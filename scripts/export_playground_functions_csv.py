#!/usr/bin/env python3
"""
Export a CSV corpus of playground functions for golden CFG testing.

Typical usage:
    uv run python scripts/export_playground_functions_csv.py \
      --output tests/playground_functions.csv

By default the exporter applies two deduplication stages while streaming rows:
    (fileformat, funcsize, function_md5)
    (fileformat, funcsize, basic_block_size_fingerprint)

This keeps one representative row for functions that look structurally identical
across multiple binaries, and then further collapses rows that keep the same
file format, function size, and lightweight basic-block-size fingerprint. We
intentionally do not require the same function name in the second pass: for
golden CFG testing we care more about deduplicating similar control-flow shapes
than about preserving distinct symbol names that still draw the same graph. The
basic-block stage uses a lightweight intra-function block traversal instead of
full CFG recovery, so it can catch some near-duplicates that differ in raw
bytes while keeping similar local block structure.

If you want the full non-deduplicated corpus, disable that behavior with:
    uv run python scripts/export_playground_functions_csv.py \
      --no-deduplicate \
      --output tests/playground_functions.csv

The script writes a CSV with these columns:
    filepath,fileformat,funcname,funcaddr,funcsize

It also prints a short report to stderr with row counts and, when deduplication
is enabled, the number of collapsed duplicate groups.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import logging as stdlib_logging
import re
import subprocess
import sys
from collections import Counter, deque
from itertools import groupby
from pathlib import Path
from typing import Iterable

from angr import Project
from loguru import logger


# angr can emit startup messages during module import, so quiet its stdlib
# loggers before importing it.
stdlib_logging.getLogger("angr").setLevel(stdlib_logging.CRITICAL)
stdlib_logging.getLogger("cle").setLevel(stdlib_logging.CRITICAL)

RawRow = tuple[str, str, str, str, int, str]
Row = tuple[str, str, str, str, int, str, tuple[int, ...] | None]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Traverse a playground directory, find binaries via 'objdump -f', "
            "extract file-owned function symbols via angr/CLE, and write CSV rows "
            "(filepath,fileformat,funcname,funcaddr,funcsize)."
        )
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("playground/angr-binaries"),
        help="Playground root directory (default: ./playground/angr-binaries)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("playground_functions.csv"),
        help="Output CSV path (default: ./playground_functions.csv)",
    )
    parser.add_argument(
        "--deduplicate",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Keep only one row per (fileformat, funcsize, function_md5) tuple. "
            "Enabled by default; use --no-deduplicate to keep the full corpus."
        ),
    )
    parser.add_argument(
        "--basic-block-deduplicate",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "After raw-byte deduplication, also collapse rows by "
            "(fileformat, funcsize, basic_block_size_fingerprint). "
            "Enabled by default; use --no-basic-block-deduplicate to disable it."
        ),
    )
    return parser.parse_args()


def run_objdump(args: list[str]) -> subprocess.CompletedProcess[str]:
    """Run objdump and capture its output."""

    return subprocess.run(args, capture_output=True, text=True, errors="replace")


def get_file_format(path: Path) -> str | None:
    """Return objdump's detected file format for a binary-like file."""

    result = run_objdump(["objdump", "-f", str(path)])
    if result.returncode != 0:
        return None

    for line in result.stdout.splitlines():
        match = re.search(r"file format\s+(.+)$", line.strip())
        if match:
            return match.group(1).strip()
    return None


def load_project(path: Path):
    """Load a binary with angr/CLE, returning ``None`` when the file is unsupported."""

    try:
        return Project(
            str(path),
            auto_load_libs=False,
        )
    except Exception as exc:
        logger.warning(f"skipping unsupported binary for angr/CLE: {path} ({type(exc).__name__}: {exc})")
        return None


def build_basic_block_size_fingerprint(project: Project, func_addr: int, func_size: int) -> tuple[int, ...]:
    """Build a cheap, deterministic basic-block-size fingerprint for one function.

    This is intentionally lighter than running CFGFast: it starts from the
    symbol address, lifts local VEX blocks, follows constant in-function branch
    targets, and also follows the fallthrough path when it remains inside the
    function range. The result is the ordered list of discovered block sizes,
    sorted by block address.
    """

    if func_size <= 0:
        return ()

    func_end = func_addr + func_size
    visited: set[int] = set()
    queue = deque([func_addr])
    blocks: list[tuple[int, int]] = []

    while queue:
        addr = queue.popleft()
        if addr in visited or not (func_addr <= addr < func_end):
            continue
        visited.add(addr)

        # Never decode past the symbol boundary, even if the lifted block would
        # otherwise continue into the next function.
        remaining_size = func_end - addr
        if remaining_size <= 0:
            continue
        try:
            block = project.factory.block(addr, size=remaining_size)
        except Exception:
            continue
        if block.size <= 0:
            continue

        blocks.append((block.addr, block.size))
        try:
            block_vex = block.vex
        except Exception:
            continue

        # Follow any statically known branch targets that stay inside the
        # current symbol range.
        jump_targets = getattr(block_vex, "constant_jump_targets", set())
        normalized_targets = sorted(
            target
            for target in (_normalize_vex_target(raw_target) for raw_target in jump_targets)
            if target is not None
        )
        for target in normalized_targets:
            if func_addr <= target < func_end:
                queue.append(target)

        # Also follow the linear fallthrough path for ordinary control flow and
        # calls, since VEX constant jump targets do not include it.
        jumpkind = getattr(block_vex, "jumpkind", "")
        if jumpkind in {"Ijk_Boring", "Ijk_Call"}:
            fallthrough = block.addr + block.size
            if func_addr <= fallthrough < func_end:
                queue.append(fallthrough)

    return tuple(size for _, size in sorted(blocks))


def _normalize_vex_target(target: object) -> int | None:
    """Convert a VEX jump target into a plain integer when possible."""

    if isinstance(target, int):
        return target

    # Some architectures expose pyvex Const objects instead of Python ints.
    value = getattr(target, "value", None)
    if isinstance(value, int):
        return value

    return None


def list_file_owned_functions(project: Project) -> list[tuple[str, str, int, str]]:
    """List non-import functions with raw-byte hashes.

    The exporter always computes raw-byte checksums first. Keep this collection
    step cheap so the caller can cheaply reject byte-for-byte duplicates before
    paying for any CFG-shaped fingerprinting.
    """

    main_object = project.loader.main_object

    entries = [
        (symbol.name, symbol.linked_addr, symbol.relative_addr, symbol.size, symbol.is_import)
        for symbol in main_object.symbols
        if symbol.is_function and symbol.name
    ]

    # CLE can expose duplicate symbols at the same address/name pair. Sort first
    # so we can collapse those duplicates deterministically.
    entries = sorted(entries, key=lambda item: (item[1], item[0]))
    entries = [next(group) for _, group in groupby(entries, key=lambda item: (item[1], item[0]))]

    # Like the app's symbol listing, infer missing sizes from the next symbol
    # when the symbol table recorded size zero.
    for idx in range(len(entries) - 1):
        name, addr, relative_addr, size, is_import = entries[idx]
        if size == 0:
            entries[idx] = (name, addr, relative_addr, entries[idx + 1][1] - addr, is_import)

    filtered: list[tuple[str, str, int, str]] = []
    for name, addr, relative_addr, size, is_import in entries:
        # Keep only non-import functions that have bytes in the binary.
        if is_import or size <= 0 or addr == 0:
            continue
        try:
            raw_bytes = bytes(main_object.memory.load(relative_addr, size))
        except KeyError:
            continue
        checksum = hashlib.md5(raw_bytes).hexdigest()
        filtered.append((name, f"0x{addr:x}", size, checksum))

    return filtered


def iter_files(root: Path) -> Iterable[Path]:
    """Yield every regular file below the selected playground root."""

    for p in root.rglob("*"):
        if any(part.startswith(".git") for part in p.parts):
            continue
        if p.is_file():
            yield p


def collect_rows(
    *,
    root: Path,
    deduplicate: bool,
    basic_block_deduplicate: bool,
) -> tuple[list[Row], int, int, int, int, Counter[tuple[str, int, str]] | None, Counter[tuple[str, int, tuple[int, ...] | None]] | None]:
    """Collect output rows while applying deduplication inline during traversal.

    This keeps project locality high: once a binary is loaded, we immediately
    decide whether each candidate survives raw-byte deduplication and, if
    needed, BB-fingerprint deduplication, instead of revisiting survivors in a
    later pass.
    """

    rows: list[Row] = []
    files_seen = 0
    binaries_seen = 0
    rows_before_dedup = 0
    rows_after_raw_dedup = 0
    raw_seen: set[tuple[str, int, str]] = set()
    bb_seen: set[tuple[str, int, tuple[int, ...] | None]] = set()
    raw_duplicate_counts: Counter[tuple[str, int, str]] | None = Counter() if deduplicate else None
    bb_duplicate_counts: Counter[tuple[str, int, tuple[int, ...] | None]] | None = (
        Counter() if deduplicate and basic_block_deduplicate else None
    )

    for path in iter_files(root):
        files_seen += 1
        fileformat = get_file_format(path)
        if not fileformat:
            continue
        project = load_project(path)
        if project is None:
            continue
        binaries_seen += 1
        relpath = str(path.relative_to(root))

        # Process each function while the owning binary is still hot in memory.
        for funcname, funcaddr, funcsize, checksum in list_file_owned_functions(project):
            rows_before_dedup += 1
            row: RawRow = (relpath, fileformat, funcname, funcaddr, funcsize, checksum)

            if not deduplicate:
                rows.append((*row, None))
                continue

            raw_key = (fileformat, funcsize, checksum)
            assert raw_duplicate_counts is not None
            raw_duplicate_counts[raw_key] += 1
            if raw_key in raw_seen:
                continue
            raw_seen.add(raw_key)
            rows_after_raw_dedup += 1

            if not basic_block_deduplicate:
                rows.append((*row, None))
                continue

            # Only pay the BB fingerprint cost for rows that already survived
            # the stricter raw-byte deduplication pass.
            block_fingerprint = build_basic_block_size_fingerprint(project, int(funcaddr, 16), funcsize)
            bb_key = (fileformat, funcsize, block_fingerprint)
            assert bb_duplicate_counts is not None
            bb_duplicate_counts[bb_key] += 1
            if bb_key in bb_seen:
                continue
            bb_seen.add(bb_key)
            rows.append((*row, block_fingerprint))

    if not deduplicate:
        rows_after_raw_dedup = rows_before_dedup

    return (
        rows,
        files_seen,
        binaries_seen,
        rows_before_dedup,
        rows_after_raw_dedup,
        raw_duplicate_counts,
        bb_duplicate_counts,
    )


def log_report(
    *,
    files_seen: int,
    binaries_seen: int,
    rows_before_dedup: int,
    rows_after_raw_dedup: int,
    rows_written: int,
    deduplicate: bool,
    duplicate_counts: Counter[tuple[str, int, str]] | None,
    basic_block_deduplicate: bool,
    basic_block_duplicate_counts: Counter[tuple[str, int, tuple[int, ...] | None]] | None,
    output: Path,
) -> None:
    """Write a compact export report to stderr."""

    logger.info(f"files scanned: {files_seen}")
    logger.info(f"binary-like files accepted: {binaries_seen}")
    logger.info(f"rows before deduplication: {rows_before_dedup}")
    logger.info(f"rows written: {rows_written}")
    logger.info(f"output: {output}")

    if not deduplicate or duplicate_counts is None:
        return

    duplicate_groups = sum(1 for count in duplicate_counts.values() if count > 1)
    duplicate_rows_removed = rows_before_dedup - rows_written
    logger.info("deduplication enabled: yes")
    logger.info(f"duplicate groups collapsed: {duplicate_groups}")
    logger.info(f"rows removed by raw-byte deduplication: {rows_before_dedup - rows_after_raw_dedup}")

    most_common_duplicates = [
        (key, count) for key, count in duplicate_counts.most_common(10) if count > 1
    ]
    if most_common_duplicates:
        logger.info("top duplicate groups:")
        for (fileformat, funcsize, _checksum), count in most_common_duplicates:
            logger.info(f"  count={count} fileformat={fileformat} funcsize={funcsize}")

    if not basic_block_deduplicate or basic_block_duplicate_counts is None:
        logger.info(f"rows removed by total deduplication: {duplicate_rows_removed}")
        return

    bb_duplicate_groups = sum(1 for count in basic_block_duplicate_counts.values() if count > 1)
    bb_rows_removed = rows_after_raw_dedup - rows_written
    logger.info("basic-block deduplication enabled: yes")
    logger.info(f"basic-block duplicate groups collapsed: {bb_duplicate_groups}")
    logger.info(f"rows removed by basic-block deduplication: {bb_rows_removed}")
    logger.info(f"rows removed by total deduplication: {duplicate_rows_removed}")

    most_common_bb_duplicates = [
        (key, count) for key, count in basic_block_duplicate_counts.most_common(10) if count > 1
    ]
    if most_common_bb_duplicates:
        logger.info("top basic-block duplicate groups:")
        for (fileformat, funcsize, block_fingerprint), count in most_common_bb_duplicates:
            logger.info(
                f"  count={count} fileformat={fileformat} funcsize={funcsize} "
                f"block_sizes={block_fingerprint}"
            )


def main() -> int:
    """Export the playground function corpus to CSV."""

    logger.remove()
    logger.add(sys.stderr, level="INFO", format="{message}")
    args = parse_args()
    root = args.root

    if not root.exists() or not root.is_dir():
        print(f"error: root directory not found or not a directory: {root}", file=sys.stderr)
        return 1

    try:
        run_objdump(["objdump", "--version"])
    except FileNotFoundError:
        print("error: 'objdump' command not found in PATH", file=sys.stderr)
        return 1

    (
        rows,
        files_seen,
        binaries_seen,
        rows_before_dedup,
        rows_after_raw_dedup,
        duplicate_counts,
        basic_block_duplicate_counts,
    ) = collect_rows(
        root=root,
        deduplicate=args.deduplicate,
        basic_block_deduplicate=args.basic_block_deduplicate,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["filepath", "fileformat", "funcname", "funcaddr", "funcsize"])
        writer.writerows(
            (filepath, fileformat, funcname, funcaddr, funcsize)
            for filepath, fileformat, funcname, funcaddr, funcsize, _checksum, _block_fingerprint in rows
        )

    log_report(
        files_seen=files_seen,
        binaries_seen=binaries_seen,
        rows_before_dedup=rows_before_dedup,
        rows_after_raw_dedup=rows_after_raw_dedup,
        rows_written=len(rows),
        deduplicate=args.deduplicate,
        duplicate_counts=duplicate_counts,
        basic_block_deduplicate=args.basic_block_deduplicate,
        basic_block_duplicate_counts=basic_block_duplicate_counts,
        output=args.output,
    )
    print(f"wrote {len(rows)} rows to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
