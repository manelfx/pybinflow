#!/usr/bin/env python3
"""
Export a CSV corpus of playground functions for golden CFG testing.

Typical usage:
    uv run python scripts/export_playground_functions_csv.py \
      --output tests/playground_functions.csv

The exporter writes one row for every candidate function that can be read from
the supported playground binaries. In addition to the base function identity,
each row stores metadata that makes offline corpus analysis easier:

    filepath,fileformat,funcname,funcaddr,funcsize,checksum,num_bbs,duplicate

Where:
    checksum  = last 5 hex characters of the function MD5
    num_bbs   = lightweight basic-block count used by the BB dedup heuristic
    duplicate = whether the row would be dropped by the raw-first, BB-second
                dedup strategy used to shrink the golden-test corpus

The exporter still computes duplicate information in streaming order, but it no
longer drops duplicate rows from the CSV. That way the committed corpus keeps
all candidates while still recording which entries are interesting enough for
CFG regression tests.
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
from dataclasses import dataclass
from itertools import groupby
from pathlib import Path
from typing import Iterable

from angr import Project
from loguru import logger


# angr can emit startup messages during module import, so quiet its stdlib
# loggers before importing it.
stdlib_logging.getLogger("angr").setLevel(stdlib_logging.CRITICAL)
stdlib_logging.getLogger("cle").setLevel(stdlib_logging.CRITICAL)


@dataclass(frozen=True)
class CorpusRow:
    filepath: str
    fileformat: str
    funcname: str
    funcaddr: str
    funcsize: int
    checksum: str
    num_bbs: int
    duplicate: bool


def parse_args() -> argparse.Namespace:
    """Parse the small CLI surface used by this exporter."""

    parser = argparse.ArgumentParser(
        description=(
            "Traverse a playground directory, find binaries via 'objdump -f', "
            "extract file-owned function symbols via angr/CLE, and write CSV rows "
            "with duplicate-analysis metadata."
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
    """Load a binary with angr/CLE, returning ``None`` when unsupported."""

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

    assert func_size > 0, "Function size must be positive before BB fingerprinting."

    func_end = func_addr + func_size
    visited: set[int] = set()
    queue = deque([func_addr])
    blocks: list[tuple[int, int]] = []

    while queue:
        addr = queue.popleft()
        if addr in visited or not (func_addr <= addr < func_end):
            continue
        visited.add(addr)

        try:
            # Ask angr for the natural block boundary at this address. Passing a
            # synthetic size here can swallow multiple real blocks, padding, or
            # even neighboring stubs, which inflates the BB count.
            block = project.factory.block(addr)
        except Exception:
            continue
        if block.size <= 0:
            continue

        # Keep the traversal fenced to the inferred function range even if the
        # natural block would otherwise run past the end.
        effective_size = min(block.size, func_end - block.addr)
        if effective_size <= 0:
            continue

        blocks.append((block.addr, effective_size))
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
            fallthrough = block.addr + effective_size
            if func_addr <= fallthrough < func_end:
                queue.append(fallthrough)

    if not blocks:
        # Keep the corpus self-contained: every exported function should have a
        # positive BB count, even if the lightweight traversal could not expand
        # any block successfully.
        return (func_size,)

    return tuple(size for _, size in sorted(blocks))


def _normalize_vex_target(target: object) -> int | None:
    """Convert a VEX jump target into a plain integer when possible."""

    if isinstance(target, int):
        return target

    value = getattr(target, "value", None)
    if isinstance(value, int):
        return value

    return None


def list_file_owned_functions(project: Project) -> list[tuple[str, str, int, str]]:
    """List non-import functions with raw-byte hashes.

    Keep this collection step focused on symbol ownership and raw bytes. The
    caller is responsible for attaching BB metadata and duplicate markers.
    """

    main_object = project.loader.main_object

    entries = [
        (symbol.name, symbol.linked_addr, symbol.relative_addr, symbol.size, symbol.is_import)
        for symbol in main_object.symbols
        if symbol.is_function and symbol.name
    ]

    # CLE can expose duplicate symbols at the same address/name pair. Sort
    # first so we can collapse those duplicates deterministically.
    entries = sorted(entries, key=lambda item: (item[1], item[0]))
    entries = [next(group) for _, group in groupby(entries, key=lambda item: (item[1], item[0]))]

    # Like the app's symbol listing, infer missing sizes when the symbol table
    # recorded size zero. Cap the inferred extent to the earliest of:
    # - the next symbol address
    # - the end of the containing section
    # This avoids size-zero symbols like `_init` accidentally absorbing PLT or
    # neighboring sections just because the next function symbol happens later.
    for idx in range(len(entries) - 1):
        name, addr, relative_addr, size, is_import = entries[idx]
        if size == 0:
            inferred_size = entries[idx + 1][1] - addr
            section = main_object.find_section_containing(addr)
            if section is not None:
                section_end = section.vaddr + section.memsize
                inferred_size = min(inferred_size, section_end - addr)
            entries[idx] = (name, addr, relative_addr, inferred_size, is_import)

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

    for path in root.rglob("*"):
        if any(part.startswith(".git") for part in path.parts):
            continue
        if path.is_file():
            yield path


def collect_rows(
    *,
    root: Path,
) -> tuple[list[CorpusRow], int, int, int, Counter[tuple[str, int, str]], Counter[tuple[str, int, tuple[int, ...]]]]:
    """Collect every candidate row while computing duplicate metadata inline.

    We still apply the same logical dedup order as before:
    1. raw-byte deduplication by (fileformat, funcsize, md5)
    2. BB-shape deduplication by (fileformat, funcsize, bb_fingerprint)

    The difference is that every candidate row is emitted, and duplicate rows
    are now marked in metadata instead of being dropped from the CSV.
    """

    rows: list[CorpusRow] = []
    files_seen = 0
    binaries_seen = 0
    candidate_rows = 0
    raw_seen: set[tuple[str, int, str]] = set()
    bb_seen: set[tuple[str, int, tuple[int, ...]]] = set()
    raw_duplicate_counts: Counter[tuple[str, int, str]] = Counter()
    bb_duplicate_counts: Counter[tuple[str, int, tuple[int, ...]]] = Counter()

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
            candidate_rows += 1
            block_fingerprint = build_basic_block_size_fingerprint(project, int(funcaddr, 16), funcsize)
            num_bbs = len(block_fingerprint)
            raw_key = (fileformat, funcsize, checksum)
            bb_key = (fileformat, funcsize, block_fingerprint)

            raw_duplicate_counts[raw_key] += 1
            is_raw_duplicate = raw_key in raw_seen
            if not is_raw_duplicate:
                raw_seen.add(raw_key)

            # Only the first raw survivor participates in BB deduplication, but
            # every row still records its BB count in the output corpus.
            is_bb_duplicate = False
            if not is_raw_duplicate:
                bb_duplicate_counts[bb_key] += 1
                is_bb_duplicate = bb_key in bb_seen
                if not is_bb_duplicate:
                    bb_seen.add(bb_key)

            rows.append(
                CorpusRow(
                    filepath=relpath,
                    fileformat=fileformat,
                    funcname=funcname,
                    funcaddr=funcaddr,
                    funcsize=funcsize,
                    checksum=checksum[-5:],
                    num_bbs=num_bbs,
                    duplicate=is_raw_duplicate or is_bb_duplicate,
                )
            )

    return rows, files_seen, binaries_seen, candidate_rows, raw_duplicate_counts, bb_duplicate_counts


def log_report(
    *,
    files_seen: int,
    binaries_seen: int,
    rows_written: int,
    raw_duplicate_counts: Counter[tuple[str, int, str]],
    bb_duplicate_counts: Counter[tuple[str, int, tuple[int, ...]]],
    duplicate_rows: int,
    output: Path,
) -> None:
    """Write a compact export report to stderr."""

    logger.info(f"files scanned: {files_seen}")
    logger.info(f"binary-like files accepted: {binaries_seen}")
    logger.info(f"candidate rows written: {rows_written}")
    logger.info(f"output: {output}")

    raw_duplicate_groups = sum(1 for count in raw_duplicate_counts.values() if count > 1)
    bb_duplicate_groups = sum(1 for count in bb_duplicate_counts.values() if count > 1)
    raw_duplicate_rows = sum(count - 1 for count in raw_duplicate_counts.values() if count > 1)
    bb_duplicate_rows = sum(count - 1 for count in bb_duplicate_counts.values() if count > 1)
    unique_rows = rows_written - duplicate_rows

    logger.info(f"rows marked duplicate: {duplicate_rows}")
    logger.info(f"rows marked unique: {unique_rows}")
    logger.info(f"raw-byte duplicate groups: {raw_duplicate_groups}")
    logger.info(f"rows flagged by raw-byte deduplication: {raw_duplicate_rows}")
    logger.info(f"basic-block duplicate groups: {bb_duplicate_groups}")
    logger.info(f"rows flagged by basic-block deduplication: {bb_duplicate_rows}")

    most_common_raw = [
        (key, count) for key, count in raw_duplicate_counts.most_common(10) if count > 1
    ]
    if most_common_raw:
        logger.info("top raw-byte duplicate groups:")
        for (fileformat, funcsize, _checksum), count in most_common_raw:
            logger.info(f"  count={count} fileformat={fileformat} funcsize={funcsize}")

    most_common_bb = [
        (key, count) for key, count in bb_duplicate_counts.most_common(10) if count > 1
    ]
    if most_common_bb:
        logger.info("top basic-block duplicate groups:")
        for (fileformat, funcsize, block_fingerprint), count in most_common_bb:
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

    rows, files_seen, binaries_seen, _candidate_rows, raw_duplicate_counts, bb_duplicate_counts = collect_rows(
        root=root
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "filepath",
                "fileformat",
                "funcname",
                "funcaddr",
                "funcsize",
                "checksum",
                "num_bbs",
                "duplicate",
            ]
        )
        writer.writerows(
            [
                row.filepath,
                row.fileformat,
                row.funcname,
                row.funcaddr,
                row.funcsize,
                row.checksum,
                row.num_bbs,
                str(row.duplicate).lower(),
            ]
            for row in rows
        )

    log_report(
        files_seen=files_seen,
        binaries_seen=binaries_seen,
        rows_written=len(rows),
        raw_duplicate_counts=raw_duplicate_counts,
        bb_duplicate_counts=bb_duplicate_counts,
        duplicate_rows=sum(1 for row in rows if row.duplicate),
        output=args.output,
    )
    print(f"wrote {len(rows)} rows to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
