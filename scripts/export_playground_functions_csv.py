#!/usr/bin/env python3
"""
Export a CSV corpus of playground functions for golden CFG testing.

Typical usage:
    uv run python scripts/export_playground_functions_csv.py \
      --output tests/playground_functions.csv

By default the exporter deduplicates rows by:
    (fileformat, funcsize, function_md5)

This keeps one representative row for functions that look structurally identical
across multiple binaries, which reduces the size of the golden-test corpus.

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
from collections import Counter
from itertools import groupby
from pathlib import Path
from typing import Iterable

from loguru import logger

# angr can emit startup messages during module import, so quiet its stdlib
# loggers before importing it.
stdlib_logging.getLogger("angr").setLevel(stdlib_logging.CRITICAL)
stdlib_logging.getLogger("cle").setLevel(stdlib_logging.CRITICAL)

from angr import Project


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


def list_file_owned_functions(main_object) -> list[tuple[str, str, int, str]]:
    """List non-import functions from CLE using canonical names and raw-byte hashes."""

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


def deduplicate_rows(
    rows: list[tuple[str, str, str, str, int, str]]
) -> tuple[list[tuple[str, str, str, str, int, str]], Counter[tuple[str, str, int, str]]]:
    """Collapse rows by (fileformat, funcsize, function_md5)."""

    deduped: list[tuple[str, str, str, str, int, str]] = []
    seen: set[tuple[str, int, str]] = set()
    duplicate_counts: Counter[tuple[str, int, str]] = Counter()

    for row in rows:
        filepath, fileformat, funcname, funcaddr, funcsize, checksum = row
        key = (fileformat, funcsize, checksum)
        duplicate_counts[key] += 1
        if key in seen:
            continue
        seen.add(key)
        deduped.append((filepath, fileformat, funcname, funcaddr, funcsize, checksum))

    return deduped, duplicate_counts


def log_report(
    *,
    files_seen: int,
    binaries_seen: int,
    rows_before_dedup: int,
    rows_written: int,
    deduplicate: bool,
    duplicate_counts: Counter[tuple[str, int, str]] | None,
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
    logger.info(f"rows removed by deduplication: {duplicate_rows_removed}")

    most_common_duplicates = [
        (key, count) for key, count in duplicate_counts.most_common(10) if count > 1
    ]
    if most_common_duplicates:
        logger.info("top duplicate groups:")
        for (fileformat, funcsize, _checksum), count in most_common_duplicates:
            logger.info(f"  count={count} fileformat={fileformat} funcsize={funcsize}")


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

    rows: list[tuple[str, str, str, str, int, str]] = []
    files_seen = 0
    binaries_seen = 0
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
        for funcname, funcaddr, funcsize, checksum in list_file_owned_functions(project.loader.main_object):
            rows.append((relpath, fileformat, funcname, funcaddr, funcsize, checksum))

    rows_before_dedup = len(rows)
    duplicate_counts: Counter[tuple[str, int, str]] | None = None
    if args.deduplicate:
        rows, duplicate_counts = deduplicate_rows(rows)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["filepath", "fileformat", "funcname", "funcaddr", "funcsize"])
        writer.writerows(
            (filepath, fileformat, funcname, funcaddr, funcsize)
            for filepath, fileformat, funcname, funcaddr, funcsize, _checksum in rows
        )

    log_report(
        files_seen=files_seen,
        binaries_seen=binaries_seen,
        rows_before_dedup=rows_before_dedup,
        rows_written=len(rows),
        deduplicate=args.deduplicate,
        duplicate_counts=duplicate_counts,
        output=args.output,
    )
    print(f"wrote {len(rows)} rows to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
