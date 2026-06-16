#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import re
import subprocess
import sys
from pathlib import Path
from typing import Iterable


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Traverse a playground directory, find binaries via 'objdump -f', "
            "extract file-owned function symbols, and write CSV rows "
            "(filepath,funcname,funcaddr)."
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
    return subprocess.run(args, capture_output=True, text=True, errors="replace")


def get_file_format(path: Path) -> str | None:
    result = run_objdump(["objdump", "-f", str(path)])
    if result.returncode != 0:
        return None

    for line in result.stdout.splitlines():
        match = re.search(r"file format\s+(.+)$", line.strip())
        if match:
            return match.group(1).strip()
    return None


def parse_objdump_symbol_lines(lines: Iterable[str]) -> list[tuple[str, str]]:
    """Return (funcname, funcaddr_hex) entries for symbols defined in file."""
    symbols: list[tuple[str, str]] = []
    seen: set[tuple[str, str]] = set()

    for line in lines:
        line = line.rstrip()
        if not line:
            continue

        # GNU objdump -t / -T typical format:
        # 00000000004005f0 g    F .text 0000000000000022 main
        parts = line.split(None, 5)
        if len(parts) < 6:
            continue

        addr, _binding, kind, section, _size, name = parts

        # Keep only function symbols (upper-case F), skip file/debug rows ("f").
        if "F" not in kind:
            continue

        # Drop undefined/external references.
        section_upper = section.upper()
        if section_upper in {"*UND*", "UND"}:
            continue

        # Trim annotation suffixes occasionally printed by some toolchains.
        clean_name = name.split("\t", 1)[0].strip()
        if not clean_name:
            continue

        # Keep canonical lowercase hex with 0x prefix.
        try:
            addr_value = int(addr, 16)
        except ValueError:
            continue

        # Drop null-address placeholder symbols.
        if addr_value == 0:
            continue

        funcaddr = f"0x{addr_value:x}"

        key = (clean_name, funcaddr)
        if key in seen:
            continue
        seen.add(key)
        symbols.append(key)

    return symbols


def list_file_owned_functions(path: Path) -> list[tuple[str, str]]:
    # Prefer full symbol table; also include dynamic symbol table for binaries
    # where only dynamic symbols are available.
    entries: list[tuple[str, str]] = []
    seen: set[tuple[str, str]] = set()

    for flag in ("-t", "-T"):
        result = run_objdump(["objdump", flag, str(path)])
        if result.returncode != 0:
            continue
        for item in parse_objdump_symbol_lines(result.stdout.splitlines()):
            if item in seen:
                continue
            seen.add(item)
            entries.append(item)

    entries.sort(key=lambda x: (int(x[1], 16), x[0]))
    return entries


def iter_files(root: Path) -> Iterable[Path]:
    for p in root.rglob("*"):
        if p.is_file():
            yield p


def main() -> int:
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

    rows: list[tuple[str, str, str, str]] = []
    for path in iter_files(root):
        fileformat = get_file_format(path)
        if not fileformat:
            continue
        relpath = str(path.relative_to(root))
        for funcname, funcaddr in list_file_owned_functions(path):
            rows.append((relpath, fileformat, funcname, funcaddr))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["filepath", "fileformat", "funcname", "funcaddr"])
        writer.writerows(rows)

    print(f"wrote {len(rows)} rows to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
