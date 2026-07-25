"""Compare application function-symbol discovery against the CSV corpus.

The CSV exporter and the application both start with CLE function symbols and
infer spans for zero-sized entries. This explicit slow suite treats the corpus
as the golden view of file-owned CFG candidates: non-import, non-zero-address
symbols with a positive size. It runs only when this file is selected directly,
so normal unit-test runs do not load every playground binary.
"""

from __future__ import annotations

import csv
from collections import defaultdict
from collections.abc import Iterable
from pathlib import Path

import pytest
from angr import Project

from bingraph.helpers.symbols import FunctionSymbol, list_function_symbols


REPOSITORY_ROOT = Path(__file__).parents[2]
PLAYGROUND_ROOT = REPOSITORY_ROOT / "angr-binaries" / "tests"
CORPUS_PATH = REPOSITORY_ROOT / "tests" / "playground_functions.csv"


def _corpus_symbols() -> dict[str, set[tuple[str, int, int]]]:
    """Return file-owned function symbols recorded by the exported corpus."""

    symbols: dict[str, set[tuple[str, int, int]]] = defaultdict(set)
    with CORPUS_PATH.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            symbols[row["filepath"]].add(
                (row["funcname"], int(row["funcaddr"], 16), int(row["funcsize"]))
            )
    return dict(symbols)


CORPUS_SYMBOLS = _corpus_symbols()


def _cfg_candidate_symbols(
    symbols: Iterable[FunctionSymbol],
) -> set[tuple[str, int, int]]:
    """Filter application symbols to the file-owned candidates exported to CSV."""

    return {
        (symbol.name, symbol.addr, symbol.size)
        for symbol in symbols
        if (
            symbol.origin == "symtab"
            and not symbol.is_import
            and symbol.addr != 0
            and symbol.size > 0
        )
    }


def _format_difference(
    filepath: str,
    app_only: set[tuple[str, int, int]],
    corpus_only: set[tuple[str, int, int]],
) -> str:
    """Render a compact mismatch report without flooding one pytest failure."""

    details = [f"symbol corpus mismatch for {filepath}"]
    for label, entries in (
        ("application only", app_only),
        ("corpus only", corpus_only),
    ):
        if not entries:
            continue
        preview = ", ".join(
            f"{name}@{addr:#x}+{size:#x}" for name, addr, size in sorted(entries)[:5]
        )
        suffix = " ..." if len(entries) > 5 else ""
        details.append(f"{label} ({len(entries)}): {preview}{suffix}")
    return "\n".join(details)


@pytest.mark.slow
@pytest.mark.parametrize("filepath", sorted(CORPUS_SYMBOLS))
def test_application_symbols_match_exported_corpus(filepath: str) -> None:
    """Keep application CFG candidates aligned with one corpus binary."""

    project = Project(PLAYGROUND_ROOT / filepath, auto_load_libs=False)
    try:
        app_symbols = _cfg_candidate_symbols(list_function_symbols(project))
    finally:
        # The application cache is useful for requests, but retaining 621 test
        # projects would make this explicit golden suite needlessly memory-heavy.
        list_function_symbols.cache_clear()

    corpus_symbols = CORPUS_SYMBOLS[filepath]
    app_only = app_symbols - corpus_symbols
    corpus_only = corpus_symbols - app_symbols

    assert not app_only and not corpus_only, _format_difference(
        filepath, app_only, corpus_only
    )
