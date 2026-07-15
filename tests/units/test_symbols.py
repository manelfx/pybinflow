"""Fast tests for symbol-table normalization used by CFG reconstruction."""

from types import SimpleNamespace

from bingraph.helpers.symbols import list_function_symbols


def _symbol(
    addr: int,
    name: str,
    size: int,
    *,
    is_function: bool = True,
) -> SimpleNamespace:
    """Create the loader-symbol shape used by symbol normalization."""

    return SimpleNamespace(
        rebased_addr=addr,
        name=name,
        size=size,
        is_function=is_function,
        is_import=False,
    )


class _Project:
    """Hashable project stand-in compatible with the symbol-list cache."""

    def __init__(self, symbols: list[SimpleNamespace]) -> None:
        """Expose loader symbols through the minimal angr-project shape."""

        self.loader = SimpleNamespace(main_object=SimpleNamespace(symbols=symbols))


def test_symbol_listing_sorts_deduplicates_and_infers_sizes() -> None:
    """Infer a zero-sized function span from the next distinct symbol address."""

    project = _Project(
        [
            _symbol(0x1020, "later", 4),
            _symbol(0x1000, "entry", 0),
            _symbol(0x1000, "entry", 0),
            _symbol(0x1010, "not_a_function", 9, is_function=False),
        ]
    )
    list_function_symbols.cache_clear()

    symbols = list_function_symbols(project)

    assert [(symbol.addr, symbol.name, symbol.size) for symbol in symbols] == [
        (0x1000, "entry", 0x20),
        (0x1020, "later", 4),
    ]


def test_symbol_listing_drops_unresolved_trailing_zero_sized_symbol() -> None:
    """Avoid returning a CFG candidate whose function bound cannot be inferred."""

    project = _Project([_symbol(0x1000, "last", 0)])
    list_function_symbols.cache_clear()

    assert list_function_symbols(project) == []
