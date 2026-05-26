from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import List
from angr import Project

from .utils import demangle
from . import get_fast_cfg

UNKNOWN_FUNC = "**UNKNOWN**"


@dataclass(frozen=True)
class FunctionSymbol:
    addr: int
    mangled_name: str
    demangled_name: str
    is_import: bool
    is_export: bool
    origin: str


@lru_cache
def list_function_symbols(project: Project) -> List[FunctionSymbol]:
    """
    List function symbols from the project's main object.

    This function iterates over the main object's symbols, filtering for functions.
    If no function symbols are found, it falls back to extracting functions from the CFG,
    and constructs a list of FunctionSymbol instances containing address, mangled and demangled names,
    import/export status, and origin. The list is sorted by address and mangled name.

    Args:
        project (angr.Project): The angr project to analyze.

    Returns:
        List[FunctionSymbol]: A sorted list of function symbols.
    """
    main_obj = project.loader.main_object
    symbols = []
    for sym in main_obj.symbols:
        if not sym.is_function:
            continue
        mangled = sym.name or UNKNOWN_FUNC
        symbols.append(
            FunctionSymbol(
                addr=int(sym.rebased_addr),
                mangled_name=mangled,
                demangled_name=demangle(mangled),
                is_import=bool(sym.is_import),
                is_export=bool(sym.is_export),
                origin="symbol",
            )
        )

    if not symbols:
        cfg = get_fast_cfg(project)
        for func in cfg.kb.functions.values():
            mangled = func.name or UNKNOWN_FUNC
            symbols.append(
                FunctionSymbol(
                    addr=int(func.addr),
                    mangled_name=mangled,
                    demangled_name=demangle(mangled),
                    is_import=func.is_plt,
                    is_export=False,
                    origin="cfg",
                )
            )

    return sorted(symbols, key=lambda s: (s.addr, s.mangled_name))
