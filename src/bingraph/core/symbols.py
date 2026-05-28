from __future__ import annotations
from functools import lru_cache
from itertools import groupby

from loguru import logger
from pydantic import BaseModel
from angr import Project, KnowledgeBase


UNKNOWN_FUNC = "**UNKNOWN**"


class FunctionSymbol(BaseModel):
    addr: int
    name: str
    size: int
    is_import: bool
    origin: str


@lru_cache
def list_function_symbols(project: Project) -> list[FunctionSymbol]:

    logger.info("Getting function symbols from symbol table")

    # grab functions from binary symbol table
    symbols = [
        FunctionSymbol(name=symbol.name or UNKNOWN_FUNC,
                       addr=symbol.rebased_addr,
                       size=symbol.size,
                       is_import=symbol.is_import,
                       origin="symtab")
        for symbol in project.loader.main_object.symbols
        if symbol.is_function
    ]

    if not symbols:
        logger.info("No function symbols were found (stripped binary?), analyzing binary ")

        # create clean knowledge base object, so project is not pulluted with this analysis
        kb = KnowledgeBase(project)

        # Super-fast CFG reconstruction, by avoiding fancy reconstruct heuristics.
        # Still slower than reading symbol table, so this might take a while for big binaries.
        # Note we don't need to cache since output symbols will be cached anyway.
        cfg = project.analyses.CFGFast(kb=kb,
                                       force_smart_scan=False,
                                       resolve_indirect_jumps=False,
                                       data_references=False)

        # grab functions from analysis output
        symbols = [
            FunctionSymbol(name=func.name or UNKNOWN_FUNC,
                           addr=func.addr,
                           size=func.size,
                           is_import=func.is_plt,
                           origin="cfg")
            for func in cfg.kb.functions.values()
            if not (func.is_simprocedure or func.is_alignment or func.is_syscall)
        ]

    # sort symbols
    symbols = sorted(symbols, key=lambda s: (s.addr, s.name))

    # remove duplicaties (it happens sometimes the CLE loader duplicate symbols)
    symbols = [next(group) for _, group in groupby(symbols, key=lambda s: (s.addr, s.name))]

    # also for CLE symbols, size it is often not defined, so do it based on next symbol address 
    # FIXME: last element is not fixed,
    # to do so we should gather all elements first, fix size, then filter only functions
    for idx in range(len(symbols) - 1):
        if symbols[idx].size == 0:
            symbols[idx].size = symbols[idx + 1].addr - symbols[idx].addr

    logger.info(f"Obtained {len(symbols)} symbols")
    return symbols
