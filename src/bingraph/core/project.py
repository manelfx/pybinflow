from __future__ import annotations
from functools import lru_cache
from pathlib import Path

from angr import Project, KnowledgeBase
from angr.analyses import CFGFast, CFGEmulated
from angr.analyses.cfg import CFGBase
from loguru import logger

from bingraph.helpers import time_it
from .symbols import list_function_symbols


@lru_cache
def _get_project(spath: str, mtime: float) -> Project:
    """
    Returns an angr project from a (maybe cathed) image.
    
    Args:
        spath (str): The file path of the binary to analyze.
        mtime (float): Binary modification time -- note this is not used but still cached,
            so updated binaries are not incorrectly cached.

    Returns:
        Project: The loaded angr project.
    """
    return Project(spath, auto_load_libs=False)


def load_project(path: Path) -> Project:
    """
    Load an angr project from the specified path.

    Args:
        path (Path): The file path of the binary to analyze.

    Returns:
        Project: The loaded angr project.
    """
    return _get_project(str(path), path.stat().st_mtime)


@lru_cache
def _get_fast_cfg(project: Project, func_addr: int) -> CFGFast:
    """
    Build or retrieve the fast control flow graph (CFGFast) for the given project.

    Args:
        project (Project): The angr project for which to build the fast CFG.
        func_addr (int): Address of the target function.

    Returns:
        CFGFast: The fast control flow graph of the project.
    """
    # find the function in binary symbols to figure out function boundaries
    function = next((sym for sym in list_function_symbols(project) if sym.addr == func_addr), None)
    if not function:
        raise KeyError(f"Function {func_addr:#x} not found binary")
    regions = [(func_addr, func_addr + function.size)]
    logger.info(f"Region for CFG reconstruct will be {[(hex(a), hex(b)) for a, b in regions]}") 

    # create clean knowledge base object, so project is not pulluted with this analysis
    kb = KnowledgeBase(project)

    # get CFG, on a defined region (function), with smart analysis
    return project.analyses.CFGFast(kb=kb,
                                    function_starts=[func_addr],
                                    regions=regions,
                                    normalize=True,
                                    force_smart_scan=True,
                                    resolve_indirect_jumps=True,
                                    data_references=True)


@lru_cache
def _get_emu_cfg(project: Project, func_addr: int) -> CFGEmulated:
    """
    Build or retrieve the emulated control flow graph (CFGEmulated) for a given function.

    Args:
        project (Project): The angr project for which to build the emulated CFG.
        func_addr (int): Address of the target function.

    Returns:
        CFGEmulated: The emulated control flow graph of the project.
    """
    # create clean knowledge base object, so project is not pulluted with this analysis
    kb = KnowledgeBase(project)

    return project.analyses.CFGEmulated(kb=kb,
                                        starts=[func_addr],
                                        call_depth=0,
                                        #keep_state=True,
                                        normalize=True)


@time_it
def get_cfg(project: Project, func_addr: int, cfg_mode: str) -> CFGBase:

    logger.info(f"Getting {cfg_mode} CFG for function {func_addr:#x}")
    return {"fast": _get_fast_cfg, "emulated": _get_emu_cfg}[cfg_mode](project, func_addr)
