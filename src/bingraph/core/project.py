from __future__ import annotations
from functools import lru_cache
from pathlib import Path
from typing import Any, cast

from angr import Project, KnowledgeBase
from angr.analyses import CFGFast, CFGEmulated
from angr.analyses.cfg import CFGBase
from loguru import logger

from bingraph.helpers import time_it, get_settings, CfgMode
from bingraph.cfg import (
    build_custom_cfg,
    log_cfg_status,
)
from .symbols import list_function_symbols


@lru_cache
def _get_project(spath: str, mtime: float) -> Project:
    """
    Return an angr project from a cached binary image.

    Args:
        spath (str): The file path of the binary to analyze.
        mtime (float): Binary modification time -- note this is not used but is still cached,
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


def _get_fast_cfg(project: Project, kb: KnowledgeBase, func_addr: int) -> CFGFast:
    """
    Build or retrieve the fast control flow graph (CFGFast) for the given project.

    Args:
        project (Project): The angr project for which to build the fast CFG.
        kb (KnowledgeBase): Shared knowledge base reused across CFG strategies.
        func_addr (int): Address of the target function.

    Returns:
        CFGFast: The fast control flow graph of the project.
    """
    # Find the function symbol first so we can bound CFGFast to its address range.
    function = next(
        (sym for sym in list_function_symbols(project) if sym.addr == func_addr), None
    )
    if not function:
        raise KeyError(f"Function {func_addr:#x} not found binary")
    regions = [(func_addr, func_addr + function.size)]
    logger.info(
        f"Region for CFG reconstruct will be {[(hex(a), hex(b)) for a, b in regions]}"
    )

    def _should_retry_cfgfast_with_safer_settings(exc: Exception) -> bool:
        """
        Return True when CFGFast should be retried with safer bounded settings.

        We have seen two angr failure modes on our region-bounded CFGFast runs:

        1. Smart-scan post-processing can dereference a `None` block
           (`AttributeError: 'NoneType' object has no attribute 'addr'`).
        2. Data-reference collection can fail inside Clinic/StackPointerTracker
           with a `KeyError(<callee-addr>)` when the bounded knowledge base does
           not contain metadata for out-of-region callees.

        In both cases we retry once with `force_smart_scan=False`, which also
        disables `data_references` in `build_cfg()`. That keeps the analysis
        bounded to the target function while avoiding the fragile angr paths.
        """
        if isinstance(exc, AttributeError):
            return "'NoneType' object has no attribute 'addr'" in str(exc)

        return isinstance(exc, KeyError)

    # Create a fresh knowledge base so this analysis does not pollute the project state.
    def build_cfg(*, force_smart_scan: bool) -> CFGFast:
        return project.analyses.CFGFast(
            kb=kb,
            # we already know the exact entry point we want
            function_starts=[func_addr],
            # big performance win, do not analyze the full binary
            regions=regions,
            # avoid extra function discovery heuristics, already gave the function start explicitly
            eh_frame=False,
            exceptions=False,
            force_complete_scan=False,
            function_prologues=False,
            start_at_entry=False,
            symbols=False,
            # Enable smarter basic-block discovery, but keep it constrained to the
            # requested function region.
            data_references=force_smart_scan,
            force_smart_scan=force_smart_scan,
            resolve_indirect_jumps=True,
            # stable, clean function graphs for rendering
            normalize=True,
        )

    # Prefer the smarter region-bounded scan, but retry without it for the
    # specific angr post-processing crash pattern we observed in the golden corpus.
    try:
        return build_cfg(force_smart_scan=True)
    except Exception as exc:
        if not _should_retry_cfgfast_with_safer_settings(exc):
            raise
        logger.warning(
            f"Retrying CFGFast with safer settings for {func_addr:#x} after angr crash: {exc}"
        )
        return build_cfg(force_smart_scan=False)


def _get_emu_cfg(
    project: Project, kb: KnowledgeBase, func_addr: int, keep_state: bool
) -> CFGEmulated:
    """
    Build or retrieve the emulated control flow graph (CFGEmulated) for a given function.

    Args:
        project (Project): The angr project for which to build the emulated CFG.
        kb (KnowledgeBase): Shared knowledge base reused across CFG strategies.
        func_addr (int): Address of the target function.
        keep_state (bool): Whether to retain full symbolic state during emulation.

    Returns:
        CFGEmulated: The emulated control flow graph of the project.
    """
    cfg_emulated = cast(Any, project.analyses.CFGEmulated)
    return cfg_emulated(
        kb=kb, starts=[func_addr], call_depth=0, keep_state=keep_state, normalize=True
    )


@lru_cache
@time_it
def get_cfg(
    project: Project, func_addr: int, cfg_mode: CfgMode | None = None
) -> CFGBase:
    """Return the CFG for one function according to the configured fallback mode."""

    resolved_cfg_mode = cfg_mode or get_settings().cfg_mode
    logger.info(
        f"Getting CFG for function {func_addr:#x} with mode '{resolved_cfg_mode}'"
    )
    # Keep one KB per high-level CFG request so a fallback CFGEmulated run can
    # reuse the metadata already discovered by CFGFast, especially comments and
    # related knowledge attached during the fast analysis.
    kb = KnowledgeBase(project)

    fast_cfg = _get_fast_cfg(project, kb, func_addr)

    if resolved_cfg_mode == "none":
        cfg = fast_cfg
    elif resolved_cfg_mode == "custom":
        # CustomCFG intentionally exposes the CFGBase subset consumed by the
        # rest of bingraph, but angr's nominal type hierarchy cannot express it.
        cfg = cast(CFGBase, build_custom_cfg(project, kb, func_addr, fast_cfg))
    else:
        raise ValueError(f"Unsupported cfg mode: {resolved_cfg_mode}")

    log_cfg_status(cfg, func_addr, f"Selected CFG ({resolved_cfg_mode})")
    return cfg
