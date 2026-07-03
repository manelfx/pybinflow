from __future__ import annotations
from functools import lru_cache
from pathlib import Path

from angr import Project, KnowledgeBase
from angr.analyses import CFGFast, CFGEmulated
from angr.analyses.cfg import CFGBase
from loguru import logger

from bingraph.helpers import time_it, get_settings, CfgMode
from .cfg import build_custom_cfg
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
    function = next((sym for sym in list_function_symbols(project) if sym.addr == func_addr), None)
    if not function:
        raise KeyError(f"Function {func_addr:#x} not found binary")
    regions = [(func_addr, func_addr + function.size)]
    logger.info(f"Region for CFG reconstruct will be {[(hex(a), hex(b)) for a, b in regions]}") 

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
            normalize=True
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


def _get_emu_cfg(project: Project, kb: KnowledgeBase, func_addr: int, keep_state: bool) -> CFGEmulated:
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
    return project.analyses.CFGEmulated(kb=kb,
                                        starts=[func_addr],
                                        call_depth=0,
                                        keep_state=keep_state,
                                        normalize=True)


def _iter_function_nodes(cfg: CFGBase, func_addr: int):
    """Yield non-simprocedure CFG nodes that belong to the requested function."""

    for node in cfg.graph.nodes():
        if node.function_address != func_addr or node.is_simprocedure:
            continue
        yield node


def _has_decoding_coverage_mismatch(node) -> bool:
    """Return True when decoded instructions do not cover the full node span."""

    if node.size == 0:
        logger.warning(
            f"CFG anomaly for function {node.function_address:#x}: zero_sized_block at "
            f"{node.addr:#x}: node size is zero"
        )
        return True

    try:
        insns = list(node.block.capstone.insns)
    except (AttributeError, KeyError) as exc:
        logger.warning(
            f"Unable to inspect decoded coverage for CFG node at {node.addr:#x}: "
            f"{type(exc).__name__}: {exc}"
        )
        insns = []

    expected_addr = node.addr

    # A well-formed node should decode contiguously from its start address all
    # the way to `node.addr + node.size`. If angr splits the node at the wrong
    # address, capstone still decodes instructions, but the decoded span will
    # show holes or end before the node boundary.
    for insn in insns:
        if insn.address != expected_addr:
            logger.warning(
                f"CFG anomaly for function {node.function_address:#x}: malformed_block at "
                f"{node.addr:#x}: decoded instruction starts at {insn.address:#x} "
                f"instead of expected {expected_addr:#x}"
            )
            return True
        expected_addr += insn.size

    node_end = node.addr + node.size
    if expected_addr != node_end:
        logger.warning(
            f"CFG anomaly for function {node.function_address:#x}: malformed_block at "
            f"{node.addr:#x}: decoded instructions end at {expected_addr:#x}, "
            f"but node size extends to {node_end:#x}"
        )
        return True

    return False


def _has_decode_gap(cfg: CFGBase, func_addr: int) -> bool:
    """Return True when the CFG contains true decoding/lifting failures."""

    if getattr(getattr(cfg, "model", None), "ident", "") == "CFGFastCustom":
        # The custom fallback is intentionally capstone-driven. VEX lifting can
        # still complain about some recovered nodes, but at that point the
        # custom graph should be judged by decoded instruction coverage instead
        # of by whether pyvex likes every block.
        return False

    for node in _iter_function_nodes(cfg, func_addr):
        # Let the weird-graph pass own malformed node boundaries or undecodable
        # capstone streams. Decode gaps are reserved for nodes that look
        # structurally fine but still lift to Ijk_NoDecode in VEX.
        if _has_decoding_coverage_mismatch(node):
            continue

        try:
            jumpkind = node.block.vex.jumpkind
        except AttributeError as exc:
            logger.warning(
                f"Unable to inspect jumpkind for CFG node at {node.addr:#x}: "
                f"{type(exc).__name__}: {exc}"
            )
            continue
        if jumpkind == "Ijk_NoDecode":
            logger.warning(
                f"CFG anomaly for function {func_addr:#x}: decode_gap at {node.addr:#x}: "
                "node ended with Ijk_NoDecode, which points to a lifting/decoding failure"
            )
            return True

    return False


def _has_weird_graph(cfg: CFGBase, func_addr: int) -> bool:
    """Return True when the CFG shows malformed structure without a decode gap."""

    for node in _iter_function_nodes(cfg, func_addr):
        # Disabled for now: unresolved jump-table style indirect jumps are a
        # useful anomaly signal, but CFGEmulated does not currently improve
        # those cases reliably enough to justify an automatic fallback.
        #
        # for succ in cfg.graph.successors(node):
        #     if succ.is_simprocedure and succ.simprocedure_name == "UnresolvableJumpTarget":
        #         logger.warning(
        #             f"CFG anomaly for function {func_addr:#x}: "
        #             f"unresolvable_indirect_jump at {node.addr:#x}: "
        #             "node flows to UnresolvableJumpTarget"
        #         )
        #         return True

        # These malformed CFG nodes are structurally wrong but do not
        # necessarily indicate a real decoding/lifting limitation. CFGEmulated
        # has been able to recover some of them in our corpus.
        if _has_decoding_coverage_mismatch(node):
            return True

    return False


def _log_post_fallback_status(cfg: CFGBase, func_addr: int, cfg_label: str) -> None:
    """Log whether fallback CFG recomputation cleared the known anomalies."""

    has_weird_graph = _has_weird_graph(cfg, func_addr)
    has_decode_gap = _has_decode_gap(cfg, func_addr)
    if not has_weird_graph and not has_decode_gap:
        logger.info(f"{cfg_label} for function {func_addr:#x} no longer shows known CFG anomalies")
    else:
        logger.warning(f"{cfg_label} for function {func_addr:#x} still shows CFG anomalies")


@lru_cache
@time_it
def get_cfg(project: Project, func_addr: int, cfg_mode: CfgMode | None = None) -> CFGBase:
    """Return the CFG for one function according to the configured fallback mode."""

    resolved_cfg_mode = cfg_mode or get_settings().cfg_mode
    logger.info(f"Getting CFG for function {func_addr:#x} with mode '{resolved_cfg_mode}'")
    # Keep one KB per high-level CFG request so a fallback CFGEmulated run can
    # reuse the metadata already discovered by CFGFast, especially comments and
    # related knowledge attached during the fast analysis.
    kb = KnowledgeBase(project)

    fast_cfg = _get_fast_cfg(project, kb, func_addr)
    has_decode_gap = _has_decode_gap(fast_cfg, func_addr)
    has_weird_graph = _has_weird_graph(fast_cfg, func_addr)

    if resolved_cfg_mode == "none":
        return fast_cfg

    if resolved_cfg_mode == "custom":
        if has_decode_gap or has_weird_graph:
            logger.warning(
                f"CFGFast produced anomalies for {func_addr:#x}; "
                "building custom CFG fallback"
            )
            custom_cfg = build_custom_cfg(project, kb, func_addr, fast_cfg)
            _log_post_fallback_status(custom_cfg, func_addr, "Custom CFG result")
            return custom_cfg
        return fast_cfg

    if has_decode_gap:
        logger.warning(
            f"CFGFast hit a decoding/lifting gap for {func_addr:#x}. "
            "CFGEmulated is not selected for this anomaly class because it still "
            "depends on VEX. A custom non-pyvex fallback is required."
        )
        return fast_cfg

    if has_weird_graph:
        keep_state = resolved_cfg_mode == "stateful"
        logger.warning(
            f"CFGFast produced a weird graph for {func_addr:#x}; "
            f"retrying with CFGEmulated(keep_state={keep_state})"
        )
        emu_cfg = _get_emu_cfg(project, kb, func_addr, keep_state)
        _log_post_fallback_status(emu_cfg, func_addr, "CFGEmulated result")
        return emu_cfg

    return fast_cfg
