"""
Custom CFG discovery and repair helpers.

This module implements the custom CFG fallback used by `cfg_mode="custom"`.
The current design is intentionally repair-oriented:

1. Run bounded CFGFast first so we keep angr metadata (KB, labels, xrefs,
   function lookup, comments).
2. Detect anomalous nodes in the seed CFGFast graph.
3. Recover replacement basic blocks only for the anomalous spans.
4. Splice those replacements back into the seed graph, leaving already-good
   CFGFast nodes untouched.

The implementation is conservative on purpose. It prefers localized repairs
over whole-function reconstruction so already-correct arch-specific behavior
from CFGFast remains intact.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Literal

from angr import KnowledgeBase, Project
from angr.analyses.cfg import CFGBase
from angr.knowledge_plugins.cfg import CFGModel, CFGNode
from capstone import CS_GRP_CALL, CS_GRP_JUMP, CS_GRP_RET, CS_OP_IMM, CsInsn
from capstone.arm import ARM_CC_AL, ARM_CC_INVALID
from capstone.x86 import X86_INS_JMP, X86_INS_LJMP
from loguru import logger
import networkx as nx
import pyvex

from .symbols import FunctionSymbol, list_function_symbols
from .vis import register_custom_graph


# Internal-only labels describing how a recovered block terminates while we are
# repairing a CFG. These are not written directly to graph edges, because the
# renderer and the rest of the angr-compatible pipeline only expect the usual
# edge jumpkinds such as Ijk_Boring / Ijk_Call / Ijk_FakeRet.
TerminatorKind = Literal["Ijk_Boring", "Ijk_Call", "Ijk_Fallthrough", "Ijk_Ret"]

# Edge jumpkinds that we actually materialize in the repaired graph. Keep this
# aligned with the jumpkind vocabulary used by normal angr CFG edges.
EdgeJumpKind = Literal["Ijk_Boring", "Ijk_Call", "Ijk_FakeRet"]


@dataclass(frozen=True)
class BlockSpec:
    """A basic-block candidate recovered from bounded disassembly."""

    addr: int
    size: int
    instruction_addrs: tuple[int, ...]
    # Internal repair metadata for the recovered block terminator. This can be
    # more descriptive than what we finally store on graph edges.
    jumpkind: TerminatorKind
    direct_targets: tuple[int, ...] = ()
    fallthrough_addr: int | None = None


@dataclass(frozen=True)
class TerminatorInfo:
    """Normalized control-flow summary for one recovered block terminator."""

    jumpkind: TerminatorKind
    direct_targets: tuple[int, ...] = ()
    fallthrough_addr: int | None = None


@dataclass(frozen=True)
class FunctionBounds:
    """Closed-open function bounds derived from the symbol table."""

    addr: int
    end_addr: int
    size: int
    symbol: FunctionSymbol


@dataclass(frozen=True)
class RepairObligation:
    """One block-entry repair task derived from a concrete CFG edge or anomaly."""

    addr: int
    reason: str
    source_addr: int | None = None
    jumpkind: EdgeJumpKind = "Ijk_Boring"


class CustomCFG(SimpleNamespace):
    """Small CFG-like wrapper exposing the attributes bingraph actually uses."""

    graph: object
    model: CFGModel
    functions: object
    kb: KnowledgeBase


class InsnSemantics:
    """
    Small wrapper around a Capstone instruction.

    The custom CFG builder uses Capstone for bounded block recovery, but wants
    higher-level predicates such as "is this a control-transfer instruction?" in
    a place that reads clearly. This helper intentionally stays lightweight:
    it only exposes generic properties needed before we ask VEX for the final
    branch shape of a recovered block.
    """

    def __init__(self, insn: CsInsn):
        self.insn = insn

    @property
    def address(self) -> int:
        return self.insn.address

    @property
    def size(self) -> int:
        return self.insn.size

    def is_ret(self) -> bool:
        return CS_GRP_RET in self.insn.groups

    def is_call(self) -> bool:
        return CS_GRP_CALL in self.insn.groups

    def is_jump(self) -> bool:
        return CS_GRP_JUMP in self.insn.groups

    def is_control_transfer(self) -> bool:
        return self.is_ret() or self.is_call() or self.is_jump()

    def is_conditional_jump(self) -> bool:
        """
        Return True when the instruction is a direct conditional branch.

        We keep this heuristic intentionally narrow: only direct jumps with a
        known branch target are considered here. Conditional branches often
        expose their condition either through an extra operand (for example
        Thumb `cbz r2, #target`) or through an architecture-specific condition
        code even when the target is the only explicit operand (for example
        x86 `jne target` or ARM `bne target`).
        """

        if not self.is_jump() or self.direct_target() is None:
            return False

        if len(self.insn.operands) > 1:
            return True

        arm_cc = getattr(self.insn, "cc", ARM_CC_INVALID)
        if arm_cc not in {ARM_CC_INVALID, ARM_CC_AL}:
            return True

        insn_id = getattr(self.insn, "id", None)
        if insn_id in {X86_INS_JMP, X86_INS_LJMP}:
            return False

        return True

    def direct_target(self) -> int | None:
        # Do not assume the branch target is always operand 0. Instructions
        # such as Thumb `cbz r2, #0x733` place the condition source first and
        # the branch destination second.
        for operand in self.insn.operands:
            if getattr(operand, "type", None) != CS_OP_IMM:
                continue

            imm = getattr(operand, "imm", None)
            if isinstance(imm, int):
                return imm

        return None

    def fallthrough_addr(self) -> int:
        return self.address + self.size


def _lookup_function_bounds(project: Project, func_addr: int) -> FunctionBounds:
    """
    Return function bounds from the symbol view used across bingraph.

    The custom repair pass needs a stable upper bound even when CFGFast itself
    missed blocks. `kb.functions[addr].size` is derived from currently
    discovered CFG blocks, so it can shrink along with a malformed CFG. By
    reusing `list_function_symbols()` we inherit the project's existing symbol
    parsing and size-inference logic instead.
    """

    function = next((sym for sym in list_function_symbols(project) if sym.addr == func_addr), None)
    if function is None:
        raise KeyError(f"Function {func_addr:#x} not found in binary")

    end_addr = func_addr + function.size
    return FunctionBounds(
        addr=func_addr,
        end_addr=end_addr,
        size=end_addr - func_addr,
        symbol=function,
    )


def _is_direct_target_valid(bounds: FunctionBounds, target: int | None) -> bool:
    """Return True when a direct branch target is inside the current function."""

    return target is not None and bounds.addr <= target < bounds.end_addr


def _decode_one(project: Project, addr: int, size: int) -> CsInsn | None:
    """
    Decode one instruction using angr's block factory but only consume the
    Capstone view.

    This preserves architecture mode details such as Thumb decoding while still
    avoiding a dependency on the full VEX lift for ordinary instruction
    discovery.
    """

    block = project.factory.block(
        addr,
        size=size,
        strict_block_end=True,
        cross_insn_opt=False,
    )
    capstone_insns = block.capstone.insns
    return capstone_insns[0].insn if capstone_insns else None


def _arch_has_delay_slot(project: Project) -> bool:
    """Return True for architectures where control transfers consume a delay slot."""

    return project.arch.name in {"MIPS32", "MIPS64"}


def _control_transfer_index(project: Project, block_insns: list[CsInsn]) -> int | None:
    """
    Return the index of the effective control-transfer instruction in a block.

    On most architectures this is simply the last instruction. On delay-slot
    architectures it may be the penultimate instruction, with the final
    instruction being the consumed delay slot.
    """

    has_delay_slot = _arch_has_delay_slot(project)

    for idx in range(len(block_insns) - 1, -1, -1):
        if not InsnSemantics(block_insns[idx]).is_control_transfer():
            continue

        if idx != len(block_insns) - 1 and not has_delay_slot:
            raise RuntimeError(
                f"Recovered non-delay block at {block_insns[0].address:#x} has trailing "
                f"instructions after control transfer {block_insns[idx].address:#x}"
            )
        return idx

    return None


def _lift_block_terminator(project: Project, bounds: FunctionBounds, block_insns: list[CsInsn]) -> TerminatorInfo:
    """
    Lift one recovered block with VEX and derive its control-flow shape.

    Capstone is used to recover the bounded instruction stream, but we defer the
    final branch classification to VEX so we can reuse the same arch-specific
    semantics that CFGFast relies on for conditional-vs-unconditional structure.
    If a recovered control-transfer block cannot be lifted, we raise instead of
    silently falling back to heuristics.
    """

    block_end_addr = block_insns[-1].address + block_insns[-1].size

    term_idx = _control_transfer_index(project, block_insns)

    if term_idx is None:
        next_addr = block_end_addr
        fallthrough_addr = next_addr if next_addr < bounds.end_addr else None
        return TerminatorInfo(jumpkind="Ijk_Fallthrough", fallthrough_addr=fallthrough_addr)

    tail_insns = block_insns[term_idx:]
    last = tail_insns[0]
    semantic = InsnSemantics(last)
    next_addr = block_end_addr

    tail_addr = tail_insns[0].address
    tail_size = sum(insn.size for insn in tail_insns)
    block_addr = block_insns[0].address
    block_size = sum(insn.size for insn in block_insns)

    def _lift(addr: int, size: int):
        return project.factory.block(
            addr,
            size=size,
            strict_block_end=True,
            cross_insn_opt=False,
        ).vex

    try:
        vex = _lift(tail_addr, tail_size)
    except Exception:
        try:
            vex = _lift(block_addr, block_size)
        except Exception as exc:
            raise RuntimeError(
                f"Custom CFG failed lifting control-transfer block at {block_addr:#x} "
                f"(terminator {last.address:#x}: {last.mnemonic} {last.op_str})"
            ) from exc

    exit_targets: list[int] = []
    for ins_addr, _, stmt in vex.exit_statements:
        if ins_addr != last.address:
            continue
        target = getattr(stmt.dst, "value", None)
        if isinstance(target, int):
            exit_targets.append(target)

    default_target = None
    if isinstance(vex.next, pyvex.expr.Const):
        default_target = vex.next.con.value

    if semantic.is_ret():
        return TerminatorInfo(jumpkind="Ijk_Ret")

    if semantic.is_call():
        direct_targets = ()
        if _is_direct_target_valid(bounds, default_target):
            direct_targets = (default_target,)
        fallthrough_addr = next_addr if next_addr < bounds.end_addr else None
        return TerminatorInfo(
            jumpkind="Ijk_Call",
            direct_targets=direct_targets,
            fallthrough_addr=fallthrough_addr,
        )

    # For jumps, trust the VEX control-flow shape instead of trying to
    # re-infer conditionality from Capstone metadata. Conditional branches
    # show up as Exit statements plus a default fallthrough target, while
    # unconditional jumps only expose the default target.
    if exit_targets or semantic.is_conditional_jump():
        all_targets: list[int] = list(exit_targets)
        if isinstance(default_target, int) and default_target not in all_targets:
            all_targets.append(default_target)
        fallthrough_addr = next_addr if next_addr in all_targets and next_addr < bounds.end_addr else None
        return TerminatorInfo(
            jumpkind="Ijk_Boring",
            direct_targets=tuple(target for target in all_targets if target != fallthrough_addr),
            fallthrough_addr=fallthrough_addr,
        )

    if semantic.is_jump() and _is_direct_target_valid(bounds, default_target):
        return TerminatorInfo(
            jumpkind="Ijk_Boring",
            direct_targets=(default_target,),
        )

    direct_target = semantic.direct_target()
    if semantic.is_jump() and direct_target is not None:
        return TerminatorInfo(
            jumpkind="Ijk_Boring",
            direct_targets=(direct_target,),
        )

    direct_targets = ()
    if _is_direct_target_valid(bounds, default_target):
        direct_targets = (default_target,)
    return TerminatorInfo(jumpkind="Ijk_Boring", direct_targets=direct_targets)




def _iter_seed_function_nodes(seed_cfg: CFGBase, func_addr: int):
    """Yield non-simprocedure nodes from the seed CFG for one function."""

    for node in seed_cfg.graph.nodes():
        if getattr(node, "function_address", None) != func_addr:
            continue
        if getattr(node, "is_simprocedure", False):
            continue
        yield node


def iter_function_nodes(cfg: CFGBase, func_addr: int):
    """Yield non-simprocedure nodes that belong to one function."""

    yield from _iter_seed_function_nodes(cfg, func_addr)


def _seed_successor_addrs(seed_cfg: CFGBase, node) -> set[int]:
    """Return the concrete successor addresses for one seed node."""

    return {
        succ.addr
        for succ in seed_cfg.graph.successors(node)
        if not getattr(succ, "is_simprocedure", False)
    }


def _seed_node_expected_successors(node) -> tuple[tuple[int, ...], int | None]:
    """
    Return the local direct targets and fallthrough encoded in one seed node.

    CFGFast may already have stitched a malformed region incorrectly, so when we
    reconnect a preserved predecessor into repaired blocks we prefer the
    predecessor's own lifted block semantics over the old graph edges.
    """

    try:
        insns = [obj.insn for obj in node.block.capstone.insns]
        vex = node.block.vex
    except (AttributeError, KeyError):
        return (), None
    except Exception:
        return (), None

    if not insns:
        return (), None

    last_insn = getattr(insns[-1], "insn", insns[-1])
    last_semantic = InsnSemantics(last_insn)
    if not last_semantic.is_control_transfer():
        return (), None

    last_addrs = {insns[-1].address}

    direct_targets: list[int] = []
    for ins_addr, _, stmt in vex.exit_statements:
        if ins_addr not in last_addrs:
            continue
        target = getattr(stmt.dst, "value", None)
        if isinstance(target, int):
            direct_targets.append(target)

    fallthrough_addr = None
    if isinstance(vex.next, pyvex.expr.Const):
        target = vex.next.con.value
        if isinstance(target, int):
            fallthrough_addr = target

    return tuple(direct_targets), fallthrough_addr


def node_has_missing_conditional_successor(graph: nx.DiGraph, bounds: FunctionBounds, node) -> bool:
    """
    Return True for a direct conditional branch missing its taken external edge.

    This intentionally targets a narrow anomaly class: CFGFast sometimes keeps
    the in-function fallthrough edge of a conditional jump, but silently drops
    the taken edge when that target lives outside the current function bounds.
    We only flag that case, since the broader "missing conditional successor"
    heuristic was too noisy on already-correct graphs.
    """

    try:
        insns = list(node.block.capstone.insns)
        vex = node.block.vex
    except (AttributeError, KeyError):
        return False
    except Exception:
        return False

    if not insns:
        return False

    last = InsnSemantics(insns[-1].insn)
    if not last.is_conditional_jump():
        return False

    exit_targets: list[int] = []
    for ins_addr, _, stmt in vex.exit_statements:
        if ins_addr != insns[-1].address:
            continue
        target = getattr(stmt.dst, "value", None)
        if isinstance(target, int):
            exit_targets.append(target)

    all_targets = list(exit_targets)
    if isinstance(vex.next, pyvex.expr.Const):
        target = vex.next.con.value
        if isinstance(target, int):
            all_targets.append(target)

    if not all_targets:
        return False

    next_addr = insns[-1].address + insns[-1].size
    fallthrough_addr = next_addr if next_addr in all_targets and _is_direct_target_valid(bounds, next_addr) else None
    direct_targets = [target for target in all_targets if target != fallthrough_addr]
    external_targets = [
        target for target in direct_targets if not _is_direct_target_valid(bounds, target)
    ]
    if not external_targets:
        return False

    if fallthrough_addr is None:
        return False

    successor_addrs = {
        succ.addr
        for succ in graph.successors(node)
        if not getattr(succ, "is_simprocedure", False)
    }
    if fallthrough_addr not in successor_addrs:
        return False

    for succ in graph.successors(node):
        if not getattr(succ, "is_simprocedure", False):
            continue
        if succ.addr in external_targets:
            return False

    logger.warning(
        f"Node {node.addr:#x} is missing conditional branch successor(s) "
        f"outside function bounds: {', '.join(hex(t) for t in external_targets)}"
    )
    return True


def node_has_decoding_coverage_mismatch(node) -> bool:
    """Return True when a CFG node clearly covers bytes incorrectly."""

    if node.size == 0:
        logger.warning(f"Node {node.addr:#x} has size zero")
        return True

    try:
        insns = list(node.block.capstone.insns)
    except (AttributeError, KeyError) as exc:
        logger.warning(
            f"Capstone inspection failed for node {node.addr:#x}: "
            f"{type(exc).__name__}: {exc}"
        )
        return True

    expected_addr = node.addr
    for insn in insns:
        if insn.address != expected_addr:
            logger.warning(
                f"Node {node.addr:#x} decodes instruction at {insn.address:#x} "
                f"instead of expected {expected_addr:#x}"
            )
            return True
        expected_addr += insn.size

    node_end = node.addr + node.size
    if expected_addr != node_end:
        logger.warning(
            f"Node {node.addr:#x} decoded instructions end at {expected_addr:#x}, "
            f"but node size extends to {node_end:#x}"
        )
        return True

    return False


def node_has_decode_gap(node) -> bool:
    """
    Return True when a CFG node has a real lifting-only gap worth flagging.

    Once a block's Capstone instruction stream covers the node span exactly, we
    treat it as structurally decodable even if VEX still reports
    `Ijk_NoDecode`. This keeps the anomaly checker focused on malformed blocks
    and missing coverage instead of on VEX-specific complaints for blocks we
    can already render correctly.
    """

    if node_has_decoding_coverage_mismatch(node):
        return False

    try:
        insns = list(node.block.capstone.insns)
    except (AttributeError, KeyError):
        insns = []

    if insns:
        return False

    try:
        return node.block.vex.jumpkind == "Ijk_NoDecode"
    except Exception:
        return False


def node_has_truncated_leaf(cfg: CFGBase, func_addr: int, node) -> bool:
    """Return True when a CFG node stops before a real terminator and has no exits."""

    try:
        if node.block.vex.jumpkind == "Ijk_Ret":
            return False
    except Exception:
        pass

    try:
        insns = list(node.block.capstone.insns)
    except (AttributeError, KeyError):
        return False

    if not insns:
        return False

    last = InsnSemantics(insns[-1].insn)
    if last.is_control_transfer():
        return False

    if any(True for _ in cfg.graph.successors(node)):
        return False

    has_later_function_node = any(
        other is not node
        and getattr(other, "function_address", None) == func_addr
        and not getattr(other, "is_simprocedure", False)
        and other.addr > node.addr
        for other in cfg.graph.nodes()
    )
    return has_later_function_node


def _is_seed_node_anomalous(seed_cfg: CFGBase, bounds: FunctionBounds, func_addr: int, node) -> bool:
    """Return True when a seed CFG node should be locally repaired."""

    return (
        node_has_decoding_coverage_mismatch(node)
        or node_has_decode_gap(node)
        or node_has_truncated_leaf(seed_cfg, func_addr, node)
        or node_has_missing_conditional_successor(seed_cfg.graph, bounds, node)
    )


def _custom_model_marker() -> SimpleNamespace:
    """Return the minimal model metadata currently needed by callers."""

    return SimpleNamespace(ident="CFGFastCustom")


def _prune_orphan_simprocedures(graph: nx.DiGraph) -> None:
    """
    Remove simprocedure nodes that no longer have any incoming edges.

    Local repairs may delete the buggy seed nodes that originally pointed to an
    angr-created simprocedure such as `UnresolvableJumpTarget`. When that
    happens, the simprocedure can survive in the graph as an orphan even though
    no repaired block still reaches it. Prune those leftovers iteratively in
    case removing one orphan exposes another orphaned simprocedure behind it.
    """

    while True:
        orphan_nodes = [
            node
            for node in list(graph.nodes())
            if getattr(node, "is_simprocedure", False) and graph.in_degree(node) == 0
        ]
        if not orphan_nodes:
            return
        _remove_nodes(graph, orphan_nodes)


def _prune_placeholders(graph: nx.DiGraph) -> None:
    """Remove any temporary placeholder nodes left after the repair pass."""

    placeholders = [
        node
        for node in list(graph.nodes())
        if _node_is_placeholder(node)
    ]
    if placeholders:
        _remove_nodes(graph, placeholders)


def _block_name(bounds: FunctionBounds, block: BlockSpec) -> str:
    """Return the function-relative label used for a recovered block."""

    if block.addr == bounds.addr:
        return bounds.symbol.name
    return f"{bounds.symbol.name}+0x{block.addr - bounds.addr:x}"


def _node_intersects_bounds(node, bounds: FunctionBounds) -> bool:
    """Return True when a node overlaps the current function address range."""

    if getattr(node, "is_simprocedure", False):
        return False
    return _ranges_overlap(node.addr, _node_range_end(node), bounds.addr, bounds.end_addr)


def _iter_graph_bound_nodes(graph: nx.DiGraph, bounds: FunctionBounds):
    """
    Yield live graph nodes that overlap the current function bounds.

    Seed CFGFast nodes may carry an incorrect `function_address` once the graph
    goes malformed. The custom repair pass therefore keys all live-graph lookups
    off address bounds, not off the stored function tag.
    """

    for node in graph.nodes():
        if _node_intersects_bounds(node, bounds):
            yield node


def _nodes_at_addr(graph: nx.DiGraph, bounds: FunctionBounds, addr: int) -> list[CFGNode]:
    """Return all non-simprocedure nodes in the function bounds that start at addr."""

    return [
        node
        for node in _iter_graph_bound_nodes(graph, bounds)
        if node.addr == addr
    ]


def _covering_nodes(graph: nx.DiGraph, bounds: FunctionBounds, addr: int) -> list[CFGNode]:
    """Return all non-simprocedure nodes in bounds whose range covers addr."""

    return [
        node
        for node in _iter_graph_bound_nodes(graph, bounds)
        if node.addr <= addr < _node_range_end(node)
    ]


def _node_range_end(node) -> int:
    """Return the closed-open end address of one node."""

    return node.addr + max(getattr(node, "size", 0), 0)


def _ranges_overlap(start_a: int, end_a: int, start_b: int, end_b: int) -> bool:
    """Return True when two closed-open address ranges overlap."""

    return start_a < end_b and start_b < end_a


def _node_is_placeholder(node) -> bool:
    """Return True when the node is a custom placeholder awaiting repair."""

    return getattr(node, "size", 0) == 0 and str(getattr(node, "name", "")).startswith("placeholder_")


def _node_has_forced_split(node, forced_block_starts: set[int]) -> bool:
    """Return True when a known block start falls inside this node's range."""

    node_end = _node_range_end(node)
    return any(node.addr < addr < node_end for addr in forced_block_starts)


def _make_cfg_node(seed_cfg: CFGBase, func_addr: int, bounds: FunctionBounds, block: BlockSpec) -> CFGNode:
    """Instantiate one CFGNode compatible with the existing rendering pipeline."""

    node = CFGNode(
        block.addr,
        block.size,
        cfg=seed_cfg.model,
        function_address=func_addr,
        block_id=block.addr,
        instruction_addrs=block.instruction_addrs,
        name=_block_name(bounds, block),
    )
    return node


def _make_placeholder_node(seed_cfg: CFGBase, func_addr: int, addr: int) -> CFGNode:
    """Create a zero-sized placeholder node for a newly discovered bad address."""

    node = CFGNode(
        addr,
        0,
        cfg=seed_cfg.model,
        function_address=func_addr,
        block_id=addr,
        instruction_addrs=(),
        name=f"placeholder_{addr:#x}",
    )
    return node


def _find_external_target_node(graph: nx.DiGraph, addr: int) -> CFGNode | None:
    """Return an existing synthetic external-target leaf for one address."""

    for node in graph.nodes():
        if not getattr(node, "is_simprocedure", False):
            continue
        if node.addr == addr:
            return node
    return None


def _external_target_name(project: Project, addr: int) -> str:
    """Return the symbol-table name for an external target when available."""

    symbol = project.loader.find_symbol(addr)
    if symbol is None:
        return f"ExternalTarget_{addr:#x}"

    name = getattr(symbol, "name", None)
    return name if isinstance(name, str) and name else f"ExternalTarget_{addr:#x}"


def _make_external_target_node(seed_cfg: CFGBase, func_addr: int, addr: int) -> CFGNode:
    """Create a synthetic leaf node for a branch target outside the function."""

    name = _external_target_name(seed_cfg.project, addr)

    return CFGNode(
        addr,
        0,
        cfg=seed_cfg.model,
        simprocedure_name=name,
        function_address=func_addr,
        block_id=addr,
        instruction_addrs=(),
        name=name,
    )


def _ensure_external_target_node(seed_cfg: CFGBase, graph: nx.DiGraph, func_addr: int, addr: int) -> CFGNode:
    """Get or create one synthetic external-target leaf node."""

    node = _find_external_target_node(graph, addr)
    if node is not None:
        return node

    node = _make_external_target_node(seed_cfg, func_addr, addr)
    graph.add_node(node)
    return node


def _node_is_acceptable(
    seed_cfg: CFGBase,
    graph: nx.DiGraph,
    bounds: FunctionBounds,
    func_addr: int,
    forced_block_starts: set[int],
    node,
) -> bool:
    """Return True when an existing node can stay as-is in the repaired graph."""

    if _node_is_placeholder(node):
        return False
    if _node_has_forced_split(node, forced_block_starts):
        return False
    if getattr(node, "function_address", None) != func_addr:
        # Malformed CFGFast regions can leave behind in-bounds nodes that are
        # spuriously attached to a predecessor's address instead of the real
        # function entry. Treat those as stale so the repair pass can replace
        # them with canonical nodes owned by the repaired function.
        return False
    if node_has_decoding_coverage_mismatch(node):
        return False
    try:
        capstone_insns = list(node.block.capstone.insns)
    except (AttributeError, KeyError):
        capstone_insns = []

    # Keep the worklist repairing nodes that were already classified as
    # structurally incomplete by the seed anomaly checks. Otherwise an initial
    # bad block like __strcmp_sse4_2+0x37 can survive forever just because its
    # byte coverage looks locally self-consistent.
    if node_has_truncated_leaf(SimpleNamespace(graph=graph), func_addr, node):
        return False

    if node_has_decode_gap(node):
        return False
    if node_has_missing_conditional_successor(graph, bounds, node):
        return False
    return True


def _recover_block(project: Project, bounds: FunctionBounds, start_addr: int, stop_addrs: set[int]) -> BlockSpec | None:
    """Decode one block starting at addr and stop on control flow or known block starts."""

    max_inst_bytes = getattr(project.arch, "max_inst_bytes", 16)
    cur = start_addr
    insns: list[CsInsn] = []
    has_delay_slot = _arch_has_delay_slot(project)

    while bounds.addr <= cur < bounds.end_addr:
        if insns and cur in stop_addrs:
            break

        insn = _decode_one(project, cur, max_inst_bytes)
        if insn is None:
            logger.warning(f"Custom CFG could not decode instruction at {cur:#x}")
            break

        insns.append(insn)
        semantic = InsnSemantics(insn)
        next_addr = insn.address + insn.size

        if semantic.is_control_transfer():
            if has_delay_slot and bounds.addr <= next_addr < bounds.end_addr:
                delay_insn = _decode_one(project, next_addr, max_inst_bytes)
                if delay_insn is not None:
                    insns.append(delay_insn)
            break

        cur = next_addr

    if not insns:
        return None

    terminator = _lift_block_terminator(project, bounds, insns)
    return BlockSpec(
        addr=insns[0].address,
        size=sum(obj.size for obj in insns),
        instruction_addrs=tuple(obj.address for obj in insns),
        jumpkind=terminator.jumpkind,
        direct_targets=terminator.direct_targets,
        fallthrough_addr=terminator.fallthrough_addr,
    )


def _enqueue_obligation(
    seed_cfg: CFGBase,
    graph: nx.DiGraph,
    bounds: FunctionBounds,
    func_addr: int,
    forced_block_starts: set[int],
    obligation: RepairObligation,
    queue: deque[RepairObligation],
    queued: set[int],
    repaired_nodes: set[CFGNode],
) -> CFGNode | None:
    """
    Ensure one repair obligation is represented and scheduled for later repair.

    Unlike the previous address-only queue, each work item corresponds to a
    concrete reason why this address should exist as a block entry. When a
    source node is known we also materialize its outgoing edge immediately,
    either to an existing node or to a placeholder, so later replacement does
    not need to guess how recovered predecessors should reconnect.
    """

    addr = obligation.addr
    if not (bounds.addr <= addr < bounds.end_addr):
        return None

    existing_nodes = _nodes_at_addr(graph, bounds, addr)
    covering_nodes = _covering_nodes(graph, bounds, addr)
    for node in covering_nodes:
        if node.addr != addr and not _node_is_placeholder(node):
            forced_block_starts.add(addr)
            placeholder = next((item for item in existing_nodes if _node_is_placeholder(item)), None)
            if placeholder is None:
                placeholder = _make_placeholder_node(seed_cfg, func_addr, addr)
                graph.add_node(placeholder)
            if obligation.source_addr is not None:
                src_nodes = _nodes_at_addr(graph, bounds, obligation.source_addr)
                if src_nodes:
                    _add_successor_edge(graph, src_nodes[0], placeholder, obligation.jumpkind)
            if node.addr not in queued:
                queue.append(
                    RepairObligation(
                        addr=node.addr,
                        reason=f"split_for_{addr:#x}",
                    )
                )
                queued.add(node.addr)
            return placeholder

    for node in existing_nodes:
        if _node_is_acceptable(seed_cfg, graph, bounds, func_addr, forced_block_starts, node):
            if obligation.source_addr is not None:
                src_nodes = _nodes_at_addr(graph, bounds, obligation.source_addr)
                if src_nodes:
                    _add_successor_edge(graph, src_nodes[0], node, obligation.jumpkind)
            return node

    for node in existing_nodes:
        if _node_is_placeholder(node):
            if obligation.source_addr is not None:
                src_nodes = _nodes_at_addr(graph, bounds, obligation.source_addr)
                if src_nodes:
                    _add_successor_edge(graph, src_nodes[0], node, obligation.jumpkind)
            if addr not in queued:
                queue.append(obligation)
                queued.add(addr)
            return node

    placeholder = _make_placeholder_node(seed_cfg, func_addr, addr)
    graph.add_node(placeholder)
    if obligation.source_addr is not None:
        src_nodes = _nodes_at_addr(graph, bounds, obligation.source_addr)
        if src_nodes:
            _add_successor_edge(graph, src_nodes[0], placeholder, obligation.jumpkind)
    if addr not in queued:
        queue.append(obligation)
        queued.add(addr)
    return placeholder


def _add_successor_edge(
    graph: nx.DiGraph,
    src: CFGNode,
    dst: CFGNode,
    jumpkind: EdgeJumpKind,
) -> None:
    """Add one successor edge if it is not already present with the same kind."""

    if graph.has_edge(src, dst):
        edge_data = graph.get_edge_data(src, dst) or {}
        if edge_data.get("jumpkind") == jumpkind:
            return
    graph.add_edge(src, dst, jumpkind=jumpkind)


def _wire_expected_successors(
    seed_cfg: CFGBase,
    graph: nx.DiGraph,
    bounds: FunctionBounds,
    func_addr: int,
    forced_block_starts: set[int],
    src,
    queue: deque[RepairObligation],
    queued: set[int],
    repaired_nodes: set[CFGNode],
) -> None:
    """
    Recreate the successor edges implied by one preserved predecessor block.

    This is the key step that keeps good CFGFast predecessors intact while still
    fixing their outgoing edges when one of their old targets was malformed or
    missing.
    """

    direct_targets, fallthrough_addr = _seed_node_expected_successors(src)
    for target in direct_targets:
        _enqueue_obligation(
            seed_cfg,
            graph,
            bounds,
            func_addr,
            forced_block_starts,
            RepairObligation(
                addr=target,
                reason=f"expected_successor_of_{src.addr:#x}",
                source_addr=src.addr,
                jumpkind="Ijk_Boring",
            ),
            queue,
            queued,
            repaired_nodes,
        )

    if fallthrough_addr is not None:
        _enqueue_obligation(
            seed_cfg,
            graph,
            bounds,
            func_addr,
            forced_block_starts,
            RepairObligation(
                addr=fallthrough_addr,
                reason=f"expected_fallthrough_of_{src.addr:#x}",
                source_addr=src.addr,
                jumpkind="Ijk_Boring",
            ),
            queue,
            queued,
            repaired_nodes,
        )


def _incoming_expected_starts(graph: nx.DiGraph, bounds: FunctionBounds, addr: int) -> set[int]:
    """
    Return extra in-function block starts implied by predecessors of `addr`.

    This is used before recovering one block so we do not swallow a sibling
    successor that the current graph already expects from a preserved
    predecessor. In `__udivmoddi4`, for example, the predecessor at `0x7e9`
    already tells us that `0x7ed` is a real block entry alongside the
    fallthrough at `0x7eb`.
    """

    starts: set[int] = set()
    seed_nodes = _nodes_at_addr(graph, bounds, addr) + _covering_nodes(graph, bounds, addr)
    for node in seed_nodes:
        for pred in graph.predecessors(node):
            if getattr(pred, "is_simprocedure", False):
                continue
            direct_targets, fallthrough_addr = _seed_node_expected_successors(pred)
            for target in direct_targets:
                if addr < target < bounds.end_addr:
                    starts.add(target)
            if fallthrough_addr is not None and addr < fallthrough_addr < bounds.end_addr:
                starts.add(fallthrough_addr)

    return starts


def _splice_block(
    seed_cfg: CFGBase,
    graph: nx.DiGraph,
    bounds: FunctionBounds,
    func_addr: int,
    forced_block_starts: set[int],
    block: BlockSpec,
    queue: deque[RepairObligation],
    queued: set[int],
    repaired_nodes: set[CFGNode],
) -> CFGNode:
    """
    Replace every stale overlapping node with one recovered block.

    Incoming edges from preserved predecessors are rewired after the stale nodes
    are removed. If a predecessor itself remains malformed it can still be
    queued later; local repairs do not need to solve every adjacent anomaly in
    one pass.
    """

    recovered_start = block.addr
    recovered_end = block.addr + block.size

    removed_nodes = [
        node
        for node in _iter_graph_bound_nodes(graph, bounds)
        if node.addr == recovered_start
        or _ranges_overlap(node.addr, _node_range_end(node), recovered_start, recovered_end)
        or (_node_is_placeholder(node) and node.addr == recovered_start)
    ]
    removed_set = set(removed_nodes)

    incoming_edges = [
        (src, dst, dict(data))
        for src, dst, data in list(graph.edges(data=True))
        if dst in removed_set and src not in removed_set
    ]

    _remove_nodes(graph, removed_nodes)

    recovered_node = _make_cfg_node(seed_cfg, func_addr, bounds, block)
    graph.add_node(recovered_node)
    repaired_nodes.add(recovered_node)

    for pred, _, data in incoming_edges:
        # Preserve the original incoming edge first. Recovered blocks replace a
        # stale node at the same entry address, so predecessors that already
        # pointed to that entry should continue to do so after the splice.
        _add_successor_edge(graph, pred, recovered_node, data.get("jumpkind", "Ijk_Boring"))

        if pred in repaired_nodes:
            continue

        if not _node_is_acceptable(seed_cfg, graph, bounds, func_addr, forced_block_starts, pred):
            continue

        # Then re-derive any additional sibling successors implied by the
        # predecessor semantics. This still lets preserved nodes discover fresh
        # split targets without relying on re-derivation for the replaced edge.
        _wire_expected_successors(
            seed_cfg,
            graph,
            bounds,
            func_addr,
            forced_block_starts,
            pred,
            queue,
            queued,
            repaired_nodes,
        )

    for target in block.direct_targets:
        if not _is_direct_target_valid(bounds, target):
            leaf = _ensure_external_target_node(seed_cfg, graph, func_addr, target)
            _add_successor_edge(
                graph,
                recovered_node,
                leaf,
                "Ijk_Call" if block.jumpkind == "Ijk_Call" else "Ijk_Boring",
            )
            continue
        _enqueue_obligation(
            seed_cfg,
            graph,
            bounds,
            func_addr,
            forced_block_starts,
            RepairObligation(
                addr=target,
                reason=f"direct_target_of_{block.addr:#x}",
                source_addr=block.addr,
                jumpkind="Ijk_Call" if block.jumpkind == "Ijk_Call" else "Ijk_Boring",
            ),
            queue,
            queued,
            repaired_nodes,
        )

    if block.fallthrough_addr is not None:
        _enqueue_obligation(
            seed_cfg,
            graph,
            bounds,
            func_addr,
            forced_block_starts,
            RepairObligation(
                addr=block.fallthrough_addr,
                reason=f"fallthrough_of_{block.addr:#x}",
                source_addr=block.addr,
                jumpkind="Ijk_FakeRet" if block.jumpkind == "Ijk_Call" else "Ijk_Boring",
            ),
            queue,
            queued,
            repaired_nodes,
        )

    return recovered_node


def _cleanup_unreachable_function_nodes(graph: nx.DiGraph, bounds: FunctionBounds, func_addr: int) -> None:
    """Remove nodes in one function that are unreachable from the entry node."""

    entry_nodes = _nodes_at_addr(graph, bounds, func_addr)
    if not entry_nodes:
        return

    reachable: set[object] = set()
    queue: deque[object] = deque(entry_nodes)

    while queue:
        node = queue.popleft()
        if node in reachable:
            continue
        reachable.add(node)
        for succ in graph.successors(node):
            queue.append(succ)

    stale_nodes = [
        node
        for node in list(graph.nodes())
        if _node_intersects_bounds(node, bounds) and node not in reachable
    ]
    if stale_nodes:
        for node in stale_nodes:
            graph.remove_node(node)


def _remove_nodes(graph, nodes: list[object]) -> None:
    """Remove a batch of nodes through the graph wrapper's public API."""

    for node in nodes:
        graph.remove_node(node)


def _repair_cfg_with_worklist(project: Project, seed_cfg: CFGBase, func_addr: int) -> CFGBase:
    """Repair only anomalous CFGFast regions by materializing blocks on demand."""

    bounds = _lookup_function_bounds(project, func_addr)
    # Mutate the live SpillingCFG wrapper in place, but stay on its public API.
    # Its private backing graph stores tuple keys that the renderer cannot
    # consume directly.
    graph = seed_cfg.graph
    queue: deque[RepairObligation] = deque()
    queued: set[int] = set()
    repaired_nodes: set[CFGNode] = set()
    forced_block_starts: set[int] = set()
    processed_counts: dict[int, int] = {}
    iterations = 0

    initial_bad_addrs = sorted(
        {
            node.addr
            for node in _iter_seed_function_nodes(seed_cfg, func_addr)
            if _is_seed_node_anomalous(seed_cfg, bounds, func_addr, node)
        }
    )
    if not initial_bad_addrs:
        return seed_cfg

    logger.info(
        f"Repairing seed CFG for function {func_addr:#x} with "
        f"{len(initial_bad_addrs)} anomalous block start(s)"
    )

    for addr in initial_bad_addrs:
        _enqueue_obligation(
            seed_cfg,
            graph,
            bounds,
            func_addr,
            forced_block_starts,
            RepairObligation(addr=addr, reason="seed_anomaly"),
            queue,
            queued,
            repaired_nodes,
        )

    while queue:
        iterations += 1
        if iterations > 5000:
            raise RuntimeError(
                f"Custom CFG worklist exceeded 5000 iterations for {func_addr:#x}; "
                f"top counts: {processed_counts}"
            )

        obligation = queue.popleft()
        addr = obligation.addr
        queued.discard(addr)
        processed_counts[addr] = processed_counts.get(addr, 0) + 1
        if processed_counts[addr] <= 5:
            logger.info(
                f"Custom CFG processing {addr:#x} for function {func_addr:#x} "
                f"(visit {processed_counts[addr]})"
            )

        if not (bounds.addr <= addr < bounds.end_addr):
            continue

        current_nodes = _nodes_at_addr(graph, bounds, addr)
        if any(node.addr != addr and not _node_is_placeholder(node) for node in _covering_nodes(graph, bounds, addr)):
            continue
        if any(_node_is_acceptable(seed_cfg, graph, bounds, func_addr, forced_block_starts, node) for node in current_nodes):
            continue

        stop_addrs = {
            node.addr
            for node in _iter_graph_bound_nodes(graph, bounds)
            if node.addr != addr
            and _node_is_acceptable(seed_cfg, graph, bounds, func_addr, forced_block_starts, node)
        }
        stop_addrs.update(
            target
            for target in forced_block_starts
            if addr < target < bounds.end_addr
        )
        stop_addrs.update(_incoming_expected_starts(graph, bounds, addr))
        block = _recover_block(project, bounds, addr, stop_addrs)
        if block is None:
            logger.warning(f"Custom CFG could not recover a block at {addr:#x}")
            continue

        _splice_block(
            seed_cfg,
            graph,
            bounds,
            func_addr,
            forced_block_starts,
            block,
            queue,
            queued,
            repaired_nodes,
        )

    _cleanup_unreachable_function_nodes(graph, bounds, func_addr)
    _prune_placeholders(graph)
    _prune_orphan_simprocedures(graph)

    for node in graph.nodes():
        register_custom_graph(node, graph)

    return CustomCFG(graph=graph, model=_custom_model_marker(), functions=seed_cfg.functions, kb=seed_cfg.kb)


def build_custom_cfg(
    project: Project,
    kb: KnowledgeBase,
    func_addr: int,
    seed_cfg: CFGBase,
) -> CFGBase:
    """
    Build a custom repaired CFG for one function starting from CFGFast output.

    The KB parameter is kept explicit because the long-term plan is to keep the
    custom path anchored to the same knowledge-base state as CFGFast, even if we
    later enrich the repair process with extra metadata.
    """

    logger.info(f"Building custom CFG for function {func_addr:#x}")
    _ = kb
    return _repair_cfg_with_worklist(project, seed_cfg, func_addr)
