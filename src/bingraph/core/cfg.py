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

from dataclasses import dataclass, field
from pathlib import Path
import sys
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
class EdgeSpec:
    """A conservative edge discovered by the custom block walker."""

    src_addr: int
    dst_addr: int
    jumpkind: EdgeJumpKind


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


@dataclass
class RepairPlan:
    """Container describing how a custom repair should rewrite one CFG span."""

    func_addr: int
    bounds: FunctionBounds
    region_start: int
    region_end: int
    replaced_node_addrs: tuple[int, ...] = ()
    block_specs: list[BlockSpec] = field(default_factory=list)
    edge_specs: list[EdgeSpec] = field(default_factory=list)
    reasons: list[str] = field(default_factory=list)


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
    """Return the symbol-bounded address range for one function."""

    function = next((sym for sym in list_function_symbols(project) if sym.addr == func_addr), None)
    if function is None:
        raise KeyError(f"Function {func_addr:#x} not found in binary")

    return FunctionBounds(
        addr=function.addr,
        end_addr=function.addr + function.size,
        size=function.size,
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


def _is_decode_barrier(project: Project, insn: CsInsn) -> bool:
    """
    Return True when an instruction is decodable by Capstone but terminal for VEX.

    Thumb `udf` is the motivating example in our corpus: Capstone decodes it
    just fine, but VEX reports `Ijk_NoDecode`. Treating it as a normal
    instruction makes it absorb following bytes into the same custom block.
    """

    try:
        jumpkind = project.factory.block(
            insn.address,
            size=insn.size,
            strict_block_end=True,
            cross_insn_opt=False,
        ).vex.jumpkind
    except Exception:
        return False

    return jumpkind == "Ijk_NoDecode"


def _decode_region_linear(
    project: Project,
    bounds: FunctionBounds,
    start_addr: int,
    end_addr: int,
) -> list[CsInsn]:
    """
    Decode one address range with a small conservative worklist.

    A plain linear sweep is not enough for repaired spans that contain a taken
    branch into bytes located after a fallthrough path ending in `ret`. In that
    situation we still want to decode the taken target block even though the
    straight-line path stops early. We therefore keep a tiny queue of direct
    in-function branch targets while still staying bounded to a local tail
    window rooted at the anomalous span.
    """

    max_inst_bytes = getattr(project.arch, "max_inst_bytes", 16)
    tail_limit = end_addr + 0x100
    decoded_by_addr: dict[int, CsInsn] = {}
    queue: list[int] = [start_addr]

    while queue:
        cur = min(queue)
        queue.remove(cur)

        while cur < tail_limit and cur not in decoded_by_addr:
            # Let the decoder see a full instruction window even when the current
            # instruction begins near the repair-span end. Otherwise variable-width
            # ISAs such as x86 may fail to decode an instruction whose start is
            # inside the span but whose bytes extend slightly past `end_addr`.
            insn = _decode_one(project, cur, max_inst_bytes)
            if insn is None:
                logger.warning(
                    f"Capstone failed decoding repair span [{start_addr:#x}, {end_addr:#x}) "
                    f"at {cur:#x}; stopping local sweep branch"
                )
                break

            decoded_by_addr[cur] = insn
            semantic = InsnSemantics(insn)
            target = semantic.direct_target()
            if _is_direct_target_valid(bounds, target) and start_addr <= target < tail_limit:
                if target not in decoded_by_addr and target not in queue:
                    queue.append(target)

            next_addr = insn.address + insn.size
            if _is_decode_barrier(project, insn):
                if start_addr <= next_addr < tail_limit:
                    if next_addr not in decoded_by_addr and next_addr not in queue:
                        queue.append(next_addr)
                break
            if semantic.is_ret():
                break
            if semantic.is_call():
                cur = next_addr
                continue
            if semantic.is_jump():
                if semantic.is_conditional_jump():
                    cur = next_addr
                    continue
                break
            cur = next_addr

    return [decoded_by_addr[addr] for addr in sorted(decoded_by_addr)]


def _arch_has_delay_slot(project: Project) -> bool:
    """Return True for architectures where control transfers consume a delay slot."""

    return project.arch.name in {"MIPS32", "MIPS64"}


def _candidate_block_starts(
    project: Project,
    bounds: FunctionBounds,
    insns: list[CsInsn],
    entry_addr: int,
) -> set[int]:
    """Return the conservative set of block starts for one repair span."""

    starts = {entry_addr}
    has_delay_slot = _arch_has_delay_slot(project)

    for idx, insn in enumerate(insns):
        semantic = InsnSemantics(insn)
        target = semantic.direct_target()

        if _is_direct_target_valid(bounds, target):
            starts.add(target)

        if _is_decode_barrier(project, insn):
            next_addr = insn.address + insn.size
            if next_addr < bounds.end_addr:
                starts.add(next_addr)
            continue

        if not semantic.is_control_transfer():
            continue

        continuation_idx = idx + 1
        if has_delay_slot:
            continuation_idx += 1

        if continuation_idx < len(insns):
            next_addr = insns[continuation_idx].address
            # Start a fresh block after any control-transfer instruction. VEX
            # will later tell us whether this boundary is a real fallthrough or
            # just decodable code after a terminating jump. On delay-slot
            # architectures the continuation begins after the consumed slot.
            starts.add(next_addr)

    return starts


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

    if _is_decode_barrier(project, block_insns[-1]):
        fallthrough_addr = block_end_addr if block_end_addr < bounds.end_addr else None
        return TerminatorInfo(jumpkind="Ijk_Boring", fallthrough_addr=fallthrough_addr)

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
        if isinstance(target, int) and _is_direct_target_valid(bounds, target):
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
    if exit_targets:
        fallthrough_addr = default_target if _is_direct_target_valid(bounds, default_target) else None
        return TerminatorInfo(
            jumpkind="Ijk_Boring",
            direct_targets=tuple(exit_targets),
            fallthrough_addr=fallthrough_addr,
        )

    if semantic.is_jump() and _is_direct_target_valid(bounds, default_target):
        return TerminatorInfo(
            jumpkind="Ijk_Boring",
            direct_targets=(default_target,),
        )

    direct_target = semantic.direct_target()
    if semantic.is_jump() and _is_direct_target_valid(bounds, direct_target):
        return TerminatorInfo(
            jumpkind="Ijk_Boring",
            direct_targets=(direct_target,),
        )

    direct_targets = ()
    if _is_direct_target_valid(bounds, default_target):
        direct_targets = (default_target,)
    return TerminatorInfo(jumpkind="Ijk_Boring", direct_targets=direct_targets)


def _build_block_specs(
    project: Project,
    bounds: FunctionBounds,
    insns: list[CsInsn],
    entry_addr: int,
) -> tuple[list[BlockSpec], list[EdgeSpec]]:
    """Split one repair span into conservative basic blocks and edges."""

    if not insns:
        return [], []

    start_set = _candidate_block_starts(project, bounds, insns, entry_addr)
    block_specs: list[BlockSpec] = []
    has_delay_slot = _arch_has_delay_slot(project)

    idx = 0
    while idx < len(insns):
        insn = insns[idx]
        if insn.address not in start_set:
            idx += 1
            continue

        block_insns = [insn]
        j = idx
        while True:
            last = block_insns[-1]
            next_idx = j + 1
            semantic = InsnSemantics(last)
            if _is_decode_barrier(project, last):
                break
            if semantic.is_control_transfer():
                if has_delay_slot and next_idx < len(insns):
                    # Keep the consumed delay-slot instruction in the same
                    # recovered block as the branch/jump that owns it.
                    block_insns.append(insns[next_idx])
                    j = next_idx
                break
            if next_idx >= len(insns):
                break
            next_insn = insns[next_idx]
            if next_insn.address in start_set:
                break
            block_insns.append(next_insn)
            j = next_idx

        terminator = _lift_block_terminator(project, bounds, block_insns)

        block_specs.append(
            BlockSpec(
                addr=block_insns[0].address,
                size=sum(obj.size for obj in block_insns),
                instruction_addrs=tuple(obj.address for obj in block_insns),
                jumpkind=terminator.jumpkind,
                direct_targets=terminator.direct_targets,
                fallthrough_addr=terminator.fallthrough_addr,
            )
        )
        idx = j + 1

    block_addrs = {block.addr for block in block_specs}
    edge_specs: list[EdgeSpec] = []
    for block in block_specs:
        for target in block.direct_targets:
            if target in block_addrs:
                edge_jumpkind: EdgeJumpKind = "Ijk_Call" if block.jumpkind == "Ijk_Call" else "Ijk_Boring"
                edge_specs.append(EdgeSpec(block.addr, target, edge_jumpkind))

        if block.fallthrough_addr is not None and block.fallthrough_addr in block_addrs:
            # Renderers expect graph edges to use the jumpkind values that
            # CFGFast typically stores on edges, not our richer internal block
            # terminator labels.
            edge_jumpkind: EdgeJumpKind = "Ijk_FakeRet" if block.jumpkind == "Ijk_Call" else "Ijk_Boring"
            edge_specs.append(EdgeSpec(block.addr, block.fallthrough_addr, edge_jumpkind))

    return block_specs, edge_specs


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
    """Return True when a CFG node lifts to Ijk_NoDecode."""

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


def _is_seed_node_anomalous(seed_cfg: CFGBase, func_addr: int, node) -> bool:
    """Return True when a seed CFG node should be locally repaired."""

    return (
        node_has_decoding_coverage_mismatch(node)
        or node_has_decode_gap(node)
        or node_has_truncated_leaf(seed_cfg, func_addr, node)
    )


def _build_repair_spans(
    seed_cfg: CFGBase,
    bounds: FunctionBounds,
    func_addr: int,
) -> list[tuple[int, int, tuple[int, ...], list[str]]]:
    """Group anomalous seed nodes into bounded repair spans."""

    nodes = sorted(_iter_seed_function_nodes(seed_cfg, func_addr), key=lambda node: node.addr)
    if not nodes:
        return []

    spans: list[tuple[int, int, tuple[int, ...], list[str]]] = []
    bad_group: list[object] = []

    def flush_group() -> None:
        nonlocal bad_group
        if not bad_group:
            return

        first_bad_addr = bad_group[0].addr
        # Keep the repair span rooted at the first malformed node. Expanding
        # backward into an already-good predecessor forces us to rediscover and
        # relift a block whose CFG semantics are already correct, which is how
        # we ended up regressing nodes such as 0x7e9 in __udivmoddi4.
        start = first_bad_addr

        end = bounds.end_addr
        last_bad_addr = bad_group[-1].addr
        for node in nodes:
            if node.addr > last_bad_addr and not _is_seed_node_anomalous(seed_cfg, func_addr, node):
                end = node.addr
                break

        reasons: list[str] = []
        for node in bad_group:
            if node_has_decoding_coverage_mismatch(node):
                reasons.append(f"malformed_node@{node.addr:#x}")
            if node_has_decode_gap(node):
                reasons.append(f"decode_gap@{node.addr:#x}")
            if node_has_truncated_leaf(seed_cfg, func_addr, node):
                reasons.append(f"truncated_leaf@{node.addr:#x}")

        replaced_node_addrs = tuple(
            node.addr for node in nodes if start <= node.addr < end
        )
        spans.append((start, end, replaced_node_addrs, reasons))
        bad_group = []

    for node in nodes:
        if _is_seed_node_anomalous(seed_cfg, func_addr, node):
            bad_group.append(node)
            continue
        flush_group()

    flush_group()
    return spans


def discover_repair_plans(project: Project, seed_cfg: CFGBase, func_addr: int) -> list[RepairPlan]:
    """Discover local repair plans for the anomalous regions of one function."""

    bounds = _lookup_function_bounds(project, func_addr)
    spans = _build_repair_spans(seed_cfg, bounds, func_addr)
    plans: list[RepairPlan] = []

    for region_start, region_end, replaced_node_addrs, reasons in spans:
        logger.info(
            f"Discovering custom repair blocks for function {func_addr:#x} "
            f"in span [{region_start:#x}, {region_end:#x})"
        )

        insns = _decode_region_linear(project, bounds, region_start, region_end)
        block_specs, edge_specs = _build_block_specs(project, bounds, insns, region_start)
        # Keep the repair plan itself strictly local to the nominal anomalous
        # span. Any edge that leaves the span will be reconnected later against
        # surviving seed nodes during the graph-surgery step.
        retained_addrs = {
            block.addr
            for block in block_specs
            if region_start <= block.addr < region_end
        }
        block_specs = [block for block in block_specs if block.addr in retained_addrs]
        edge_specs = [
            edge
            for edge in edge_specs
            if edge.src_addr in retained_addrs and edge.dst_addr in retained_addrs
        ]
        plan = RepairPlan(
            func_addr=func_addr,
            bounds=bounds,
            region_start=region_start,
            region_end=region_end,
            replaced_node_addrs=replaced_node_addrs,
        )
        plan.block_specs.extend(block_specs)
        plan.edge_specs.extend(edge_specs)
        plan.reasons.extend(reasons)
        plan.reasons.append("bounded_capstone_span_sweep")
        plans.append(plan)

    return plans


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
        graph.remove_nodes_from(orphan_nodes)


def _block_name(bounds: FunctionBounds, block: BlockSpec) -> str:
    """Return the function-relative label used for a recovered block."""

    if block.addr == bounds.addr:
        return bounds.symbol.name
    return f"{bounds.symbol.name}+0x{block.addr - bounds.addr:x}"


def repair_cfg_from_blocks(seed_cfg: CFGBase, plans: list[RepairPlan]) -> CFGBase:
    """
    Repair a seed angr CFG using previously discovered local block plans.

    The repair path preserves the seed graph wholesale, removes only anomalous
    nodes inside the planned spans, and splices custom replacement blocks into
    those regions.
    """

    if not plans:
        return seed_cfg

    logger.info(
        f"Repairing seed CFG for function {plans[0].func_addr:#x} with "
        f"{len(plans)} local custom span(s)"
    )

    for plan in plans:
        dump_repair_plan(plan)
    #return seed_cfg

    graph = nx.DiGraph()
    graph.add_nodes_from(seed_cfg.graph.nodes())
    for src, dst, data in seed_cfg.graph.edges(data=True):
        graph.add_edge(src, dst, **dict(data))

    for plan in plans:
        replacement_cover_end = max(
            plan.region_end,
            max((block.addr + block.size for block in plan.block_specs), default=plan.region_end),
        )
        nodes_in_span = [
            node
            for node in list(graph.nodes())
            if not getattr(node, "is_simprocedure", False)
            and plan.region_start <= node.addr < replacement_cover_end
        ]

        removed_nodes = set(nodes_in_span)
        removed_addrs = {node.addr for node in nodes_in_span}

        incoming_edges = []
        outgoing_edges = []

        for src, dst, data in list(graph.edges(data=True)):
            if dst in removed_nodes and src not in removed_nodes:
                incoming_edges.append((src, dst, dict(data)))
            if src in removed_nodes and dst not in removed_nodes:
                outgoing_edges.append((src, dst, dict(data)))

        graph.remove_nodes_from(nodes_in_span)

        replacement_nodes_by_addr: dict[int, CFGNode] = {}
        for block in plan.block_specs:
            node = CFGNode(
                block.addr,
                block.size,
                cfg=seed_cfg.model,
                function_address=plan.func_addr,
                block_id=block.addr,
                instruction_addrs=block.instruction_addrs,
                name=_block_name(plan.bounds, block),
            )
            graph.add_node(node)
            replacement_nodes_by_addr[block.addr] = node

        for edge in plan.edge_specs:
            src = replacement_nodes_by_addr.get(edge.src_addr)
            dst = replacement_nodes_by_addr.get(edge.dst_addr)
            if src is None or dst is None:
                continue
            graph.add_edge(src, dst, jumpkind=edge.jumpkind)

        if replacement_nodes_by_addr:
            first_replacement = replacement_nodes_by_addr[min(replacement_nodes_by_addr)]
        else:
            first_replacement = None

        outgoing_by_addr: dict[int, list[tuple[object, object, dict]]] = {}
        for old_src, old_dst, data in outgoing_edges:
            outgoing_by_addr.setdefault(old_dst.addr, []).append((old_src, old_dst, data))

        # Preserve entry into the repaired span. Prefer the preserved
        # predecessor's own block semantics over the old CFGFast edge set,
        # because a malformed span may already have been stitched with missing
        # targets (for example Thumb `cbnz` reaching only its fallthrough but
        # not its taken edge). If we cannot recover any precise successor from
        # the predecessor block, fall back to the original incoming edge shape.
        for src, old_dst, data in incoming_edges:
            connected = False
            direct_targets, fallthrough_addr = _seed_node_expected_successors(src)

            for target in direct_targets:
                new_dst = replacement_nodes_by_addr.get(target)
                if new_dst is None:
                    continue
                edge_data = dict(data)
                edge_data["jumpkind"] = "Ijk_Boring"
                graph.add_edge(src, new_dst, **edge_data)
                connected = True

            if fallthrough_addr is not None:
                new_dst = replacement_nodes_by_addr.get(fallthrough_addr)
                if new_dst is not None:
                    edge_data = dict(data)
                    edge_data["jumpkind"] = "Ijk_Boring"
                    graph.add_edge(src, new_dst, **edge_data)
                    connected = True

            if connected:
                continue

            new_dst = replacement_nodes_by_addr.get(old_dst.addr, first_replacement)
            if new_dst is not None:
                graph.add_edge(src, new_dst, **data)

        # Reconnect replacement blocks to seed nodes outside the repaired span
        # using only explicit custom block exits. This keeps the patch local and
        # avoids manufacturing extra edges from unrelated seed nodes.
        for block in plan.block_specs:
            src = replacement_nodes_by_addr.get(block.addr)
            if src is None:
                continue

            for target in block.direct_targets:
                if target in removed_addrs:
                    continue
                for old_src, old_dst, data in outgoing_by_addr.get(target, []):
                    edge_data = dict(data)
                    edge_data["jumpkind"] = "Ijk_Boring"
                    graph.add_edge(src, old_dst, **edge_data)

            if block.fallthrough_addr is not None and block.fallthrough_addr not in removed_addrs:
                for old_src, old_dst, data in outgoing_by_addr.get(block.fallthrough_addr, []):
                    edge_jumpkind = "Ijk_FakeRet" if block.jumpkind == "Ijk_Call" else "Ijk_Boring"
                    edge_data = dict(data)
                    edge_data["jumpkind"] = edge_jumpkind
                    graph.add_edge(src, old_dst, **edge_data)

    # Renderers consult the source node for its "current graph" when deciding
    # how to style outgoing edges. Point every surviving node at the repaired
    # custom graph so conditional-vs-fallthrough classification uses the
    # patched edge set instead of the original CFGFast one.
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
    plans = discover_repair_plans(project, seed_cfg, func_addr)
    return repair_cfg_from_blocks(seed_cfg, plans)


def dump_repair_plan(plan: RepairPlan, output_path: Path | None = None) -> None:
    """Persist a human-readable repair-plan sketch for debugging."""

    lines = [
        f"func_addr={plan.func_addr:#x}",
        f"range=[{plan.bounds.addr:#x}, {plan.bounds.end_addr:#x})",
        f"size={plan.bounds.size}",
        f"repair_span=[{plan.region_start:#x}, {plan.region_end:#x})",
        "replaced_nodes=" + (
            ", ".join(f"{addr:#x}" for addr in plan.replaced_node_addrs)
            if plan.replaced_node_addrs
            else "<none>"
        ),
        "reasons=" + (", ".join(plan.reasons) if plan.reasons else "<none>"),
        "",
        "[blocks]",
    ]

    for block in plan.block_specs:
        targets = ", ".join(f"{addr:#x}" for addr in block.direct_targets) or "-"
        fallthrough = "-" if block.fallthrough_addr is None else f"{block.fallthrough_addr:#x}"
        lines.append(
            f"{block.addr:#x} size={block.size} jumpkind={block.jumpkind} "
            f"targets=[{targets}] fallthrough={fallthrough}"
        )

    lines.append("")
    lines.append("[edges]")
    for edge in plan.edge_specs:
        lines.append(f"{edge.src_addr:#x} -> {edge.dst_addr:#x} ({edge.jumpkind})")

    text = "\n".join(lines) + "\n"
    if output_path is None:
        sys.stdout.write(text)
        return
    output_path.write_text(text, encoding="utf-8")
