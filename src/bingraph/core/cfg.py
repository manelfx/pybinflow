"""
Custom CFG discovery and repair helpers.

This module implements the `cfg_mode="custom"` path as a localized repair pass
over a bounded CFGFast graph. The intent is to preserve angr metadata and any
already-correct CFGFast structure while patching only the malformed parts of the
graph.

Terminology used throughout the module:

- seed CFG:
  The initial bounded CFGFast graph for one function.
- anomaly:
  A block shape we do not trust, such as malformed byte coverage, a missing
  jump successor, or a truncated leaf.
- repair:
  Replacing stale seed nodes with newly decoded blocks while preserving good
  incoming/outgoing structure around them.
- obligation:
  A queued request to ensure that one address exists as a block entry in the
  repaired graph. Requests for the same action and address are merged, keeping
  every source-edge claim, split requirement, and reason that led to them.
  Obligations are created from seed anomalies and from edges discovered while
  repairing neighboring blocks.
- resolution policy:
  Immediate resolution attempts local recovery for a missing successor of a
  live node. If it cannot recover that target, the request becomes ordinary
  queued work; all other requests begin queued.
- terminator:
  The control-transfer summary of a recovered block: return, call, direct jump,
  or plain fallthrough, plus any direct targets or fallthrough address.
- placeholder:
  A temporary zero-sized target node that records an unresolved block entry.
  It may receive incoming edge claims, but never supplies control-flow
  semantics itself; recovery replaces it with a decoded block or cleanup
  removes it.

High-level algorithm:

1. Build a bounded CFGFast graph for the target function.
2. If the seed graph shows no known anomalies, return it unchanged.
3. Otherwise, seed a worklist with one repair obligation per anomalous block.
4. Process obligations one by one:
   - decode a bounded replacement block at the requested address,
   - splice it into the live graph,
   - queue new obligations for block starts implied by the recovered
     terminator or by preserved predecessor semantics,
   - reconcile the replacement block and its immediate graph neighborhood.
5. When the worklist is truly empty, remove unreachable stale nodes and
   temporary placeholders, then expose the repaired graph through a small
   CFG-like wrapper.

The implementation is intentionally conservative. It prefers localized repairs
over whole-function reconstruction so already-correct arch-specific behavior
from CFGFast remains intact.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Iterable
from dataclasses import dataclass, replace
from types import SimpleNamespace
from typing import Any, Literal, Protocol, cast

from angr import KnowledgeBase, Project
from angr.analyses.cfg import CFGBase
from angr.knowledge_plugins.cfg import CFGModel, CFGNode
from capstone import CS_GRP_CALL, CS_GRP_JUMP, CS_GRP_RET, CS_OP_IMM, CsInsn
from capstone.arm import ARM_CC_AL, ARM_CC_INVALID
from capstone.x86 import X86_INS_JMP, X86_INS_LJMP
from loguru import logger
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

# Entry resolution is deliberately explicit: most discovered block starts can
# wait for normal worklist processing, while a missing successor of a live node
# may need immediate local recovery to preserve convergence.
EntryResolutionPolicy = Literal["queued", "immediate"]


class CFGGraph(Protocol):
    """Public graph operations shared by NetworkX and angr's SpillingCFG."""

    def nodes(self) -> Iterable[CFGNode]: ...

    def edges(
        self, data: bool = False
    ) -> Iterable[tuple[CFGNode, CFGNode, dict[str, Any]]]: ...

    def predecessors(self, node: CFGNode) -> Iterable[CFGNode]: ...

    def successors(self, node: CFGNode) -> Iterable[CFGNode]: ...

    def in_degree(self, node: CFGNode) -> int: ...

    def has_edge(self, src: CFGNode, dst: CFGNode) -> bool: ...

    def get_edge_data(self, src: CFGNode, dst: CFGNode) -> dict[str, Any] | None: ...

    def add_node(self, node: CFGNode) -> None: ...

    def add_edge(self, src: CFGNode, dst: CFGNode, **attrs: Any) -> None: ...

    def remove_node(self, node: CFGNode) -> None: ...


def _cfg_graph(cfg: CFGBase) -> CFGGraph:
    """Return the CFG's public graph wrapper with the operations used here."""

    return cast(CFGGraph, cfg.graph)


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
class BlockLeaderRegistry:
    """Reasons that an address must remain a basic-block entry during recovery."""

    reasons: dict[int, set[str]]

    def copy(self) -> BlockLeaderRegistry:
        """Return an independent snapshot suitable for one recovery attempt."""

        return BlockLeaderRegistry(
            {addr: set(reasons) for addr, reasons in self.reasons.items()}
        )

    def add(self, addr: int, reason: str) -> bool:
        """Record one leader reason and return whether the registry changed."""

        reasons = self.reasons.setdefault(addr, set())
        if reason in reasons:
            return False
        reasons.add(reason)
        return True

    def starts(self) -> set[int]:
        """Return all addresses currently required to begin a block."""

        return set(self.reasons)

    def starts_with_reason(self, reason: str) -> set[int]:
        """Return leader addresses that carry one particular reason."""

        return {addr for addr, reasons in self.reasons.items() if reason in reasons}


@dataclass(frozen=True)
class RepairObligation:
    """One request to recover or reconcile a CFG address."""

    addr: int
    reason: str
    action: Literal["recover", "reconcile"] = "recover"
    source_node: CFGNode | None = None
    jumpkind: EdgeJumpKind = "Ijk_Boring"
    preserve_exact_addr: bool = False
    resolution_policy: EntryResolutionPolicy = "queued"


@dataclass(frozen=True)
class EdgeClaim:
    """One required edge from one exact live source node into an obligation."""

    source_node: CFGNode
    jumpkind: EdgeJumpKind


@dataclass
class PendingObligation:
    """Merged queued work for one action at one CFG address."""

    addr: int
    action: Literal["recover", "reconcile"]
    reasons: set[str]
    edge_claims: set[EdgeClaim]
    preserve_exact_addr: bool = False

    @classmethod
    def from_request(cls, request: RepairObligation) -> PendingObligation:
        """Create pending state from one first-in request."""

        claims = set()
        if request.source_node is not None:
            claims.add(EdgeClaim(request.source_node, request.jumpkind))
        return cls(
            addr=request.addr,
            action=request.action,
            reasons={request.reason},
            edge_claims=claims,
            preserve_exact_addr=request.preserve_exact_addr,
        )

    def merge(self, request: RepairObligation) -> None:
        """Accumulate another request without changing queue order."""

        self.reasons.add(request.reason)
        if request.source_node is not None:
            self.edge_claims.add(EdgeClaim(request.source_node, request.jumpkind))
        self.preserve_exact_addr |= request.preserve_exact_addr

    def fingerprint(self) -> tuple[bool, tuple[tuple[int, EdgeJumpKind], ...]]:
        """Return the repair-relevant state used to detect a stalled requeue."""

        # Source identity matters because distinct CFG nodes can share an
        # address. Reasons are diagnostic only, so they do not affect whether
        # retrying this obligation can change the graph.
        claims = tuple(
            sorted(
                (id(claim.source_node), claim.jumpkind) for claim in self.edge_claims
            )
        )
        return self.preserve_exact_addr, claims


@dataclass(frozen=True)
class JumpSuccessorExpectation:
    """One expected direct jump successor and its repair metadata."""

    addr: int
    jumpkind: EdgeJumpKind
    preserve_exact_addr: bool


@dataclass(frozen=True)
class JumpSuccessorAnalysis:
    """Expected and present successors for one decoded jump-terminating block."""

    kind: Literal["conditional", "direct"]
    expected: tuple[JumpSuccessorExpectation, ...]
    present: frozenset[int]


@dataclass(frozen=True)
class DecodedNode:
    """The Capstone instruction view of one CFG node, when it is available."""

    insns: tuple[CsInsn, ...] | None
    inspection_error: Exception | None = None

    @classmethod
    def from_node(cls, node) -> DecodedNode:
        """Read a node's Capstone instructions with the project's usual fallback."""

        try:
            return cls(tuple(item.insn for item in node.block.capstone.insns))
        except (AttributeError, KeyError) as exc:
            return cls(None, exc)

    @property
    def is_empty(self) -> bool:
        """Return whether Capstone found no instructions in the node."""

        return not self.insns

    @property
    def last(self) -> CsInsn | None:
        """Return the final decoded instruction, if one exists."""

        return self.insns[-1] if self.insns else None

    def has_exact_coverage(self, node) -> bool:
        """Return whether instructions exactly cover the node's declared range."""

        if node.size == 0 or self.insns is None:
            return False

        expected_addr = node.addr
        for insn in self.insns:
            if insn.address != expected_addr:
                return False
            expected_addr += insn.size

        return expected_addr == node.addr + node.size

    def contains_mid_instruction_addr(self, addr: int) -> bool:
        """Return whether ``addr`` falls strictly inside a decoded instruction."""

        if self.insns is None:
            return False
        return any(
            insn.address < addr < insn.address + insn.size for insn in self.insns
        )


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
        x86 `jne target` or ARM `bne target`). Plain ARM `b target` carries the
        unconditional `AL` condition code and must not be treated as
        conditional just because it has one immediate operand.
        """

        if not self.is_jump() or self.direct_target() is None:
            return False

        if len(self.insn.operands) > 1:
            return True

        arm_cc = getattr(self.insn, "cc", ARM_CC_INVALID)
        if arm_cc == ARM_CC_AL:
            return False
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


def _lookup_function_bounds(project: Project, func_addr: int) -> FunctionBounds:
    """
    Return function bounds from the symbol view used across bingraph.

    The custom repair pass needs a stable upper bound even when CFGFast itself
    missed blocks. `kb.functions[addr].size` is derived from currently
    discovered CFG blocks, so it can shrink along with a malformed CFG. By
    reusing `list_function_symbols()` we inherit the project's existing symbol
    parsing and size-inference logic instead.
    """

    function = next(
        (sym for sym in list_function_symbols(project) if sym.addr == func_addr), None
    )
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


def _lift_block_terminator(
    project: Project, bounds: FunctionBounds, block_insns: list[CsInsn]
) -> TerminatorInfo:
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
        return TerminatorInfo(
            jumpkind="Ijk_Fallthrough", fallthrough_addr=fallthrough_addr
        )

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

    default_target: int | None = None
    if isinstance(vex.next, pyvex.expr.Const):
        target = vex.next.con.value
        if isinstance(target, int):
            default_target = target

    if semantic.is_ret():
        return TerminatorInfo(jumpkind="Ijk_Ret")

    if semantic.is_call():
        direct_targets: tuple[int, ...] = ()
        if isinstance(default_target, int) and _is_direct_target_valid(
            bounds, default_target
        ):
            direct_targets = (default_target,)
        fallthrough_addr = next_addr if next_addr < bounds.end_addr else None
        return TerminatorInfo(
            jumpkind="Ijk_Call",
            direct_targets=direct_targets,
            fallthrough_addr=fallthrough_addr,
        )

    # For jumps, prefer the VEX control-flow shape when it is available:
    # conditional branches produce exit statements, while direct
    # unconditional jumps do not. However, the concrete branch target still
    # comes from Capstone because malformed seed nodes can expose a stale
    # VEX `next` value even when the decoded terminator target is correct.
    if exit_targets or semantic.is_conditional_jump():
        all_targets: list[int] = list(exit_targets)
        if isinstance(default_target, int) and default_target not in all_targets:
            all_targets.append(default_target)
        fallthrough_addr = (
            next_addr
            if next_addr in all_targets and next_addr < bounds.end_addr
            else None
        )
        return TerminatorInfo(
            jumpkind="Ijk_Boring",
            direct_targets=tuple(
                target for target in all_targets if target != fallthrough_addr
            ),
            fallthrough_addr=fallthrough_addr,
        )

    if (
        semantic.is_jump()
        and isinstance(default_target, int)
        and _is_direct_target_valid(bounds, default_target)
    ):
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

    direct_targets: tuple[int, ...] = ()
    if isinstance(default_target, int) and _is_direct_target_valid(
        bounds, default_target
    ):
        direct_targets = (default_target,)
    return TerminatorInfo(jumpkind="Ijk_Boring", direct_targets=direct_targets)


def _iter_seed_function_nodes(seed_cfg: CFGBase, func_addr: int):
    """Yield non-simprocedure nodes from the seed CFG for one function."""

    for node in _cfg_graph(seed_cfg).nodes():
        if getattr(node, "function_address", None) != func_addr:
            continue
        if getattr(node, "is_simprocedure", False):
            continue
        yield node


def iter_function_nodes(cfg: CFGBase, func_addr: int):
    """Yield non-simprocedure nodes that belong to one function."""

    yield from _iter_seed_function_nodes(cfg, func_addr)


def _seed_node_expected_successors(node) -> tuple[tuple[int, ...], int | None]:
    """
    Return the local direct targets and fallthrough encoded in one seed node.

    CFGFast may already have stitched a malformed region incorrectly, so when we
    reconnect a preserved predecessor into repaired blocks we prefer the
    predecessor's own lifted block semantics over the old graph edges.
    """

    try:
        decoded = DecodedNode.from_node(node)
    except Exception:
        return (), None
    if decoded.insns is None:
        return (), None

    try:
        vex = node.block.vex
    except Exception:
        return (), None

    if decoded.is_empty:
        return (), None

    last_insn = decoded.last
    if last_insn is None:
        return (), None

    last_semantic = InsnSemantics(last_insn)
    if not last_semantic.is_control_transfer():
        return (), None

    last_addrs = {last_insn.address}

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


def _seed_graph_direct_targets(graph: CFGGraph, node) -> tuple[int, ...]:
    """
    Return direct branch targets that are explicitly present in the seed graph.

    For unrepaired seed nodes we only want to preserve leaders that are backed
    by a concrete branch edge already materialized in CFGFast. This is narrower
    than trusting the node's fallthrough layout and avoids swallowing real
    branch-target leaders such as `0x806a85a` in `__strcasecmp_l_sse4_2`.
    """

    try:
        decoded = DecodedNode.from_node(node)
    except Exception:
        return ()

    if decoded.is_empty:
        return ()

    last_insn = decoded.last
    if last_insn is None:
        return ()

    last = InsnSemantics(last_insn)
    if not last.is_jump():
        return ()

    target = last.direct_target()
    if not isinstance(target, int):
        return ()

    successor_addrs = {
        succ.addr for succ in graph.successors(node) if hasattr(succ, "addr")
    }
    if target not in successor_addrs:
        return ()

    return (target,)


def _analyze_jump_successors(
    graph: CFGGraph,
    bounds: FunctionBounds,
    node,
) -> JumpSuccessorAnalysis | None:
    """
    Return expected vs present successors for one decoded jump block.

    We classify the jump shape from VEX exit statements when possible, but we
    keep using the decoded Capstone target as the authoritative direct target.
    This avoids trusting stale `vex.next` values on malformed CFGFast nodes
    while still recognizing conditional-vs-direct structure on repaired nodes.
    """

    try:
        decoded = DecodedNode.from_node(node)
    except Exception:
        return None

    if decoded.is_empty or node_has_decoding_coverage_mismatch(node):
        return None

    last_insn = decoded.last
    if last_insn is None:
        return None

    last = InsnSemantics(last_insn)
    if not last.is_jump():
        return None

    expected: list[JumpSuccessorExpectation] = []
    kind: Literal["conditional", "direct"]
    direct_target = last.direct_target()
    if direct_target is None:
        return None
    fallthrough_addr = last_insn.address + last_insn.size

    try:
        vex = node.block.vex
    except Exception:
        vex = None

    exit_targets: list[int] = []
    if vex is not None:
        for ins_addr, _, stmt in vex.exit_statements:
            if ins_addr != last_insn.address:
                continue
            target = getattr(stmt.dst, "value", None)
            if isinstance(target, int) and target not in exit_targets:
                exit_targets.append(target)

    if exit_targets or last.is_conditional_jump():
        expected.append(JumpSuccessorExpectation(direct_target, "Ijk_Boring", True))
        if bounds.addr <= fallthrough_addr < bounds.end_addr:
            expected.append(
                JumpSuccessorExpectation(fallthrough_addr, "Ijk_Boring", False)
            )
        kind = "conditional"
    else:
        expected.append(JumpSuccessorExpectation(direct_target, "Ijk_Boring", True))
        kind = "direct"

    present = frozenset(
        succ.addr for succ in graph.successors(node) if not _node_is_placeholder(succ)
    )
    return JumpSuccessorAnalysis(
        kind=kind,
        expected=tuple(expected),
        present=present,
    )


def node_has_missing_jump_successor(
    graph: CFGGraph,
    bounds: FunctionBounds,
    node,
) -> bool:
    """Return True when a jump block is missing one or more successor edges."""

    analysis = _analyze_jump_successors(graph, bounds, node)
    if analysis is None:
        return False

    expected_addrs = {item.addr for item in analysis.expected}
    if analysis.present == expected_addrs:
        return False

    label = "conditional branch" if analysis.kind == "conditional" else "direct jump"
    logger.warning(
        f"Node {node.addr:#x} is missing {label} successor(s) "
        f"or shows unexpected ones: expected "
        f"{', '.join(hex(t) for t in sorted(expected_addrs))}, got "
        f"{', '.join(hex(t) for t in sorted(analysis.present)) if analysis.present else '<none>'}"
    )
    return True


def _missing_jump_successors(
    graph: CFGGraph,
    bounds: FunctionBounds,
    node,
) -> tuple[JumpSuccessorExpectation, ...]:
    """Return the subset of expected jump successors still missing in the graph."""

    analysis = _analyze_jump_successors(graph, bounds, node)
    if analysis is None:
        return ()

    return tuple(
        item for item in analysis.expected if item.addr not in analysis.present
    )


def node_has_linear_merge_successor(graph: CFGGraph, node) -> bool:
    """
    Return True when `node` should absorb its only straight-line successor.

    This targets the specific malformed shape where CFGFast left an artificial
    split inside one linear byte range: A has one `Ijk_Boring` successor B, B
    has exactly one predecessor, B starts exactly where A ends, and A itself
    does not end in a control-transfer instruction.
    """

    successors = list(graph.successors(node))
    if len(successors) != 1:
        return False

    succ = successors[0]
    if getattr(succ, "is_simprocedure", False):
        return False
    if _node_is_placeholder(succ):
        return False
    if graph.in_degree(succ) != 1:
        return False

    edge_data = graph.get_edge_data(node, succ) or {}
    if edge_data.get("jumpkind") != "Ijk_Boring":
        return False
    if _node_range_end(node) != succ.addr:
        return False

    try:
        decoded = DecodedNode.from_node(node)
    except Exception:
        return False

    if decoded.is_empty:
        return False

    last_insn = decoded.last
    if last_insn is None:
        return False
    if InsnSemantics(last_insn).is_control_transfer():
        return False

    logger.warning(
        f"Node {node.addr:#x} is split from straight-line successor {succ.addr:#x}"
    )
    return True


def node_has_decoding_coverage_mismatch(node) -> bool:
    """Return True when a CFG node clearly covers bytes incorrectly."""

    if node.size == 0:
        logger.warning(f"Node {node.addr:#x} has size zero")
        return True

    decoded = DecodedNode.from_node(node)
    if decoded.insns is None:
        logger.warning(
            f"Capstone inspection failed for node {node.addr:#x}: "
            f"{type(decoded.inspection_error).__name__}: "
            f"{decoded.inspection_error}"
        )
        return True

    if decoded.has_exact_coverage(node):
        return False

    expected_addr = node.addr
    for insn in decoded.insns:
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

    decoded = DecodedNode.from_node(node)
    if not decoded.is_empty:
        return False

    try:
        return node.block.vex.jumpkind == "Ijk_NoDecode"
    except Exception:
        return False


def node_has_truncated_leaf(graph: CFGGraph, func_addr: int, node) -> bool:
    """Return True when a CFG node stops before a real terminator and has no exits."""

    try:
        if node.block.vex.jumpkind == "Ijk_Ret":
            return False
    except Exception:
        pass

    decoded = DecodedNode.from_node(node)
    if decoded.insns is None:
        return False

    if decoded.is_empty:
        return False

    last_insn = decoded.last
    if last_insn is None:
        return False

    last = InsnSemantics(last_insn)
    if last.is_control_transfer():
        return False

    if any(True for _ in graph.successors(node)):
        return False

    has_later_function_node = any(
        other is not node
        and getattr(other, "function_address", None) == func_addr
        and not getattr(other, "is_simprocedure", False)
        and other.addr > node.addr
        for other in graph.nodes()
    )
    return has_later_function_node


def _node_needs_repair(
    graph: CFGGraph,
    bounds: FunctionBounds,
    func_addr: int,
    node,
) -> bool:
    """Return True when a node violates one of the repair invariants."""

    return (
        node_has_decoding_coverage_mismatch(node)
        or node_has_decode_gap(node)
        or node_has_truncated_leaf(graph, func_addr, node)
        or node_has_missing_jump_successor(graph, bounds, node)
        or node_has_linear_merge_successor(graph, node)
    )


def _has_decoding_coverage_mismatch(cfg: CFGBase, node) -> bool:
    """Return True when decoded instructions do not cover the full node span."""

    if not node_has_decoding_coverage_mismatch(node):
        return False

    logger.warning(
        f"CFG anomaly for function {node.function_address:#x}:"
        f" {'zero_sized_block' if node.size == 0 else 'malformed_block'}"
        f" at {node.addr:#x}"
    )
    return True


def _has_truncated_leaf(cfg: CFGBase, func_addr: int, node) -> bool:
    """Return True when a block stops before a real terminator and has no exits."""

    if not node_has_truncated_leaf(_cfg_graph(cfg), func_addr, node):
        return False

    try:
        insns = list(node.block.capstone.insns)
    except (AttributeError, KeyError):
        insn_text = "<unknown>"
    else:
        if insns:
            last = insns[-1]
            insn_text = f"{last.mnemonic} {last.op_str}".strip()
        else:
            insn_text = "<empty>"

    logger.warning(
        f"CFG anomaly for function {func_addr:#x}: truncated_leaf at {node.addr:#x}: "
        f"block ends with non-terminating instruction {insn_text} and has no CFG successors"
    )
    return True


def _has_decode_gap(cfg: CFGBase, func_addr: int) -> bool:
    """Return True when the CFG contains true decoding/lifting failures."""

    if getattr(getattr(cfg, "model", None), "ident", "") == "CFGFastCustom":
        # The custom fallback is intentionally capstone-driven. VEX lifting can
        # still complain about some recovered nodes, but at that point the
        # custom graph should be judged by decoded instruction coverage instead
        # of by whether pyvex likes every block.
        return False

    for node in iter_function_nodes(cfg, func_addr):
        if _has_decoding_coverage_mismatch(cfg, node):
            continue

        if node_has_decode_gap(node):
            logger.warning(
                f"CFG anomaly for function {func_addr:#x}: decode_gap at {node.addr:#x}: "
                "node ended with Ijk_NoDecode, which points to a lifting/decoding failure"
            )
            return True

    return False


def _has_weird_graph(cfg: CFGBase, func_addr: int) -> bool:
    """Return True when the CFG shows malformed structure without a decode gap."""

    project = getattr(cfg, "project", None)
    if project is None:
        project = getattr(getattr(cfg, "kb", None), "_project", None)

    for node in iter_function_nodes(cfg, func_addr):
        if _has_decoding_coverage_mismatch(cfg, node):
            return True
        if _has_truncated_leaf(cfg, func_addr, node):
            return True
        if project is not None and node_has_missing_jump_successor(
            _cfg_graph(cfg),
            _lookup_function_bounds(project, func_addr),
            node,
        ):
            logger.warning(
                f"CFG anomaly for function {func_addr:#x}: missing_jump_successor "
                f"at {node.addr:#x}: direct jump block is missing one or more CFG edges"
            )
            return True
        if node_has_linear_merge_successor(_cfg_graph(cfg), node):
            logger.warning(
                f"CFG anomaly for function {func_addr:#x}: linear_split at {node.addr:#x}: "
                "straight-line successor should be merged into the current block"
            )
            return True
    return False


def log_cfg_status(cfg: CFGBase, func_addr: int, cfg_label: str) -> None:
    """Log whether a CFG still shows the anomaly classes we currently track."""

    has_weird_graph = _has_weird_graph(cfg, func_addr)
    has_decode_gap = _has_decode_gap(cfg, func_addr)
    if not has_weird_graph and not has_decode_gap:
        logger.info(
            f"{cfg_label} for function {func_addr:#x} no longer shows known CFG anomalies"
        )
    else:
        logger.warning(
            f"{cfg_label} for function {func_addr:#x} still shows CFG anomalies"
        )


def _custom_model_marker() -> SimpleNamespace:
    """Return the minimal model metadata currently needed by callers."""

    return SimpleNamespace(ident="CFGFastCustom")


def _prune_orphan_simprocedures(graph: CFGGraph) -> None:
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
            if _node_is_simprocedure(node) and graph.in_degree(node) == 0
        ]
        if not orphan_nodes:
            return
        _remove_nodes(graph, orphan_nodes)


def _prune_placeholders(graph: CFGGraph) -> None:
    """Remove any temporary placeholder nodes left after the repair pass."""

    placeholders = [node for node in list(graph.nodes()) if _node_is_placeholder(node)]
    if placeholders:
        _remove_nodes(graph, placeholders)


def _block_name(bounds: FunctionBounds, block: BlockSpec) -> str:
    """Return the function-relative label used for a recovered block."""

    if block.addr == bounds.addr:
        return bounds.symbol.name
    return f"{bounds.symbol.name}+0x{block.addr - bounds.addr:x}"


def _node_intersects_bounds(node, bounds: FunctionBounds) -> bool:
    """Return True when a node overlaps the current function address range."""

    if _node_is_simprocedure(node):
        return False
    return _ranges_overlap(
        node.addr, _node_range_end(node), bounds.addr, bounds.end_addr
    )


def _iter_graph_bound_nodes(graph: CFGGraph, bounds: FunctionBounds):
    """
    Yield live graph nodes that overlap the current function bounds.

    Seed CFGFast nodes may carry an incorrect `function_address` once the graph
    goes malformed. The custom repair pass therefore keys all live-graph lookups
    off address bounds, not off the stored function tag.
    """

    for node in graph.nodes():
        if _node_intersects_bounds(node, bounds):
            yield node


def _nodes_at_addr(graph: CFGGraph, bounds: FunctionBounds, addr: int) -> list[CFGNode]:
    """Return all non-simprocedure nodes in the function bounds that start at addr."""

    return [
        node for node in _iter_graph_bound_nodes(graph, bounds) if node.addr == addr
    ]


def _covering_nodes(
    graph: CFGGraph, bounds: FunctionBounds, addr: int
) -> list[CFGNode]:
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

    return getattr(node, "size", 0) == 0 and str(getattr(node, "name", "")).startswith(
        "placeholder_"
    )


def _node_is_simprocedure(node) -> bool:
    """Return True when `node` is a synthetic/simprocedure CFG node."""

    return getattr(node, "is_simprocedure", False)


def _node_is_materialized_cfg_node(node) -> bool:
    """Return True for normal in-graph nodes that are neither simprocs nor placeholders."""

    return not _node_is_simprocedure(node) and not _node_is_placeholder(node)


def _node_has_forced_split(node, forced_block_starts: set[int]) -> bool:
    """Return True when a known block start falls inside this node's range."""

    node_end = _node_range_end(node)
    return any(node.addr < addr < node_end for addr in forced_block_starts)


def _addr_is_mid_instruction_start(node, addr: int) -> bool:
    """
    Return True when `addr` falls inside one decoded instruction of `node`.

    This is stricter than merely checking whether `addr` is covered by the
    node's nominal byte range. Malformed CFGFast nodes often advertise a stale
    size that extends beyond the last decoded instruction. Those trailing bytes
    may still be legitimate new block leaders and must not be suppressed.
    """

    try:
        return DecodedNode.from_node(node).contains_mid_instruction_addr(addr)
    except Exception:
        return False


def _make_cfg_node(
    seed_cfg: CFGBase, func_addr: int, bounds: FunctionBounds, block: BlockSpec
) -> CFGNode:
    """Instantiate one CFGNode compatible with the existing rendering pipeline."""

    return CFGNode(
        block.addr,
        block.size,
        cfg=seed_cfg.model,
        function_address=func_addr,
        block_id=block.addr,
        instruction_addrs=block.instruction_addrs,
        name=_block_name(bounds, block),
    )


def _make_placeholder_node(seed_cfg: CFGBase, func_addr: int, addr: int) -> CFGNode:
    """Create a zero-sized placeholder node for a newly discovered bad address."""

    return CFGNode(
        addr,
        0,
        cfg=seed_cfg.model,
        function_address=func_addr,
        block_id=addr,
        instruction_addrs=(),
        name=f"placeholder_{addr:#x}",
    )


def _find_external_target_node(graph: CFGGraph, addr: int) -> CFGNode | None:
    """Return an existing synthetic external-target leaf for one address."""

    for node in graph.nodes():
        if not _node_is_simprocedure(node):
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


def _ensure_external_target_node(
    seed_cfg: CFGBase,
    graph: CFGGraph,
    func_addr: int,
    addr: int,
) -> tuple[CFGNode, bool]:
    """Get or create one synthetic external-target leaf and report creation."""

    node = _find_external_target_node(graph, addr)
    if node is not None:
        return node, False

    node = _make_external_target_node(seed_cfg, func_addr, addr)
    graph.add_node(node)
    return node, True


def _node_is_acceptable(
    seed_cfg: CFGBase,
    graph: CFGGraph,
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
    # Keep the worklist repairing nodes that were already classified as
    # structurally incomplete by the seed anomaly checks. Otherwise an initial
    # bad block like __strcmp_sse4_2+0x37 can survive forever just because its
    # byte coverage looks locally self-consistent.
    if _node_needs_repair(graph, bounds, func_addr, node):
        return False
    return True


def _recover_block(
    project: Project, bounds: FunctionBounds, start_addr: int, stop_addrs: set[int]
) -> BlockSpec | None:
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
    block = BlockSpec(
        addr=insns[0].address,
        size=sum(obj.size for obj in insns),
        instruction_addrs=tuple(obj.address for obj in insns),
        jumpkind=terminator.jumpkind,
        direct_targets=terminator.direct_targets,
        fallthrough_addr=terminator.fallthrough_addr,
    )

    block_end = block.addr + block.size
    internal_targets = sorted(
        target
        for target in block.direct_targets
        if block.addr < target < block_end and target not in stop_addrs
    )
    if internal_targets:
        # A direct branch target that lands inside the bytes we just recovered
        # identifies a missing basic-block leader, typically a loop header that
        # CFGFast failed to seed. Re-run bounded recovery with that leader as a
        # hard stop so we do not absorb the target block into its predecessor.
        return _recover_block(
            project, bounds, start_addr, stop_addrs | {internal_targets[0]}
        )

    return block


def _add_successor_edge(
    graph: CFGGraph,
    src: CFGNode,
    dst: CFGNode,
    jumpkind: EdgeJumpKind,
) -> bool:
    """Add one successor edge if it is not already present with the same kind."""

    if graph.has_edge(src, dst):
        edge_data = graph.get_edge_data(src, dst) or {}
        if edge_data.get("jumpkind") == jumpkind:
            return False
    graph.add_edge(src, dst, jumpkind=jumpkind)
    return True


class _RepairSession:
    """Mutable state and helpers for one worklist-driven CFG repair run."""

    def __init__(self, project: Project, seed_cfg: CFGBase, func_addr: int):
        self.project = project
        self.seed_cfg = seed_cfg
        self.func_addr = func_addr
        self.bounds = _lookup_function_bounds(project, func_addr)
        # Mutate the live SpillingCFG wrapper in place, but stay on its public
        # API. Its private backing graph stores tuple keys that the renderer
        # cannot consume directly.
        self.graph = _cfg_graph(seed_cfg)
        self.queue: deque[tuple[str, int]] = deque()
        self.pending: dict[tuple[str, int], PendingObligation] = {}
        self.repaired_nodes: set[CFGNode] = set()
        self.leaders = BlockLeaderRegistry({func_addr: {"function_entry"}})
        self.processed_counts: dict[int, int] = {}
        self.mutation_revision = 0
        self.last_requeue_states: dict[
            tuple[str, int],
            tuple[int, tuple[bool, tuple[tuple[int, EdgeJumpKind], ...]]],
        ] = {}
        self.iterations = 0

    def _is_preservable_seed_node(self, node) -> bool:
        """
        Return True when an unrepaired seed node may still preserve CFG structure.

        Malformed seed fragments should not freeze extra block leaders or
        successor expectations. Once repaired, the replacement node will drive
        discovery with its own recovered semantics.
        """

        return node in self.repaired_nodes or _node_is_acceptable(
            self.seed_cfg,
            self.graph,
            self.bounds,
            self.func_addr,
            self._explicit_split_starts(),
            node,
        )

    def _note_mutation(self) -> None:
        """Advance the revision after a live CFG graph mutation."""

        self.mutation_revision += 1

    def _add_edge(
        self,
        src: CFGNode,
        dst: CFGNode,
        jumpkind: EdgeJumpKind,
    ) -> bool:
        """Add an edge and record whether it changed the live graph."""

        changed = _add_successor_edge(self.graph, src, dst, jumpkind)
        if changed:
            self._note_mutation()
        return changed

    def _record_obligation_progress(
        self,
        key: tuple[str, int],
        obligation: PendingObligation,
    ) -> None:
        """Reject a requeued obligation whose graph and repair state are unchanged."""

        pending = self.pending.get(key)
        if pending is None:
            self.last_requeue_states.pop(key, None)
            return

        state = self.mutation_revision, pending.fingerprint()
        previous_state = self.last_requeue_states.get(key)
        if state != previous_state:
            self.last_requeue_states[key] = state
            return

        reasons = ", ".join(sorted(pending.reasons))
        raise RuntimeError(
            f"Custom CFG stalled while {obligation.action} at {obligation.addr:#x}: "
            "the graph and pending repair state did not change "
            f"(reasons: {reasons})"
        )

    def _preserved_successor_starts(self, node) -> tuple[tuple[int, ...], int | None]:
        """
        Return the starts that `node` is allowed to preserve in the live graph.

        Repaired nodes preserve both direct targets and true fallthroughs. Seed
        nodes only preserve direct targets already materialized in the graph,
        since CFGFast fallthrough splits are often the very artifacts we are
        trying to erase.
        """

        if node in self.repaired_nodes:
            return _seed_node_expected_successors(node)

        return _seed_graph_direct_targets(self.graph, node), None

    def _explicit_split_starts(self) -> set[int]:
        """Return leaders created by a requested split through an old node."""

        return self.leaders.starts_with_reason("explicit_split")

    def _current_leaders(self) -> BlockLeaderRegistry:
        """Build the leader snapshot that bounds one local block recovery."""

        leaders = self.leaders.copy()
        for node in _iter_graph_bound_nodes(self.graph, self.bounds):
            if not _node_is_materialized_cfg_node(node):
                continue
            if not self._is_preservable_seed_node(node):
                continue

            direct_targets, fallthrough_addr = self._preserved_successor_starts(node)
            for target in direct_targets:
                if self.bounds.addr <= target < self.bounds.end_addr:
                    leaders.add(target, "direct_target")
            if (
                fallthrough_addr is not None
                and self.bounds.addr <= fallthrough_addr < self.bounds.end_addr
            ):
                leaders.add(fallthrough_addr, "fallthrough")

        return leaders

    def ensure_block_entry(
        self,
        obligation: RepairObligation,
    ) -> CFGNode | None:
        """
        Ensure one obligation is represented in the graph and queued for repair.

        If the obligation address is currently covered by a larger node, this
        method also decides whether we should re-run repair at the covering
        node's start or at the requested split address itself.
        """

        addr = obligation.addr
        if not (self.bounds.addr <= addr < self.bounds.end_addr):
            return None

        if obligation.resolution_policy == "immediate":
            recovered = self._recover_entry_now(obligation)
            if recovered is not None:
                return recovered
            # An immediate local attempt is a convergence aid, not a separate
            # execution path. Once it cannot recover the entry, hand the same
            # request to normal worklist processing.
            obligation = replace(obligation, resolution_policy="queued")

        existing_nodes = _nodes_at_addr(self.graph, self.bounds, addr)
        covering_nodes = _covering_nodes(self.graph, self.bounds, addr)
        for node in covering_nodes:
            if node.addr == addr or _node_is_placeholder(node):
                continue

            if _addr_is_mid_instruction_start(node, addr):
                allow_exact_split = (
                    obligation.preserve_exact_addr
                    and not self._is_preservable_seed_node(node)
                )
                if allow_exact_split:
                    continue
                # The requested address falls inside a decoded instruction of
                # a covering node. Unless this is an explicit direct-branch
                # target punching through a stale covering node, do not freeze
                # the byte offset as a synthetic block leader: it only causes
                # placeholder ping-pong around invalid starts such as 0x4358ff
                # / 0x43597e in __strstr_avx512. Legitimate taken targets like
                # 0x42367a / 0x445a5e still get through when the covering node
                # is itself stale.
                self._queue_if_needed(
                    RepairObligation(
                        addr=node.addr,
                        reason=f"covering_node_for_{addr:#x}",
                    )
                )
                return node

            if self.leaders.add(addr, "explicit_split"):
                self._note_mutation()
            placeholder = self._claim_placeholder(obligation)

            repair_addr = node.addr
            repair_reason = f"split_for_{addr:#x}"
            if not self._is_preservable_seed_node(node):
                repair_addr = addr
                repair_reason = obligation.reason

            self._queue_if_needed(
                RepairObligation(
                    addr=repair_addr,
                    reason=repair_reason,
                    source_node=obligation.source_node if repair_addr == addr else None,
                    jumpkind=obligation.jumpkind,
                )
            )
            # Keep the requested split point alive as its own obligation. The
            # covering block must be repaired first, but we still need a later
            # pass to materialize the block that starts exactly at `addr`.
            self._queue_if_needed(obligation)
            return placeholder

        for node in existing_nodes:
            if self._is_preservable_seed_node(node):
                self._connect_source_to_node(obligation, node)
                return node

        placeholder = self._claim_placeholder(obligation)
        self._queue_if_needed(obligation)
        return placeholder

    def _claim_placeholder(
        self,
        obligation: RepairObligation,
    ) -> CFGNode:
        """Get the target placeholder for an entry and attach its edge claims."""

        placeholder = next(
            (
                node
                for node in _nodes_at_addr(self.graph, self.bounds, obligation.addr)
                if _node_is_placeholder(node)
            ),
            None,
        )
        if placeholder is None:
            placeholder = _make_placeholder_node(
                self.seed_cfg,
                self.func_addr,
                obligation.addr,
            )
            self.graph.add_node(placeholder)
            self._note_mutation()
        self._connect_source_to_node(obligation, placeholder)
        return placeholder

    def _recover_entry_now(self, obligation: RepairObligation) -> CFGNode | None:
        """Resolve an existing entry or decode one immediately for local repair."""

        exact_nodes = [
            node
            for node in _nodes_at_addr(self.graph, self.bounds, obligation.addr)
            if _node_is_materialized_cfg_node(node)
        ]
        acceptable_node = self._first_acceptable_entry(exact_nodes)
        if acceptable_node is not None:
            return acceptable_node

        block = _recover_block(
            self.project,
            self.bounds,
            obligation.addr,
            self.current_stop_addrs(obligation.addr),
        )
        if block is None or block.addr != obligation.addr:
            return None

        recovered_node = self.splice_block(block)
        return recovered_node

    def _first_acceptable_entry(
        self,
        nodes: Iterable[CFGNode],
    ) -> CFGNode | None:
        """Return the first live node that can satisfy an entry request unchanged."""

        return next(
            (
                node
                for node in nodes
                if _node_is_acceptable(
                    self.seed_cfg,
                    self.graph,
                    self.bounds,
                    self.func_addr,
                    self._explicit_split_starts(),
                    node,
                )
            ),
            None,
        )

    def _connect_source_to_node(
        self,
        obligation: RepairObligation | PendingObligation,
        node: CFGNode,
    ) -> None:
        """Materialize every available source-edge claim into ``node``."""

        if isinstance(obligation, RepairObligation):
            claims = (
                {EdgeClaim(obligation.source_node, obligation.jumpkind)}
                if obligation.source_node is not None
                else set()
            )
        else:
            claims = obligation.edge_claims

        for claim in claims:
            if not any(source is claim.source_node for source in self.graph.nodes()):
                continue
            self._add_edge(claim.source_node, node, claim.jumpkind)

    def _queue_if_needed(self, request: RepairObligation) -> bool:
        """Merge a request into the pending work item for its action and address."""

        if request.resolution_policy != "queued":
            raise ValueError("Only queued obligations may enter the worklist")

        key = (request.action, request.addr)
        pending = self.pending.get(key)
        if pending is not None:
            pending.merge(request)
            return False

        self.pending[key] = PendingObligation.from_request(request)
        self.queue.append(key)
        return True

    def _requeue_pending(
        self,
        obligation: PendingObligation,
        *,
        addr: int | None = None,
        reason: str | None = None,
        include_claims: bool = True,
    ) -> None:
        """Turn merged work back into one or more ordinary queue requests."""

        target_addr = obligation.addr if addr is None else addr
        request_reason = reason or ", ".join(sorted(obligation.reasons))
        claims = obligation.edge_claims if include_claims else set()
        if not claims:
            self._queue_if_needed(
                RepairObligation(
                    addr=target_addr,
                    reason=request_reason,
                    action=obligation.action,
                    preserve_exact_addr=obligation.preserve_exact_addr,
                )
            )
            return

        for claim in claims:
            self._queue_if_needed(
                RepairObligation(
                    addr=target_addr,
                    reason=request_reason,
                    action=obligation.action,
                    source_node=claim.source_node,
                    jumpkind=claim.jumpkind,
                    preserve_exact_addr=obligation.preserve_exact_addr,
                )
            )

    def _queue_reconciliation(self, node: CFGNode, reason: str) -> None:
        """Queue a local invariant check for one materialized node."""

        if not _node_is_materialized_cfg_node(node):
            return
        if not _node_intersects_bounds(node, self.bounds):
            return
        self._queue_if_needed(
            RepairObligation(addr=node.addr, reason=reason, action="reconcile")
        )

    def _queue_reconciliation_neighborhood(self, node: CFGNode) -> None:
        """Recheck the nodes whose local invariants a splice may have changed."""

        neighbors = [node, *self.graph.predecessors(node), *self.graph.successors(node)]
        for neighbor in neighbors:
            self._queue_reconciliation(neighbor, f"neighbor_of_{node.addr:#x}")

    def _reconcile_node(self, node: CFGNode) -> None:
        """Satisfy local edge invariants before scheduling a full block recovery."""

        for expectation in _missing_jump_successors(self.graph, self.bounds, node):
            self._resolve_successor(
                node,
                expectation.addr,
                expectation.jumpkind,
                reason=f"missing_successor_of_{node.addr:#x}",
                preserve_exact_addr=expectation.preserve_exact_addr,
                resolution_policy="immediate",
                materialize_external=True,
            )

        if _node_needs_repair(
            self.graph,
            self.bounds,
            self.func_addr,
            node,
        ):
            self._queue_if_needed(
                RepairObligation(
                    addr=node.addr,
                    reason=f"remaining_anomaly_at_{node.addr:#x}",
                )
            )

    def _reconcile_addr(self, addr: int) -> None:
        """Reconcile every materialized node currently starting at ``addr``."""

        for node in _nodes_at_addr(self.graph, self.bounds, addr):
            if _node_is_materialized_cfg_node(node):
                self._reconcile_node(node)

    def _materialize_external_successor(
        self,
        src: CFGNode,
        target: int,
        jumpkind: EdgeJumpKind,
    ) -> bool:
        """
        Attach one out-of-function successor through a shared synthetic leaf.

        This keeps the policy for external targets centralized regardless of
        whether the source block is conditional, unconditional, or call-like.
        """

        if _is_direct_target_valid(self.bounds, target):
            return False

        leaf, created = _ensure_external_target_node(
            self.seed_cfg,
            self.graph,
            self.func_addr,
            target,
        )
        if created:
            self._note_mutation()
        self._add_edge(src, leaf, jumpkind)
        return True

    def _resolve_successor(
        self,
        src: CFGNode,
        target: int,
        jumpkind: EdgeJumpKind,
        *,
        reason: str,
        preserve_exact_addr: bool,
        resolution_policy: EntryResolutionPolicy = "queued",
        materialize_external: bool = False,
    ) -> None:
        """
        Resolve one successor target and connect it from `src`.

        Direct targets may materialize an external leaf. In-function targets
        enter through `ensure_block_entry()`, which owns the
        preserve/split/placeholder/recovery decision. Callers choose immediate
        resolution only when a live node is missing a required successor.
        """

        if materialize_external and self._materialize_external_successor(
            src, target, jumpkind
        ):
            return

        node = self.ensure_block_entry(
            RepairObligation(
                addr=target,
                reason=reason,
                source_node=src,
                jumpkind=jumpkind,
                preserve_exact_addr=preserve_exact_addr,
                resolution_policy=resolution_policy,
            ),
        )
        if node is not None:
            self._add_edge(src, node, jumpkind)

    def ensure_expected_successors(self, src: CFGNode) -> None:
        """
        Recreate the successor edges implied by one preserved predecessor node.

        This keeps good CFGFast predecessors intact while still discovering
        repaired targets or fresh split points around them.
        """

        direct_targets, fallthrough_addr = _seed_node_expected_successors(src)
        for target in direct_targets:
            self._resolve_successor(
                src,
                target,
                "Ijk_Boring",
                reason=f"expected_successor_of_{src.addr:#x}",
                preserve_exact_addr=True,
                materialize_external=True,
            )

        if fallthrough_addr is not None:
            self._resolve_successor(
                src,
                fallthrough_addr,
                "Ijk_Boring",
                reason=f"expected_fallthrough_of_{src.addr:#x}",
                preserve_exact_addr=False,
            )

    def current_stop_addrs(self, addr: int) -> set[int]:
        """Return the hard stop addresses used for bounded recovery at `addr`."""

        leaders = self._current_leaders()
        return {
            target
            for target in leaders.starts()
            if addr < target < self.bounds.end_addr
            if not self._addr_is_linear_tail_start(target)
        }

    def _addr_is_linear_tail_start(self, addr: int) -> bool:
        """
        Return True when `addr` is only the straight-line tail of a predecessor.

        Such addresses should not act as hard recovery boundaries: if the live
        graph currently has `A -> B` as a linear split candidate, revisiting A
        should be allowed to absorb B into one larger block even when other
        stop-set sources still mention B.
        """

        nodes = [
            node
            for node in _nodes_at_addr(self.graph, self.bounds, addr)
            if _node_is_materialized_cfg_node(node)
        ]
        for node in nodes:
            preds = [
                pred
                for pred in self.graph.predecessors(node)
                if _node_is_materialized_cfg_node(pred)
            ]
            if len(preds) != 1:
                continue
            if node_has_linear_merge_successor(self.graph, preds[0]):
                return True
        return False

    def splice_block(self, block: BlockSpec) -> CFGNode:
        """
        Replace every stale overlapping node with one recovered block.

        Incoming edges from preserved predecessors are rewired after stale nodes
        are removed. Preserved predecessors may also enqueue additional sibling
        successors implied by their own semantics.
        """

        recovered_start = block.addr
        recovered_end = block.addr + block.size

        removed_nodes = [
            node
            for node in _iter_graph_bound_nodes(self.graph, self.bounds)
            if node.addr == recovered_start
            or _ranges_overlap(
                node.addr, _node_range_end(node), recovered_start, recovered_end
            )
            or (_node_is_placeholder(node) and node.addr == recovered_start)
        ]
        removed_set = set(removed_nodes)

        incoming_edges = [
            (src, dst, dict(data))
            for src, dst, data in list(self.graph.edges(data=True))
            if dst in removed_set and src not in removed_set
        ]

        _remove_nodes(self.graph, removed_nodes)

        recovered_node = _make_cfg_node(
            self.seed_cfg, self.func_addr, self.bounds, block
        )
        self.graph.add_node(recovered_node)
        self._note_mutation()
        self.repaired_nodes.add(recovered_node)

        for pred, _, data in incoming_edges:
            self._add_edge(
                pred,
                recovered_node,
                data.get("jumpkind", "Ijk_Boring"),
            )

            if pred in self.repaired_nodes:
                continue
            if not _node_is_acceptable(
                self.seed_cfg,
                self.graph,
                self.bounds,
                self.func_addr,
                self._explicit_split_starts(),
                pred,
            ):
                continue

            self.ensure_expected_successors(pred)

        for target in block.direct_targets:
            edge_jumpkind = "Ijk_Call" if block.jumpkind == "Ijk_Call" else "Ijk_Boring"
            self._resolve_successor(
                recovered_node,
                target,
                edge_jumpkind,
                reason=f"direct_target_of_{block.addr:#x}",
                preserve_exact_addr=True,
                materialize_external=True,
            )

        if block.fallthrough_addr is not None:
            self._resolve_successor(
                recovered_node,
                block.fallthrough_addr,
                "Ijk_FakeRet" if block.jumpkind == "Ijk_Call" else "Ijk_Boring",
                reason=f"fallthrough_of_{block.addr:#x}",
                preserve_exact_addr=False,
            )

        self._queue_reconciliation_neighborhood(recovered_node)

        return recovered_node

    def _cleanup(self) -> None:
        """Prune temporary and unreachable nodes after the worklist finishes."""

        _cleanup_unreachable_function_nodes(self.graph, self.bounds, self.func_addr)
        _prune_placeholders(self.graph)
        _prune_orphan_simprocedures(self.graph)

    def _register_custom_graphs(self) -> None:
        """Attach the repaired live graph to every node wrapper used by rendering."""

        for node in self.graph.nodes():
            register_custom_graph(node, self.graph)

    def _process_reconciliation(self, obligation: PendingObligation) -> None:
        """Reconcile every live node at one queued address."""

        self._reconcile_addr(obligation.addr)

    def _process_recovery(self, obligation: PendingObligation) -> None:
        """Recover a queued address or requeue the work needed to expose it."""

        addr = obligation.addr
        current_nodes = _nodes_at_addr(self.graph, self.bounds, addr)
        covering_nodes = [
            node
            for node in _covering_nodes(self.graph, self.bounds, addr)
            if node.addr != addr and not _node_is_placeholder(node)
        ]
        if covering_nodes:
            if addr in self._explicit_split_starts():
                # Another node still covers this forced split point. Requeue
                # both the covering node and the split address so the target
                # is revisited after the prefix block gets truncated.
                for node in covering_nodes:
                    self._queue_if_needed(
                        RepairObligation(
                            addr=node.addr,
                            reason=f"split_for_{addr:#x}",
                        )
                    )
                self._requeue_pending(obligation)
            return

        acceptable_node = self._first_acceptable_entry(current_nodes)
        if acceptable_node is not None:
            self._connect_source_to_node(obligation, acceptable_node)
            return

        block = _recover_block(
            self.project, self.bounds, addr, self.current_stop_addrs(addr)
        )
        if block is None:
            logger.warning(f"Custom CFG could not recover a block at {addr:#x}")
            return

        recovered_node = self.splice_block(block)
        self._connect_source_to_node(obligation, recovered_node)

    def _process_obligation(self, obligation: PendingObligation) -> None:
        """Dispatch one in-bounds worklist item to its action-specific handler."""

        if not (self.bounds.addr <= obligation.addr < self.bounds.end_addr):
            return

        if obligation.action == "reconcile":
            self._process_reconciliation(obligation)
            return

        self._process_recovery(obligation)

    def run(self) -> CFGBase | CustomCFG:
        """Execute the repair worklist and return the repaired CFG wrapper."""

        initial_bad_addrs = sorted(
            {
                node.addr
                for node in _iter_seed_function_nodes(self.seed_cfg, self.func_addr)
                if _node_needs_repair(self.graph, self.bounds, self.func_addr, node)
            }
        )
        if not initial_bad_addrs:
            logger.info(
                f"Seed CFG for function {self.func_addr:#x} has no known anomalies; "
                "skipping custom repair"
            )
            return self.seed_cfg

        logger.info(
            f"Repairing seed CFG for function {self.func_addr:#x} with "
            f"{len(initial_bad_addrs)} anomalous block start(s)"
        )

        for addr in initial_bad_addrs:
            self.ensure_block_entry(RepairObligation(addr=addr, reason="seed_anomaly"))

        while self.queue:
            self.iterations += 1
            if self.iterations > 5000:
                raise RuntimeError(
                    f"Custom CFG worklist exceeded 5000 iterations for {self.func_addr:#x}; "
                    f"top counts: {self.processed_counts}"
                )

            key = self.queue.popleft()
            obligation = self.pending.pop(key)
            addr = obligation.addr
            self.processed_counts[addr] = self.processed_counts.get(addr, 0) + 1
            if self.processed_counts[addr] <= 5:
                logger.info(
                    f"Custom CFG processing {addr:#x} for function {self.func_addr:#x} "
                    f"(visit {self.processed_counts[addr]})"
                )

            self._process_obligation(obligation)
            self._record_obligation_progress(key, obligation)

        self._cleanup()
        self._register_custom_graphs()
        return CustomCFG(
            graph=self.graph,
            model=_custom_model_marker(),
            functions=self.seed_cfg.functions,
            kb=self.seed_cfg.kb,
        )


def _cleanup_unreachable_function_nodes(
    graph: CFGGraph, bounds: FunctionBounds, func_addr: int
) -> None:
    """Remove nodes in one function that are unreachable from the entry node."""

    entry_nodes = _nodes_at_addr(graph, bounds, func_addr)
    if not entry_nodes:
        return

    reachable: set[CFGNode] = set()
    queue: deque[CFGNode] = deque(entry_nodes)

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


def _remove_nodes(graph: CFGGraph, nodes: Iterable[CFGNode]) -> None:
    """Remove a batch of nodes through the graph wrapper's public API."""

    for node in nodes:
        graph.remove_node(node)


def _repair_cfg_with_worklist(
    project: Project,
    seed_cfg: CFGBase,
    func_addr: int,
) -> CFGBase | CustomCFG:
    """Repair only anomalous CFGFast regions by materializing blocks on demand."""

    return _RepairSession(project, seed_cfg, func_addr).run()


def build_custom_cfg(
    project: Project,
    kb: KnowledgeBase,
    func_addr: int,
    seed_cfg: CFGBase,
) -> CFGBase | CustomCFG:
    """
    Build a custom repaired CFG for one function starting from CFGFast output.

    The explicit KB parameter mirrors the higher-level CFG plumbing even though
    the current repair pass operates directly on the provided seed CFG graph.
    `_RepairSession.run()` owns both initial anomaly discovery and repair, so
    the custom path performs one coherent classification before it mutates the
    seed graph.
    """

    logger.info(f"Building custom CFG for function {func_addr:#x}")
    return _repair_cfg_with_worklist(project, seed_cfg, func_addr)
