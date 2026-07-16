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
  A queued worklist item for either recovering a block entry or reconciling a
  live block's local invariants. Requests for the same action and address are
  merged, retaining all their edge claims, split requirements, and reasons.
- edge claim:
  A required outgoing edge from one exact source node to an obligation's
  address. Identity matters because distinct CFG nodes can share an address.
- resolution policy:
  Immediate resolution attempts local recovery for a missing successor of a
  live node. If it cannot recover that target, the request becomes ordinary
  queued work; all other requests begin queued.
- leader:
  An address that must begin a recovered block. The repair session derives
  leaders from the function entry, direct targets, fallthroughs, and explicit
  splits through stale nodes, then uses them to bound local decoding.
- reconciliation:
  The local worklist action that restores required direct successors for a live
  node and queues recovery again only if the node still violates an invariant.
- terminator:
  The control-transfer summary of a recovered block: return, call, direct jump,
  or plain fallthrough, plus any direct targets or fallthrough address.
- placeholder:
  A temporary zero-sized target node that records an unresolved block entry.
  It may receive incoming edge claims, but never supplies control-flow
  semantics itself; recovery replaces it with a decoded block or cleanup
  removes it.
- unresolved-jump fallback:
  An `UnresolvableJumpTarget` simprocedure keeps an indirect dispatch visible
  when static recovery cannot prove its targets. It is connected to otherwise
  disconnected in-function blocks so cleanup preserves those CFGFast-discovered
  regions without inventing direct case edges from the original dispatch.

High-level algorithm:

1. Build a bounded CFGFast graph for the target function.
2. If the seed graph shows no known anomalies, return it unchanged.
3. Otherwise, seed a worklist with one recovery obligation per anomalous
   address. The queue merges requests for the same action and address.
4. Dispatch each obligation:
   - recovery resolves an existing entry, a covered entry, or a placeholder;
     it decodes and splices a bounded replacement block when required,
   - reconciliation restores direct successor edges and requeues recovery only
     for nodes that still violate an invariant,
   - both actions may add edge claims, leaders, and neighboring reconciliation
     work as the live graph changes.
5. Reject a requeued obligation when neither the graph revision nor its merged
   repair state changed. A separate iteration limit protects against a graph
   that continues changing without converging.
6. When the worklist is empty, connect any remaining unresolved indirect-jump
   placeholder to disconnected seed blocks. A placeholder with exactly one
   indirect source is flattened into explicitly marked unresolved candidate
   edges, preserving uncertainty without retaining a synthetic intermediary.
   Then remove genuinely unreachable stale nodes and temporary placeholders.
   These preserving edges deliberately do not claim new basic-block leaders.
   Cleanup can expose a new local anomaly, such as a linear split whose extra
   predecessors were stale. If cleanup changed the graph, queue those newly
   visible anomalies for another worklist pass. If cleanup made no change,
   report any remaining anomalies instead of retrying them indefinitely.
7. Expose the repaired graph through a small CFG-like wrapper.

The implementation is intentionally conservative. It prefers localized repairs
over whole-function reconstruction so already-correct arch-specific behavior
from CFGFast remains intact.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Iterable
from dataclasses import replace
from types import SimpleNamespace
from typing import Any, Literal, cast

from angr import KnowledgeBase, Project
from angr.analyses.cfg import CFGBase
from angr.knowledge_plugins.cfg import CFGNode
from capstone import CsInsn
from loguru import logger
import pyvex

from bingraph.helpers.symbols import list_function_symbols
from bingraph.helpers.capstone import (
    InsnSemantics,
    arch_has_delay_slot,
    control_transfer_index,
)
from .graph import CFGGraph, cfg_graph as _cfg_graph
from .jumps import (
    MAX_STATIC_JUMPTABLE_ENTRIES,
    is_direct_target_valid as _is_direct_target_valid,
)
from .decode import DecodedNode
from .models import (
    BlockLeaderRegistry,
    BlockSpec,
    CFGAnomaly,
    CustomCFG,
    CustomCFGStats,
    EdgeClaim,
    EdgeJumpKind,
    EntryResolutionPolicy,
    FunctionBounds,
    JumpSuccessorAnalysis,
    JumpSuccessorExpectation,
    PendingObligation,
    RepairObligation,
    StaticJumpTable,
    TerminatorInfo,
)


# Last-resort protection for repair loops that keep mutating the graph without
# converging. Stable requeues are diagnosed earlier by PendingObligation state.
MAX_CUSTOM_CFG_WORKLIST_ITERATIONS = 5_000


def _node_vex(node: CFGNode) -> Any | None:
    """Return VEX for a native CFG node, or None when angr cannot lift it."""

    try:
        return cast(Any, node.block).vex
    # CFGFast can retain zero-sized or otherwise unliftable seed nodes. VEX is
    # optional for the callers of this helper, so preserve their skip behavior.
    except Exception:
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


def _vex_tmp_definitions(vex) -> dict[int, Any]:
    """Return the local VEX temporary definitions used to unfold expressions."""

    return {
        stmt.tmp: stmt.data
        for stmt in vex.statements
        if isinstance(stmt, pyvex.stmt.WrTmp)
    }


def _resolve_vex_expr(expr, definitions: dict[int, Any]):
    """Follow local VEX temporary references until reaching a concrete expression."""

    seen: set[int] = set()
    while isinstance(expr, pyvex.expr.RdTmp):
        if expr.tmp in seen:
            return None
        seen.add(expr.tmp)
        expr = definitions.get(expr.tmp)
        if expr is None:
            return None
    return expr


def _vex_const_value(expr, definitions: dict[int, Any]) -> int | None:
    """Return a VEX constant's value after resolving local temporaries."""

    expr = _resolve_vex_expr(expr, definitions)
    if not isinstance(expr, pyvex.expr.Const):
        return None
    value = expr.con.value
    return value if isinstance(value, int) else None


def _vex_get_key(expr, definitions: dict[int, Any], vex) -> tuple[int, int] | None:
    """Return ``(register_offset, bits)`` for a VEX register read expression."""

    expr = _resolve_vex_expr(expr, definitions)
    if not isinstance(expr, pyvex.expr.Get):
        return None
    return expr.offset, expr.result_size(vex.tyenv)


def _vex_add_terms(expr, definitions: dict[int, Any]) -> list[Any] | None:
    """Flatten a VEX integer-addition expression into its non-additive terms."""

    expr = _resolve_vex_expr(expr, definitions)
    if expr is None:
        return None
    if isinstance(expr, pyvex.expr.Binop) and expr.op.startswith("Iop_Add"):
        left = _vex_add_terms(expr.args[0], definitions)
        right = _vex_add_terms(expr.args[1], definitions)
        if left is None or right is None:
            return None
        return [*left, *right]
    return [expr]


def _vex_index_key(expr, definitions: dict[int, Any], vex) -> tuple[int, int] | None:
    """Return the original register identity for a zero-extended table index."""

    expr = _resolve_vex_expr(expr, definitions)
    while isinstance(expr, pyvex.expr.Unop) and "Uto" in expr.op:
        expr = _resolve_vex_expr(expr.args[0], definitions)
    if isinstance(expr, pyvex.expr.Binop) and expr.op.startswith("Iop_And"):
        left = _vex_index_key(expr.args[0], definitions, vex)
        right = _vex_index_key(expr.args[1], definitions, vex)
        return left or right
    key = _vex_get_key(expr, definitions, vex)
    if key is not None and 0 < key[1] <= 8:
        return key
    return None


def _vex_static_int(expr, definitions: dict[int, Any]) -> int | None:
    """Evaluate the small constant-only VEX expressions used in branch guards."""

    value = _vex_const_value(expr, definitions)
    if value is not None:
        return value

    expr = _resolve_vex_expr(expr, definitions)
    if not isinstance(expr, pyvex.expr.Binop):
        return None
    left = _vex_static_int(expr.args[0], definitions)
    right = _vex_static_int(expr.args[1], definitions)
    if left is None or right is None:
        return None
    if expr.op.startswith("Iop_And"):
        return left & right
    return None


def _vex_guarded_index_upper_bound(
    vex, target_addr: int, index_key: tuple[int, int]
) -> int | None:
    """Return a proven unsigned upper bound for an exit entering ``target_addr``."""

    definitions = _vex_tmp_definitions(vex)
    for stmt in vex.statements:
        if not isinstance(stmt, pyvex.stmt.Exit):
            continue
        if getattr(stmt.dst, "value", None) != target_addr:
            continue

        guard = _resolve_vex_expr(stmt.guard, definitions)
        while isinstance(guard, pyvex.expr.Unop):
            guard = _resolve_vex_expr(guard.args[0], definitions)
        if not isinstance(guard, pyvex.expr.Binop):
            continue
        if "CmpLE" not in guard.op or not guard.op.endswith("U"):
            continue
        if _vex_index_key(guard.args[0], definitions, vex) != index_key:
            continue
        bound = _vex_static_int(guard.args[1], definitions)
        if bound is not None:
            return bound
    return None


def _guarded_jump_table_entry_count(
    graph: CFGGraph,
    bounds: FunctionBounds,
    node,
    table: StaticJumpTable,
) -> int | None:
    """Return the bounded table length proven by a predecessor branch."""

    index_key = table.index_register_offset, table.index_bits
    bounds_found: set[int] = set()
    for predecessor in graph.predecessors(node):
        if not _node_is_materialized_cfg_node(predecessor):
            continue
        if not _node_intersects_bounds(predecessor, bounds):
            continue
        vex = _node_vex(predecessor)
        if vex is None:
            continue
        upper_bound = _vex_guarded_index_upper_bound(vex, node.addr, index_key)
        if upper_bound is not None:
            bounds_found.add(upper_bound)

    if len(bounds_found) != 1:
        return None
    upper_bound = next(iter(bounds_found))
    entry_count = upper_bound + 1
    return entry_count if entry_count <= MAX_STATIC_JUMPTABLE_ENTRIES else None


def _vex_relative_jump_table(vex) -> StaticJumpTable | None:
    """
    Describe a bounded relative jump table encoded in one VEX indirect jump.

    The accepted form is intentionally narrow: ``next`` must add a register
    base to a loaded (optionally sign-extended) table entry, while the load
    address must be that same base plus an index scaled by the entry size. This
    covers common PIC tables without treating arbitrary computed jumps as CFG
    targets.
    """

    if vex.jumpkind != "Ijk_Boring":
        return None

    definitions = _vex_tmp_definitions(vex)
    next_expr = _resolve_vex_expr(vex.next, definitions)
    if not isinstance(next_expr, pyvex.expr.Binop) or not next_expr.op.startswith(
        "Iop_Add"
    ):
        return None

    left, right = (
        _resolve_vex_expr(next_expr.args[0], definitions),
        _resolve_vex_expr(next_expr.args[1], definitions),
    )
    candidates = ((left, right), (right, left))
    for entry_expr, base_expr in candidates:
        base_key = _vex_get_key(base_expr, definitions, vex)
        if base_key is None:
            continue

        signed_entries = False
        entry_expr = _resolve_vex_expr(entry_expr, definitions)
        if isinstance(entry_expr, pyvex.expr.Unop):
            signed_entries = "Sto" in entry_expr.op
            entry_expr = _resolve_vex_expr(entry_expr.args[0], definitions)
        if not isinstance(entry_expr, pyvex.expr.Load):
            continue

        entry_size = entry_expr.result_size(vex.tyenv) // 8
        if entry_size not in {1, 2, 4, 8}:
            continue

        address_terms = _vex_add_terms(entry_expr.addr, definitions)
        if address_terms is None:
            continue

        displacement = 0
        saw_base = False
        index_bits: int | None = None
        for term in address_terms:
            value = _vex_const_value(term, definitions)
            if value is not None:
                displacement += value
                continue
            if _vex_get_key(term, definitions, vex) == base_key:
                saw_base = True
                continue
            term = _resolve_vex_expr(term, definitions)
            if not isinstance(term, pyvex.expr.Binop) or not term.op.startswith(
                "Iop_Shl"
            ):
                break
            shift = _vex_const_value(term.args[1], definitions)
            index_key = _vex_index_key(term.args[0], definitions, vex)
            if (
                shift is None
                or index_key is None
                or 1 << shift != entry_size
                or index_bits is not None
            ):
                break
            _, index_bits = index_key
        else:
            if saw_base and index_key is not None and index_bits is not None:
                offset, bits = base_key
                return StaticJumpTable(
                    base_register_offset=offset,
                    base_bits=bits,
                    table_displacement=displacement,
                    index_register_offset=index_key[0],
                    index_bits=index_bits,
                    entry_size=entry_size,
                    endness=entry_expr.end,
                    signed_entries=signed_entries,
                )

    return None


def _constant_register_from_predecessors(
    graph: CFGGraph,
    bounds: FunctionBounds,
    node,
    register_offset: int,
) -> int | None:
    """Return one unambiguous constant register definition reaching ``node``."""

    definitions: set[int] = set()
    queue: deque[CFGNode] = deque([node])
    seen: set[CFGNode] = set()
    while queue:
        current = queue.popleft()
        if current in seen:
            continue
        seen.add(current)
        vex = _node_vex(current)
        if vex is None:
            return None

        tmp_definitions = _vex_tmp_definitions(vex)
        found_definition = False
        for stmt in reversed(vex.statements):
            if not isinstance(stmt, pyvex.stmt.Put) or stmt.offset != register_offset:
                continue
            value = _vex_const_value(stmt.data, tmp_definitions)
            if value is None:
                return None
            definitions.add(value)
            found_definition = True
            break

        if found_definition:
            continue
        queue.extend(
            predecessor
            for predecessor in graph.predecessors(current)
            if _node_is_materialized_cfg_node(predecessor)
            and _node_intersects_bounds(predecessor, bounds)
        )

    return next(iter(definitions)) if len(definitions) == 1 else None


def _unique_static_register_value(
    graph: CFGGraph,
    bounds: FunctionBounds,
    register_offset: int,
) -> int | None:
    """Return one in-bounds VEX-proven static value assigned to a register."""

    values: set[int] = set()
    for node in _iter_graph_bound_nodes(graph, bounds):
        if not _node_is_materialized_cfg_node(node):
            continue
        try:
            vex = node.block.vex
        except Exception:
            continue
        definitions = _vex_tmp_definitions(vex)
        for stmt in vex.statements:
            if not isinstance(stmt, pyvex.stmt.Put) or stmt.offset != register_offset:
                continue
            value = _vex_static_int(stmt.data, definitions)
            if value is not None:
                values.add(value)

    return next(iter(values)) if len(values) == 1 else None


def _read_static_jump_table_targets(
    project: Project,
    table: StaticJumpTable,
    base_addr: int,
    entry_count: int,
) -> tuple[int, ...]:
    """Read all targets from one VEX-proven bounded relative jump table."""

    if entry_count <= 0 or entry_count > MAX_STATIC_JUMPTABLE_ENTRIES:
        return ()

    table_addr = base_addr + table.table_displacement
    try:
        raw = project.loader.memory.load(table_addr, entry_count * table.entry_size)
    except Exception as exc:
        logger.debug(f"Custom CFG could not read jump table at {table_addr:#x}: {exc}")
        return ()

    byteorder = "little" if table.endness == "Iend_LE" else "big"
    targets: set[int] = set()
    for offset in range(0, len(raw), table.entry_size):
        entry = int.from_bytes(
            raw[offset : offset + table.entry_size],
            byteorder=byteorder,
            signed=table.signed_entries,
        )
        targets.add(base_addr + entry)

    return tuple(sorted(targets))


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


def _instruction_has_nonfallthrough_vex_semantics(
    project: Project, insn: CsInsn
) -> bool:
    """Return whether VEX models one exceptional instruction as terminal."""

    semantic = InsnSemantics(insn)
    if not semantic.may_have_nonfallthrough_vex_semantics():
        return False

    try:
        vex = project.factory.block(
            insn.address,
            size=insn.size,
            strict_block_end=True,
            cross_insn_opt=False,
        ).vex
    except Exception as exc:
        logger.warning(
            f"Custom CFG could not lift exceptional instruction at {insn.address:#x}: {exc}"
        )
        return False

    return _vex_jumpkind_is_terminal(vex.jumpkind)


def _instruction_has_unclassified_vex_transfer(project: Project, insn: CsInsn) -> bool:
    """Return whether VEX identifies an executable direct target as control flow.

    Capstone normally provides generic call/jump groups, but a few backends do
    not.  S390's ``brasl`` is one example: it carries an executable immediate
    target yet exposes no call or jump group.  Restricting the VEX check to
    that narrow shape avoids lifting every ordinary instruction during custom
    recovery while still letting VEX provide the architecture-specific answer.
    """

    semantic = InsnSemantics(insn)
    if semantic.is_control_transfer():
        return False

    target = semantic.direct_target()
    if target is None:
        return False

    obj = project.loader.find_object_containing(target)
    if obj is None:
        return False
    section = obj.find_section_containing(target)
    if section is None or not section.is_executable:
        return False

    try:
        vex = project.factory.block(
            insn.address,
            size=insn.size,
            strict_block_end=True,
            cross_insn_opt=False,
        ).vex
    except Exception as exc:
        logger.warning(
            f"Custom CFG could not lift possible control transfer at "
            f"{insn.address:#x}: {exc}"
        )
        return False

    return vex.jumpkind == "Ijk_Call" or _vex_jumpkind_is_terminal(vex.jumpkind)


def _vex_jumpkind_is_terminal(jumpkind: str) -> bool:
    """Return whether VEX marks a block as a return or a synchronous trap."""

    return jumpkind == "Ijk_Ret" or jumpkind.startswith("Ijk_Sig")


def _lift_block_terminator(
    project: Project,
    bounds: FunctionBounds,
    block_insns: list[CsInsn],
    has_nonfallthrough_vex_terminator: bool = False,
    unclassified_vex_terminator_addr: int | None = None,
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

    term_idx = control_transfer_index(project.arch.name, block_insns)
    if term_idx is None and unclassified_vex_terminator_addr is not None:
        term_idx = next(
            (
                index
                for index, insn in enumerate(block_insns)
                if insn.address == unclassified_vex_terminator_addr
            ),
            None,
        )

    if term_idx is None:
        if has_nonfallthrough_vex_terminator:
            return TerminatorInfo(jumpkind="Ijk_Terminal")

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

    # Some Capstone backends classify direct calls only as jumps. VEX is the
    # authoritative source for this semantic distinction during repair.
    if semantic.is_call() or vex.jumpkind == "Ijk_Call":
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
    project: Project,
    graph: CFGGraph,
    bounds: FunctionBounds,
    node,
) -> JumpSuccessorAnalysis | None:
    """
    Return expected vs present successors for one decoded jump block.

    VEX helps recognize conditional control flow, while Capstone remains
    authoritative for the direct target because malformed CFGFast nodes can
    expose stale VEX exit addresses. `InsnSemantics` selects the final immediate
    operand so compare-and-branch instructions do not use a condition value,
    such as S390's `-1`, as the branch address.
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

    try:
        vex = node.block.vex
    except Exception:
        vex = None

    last = InsnSemantics(last_insn)
    if not last.is_jump():
        return None
    # Calls may be members of Capstone's generic jump group. They have their
    # own call/fake-return edge semantics and must not be checked as branches.
    if last.is_call() or (vex is not None and vex.jumpkind == "Ijk_Call"):
        return None

    exit_targets: list[int] = []
    if vex is not None:
        for ins_addr, _, stmt in vex.exit_statements:
            if ins_addr != last_insn.address:
                continue
            target = getattr(stmt.dst, "value", None)
            if isinstance(target, int) and target not in exit_targets:
                exit_targets.append(target)

    is_conditional = bool(exit_targets)
    if not is_conditional and last.is_conditional_jump():
        # A malformed CFG node can retain stale VEX without the final branch
        # exit. Re-lift only the ambiguous terminator: this keeps genuine x86
        # conditionals conditional, while correctly classifying S390 `j` as a
        # direct jump instead of inventing a fallthrough edge.
        try:
            fresh_vex = project.factory.block(
                last_insn.address,
                size=last_insn.size,
                strict_block_end=True,
                cross_insn_opt=False,
            ).vex
        except Exception:
            # Preserve the conservative Capstone classification if a fresh
            # lift is unavailable; repair is safer than silently omitting an
            # actual branch successor.
            is_conditional = True
        else:
            is_conditional = any(
                ins_addr == last_insn.address
                for ins_addr, _, _ in fresh_vex.exit_statements
            )

    direct_target = last.direct_target()
    if direct_target is None:
        return None

    expected: list[JumpSuccessorExpectation] = []
    kind: Literal["conditional", "direct"]
    fallthrough_addr = last_insn.address + last_insn.size
    if is_conditional:
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
    project: Project,
    graph: CFGGraph,
    bounds: FunctionBounds,
    node,
) -> bool:
    """Return True when a jump block is missing one or more successor edges."""

    analysis = _analyze_jump_successors(project, graph, bounds, node)
    if analysis is None:
        return False

    expected_addrs = {item.addr for item in analysis.expected}
    if analysis.present == expected_addrs:
        return False

    return True


def _missing_jump_successors(
    project: Project,
    graph: CFGGraph,
    bounds: FunctionBounds,
    node,
) -> tuple[JumpSuccessorExpectation, ...]:
    """Return the subset of expected jump successors still missing in the graph."""

    analysis = _analyze_jump_successors(project, graph, bounds, node)
    if analysis is None:
        return ()

    return tuple(
        item for item in analysis.expected if item.addr not in analysis.present
    )


def _call_target_is_known_nonreturning(project: Project, node) -> bool:
    """Return True only for a direct call to an explicitly non-returning hook."""

    decoded = DecodedNode.from_node(node)
    last_insn = decoded.last
    if last_insn is None:
        return False

    target = InsnSemantics(last_insn).direct_target()
    if target is None or not project.is_hooked(target):
        return False

    return bool(getattr(project.hooked_by(target), "NO_RET", False))


def node_has_missing_call_fallthrough(
    project: Project,
    graph: CFGGraph,
    bounds: FunctionBounds,
    node,
) -> bool:
    """Return True when a returning call is missing its in-function fake return."""

    if node_has_decoding_coverage_mismatch(node):
        return False

    try:
        is_call = node.block.vex.jumpkind == "Ijk_Call"
    except Exception:
        return False
    if not is_call or _call_target_is_known_nonreturning(project, node):
        return False

    fallthrough_addr = _node_range_end(node)
    if not _is_direct_target_valid(bounds, fallthrough_addr):
        return False

    for successor in graph.successors(node):
        if successor.addr != fallthrough_addr:
            continue
        edge_data = graph.get_edge_data(node, successor) or {}
        if edge_data.get("jumpkind") == "Ijk_FakeRet":
            return False

    return True


def _is_linear_merge_successor(graph: CFGGraph, node) -> bool:
    """Return whether ``node`` should absorb its only straight-line successor.

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
    # Synthetic unresolved-jump fallbacks preserve disconnected regions but do
    # not represent real branch targets. They must not prevent a normal linear
    # merge between two adjacent materialized blocks.
    materialized_predecessors = [
        predecessor
        for predecessor in graph.predecessors(succ)
        if not _node_is_simprocedure(predecessor)
    ]
    if len(materialized_predecessors) != 1:
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

    return True


def node_has_linear_merge_successor(
    graph: CFGGraph,
    node,
) -> bool:
    """Return True when ``node`` should absorb a linear successor."""

    return _is_linear_merge_successor(graph, node)


def _decoding_coverage_anomaly(node) -> CFGAnomaly | None:
    """Return the byte-coverage anomaly for ``node``, when present."""

    if node.size == 0:
        return CFGAnomaly(
            "decoding_coverage_mismatch",
            node.addr,
            f"Node {node.addr:#x} has size zero",
        )

    decoded = DecodedNode.from_node(node)
    if decoded.insns is None:
        return CFGAnomaly(
            "decoding_coverage_mismatch",
            node.addr,
            f"Capstone inspection failed for node {node.addr:#x}: "
            f"{type(decoded.inspection_error).__name__}: {decoded.inspection_error}",
        )

    if decoded.has_exact_coverage(node):
        return None

    expected_addr = node.addr
    for insn in decoded.insns:
        if insn.address != expected_addr:
            return CFGAnomaly(
                "decoding_coverage_mismatch",
                node.addr,
                f"Node {node.addr:#x} decodes instruction at {insn.address:#x} "
                f"instead of expected {expected_addr:#x}",
            )
        expected_addr += insn.size

    node_end = node.addr + node.size
    if expected_addr != node_end:
        return CFGAnomaly(
            "decoding_coverage_mismatch",
            node.addr,
            f"Node {node.addr:#x} decoded instructions end at {expected_addr:#x}, "
            f"but node size extends to {node_end:#x}",
        )

    return None


def node_has_decoding_coverage_mismatch(node) -> bool:
    """Return True when a CFG node clearly covers bytes incorrectly."""

    return _decoding_coverage_anomaly(node) is not None


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
        if _vex_jumpkind_is_terminal(node.block.vex.jumpkind):
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


def node_has_foreign_function_owner(
    graph: CFGGraph,
    func_addr: int,
    node,
) -> bool:
    """Return True for an in-function successor mis-owned by CFGFast."""

    if node.function_address == func_addr:
        return False

    if not any(
        predecessor.function_address == func_addr
        for predecessor in graph.predecessors(node)
    ):
        return False

    return True


class CFGAnomalyDetector:
    """Classify node-local CFG anomalies and report each one only once."""

    def __init__(
        self,
        project: Project,
        graph: CFGGraph,
        bounds: FunctionBounds,
        func_addr: int,
    ) -> None:
        """Bind anomaly checks to one live CFG graph and function range."""

        self.project = project
        self.graph = graph
        self.bounds = bounds
        self.func_addr = func_addr
        self.reported_anomalies: set[tuple[str, int]] = set()

    def _report(self, anomaly: CFGAnomaly) -> None:
        """Emit ``anomaly`` once for this detector."""

        key = anomaly.kind, anomaly.addr
        if key in self.reported_anomalies:
            return
        self.reported_anomalies.add(key)
        logger.warning(anomaly.message)

    def node_has_decoding_coverage_mismatch(self, node) -> bool:
        """Check byte coverage and report the precise mismatch once."""

        anomaly = _decoding_coverage_anomaly(node)
        if anomaly is None:
            return False
        self._report(anomaly)
        return True

    def node_has_missing_jump_successor(self, node) -> bool:
        """Check direct branch edges and report missing or unexpected targets."""

        analysis = _analyze_jump_successors(self.project, self.graph, self.bounds, node)
        if analysis is None:
            return False

        expected_addrs = {item.addr for item in analysis.expected}
        if analysis.present == expected_addrs:
            return False

        label = (
            "conditional branch" if analysis.kind == "conditional" else "direct jump"
        )
        self._report(
            CFGAnomaly(
                "missing_jump_successor",
                node.addr,
                f"Node {node.addr:#x} is missing {label} successor(s) "
                f"or shows unexpected ones: expected "
                f"{', '.join(hex(t) for t in sorted(expected_addrs))}, got "
                f"{', '.join(hex(t) for t in sorted(analysis.present)) if analysis.present else '<none>'}",
            )
        )
        return True

    def missing_jump_successors(self, node) -> tuple[JumpSuccessorExpectation, ...]:
        """Return missing direct branch targets for immediate edge recovery."""

        return _missing_jump_successors(self.project, self.graph, self.bounds, node)

    def node_has_missing_call_fallthrough(self, node) -> bool:
        """Check fake-return coverage for calls and report a missing edge once."""

        if not node_has_missing_call_fallthrough(
            self.project, self.graph, self.bounds, node
        ):
            return False

        self._report(
            CFGAnomaly(
                "missing_call_fallthrough",
                node.addr,
                f"Call node {node.addr:#x} is missing fake-return successor "
                f"{_node_range_end(node):#x}",
            )
        )
        return True

    def node_has_linear_merge_successor(self, node) -> bool:
        """Check for an artificial linear split and report it once."""

        if not node_has_linear_merge_successor(self.graph, node):
            return False
        successor = next(iter(self.graph.successors(node)))
        self._report(
            CFGAnomaly(
                "linear_merge_successor",
                node.addr,
                f"Node {node.addr:#x} is split from straight-line successor "
                f"{successor.addr:#x}",
            )
        )
        return True

    def node_has_foreign_function_owner(self, node) -> bool:
        """Check for an in-bounds node retained under another function owner."""

        if not node_has_foreign_function_owner(self.graph, self.func_addr, node):
            return False
        self._report(
            CFGAnomaly(
                "foreign_function_owner",
                node.addr,
                f"Node {node.addr:#x} is reached from function {self.func_addr:#x} "
                f"but is owned by CFGFast function {node.function_address:#x}",
            )
        )
        return True

    def node_needs_repair(self, node) -> bool:
        """Return True when ``node`` violates a repair invariant."""

        return (
            self.node_has_decoding_coverage_mismatch(node)
            or node_has_decode_gap(node)
            or node_has_truncated_leaf(self.graph, self.func_addr, node)
            or self.node_has_missing_jump_successor(node)
            or self.node_has_missing_call_fallthrough(node)
            or self.node_has_linear_merge_successor(node)
        )

    def node_is_acceptable(self, forced_block_starts: set[int], node) -> bool:
        """Return True when an existing node may remain unchanged in the graph."""

        if _node_is_placeholder(node):
            return False
        if _node_has_forced_split(node, forced_block_starts):
            return False
        if getattr(node, "function_address", None) != self.func_addr:
            return False
        return not self.node_needs_repair(node)


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
            project,
            _cfg_graph(cfg),
            _lookup_function_bounds(project, func_addr),
            node,
        ):
            logger.warning(
                f"CFG anomaly for function {func_addr:#x}: missing_jump_successor "
                f"at {node.addr:#x}: direct jump block is missing one or more CFG edges"
            )
            return True
        if project is not None and node_has_missing_call_fallthrough(
            project,
            _cfg_graph(cfg),
            _lookup_function_bounds(project, func_addr),
            node,
        ):
            logger.warning(
                f"CFG anomaly for function {func_addr:#x}: "
                f"missing_call_fallthrough at {node.addr:#x}: call block is missing "
                "an in-function fake-return edge"
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


def _prune_orphan_simprocedures(graph: CFGGraph) -> int:
    """
    Remove simprocedure nodes that no longer have any incoming edges.

    Local repairs may delete the buggy seed nodes that originally pointed to an
    angr-created simprocedure such as `UnresolvableJumpTarget`. When that
    happens, the simprocedure can survive in the graph as an orphan even though
    no repaired block still reaches it. Prune those leftovers iteratively in
    case removing one orphan exposes another orphaned simprocedure behind it.
    """

    removed = 0
    while True:
        orphan_nodes = [
            node
            for node in list(graph.nodes())
            if _node_is_simprocedure(node) and graph.in_degree(node) == 0
        ]
        if not orphan_nodes:
            return removed
        _remove_nodes(graph, orphan_nodes)
        removed += len(orphan_nodes)


def _prune_placeholders(graph: CFGGraph) -> int:
    """Remove temporary placeholder nodes and return how many were pruned."""

    placeholders = [node for node in list(graph.nodes()) if _node_is_placeholder(node)]
    if placeholders:
        _remove_nodes(graph, placeholders)
    return len(placeholders)


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


def _is_unresolvable_jump_target(node) -> bool:
    """Return True for angr's synthetic unresolved indirect-jump target node."""

    return _node_is_simprocedure(node) and getattr(node, "simprocedure_name", None) == (
        "UnresolvableJumpTarget"
    )


def _node_ends_in_indirect_jump(node) -> bool:
    """Return whether VEX identifies ``node`` as an indirect boring jump."""

    try:
        vex = node.block.vex
    except (AttributeError, KeyError):
        return False
    return vex.jumpkind == "Ijk_Boring" and not isinstance(vex.next, pyvex.expr.Const)


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


def _block_has_unresolved_indirect_jump(block: BlockSpec) -> bool:
    """Return whether a recovered block needs an unresolved indirect-jump leaf."""

    return (
        block.jumpkind == "Ijk_Boring"
        and not block.direct_targets
        and block.fallthrough_addr is None
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


def _recover_block(
    project: Project, bounds: FunctionBounds, start_addr: int, stop_addrs: set[int]
) -> BlockSpec | None:
    """Decode one block starting at addr and stop on control flow or known block starts."""

    max_inst_bytes = getattr(project.arch, "max_inst_bytes", 16)
    cur = start_addr
    insns: list[CsInsn] = []
    has_delay_slot = arch_has_delay_slot(project.arch.name)
    has_nonfallthrough_vex_terminator = False
    unclassified_vex_terminator_addr: int | None = None

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

        if _instruction_has_nonfallthrough_vex_semantics(project, insn):
            has_nonfallthrough_vex_terminator = True
            break

        if _instruction_has_unclassified_vex_transfer(project, insn):
            unclassified_vex_terminator_addr = insn.address
            break

        cur = next_addr

    if not insns:
        return None

    terminator = _lift_block_terminator(
        project,
        bounds,
        insns,
        has_nonfallthrough_vex_terminator=has_nonfallthrough_vex_terminator,
        unclassified_vex_terminator_addr=unclassified_vex_terminator_addr,
    )
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
    *,
    unresolved_indirect: bool = False,
) -> bool:
    """Add one successor edge if it is not already present with the same kind."""

    if graph.has_edge(src, dst):
        edge_data = graph.get_edge_data(src, dst) or {}
        if (
            edge_data.get("jumpkind") == jumpkind
            and edge_data.get("unresolved_indirect", False) == unresolved_indirect
        ):
            return False
    graph.add_edge(
        src,
        dst,
        jumpkind=jumpkind,
        unresolved_indirect=unresolved_indirect,
    )
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
        self.anomalies = CFGAnomalyDetector(project, self.graph, self.bounds, func_addr)
        self.mutation_revision = 0
        self._bound_nodes_revision = -1
        self._bound_nodes_snapshot: tuple[CFGNode, ...] = ()
        self.stats = CustomCFGStats()
        self.stats.input_blocks, self.stats.input_edges = self._graph_shape()
        self.queue: deque[tuple[str, int]] = deque()
        self.pending: dict[tuple[str, int], PendingObligation] = {}
        self.repaired_nodes: set[CFGNode] = set()
        self.leaders = BlockLeaderRegistry({func_addr: {"function_entry"}})
        self.processed_counts: dict[int, int] = {}
        self.last_requeue_states: dict[
            tuple[str, int],
            tuple[int, tuple[bool, tuple[tuple[int, EdgeJumpKind], ...]]],
        ] = {}
        self.resolved_static_table_sources: set[int] = set()
        self.iterations = 0
        self._stop_starts_revision = -1
        self._stop_starts: set[int] = set()

    def _graph_shape(self) -> tuple[int, int]:
        """Return the number of in-bounds blocks and their outgoing CFG edges."""

        blocks = {
            node for node in self._bound_nodes() if _node_is_materialized_cfg_node(node)
        }
        edges = sum(
            1 for source, _, _ in self.graph.edges(data=True) if source in blocks
        )
        return len(blocks), edges

    def _bound_nodes(self) -> tuple[CFGNode, ...]:
        """Return a revision-scoped snapshot of live in-bounds CFG nodes."""

        if self._bound_nodes_revision != self.mutation_revision:
            self._bound_nodes_snapshot = tuple(
                _iter_graph_bound_nodes(self.graph, self.bounds)
            )
            self._bound_nodes_revision = self.mutation_revision
        return self._bound_nodes_snapshot

    def _nodes_at_addr(self, addr: int) -> list[CFGNode]:
        """Return cached in-bounds nodes beginning at ``addr``."""

        return [node for node in self._bound_nodes() if node.addr == addr]

    def _covering_nodes(self, addr: int) -> list[CFGNode]:
        """Return cached in-bounds nodes whose byte range covers ``addr``."""

        return [
            node
            for node in self._bound_nodes()
            if node.addr <= addr < _node_range_end(node)
        ]

    def log_stats(self) -> None:
        """Capture the final graph shape and emit one custom-repair summary."""

        try:
            self.stats.output_blocks, self.stats.output_edges = self._graph_shape()
            self.stats.output_anomalies = len(self._anomalous_addrs())
        except Exception as exc:
            logger.warning(
                f"Custom CFG could not finish collecting stats for {self.func_addr:#x}: "
                f"{exc}"
            )
        logger.info(
            f"Custom CFG stats for function {self.func_addr:#x}: {self.stats.as_dict()}"
        )

    def _is_preservable_seed_node(self, node) -> bool:
        """
        Return True when an unrepaired seed node may still preserve CFG structure.

        Malformed seed fragments should not freeze extra block leaders or
        successor expectations. Once repaired, the replacement node will drive
        discovery with its own recovered semantics.
        """

        return node in self.repaired_nodes or self.anomalies.node_is_acceptable(
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
        *,
        unresolved_indirect: bool = False,
    ) -> bool:
        """Add an edge and record whether it changed the live graph."""

        changed = _add_successor_edge(
            self.graph,
            src,
            dst,
            jumpkind,
            unresolved_indirect=unresolved_indirect,
        )
        if changed:
            self.stats.edges_added += 1
            self._note_mutation()
        return changed

    def _canonicalize_function_ownership(self) -> None:
        """Assign all in-bounds custom-CFG blocks to the requested function."""

        reassigned = 0
        for node in self._bound_nodes():
            if not _node_is_materialized_cfg_node(node):
                continue
            if node.function_address == self.func_addr:
                continue
            # CFGFast can create provisional functions for disconnected code
            # regions. Once custom repair includes those in the symbol-bounded
            # graph, renderers must not hide them based on that stale owner.
            node.function_address = self.func_addr
            reassigned += 1
        if reassigned:
            self.stats.function_owners_canonicalized += reassigned
            logger.info(
                f"Assigned {reassigned} in-bounds CFG block(s) to function "
                f"{self.func_addr:#x}"
            )

    def _resolve_static_jump_tables(self) -> int:
        """Recover high-confidence relative table targets from unresolved jumps."""

        resolved_sources = 0
        for node in self._bound_nodes():
            if not _node_is_materialized_cfg_node(node):
                continue

            vex = _node_vex(node)
            if vex is None:
                continue
            table = _vex_relative_jump_table(vex)
            if table is None or table.base_bits != self.project.arch.bits:
                continue
            entry_count = _guarded_jump_table_entry_count(
                self.graph,
                self.bounds,
                node,
                table,
            )
            if entry_count is None:
                continue
            base_addr = _constant_register_from_predecessors(
                self.graph,
                self.bounds,
                node,
                table.base_register_offset,
            )
            if base_addr is None:
                # A disconnected table dispatcher may not have a complete
                # predecessor path back to its base definition. Scan the
                # bounded VEX blocks instead, but accept a value only when all
                # static definitions for this register agree.
                base_addr = _unique_static_register_value(
                    self.graph,
                    self.bounds,
                    table.base_register_offset,
                )
            if base_addr is None:
                continue
            targets = _read_static_jump_table_targets(
                self.project,
                table,
                base_addr,
                entry_count,
            )
            if not targets:
                continue

            existing_target_addrs = {
                successor.addr for successor in self.graph.successors(node)
            }
            missing_targets = [
                target for target in targets if target not in existing_target_addrs
            ]
            if not missing_targets:
                unresolved_targets = []
            else:
                unresolved_targets = [
                    successor
                    for successor in self.graph.successors(node)
                    if _is_unresolvable_jump_target(successor)
                ]

            if not missing_targets and not unresolved_targets:
                continue

            logger.info(
                f"Resolved static jump table at {node.addr:#x} with "
                f"{len(missing_targets)} missing target(s)"
            )
            for target_addr in missing_targets:
                self._resolve_successor(
                    node,
                    target_addr,
                    "Ijk_Boring",
                    reason=f"static_jump_table_of_{node.addr:#x}",
                    preserve_exact_addr=True,
                    materialize_external=True,
                )

            # The guarded VEX form and every table entry have now been proven,
            # so the original indirect-jump placeholder is no longer needed.
            for unresolved_target in unresolved_targets:
                self.graph.remove_edge(node, unresolved_target)
                self._note_mutation()
            self.stats.static_jump_targets_added += len(missing_targets)
            self.stats.unresolved_jump_edges_removed += len(unresolved_targets)
            self.resolved_static_table_sources.add(node.addr)
            self.stats.static_jump_tables_resolved = len(
                self.resolved_static_table_sources
            )
            resolved_sources += 1

        return resolved_sources

    def _unresolved_jump_fallback_nodes(self) -> list[CFGNode]:
        """Return synthetic leaves still reached from unresolved indirect jumps."""

        return sorted(
            {
                successor
                for node in self._bound_nodes()
                if _node_is_materialized_cfg_node(node)
                for successor in self.graph.successors(node)
                if _is_unresolvable_jump_target(successor)
            },
            key=lambda node: node.addr,
        )

    def _reachable_from_entry(self) -> set[CFGNode]:
        """Return all graph nodes reachable through the current entry edges."""

        reachable: set[CFGNode] = set()
        queue: deque[CFGNode] = deque(self._nodes_at_addr(self.func_addr))
        while queue:
            node = queue.popleft()
            if node in reachable:
                continue
            reachable.add(node)
            queue.extend(self.graph.successors(node))
        return reachable

    def _disconnected_function_nodes(self) -> list[CFGNode]:
        """Return every in-function node disconnected from the entry graph."""

        reachable = self._reachable_from_entry()
        return sorted(
            (
                node
                for node in self._bound_nodes()
                if _node_is_materialized_cfg_node(node) and node not in reachable
            ),
            key=lambda node: node.addr,
        )

    def _attach_unresolved_jump_fallbacks(self) -> bool:
        """Keep disconnected seed regions reachable through unresolved jump leaves."""

        fallback_nodes = self._unresolved_jump_fallback_nodes()
        if not fallback_nodes:
            return False
        disconnected_nodes = self._disconnected_function_nodes()
        changed = False
        for fallback in fallback_nodes:
            for node in disconnected_nodes:
                if self._add_edge(fallback, node, "Ijk_Boring"):
                    self.stats.unresolved_fallback_edges_added += 1
                    changed = True
        if changed:
            logger.info(
                f"Connected {len(disconnected_nodes)} disconnected function block(s) through "
                f"{len(fallback_nodes)} unresolved indirect-jump target(s)"
            )
        return changed

    def _flatten_single_source_unresolved_fallback(self) -> bool:
        """Replace one unambiguous unresolved-jump placeholder with candidate edges.

        A single indirect-jump source and a single
        ``UnresolvableJumpTarget`` form a synthetic intermediary rather than a
        meaningful CFG block. Once its in-function candidate targets have been
        attached, move those edges to the dispatcher itself while retaining
        their ``unresolved_indirect`` marker for rendering. Multiple fallback
        nodes, extra dispatcher successors, or non-local targets are left
        untouched because the intermediary still carries useful structure.
        """

        fallbacks = self._unresolved_jump_fallback_nodes()
        if len(fallbacks) != 1:
            return False

        fallback = fallbacks[0]
        predecessors = list(self.graph.predecessors(fallback))
        targets = list(self.graph.successors(fallback))
        if len(predecessors) != 1 or not targets:
            return False

        source = predecessors[0]
        if (
            not _node_is_materialized_cfg_node(source)
            or not _node_intersects_bounds(source, self.bounds)
            or not _node_ends_in_indirect_jump(source)
            or list(self.graph.successors(source)) != [fallback]
            or any(
                not _node_is_materialized_cfg_node(target)
                or not _node_intersects_bounds(target, self.bounds)
                for target in targets
            )
        ):
            return False

        for target in targets:
            edge_data = self.graph.get_edge_data(fallback, target) or {}
            self._add_edge(
                source,
                target,
                edge_data.get("jumpkind", "Ijk_Boring"),
                unresolved_indirect=True,
            )

        self.graph.remove_edge(source, fallback)
        self.graph.remove_node(fallback)
        self.stats.unresolved_fallbacks_flattened += 1
        self.stats.unresolved_candidate_edges_flattened += len(targets)
        self._note_mutation()
        logger.info(
            f"Flattened unresolved indirect jump at {source.addr:#x} to "
            f"{len(targets)} in-function candidate target(s)"
        )
        return True

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

        covering_entry = self._resolve_covering_entry(obligation)
        if covering_entry is not None:
            return covering_entry

        for node in self._nodes_at_addr(addr):
            if self._is_preservable_seed_node(node):
                self._connect_source_to_node(obligation, node)
                return node

        placeholder = self._claim_placeholder(obligation)
        self._queue_if_needed(obligation)
        return placeholder

    def _resolve_covering_entry(
        self,
        obligation: RepairObligation,
    ) -> CFGNode | None:
        """Resolve an entry covered by an existing node, or leave it deferred."""

        addr = obligation.addr
        for node in self._covering_nodes(addr):
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
                self.stats.explicit_splits += 1
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

        return None

    def _claim_placeholder(
        self,
        obligation: RepairObligation,
    ) -> CFGNode:
        """Get the target placeholder for an entry and attach its edge claims."""

        placeholder = next(
            (
                node
                for node in self._nodes_at_addr(obligation.addr)
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
            self.stats.placeholders_created += 1
            self._note_mutation()
        self._connect_source_to_node(obligation, placeholder)
        return placeholder

    def _recover_entry_now(self, obligation: RepairObligation) -> CFGNode | None:
        """Resolve an existing entry or decode one immediately for local repair."""

        exact_nodes = [
            node
            for node in self._nodes_at_addr(obligation.addr)
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
                if self.anomalies.node_is_acceptable(
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
    ) -> None:
        """Turn merged work back into one or more ordinary queue requests."""

        reason = ", ".join(sorted(obligation.reasons))
        if not obligation.edge_claims:
            self._queue_if_needed(
                RepairObligation(
                    addr=obligation.addr,
                    reason=reason,
                    action=obligation.action,
                    preserve_exact_addr=obligation.preserve_exact_addr,
                )
            )
            return

        for claim in obligation.edge_claims:
            self._queue_if_needed(
                RepairObligation(
                    addr=obligation.addr,
                    reason=reason,
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

        for expectation in self.anomalies.missing_jump_successors(node):
            self._resolve_successor(
                node,
                expectation.addr,
                expectation.jumpkind,
                reason=f"missing_successor_of_{node.addr:#x}",
                preserve_exact_addr=expectation.preserve_exact_addr,
                resolution_policy="immediate",
                materialize_external=True,
            )

        if self.anomalies.node_needs_repair(node):
            self._queue_if_needed(
                RepairObligation(
                    addr=node.addr,
                    reason=f"remaining_anomaly_at_{node.addr:#x}",
                )
            )

    def _reconcile_addr(self, addr: int) -> None:
        """Reconcile every materialized node currently starting at ``addr``."""

        for node in self._nodes_at_addr(addr):
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
            self.stats.external_targets_created += 1
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

        if self._stop_starts_revision != self.mutation_revision:
            self._stop_starts = self._current_stop_starts()
            self._stop_starts_revision = self.mutation_revision

        return {
            target
            for target in self._stop_starts
            if addr < target < self.bounds.end_addr
        }

    def _current_stop_starts(self) -> set[int]:
        """
        Return the live leader starts that must bound recovered blocks.

        A linear split candidate ``A -> B`` must not make ``B`` a hard stop:
        revisiting ``A`` needs to absorb its straight-line tail.  Collecting
        every such tail in one pass is equivalent to checking each leader
        separately, but avoids repeatedly iterating angr's spilled CFG nodes.
        The result is invalidated by ``mutation_revision`` whenever the graph
        or leader set changes.
        """

        leaders = self.leaders.copy()
        linear_tail_starts: set[int] = set()
        for node in self._bound_nodes():
            if not _node_is_materialized_cfg_node(node):
                continue

            if self._is_preservable_seed_node(node):
                direct_targets, fallthrough_addr = self._preserved_successor_starts(
                    node
                )
                for target in direct_targets:
                    if self.bounds.addr <= target < self.bounds.end_addr:
                        leaders.add(target, "direct_target")
                if (
                    fallthrough_addr is not None
                    and self.bounds.addr <= fallthrough_addr < self.bounds.end_addr
                ):
                    leaders.add(fallthrough_addr, "fallthrough")

            if not self.anomalies.node_has_linear_merge_successor(node):
                continue
            linear_tail_starts.update(
                successor.addr for successor in self.graph.successors(node)
            )

        return leaders.starts() - linear_tail_starts

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
            for node in self._bound_nodes()
            if node.addr == recovered_start
            or _ranges_overlap(
                node.addr, _node_range_end(node), recovered_start, recovered_end
            )
            or (_node_is_placeholder(node) and node.addr == recovered_start)
        ]
        removed_set = set(removed_nodes)
        removed_blocks = [
            node for node in removed_nodes if _node_is_materialized_cfg_node(node)
        ]
        linear_merges = sum(
            1
            for node in removed_blocks
            if _is_linear_merge_successor(self.graph, node)
            and any(
                successor in removed_set for successor in self.graph.successors(node)
            )
        )

        incoming_edges = [
            (src, dst, dict(data))
            for src, dst, data in list(self.graph.edges(data=True))
            if dst in removed_set and src not in removed_set
        ]
        unresolved_jump_edges = [
            (dst, data.get("jumpkind", "Ijk_Boring"))
            for src, dst, data in list(self.graph.edges(data=True))
            if src in removed_set and _is_unresolvable_jump_target(dst)
        ]

        _remove_nodes(self.graph, removed_nodes)
        self.stats.blocks_redecoded += 1
        self.stats.blocks_replaced += len(removed_blocks)
        self.stats.linear_block_merges += linear_merges

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
            if not self.anomalies.node_is_acceptable(
                self._explicit_split_starts(),
                pred,
            ):
                continue

            self.ensure_expected_successors(pred)

        # A recovered indirect jump has no concrete BlockSpec target. Retain
        # CFGFast's unresolved placeholder until static table recovery proves
        # real targets and deliberately replaces it.
        if _block_has_unresolved_indirect_jump(block):
            for unresolved_target, jumpkind in unresolved_jump_edges:
                self._add_edge(recovered_node, unresolved_target, jumpkind)

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

    def _cleanup(self) -> bool:
        """Prune stale nodes and report whether cleanup changed the live graph."""

        self.stats.cleanup_rounds += 1
        changed = self._attach_unresolved_jump_fallbacks()
        changed |= self._flatten_single_source_unresolved_fallback()
        unreachable_removed = _cleanup_unreachable_function_nodes(
            self.graph, self.bounds, self.func_addr
        )
        placeholders_pruned = _prune_placeholders(self.graph)
        orphan_simprocedures_pruned = _prune_orphan_simprocedures(self.graph)
        self.stats.unreachable_blocks_removed += unreachable_removed
        self.stats.placeholders_pruned += placeholders_pruned
        self.stats.orphan_simprocedures_pruned += orphan_simprocedures_pruned
        changed |= bool(
            unreachable_removed or placeholders_pruned or orphan_simprocedures_pruned
        )
        if changed:
            self._note_mutation()
        return changed

    def _anomalous_addrs(self) -> list[int]:
        """Return the current in-bounds materialized block starts needing repair."""

        return sorted(
            {
                node.addr
                for node in self._bound_nodes()
                if _node_is_materialized_cfg_node(node)
                and self.anomalies.node_needs_repair(node)
            }
        )

    def _initial_anomalous_addrs(self) -> list[int]:
        """Return seed anomalies plus in-bounds nodes mis-owned by CFGFast."""

        seed_anomalies = {
            node.addr
            for node in _iter_seed_function_nodes(self.seed_cfg, self.func_addr)
            if self.anomalies.node_needs_repair(node)
        }
        ownership_boundaries = {
            node.addr
            for node in self._bound_nodes()
            if self.anomalies.node_has_foreign_function_owner(node)
        }
        return sorted(seed_anomalies | ownership_boundaries)

    def _queue_recoveries(self, addrs: Iterable[int], reason: str) -> None:
        """Seed ordinary recovery work for a group of anomalous block starts."""

        for addr in addrs:
            self.ensure_block_entry(RepairObligation(addr=addr, reason=reason))

    def _drain_worklist(self) -> None:
        """Process queued work while enforcing the global repair iteration limit."""

        while self.queue:
            self.iterations += 1
            if self.iterations > MAX_CUSTOM_CFG_WORKLIST_ITERATIONS:
                raise RuntimeError(
                    "Custom CFG worklist exceeded "
                    f"{MAX_CUSTOM_CFG_WORKLIST_ITERATIONS} iterations for {self.func_addr:#x}; "
                    f"top counts: {self.processed_counts}"
                )

            key = self.queue.popleft()
            obligation = self.pending.pop(key)
            self.stats.worklist_obligations += 1
            addr = obligation.addr
            self.processed_counts[addr] = self.processed_counts.get(addr, 0) + 1
            if self.processed_counts[addr] <= 5:
                logger.debug(
                    f"Custom CFG processing {addr:#x} for function {self.func_addr:#x} "
                    f"(visit {self.processed_counts[addr]})"
                )

            self._process_obligation(obligation)
            self._record_obligation_progress(key, obligation)

    def _process_reconciliation(self, obligation: PendingObligation) -> None:
        """Reconcile every live node at one queued address."""

        self._reconcile_addr(obligation.addr)

    def _process_recovery(self, obligation: PendingObligation) -> None:
        """Recover a queued address or requeue the work needed to expose it."""

        addr = obligation.addr
        current_nodes = self._nodes_at_addr(addr)
        covering_nodes = [
            node
            for node in self._covering_nodes(addr)
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

        # Capture seed anomalies before any static table recovery mutates the
        # input graph. The normal initial classification below remains in its
        # original order so the repair behavior itself does not change.
        self.stats.input_anomalies = len(self._initial_anomalous_addrs())
        resolved_tables = self._resolve_static_jump_tables()
        initial_bad_addrs = self._initial_anomalous_addrs()
        if initial_bad_addrs:
            logger.info(
                f"Repairing seed CFG for function {self.func_addr:#x} with "
                f"{len(initial_bad_addrs)} anomalous block start(s)"
            )
        elif resolved_tables:
            logger.info(
                f"Repairing seed CFG for function {self.func_addr:#x} after resolving "
                f"{resolved_tables} static jump table(s)"
            )

        self._queue_recoveries(initial_bad_addrs, "seed_anomaly")

        while True:
            self._drain_worklist()

            # Worklist recovery can replace an indirect-dispatch source and
            # therefore discard table edges found before repair. Re-scan the
            # live nodes so a recovered source receives its proven targets.
            resolved_tables += self._resolve_static_jump_tables()

            cleanup_changed = self._cleanup()
            remaining_bad_addrs = self._anomalous_addrs()
            if not remaining_bad_addrs:
                break
            if not cleanup_changed:
                logger.warning(
                    f"Custom CFG repair for {self.func_addr:#x} stopped with "
                    f"{len(remaining_bad_addrs)} unresolved anomaly start(s): "
                    f"{', '.join(hex(addr) for addr in remaining_bad_addrs)}"
                )
                break

            logger.info(
                f"Cleanup exposed {len(remaining_bad_addrs)} anomaly start(s) for "
                f"function {self.func_addr:#x}; continuing repair"
            )
            self._queue_recoveries(remaining_bad_addrs, "post_cleanup_anomaly")
            if not self.queue:
                logger.warning(
                    f"Custom CFG repair for {self.func_addr:#x} could not queue "
                    "cleanup-exposed anomalies"
                )
                break

        self._canonicalize_function_ownership()
        result = CustomCFG(
            graph=self.graph,
            model=_custom_model_marker(),
            functions=self.seed_cfg.functions,
            kb=self.seed_cfg.kb,
        )
        return result


def _cleanup_unreachable_function_nodes(
    graph: CFGGraph, bounds: FunctionBounds, func_addr: int
) -> int:
    """Remove unreachable function nodes and return how many were pruned."""

    entry_nodes = _nodes_at_addr(graph, bounds, func_addr)
    if not entry_nodes:
        return 0

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
    return len(stale_nodes)


def _remove_nodes(graph: CFGGraph, nodes: Iterable[CFGNode]) -> None:
    """Remove a batch of nodes through the graph wrapper's public API."""

    for node in nodes:
        graph.remove_node(node)


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
    session = _RepairSession(project, seed_cfg, func_addr)
    try:
        return session.run()
    finally:
        # Keep transformation counters available even when custom repair fails.
        session.log_stats()
