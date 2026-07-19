"""Jump-target primitives shared by custom CFG reconstruction."""

from __future__ import annotations

from collections import deque
from collections.abc import Iterable
from typing import Any, Literal

from angr import Project
from angr.knowledge_plugins.cfg import CFGNode
from loguru import logger
import pyvex

from bingraph.helpers.capstone import InsnSemantics
from .decode import DecodedNode, lift_instruction_vex
from .graph import (
    CFGGraph,
    iter_graph_bound_nodes as _iter_graph_bound_nodes,
    node_intersects_bounds as _node_intersects_bounds,
    node_is_materialized_cfg_node as _node_is_materialized_cfg_node,
    node_is_placeholder as _node_is_placeholder,
    node_vex as _node_vex,
)
from .models import (
    CFGAnomaly,
    FunctionBounds,
    JumpSuccessorAnalysis,
    JumpSuccessorExpectation,
    StaticJumpTable,
)


# Static table recovery is deliberately bounded. Larger index domains require a
# stronger range proof than the local VEX matcher currently provides.
MAX_STATIC_JUMPTABLE_ENTRIES = 256


def is_direct_target_valid(bounds: FunctionBounds, target: int | None) -> bool:
    """Return whether one direct target remains inside the function range."""

    return target is not None and bounds.addr <= target < bounds.end_addr


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
    if vex.jumpkind == "Ijk_NoDecode":
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


def _resolve_direct_branch_target(
    project: Project,
    bounds: FunctionBounds,
    capstone_target: int | None,
    vex_exit_targets: Iterable[int],
) -> int | None:
    """Choose a direct branch target from Capstone with a bounded VEX fallback.

    Capstone normally supplies the concrete target, but some architectures
    expose a PC-relative displacement instead. When that value is not mapped
    in the loaded binary and VEX supplies exactly one mapped exit, the VEX
    target is the unambiguous absolute address. This includes a direct branch
    outside the current symbol bounds, which must become an external target
    rather than repeatedly repairing an impossible relative displacement.
    Mapped Capstone targets may point outside this function, so retain them:
    malformed CFGFast VEX metadata can still be stale.
    """

    if capstone_target is not None and project.loader.find_object_containing(
        capstone_target
    ):
        return capstone_target

    mapped_exits = {
        target
        for target in vex_exit_targets
        if project.loader.find_object_containing(target)
    }
    if len(mapped_exits) == 1:
        return next(iter(mapped_exits))

    return capstone_target


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

    from .anomalies import _can_decode_block_at, node_has_decoding_coverage_mismatch

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

    # Some instruction encodings, including RISC-V ``c.jr ra``, are exposed
    # by Capstone as generic jumps rather than returns. Re-lift just the
    # terminator to avoid treating an old CFG node's stale fallthrough as a
    # successor required by the recovered block.
    terminator_vex = lift_instruction_vex(project, last_insn)
    terminator_jumpkind = getattr(terminator_vex, "jumpkind", "")
    if terminator_jumpkind == "Ijk_Ret" or terminator_jumpkind.startswith("Ijk_Sig"):
        return None

    # A normalized CFGFast node can span many instructions. Its full-block
    # VEX ``next`` may therefore describe an earlier stale split rather than
    # the final branch Capstone decoded above. Use a fresh lift of exactly that
    # terminator for its unconditional target, but retain the full node's exit
    # statements for conditional classification: some ARM encodings expose
    # predicated direct branches as generic ``b`` instructions in Capstone.
    fresh_vex_has_control_flow = (
        terminator_vex is not None and terminator_jumpkind != "Ijk_NoDecode"
    )
    target_vex = terminator_vex if fresh_vex_has_control_flow else vex
    vex_has_control_flow = vex is not None and vex.jumpkind != "Ijk_NoDecode"
    exit_targets: list[int] = []
    if vex_has_control_flow:
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
        # direct jump instead of inventing a fallthrough edge. Keep the fresh
        # targets too: some architectures expose a PC-relative Capstone
        # operand, while the short VEX lift provides the resolved address.
        fresh_vex = terminator_vex
        if fresh_vex is None:
            # Preserve the conservative Capstone classification if a fresh
            # lift is unavailable; repair is safer than silently omitting an
            # actual branch successor.
            is_conditional = True
        else:
            exit_targets = [
                target
                for ins_addr, _, stmt in fresh_vex.exit_statements
                if ins_addr == last_insn.address
                if isinstance(target := getattr(stmt.dst, "value", None), int)
            ]
            is_conditional = bool(exit_targets)

    vex_branch_targets = list(exit_targets)
    if (
        not is_conditional
        and vex_has_control_flow
        and target_vex is not None
        and isinstance(target_vex.next, pyvex.expr.Const)
        and isinstance(target_vex.next.con.value, int)
    ):
        # An unconditional branch is represented by VEX's default successor,
        # not an Exit statement. This matters for architectures whose
        # Capstone operands retain a PC-relative displacement.
        vex_branch_targets.append(target_vex.next.con.value)

    direct_target = _resolve_direct_branch_target(
        project,
        bounds,
        last.direct_target(),
        vex_branch_targets,
    )
    if direct_target is None:
        return None

    expected: list[JumpSuccessorExpectation] = []
    kind: Literal["conditional", "direct"]
    fallthrough_addr = last_insn.address + last_insn.size
    if is_conditional:
        expected.append(JumpSuccessorExpectation(direct_target, "Ijk_Boring", True))
        if _can_decode_block_at(project, bounds, fallthrough_addr):
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


def _missing_jump_successor_anomaly(
    project: Project,
    graph: CFGGraph,
    bounds: FunctionBounds,
    node,
) -> CFGAnomaly | None:
    """Return the direct-jump successor anomaly for ``node``, if any."""

    analysis = _analyze_jump_successors(project, graph, bounds, node)
    if analysis is None:
        return None

    expected_addrs = {item.addr for item in analysis.expected}
    if analysis.present == expected_addrs:
        return None

    label = "conditional branch" if analysis.kind == "conditional" else "direct jump"
    return CFGAnomaly(
        "missing_jump_successor",
        node.addr,
        f"Node {node.addr:#x} is missing {label} successor(s) or shows "
        f"unexpected ones: expected {', '.join(hex(t) for t in sorted(expected_addrs))}, "
        f"got {', '.join(hex(t) for t in sorted(analysis.present)) if analysis.present else '<none>'}",
    )


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
