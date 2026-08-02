"""Jump-target primitives shared by custom CFG reconstruction.

Static jump-table recovery is deliberately VEX-driven and therefore portable
across architectures when their lifted transfer has a recognized shape.  It
supports direct and relative entries, one-, two-, four-, and eight-byte table
entries, VEX endianness and signed-entry semantics, guard-derived index bounds,
constant bases, and bases proven from predecessor register definitions.  The
32-bit x86 PC-thunk helper is a narrow supplement for PIC code whose table base
is not retained as a VEX constant.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Iterable
from typing import Any, Literal

from angr import Project
from angr.knowledge_plugins.cfg import CFGNode
from loguru import logger
import pyvex

from bingraph.helpers.capstone import (
    InsnSemantics,
    arch_has_delay_slot,
    control_transfer_index,
)
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


def static_jump_target_rejection_reason(project: Project, target: int) -> str | None:
    """Return why one static-table entry is unsafe to materialize, if any.

    A table shape and finite index prove that entries are consulted, but do not
    prove arbitrary values read from the table are executable CFG destinations.
    Executable targets outside the function remain valid external leaves; data
    and CLE's synthetic extern-address space remain unresolved.
    """

    obj = project.loader.find_object_containing(target)
    if obj is None:
        return "unmapped"
    if obj is getattr(project.loader, "extern_object", None):
        return "synthetic"

    find_section = getattr(obj, "find_section_containing", None)
    section = find_section(target) if callable(find_section) else None
    if section is not None:
        return None if getattr(section, "is_executable", False) else "non_executable"

    find_segment = getattr(obj, "find_segment_containing", None)
    segment = find_segment(target) if callable(find_segment) else None
    if segment is not None:
        return None if getattr(segment, "is_executable", False) else "non_executable"

    return "non_executable"


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


def _vex_register_with_displacement(
    expr, definitions: dict[int, Any], vex
) -> tuple[tuple[int, int], int] | None:
    """Return one register plus its static displacement from a VEX expression."""

    register_key = _vex_get_key(expr, definitions, vex)
    if register_key is not None:
        return register_key, 0

    terms = _vex_add_terms(expr, definitions)
    if terms is None:
        return None

    displacement = 0
    register_key = None
    for term in terms:
        value = _vex_const_value(term, definitions)
        if value is not None:
            displacement += value
            continue
        key = _vex_get_key(term, definitions, vex)
        if key is None or register_key is not None:
            return None
        register_key = key
    return (register_key, displacement) if register_key is not None else None


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


def _vex_index_key(
    expr,
    definitions: dict[int, Any],
    vex,
    *,
    allow_full_width: bool = False,
) -> tuple[int, int] | None:
    """Return the original register identity for a table index expression."""

    expr = _resolve_vex_expr(expr, definitions)
    # A range guard often compares a narrowed view of the register used to
    # address the table (for example, x86 ``cmp r8d, limit`` before indexing
    # with ``r8``). Preserve the source register across integer-width casts.
    while (
        isinstance(expr, pyvex.expr.Unop)
        and expr.op.startswith("Iop_")
        and "to" in expr.op
    ):
        expr = _resolve_vex_expr(expr.args[0], definitions)
    if isinstance(expr, pyvex.expr.Binop) and expr.op.startswith("Iop_And"):
        left = _vex_index_key(expr.args[0], definitions, vex)
        right = _vex_index_key(expr.args[1], definitions, vex)
        return left or right
    key = _vex_get_key(expr, definitions, vex)
    if key is not None and key[1] > 0 and (allow_full_width or key[1] <= 8):
        return key
    return None


def _vex_static_int(expr, definitions: dict[int, Any]) -> int | None:
    """Evaluate the small constant-only VEX expressions used in branch guards."""

    value = _vex_const_value(expr, definitions)
    if value is not None:
        return value

    expr = _resolve_vex_expr(expr, definitions)
    if (
        isinstance(expr, pyvex.expr.Unop)
        and expr.op.startswith("Iop_")
        and "to" in expr.op
    ):
        value = _vex_static_int(expr.args[0], definitions)
        result_bits = expr.op.rsplit("to", maxsplit=1)[-1]
        if value is None or not result_bits.isdecimal():
            return None
        return value & ((1 << int(result_bits)) - 1)
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
    for exit_index, stmt in enumerate(vex.statements):
        if not isinstance(stmt, pyvex.stmt.Exit):
            continue
        if getattr(stmt.dst, "value", None) != target_addr:
            continue

        guard = _resolve_vex_expr(stmt.guard, definitions)
        while isinstance(guard, pyvex.expr.Unop):
            guard = _resolve_vex_expr(guard.args[0], definitions)
        if not isinstance(guard, pyvex.expr.Binop):
            continue
        if not guard.op.endswith("U"):
            continue
        if not _vex_guard_matches_index_register(
            guard.args[0], index_key, definitions, vex, vex.statements[:exit_index]
        ):
            continue
        bound = _vex_static_int(guard.args[1], definitions)
        if bound is None:
            continue
        if "CmpLE" in guard.op:
            return bound
        if "CmpLT" in guard.op and bound > 0:
            return bound - 1
    return None


def _vex_guard_matches_index_register(
    expr,
    index_key: tuple[int, int],
    definitions: dict[int, Any],
    vex,
    preceding_statements: tuple[Any, ...] | list[Any],
) -> bool:
    """Return whether a guard expression is the index register's current value."""

    if _vex_index_key(expr, definitions, vex, allow_full_width=True) == index_key:
        return True

    resolved_expr = _resolve_vex_expr(expr, definitions)
    if resolved_expr is None:
        return False
    for stmt in reversed(preceding_statements):
        if not isinstance(stmt, pyvex.stmt.Put) or stmt.offset != index_key[0]:
            continue
        value = _resolve_vex_expr(stmt.data, definitions)
        return value is resolved_expr or value == resolved_expr
    return False


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


def _vex_relative_jump_table(
    vex, *, allow_full_width_index: bool = False
) -> StaticJumpTable | None:
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
        base = _vex_register_with_displacement(base_expr, definitions, vex)
        static_base_addr = None
        if base is None:
            static_base_addr = _vex_const_value(base_expr, definitions)
            if static_base_addr is None:
                continue
            base_key = None
            target_displacement = 0
            base_bits = base_expr.result_size(vex.tyenv)
        else:
            base_key, target_displacement = base
            base_bits = base_key[1]

        if base_bits <= 0:
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

        constant_total = 0
        displacement = 0
        saw_base = False
        index_key: tuple[int, int] | None = None
        index_bits: int | None = None
        for term in address_terms:
            value = _vex_const_value(term, definitions)
            if value is not None:
                constant_total += value
                continue
            if (
                base_key is not None
                and _vex_get_key(term, definitions, vex) == base_key
            ):
                saw_base = True
                continue
            term = _resolve_vex_expr(term, definitions)
            if not isinstance(term, pyvex.expr.Binop) or not term.op.startswith(
                "Iop_Shl"
            ):
                break
            shift = _vex_const_value(term.args[1], definitions)
            index_key = _vex_index_key(
                term.args[0],
                definitions,
                vex,
                allow_full_width=allow_full_width_index,
            )
            if (
                shift is None
                or index_key is None
                or 1 << shift != entry_size
                or index_bits is not None
            ):
                break
            _, index_bits = index_key
        else:
            if static_base_addr is not None:
                mask = (1 << base_bits) - 1
                displacement = (constant_total - static_base_addr) & mask
                saw_base = True
            else:
                displacement = constant_total
            if saw_base and index_key is not None and index_bits is not None:
                offset = base_key[0] if base_key is not None else None
                table_displacement = displacement & ((1 << base_bits) - 1)
                return StaticJumpTable(
                    base_register_offset=offset,
                    base_bits=base_bits,
                    table_displacement=table_displacement,
                    index_register_offset=index_key[0],
                    index_bits=index_bits,
                    entry_size=entry_size,
                    endness=entry_expr.end,
                    signed_entries=signed_entries,
                    target_displacement=target_displacement,
                    static_base_addr=static_base_addr,
                )

    return None


def _vex_direct_jump_table(
    vex, *, allow_full_width_index: bool = False
) -> StaticJumpTable | None:
    """
    Describe a bounded table whose entries are absolute jump destinations.

    The accepted VEX form is ``next = Load(base + index * entry_size + disp)``.
    Unlike relative tables, the loaded entry is itself the target address. The
    index must be a register. The caller separately requires a matching
    predecessor guard before it reads any finite number of table entries.
    """

    if vex.jumpkind != "Ijk_Boring":
        return None

    definitions = _vex_tmp_definitions(vex)
    entry_expr = _resolve_vex_expr(vex.next, definitions)
    if not isinstance(entry_expr, pyvex.expr.Load):
        return None

    entry_size = entry_expr.result_size(vex.tyenv) // 8
    if entry_size not in {1, 2, 4, 8}:
        return None

    address_terms = _vex_add_terms(entry_expr.addr, definitions)
    if address_terms is None:
        return None

    displacement = 0
    base_key = None
    index_key = None
    index_bits = None
    for term in address_terms:
        value = _vex_const_value(term, definitions)
        if value is not None:
            displacement += value
            continue

        register_key = _vex_get_key(term, definitions, vex)
        if register_key is not None and base_key is None:
            base_key = register_key
            continue

        term = _resolve_vex_expr(term, definitions)
        if not isinstance(term, pyvex.expr.Binop) or not term.op.startswith("Iop_Shl"):
            return None
        shift = _vex_const_value(term.args[1], definitions)
        candidate_index = _vex_index_key(
            term.args[0],
            definitions,
            vex,
            allow_full_width=allow_full_width_index,
        )
        if (
            candidate_index is None
            or shift is None
            or 1 << shift != entry_size
            or index_key is not None
        ):
            return None
        index_key = candidate_index
        index_bits = candidate_index[1]
    if base_key is None or index_key is None or index_bits is None:
        return None

    return StaticJumpTable(
        base_register_offset=base_key[0],
        base_bits=base_key[1],
        table_displacement=displacement & ((1 << base_key[1]) - 1),
        index_register_offset=index_key[0],
        index_bits=index_bits,
        entry_size=entry_size,
        endness=entry_expr.end,
        signed_entries=False,
        entries_are_relative=False,
    )


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


def _x86_pc_thunk_predecessors(
    project: Project,
    graph: CFGGraph,
    bounds: FunctionBounds,
    node,
    table: StaticJumpTable,
) -> tuple[CFGNode, ...]:
    """Return matching x86 PC-thunk calls whose fake return enters ``node``."""

    if project.arch.name != "X86" or project.arch.bits != 32:
        return ()
    if table.base_register_offset is None:
        return ()
    register_name = project.arch.register_names.get(table.base_register_offset)
    if register_name is None:
        return ()
    # GCC names 32-bit x86 thunks after the 16-bit register suffix: ``ebx``
    # is initialized by ``__x86.get_pc_thunk.bx``.
    thunk_register = register_name.removeprefix("e")
    expected_name = f"__x86.get_pc_thunk.{thunk_register}"

    predecessors: list[CFGNode] = []
    for predecessor in graph.predecessors(node):
        if not _node_is_materialized_cfg_node(predecessor):
            continue
        if not _node_intersects_bounds(predecessor, bounds):
            continue
        if predecessor.addr + predecessor.size != node.addr:
            continue
        vex = _node_vex(predecessor)
        if vex is None or vex.jumpkind != "Ijk_Call":
            continue
        definitions = _vex_tmp_definitions(vex)
        call_target = _vex_const_value(vex.next, definitions)
        if call_target is None:
            continue
        symbol = project.loader.find_symbol(call_target)
        if symbol is not None and symbol.name == expected_name:
            predecessors.append(predecessor)

    return tuple(predecessors)


def _x86_pc_thunk_base_addr(
    project: Project,
    graph: CFGGraph,
    bounds: FunctionBounds,
    node,
    table: StaticJumpTable,
) -> int | None:
    """Return an x86 PIC table base register value proven by a PC thunk call."""

    if project.arch.name != "X86" or project.arch.bits != 32:
        return None
    if table.base_register_offset is None:
        return None
    register_name = project.arch.register_names.get(table.base_register_offset)
    if register_name is None:
        return None
    # GCC names 32-bit x86 thunks after the 16-bit register suffix: ``ebx``
    # is initialized by ``__x86.get_pc_thunk.bx``.
    thunk_register = register_name.removeprefix("e")
    expected_name = f"__x86.get_pc_thunk.{thunk_register}"

    if _x86_pc_thunk_predecessors(project, graph, bounds, node, table):
        return node.addr

    values: set[int] = set()

    # Tables are often far from the prologue that initializes their PIC base.
    # GCC emits ``call __x86.get_pc_thunk.<reg>; add $offset, %reg``: the
    # fallthrough address is the thunk's result, so the following Add defines
    # the exact table base for every later use of the register in this function.
    for thunk_call in _iter_graph_bound_nodes(graph, bounds):
        vex = _node_vex(thunk_call)
        try:
            is_thunk_call = vex is not None and vex.jumpkind == "Ijk_Call"
        except AttributeError:
            continue
        if not is_thunk_call:
            continue
        definitions = _vex_tmp_definitions(vex)
        call_target = _vex_const_value(vex.next, definitions)
        symbol = project.loader.find_symbol(call_target) if call_target else None
        if symbol is None or symbol.name != expected_name:
            continue
        for successor in graph.successors(thunk_call):
            if not _node_is_materialized_cfg_node(successor):
                continue
            if successor.addr != thunk_call.addr + thunk_call.size:
                continue
            successor_vex = _node_vex(successor)
            if successor_vex is None:
                continue
            try:
                successor_defs = _vex_tmp_definitions(successor_vex)
            except (AttributeError, TypeError):
                continue
            for stmt in successor_vex.statements:
                if (
                    not isinstance(stmt, pyvex.stmt.Put)
                    or stmt.offset != table.base_register_offset
                ):
                    continue
                base = _vex_register_with_displacement(
                    stmt.data, successor_defs, successor_vex
                )
                if base is None or base[0] != (
                    table.base_register_offset,
                    table.base_bits,
                ):
                    continue
                values.add(successor.addr + base[1])

    return next(iter(values)) if len(values) == 1 else None


def _x86_pc_thunk_guarded_entry_count(
    project: Project,
    graph: CFGGraph,
    bounds: FunctionBounds,
    node,
    table: StaticJumpTable,
) -> int | None:
    """Return a table length proven by a guard immediately before an x86 thunk."""

    index_key = table.index_register_offset, table.index_bits
    upper_bounds: set[int] = set()
    for thunk_call in _x86_pc_thunk_predecessors(project, graph, bounds, node, table):
        for predecessor in graph.predecessors(thunk_call):
            if not _node_is_materialized_cfg_node(predecessor):
                continue
            if not _node_intersects_bounds(predecessor, bounds):
                continue
            vex = _node_vex(predecessor)
            if vex is None:
                continue
            upper_bound = _vex_guarded_index_upper_bound(
                vex, thunk_call.addr, index_key
            )
            if upper_bound is not None:
                upper_bounds.add(upper_bound)

    if len(upper_bounds) != 1:
        return None
    entry_count = next(iter(upper_bounds)) + 1
    return entry_count if entry_count <= MAX_STATIC_JUMPTABLE_ENTRIES else None


def _in_function_jump_table_entry_count(
    project: Project,
    bounds: FunctionBounds,
    table: StaticJumpTable,
    base_addr: int,
) -> int | None:
    """Estimate a table extent from contiguous in-function target entries."""

    table_addr = _jump_table_addr(base_addr, table)
    try:
        raw = project.loader.memory.load(
            table_addr, MAX_STATIC_JUMPTABLE_ENTRIES * table.entry_size
        )
    except Exception as exc:
        logger.debug(f"Custom CFG could not read jump table at {table_addr:#x}: {exc}")
        return None

    byteorder = "little" if table.endness == "Iend_LE" else "big"
    count = 0
    for offset in range(0, len(raw), table.entry_size):
        entry = int.from_bytes(
            raw[offset : offset + table.entry_size],
            byteorder=byteorder,
            signed=table.signed_entries,
        )
        target = _jump_table_target_addr(base_addr, table, entry)
        if not is_direct_target_valid(bounds, target):
            break
        count += 1
    return count if count >= 2 else None


def _jump_table_target_addr(base_addr: int, table: StaticJumpTable, entry: int) -> int:
    """Apply the architecture-width arithmetic used by a relative table jump."""

    if not table.entries_are_relative:
        return entry

    return (base_addr + table.target_displacement + entry) & (
        (1 << table.base_bits) - 1
    )


def _jump_table_addr(base_addr: int, table: StaticJumpTable) -> int:
    """Apply the table-address arithmetic in the architecture's address width."""

    return (base_addr + table.table_displacement) & ((1 << table.base_bits) - 1)


def _read_static_jump_table_targets(
    project: Project,
    table: StaticJumpTable,
    base_addr: int,
    entry_count: int,
) -> tuple[int, ...] | None:
    """Read targets from one VEX-proven table, or None when memory is unreadable."""

    if entry_count <= 0 or entry_count > MAX_STATIC_JUMPTABLE_ENTRIES:
        return None

    table_addr = _jump_table_addr(base_addr, table)
    try:
        raw = project.loader.memory.load(table_addr, entry_count * table.entry_size)
    except Exception as exc:
        logger.debug(f"Custom CFG could not read jump table at {table_addr:#x}: {exc}")
        return None

    byteorder = "little" if table.endness == "Iend_LE" else "big"
    targets: set[int] = set()
    for offset in range(0, len(raw), table.entry_size):
        entry = int.from_bytes(
            raw[offset : offset + table.entry_size],
            byteorder=byteorder,
            signed=table.signed_entries,
        )
        targets.add(_jump_table_target_addr(base_addr, table, entry))

    return tuple(sorted(targets))


def _vex_expr_key(expr, definitions: dict[int, Any]) -> tuple[Any, ...] | None:
    """Return a structural key for a local VEX expression.

    VEX temporary numbers are local to one lifted block, so matching branch
    predicates across adjacent blocks requires recursively replacing them with
    their definitions. Unsupported expressions deliberately return ``None``:
    callers use this only as a proof, never as a best-effort guess.
    """

    expr = _resolve_vex_expr(expr, definitions)
    if expr is None:
        return None
    if isinstance(expr, pyvex.expr.Const):
        value = expr.con.value
        return ("const", value) if isinstance(value, int) else None
    if isinstance(expr, pyvex.expr.Get):
        return ("get", expr.offset, expr.result_size(None))
    if isinstance(expr, pyvex.expr.Unop):
        argument = _vex_expr_key(expr.args[0], definitions)
        return ("unop", expr.op, argument) if argument is not None else None
    if isinstance(expr, pyvex.expr.Binop):
        arguments = tuple(_vex_expr_key(arg, definitions) for arg in expr.args)
        return ("binop", expr.op, *arguments) if all(arguments) else None
    if isinstance(expr, pyvex.expr.ITE):
        condition = _vex_expr_key(expr.cond, definitions)
        if_true = _vex_expr_key(expr.iftrue, definitions)
        if_false = _vex_expr_key(expr.iffalse, definitions)
        if condition is None or if_true is None or if_false is None:
            return None
        return ("ite", condition, if_true, if_false)
    if isinstance(expr, pyvex.expr.CCall):
        arguments = tuple(_vex_expr_key(arg, definitions) for arg in expr.args)
        if not all(arguments):
            return None
        return ("ccall", expr.cee.name, *arguments)
    return None


def _vex_scaled_register_target(
    vex,
) -> tuple[tuple[Any, ...], tuple[int, int], int, int] | None:
    """Describe one conditional ``base + (register << shift)`` VEX target."""

    definitions = _vex_tmp_definitions(vex)
    next_expr = _resolve_vex_expr(vex.next, definitions)
    if not isinstance(next_expr, pyvex.expr.ITE):
        return None

    target_expr = _resolve_vex_expr(next_expr.iftrue, definitions)
    if not isinstance(target_expr, pyvex.expr.Binop) or not target_expr.op.startswith(
        "Iop_Add"
    ):
        return None

    base_addr = None
    index_key = None
    shift = None
    for term in _vex_add_terms(target_expr, definitions) or ():
        value = _vex_const_value(term, definitions)
        if value is not None and base_addr is None:
            base_addr = value
            continue
        term = _resolve_vex_expr(term, definitions)
        if not isinstance(term, pyvex.expr.Binop) or not term.op.startswith("Iop_Shl"):
            return None
        candidate_shift = _vex_const_value(term.args[1], definitions)
        candidate_index = _vex_get_key(term.args[0], definitions, vex)
        if candidate_shift is None or candidate_index is None or index_key is not None:
            return None
        index_key = candidate_index
        shift = candidate_shift

    condition = _vex_expr_key(next_expr.cond, definitions)
    if condition is None or base_addr is None or index_key is None or shift is None:
        return None
    return condition, index_key, base_addr, shift


def _vex_conditionally_scaled_register(
    vex,
    register_key: tuple[int, int],
    condition_key: tuple[Any, ...],
) -> int | None:
    """Return a conditionally assigned register multiplier, if VEX proves one."""

    definitions = _vex_tmp_definitions(vex)
    register_expr = ("get", *register_key)
    for statement in reversed(vex.statements):
        if (
            not isinstance(statement, pyvex.stmt.Put)
            or statement.offset != register_key[0]
        ):
            continue
        assignment = _resolve_vex_expr(statement.data, definitions)
        if not isinstance(assignment, pyvex.expr.ITE):
            return None
        if _vex_expr_key(assignment.cond, definitions) != condition_key:
            return None

        unchanged = _vex_expr_key(assignment.iffalse, definitions)
        scaled_expr = _resolve_vex_expr(assignment.iftrue, definitions)
        if unchanged != register_expr or not isinstance(scaled_expr, pyvex.expr.Binop):
            return None
        if not scaled_expr.op.startswith("Iop_Add"):
            return None

        terms = _vex_add_terms(scaled_expr, definitions)
        if terms is None or len(terms) != 2:
            return None
        direct_reads = [
            _vex_expr_key(term, definitions) == register_expr for term in terms
        ]
        shifted_terms = [
            _resolve_vex_expr(term, definitions)
            for term, is_direct_read in zip(terms, direct_reads, strict=True)
            if not is_direct_read
        ]
        if direct_reads.count(True) != 1 or len(shifted_terms) != 1:
            return None

        shifted = shifted_terms[0]
        if not isinstance(shifted, pyvex.expr.Binop) or not shifted.op.startswith(
            "Iop_Shl"
        ):
            return None
        if _vex_expr_key(shifted.args[0], definitions) != register_expr:
            return None
        shift = _vex_const_value(shifted.args[1], definitions)
        return 1 + (1 << shift) if shift is not None else None
    return None


def _immediate_linear_predecessor(
    graph: CFGGraph, bounds: FunctionBounds, node
) -> CFGNode | None:
    """Return one sole fallthrough predecessor ending immediately before ``node``."""

    predecessors = [
        predecessor
        for predecessor in graph.predecessors(node)
        if _node_is_materialized_cfg_node(predecessor)
        and _node_intersects_bounds(predecessor, bounds)
        and predecessor.addr + predecessor.size == node.addr
        and tuple(graph.successors(predecessor)) == (node,)
    ]
    return predecessors[0] if len(predecessors) == 1 else None


def arithmetic_pc_dispatch_targets(
    project: Project, graph: CFGGraph, bounds: FunctionBounds, node
) -> tuple[int, ...] | None:
    """Return a proven target subset for one arithmetic computed-PC dispatch.

    This covers a VEX-level pattern where a conditional indirect branch writes
    ``base + (index << shift)`` to the program counter, and a sole linear
    predecessor conditionally scales that same index under the exact same VEX
    predicate. CFGFast may conservatively fan this out to every instruction
    boundary. We retain only the stride-aligned candidates it already found;
    the function never invents targets or applies architecture-specific
    mnemonic rules.
    """

    vex = _node_vex(node)
    if vex is None:
        return None
    dispatch = _vex_scaled_register_target(vex)
    if dispatch is None:
        return None
    condition_key, index_key, base_addr, shift = dispatch

    predecessor = _immediate_linear_predecessor(graph, bounds, node)
    # A neutral instruction may separate the scale and computed branch. Walk
    # only unique linear predecessors, so no unproven control-flow path is
    # included in the proof.
    for _ in range(3):
        if predecessor is None:
            return None
        predecessor_vex = _node_vex(predecessor)
        if predecessor_vex is not None:
            multiplier = _vex_conditionally_scaled_register(
                predecessor_vex, index_key, condition_key
            )
            if multiplier is None:
                # CFGFast may group the scale with the instruction that sets
                # condition flags, allowing VEX to simplify its predicate.
                # Re-lift only the last instruction to compare the preserved
                # condition-code form used by the computed-PC branch.
                last_insn = DecodedNode.from_node(predecessor).last
                if last_insn is not None:
                    single_insn_vex = lift_instruction_vex(project, last_insn)
                    if single_insn_vex is not None:
                        multiplier = _vex_conditionally_scaled_register(
                            single_insn_vex, index_key, condition_key
                        )
            if multiplier is not None:
                stride = (1 << shift) * multiplier
                break
        predecessor = _immediate_linear_predecessor(graph, bounds, predecessor)
    else:
        return None

    direct_exit_targets = {
        statement.dst.value
        for statement in vex.statements
        if isinstance(statement, pyvex.stmt.Exit)
        and isinstance(getattr(statement.dst, "value", None), int)
    }
    candidates = {
        successor.addr
        for successor in graph.successors(node)
        if _node_is_materialized_cfg_node(successor)
        and _node_intersects_bounds(successor, bounds)
        and successor.addr not in direct_exit_targets
    }
    if base_addr not in candidates:
        return None
    upper_bound = max(candidates)
    expected_targets = tuple(range(base_addr, upper_bound + 1, stride))
    if not expected_targets or not set(expected_targets).issubset(candidates):
        return None
    return expected_targets


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

    insns = decoded.insns
    if not insns:
        return ()

    transfer_index = control_transfer_index(
        node.block.arch.name, list(insns), strict=False
    )
    if transfer_index is None:
        return ()

    transfer = InsnSemantics(insns[transfer_index])
    if not transfer.is_jump():
        return ()

    target = transfer.direct_target()
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

    insns = decoded.insns
    if not insns:
        return None

    # On delayed-branch architectures, the final decoded instruction is the
    # delay slot. Classify the branch itself, but retain the final instruction
    # for the architectural fall-through address after that delay slot.
    try:
        arch_name = node.block.arch.name
    except AttributeError:
        arch_name = ""
    transfer_index = (
        control_transfer_index(arch_name, list(insns), strict=False)
        if arch_has_delay_slot(arch_name)
        else len(insns) - 1
    )
    if transfer_index is None:
        return None
    transfer_insn = insns[transfer_index]
    final_insn = insns[-1]

    try:
        vex = node.block.vex
    except Exception:
        vex = None

    transfer = InsnSemantics(transfer_insn)
    if not transfer.is_jump():
        return None
    # Calls may be members of Capstone's generic jump group. They have their
    # own call/fake-return edge semantics and must not be checked as branches.
    if transfer.is_call() or (vex is not None and vex.jumpkind == "Ijk_Call"):
        return None

    # Some instruction encodings, including RISC-V ``c.jr ra``, are exposed
    # by Capstone as generic jumps rather than returns. Re-lift just the
    # terminator to avoid treating an old CFG node's stale fallthrough as a
    # successor required by the recovered block.
    terminator_vex = lift_instruction_vex(project, transfer_insn)
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
            if ins_addr not in {transfer_insn.address, final_insn.address}:
                continue
            target = getattr(stmt.dst, "value", None)
            if isinstance(target, int) and target not in exit_targets:
                exit_targets.append(target)

    # A one-instruction VEX lift on a delay-slot ISA can omit the branch Exit
    # entirely. Preserve only an explicit Capstone condition operand here: a
    # one-target branch can be unconditional and must not gain a fallthrough.
    is_conditional = bool(exit_targets) or (
        arch_has_delay_slot(arch_name) and transfer.has_explicit_branch_condition()
    )
    if not is_conditional and transfer.is_conditional_jump():
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
                if ins_addr in {transfer_insn.address, final_insn.address}
                if isinstance(target := getattr(stmt.dst, "value", None), int)
            ]
            is_conditional = bool(exit_targets)

    vex_branch_targets = list(exit_targets)
    if (
        not is_conditional
        and vex_has_control_flow
        and target_vex is not None
        and isinstance(getattr(target_vex, "next", None), pyvex.expr.Const)
        and isinstance(target_vex.next.con.value, int)
    ):
        # An unconditional branch is represented by VEX's default successor,
        # not an Exit statement. This matters for architectures whose
        # Capstone operands retain a PC-relative displacement.
        vex_branch_targets.append(target_vex.next.con.value)

    direct_target = _resolve_direct_branch_target(
        project,
        bounds,
        transfer.direct_target(),
        vex_branch_targets,
    )
    if direct_target is None:
        return None

    expected: list[JumpSuccessorExpectation] = []
    kind: Literal["conditional", "direct"]
    fallthrough_addr = final_insn.address + final_insn.size
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
