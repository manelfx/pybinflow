"""Read-only executable-range discovery used to audit unresolved dispatches."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Mapping

from angr import Project
import networkx as nx

from bingraph.cfg.jumps import static_jump_target_rejection_reason
from bingraph.cfg.models import BlockSpec, FunctionBounds
from bingraph.cfg.decode import decode_bounded_block


@dataclass(frozen=True)
class ExecutableSweepAudit:
    """Candidate blocks found without changing the extracted CFG."""

    candidate_blocks: int
    candidate_instructions: int
    candidate_components: int
    decode_failures: int
    non_executable_bytes: int


@dataclass(frozen=True)
class ExecutableSweep:
    """Closed direct-flow components discovered outside extracted reachability."""

    blocks: Mapping[int, BlockSpec]
    reachable_addrs: frozenset[int]
    disconnected_addrs: frozenset[int]
    audit: ExecutableSweepAudit


@dataclass(frozen=True)
class ReconnectingComponents:
    """Disconnected components safe to expose behind one unknown dispatcher."""

    blocks: Mapping[int, BlockSpec]
    roots: frozenset[int]
    component_count: int


def _covering_end(blocks: Mapping[int, BlockSpec], addr: int) -> int | None:
    """Return the end of a recovered block covering ``addr``, if any."""

    for start, block in blocks.items():
        end = start + block.size
        if start <= addr < end:
            return end
    return None


def _direct_flow_graph(blocks: Mapping[int, BlockSpec]) -> nx.DiGraph:
    """Build the decoded direct-flow graph for a recovered block mapping."""

    graph = nx.DiGraph()
    graph.add_nodes_from(blocks)
    for addr, block in blocks.items():
        for target in (*block.direct_targets, block.fallthrough_addr):
            if target in blocks:
                graph.add_edge(addr, target)
    return graph


def _reachable_addrs(
    blocks: Mapping[int, BlockSpec], bounds: FunctionBounds
) -> set[int]:
    """Return block starts reachable from the function entry by direct flow."""

    graph = _direct_flow_graph(blocks)
    if bounds.addr not in graph:
        return set()
    return nx.descendants(graph, bounds.addr) | {bounds.addr}


def _disconnected_component_count(
    blocks: Mapping[int, BlockSpec], disconnected_addrs: set[int]
) -> int:
    """Count direct-flow components outside the function entry's reachability."""

    if not disconnected_addrs:
        return 0
    return nx.number_weakly_connected_components(
        _direct_flow_graph(blocks).subgraph(disconnected_addrs)
    )


def _recover_direct_closure(
    project: Project,
    bounds: FunctionBounds,
    blocks: dict[int, BlockSpec],
    leaders: set[int],
) -> int:
    """Re-decode blocks until every decoded direct target is a block leader.

    The initial byte sweep intentionally follows sequential bytes and therefore
    can discover a branch target in the middle of one of its provisional
    blocks. Before treating scanned code as a candidate CFG component, apply
    the same leader invariant as normal extraction: split that block and
    continue until direct-flow edges land on exact block starts.
    """

    pending: deque[int] = deque()
    pending_addrs: set[int] = set()
    decode_failures = 0

    def queue(addr: int) -> None:
        if addr not in pending_addrs:
            pending.append(addr)
            pending_addrs.add(addr)

    def add_leader(addr: int) -> None:
        if not bounds.addr <= addr < bounds.end_addr:
            return
        if addr not in leaders:
            leaders.add(addr)
            for start, block in tuple(blocks.items()):
                if start < addr < start + block.size:
                    del blocks[start]
                    queue(start)
        if addr not in blocks:
            queue(addr)

    for block in tuple(blocks.values()):
        for target in (*block.direct_targets, block.fallthrough_addr):
            if target is not None:
                add_leader(target)

    while pending:
        addr = pending.popleft()
        pending_addrs.remove(addr)
        block = decode_bounded_block(
            project,
            bounds,
            addr,
            leaders - {addr},
            preserve_conditional_return_fallthrough=True,
        )
        if block is None or block.size <= 0:
            decode_failures += 1
            continue
        blocks[addr] = block
        for target in (*block.direct_targets, block.fallthrough_addr):
            if target is not None:
                add_leader(target)

    return decode_failures


def recover_executable_components(
    project: Project,
    bounds: FunctionBounds,
    recovered_blocks: Mapping[int, BlockSpec],
) -> ExecutableSweep:
    """Recover and validate disconnected executable components without a CFG.

    The sweep starts from unclaimed executable bytes, then closes the decoded
    direct flow around each recovered target. It deliberately returns data only:
    callers decide whether unresolved indirect dispatches justify materializing
    these speculative components.
    """

    blocks = dict(recovered_blocks)
    leaders = set(blocks)
    cursor = bounds.addr
    decode_failures = 0
    non_executable_bytes = 0

    while cursor < bounds.end_addr:
        covered_end = _covering_end(blocks, cursor)
        if covered_end is not None:
            cursor = covered_end
            continue

        if static_jump_target_rejection_reason(project, cursor) is not None:
            non_executable_bytes += 1
            cursor += 1
            continue

        block = decode_bounded_block(
            project,
            bounds,
            cursor,
            leaders,
            preserve_conditional_return_fallthrough=True,
        )
        if block is None or block.size <= 0:
            decode_failures += 1
            cursor += 1
            continue

        blocks[block.addr] = block
        leaders.add(block.addr)
        cursor = block.addr + block.size

    decode_failures += _recover_direct_closure(project, bounds, blocks, leaders)
    reachable_addrs = _reachable_addrs(blocks, bounds)
    disconnected_addrs = set(blocks) - reachable_addrs

    audit = ExecutableSweepAudit(
        candidate_blocks=len(disconnected_addrs),
        candidate_instructions=sum(
            len(blocks[addr].instruction_addrs) for addr in disconnected_addrs
        ),
        candidate_components=_disconnected_component_count(blocks, disconnected_addrs),
        decode_failures=decode_failures,
        non_executable_bytes=non_executable_bytes,
    )
    return ExecutableSweep(
        blocks,
        frozenset(reachable_addrs),
        frozenset(disconnected_addrs),
        audit,
    )


def select_reconnecting_components(
    sweep: ExecutableSweep,
    recovered_addrs: set[int],
) -> ReconnectingComponents:
    """Select direct-flow components that rejoin known function code.

    Executable bytes alone do not prove an indirect-jump target. A component
    becomes a candidate only when its decoded direct flow reaches a block from
    the original extraction and it contains no additional unresolved indirect
    branch. The latter would need its own target evidence, not inherited trust
    from the outer dispatcher.
    """

    graph = _direct_flow_graph(sweep.blocks)
    disconnected_graph = graph.subgraph(sweep.disconnected_addrs)
    # Preserve leader-closed revisions of entry-reachable blocks. A selected
    # component can branch into the middle of an original block, so retaining
    # the pre-sweep block would undo the exact-target split.
    selected_blocks = {addr: sweep.blocks[addr] for addr in sweep.reachable_addrs}
    roots: set[int] = set()
    component_count = 0

    for component in nx.weakly_connected_components(disconnected_graph):
        has_rejoin = any(
            target in recovered_addrs
            for addr in component
            for target in graph.successors(addr)
        )
        has_nested_unresolved = any(
            sweep.blocks[addr].jumpkind == "Ijk_Boring"
            and not sweep.blocks[addr].direct_targets
            and sweep.blocks[addr].fallthrough_addr is None
            for addr in component
        )
        if not has_rejoin or has_nested_unresolved:
            continue

        component_count += 1
        selected_blocks.update((addr, sweep.blocks[addr]) for addr in component)
        roots.update(
            addr for addr in component if disconnected_graph.in_degree(addr) == 0
        )

    return ReconnectingComponents(selected_blocks, frozenset(roots), component_count)


def audit_executable_range(
    project: Project,
    bounds: FunctionBounds,
    recovered_blocks: Mapping[int, BlockSpec],
) -> ExecutableSweepAudit:
    """Return read-only statistics for disconnected executable components."""

    return recover_executable_components(project, bounds, recovered_blocks).audit
