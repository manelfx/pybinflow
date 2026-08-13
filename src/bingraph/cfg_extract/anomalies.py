"""Structural validation for CFGs built by the independent extractor."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Mapping

from angr.knowledge_plugins.cfg import CFGNode

from bingraph.cfg.graph import CFGGraph, node_range_end, ranges_overlap
from bingraph.cfg.models import BlockSpec, FunctionBounds


@dataclass(frozen=True)
class ExtractedCFGAnomaly:
    """One invariant violation in a graph wholly constructed by extraction."""

    kind: str
    addr: int
    message: str


def _normal_nodes(graph: CFGGraph, func_addr: int) -> tuple[CFGNode, ...]:
    """Return this function's materialized nodes in stable address order."""

    return tuple(
        sorted(
            (
                node
                for node in graph.nodes()
                if not node.is_simprocedure and node.function_address == func_addr
            ),
            key=lambda node: node.addr,
        )
    )


def _reachable_nodes(graph: CFGGraph, entry: CFGNode) -> set[CFGNode]:
    """Return every graph node reachable from the extracted entry block."""

    reachable: set[CFGNode] = set()
    pending = deque([entry])
    while pending:
        node = pending.popleft()
        if node in reachable:
            continue
        reachable.add(node)
        pending.extend(graph.successors(node))
    return reachable


def find_extracted_cfg_anomalies(
    graph: CFGGraph,
    bounds: FunctionBounds,
    func_addr: int,
    blocks: Mapping[int, BlockSpec],
) -> tuple[ExtractedCFGAnomaly, ...]:
    """Validate invariants that the extractor itself promises to establish.

    Unresolved indirect transfers are not anomalies: they remain explicit
    leaves until a table resolver proves their targets. The checks below cover
    only errors the bounded leader worklist should never leave behind.
    """

    anomalies: list[ExtractedCFGAnomaly] = []
    nodes = _normal_nodes(graph, func_addr)
    node_by_addr = {node.addr: node for node in nodes}
    instruction_owners: dict[int, int] = {}

    for index, node in enumerate(nodes):
        for other in nodes[index + 1 :]:
            if other.addr >= node_range_end(node):
                break
            if ranges_overlap(
                node.addr, node_range_end(node), other.addr, node_range_end(other)
            ):
                anomalies.append(
                    ExtractedCFGAnomaly(
                        "overlapping_blocks",
                        node.addr,
                        f"Extracted blocks at {node.addr:#x} and {other.addr:#x} overlap",
                    )
                )

        for insn_addr in node.instruction_addrs:
            previous = instruction_owners.setdefault(insn_addr, node.addr)
            if previous != node.addr:
                anomalies.append(
                    ExtractedCFGAnomaly(
                        "duplicate_instruction",
                        insn_addr,
                        f"Instruction {insn_addr:#x} appears in extracted blocks "
                        f"{previous:#x} and {node.addr:#x}",
                    )
                )

        block = blocks.get(node.addr)
        if block is None:
            anomalies.append(
                ExtractedCFGAnomaly(
                    "missing_block_spec",
                    node.addr,
                    f"Extracted node {node.addr:#x} has no recovered block specification",
                )
            )
            continue

        successor_addrs = {successor.addr for successor in graph.successors(node)}
        for target in block.direct_targets:
            if target not in successor_addrs:
                anomalies.append(
                    ExtractedCFGAnomaly(
                        "missing_direct_edge",
                        node.addr,
                        f"Extracted block {node.addr:#x} is missing direct edge to "
                        f"{target:#x}",
                    )
                )
            covering = next(
                (
                    candidate
                    for candidate in nodes
                    if candidate.addr < target < node_range_end(candidate)
                ),
                None,
            )
            if covering is not None:
                anomalies.append(
                    ExtractedCFGAnomaly(
                        "target_inside_block",
                        node.addr,
                        f"Extracted target {target:#x} from {node.addr:#x} lands "
                        f"inside block {covering.addr:#x}",
                    )
                )

        fallthrough = block.fallthrough_addr
        if fallthrough is not None and fallthrough in node_by_addr:
            if fallthrough not in successor_addrs:
                anomalies.append(
                    ExtractedCFGAnomaly(
                        "missing_fallthrough_edge",
                        node.addr,
                        f"Extracted block {node.addr:#x} is missing fallthrough edge "
                        f"to {fallthrough:#x}",
                    )
                )

    entry = node_by_addr.get(func_addr)
    if entry is None:
        anomalies.append(
            ExtractedCFGAnomaly(
                "missing_entry",
                func_addr,
                f"Extracted CFG has no entry at {func_addr:#x}",
            )
        )
    else:
        reachable = _reachable_nodes(graph, entry)
        for node in nodes:
            if node not in reachable:
                anomalies.append(
                    ExtractedCFGAnomaly(
                        "unreachable_block",
                        node.addr,
                        f"Extracted block {node.addr:#x} is unreachable from {func_addr:#x}",
                    )
                )

    return tuple(anomalies)
