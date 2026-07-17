"""CFG node construction and cleanup helpers for custom repair."""

from __future__ import annotations

from angr import Project
from angr.analyses.cfg import CFGBase
from angr.knowledge_plugins.cfg import CFGNode

from .graph import (
    CFGGraph,
    node_is_placeholder,
    node_is_simprocedure,
    remove_nodes,
)
from .models import BlockSpec, FunctionBounds


def prune_orphan_simprocedures(graph: CFGGraph) -> int:
    """Remove simprocedure nodes that no longer have incoming edges."""

    removed = 0
    while True:
        orphan_nodes = [
            node
            for node in list(graph.nodes())
            if node_is_simprocedure(node) and graph.in_degree(node) == 0
        ]
        if not orphan_nodes:
            return removed
        remove_nodes(graph, orphan_nodes)
        removed += len(orphan_nodes)


def prune_placeholders(graph: CFGGraph) -> int:
    """Remove temporary placeholder nodes and return how many were pruned."""

    placeholders = [node for node in list(graph.nodes()) if node_is_placeholder(node)]
    if placeholders:
        remove_nodes(graph, placeholders)
    return len(placeholders)


def make_cfg_node(
    seed_cfg: CFGBase, func_addr: int, bounds: FunctionBounds, block: BlockSpec
) -> CFGNode:
    """Instantiate one recovered node compatible with the render pipeline."""

    name = (
        bounds.name
        if block.addr == bounds.addr
        else f"{bounds.name}+0x{block.addr - bounds.addr:x}"
    )
    return CFGNode(
        block.addr,
        block.size,
        cfg=seed_cfg.model,
        function_address=func_addr,
        block_id=block.addr,
        instruction_addrs=block.instruction_addrs,
        name=name,
    )


def make_placeholder_node(seed_cfg: CFGBase, func_addr: int, addr: int) -> CFGNode:
    """Create a zero-sized placeholder node for a newly discovered entry."""

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
        if node_is_simprocedure(node) and node.addr == addr:
            return node
    return None


def _external_target_name(project: Project, addr: int) -> str:
    """Return the symbol-table name for an external target when available."""

    symbol = project.loader.find_symbol(addr)
    if symbol is None:
        return f"ExternalTarget_{addr:#x}"

    name = getattr(symbol, "name", None)
    return name if isinstance(name, str) and name else f"ExternalTarget_{addr:#x}"


def make_external_target_node(seed_cfg: CFGBase, func_addr: int, addr: int) -> CFGNode:
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


def ensure_external_target_node(
    seed_cfg: CFGBase,
    graph: CFGGraph,
    func_addr: int,
    addr: int,
) -> tuple[CFGNode, bool]:
    """Get or create one synthetic external-target leaf and report creation."""

    node = _find_external_target_node(graph, addr)
    if node is not None:
        return node, False

    node = make_external_target_node(seed_cfg, func_addr, addr)
    graph.add_node(node)
    return node, True
