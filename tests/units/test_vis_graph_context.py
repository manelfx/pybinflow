"""Fast tests for graph ownership passed into visualization nodes."""

from types import SimpleNamespace
from typing import cast

from angr.knowledge_plugins.cfg import CFGNode

from bingraph.core.vis import Node


def _cfg_node(fallback_graph: object) -> CFGNode:
    """Build the minimal CFG-node shape consumed by ``Node``."""

    return cast(
        CFGNode,
        SimpleNamespace(
            block_id=0x1000,
            _cfg_model=SimpleNamespace(graph=fallback_graph),
        ),
    )


def test_node_graph_uses_explicit_render_context() -> None:
    """Prefer the graph selected by CFG source parsing over angr model state."""

    fallback_graph = object()
    render_graph = object()

    assert Node(_cfg_node(fallback_graph), render_graph).graph is render_graph


def test_node_graph_falls_back_to_angr_model_graph() -> None:
    """Keep ordinary angr CFG nodes compatible without an explicit graph."""

    fallback_graph = object()

    assert Node(_cfg_node(fallback_graph)).graph is fallback_graph
