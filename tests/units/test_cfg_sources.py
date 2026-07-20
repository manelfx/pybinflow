"""Unit tests for selecting CFG nodes that are relevant to one function."""

from dataclasses import dataclass

import networkx as nx

from bingraph.core.sources import _select_cfg_nodes


@dataclass(frozen=True)
class _Node:
    """Minimal CFG node shape used to exercise source selection."""

    addr: int
    function_address: int | None
    is_simprocedure: bool = False
    simprocedure_name: str | None = None


def test_select_cfg_nodes_keeps_only_connected_semantic_exits() -> None:
    """Render known function exits without rendering ordinary external calls."""

    graph = nx.DiGraph()
    function_node = _Node(0x1000, 0x1000)
    external_call = _Node(0x2000, 0x2000)
    fake_return = _Node(0x3000, 0x3000)
    direct_branch = _Node(0x3500, 0x3500)
    simprocedure = _Node(0x4000, None, is_simprocedure=True)
    disconnected_simprocedure = _Node(0x5000, None, is_simprocedure=True)
    external_child = _Node(0x6000, 0x6000)

    graph.add_edge(function_node, external_call, jumpkind="Ijk_Call")
    graph.add_edge(function_node, fake_return, jumpkind="Ijk_FakeRet")
    graph.add_edge(function_node, direct_branch, jumpkind="Ijk_Boring")
    graph.add_edge(function_node, simprocedure, jumpkind="Ijk_Call")
    graph.add_edge(fake_return, external_child, jumpkind="Ijk_Boring")
    graph.add_node(disconnected_simprocedure)

    assert _select_cfg_nodes(graph, 0x1000) == {
        function_node,
        fake_return,
        direct_branch,
        simprocedure,
    }


def test_select_cfg_nodes_excludes_path_terminators() -> None:
    """Never render angr's synthetic path-end marker."""

    graph = nx.DiGraph()
    function_node = _Node(0x1000, 0x1000)
    terminator = _Node(
        0x0,
        None,
        is_simprocedure=True,
        simprocedure_name="PathTerminator",
    )
    graph.add_edge(function_node, terminator, jumpkind="Ijk_Boring")

    assert _select_cfg_nodes(graph, 0x1000) == {function_node}
