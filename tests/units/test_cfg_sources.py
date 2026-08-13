"""Unit tests for selecting CFG nodes that are relevant to one function."""

from dataclasses import dataclass

import networkx as nx

from bingraph.core.sources import _select_cfg_nodes, _selected_cfg_edges


@dataclass(frozen=True)
class _Node:
    """Minimal CFG node shape used to exercise source selection."""

    addr: int
    function_address: int | None
    is_simprocedure: bool = False
    simprocedure_name: str | None = None


def test_select_cfg_nodes_shows_non_call_exits_by_default() -> None:
    """Keep direct external branches while omitting optional call leaves."""

    graph = nx.DiGraph()
    function_node = _Node(0x1000, 0x1000)
    external_call = _Node(0x2000, 0x2000)
    fake_return = _Node(0x3000, 0x3000)
    direct_branch = _Node(0x3500, 0x3500)
    simprocedure = _Node(0x4000, None, is_simprocedure=True)
    syscall = _Node(0x4100, None, is_simprocedure=True)
    disconnected_simprocedure = _Node(0x5000, None, is_simprocedure=True)
    external_child = _Node(0x6000, 0x6000)

    graph.add_edge(function_node, external_call, jumpkind="Ijk_Call")
    graph.add_edge(function_node, fake_return, jumpkind="Ijk_FakeRet")
    graph.add_edge(function_node, direct_branch, jumpkind="Ijk_Boring")
    graph.add_edge(function_node, simprocedure, jumpkind="Ijk_Call")
    graph.add_edge(function_node, syscall, jumpkind="Ijk_Sys_syscall")
    graph.add_edge(fake_return, external_child, jumpkind="Ijk_Boring")
    graph.add_node(disconnected_simprocedure)

    assert _select_cfg_nodes(graph, 0x1000) == {
        function_node,
        fake_return,
        direct_branch,
        syscall,
    }


def test_select_cfg_nodes_respects_the_exit_display_policy() -> None:
    """Select external call and branch leaves according to the exit policy."""

    graph = nx.DiGraph()
    function_node = _Node(0x1000, 0x1000)
    # Extractor-created leaves carry the current function owner, so selection
    # must classify them by their incoming edge instead of their ownership.
    callee = _Node(0x4000, 0x1000, is_simprocedure=True)
    direct_branch = _Node(0x5000, 0x1000, is_simprocedure=True)
    graph.add_edge(function_node, callee, jumpkind="Ijk_Call")
    graph.add_edge(function_node, direct_branch, jumpkind="Ijk_Boring")

    assert _select_cfg_nodes(graph, 0x1000, "never") == {function_node}
    assert _select_cfg_nodes(graph, 0x1000) == {function_node, direct_branch}
    assert _select_cfg_nodes(graph, 0x1000, "always") == {
        function_node,
        callee,
        direct_branch,
    }


def test_select_cfg_nodes_keeps_structural_indirect_dispatchers() -> None:
    """Keep a dispatcher that connects selected blocks under every policy."""

    graph = nx.DiGraph()
    function_node = _Node(0x1000, 0x1000)
    target = _Node(0x1100, 0x1000)
    dispatcher = _Node(
        0xFFFFFFFFFFFFFFF0,
        0x1000,
        is_simprocedure=True,
        simprocedure_name="UnresolvableJumpTarget",
    )
    graph.add_edge(function_node, dispatcher, jumpkind="Ijk_Boring")
    graph.add_edge(dispatcher, target, jumpkind="Ijk_Boring")

    expected = {function_node, target, dispatcher}
    assert _select_cfg_nodes(graph, 0x1000, "never") == expected
    assert _select_cfg_nodes(graph, 0x1000, "jump") == expected
    assert _select_cfg_nodes(graph, 0x1000, "always") == expected


def test_select_cfg_nodes_prefers_a_simp_leaf_for_foreign_target_code() -> None:
    """Render all target edges through one SIMP leaf when addresses collide."""

    graph = nx.DiGraph()
    function_node = _Node(0x1000, 0x1000)
    tail_branch = _Node(0x1010, 0x1000)
    foreign_target = _Node(0x2000, 0x2000)
    external_leaf = _Node(0x2000, 0x1000, is_simprocedure=True)
    graph.add_edge(function_node, foreign_target, jumpkind="Ijk_Call")
    graph.add_edge(tail_branch, external_leaf, jumpkind="Ijk_Boring")

    selected = _select_cfg_nodes(graph, 0x1000, "always")
    assert selected == {function_node, tail_branch, external_leaf}
    assert list(_selected_cfg_edges(graph, selected)) == [
        (function_node, external_leaf, {"jumpkind": "Ijk_Call"}),
        (tail_branch, external_leaf, {"jumpkind": "Ijk_Boring"}),
    ]


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
