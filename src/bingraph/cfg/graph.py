"""Graph and node primitives shared by custom CFG validation and repair."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any, Protocol, cast

from angr.analyses.cfg import CFGBase
from angr.knowledge_plugins.cfg import CFGNode


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

    def remove_edge(self, src: CFGNode, dst: CFGNode) -> None: ...

    def remove_node(self, node: CFGNode) -> None: ...


def cfg_graph(cfg: CFGBase) -> CFGGraph:
    """Return the CFG's public graph wrapper used by custom repair."""

    return cast(CFGGraph, cfg.graph)
