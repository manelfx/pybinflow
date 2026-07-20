from dataclasses import dataclass
from typing import Any

from angr.analyses.cfg import CFGBase

from .vis import Edge, Graph, Node, Source, VisError


_EXTERNAL_FRONTIER_JUMPKINDS = frozenset({"Ijk_Boring", "Ijk_FakeRet"})


def _is_path_terminator(node: Any) -> bool:
    """Return whether ``node`` is angr's artificial path-end marker."""

    return node.is_simprocedure and node.simprocedure_name == "PathTerminator"


def _select_cfg_nodes(cfg_graph: Any, func_addr: int | None) -> set[Any]:
    """Choose function nodes and the one-hop exits needed to explain them.

    CFGFast may attach targets that belong to another function or a synthetic
    procedure. Keep only nodes owned by the requested function, plus direct
    semantic exits from those nodes. In particular, real external calls remain
    hidden while fake returns and direct branches can show their known exit.
    """

    if func_addr is None:
        return {node for node in cfg_graph if not _is_path_terminator(node)}

    selected = {
        node
        for node in cfg_graph
        if not _is_path_terminator(node) and node.function_address == func_addr
    }

    for source in tuple(selected):
        for destination in cfg_graph.successors(source):
            if _is_path_terminator(destination):
                continue

            edge_data = cfg_graph.get_edge_data(source, destination)
            jumpkind = edge_data.get("jumpkind")
            if destination.is_simprocedure or jumpkind in _EXTERNAL_FRONTIER_JUMPKINDS:
                selected.add(destination)

    return selected


@dataclass
class CFGSource(Source):
    func_addr: int | None = None

    def parse(self, cfg: CFGBase) -> Graph:
        """Convert the selected function and its direct exits to render nodes."""

        obj = cfg.graph
        graph = Graph(cfg)
        selected_nodes = _select_cfg_nodes(obj, self.func_addr)
        lookup = {}

        for n in selected_nodes:
            if n in lookup:
                raise VisError("Duplicate node %s" % str(n))

            wn = Node(n, obj)
            lookup[n] = wn
            graph.add_node(wn)

        for src, dst, data in obj.edges(data=True):
            if src not in lookup or dst not in lookup:
                continue

            graph.add_edge(Edge(lookup[src], lookup[dst], data))

        return graph
