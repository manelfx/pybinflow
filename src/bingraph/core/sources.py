
from angr.analyses.cfg import CFGBase
from .vis import Source, VisError, Node, Edge, Graph


class CFGSource(Source):
    func_addr: int | None = None

    def parse(self, cfg: CFGBase) -> Graph:

        lookup = {}

        obj = cfg.graph
        graph = Graph(cfg)

        # traverse angr nodes
        for n in obj.nodes():
            # consider only nodes of given function (if provided) and simbolic procedures
            if not n.is_simprocedure and (self.func_addr and n.function_address != self.func_addr):
                continue

            if n in lookup:
                raise VisError("Duplicate node %s" % str(n))

            # add node to graph
            wn = Node(n)
            lookup[n] = wn
            graph.add_node(wn)

        # traverse angr edges
        for src, dst, data in obj.edges(data=True):
            # ignore all edges coming from / going to outside this function, if provided
            if self.func_addr and (src not in lookup or dst not in lookup):
                continue

            # add edge to graph
            graph.add_edge(Edge(lookup[src], lookup[dst], data))

        return graph
