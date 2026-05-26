
from angr.analyses.cfg import CFGBase
from .vis import Source, VisError, Node, Edge, Graph


class CFGSource(Source):

    def parse(self, cfg: CFGBase) -> Graph:

        lookup = {}
        seq = 0

        obj = cfg.graph
        graph = Graph(cfg)

        for n in obj.nodes():
            if n not in lookup:
                wn = Node(seq, n, cfg)
                seq += 1
                lookup[n] = wn
                graph.add_node(wn)
            else:
                raise VisError("Duplicate node %s" % str(n))

        for src, dst, data in obj.edges(data=True):
            if src not in lookup or dst not in lookup:
                raise VisError("Missing nodes %s %s" % str(src), str(dst))
            wsrc = lookup[src]
            wdst = lookup[dst]
            graph.add_edge(Edge(wsrc, wdst, data))

        return graph


# NOTE: add call graph (CG) source view as well
