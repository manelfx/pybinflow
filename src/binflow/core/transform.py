
from typing import Callable, Any, Set
from angr import Project
from .vis import Transformer, Graph


class FilterNodes(Transformer):
    node_filter: Callable[[Any], bool]

    def transform(self, graph: Graph) -> None:
        graph.filter_nodes(self.node_filter)


class RemovePathTerminator(Transformer):
    def transform(self, graph: Graph) -> None:
        remove = []
        for n in graph.nodes:
            if hasattr(n.obj, 'is_simprocedure') and n.obj.is_simprocedure and n.obj.simprocedure_name == 'PathTerminator':
                remove.append(n)
        for r in remove:
            graph.remove_node(r)


class RemoveSimProcedures(Transformer):
    def transform(self, graph: Graph) -> None:
        remove = []
        for n in graph.nodes:
            if n.obj.is_simprocedure:
                remove.append(n)
                cs = []
                for e in graph.edges:
                    if e.dst == n:
                        cs.append(e.src)
                found = False
                for c in cs:
                    for e in graph.edges:
                        if e.src == c and e.dst != n:
                            found = True
                            break
                    if not found:
                        remove.append(c)
        for r in remove:
            graph.remove_node(r)


class RemoveImports(Transformer):
    def import_addrs(self, project: Project) -> Set[int]:
        eaddrs=[]
        for _ in project.loader.main_object.imports.values():
            if _.resolvedby is not None:
                eaddrs.append(_.value)
        return set(eaddrs)

    def transform(self, graph: Graph) -> None:
        remove = set()
        eaddrs = self.import_addrs(graph.cfg.project)
        for n in graph.nodes:
            if n.obj.addr in eaddrs:
                remove.add(n)
                cs = []
                for e in graph.edges:
                    if e.dst == n:
                        cs.append(e.src)
                found = False
                for c in cs:
                    for e in graph.edges:
                        if e.src == c and e.dst != n:
                            found = True
                            break
                    if not found:
                        remove.add(c)
        for r in remove:
            graph.remove_node(r)
