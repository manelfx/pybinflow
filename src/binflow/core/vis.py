
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Callable
import itertools

from angr.analyses.cfg import CFGBase
from pydantic import BaseModel


class VisError(Exception):
    pass


class Node:
    def __init__(self, seq: int, obj: Any, cfg: CFGBase) -> None:
        self.obj = obj
        self.cfg = cfg
        self.seq = seq
        self.content = {}
        self.style = None
        self.fillcolor = None
        self.color = None
        self.width = None
        self.url = None
        self.tooltip = None

    def __eq__(self, other: 'Node') -> bool:
        return self.obj.__eq__(other.obj) and self.seq == other.seq

    def __hash__(self) -> int:
        return self.obj.__hash__()


class Edge:
    def __init__(self, src: Node, dst: Node, meta: Dict[str, Any] = {}, color: Optional[str] = None, label: Optional[str] = None, style: Optional[str] = None, width: Optional[float] = None, weight: Optional[float] = None) -> None:
        self.src = src
        self.dst = dst
        self.meta = meta
        self.color = color
        self.label = label
        self.style = style
        self.width = width
        self.weight = weight

    def __eq__(self, other: 'Edge') -> bool:
        return self.src == other.src and self.dst == other.dst

    #def __hash__(self) -> int:
    #    return hash(self.src, self.dst)


class Annotator(ABC, BaseModel):
    """Base class for all annotators."""
    pass


class NodeAnnotator(Annotator):
    @abstractmethod
    def annotate_node(self, node: Node) -> None:
        pass


class EdgeAnnotator(Annotator):
    @abstractmethod
    def annotate_edge(self, edge: Edge) -> None:
        pass


class ContentAnnotator(Annotator):
    name: str
    column: str

    @abstractmethod
    def annotate_content(self, node: Node, content: Dict[str, Any]) -> None:
        pass


class Content(ABC, BaseModel):
    name: str
    columns: List[str]
    annotators: List[Any] = []

    def append_column(self, column: str) -> None:
        if column not in self.columns:
            self.columns.append(column)

    def add_annotator(self, obj: ContentAnnotator) -> None:
        self.append_column(obj.column)
        self.annotators.append(obj)

    @abstractmethod
    def gen_render(self, node: Node):
        pass

    def render(self, n: Node) -> None:
        self.gen_render(n)
        for an in self.annotators:
            if self.name in n.content:
                an.annotate_content(n, n.content[self.name])


class Graph:
    def __init__(self, cfg: CFGBase, nodes: List[Node] = None, edges: List[Edge] = None) -> None:
        self.cfg = cfg
        self.obj = cfg.graph
        self.nodes = nodes if nodes else set()
        self.edges = edges if edges else []
        self.seqctr = itertools.count()
        self.seqmap = {}

    def add_node(self, node: Node) -> None:
        self.nodes.add(node)

    def add_edge(self, edge: Edge) -> None:
        self.edges.append(edge)

    def remove_node(self, node: Node) -> None:
        self.nodes.remove(node)
        self.edges = list(filter(lambda edge: edge.src != node and edge.dst != node, self.edges))

    def remove_edge(self, edge: Edge) -> None:
        self.edges.remove(edge)

    def filter_nodes(self, node_filter: Callable[[Node], bool]) -> None:
        new_graph = self.filtered_view(node_filter)
        self.nodes = new_graph.nodes
        self.edges = new_graph.edges

    def filtered_view(self, node_filter: Callable[[Node], bool]) -> 'Graph':
        nodes = list(filter(lambda _: node_filter(_), self.nodes))
        edges = list(filter(lambda edge: node_filter(edge.src) and node_filter(edge.dst), self.edges))
        return Graph(self.cfg, nodes, edges)


class Source(BaseModel, ABC):
    @abstractmethod
    def parse(self, cfg: CFGBase) -> Graph:
        pass


class Transformer(BaseModel, ABC):
    @abstractmethod
    def transform(self, graph: Graph) -> None:
        pass


class VisPipeLine:
    def __init__(self) -> None:
        self.content = {}
        self.node_annotators = []
        self.edge_annotators = []
        self.transformers = []
        self.graph = None

    def set_source(self, obj: Source) -> None:
        self.source = obj
        self.graph = None  # Initialize graph here

    def add_content(self, obj: Content) -> None:
        self.content[obj.name] = obj

    def add_node_annotator(self, obj: NodeAnnotator) -> None:
        self.node_annotators.append(obj)

    def add_edge_annotator(self, obj: EdgeAnnotator) -> None:
        self.edge_annotators.append(obj)

    def add_content_annotator(self, obj: ContentAnnotator) -> None:
        cname = obj.name
        if cname not in self.content:
            raise VisError("Content '%s' not found, required by annotator '%s'" % (cname, type(obj)))
        self.content[cname].add_annotator(obj)

    def add_transformer(self, obj: Transformer) -> None:
        self.transformers.append(obj)

    def set_input(self, cfg: CFGBase) -> None:
        self.graph = self.source.parse(cfg)

    def preprocess(self, cfg: CFGBase) -> None:
        self.set_input(cfg)
        for t in self.transformers:
            t.transform(self.graph)

    def process(self) -> Graph:
        graph = self.graph

        for n in graph.nodes:
            for c in self.content.values():
                c.render(n)
            for na in self.node_annotators:
                na.annotate_node(n)

        for e in graph.edges:
            for ea in self.edge_annotators:
                ea.annotate_edge(e)

        return graph


class Output(BaseModel, ABC):
    @abstractmethod
    def generate(self, graph: Graph) -> str:
        pass


class Vis:
    def __init__(self) -> None:
        self.pipeline = VisPipeLine()

    def preprocess(self, cfg: CFGBase) -> None:
        self.pipeline.preprocess(cfg)

    def process(self, cfg: CFGBase) -> str:
        self.preprocess(cfg)
        graph = self.pipeline.process()
        return self.output.generate(graph)

    def set_source(self, source: Source) -> None:
        self.pipeline.set_source(source)

    def add_content(self, obj: Content) -> None:
        self.pipeline.add_content(obj)

    def add_node_annotator(self, obj: NodeAnnotator) -> None:
        self.pipeline.add_node_annotator(obj)

    def add_edge_annotator(self, obj: EdgeAnnotator) -> None:
        self.pipeline.add_edge_annotator(obj)

    def add_content_annotator(self, obj: ContentAnnotator) -> None:
        self.pipeline.add_content_annotator(obj)

    def add_transformer(self, obj: Transformer) -> None:
        self.pipeline.add_transformer(obj)

    def set_output(self, obj: Output) -> None:
        self.output = obj

######

class XVis(BaseModel):

    source: Source
    output: Output

    transformers: List[Transformer]
    contents: List[Content]
    annotators: List[Annotator]

    def _split_annotators(self) -> None:

        # split annotators by type
        self.node_annotators = []
        self.edge_annotators = []
        self.content_annotators = []

        for annotator in self.annotators:
            if isinstance(annotator, NodeAnnotator):
                self.node_annotators.append(annotator)
            elif isinstance(annotator, EdgeAnnotator):
                self.edge_annotators.append(annotator)
            elif isinstance(annotator, ContentAnnotator):
                # FIXME! -- this should be a dictionary
                self.content_annotators.append(annotator)
            else:
                VisError(f"Unexpected annotator of type {type(annotator)}")

    def preprocess(self, cfg: CFGBase) -> None:

        # parse input graph
        self.graph = self.source.parse(cfg)

        # apply graph transformations
        for t in self.transformers:
            t.transform(self.graph)

        # split annotators by type
        self._split_annotators()

    def process(self, cfg: CFGBase) -> str:
        self.preprocess(cfg)

        for n in self.graph.nodes:
            for c in self.content.values():
                c.render(n)
            for na in self.node_annotators:
                na.annotate_node(n)

        for e in self.graph.edges:
            for ea in self.edge_annotators:
                ea.annotate_edge(e)

        return self.output.generate(self.graph)
