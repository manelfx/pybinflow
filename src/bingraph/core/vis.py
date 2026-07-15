from abc import ABC, abstractmethod
from typing import Any, Callable

from angr.analyses.cfg import CFGBase
from angr.knowledge_plugins.cfg import CFGNode
from pydantic import BaseModel, model_validator
from pydot import Node as PydotNode, Edge as PydotEdge

from bingraph.helpers import time_it

class VisError(Exception):
    pass


class Node:
    def __init__(self, obj: CFGNode, graph: Any | None = None) -> None:
        """Wrap one CFG node with the graph selected for this render pass."""

        self.obj = obj
        # CFGFast nodes keep their original angr model. Custom repair mutates a
        # graph wrapper in place, so rendering must receive that live graph
        # explicitly instead of consulting process-global node registration.
        self._graph = graph
        self.pydot = PydotNode(self.seq)
        self.content = {}

    @property
    def graph(self):
        """Returns NetworkX graph."""
        return self._graph or self.obj._cfg_model.graph

    @property
    def project(self):
        """Returns static project info, no change between analyses."""
        return self.obj._cfg_model.project

    @property
    def kb(self):
        """Returns Angr Knowledge base info, analysis dependent."""
        return self.obj._cfg_model._cfg_manager._kb

    @property
    def seq(self):
        """Return a readable, stable DOT identifier for this CFG node."""

        block_id = self.obj.block_id
        if isinstance(block_id, int):
            return hex(block_id)
        return str(block_id)

    def __eq__(self, other: "Node") -> bool:
        return self.obj == other.obj  # and self.seq == other.seq

    def __hash__(self) -> int:
        return hash(self.obj)


class Edge:
    def __init__(
        self, src: Node, dst: Node, meta: dict[str, Any] | None = None
    ) -> None:
        self.src = src
        self.dst = dst
        self.pydot = PydotEdge(src.seq, dst.seq)
        self.meta = meta or {}

    def __eq__(self, other: "Edge") -> bool:
        return self.src == other.src and self.dst == other.dst

    def __hash__(self) -> int:
        return hash((self.src, self.dst))


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
    def annotate_content(self, node: Node, content: dict[str, Any]) -> None:
        pass


class Content(ABC, BaseModel):
    name: str
    columns: list[str]
    annotators: list[Any] = []

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
    def __init__(
        self, cfg: CFGBase, nodes: list[Node] = None, edges: list[Edge] = None
    ) -> None:
        self.cfg = cfg
        self.obj = cfg.graph
        self.nodes = nodes if nodes else set()
        self.edges = edges if edges else []

    def add_node(self, node: Node) -> None:
        self.nodes.add(node)

    def add_edge(self, edge: Edge) -> None:
        self.edges.append(edge)

    def remove_node(self, node: Node) -> None:
        self.nodes.remove(node)
        self.edges = list(
            filter(lambda edge: edge.src != node and edge.dst != node, self.edges)
        )

    def remove_edge(self, edge: Edge) -> None:
        self.edges.remove(edge)

    def filter_nodes(self, node_filter: Callable[[Node], bool]) -> None:
        new_graph = self.filtered_view(node_filter)
        self.nodes = new_graph.nodes
        self.edges = new_graph.edges

    def filtered_view(self, node_filter: Callable[[Node], bool]) -> "Graph":
        nodes = list(filter(lambda _: node_filter(_), self.nodes))
        edges = list(
            filter(
                lambda edge: node_filter(edge.src) and node_filter(edge.dst), self.edges
            )
        )
        return Graph(self.cfg, nodes, edges)


class Source(BaseModel, ABC):
    @abstractmethod
    def parse(self, cfg: CFGBase) -> Graph:
        pass


class Transformer(BaseModel, ABC):
    @abstractmethod
    def transform(self, graph: Graph) -> None:
        pass


class Output(BaseModel, ABC):
    @abstractmethod
    def generate(self, graph: Graph) -> str:
        pass


class Vis(BaseModel):
    source: Source
    output: Output

    transformers: list[Transformer]
    contents: list[Content]
    annotators: list[Annotator]

    @model_validator(mode="after")
    def setup(self):

        # create a content dictionary out of the content list
        self._contents = {obj.name: obj for obj in self.contents}

        # split annotators by type
        self._node_annotators = []
        self._edge_annotators = []
        self._content_annotators = {}

        for annotator in self.annotators:
            if isinstance(annotator, NodeAnnotator):
                self._node_annotators.append(annotator)
            elif isinstance(annotator, EdgeAnnotator):
                self._edge_annotators.append(annotator)
            elif isinstance(annotator, ContentAnnotator):
                if annotator.name not in self._contents:
                    raise VisError(
                        f"Content '{annotator.name}' not found, required by annotator '{type(annotator)}'"
                    )
                self._contents[annotator.name].add_annotator(annotator)
            else:
                VisError(f"Unexpected annotator of type {type(annotator)}")

        return self

    @time_it
    def process(self, cfg: CFGBase) -> str:

        # parse input graph
        graph = self.source.parse(cfg)

        # apply graph transformations
        for t in self.transformers:
            t.transform(graph)

        for n in graph.nodes:
            for c in self._contents.values():
                c.render(n)
            for na in self._node_annotators:
                na.annotate_node(n)

        for e in graph.edges:
            for ea in self._edge_annotators:
                ea.annotate_edge(e)

        return self.output.generate(graph)
