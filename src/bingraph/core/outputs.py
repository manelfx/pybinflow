
from typing import Any
from pydot import Dot
from .vis import Output, Graph, Node


escape_map = {
    "!" : "&#33;",
    "#" : "&#35;",
    ":" : "&#58;",
    "{" : "&#123;",
    "}" : "&#125;",
    "<" : "&#60;",
    ">" : "&#62;",
    "\t": "&nbsp;",
    "&" : "&amp",
    "|" : "&#124;",
}


def escape(text: str) -> str:
    return "".join(escape_map.get(c,c) for c in text)


default_node_attributes = {
    'shape'    : 'Mrecord',
    'fontname' : 'monospace',
    'fontsize' : '8',
}


default_edge_attributes = {
    'fontname' : 'monospace',
    'fontsize' : '8',
}


class DotOutput(Output):
    fname: str
    format: str = "png"

    def render_cell(self, key: str, data: dict[str, Any] | None) -> str:
        if data is not None and data['content'] is not None and data['content'].strip() != '':
            ret = '<TD '+ ('bgcolor="'+data['bgcolor']+'" ' if 'bgcolor' in data else '') + ('ALIGN="'+data['align']+'"' if 'align' in data else '' )+'>'
            if 'color' in data:
                ret += '<FONT COLOR="'+data['color']+'">'
            if 'style' in data:
                ret += '<'+data['style']+'>'

            if isinstance(data['content'], list):
                ret += '<TABLE BORDER="0">'
                for c in data['content']:
                    ret += '<TR><TD ' + ('ALIGN="'+data['align']+'"' if 'align' in data else '' )+'>'
                    ret += escape(c)
                    ret += '</TD></TR>'
                ret += '</TABLE>'
            else:
                ret += escape(data['content'])
            if 'style' in data:
                ret += '</'+data['style']+'>'
            if 'color' in data:
                ret += '</FONT>'
            ret += "</TD>"
            return ret
        else:
            return "<TD></TD>"

    def render_row(self, row: dict[str, Any], colmeta: list[str]) -> str:
        ret = "<TR>"
        for k in colmeta:
            ret += self.render_cell(k, row[k] if k in row else None)
        ret += "</TR>"
        return ret

    def render_content(self, c: dict[str, Any]) -> str:
        ret = ''
        if len(c['data']) > 0:
            ret = '<TABLE BORDER="0" CELLPADDING="1" ALIGN="LEFT">'
            for r in c['data']:
                ret += self.render_row(r, c['columns'])
            ret += '</TABLE>'
        return ret

    def set_node_label(self, node: Node):

        label = " | ".join([self.render_content(c) for c in node.content.values()])
        if label:
            node.pydot.set_label('<{ %s }>' % label)

    def generate(self, graph: Graph) -> str:

        digraph = Dot(graph_type="digraph", rankdir="TB")
        digraph.set_node_defaults(**default_node_attributes)
        digraph.set_edge_defaults(**default_edge_attributes)

        # add nodes, sorted by node (addr)
        nodes = sorted(graph.nodes, key=lambda n: n.obj.addr)
        for node in nodes:
            self.set_node_label(node)
            digraph.add_node(node.pydot)

        # add edges
        for edge in graph.edges:
            digraph.add_edge(edge.pydot)

        # write graph to output file
        digraph.write("{}.{}".format(self.fname, self.format), format=self.format)

        return digraph.to_string()
