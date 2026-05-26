
from typing import Dict, Any, Optional, List
from pydot import Dot
from .vis import Output, Graph

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
    'fontsize' : '8.0',
}

default_edge_attributes = {
    'fontname' : 'monospace',
    'fontsize' : '8.0',
}

class XDot(Dot):
    def __init__(self, content: str) -> None:
        super().__init__()
        self.content = content

    def to_string(self) -> str:
        return self.content

class DotOutput(Output):
    fname: str
    format: str = "png"

    def render_attributes(self, default: Dict[str, str], attrs: Dict[str, str]) -> str:
        a = {}
        a.update(default)
        a.update(attrs)
        r = []
        for k,v in a.items():
            r.append(k+"="+v)

        return "["+", ".join(r)+"]"

    def render_cell(self, key: str, data: Optional[Dict[str, Any]]) -> str:
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

    def render_row(self, row: Dict[str, Any], colmeta: List[str]) -> str:
        ret = "<TR>"
        for k in colmeta:
            ret += self.render_cell(k, row[k] if k in row else None)
        ret += "</TR>"
        return ret

    def render_content(self, c: Dict[str, Any]) -> str:
        ret = ''
        if len(c['data']) > 0:
            ret = '<TABLE BORDER="0" CELLPADDING="1" ALIGN="LEFT">'
            for r in c['data']:
                ret += self.render_row(r, c['columns'])
            ret += '</TABLE>'
        return ret

    def render_node(self, n: Any) -> str:
        attrs = {}
        if n.style:
            attrs['style'] = n.style
        if n.fillcolor:
            attrs['fillcolor'] = '"'+n.fillcolor+'"'
        if n.color:
            attrs['color'] = n.color
        if n.width:
            attrs['penwidth'] = str(n.width)
        if n.url:
            attrs['URL'] = '"'+n.url+'"'
        if n.tooltip:
            attrs['tooltip'] = '"'+n.tooltip+'"'

        label = "|".join([self.render_content(c) for c in n.content.values()])
        if label:
            attrs['label'] = '<{ %s }>' % label

        return "%s %s" % (str(n.seq), self.render_attributes(default_node_attributes, attrs))

    def render_edge(self, e: Any) -> str:
        attrs = {}
        if e.color:
            attrs['color'] = e.color
        if e.label:
            attrs['label'] = '"'+e.label+'"'
        if e.style:
            attrs['style'] = e.style
        if e.width:
            attrs['penwidth'] = str(e.width)
        if e.weight:
            attrs['weight'] = str(e.weight)

        return "%s -> %s %s" % (str(e.src.seq), str(e.dst.seq), self.render_attributes(default_edge_attributes, attrs))

    def generate_cluster(self, graph: Graph) -> str:
        ret = ""

        nodes = graph.nodes

        try:
            if len(nodes) > 0:
                nodes = sorted(nodes, key=lambda n: n.obj.addr)
        except AttributeError:
            # if the nodes don't have address
            pass

        for n in nodes:
            ret += self.render_node(n) + "\n"

        return ret

    def generate(self, graph: Graph) -> str:
        ret  = "digraph \"\" {\n"
        ret += "rankdir=TB;\n"
        ret += "newrank=true;\n"
        ret += "labeljust=l;\n"

        ret += self.generate_cluster(graph)

        for e in graph.edges:
            ret += self.render_edge(e) + "\n"

        ret += "}\n"

        if self.fname:
            dotfile = XDot(ret)
            dotfile.write("{}.{}".format(self.fname, self.format), format=self.format)


class DumpOutput(Output):
    def generate(self, graph: Graph) -> str:
        ret = ""
        for e in graph.edges:
            ret += self.render_edge(e) + "\n"
        print(ret)

    def render_edge(self, e: Any) -> str:
        return "%s %s %s" % (hex(e.src.obj.addr), hex(e.dst.obj.addr), e.meta['jumpkind'])
