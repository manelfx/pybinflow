
from typing import Dict, Any
from .vis import NodeAnnotator, ContentAnnotator, EdgeAnnotator, Node
from .style import get_style


class ColorSimprocedures(NodeAnnotator):
    def annotate_node(self, node: Node) -> None:
        if node.obj.is_simprocedure:
            if node.obj.simprocedure_name in ['PathTerminator','ReturnUnconstrained','UnresolvableTarget']:
                node.style = 'filled'
                node.fillcolor = '#ffcccc'
            else:
                node.style = 'filled'
                node.fillcolor = '#dddddd'


class CommentsAsm(ContentAnnotator):
    name: str = "asm"
    column: str = "comment"

    def annotate_content(self, node: Node, content: Dict[str, Any]) -> None:
        if node.obj.is_simprocedure or node.obj.is_syscall:
            return

        comments_by_addr = {}
        if len(node.obj.final_states) > 0:
            state = node.obj.final_states[0]
            for action in state.log.actions:
                label = ''
                if action.type == 'mem' or action.type == 'reg':
                    if isinstance(action.data.ast, int) or action.data.ast.concrete:
                        d = state.solver.eval(action.data.ast)
                        if d in node.cfg.project.kb.labels:
                            label += 'data=' + node.cfg.project.kb.labels[d] + ' '
                    if isinstance(action.addr.ast, int) or action.addr.ast.concrete:
                        a = state.solver.eval(action.addr.ast)
                        if a in node.cfg.project.kb.labels:
                            label += 'addr=' + node.cfg.project.kb.labels[a] + ' '

                if action.type == 'exit':
                    if action.target.ast.concrete:
                        a = state.solver.eval(action.target.ast)
                        if a in node.cfg.project.kb.labels:
                            label += node.cfg.project.kb.labels[a] + ' '

                if label != '':
                    comments_by_addr[action.ins_addr] = label

        for k in content['data']:
            ins = k['_ins']
            if ins.address in comments_by_addr:
                if not ('comment' in k and 'content' in k['comment']):
                    k['comment'] = {
                        'content': "; " + comments_by_addr[ins.address][:100]
                    }
                else:
                    k['comment']['content'] += ", " + comments_by_addr[ins.address][:100]

                k['comment']['color'] = 'gray'
                k['comment']['align'] = 'LEFT'


class CommentsDataRef(ContentAnnotator):
    name: str = "asm"
    column: str = "comment"

    def annotate_content(self, node: Node, content: Dict[str, Any]):
        if node.obj.is_simprocedure or node.obj.is_syscall:
            return

        comments_by_addr = {}
        for dr in node.obj.accessed_data_references:
            comments_by_addr[dr.ins_addr] = str(dr)
            if dr.memory_data.sort == 'string':
                comments_by_addr[dr.ins_addr] = str(dr.memory_data.content)

        for k in content['data']:
            ins = k['_ins']
            if ins.address in comments_by_addr:
                if not ('comment' in k and 'content' in k['comment']):
                    k['comment'] = {
                        'content': "; " + comments_by_addr[ins.address][:100]
                    }
                else:
                    k['comment']['content'] += ", " + comments_by_addr[ins.address][:100]

                k['comment']['color'] = 'gray'
                k['comment']['align'] = 'LEFT'


class ColorEdgesVex(EdgeAnnotator):
    def annotate_edge(self, edge):
        style = get_style()

        if 'jumpkind' in edge.meta:
            jk = edge.meta['jumpkind']
            if jk == 'Ijk_Ret':
                style.make_edge(edge, 'RET')
            elif jk == 'Ijk_FakeRet':
                style.make_edge(edge, 'FAKE_RET')
            elif jk == 'Ijk_Call':
                style.make_edge(edge, 'CALL')
            elif jk == 'Ijk_Boring':
                # Check if edge is a conditional jump by counting the "boring"
                # edges exiting from source node.
                source_node = edge.src.obj
                out_edges = edge.src.cfg.graph.out_edges(source_node, data=True)
                boring_edges_count = sum(1 for _, _, edge_data in out_edges
                                         if edge_data.get('jumpkind') == 'Ijk_Boring')
                # only one edge found, this must be unconditional
                if boring_edges_count == 1:
                    style.make_edge(edge, 'UNCONDITIONAL')
                # this is a conditional jump if we find 2 edges
                elif boring_edges_count == 2:
                    # look at the source node to figure out the fall-through address
                    fall_through_addr = source_node.addr + source_node.size
                    # lood at destination node to see the branch type
                    if edge.dst.obj.addr == fall_through_addr:
                        style.make_edge(edge, 'CONDITIONAL_FALSE')
                    else:
                        style.make_edge(edge, 'CONDITIONAL_TRUE')
                # this should be an indirect jump with many targets, or something else
                else:
                    style.make_edge(edge, 'UNKNOWN')
            else:
                #TODO warning
                style.make_edge(edge, 'UNKNOWN')
        else:
            style.make_edge(edge, 'UNKNOWN')
