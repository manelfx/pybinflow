
from abc import abstractmethod
from typing import Any
from loguru import logger

from bingraph.helpers import get_style
from .vis import NodeAnnotator, ContentAnnotator, EdgeAnnotator, Node


class ColorSimprocedures(NodeAnnotator):
    def annotate_node(self, node: Node) -> None:
        if not node.obj.is_simprocedure:
            return

        node.pydot.set_style("filled")
        if node.obj.simprocedure_name in ['PathTerminator','ReturnUnconstrained','UnresolvableTarget']:
            node.pydot.set_fillcolor("#ffcccc")
        else:
            node.pydot.set_fillcolor("#dddddd")


class CommentsAnnotator(ContentAnnotator):
    name: str = "asm"
    column: str = "comment"

    @abstractmethod
    def get_comments_by_addr(self, node: Node) -> dict[int, list[str]]:
        pass

    def annotate_content(self, node: Node, content: dict[str, Any]):
        if node.obj.is_simprocedure or node.obj.is_syscall:
            return

        comments_by_addr = self.get_comments_by_addr(node)

        for k in content['data']:
            ins = k['_ins']
            if ins.address in comments_by_addr:
                k['comment'] = {'content': "; " + "\n".join(comments_by_addr[ins.address])}
                k['comment']['color'] = 'gray'
                k['comment']['align'] = 'LEFT'


class CommentsDataRef(CommentsAnnotator):

    @staticmethod
    def _symbol_name_at(node: Node, addr: int) -> str | None:

        # Check if it maps to a known internal label
        project = node.project
        if addr in project.kb.labels:
            return project.kb.labels[addr]

        # Check if it maps to a global symbol or imported function
        sym = project.loader.find_symbol(addr)
        if sym:
            return sym.name

        return None

    @staticmethod
    def _truncate_comment(text: str, max_len: int = 64) -> str:
        text = text.replace("\n", "\\n").replace("\r", "\\r")

        if len(text) <= max_len:
            return text

        return text[: max_len - 3] + "..."

    def _format_memory_data_comment(self, node: Node, md) -> str:

        content = getattr(md, "content", None)
        if isinstance(content, (bytes, bytearray)):
            text = content.decode("utf-8", errors="ignore").strip("\x00")
            if text:
                text = self._truncate_comment(text)
                return f'"{text}"'

        addr = getattr(md, "addr", None)
        sort = str(getattr(md, "sort", "data")).lower()

        if addr is not None:
            symbol = self._symbol_name_at(node, addr)
            if symbol:
                return symbol

        if addr is None:
            return f"data: {sort}"

        if "pointer" in sort or sort == "ptr":
            target = getattr(md, "pointer_addr", None)
            if isinstance(target, int):
                target_name = self._symbol_name_at(node, target)
                if target_name:
                    return f"ptr -> {target_name}"
                return f"ptr -> {hex(target)}"

            return f"ptr @ {hex(addr)}"

        if "string" in sort:
            return f"string @ {hex(addr)}"

        if "jumptable" in sort or "jump table" in sort:
            return f"jump table @ {hex(addr)}"

        if "integer" in sort or "int" in sort:
            return f"int @ {hex(addr)}"

        return f"{sort} @ {hex(addr)}"

    def _format_address_comment(self, node: "Node", addr: int) -> str:
        symbol = self._symbol_name_at(node, addr)
        if symbol:
            return symbol

        return f"ref {hex(addr)}"

    def _format_xref_comment(self, node: Node, xref) -> str | None:

        md = getattr(xref, "memory_data", None)
        if md is not None:
            # Case 1: angr resolved a MemoryData object
            return self._format_memory_data_comment(node, md)

        dst = getattr(xref, "dst", None)
        if isinstance(dst, int):
            # Case 2: raw destination address only
            return self._format_address_comment(node, dst)

        return None

    def get_comments_by_addr(self, node: Node) -> dict[int, list[str]]:
        comments_by_addr: dict[int, str] = {}

        kb_xrefs = getattr(node.kb, "xrefs", None)
        if kb_xrefs is None:
            return comments_by_addr

        block_start = node.obj.addr
        block_end = block_start + node.obj.size

        xrefs = kb_xrefs.get_xrefs_by_ins_addr_region(block_start, block_end)
        if not xrefs:
            xrefs = list(getattr(node.obj, "accessed_data_references", []))
        logger.info(f"Found {len(xrefs)} reference(s) in block {hex(block_start)}-{hex(block_end)}")

        for xref in xrefs:
            comment = self._format_xref_comment(node, xref)
            if not comment:
                continue

            # add some space for pretty printing on BB boundary
            comment += " "

            # Merge multiple references originating from the same instruction
            ins_addr = xref.ins_addr
            if ins_addr in comments_by_addr:
                if comment not in comments_by_addr[ins_addr]:
                    comments_by_addr[ins_addr].append(comment)
            else:
                comments_by_addr[ins_addr] = [comment]

        return comments_by_addr


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
                out_edges = edge.src.graph.out_edges(source_node, data=True)
                boring_edges_count = sum(1 for _, _, edge_data in out_edges
                                         if edge_data.get('jumpkind') == 'Ijk_Boring')
                # only one edge found, this must be unconditional
                if boring_edges_count == 1:
                    # check for unconditional branch or fall through edge
                    if edge.dst.obj.addr != source_node.addr + source_node.size:
                        style.make_edge(edge, 'UNCONDITIONAL')
                    else:
                        style.make_edge(edge, 'NEXT')
                # this is a conditional jump if we find 2 edges
                elif boring_edges_count == 2:
                    # look at the source node to figure out the fall-through address
                    fall_through_addr = source_node.addr + source_node.size
                    # lood at destination node to see the branch type
                    if edge.dst.obj.addr == fall_through_addr:
                        style.make_edge(edge, 'CONDITIONAL_FALSE')
                    else:
                        style.make_edge(edge, 'CONDITIONAL_TRUE')
                else:
                    # this should be an indirect jump with many targets, or something else
                    logger.info("found unconditional branch with many targets for edge"
                                f" {source_node.addr:#x} -> {edge.dst.obj.addr:#x}")
                    style.make_edge(edge, 'UNKNOWN')
            else:
                logger.warning(f"Unexpected {jk} type for edge"
                               f" {source_node.addr:#x} -> {edge.dst.obj.addr:#x}")
                style.make_edge(edge, 'UNKNOWN')
        else:
            style.make_edge(edge, 'UNKNOWN')
