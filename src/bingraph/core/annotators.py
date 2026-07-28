from abc import abstractmethod
from typing import Any
from loguru import logger

from bingraph.helpers import get_style
from bingraph.helpers.capstone import InsnSemantics, control_transfer_index
from bingraph.cfg.recovery import vex_jumpkind_is_terminal
from .vis import NodeAnnotator, ContentAnnotator, EdgeAnnotator, Node


class ColorSimprocedures(NodeAnnotator):
    def annotate_node(self, node: Node) -> None:
        if not node.obj.is_simprocedure:
            return

        node.pydot.set_style("filled")
        if node.obj.simprocedure_name in [
            "PathTerminator",
            "ReturnUnconstrained",
            "UnresolvableTarget",
        ]:
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

        for k in content["data"]:
            ins = k["_ins"]
            if ins.address in comments_by_addr:
                k["comment"] = {
                    "content": " ; " + "\n".join(comments_by_addr[ins.address])
                }
                k["comment"]["color"] = "gray"
                k["comment"]["align"] = "LEFT"


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
        comments_by_addr: dict[int, list[str]] = {}

        kb_xrefs = getattr(node.kb, "xrefs", None)
        if kb_xrefs is None:
            return comments_by_addr

        block_start = node.obj.addr
        block_end = block_start + node.obj.size

        xrefs = kb_xrefs.get_xrefs_by_ins_addr_region(block_start, block_end)
        if not xrefs:
            # Do not use "accessed_data_refernces" sice CFGFast is required
            # e.g., xrefs = list(getattr(node.obj, "accessed_data_references", []))
            for instr_addr in node.obj.instruction_addrs:
                xrefs.update(kb_xrefs.get_xrefs_by_ins_addr(instr_addr))
        if len(xrefs):
            logger.info(
                f"Found {len(xrefs)} reference(s) in block {hex(block_start)}-{hex(block_end)}"
            )

        def _xref_sort_key(xref):
            md = getattr(xref, "memory_data", None)
            return (
                getattr(xref, "ins_addr", -1),
                getattr(xref, "dst", -1),
                getattr(md, "addr", -1) if md is not None else -1,
                str(getattr(md, "sort", "")) if md is not None else "",
            )

        # Keep comment emission deterministic across runs. angr xref iteration
        # order is not stable enough for golden-file tests when an instruction
        # accumulates multiple references/comments.
        for xref in sorted(xrefs, key=_xref_sort_key):
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

        for comments in comments_by_addr.values():
            comments.sort()

        return comments_by_addr


def _is_unresolvable_jump_target(node: Node) -> bool:
    """Return whether ``node`` is angr's unresolved indirect-jump placeholder."""

    return (
        node.obj.is_simprocedure
        and node.obj.simprocedure_name == "UnresolvableJumpTarget"
    )


def _control_transfer_tail(edge):
    """Return a block's control-transfer instruction and any delay-slot tail."""

    source_node = edge.src.obj
    try:
        insns = [wrapped.insn for wrapped in source_node.block.capstone.insns]
        terminator_index = control_transfer_index(edge.src.project.arch.name, insns)
        if terminator_index is None:
            return None
        return insns[terminator_index:]
    except (AttributeError, KeyError, RuntimeError):
        return None


def _lift_control_transfer_tail(edge, tail):
    """Lift a Capstone-discovered control-transfer tail for classification."""

    try:
        return edge.src.project.factory.block(
            tail[0].address,
            size=sum(insn.size for insn in tail),
            strict_block_end=True,
            cross_insn_opt=False,
        ).vex
    except Exception:
        # This is a presentation-only recovery attempt. A failed tail lift
        # means the edge is genuinely unclassifiable, not a render failure.
        return None


def _capstone_folded_conditional_edge_type(edge, exit_targets: set[int]) -> str | None:
    """Classify a direct conditional branch that VEX has constant-folded.

    VEX may model a system-register read as a concrete value and eliminate a
    later conditional branch. When CFGFast still retains both architectural
    successors, preserve that static CFG information for presentation.
    """

    if exit_targets:
        return None
    tail = _control_transfer_tail(edge)
    if tail is None:
        return None
    semantics = InsnSemantics(tail[0])
    if not semantics.is_conditional_jump():
        return None
    target_addr = semantics.direct_target()
    if target_addr is None:
        return None
    fallthrough_addr = tail[-1].address + tail[-1].size
    # Some control-transfer instructions, such as x86 XBEGIN with a zero
    # displacement, encode the sequential address as their only target. They
    # do not represent a distinct taken edge in the rendered CFG.
    if target_addr == fallthrough_addr:
        return None
    try:
        successor_addrs = {
            successor.addr for successor in edge.src.graph.successors(edge.src.obj)
        }
    except (AttributeError, KeyError):
        return None
    if {target_addr, fallthrough_addr} - successor_addrs:
        return None
    if edge.dst.obj.addr == target_addr:
        return "CONDITIONAL_TRUE"
    if edge.dst.obj.addr == fallthrough_addr:
        return "CONDITIONAL_FALSE"
    return None


def _vex_boring_edge_type(edge) -> str:
    """Classify one ordinary edge from its source block's lifted terminator."""

    source_node = edge.src.obj
    try:
        vex = source_node.block.vex
    except (AttributeError, KeyError):
        return "UNKNOWN"

    if vex.jumpkind == "Ijk_NoDecode":
        # Lift only the branch tail when preceding SIMD/extension instructions
        # make VEX reject the complete block.
        tail = _control_transfer_tail(edge)
        if tail is None:
            if edge.dst.obj.addr == source_node.addr + source_node.size:
                return "NEXT"
            return "UNKNOWN"
        vex = _lift_control_transfer_tail(edge, tail)
        if vex is None:
            return "UNKNOWN"

    if vex.jumpkind == "Ijk_Call":
        try:
            call_target = vex.next.con.value
        except AttributeError:
            return "UNKNOWN"
        # CFGFast occasionally retains a direct call target as Ijk_Boring even
        # though VEX identifies the source transfer as a call. Only recover
        # the style when the edge lands at that exact lifted target.
        return "CALL" if edge.dst.obj.addr == call_target else "UNKNOWN"

    terminal_default = vex_jumpkind_is_terminal(vex.jumpkind)
    if vex.jumpkind != "Ijk_Boring" and not terminal_default:
        return "UNKNOWN"

    try:
        next_addr = vex.next.con.value
    except AttributeError:
        next_addr = None

    # VEX records explicit Exit statements for conditional branches. Depending
    # on the lifter, either the exit or the default `next` can be the taken
    # destination, so both are valid non-fall-through successors.
    exit_targets: set[int] = set()
    for _, _, stmt in vex.exit_statements:
        if stmt.jumpkind != "Ijk_Boring":
            continue
        try:
            target = stmt.dst.value
        except AttributeError:
            continue
        if isinstance(target, int):
            exit_targets.add(target)

    folded_conditional_type = _capstone_folded_conditional_edge_type(edge, exit_targets)
    if folded_conditional_type is not None:
        return folded_conditional_type

    if exit_targets:
        if terminal_default:
            # Conditional return instructions can use a terminal default VEX
            # jumpkind together with an explicit Ijk_Boring exit for their
            # non-returning path. The explicit exit remains a real branch.
            if edge.dst.obj.addr in exit_targets:
                return "CONDITIONAL_TRUE"
            if edge.dst.obj.addr == next_addr:
                return "CONDITIONAL_FALSE"
            return "UNKNOWN"

        fallthrough_addr = source_node.addr + source_node.size
        if (
            edge.dst.obj.addr == fallthrough_addr
            and exit_targets == {source_node.addr}
            and _control_transfer_tail(edge) is None
        ):
            # VEX models some atomic x86 instructions with a self-targeting
            # internal Exit. Capstone sees no control transfer, so the only
            # CFG successor is the ordinary next instruction.
            return "NEXT"
        if edge.dst.obj.addr == fallthrough_addr:
            return "CONDITIONAL_FALSE"
        if edge.dst.obj.addr in exit_targets or edge.dst.obj.addr == next_addr:
            return "CONDITIONAL_TRUE"
        return "UNKNOWN"

    if next_addr is None:
        # A non-constant VEX `next` is an indirect branch. Recovered table
        # entries are concrete edges, but the dispatch itself remains indirect.
        return "INDIRECT"

    if not isinstance(next_addr, int) or edge.dst.obj.addr != next_addr:
        return "UNKNOWN"
    if next_addr == source_node.addr + source_node.size:
        return "NEXT"
    return "UNCONDITIONAL"


def _edge_type(edge) -> str:
    """Return the visual category for one CFG edge."""

    # Custom CFG repair may flatten an UnresolvableJumpTarget placeholder into
    # direct candidate edges. The marker keeps that unresolved semantics
    # visible after the synthetic endpoint itself has been removed.
    if edge.meta.get("unresolved_indirect"):
        return "UNRESOLVED_INDIRECT"

    # Both sides of this synthetic node express unresolved control flow: the
    # incoming edge is the unresolved jump and outgoing edges are candidates.
    if _is_unresolvable_jump_target(edge.src) or _is_unresolvable_jump_target(edge.dst):
        return "UNRESOLVED_INDIRECT"

    jumpkind = edge.meta.get("jumpkind")
    if jumpkind == "Ijk_Ret":
        return "RET"
    if jumpkind == "Ijk_FakeRet":
        return "FAKE_RET"
    # System transfers such as x86 ``int 0x80`` cross into an OS service just
    # like calls. Keep their explicit fake-return edges distinct below.
    if jumpkind == "Ijk_Call" or (
        isinstance(jumpkind, str) and jumpkind.startswith("Ijk_Sys_")
    ):
        return "CALL"
    if jumpkind == "Ijk_Boring":
        return _vex_boring_edge_type(edge)

    logger.warning(
        f"Unexpected {jumpkind!r} edge type for "
        f"{edge.src.obj.addr:#x} -> {edge.dst.obj.addr:#x}"
    )
    return "UNKNOWN"


class ColorEdgesVex(EdgeAnnotator):
    """Apply semantic edge styles derived from VEX and repair metadata."""

    def annotate_edge(self, edge) -> None:
        """Style one edge without inferring branch kind from successor count."""

        get_style().make_edge(edge, _edge_type(edge))
