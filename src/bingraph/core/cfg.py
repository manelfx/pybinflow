"""
Custom CFG discovery and repair helpers.

This module implements the first prototype of the custom CFG fallback used by
`cfg_mode="custom"`. The current design is intentionally hybrid:

1. Run bounded CFGFast first so we keep angr metadata (KB, labels, xrefs,
   function lookup, comments).
2. Recover bounded basic blocks ourselves using capstone-only disassembly.
3. Rebuild a per-function graph with angr-compatible CFGNode objects so the
   rest of bingraph can render it without special cases.

The implementation is conservative on purpose. It prefers a structurally sane
graph with possibly missing edges over a graph that invents control flow.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Literal

from angr import KnowledgeBase, Project
from angr.analyses.cfg import CFGBase
from angr.knowledge_plugins.cfg import CFGModel, CFGNode
from capstone import CS_GRP_CALL, CS_GRP_JUMP, CS_GRP_RET, CS_OP_IMM, CsInsn
from loguru import logger
import pyvex

from .symbols import FunctionSymbol, list_function_symbols


# Internal-only labels describing how a recovered block terminates while we are
# repairing a CFG. These are not written directly to graph edges, because the
# renderer and the rest of the angr-compatible pipeline only expect the usual
# edge jumpkinds such as Ijk_Boring / Ijk_Call / Ijk_FakeRet.
TerminatorKind = Literal["Ijk_Boring", "Ijk_Call", "Ijk_Fallthrough", "Ijk_Ret"]

# Edge jumpkinds that we actually materialize in the repaired graph. Keep this
# aligned with the jumpkind vocabulary used by normal angr CFG edges.
EdgeJumpKind = Literal["Ijk_Boring", "Ijk_Call", "Ijk_FakeRet"]


@dataclass(frozen=True)
class EdgeSpec:
    """A conservative edge discovered by the custom block walker."""

    src_addr: int
    dst_addr: int
    jumpkind: EdgeJumpKind


@dataclass(frozen=True)
class BlockSpec:
    """A basic-block candidate recovered from bounded disassembly."""

    addr: int
    size: int
    instruction_addrs: tuple[int, ...]
    # Internal repair metadata for the recovered block terminator. This can be
    # more descriptive than what we finally store on graph edges.
    jumpkind: TerminatorKind
    direct_targets: tuple[int, ...] = ()
    fallthrough_addr: int | None = None


@dataclass(frozen=True)
class TerminatorInfo:
    """Normalized control-flow summary for one recovered block terminator."""

    jumpkind: TerminatorKind
    direct_targets: tuple[int, ...] = ()
    fallthrough_addr: int | None = None


@dataclass(frozen=True)
class FunctionBounds:
    """Closed-open function bounds derived from the symbol table."""

    addr: int
    end_addr: int
    size: int
    symbol: FunctionSymbol


@dataclass
class RepairPlan:
    """Container describing how a custom repair should rewrite a seed CFG."""

    func_addr: int
    bounds: FunctionBounds
    block_specs: list[BlockSpec] = field(default_factory=list)
    edge_specs: list[EdgeSpec] = field(default_factory=list)
    reasons: list[str] = field(default_factory=list)


class CustomCFG(SimpleNamespace):
    """Small CFG-like wrapper exposing the attributes bingraph actually uses."""

    graph: object
    model: CFGModel
    functions: object
    kb: KnowledgeBase


class InsnSemantics:
    """
    Small wrapper around a Capstone instruction.

    The custom CFG builder uses Capstone for bounded block recovery, but wants
    higher-level predicates such as "is this a control-transfer instruction?" in
    a place that reads clearly. This helper intentionally stays lightweight:
    it only exposes generic properties needed before we ask VEX for the final
    branch shape of a recovered block.
    """

    def __init__(self, insn: CsInsn):
        self.insn = insn

    @property
    def address(self) -> int:
        return self.insn.address

    @property
    def size(self) -> int:
        return self.insn.size

    def is_ret(self) -> bool:
        return CS_GRP_RET in self.insn.groups

    def is_call(self) -> bool:
        return CS_GRP_CALL in self.insn.groups

    def is_jump(self) -> bool:
        return CS_GRP_JUMP in self.insn.groups

    def is_control_transfer(self) -> bool:
        return self.is_ret() or self.is_call() or self.is_jump()

    def direct_target(self) -> int | None:
        # Do not assume the branch target is always operand 0. Instructions
        # such as Thumb `cbz r2, #0x733` place the condition source first and
        # the branch destination second.
        for operand in self.insn.operands:
            if getattr(operand, "type", None) != CS_OP_IMM:
                continue

            imm = getattr(operand, "imm", None)
            if isinstance(imm, int):
                return imm

        return None

    def fallthrough_addr(self) -> int:
        return self.address + self.size


def _lookup_function_bounds(project: Project, func_addr: int) -> FunctionBounds:
    """Return the symbol-bounded address range for one function."""

    function = next((sym for sym in list_function_symbols(project) if sym.addr == func_addr), None)
    if function is None:
        raise KeyError(f"Function {func_addr:#x} not found in binary")

    return FunctionBounds(
        addr=function.addr,
        end_addr=function.addr + function.size,
        size=function.size,
        symbol=function,
    )


def _is_direct_target_valid(bounds: FunctionBounds, target: int | None) -> bool:
    """Return True when a direct branch target is inside the current function."""

    return target is not None and bounds.addr <= target < bounds.end_addr


def _decode_one(project: Project, addr: int, size: int) -> CsInsn | None:
    """Decode a single instruction from the loader-backed memory image."""

    md = project.arch.capstone
    md.detail = True
    blob = project.loader.memory.load(addr, size)
    return next(md.disasm(blob, addr, count=1), None)


def _decode_function_linear(project: Project, bounds: FunctionBounds) -> list[CsInsn]:
    """
    Decode the function bytes linearly with capstone.

    This intentionally follows a disassembly-style walk instead of a
    reachability-driven walk so we can still recover decodable blocks that CFG
    analyses omit after unresolved indirect jumps.
    """

    max_inst_bytes = getattr(project.arch, "max_inst_bytes", 16)
    insns: list[CsInsn] = []
    cur = bounds.addr

    while cur < bounds.end_addr:
        insn = _decode_one(project, cur, min(max_inst_bytes, bounds.end_addr - cur))
        if insn is None:
            logger.warning(
                f"Capstone failed decoding function {bounds.addr:#x} at {cur:#x}; "
                "stopping linear sweep"
            )
            break

        insns.append(insn)
        cur = insn.address + insn.size

    return insns


def _arch_has_delay_slot(project: Project) -> bool:
    """Return True for architectures where control transfers consume a delay slot."""

    return project.arch.name in {"MIPS32", "MIPS64"}


def _candidate_block_starts(
    project: Project,
    bounds: FunctionBounds,
    insns: list[CsInsn],
) -> set[int]:
    """Return the conservative set of block-start addresses for one function."""

    starts = {bounds.addr}
    has_delay_slot = _arch_has_delay_slot(project)

    for idx, insn in enumerate(insns):
        semantic = InsnSemantics(insn)
        target = semantic.direct_target()

        if _is_direct_target_valid(bounds, target):
            starts.add(target)

        if not semantic.is_control_transfer():
            continue

        continuation_idx = idx + 1
        if has_delay_slot:
            continuation_idx += 1

        if continuation_idx < len(insns):
            next_addr = insns[continuation_idx].address
            # Start a fresh block after any control-transfer instruction. VEX
            # will later tell us whether this boundary is a real fallthrough or
            # just decodable code after a terminating jump. On delay-slot
            # architectures the continuation begins after the consumed slot.
            starts.add(next_addr)

    return starts


def _control_transfer_index(project: Project, block_insns: list[CsInsn]) -> int | None:
    """
    Return the index of the effective control-transfer instruction in a block.

    On most architectures this is simply the last instruction. On delay-slot
    architectures it may be the penultimate instruction, with the final
    instruction being the consumed delay slot.
    """

    has_delay_slot = _arch_has_delay_slot(project)

    for idx in range(len(block_insns) - 1, -1, -1):
        if not InsnSemantics(block_insns[idx]).is_control_transfer():
            continue

        if idx != len(block_insns) - 1 and not has_delay_slot:
            raise RuntimeError(
                f"Recovered non-delay block at {block_insns[0].address:#x} has trailing "
                f"instructions after control transfer {block_insns[idx].address:#x}"
            )
        return idx

    return None


def _lift_block_terminator(project: Project, bounds: FunctionBounds, block_insns: list[CsInsn]) -> TerminatorInfo:
    """
    Lift one recovered block with VEX and derive its control-flow shape.

    Capstone is used to recover the bounded instruction stream, but we defer the
    final branch classification to VEX so we can reuse the same arch-specific
    semantics that CFGFast relies on for conditional-vs-unconditional structure.
    If a recovered control-transfer block cannot be lifted, we raise instead of
    silently falling back to heuristics.
    """

    term_idx = _control_transfer_index(project, block_insns)
    block_end_addr = block_insns[-1].address + block_insns[-1].size

    if term_idx is None:
        next_addr = block_end_addr
        fallthrough_addr = next_addr if next_addr < bounds.end_addr else None
        return TerminatorInfo(jumpkind="Ijk_Fallthrough", fallthrough_addr=fallthrough_addr)

    tail_insns = block_insns[term_idx:]
    last = tail_insns[0]
    semantic = InsnSemantics(last)
    next_addr = block_end_addr

    tail_addr = tail_insns[0].address
    tail_size = sum(insn.size for insn in tail_insns)
    block_addr = block_insns[0].address
    block_size = sum(insn.size for insn in block_insns)

    def _lift(addr: int, size: int):
        return project.factory.block(
            addr,
            size=size,
            strict_block_end=True,
            cross_insn_opt=False,
        ).vex

    try:
        vex = _lift(tail_addr, tail_size)
    except Exception:
        try:
            vex = _lift(block_addr, block_size)
        except Exception as exc:
            raise RuntimeError(
                f"Custom CFG failed lifting control-transfer block at {block_addr:#x} "
                f"(terminator {last.address:#x}: {last.mnemonic} {last.op_str})"
            ) from exc

    exit_targets: list[int] = []
    for ins_addr, _, stmt in vex.exit_statements:
        if ins_addr != last.address:
            continue
        if isinstance(stmt.dst, pyvex.expr.Const):
            target = stmt.dst.con.value
            if _is_direct_target_valid(bounds, target):
                exit_targets.append(target)

    default_target = None
    if isinstance(vex.next, pyvex.expr.Const):
        default_target = vex.next.con.value

    if semantic.is_ret():
        return TerminatorInfo(jumpkind="Ijk_Ret")

    if semantic.is_call():
        direct_targets = ()
        if _is_direct_target_valid(bounds, default_target):
            direct_targets = (default_target,)
        fallthrough_addr = next_addr if next_addr < bounds.end_addr else None
        return TerminatorInfo(
            jumpkind="Ijk_Call",
            direct_targets=direct_targets,
            fallthrough_addr=fallthrough_addr,
        )

    if exit_targets:
        fallthrough_addr = default_target if _is_direct_target_valid(bounds, default_target) else None
        return TerminatorInfo(
            jumpkind="Ijk_Boring",
            direct_targets=tuple(exit_targets),
            fallthrough_addr=fallthrough_addr,
        )

    direct_targets = ()
    if _is_direct_target_valid(bounds, default_target):
        direct_targets = (default_target,)
    return TerminatorInfo(jumpkind="Ijk_Boring", direct_targets=direct_targets)


def _build_block_specs(
    project: Project,
    bounds: FunctionBounds,
    insns: list[CsInsn],
) -> tuple[list[BlockSpec], list[EdgeSpec]]:
    """Split a linear disassembly into conservative basic blocks and edges."""

    if not insns:
        return [], []

    start_set = _candidate_block_starts(project, bounds, insns)
    block_specs: list[BlockSpec] = []
    has_delay_slot = _arch_has_delay_slot(project)

    idx = 0
    while idx < len(insns):
        insn = insns[idx]
        if insn.address not in start_set:
            idx += 1
            continue

        block_insns = [insn]
        j = idx
        while True:
            last = block_insns[-1]
            next_idx = j + 1
            semantic = InsnSemantics(last)
            if semantic.is_control_transfer():
                if has_delay_slot and next_idx < len(insns):
                    # Keep the consumed delay-slot instruction in the same
                    # recovered block as the branch/jump that owns it.
                    block_insns.append(insns[next_idx])
                    j = next_idx
                break
            if next_idx >= len(insns):
                break
            next_insn = insns[next_idx]
            if next_insn.address in start_set:
                break
            block_insns.append(next_insn)
            j = next_idx

        terminator = _lift_block_terminator(project, bounds, block_insns)

        block_specs.append(
            BlockSpec(
                addr=block_insns[0].address,
                size=sum(obj.size for obj in block_insns),
                instruction_addrs=tuple(obj.address for obj in block_insns),
                jumpkind=terminator.jumpkind,
                direct_targets=terminator.direct_targets,
                fallthrough_addr=terminator.fallthrough_addr,
            )
        )
        idx = j + 1

    block_addrs = {block.addr for block in block_specs}
    edge_specs: list[EdgeSpec] = []
    for block in block_specs:
        for target in block.direct_targets:
            if target in block_addrs:
                edge_jumpkind: EdgeJumpKind = "Ijk_Call" if block.jumpkind == "Ijk_Call" else "Ijk_Boring"
                edge_specs.append(EdgeSpec(block.addr, target, edge_jumpkind))

        if block.fallthrough_addr is not None and block.fallthrough_addr in block_addrs:
            # Renderers expect graph edges to use the jumpkind values that
            # CFGFast typically stores on edges, not our richer internal block
            # terminator labels.
            edge_jumpkind: EdgeJumpKind = "Ijk_FakeRet" if block.jumpkind == "Ijk_Call" else "Ijk_Boring"
            edge_specs.append(EdgeSpec(block.addr, block.fallthrough_addr, edge_jumpkind))

    return block_specs, edge_specs


def discover_function_blocks(project: Project, func_addr: int) -> RepairPlan:
    """
    Discover bounded block candidates for one function.

    The current implementation uses a linear capstone sweep bounded by the
    symbol-reported function range, then splits the instruction stream into
    conservative basic blocks.
    """

    bounds = _lookup_function_bounds(project, func_addr)
    logger.info(
        f"Discovering custom CFG blocks for function {func_addr:#x} "
        f"in range [{bounds.addr:#x}, {bounds.end_addr:#x})"
    )

    insns = _decode_function_linear(project, bounds)
    block_specs, edge_specs = _build_block_specs(project, bounds, insns)

    plan = RepairPlan(func_addr=func_addr, bounds=bounds)
    plan.block_specs.extend(block_specs)
    plan.edge_specs.extend(edge_specs)
    plan.reasons.append("bounded_capstone_linear_sweep")

    return plan


def _clone_model(seed_cfg: CFGBase) -> CFGModel:
    """Return a fresh CFGModel tied to the same angr KB/project as the seed."""

    seed_model = seed_cfg.model
    model = CFGModel(
        ident="CFGFastCustom",
        cfg_manager=seed_model._cfg_manager,
        is_arm=seed_model.is_arm,
        cache_limit=None,
        addr_type=seed_model.addr_type,
    )
    model._iropt_level = seed_model._iropt_level
    return model


def _block_name(bounds: FunctionBounds, block: BlockSpec) -> str:
    """Return the function-relative label used for a recovered block."""

    if block.addr == bounds.addr:
        return bounds.symbol.name
    return f"{bounds.symbol.name}+0x{block.addr - bounds.addr:x}"


def repair_cfg_from_blocks(seed_cfg: CFGBase, plan: RepairPlan) -> CFGBase:
    """
    Repair a seed angr CFG using a previously discovered custom block plan.

    For the first prototype, we rebuild a per-function graph from scratch using
    real angr CFGNode objects backed by a fresh CFGModel, while preserving the
    seed function manager and knowledge base.
    """

    logger.info(
        f"Repairing seed CFG for function {plan.func_addr:#x} with "
        f"{len(plan.block_specs)} custom block(s)"
    )

    model = _clone_model(seed_cfg)
    nodes_by_addr: dict[int, CFGNode] = {}

    for block in plan.block_specs:
        node = CFGNode(
            block.addr,
            block.size,
            cfg=model,
            function_address=plan.func_addr,
            block_id=block.addr,
            instruction_addrs=block.instruction_addrs,
            name=_block_name(plan.bounds, block),
        )
        model.graph.add_node(node)
        nodes_by_addr[block.addr] = node

    for edge in plan.edge_specs:
        src = nodes_by_addr.get(edge.src_addr)
        dst = nodes_by_addr.get(edge.dst_addr)
        if src is None or dst is None:
            continue
        model.graph.add_edge(src, dst, jumpkind=edge.jumpkind)

    return CustomCFG(
        graph=model.graph,
        model=model,
        functions=seed_cfg.functions,
        kb=seed_cfg.kb,
    )


def build_custom_cfg(
    project: Project,
    kb: KnowledgeBase,
    func_addr: int,
    seed_cfg: CFGBase,
) -> CFGBase:
    """
    Build a custom repaired CFG for one function starting from CFGFast output.

    The KB parameter is kept explicit because the long-term plan is to keep the
    custom path anchored to the same knowledge-base state as CFGFast, even if we
    later enrich the repair process with extra metadata.
    """

    _ = kb
    logger.info(f"Building custom CFG for function {func_addr:#x}")
    plan = discover_function_blocks(project, func_addr)
    return repair_cfg_from_blocks(seed_cfg, plan)


def dump_repair_plan(plan: RepairPlan, output_path: Path) -> None:
    """Persist a human-readable repair-plan sketch for debugging."""

    lines = [
        f"func_addr={plan.func_addr:#x}",
        f"range=[{plan.bounds.addr:#x}, {plan.bounds.end_addr:#x})",
        f"size={plan.bounds.size}",
        "reasons=" + (", ".join(plan.reasons) if plan.reasons else "<none>"),
        "",
        "[blocks]",
    ]

    for block in plan.block_specs:
        targets = ", ".join(f"{addr:#x}" for addr in block.direct_targets) or "-"
        fallthrough = "-" if block.fallthrough_addr is None else f"{block.fallthrough_addr:#x}"
        lines.append(
            f"{block.addr:#x} size={block.size} jumpkind={block.jumpkind} "
            f"targets=[{targets}] fallthrough={fallthrough}"
        )

    lines.append("")
    lines.append("[edges]")
    for edge in plan.edge_specs:
        lines.append(f"{edge.src_addr:#x} -> {edge.dst_addr:#x} ({edge.jumpkind})")

    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
