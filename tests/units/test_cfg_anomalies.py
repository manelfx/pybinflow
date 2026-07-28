"""Fast tests for custom-CFG anomaly classification and reporting."""

from types import SimpleNamespace

from capstone import CS_GRP_JUMP, CS_OP_IMM
from capstone.x86 import X86_INS_UD2
from loguru import logger
import pyvex

from bingraph.cfg import anomalies
from bingraph.cfg import jumps
from bingraph.cfg.decode import DecodedNode
from bingraph.cfg.models import CFGAnomaly, FunctionBounds


def test_decoding_coverage_detects_zero_sized_nodes() -> None:
    """Classify zero-sized nodes without attempting Capstone inspection."""

    anomaly = anomalies._decoding_coverage_anomaly(SimpleNamespace(addr=0x1000, size=0))

    assert anomaly == CFGAnomaly(
        "decoding_coverage_mismatch", 0x1000, "Node 0x1000 has size zero"
    )


def test_decoding_coverage_accepts_exact_capstone_span(
    monkeypatch,
) -> None:
    """Avoid a coverage anomaly when Capstone exactly covers the node span."""

    insn = SimpleNamespace(address=0x1000, size=2)
    monkeypatch.setattr(
        anomalies.DecodedNode,
        "from_node",
        lambda node: DecodedNode((insn,)),
    )

    assert (
        anomalies._decoding_coverage_anomaly(SimpleNamespace(addr=0x1000, size=2))
        is None
    )


def test_truncated_leaf_accepts_fully_decoded_ud2_trap(monkeypatch) -> None:
    """Accept a complete x86 ud2 trap even though VEX reports Ijk_NoDecode."""

    node = SimpleNamespace(
        addr=0x1000,
        size=2,
        block=SimpleNamespace(vex=SimpleNamespace(jumpkind="Ijk_NoDecode")),
    )
    ud2 = SimpleNamespace(address=0x1000, size=2, id=X86_INS_UD2)
    monkeypatch.setattr(
        anomalies.DecodedNode,
        "from_node",
        lambda _node: DecodedNode((ud2,)),
    )

    assert not anomalies.node_has_truncated_leaf(
        SimpleNamespace(arch=SimpleNamespace(name="AMD64")),
        SimpleNamespace(successors=lambda _node: (), nodes=lambda: (node,)),
        FunctionBounds(0x1000, 0x1010, 0x10, SimpleNamespace(name="f")),
        0x1000,
        node,
    )


def test_truncated_leaf_keeps_non_trap_nodecode_blocks_repairable(monkeypatch) -> None:
    """Do not treat every fully decoded Ijk_NoDecode node as a valid leaf."""

    node = SimpleNamespace(
        addr=0x1000,
        size=2,
        block=SimpleNamespace(vex=SimpleNamespace(jumpkind="Ijk_NoDecode")),
        function_address=0x1000,
        is_simprocedure=False,
    )
    later_node = SimpleNamespace(
        addr=0x1002,
        function_address=0x1000,
        is_simprocedure=False,
    )
    ordinary_insn = SimpleNamespace(address=0x1000, size=2, id=None, groups=())
    monkeypatch.setattr(
        anomalies.DecodedNode,
        "from_node",
        lambda _node: DecodedNode((ordinary_insn,)),
    )
    monkeypatch.setattr(anomalies, "_can_decode_block_at", lambda *_args: True)

    assert anomalies.node_has_truncated_leaf(
        SimpleNamespace(arch=SimpleNamespace(name="AMD64")),
        SimpleNamespace(
            successors=lambda _node: (),
            nodes=lambda: (node, later_node),
        ),
        FunctionBounds(0x1000, 0x1010, 0x10, SimpleNamespace(name="f")),
        0x1000,
        node,
    )


def test_anomaly_detector_logs_each_kind_and_address_once() -> None:
    """Suppress repeated warnings while retaining distinct anomaly categories."""

    messages: list[str] = []
    sink_id = logger.add(messages.append, format="{message}")
    try:
        detector = anomalies.CFGAnomalyDetector(
            object(),
            object(),
            FunctionBounds(0x1000, 0x1010, 0x10, SimpleNamespace(name="f")),
            0x1000,
        )
        detector._report(CFGAnomaly("coverage", 0x1000, "coverage warning"))
        detector._report(CFGAnomaly("coverage", 0x1000, "duplicate warning"))
        detector._report(CFGAnomaly("jump", 0x1000, "jump warning"))
    finally:
        logger.remove(sink_id)

    assert messages == ["coverage warning\n", "jump warning\n"]


def test_terminal_vex_node_with_successor_needs_repair() -> None:
    """Reject CFGFast's stale fallthrough after a VEX-level return."""

    node = SimpleNamespace(
        addr=0x1000,
        block=SimpleNamespace(vex=SimpleNamespace(jumpkind="Ijk_Ret")),
    )
    successor = SimpleNamespace(addr=0x1004)
    graph = SimpleNamespace(
        successors=lambda candidate: (successor,) if candidate is node else ()
    )

    assert anomalies._terminal_successor_anomaly(graph, node) == CFGAnomaly(
        "terminal_successor",
        0x1000,
        "Node 0x1000 has terminal VEX jumpkind Ijk_Ret but retains "
        "successor(s): 0x1004",
    )


def test_terminal_vex_node_keeps_conditional_return_fallthrough() -> None:
    """Keep the explicit not-taken path of a conditional return instruction."""

    successor = SimpleNamespace(addr=0x1004)
    node = SimpleNamespace(
        addr=0x1000,
        size=4,
        block=SimpleNamespace(
            vex=SimpleNamespace(
                jumpkind="Ijk_Ret",
                exit_statements=(
                    (
                        0x1000,
                        0,
                        SimpleNamespace(
                            jumpkind="Ijk_Boring",
                            dst=SimpleNamespace(value=0x1004),
                        ),
                    ),
                ),
            )
        ),
    )
    graph = SimpleNamespace(
        successors=lambda candidate: (successor,) if candidate is node else ()
    )

    assert anomalies._terminal_successor_anomaly(graph, node) is None


def test_terminal_vex_node_rejects_non_fallthrough_successor() -> None:
    """Retain the stale-edge check beside a conditional return fall-through."""

    fallthrough = SimpleNamespace(addr=0x1004)
    stale_successor = SimpleNamespace(addr=0x2000)
    node = SimpleNamespace(
        addr=0x1000,
        size=4,
        block=SimpleNamespace(
            vex=SimpleNamespace(
                jumpkind="Ijk_Ret",
                exit_statements=(
                    (
                        0x1000,
                        0,
                        SimpleNamespace(
                            jumpkind="Ijk_Boring",
                            dst=SimpleNamespace(value=0x1004),
                        ),
                    ),
                ),
            )
        ),
    )
    graph = SimpleNamespace(
        successors=lambda candidate: (
            (fallthrough, stale_successor) if candidate is node else ()
        )
    )

    assert anomalies._terminal_successor_anomaly(graph, node) == CFGAnomaly(
        "terminal_successor",
        0x1000,
        "Node 0x1000 has terminal VEX jumpkind Ijk_Ret but retains "
        "successor(s): 0x2000",
    )


def test_overlapping_instruction_entry_requires_an_external_predecessor(
    monkeypatch,
) -> None:
    """Split a covering node only for an independently entered instruction."""

    covering = SimpleNamespace(addr=0x1000, size=6, is_simprocedure=False)
    entry = SimpleNamespace(addr=0x1002, size=2, is_simprocedure=False)
    source = SimpleNamespace(addr=0x2000)
    graph = SimpleNamespace(
        nodes=lambda: (covering, entry),
        predecessors=lambda node: (source,) if node is entry else (),
    )
    first = SimpleNamespace(address=0x1000, size=2)
    second = SimpleNamespace(address=0x1002, size=2)
    monkeypatch.setattr(
        anomalies.DecodedNode,
        "from_node",
        lambda node: (
            DecodedNode((first, second)) if node is covering else DecodedNode(())
        ),
    )

    assert anomalies.overlapping_instruction_entries(graph, (covering, entry)) == (
        (covering, (0x1002,)),
    )


def test_fresh_vex_target_overrides_relative_capstone_branch_operand(
    monkeypatch,
) -> None:
    """Use a terminator lift when Capstone reports a PC-relative displacement."""

    branch = SimpleNamespace(
        address=0x1000,
        size=4,
        groups=(CS_GRP_JUMP,),
        operands=(SimpleNamespace(type=CS_OP_IMM, imm=-0x20),),
        id=None,
    )
    node = SimpleNamespace(
        addr=0x1000,
        size=4,
        block=SimpleNamespace(
            vex=SimpleNamespace(jumpkind="Ijk_Boring", exit_statements=())
        ),
    )
    fresh_vex = SimpleNamespace(
        jumpkind="Ijk_Boring",
        exit_statements=[
            (0x1000, None, SimpleNamespace(dst=SimpleNamespace(value=0x4000)))
        ],
    )
    project = SimpleNamespace(
        loader=SimpleNamespace(
            find_object_containing=lambda addr: object() if addr == 0x4000 else None
        ),
        factory=SimpleNamespace(
            block=lambda *_args, **_kwargs: SimpleNamespace(vex=fresh_vex)
        ),
    )
    graph = SimpleNamespace(successors=lambda _node: ())
    bounds = FunctionBounds(0x1000, 0x3000, 0x2000, SimpleNamespace(name="f"))
    monkeypatch.setattr(
        jumps.DecodedNode, "from_node", lambda _node: DecodedNode((branch,))
    )
    monkeypatch.setattr(
        anomalies, "node_has_decoding_coverage_mismatch", lambda _node: False
    )
    monkeypatch.setattr(anomalies, "_can_decode_block_at", lambda *_args: True)

    analysis = jumps._analyze_jump_successors(project, graph, bounds, node)

    assert analysis is not None
    assert [expectation.addr for expectation in analysis.expected] == [0x4000, 0x1004]


def test_fresh_vex_target_overrides_stale_node_target_for_direct_branch(
    monkeypatch,
) -> None:
    """Use a terminator lift instead of stale full-node VEX branch metadata."""

    branch = SimpleNamespace(
        address=0x1000,
        size=2,
        groups=(CS_GRP_JUMP,),
        operands=(SimpleNamespace(type=CS_OP_IMM, imm=6),),
        id=None,
    )
    node = SimpleNamespace(
        addr=0x1000,
        size=0x100,
        block=SimpleNamespace(
            vex=SimpleNamespace(
                jumpkind="Ijk_Boring",
                exit_statements=(),
                next=pyvex.expr.Const(pyvex.const.U64(0x2000)),
            )
        ),
    )
    fresh_vex = SimpleNamespace(
        jumpkind="Ijk_Boring",
        exit_statements=(),
        next=pyvex.expr.Const(pyvex.const.U64(0x3000)),
    )
    project = SimpleNamespace(
        loader=SimpleNamespace(
            find_object_containing=lambda addr: (
                object() if addr in {0x2000, 0x3000} else None
            )
        )
    )
    graph = SimpleNamespace(successors=lambda _node: ())
    bounds = FunctionBounds(0x1000, 0x4000, 0x3000, SimpleNamespace(name="f"))
    monkeypatch.setattr(
        jumps.DecodedNode, "from_node", lambda _node: DecodedNode((branch,))
    )
    monkeypatch.setattr(
        anomalies, "node_has_decoding_coverage_mismatch", lambda _node: False
    )
    monkeypatch.setattr(jumps, "lift_instruction_vex", lambda *_args: fresh_vex)

    analysis = jumps._analyze_jump_successors(project, graph, bounds, node)

    assert analysis is not None
    assert [expectation.addr for expectation in analysis.expected] == [0x3000]


def test_terminator_vex_return_is_not_treated_as_an_indirect_jump(
    monkeypatch,
) -> None:
    """Honor a lifter-only return classification for compact encodings."""

    terminal_jump = SimpleNamespace(
        address=0x1000,
        size=2,
        groups=(CS_GRP_JUMP,),
        operands=(),
        id=None,
    )
    node = SimpleNamespace(
        addr=0x1000,
        size=2,
        block=SimpleNamespace(
            vex=SimpleNamespace(jumpkind="Ijk_Boring", exit_statements=())
        ),
    )
    project = SimpleNamespace()
    graph = SimpleNamespace(successors=lambda _node: ())
    bounds = FunctionBounds(0x1000, 0x1010, 0x10, SimpleNamespace(name="f"))
    monkeypatch.setattr(
        jumps.DecodedNode, "from_node", lambda _node: DecodedNode((terminal_jump,))
    )
    monkeypatch.setattr(
        anomalies, "node_has_decoding_coverage_mismatch", lambda _node: False
    )
    monkeypatch.setattr(
        jumps,
        "lift_instruction_vex",
        lambda *_args: SimpleNamespace(jumpkind="Ijk_Ret"),
    )

    assert jumps._analyze_jump_successors(project, graph, bounds, node) is None


def test_inner_call_does_not_require_a_fake_return(
    monkeypatch,
) -> None:
    """Ignore a stale whole-node call lift when the final instruction is not a call."""

    final_branch = SimpleNamespace(
        address=0x1004,
        size=2,
        groups=(CS_GRP_JUMP,),
        operands=(),
        id=None,
    )
    node = SimpleNamespace(
        addr=0x1000,
        size=6,
        block=SimpleNamespace(vex=SimpleNamespace(jumpkind="Ijk_Call")),
    )
    project = SimpleNamespace()
    graph = SimpleNamespace()
    bounds = FunctionBounds(0x1000, 0x1010, 0x10, SimpleNamespace(name="f"))
    monkeypatch.setattr(
        anomalies.DecodedNode, "from_node", lambda _node: DecodedNode((final_branch,))
    )
    monkeypatch.setattr(
        anomalies, "node_has_decoding_coverage_mismatch", lambda _node: False
    )
    monkeypatch.setattr(
        anomalies,
        "lift_instruction_vex",
        lambda *_args: SimpleNamespace(jumpkind="Ijk_Boring"),
    )

    assert (
        anomalies._missing_call_fallthrough_anomaly(project, graph, bounds, node)
        is None
    )


def test_call_fallthrough_requires_a_complete_capstone_continuation(
    monkeypatch,
) -> None:
    """Reject partial data decoding but retain a Capstone-only control transfer."""

    partial = SimpleNamespace(address=0x1004, size=2, control_transfer=False)
    transfer = SimpleNamespace(address=0x1004, size=2, control_transfer=True)

    class Project:
        """Provide a hashable mapping of continuation instructions."""

        arch = SimpleNamespace(max_inst_bytes=16)

        def __init__(self, insns) -> None:
            self.insns = insns

    bounds = FunctionBounds(0x1000, 0x1010, 0x10, SimpleNamespace(name="f"))
    monkeypatch.setattr(
        anomalies,
        "decode_one",
        lambda project, addr, _size: project.insns.get(addr),
    )
    monkeypatch.setattr(
        anomalies,
        "InsnSemantics",
        lambda insn: SimpleNamespace(is_control_transfer=lambda: insn.control_transfer),
    )

    assert not anomalies._has_complete_capstone_block_at(
        Project({0x1004: partial}), bounds, 0x1004
    )
    assert anomalies._has_complete_capstone_block_at(
        Project({0x1004: transfer}), bounds, 0x1004
    )
