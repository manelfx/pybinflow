"""Fast tests for Capstone instruction semantics shared by CFG code."""

from types import SimpleNamespace

from capstone import CS_GRP_JUMP, CS_OP_IMM
from capstone.systemz import SYSZ_INS_BC

from bingraph.helpers.capstone import InsnSemantics


def test_branch_to_next_instruction_is_linear_for_cfg() -> None:
    """Keep PC-relative branch-to-next idioms inside their basic block."""

    insn = SimpleNamespace(
        address=0x1000,
        size=4,
        groups=(CS_GRP_JUMP,),
        id=0,
        operands=(SimpleNamespace(type=CS_OP_IMM, imm=0x1004),),
    )

    assert not InsnSemantics(insn).is_control_transfer()


def test_systemz_zero_mask_bc_is_linear_for_cfg() -> None:
    """Treat SystemZ's never-taken BC mask as a no-op, not a branch."""

    insn = SimpleNamespace(
        address=0x1000,
        size=4,
        groups=(CS_GRP_JUMP,),
        id=SYSZ_INS_BC,
        operands=(
            SimpleNamespace(type=CS_OP_IMM, imm=0),
            SimpleNamespace(type=CS_OP_IMM, imm=0),
        ),
    )

    assert not InsnSemantics(insn).is_control_transfer()
