"""Fast tests for Capstone instruction semantics shared by CFG code."""

from types import SimpleNamespace

import pytest
from capstone import CS_ARCH_X86, CS_GRP_JUMP, CS_MODE_64, CS_OP_IMM, Cs
from capstone.arm import ARM_CC_AL, ARM_INS_IT
from capstone.systemz import SYSZ_INS_BC

from bingraph.helpers.capstone import (
    InsnSemantics,
    control_transfer_index,
    proven_unconditional_direct_target,
)


def test_branch_to_next_instruction_remains_a_cfg_terminator() -> None:
    """Keep an adjacent direct jump as a basic-block boundary."""

    insn = SimpleNamespace(
        address=0x1000,
        size=4,
        groups=(CS_GRP_JUMP,),
        id=0,
        operands=(SimpleNamespace(type=CS_OP_IMM, imm=0x1004),),
    )

    assert InsnSemantics(insn).is_control_transfer()


def test_thumb_it_context_prevents_an_unconditional_branch_proof() -> None:
    """Keep an IT-predicated Thumb branch under VEX's control-flow model."""

    it = SimpleNamespace(
        address=0x1000,
        size=2,
        groups=(),
        id=ARM_INS_IT,
        bytes=b"\x08\xbf",
        operands=(),
    )
    branch = SimpleNamespace(
        address=0x1002,
        size=2,
        groups=(CS_GRP_JUMP,),
        id=0,
        cc=ARM_CC_AL,
        operands=(SimpleNamespace(type=CS_OP_IMM, imm=0x2000),),
    )

    assert proven_unconditional_direct_target("ARMCortexM", [it, branch], 1) is None


def test_thumb_expired_it_context_allows_an_unconditional_branch_proof() -> None:
    """Do not extend Thumb IT predication beyond its encoded instruction span."""

    it = SimpleNamespace(
        address=0x1000,
        size=2,
        groups=(),
        id=ARM_INS_IT,
        bytes=b"\x24\xbf",
        operands=(),
    )
    predicated = SimpleNamespace(
        address=0x1002,
        size=2,
        groups=(),
        id=0,
        operands=(),
    )
    another_predicated = SimpleNamespace(
        address=0x1004,
        size=2,
        groups=(),
        id=0,
        operands=(),
    )
    branch = SimpleNamespace(
        address=0x1006,
        size=2,
        groups=(CS_GRP_JUMP,),
        id=0,
        cc=ARM_CC_AL,
        operands=(SimpleNamespace(type=CS_OP_IMM, imm=0x2000),),
    )

    assert (
        proven_unconditional_direct_target(
            "ARMCortexM", [it, predicated, another_predicated, branch], 3
        )
        == 0x2000
    )


def test_xbegin_to_next_instruction_remains_a_cfg_terminator() -> None:
    """Do not merge across transactional-abort control flow."""

    capstone = Cs(CS_ARCH_X86, CS_MODE_64)
    capstone.detail = True
    insn = next(capstone.disasm(b"\xc7\xf8\x00\x00\x00\x00", 0x1000))

    assert insn.mnemonic == "xbegin"
    assert insn.address + insn.size == 0x1006
    assert InsnSemantics(insn).direct_target() == 0x1006
    assert InsnSemantics(insn).is_control_transfer()


def test_lenient_transfer_lookup_accepts_cfgfast_linearized_seed_blocks() -> None:
    """Allow anomaly checks to inspect a seed block with an interior transfer."""

    branch = SimpleNamespace(
        address=0x1000,
        size=4,
        groups=(CS_GRP_JUMP,),
        id=0,
        operands=(SimpleNamespace(type=CS_OP_IMM, imm=0x1004),),
    )
    trailing = SimpleNamespace(
        address=0x1004,
        size=4,
        groups=(),
        id=0,
        operands=(),
    )

    with pytest.raises(RuntimeError, match="trailing instructions"):
        control_transfer_index("PPC32", [branch, trailing])

    assert control_transfer_index("PPC32", [branch, trailing], strict=False) == 0


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
