"""Fast tests for Capstone-backed CFG node inspection."""

from types import SimpleNamespace

from bingraph.cfg.decode import DecodedNode


def _insn(addr: int, size: int) -> SimpleNamespace:
    """Create the minimal instruction shape required by ``DecodedNode``."""

    return SimpleNamespace(address=addr, size=size)


def test_decoded_node_reports_exact_instruction_coverage() -> None:
    """Accept a contiguous instruction stream covering the complete node span."""

    decoded = DecodedNode((_insn(0x1000, 2), _insn(0x1002, 3)))
    node = SimpleNamespace(addr=0x1000, size=5)

    assert decoded.has_exact_coverage(node)
    assert not decoded.contains_mid_instruction_addr(0x1002)
    assert decoded.contains_mid_instruction_addr(0x1003)


def test_decoded_node_rejects_a_gap_or_trailing_bytes() -> None:
    """Reject instruction streams that do not match the declared node range."""

    node = SimpleNamespace(addr=0x1000, size=5)

    assert not DecodedNode((_insn(0x1000, 2), _insn(0x1003, 2))).has_exact_coverage(
        node
    )
    assert not DecodedNode((_insn(0x1000, 2),)).has_exact_coverage(node)


def test_decoded_node_handles_missing_capstone_inspection() -> None:
    """Treat a missing Capstone inspection as non-empty-coverage failure."""

    decoded = DecodedNode(None, KeyError("block"))

    assert decoded.is_empty
    assert decoded.last is None
    assert not decoded.has_exact_coverage(SimpleNamespace(addr=0x1000, size=1))
