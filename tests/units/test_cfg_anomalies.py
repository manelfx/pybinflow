"""Fast tests for custom-CFG anomaly classification and reporting."""

from types import SimpleNamespace

from loguru import logger

from bingraph.cfg import anomalies
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
