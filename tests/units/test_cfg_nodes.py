"""Unit tests for custom CFG node construction."""

from types import SimpleNamespace

from bingraph.cfg import nodes as nodes_module
from bingraph.cfg.models import BlockSpec


def test_make_cfg_node_preserves_thumb_mode(monkeypatch) -> None:
    """Pass the address-selected Thumb mode to recovered CFG nodes."""

    captured: dict[str, object] = {}

    def make_node(*_args, **kwargs):
        captured.update(kwargs)
        return SimpleNamespace()

    monkeypatch.setattr(nodes_module, "CFGNode", make_node)
    seed_cfg = SimpleNamespace(
        model=object(),
        project=SimpleNamespace(
            arch=SimpleNamespace(is_thumb=lambda addr: addr == 0x1001)
        ),
    )
    bounds = SimpleNamespace(addr=0x1001, name="thumb_function")
    block = BlockSpec(0x1001, 2, (0x1001,), "Ijk_Fallthrough")

    nodes_module.make_cfg_node(seed_cfg, 0x1001, bounds, block)

    assert captured["thumb"] is True


def test_make_cfg_node_defaults_to_non_thumb_without_arch_support(monkeypatch) -> None:
    """Keep the normal CFGNode default when the architecture has no Thumb mode."""

    captured: dict[str, object] = {}

    def make_node(*_args, **kwargs):
        captured.update(kwargs)
        return SimpleNamespace()

    monkeypatch.setattr(nodes_module, "CFGNode", make_node)
    seed_cfg = SimpleNamespace(model=object(), project=SimpleNamespace(arch=object()))
    bounds = SimpleNamespace(addr=0x1000, name="generic_function")
    block = BlockSpec(0x1000, 1, (0x1000,), "Ijk_Fallthrough")

    nodes_module.make_cfg_node(seed_cfg, 0x1000, bounds, block)

    assert captured["thumb"] is False
