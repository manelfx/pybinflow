from __future__ import annotations

from fastapi.testclient import TestClient

import bingraph.api.app as app_module
import bingraph.helpers.settings as settings_module
from bingraph.helpers import Settings


def test_cfg_api_exits_query_overrides_the_default(monkeypatch, tmp_path) -> None:
    """Pass the request exit-display policy through to the renderer."""

    binary = tmp_path / "binary"
    binary.touch()
    settings = Settings.model_construct(
        root=tmp_path,
        cfg_mode="custom",
        cfg_exits="jump",
        comments=False,
        dfs_rank=False,
        log_level="INFO",
        debug=False,
        server=None,
        client=None,
    )
    monkeypatch.setattr(settings_module, "_settings", settings)
    monkeypatch.setattr(app_module, "load_project", lambda _: object())

    rendered_calls: list[tuple[object, ...]] = []

    def render_cfg(*args: object) -> str:
        rendered_calls.append(args)
        return "digraph G {}"

    monkeypatch.setattr(app_module, "render_cfg", render_cfg)
    client = TestClient(app_module.create_app())

    response = client.get(
        "/api/cfg",
        params={
            "filepath": binary.name,
            "function": "0x10",
            "format": "raw",
            "exits": "always",
        },
    )

    assert response.status_code == 200
    assert response.json() == {"graph": "digraph G {}"}
    assert [args[5] for args in rendered_calls] == ["always"]
