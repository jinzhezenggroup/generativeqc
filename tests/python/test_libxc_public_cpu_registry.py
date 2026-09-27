from __future__ import annotations

from tools import generate_libxc_public_cpu_registry as registry


def test_empty_public_registry_is_fail_closed(monkeypatch) -> None:
    monkeypatch.setattr(registry, "PUBLIC_EVIDENCE", {})
    rendered = registry.render_registry()
    assert "bulk_public::find" in rendered
    assert "return nullptr;" in rendered
    assert "point_program()" not in rendered
