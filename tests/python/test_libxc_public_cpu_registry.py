from __future__ import annotations

from typing import TYPE_CHECKING

from tools import generate_libxc_public_cpu_registry as registry

if TYPE_CHECKING:
    import pytest


def test_empty_public_registry_is_fail_closed(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(registry, "PUBLIC_EVIDENCE", {})
    rendered = registry.render_registry()
    assert "bulk_public::find" in rendered
    assert "return nullptr;" in rendered
    assert "point_program()" not in rendered
