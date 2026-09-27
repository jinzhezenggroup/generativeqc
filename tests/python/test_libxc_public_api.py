"""Public API tests for retained Libxc functional discovery."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

import vibeqc.libxc as public


def test_public_inventory_is_discoverable_without_native_loading(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(public, "available_public_functionals", lambda: ("GGA_X_APBE",))
    assert public.available_libxc_functionals() == ("GGA_X_APBE",)


def test_public_resolution_consumes_retained_evidence(monkeypatch: pytest.MonkeyPatch) -> None:
    seen = {}
    evidence = {"public-method": {"stage": "public-method"}}
    expected = SimpleNamespace(method="resolved")
    monkeypatch.setattr(public, "public_evidence", lambda name: evidence)

    def resolve(name: str, **kwargs: object) -> object:
        seen["name"] = name
        seen.update(kwargs)
        return expected

    monkeypatch.setattr(public, "resolve_public_bulk_ks", resolve)
    assert public.resolve_libxc_functional("gga_x_apbe", spin="polarized") is expected
    assert seen == {
        "name": "gga_x_apbe",
        "spin": "polarized",
        "backend": "cpu",
        "evidence": evidence,
        "identifier": "LIBXC:GGA_X_APBE",
    }


def test_public_resolution_rejects_unknown_spin_before_evidence(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        public,
        "public_evidence",
        lambda name: pytest.fail("invalid spin reached evidence lookup"),
    )
    with pytest.raises(ValueError, match="Libxc spin"):
        public.resolve_libxc_functional("GGA_X_APBE", spin="restricted")
