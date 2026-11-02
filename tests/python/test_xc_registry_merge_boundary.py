"""Execute registry dispatch boundaries without a native library or GPU."""

from __future__ import annotations

import typing

import pytest
from generativeqc import ks
from generativeqc_compiler.xc.spec import UnsupportedXC

# Native automatic-selector namespace plus the pinned Libxc GGA_X_APBE ID.
APBE_CODE = 0x30000 | 184


def _unexpected(*args: typing.Any, **kwargs: typing.Any) -> typing.NoReturn:
    raise AssertionError("dispatch reached an unrelated lowerer")


def test_automatic_code_precedes_curated_and_split_dispatch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(ks, "_is_pbe_d4_composition", lambda value: False)
    monkeypatch.setattr(ks, "_automatic_semilocal_name", lambda value: "GGA_X_APBE")
    monkeypatch.setattr(ks, "_split_hybrid_record", _unexpected)
    monkeypatch.setattr(ks, "_native_semilocal_record", _unexpected)

    assert ks._native_semilocal_code(object()) == APBE_CODE


@pytest.mark.parametrize(
    "selector", ("libxc:GGA_X_APBE", "libxc-rks:GGA_X_APBE", "libxc-uks:GGA_X_APBE")
)
def test_automatic_public_selector_preserves_atomic_recipe(selector: str) -> None:
    # Public aliases now share ordinary resolution, but must retain precisely
    # the requested component rather than adding a curated X/C partner.
    method, _ = ks.resolve_ks_method(selector)
    assert method.primitives[0].functional.components == (("GGA_X_APBE", 1),)
    assert ks.native_xc_functional_code(selector) == APBE_CODE
    assert ks._native_semilocal_code(method) == APBE_CODE


@pytest.mark.parametrize(
    ("selector", "error", "message"),
    (
        ("libxc:", ValueError, "requires a functional registration name"),
        (
            "libxc:NOT_A_FUNCTIONAL",
            UnsupportedXC,
            "not an automatic semilocal registration",
        ),
    ),
)
def test_invalid_automatic_selector_fails_before_native_lowering(
    monkeypatch: pytest.MonkeyPatch,
    selector: str,
    error: type[Exception],
    message: str,
) -> None:
    monkeypatch.setattr(ks, "_native_semilocal_code", _unexpected)

    with pytest.raises(error, match=message):
        ks.native_xc_functional_code(selector)


@pytest.mark.parametrize("automatic", (False, True))
def test_d4_domain_is_selected_from_the_electronic_projection(
    monkeypatch: pytest.MonkeyPatch, automatic: bool
) -> None:
    composite, electronic = object(), object()

    def project(value: object) -> object:
        assert value is composite
        return electronic

    def automatic_name(value: object) -> str | None:
        assert value is electronic
        return "GGA_X_APBE" if automatic else None

    def record(value: object) -> dict[str, str]:
        assert value is electronic
        return {"scf_domain": "curated-domain"}

    monkeypatch.setattr(ks, "_d4_electronic_projection", project)
    monkeypatch.setattr(ks, "_automatic_semilocal_name", automatic_name)
    monkeypatch.setattr(ks, "_is_pbe_d4_composition", lambda value: False)
    monkeypatch.setattr(ks, "_split_hybrid_record", lambda value: None)
    monkeypatch.setattr(ks, "_native_semilocal_record", record)
    monkeypatch.setattr(ks, "automatic_scf_domain", lambda name: "automatic-domain")

    assert ks._scf_domain_for_ir(composite) == (
        "automatic-domain" if automatic else "curated-domain"
    )


def test_legacy_pbe_d4_keeps_its_intrinsic_owner_dispatch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(ks, "_is_pbe_d4_composition", lambda value: True)
    monkeypatch.setattr(ks, "_automatic_semilocal_name", _unexpected)
    monkeypatch.setattr(ks, "_split_hybrid_record", _unexpected)
    monkeypatch.setattr(ks, "_native_semilocal_record", lambda value: {"code": 1})

    assert ks._native_semilocal_code(object()) == 1
