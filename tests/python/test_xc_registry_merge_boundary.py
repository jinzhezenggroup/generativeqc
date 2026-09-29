"""Execute registry dispatch boundaries without a native library or GPU."""

from __future__ import annotations

import ast
import typing
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]


def _function(name: str, namespace: dict[str, typing.Any]) -> typing.Callable:
    path = ROOT / "python/generativeqc/ks.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    node = next(
        item
        for item in tree.body
        if isinstance(item, ast.FunctionDef) and item.name == name
    )
    scope = {"typing": typing, **namespace}
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), "exec"), scope)  # noqa: S102
    return scope[name]


def _unexpected(*args: typing.Any, **kwargs: typing.Any) -> typing.NoReturn:
    raise AssertionError("dispatch reached an unrelated lowerer")


def test_automatic_code_precedes_curated_and_split_dispatch() -> None:
    method = object()
    code = _function(
        "_native_semilocal_code",
        {
            "_is_pbe_d4_composition": lambda value: False,
            "_automatic_semilocal_name": lambda value: "LDA_X",
            "automatic_functional_code": lambda name: 0x30001,
            "_split_hybrid_record": _unexpected,
            "_native_semilocal_record": _unexpected,
        },
    )
    assert code(method) == 0x30001


def test_automatic_public_selector_does_not_enter_named_resolution() -> None:
    code = _function(
        "native_xc_functional_code",
        {
            "parse_automatic_libxc_selector": lambda value: ("LDA_X", "unpolarized"),
            "automatic_functional_code": lambda name: 0x30001,
            "resolve_ks_method": _unexpected,
        },
    )
    assert code("libxc:LDA_X") == 0x30001


@pytest.mark.parametrize("automatic", (False, True))
def test_d4_domain_is_selected_from_the_electronic_projection(automatic: bool) -> None:
    composite, electronic = object(), object()

    def project(value: object) -> object:
        assert value is composite
        return electronic

    def automatic_name(value: object) -> str | None:
        assert value is electronic
        return "LDA_X" if automatic else None

    def record(value: object) -> dict[str, str]:
        assert value is electronic
        return {"scf_domain": "curated-domain"}

    domain = _function(
        "_scf_domain_for_ir",
        {
            "_d4_electronic_projection": project,
            "_automatic_semilocal_name": automatic_name,
            "_is_pbe_d4_composition": lambda value: False,
            "_split_hybrid_record": lambda value: None,
            "_native_semilocal_record": record,
            "AUTOMATIC_SCF_DOMAIN": "automatic-domain",
        },
    )
    assert domain(composite) == ("automatic-domain" if automatic else "curated-domain")


def test_legacy_pbe_d4_keeps_its_intrinsic_owner_dispatch() -> None:
    code = _function(
        "_native_semilocal_code",
        {
            "_is_pbe_d4_composition": lambda value: True,
            "_automatic_semilocal_name": _unexpected,
            "_split_hybrid_record": _unexpected,
            "_native_semilocal_record": lambda value: {"code": 1},
        },
    )
    assert code(object()) == 1
