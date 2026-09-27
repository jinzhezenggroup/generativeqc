"""Exercise the actual default-selection guards without native SCF execution."""

from __future__ import annotations

import ast
import typing
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[2]
NATIVE = SimpleNamespace(
    METHOD_RCCSD=1,
    METHOD_RCCSD_T=2,
    METHOD_MP2=3,
    METHOD_RHF=4,
    DENSITY_FITTING_NONE=0,
)
SUPPORTED = frozenset({"energy", "forces"})


@pytest.fixture(params=("singlepoint", "batch"))
def selector(request: pytest.FixtureRequest) -> typing.Any:
    batched = request.param == "batch"
    filename, owner, method = (
        ("batch.py", "PreparedBatch", "execute")
        if batched
        else ("calculator.py", "Calculator", "singlepoint")
    )
    tree = ast.parse(
        (ROOT / "python/vibeqc" / filename).read_text(encoding="utf-8")
    )
    cls = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == owner
    )
    function = next(
        node
        for node in cls.body
        if isinstance(node, ast.FunctionDef) and node.name == method
    )
    guard = next(
        node
        for node in function.body
        if isinstance(node, ast.If) and ast.unparse(node.test) == "properties is None"
    )
    wrapper = ast.parse("def select(self, properties):\n    return properties\n")
    wrapper.body[0].body.insert(0, guard)
    namespace = {"_native": NATIVE}
    exec(compile(ast.fix_missing_locations(wrapper), filename, "exec"), namespace)
    return batched, namespace["select"]


def _owner(batched: bool, method: int, *, fitted: bool = False) -> SimpleNamespace:
    calculator = SimpleNamespace(
        _method=method,
        _density_fitting_mode=int(fitted),
        _capabilities=SimpleNamespace(supported_properties=SUPPORTED),
    )
    return SimpleNamespace(_calculator=calculator) if batched else calculator


def test_rccsd_omitted_properties_remain_energy_only(selector: typing.Any) -> None:
    batched, select = selector
    assert select(_owner(batched, NATIVE.METHOD_RCCSD), None) == {"energy"}


@pytest.mark.parametrize("properties", (("energy",), ("energy", "forces")))
def test_explicit_rccsd_requests_are_not_replaced(
    selector: typing.Any, properties: tuple[str, ...]
) -> None:
    batched, select = selector
    assert select(_owner(batched, NATIVE.METHOD_RCCSD), properties) is properties


@pytest.mark.parametrize("method", (NATIVE.METHOD_RHF, NATIVE.METHOD_RCCSD_T))
def test_other_method_defaults_remain_unchanged(
    selector: typing.Any, method: int
) -> None:
    batched, select = selector
    assert select(_owner(batched, method), None) is SUPPORTED


@pytest.mark.parametrize("fitted", (False, True))
def test_existing_mp2_default_policy_is_preserved(
    selector: typing.Any, fitted: bool
) -> None:
    batched, select = selector
    expected = {"energy"} if fitted and not batched else SUPPORTED
    assert select(_owner(batched, NATIVE.METHOD_MP2, fitted=fitted), None) == expected
