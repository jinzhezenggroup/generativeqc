"""Do not infer native route-dependent work from the WB97M-V method name."""

import ast
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]


def _work_fields() -> dict[str, ast.expr]:
    module = ast.parse(
        (ROOT / "python/generativeqc/_stationary_wb97mv_cuda.py").read_text()
    )
    owner = next(
        node
        for node in module.body
        if isinstance(node, ast.ClassDef) and node.name == "PreparedWb97mvCudaGradient"
    )
    execute = next(
        node
        for node in owner.body
        if isinstance(node, ast.FunctionDef) and node.name == "_execute"
    )
    work = next(
        node.value
        for node in execute.body
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id == "work"
            for target in node.targets
        )
    )
    assert isinstance(work, ast.Dict)
    return {
        key.value: value
        for key, value in zip(work.keys, work.values, strict=True)
        if isinstance(key, ast.Constant) and isinstance(key.value, str)
    }


@pytest.mark.parametrize(
    "field",
    [
        "symmetry_unique_quartets_per_integral_source",
        "two_electron_quartet_traversals",
        "maximum_center_dual3_evaluations_total",
        "two_electron_shell_traversals",
        "range_recurrences_per_participating_center",
    ],
)
def test_unexported_native_route_does_not_become_measured_work(field: str) -> None:
    # The v1 native ABI exposes resources, not the selected derivative route.
    # A shell-capable owner and a successful public-AO fallback must therefore
    # both remain unmeasured here until native execution evidence is exported.
    assert ast.literal_eval(_work_fields()[field]) is None


def test_unavailable_route_is_disclosed_without_losing_radial_identity() -> None:
    fields = _work_fields()
    scope = ast.literal_eval(fields["two_electron_work_scope"])
    assert "unavailable" in scope
    assert "native execution route is not exported" in scope
    assert "public-AO capability fallback" in scope
    assert ast.literal_eval(fields["two_electron_radial_operators"]) == [
        "full-range",
        "short-range",
        "long-range",
    ]
