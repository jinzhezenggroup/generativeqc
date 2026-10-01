"""Source-only checks for the force route audited by the FP64 capacity report.

These exercise the real Calculator predicate and MethodIR eligibility helper;
they do not load a native library or claim AUTO numerical qualification.
"""

import ast
from pathlib import Path
from types import SimpleNamespace

import pytest
from generativeqc import _generated_methods as method_manifest
from generativeqc import _native
from generativeqc.ks import cuda_global_hybrid_force_eligible
from generativeqc_compiler.method import resolve_method

from tools.dft_mp_v1 import qualify_capacity

ROOT = Path(__file__).resolve().parents[2]


def _nodes() -> tuple[str, ast.Assign, ast.If]:
    source = (ROOT / "python/generativeqc/calculator.py").read_text(encoding="utf-8")
    owner = next(
        node
        for node in ast.parse(source).body
        if isinstance(node, ast.ClassDef) and node.name == "Calculator"
    )
    constructor = next(
        node
        for node in owner.body
        if isinstance(node, ast.FunctionDef) and node.name == "__init__"
    )
    assignments = [
        node
        for node in ast.walk(constructor)
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id == "cuda_hybrid_force"
            for target in node.targets
        )
    ]
    promotions = [
        node
        for node in constructor.body
        if isinstance(node, ast.If)
        and any(
            isinstance(item, ast.Name) and item.id == "semilocal_force"
            for item in ast.walk(node.test)
        )
    ]
    assert len(assignments) == len(promotions) == 1
    return source, assignments[0], promotions[0]


def _promoted(
    *,
    precision: int,
    method: str = "PBE0",
    spin: str = "unpolarized",
    blocked: str | None = None,
) -> bool:
    _, assignment, promotion = _nodes()
    options = SimpleNamespace(
        xc_schedule="device_fused", method_ir=resolve_method(method, spin=spin)
    )
    owner = SimpleNamespace(
        _device_name="cuda",
        _precision_mode=precision,
        _ks_options=options,
        _capabilities=SimpleNamespace(family="density_functional"),
        _method=(
            _native.METHOD_PBE_UKS if spin == "polarized" else _native.METHOD_PBE_RKS
        ),
        _basis="sto-3g",
    )
    scope = {
        "self": owner,
        "basis_has_ecp": blocked == "ecp",
        "density_fitting_mode": _native.DENSITY_FITTING_NONE,
        "semilocal_force": False,
        "named_cpu_all_electron_force": False,
        "cuda_wb97mv_force": False,
        "qualified_basis": lambda basis: True,
        "cuda_global_hybrid_force_eligible": cuda_global_hybrid_force_eligible,
        "_native": _native,
        "_method_manifest": method_manifest,
    }
    if blocked == "cpu":
        owner._device_name = "cpu"
    elif blocked == "missing-options":
        owner._ks_options = None
    elif blocked == "reference-xc":
        options.xc_schedule = "reference"
    elif blocked == "density-fitting":
        scope["density_fitting_mode"] = _native.DENSITY_FITTING_NONE + 1
    elif blocked == "wrong-family":
        owner._capabilities.family = "hartree_fock"
    elif blocked == "unregistered-method":
        owner._method = -1
    for name, expression in (
        ("cuda_hybrid_force", assignment.value),
        ("promoted", promotion.test),
    ):
        scope[name] = eval(  # noqa: S307 - execute only the trusted repository predicate
            compile(ast.Expression(expression), "<Calculator force route>", "eval"),
            {"__builtins__": {}},
            scope,
        )
    return scope["promoted"]


@pytest.mark.parametrize("precision", (_native.PRECISION_FP64, _native.PRECISION_AUTO))
@pytest.mark.parametrize("spin", ("unpolarized", "polarized"))
@pytest.mark.parametrize("method", ("PBE0", "B3LYP", "M06-2X", "MN15"))
def test_audited_hybrid_route_admits_both_precision_requests(
    precision: int, spin: str, method: str
) -> None:
    assert _promoted(precision=precision, spin=spin, method=method)


@pytest.mark.parametrize("precision", (_native.PRECISION_FP64, _native.PRECISION_AUTO))
@pytest.mark.parametrize(
    "blocked",
    (
        "cpu",
        "ecp",
        "missing-options",
        "reference-xc",
        "density-fitting",
        "wrong-family",
        "unregistered-method",
    ),
)
def test_auto_force_route_preserves_nonprecision_boundaries(
    precision: int, blocked: str
) -> None:
    assert not _promoted(precision=precision, blocked=blocked)


@pytest.mark.parametrize("precision", (_native.PRECISION_FP64, _native.PRECISION_AUTO))
@pytest.mark.parametrize("method", ("PBE", "CAM-B3LYP", "WB97M-V"))
def test_hybrid_promotion_does_not_admit_other_source_contracts(
    precision: int, method: str
) -> None:
    assert not _promoted(precision=precision, method=method)


@pytest.mark.parametrize(
    "guard",
    (
        'self._device_name == "cuda"',
        "not basis_has_ecp",
        "self._ks_options is not None",
        'self._ks_options.xc_schedule == "device_fused"',
        "cuda_global_hybrid_force_eligible(self._ks_options.method_ir)",
    ),
)
def test_capacity_audit_rejects_a_changed_hybrid_guard(
    tmp_path: Path, guard: str
) -> None:
    source, assignment, _ = _nodes()
    segment = ast.get_source_segment(source, assignment)
    assert segment is not None and segment.count(guard) == 1
    for name in ("calculator.py", "batch.py"):
        target = tmp_path / "python/generativeqc" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        text = (ROOT / "python/generativeqc" / name).read_text(encoding="utf-8")
        if name == "calculator.py":
            text = text.replace(segment, segment.replace(guard, "True", 1), 1)
        target.write_text(text, encoding="utf-8")
    with pytest.raises(
        RuntimeError, match="public global-hybrid force predicate changed"
    ):
        qualify_capacity._source_public_route(tmp_path)
