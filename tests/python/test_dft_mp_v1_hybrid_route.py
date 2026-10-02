"""Source-only checks for the force route audited by the FP64 capacity report.

These exercise the real Calculator predicate and MethodIR eligibility helper;
they do not load a native library or claim AUTO numerical qualification.
"""

import ast
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
from generativeqc import _generated_methods as method_manifest
from generativeqc import _native
from generativeqc.ks import (
    SPLIT_HYBRID_SCF_DOMAIN,
    _scf_domain_for_ir,
    cuda_global_hybrid_force_eligible,
    ks_coefficients,
)
from generativeqc_compiler.method import compile_ks_execution_plan, resolve_method

from tools.dft_mp_v1 import qualify_capacity

ROOT = Path(__file__).resolve().parents[2]


def _nodes() -> tuple[str, dict[str, ast.Assign], ast.If]:
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
    predicate_names = {
        "semilocal_force",
        "density_fitted_force",
        "cuda_hybrid_force",
    }
    assignments = {
        target.id: node
        for node in ast.walk(constructor)
        if isinstance(node, ast.Assign)
        for target in node.targets
        if isinstance(target, ast.Name) and target.id in predicate_names
    }
    promotions = [
        node
        for node in constructor.body
        if isinstance(node, ast.If)
        and any(
            isinstance(item, ast.Name) and item.id == "semilocal_force"
            for item in ast.walk(node.test)
        )
    ]
    assert set(assignments) == predicate_names and len(promotions) == 1
    return source, assignments, promotions[0]


def _promoted(
    *,
    precision: int,
    method: str = "PBE0",
    spin: str = "unpolarized",
    blocked: str | None = None,
    renamed: bool = False,
    density_fitting: bool = False,
    include_semilocal: bool = False,
    device: str = "cuda",
) -> bool:
    _, assignments, promotion = _nodes()
    options = SimpleNamespace(
        xc_schedule="device_fused",
        method_ir=resolve_method(method, spin=spin),
    )
    if renamed:
        options.method_ir = replace(options.method_ir, identifier="opaque-hybrid-alias")
    options.execution_plan = compile_ks_execution_plan(options.method_ir)
    if include_semilocal:
        options.coefficients = ks_coefficients(options.method_ir)
    # Resolve the actual native domain instead of inferring it from a label.
    # Ineligible graphs must short-circuit without querying their native domain;
    # CAM-B3LYP deliberately has no native semilocal lowerer.
    options.scf_domain = (
        _scf_domain_for_ir(options.method_ir)
        if cuda_global_hybrid_force_eligible(options.method_ir)
        else None
    )
    owner = SimpleNamespace(
        _device_name=device,
        _precision_mode=precision,
        _ks_options=options,
        _capabilities=SimpleNamespace(family="density_functional"),
        _method=(
            _native.METHOD_PBE_UKS if spin == "polarized" else _native.METHOD_PBE_RKS
        ),
        _basis="sto-3g",
        _method_name=method.lower(),
        _automatic_libxc_name=None,
        _dispersion_method_ir=None,
    )
    scope = {
        "self": owner,
        "basis_has_ecp": blocked in ("ecp", "unqualified-ecp"),
        "density_fitting_mode": (
            _native.DENSITY_FITTING_AUTO
            if density_fitting
            else _native.DENSITY_FITTING_NONE
        ),
        "semilocal_force": False,
        "named_cpu_all_electron_force": False,
        "cuda_wb97mv_force": False,
        "qualified_basis": lambda basis: blocked != "unqualified-ecp",
        "cuda_global_hybrid_force_eligible": cuda_global_hybrid_force_eligible,
        "SPLIT_HYBRID_SCF_DOMAIN": SPLIT_HYBRID_SCF_DOMAIN,
        "_native": _native,
        "_method_manifest": method_manifest,
    }
    if blocked == "cpu":
        owner._device_name = "cpu"
    elif blocked == "missing-options":
        owner._ks_options = None
    elif blocked == "reference-xc":
        options.xc_schedule = "reference"
    elif blocked == "wrong-family":
        owner._capabilities.family = "hartree_fock"
    elif blocked == "unregistered-method":
        owner._method = -1
    elif blocked == "automatic-libxc":
        owner._automatic_libxc_name = "unqualified"
    elif blocked == "unsupported-device":
        owner._device_name = "unsupported"
    elif blocked == "dispersion":
        owner._dispersion_method_ir = object()
    elif blocked in ("exchange", "nonlocal"):
        options.execution_plan = SimpleNamespace(
            exchange=(SimpleNamespace(operator="short-range"),)
            if blocked == "exchange"
            else (),
            nonlocal_correlation=object() if blocked == "nonlocal" else None,
            post_scf=(),
        )
    predicates = (
        [("semilocal_force", assignments["semilocal_force"].value)]
        if include_semilocal
        else []
    ) + [
        ("density_fitted_force", assignments["density_fitted_force"].value),
        ("cuda_hybrid_force", assignments["cuda_hybrid_force"].value),
        ("promoted", promotion.test),
    ]
    for name, expression in predicates:
        scope[name] = eval(  # noqa: S307 - execute only the trusted repository predicate
            compile(ast.Expression(expression), "<Calculator force route>", "eval"),
            {"__builtins__": {"all": all}},
            scope,
        )
    return scope["promoted"]


@pytest.mark.parametrize("precision", (_native.PRECISION_FP64, _native.PRECISION_AUTO))
@pytest.mark.parametrize("spin", ("unpolarized", "polarized"))
@pytest.mark.parametrize("method", ("PBE0", "B3LYP"))
def test_qualified_non_split_hybrid_route_admits_both_precision_requests(
    precision: int, spin: str, method: str
) -> None:
    assert _promoted(precision=precision, spin=spin, method=method)


@pytest.mark.parametrize("spin", ("unpolarized", "polarized"))
@pytest.mark.parametrize("method", ("M06-2X", "MN15"))
def test_generated_split_hybrid_force_keeps_auto_fail_closed(
    spin: str, method: str
) -> None:
    assert _promoted(precision=_native.PRECISION_FP64, spin=spin, method=method)
    assert not _promoted(precision=_native.PRECISION_AUTO, spin=spin, method=method)


@pytest.mark.parametrize("spin", ("unpolarized", "polarized"))
@pytest.mark.parametrize("method", ("PBE0", "B3LYP", "M06-2X", "MN15"))
def test_auto_force_admission_uses_resolved_domain_under_renaming(
    spin: str, method: str
) -> None:
    assert _promoted(
        precision=_native.PRECISION_AUTO, spin=spin, method=method, renamed=True
    ) is (method in ("PBE0", "B3LYP"))


@pytest.mark.parametrize("precision", (_native.PRECISION_FP64, _native.PRECISION_AUTO))
@pytest.mark.parametrize(
    "blocked",
    (
        "cpu",
        "ecp",
        "missing-options",
        "reference-xc",
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


@pytest.mark.parametrize("precision", (_native.PRECISION_FP64, _native.PRECISION_AUTO))
@pytest.mark.parametrize("spin", ("unpolarized", "polarized"))
@pytest.mark.parametrize("method", ("LDA_XC_PW", "PBE", "R2SCAN"))
@pytest.mark.parametrize("density_fitting", (False, True))
def test_semilocal_force_precision_boundary_is_separate_from_hybrid_admission(
    precision: int, spin: str, method: str, density_fitting: bool
) -> None:
    assert _promoted(
        precision=precision,
        spin=spin,
        method=method,
        density_fitting=density_fitting,
        include_semilocal=True,
    ) is (not density_fitting or precision == _native.PRECISION_FP64)


@pytest.mark.parametrize("precision", (_native.PRECISION_FP64, _native.PRECISION_AUTO))
@pytest.mark.parametrize("spin", ("unpolarized", "polarized"))
@pytest.mark.parametrize("method", ("PBE0", "B3LYP", "M06-2X", "MN15", "WB97M-V"))
@pytest.mark.parametrize("device", ("cpu", "cuda"))
def test_density_fitted_force_uses_own_strict_precision_admission(
    precision: int, spin: str, method: str, device: str
) -> None:
    # This executes capability admission only; the native final-state owner
    # independently validates the particular fitted functional composition.
    assert _promoted(
        precision=precision,
        spin=spin,
        method=method,
        density_fitting=True,
        include_semilocal=True,
        device=device,
    ) is (precision == _native.PRECISION_FP64 and method != "WB97M-V")


@pytest.mark.parametrize(
    "blocked",
    (
        "unsupported-device",
        "missing-options",
        "wrong-family",
        "unregistered-method",
        "automatic-libxc",
        "unqualified-ecp",
        "exchange",
        "nonlocal",
        "dispersion",
    ),
)
def test_df_force_preserves_nonprecision_boundaries(blocked: str) -> None:
    assert not _promoted(
        precision=_native.PRECISION_FP64,
        method="PBE",
        blocked=blocked,
        density_fitting=True,
        include_semilocal=True,
    )


@pytest.mark.parametrize(
    "guard",
    (
        'self._device_name == "cuda"',
        "not basis_has_ecp",
        "self._ks_options is not None",
        'self._ks_options.xc_schedule == "device_fused"',
        "cuda_global_hybrid_force_eligible(self._ks_options.method_ir)",
        "self._precision_mode == _native.PRECISION_FP64",
        "self._ks_options.scf_domain != SPLIT_HYBRID_SCF_DOMAIN",
    ),
)
def test_capacity_audit_rejects_a_changed_hybrid_guard(
    tmp_path: Path, guard: str
) -> None:
    source, assignments, _ = _nodes()
    assignment = assignments["cuda_hybrid_force"]
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
