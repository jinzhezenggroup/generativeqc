"""Source-only checks for the force route audited by the FP64 capacity report.

These exercise the real Calculator predicate and MethodIR eligibility helper;
they do not load a native library or claim AUTO numerical qualification.
"""

import ast
from dataclasses import replace
from fractions import Fraction
from pathlib import Path
from types import MethodType, SimpleNamespace

import pytest
from generativeqc import Calculator, _native
from generativeqc import _generated_methods as method_manifest
from generativeqc._cpu_force_resources import qualified_direct_semilocal_context
from generativeqc._df_precision import fitted_scf_budget_admitted
from generativeqc._model_resolution import snapshot_basis
from generativeqc.ks import (
    SPLIT_HYBRID_SCF_DOMAIN,
    _scf_domain_for_ir,
    cuda_global_hybrid_force_eligible,
    cuda_mixed_density_fitted_force_eligible,
    cuda_nonlocal_force_basis_eligible,
    ks_coefficients,
    mixed_density_fitted_force_eligible,
    split_hybrid_df_force_admitted,
)
from generativeqc_compiler.method import (
    D4Spec,
    DispersionCorrectionPrimitive,
    ExactExchangePrimitive,
    compile_ks_execution_plan,
    resolve_bulk_ks,
    resolve_method,
)

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
        "cpu_direct_semilocal_force",
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
    automatic_name: str | None = None,
    composite: str | None = None,
    automatic_package: bool = False,
    discovery_calls: list[dict] | None = None,
) -> bool:
    _, assignments, promotion = _nodes()
    options = SimpleNamespace(
        xc_schedule="device_fused",
        method_ir=(
            resolve_bulk_ks(automatic_name, spin=spin).method
            if automatic_name is not None
            else resolve_method(method, spin=spin)
        ),
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
        _library=SimpleNamespace(),
        _max_iterations=100,
        _diis_history=8,
        _energy_tolerance=1e-10,
        _density_tolerance=1e-8,
        _screening_tolerance=1e-12,
        _precision_mode=precision,
        _ks_options=options,
        _capabilities=SimpleNamespace(family="density_functional"),
        _method=(
            _native.METHOD_PBE_UKS if spin == "polarized" else _native.METHOD_PBE_RKS
        ),
        _basis=snapshot_basis("sto-3g"),
        _method_name=(
            f"libxc-{'uks' if spin == 'polarized' else 'rks'}:{automatic_name}"
            if automatic_name is not None
            else ("lda" if method == "LDA_XC_PW" else method.lower())
            + ("-uks" if spin == "polarized" else "-rks")
        ),
        _density_fitting_mode=(
            _native.DENSITY_FITTING_AUTO
            if density_fitting
            else _native.DENSITY_FITTING_NONE
        ),
        _automatic_libxc_name=automatic_name,
        _dispersion_method_ir=(
            resolve_method(composite, spin=spin) if composite is not None else None
        ),
    )
    owner._density_fitted_composite_force_eligible = MethodType(
        Calculator._density_fitted_composite_force_eligible, owner
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
        # These force-route probes request no application error budget. Resolve
        # its admission through the real helper; AUTO alone must not grant it.
        "fitted_budget": fitted_scf_budget_admitted(owner, None),
        "cpu_composed_all_electron_force": False,
        "cuda_nonlocal_force": False,
        "qualified_basis": lambda basis: blocked == "ecp",
        "qualified_direct_semilocal_context": qualified_direct_semilocal_context,
        "cuda_global_hybrid_force_eligible": cuda_global_hybrid_force_eligible,
        "cuda_mixed_density_fitted_force_eligible": cuda_mixed_density_fitted_force_eligible,
        "cuda_nonlocal_force_basis_eligible": cuda_nonlocal_force_basis_eligible,
        "mixed_density_fitted_force_eligible": mixed_density_fitted_force_eligible,
        "split_hybrid_df_force_admitted": split_hybrid_df_force_admitted,
        "SPLIT_HYBRID_SCF_DOMAIN": SPLIT_HYBRID_SCF_DOMAIN,
        "D4Spec": D4Spec,
        "DispersionCorrectionPrimitive": DispersionCorrectionPrimitive,
        "_native": _native,
        "_method_manifest": method_manifest,
    }
    if blocked == "cpu":
        owner._device_name = "cpu"
    elif blocked == "missing-options":
        owner._ks_options = None
    elif blocked == "reference-xc":
        options.xc_schedule = "reference"
    elif blocked == "host-unfused":
        options.xc_schedule = "host_unfused"
    elif blocked == "wrong-family":
        owner._capabilities.family = "hartree_fock"
    elif blocked == "unregistered-method":
        owner._method = -1
    elif blocked == "automatic-libxc":
        owner._automatic_libxc_name = "unqualified"
    elif blocked == "unsupported-device":
        owner._device_name = "unsupported"
    elif blocked == "dispersion":
        owner._dispersion_method_ir = resolve_method("R2SCAN-D4(BJ-EEQ-ATM)", spin=spin)
    elif blocked in ("exchange", "nonlocal"):
        options.execution_plan = SimpleNamespace(
            exchange=(SimpleNamespace(operator="short-range"),)
            if blocked == "exchange"
            else (),
            nonlocal_correlation=object() if blocked == "nonlocal" else None,
            post_scf=(),
        )
    # Execute the actual initializer and guarded discovery dependency before
    # evaluating predicates. An absent package is this source-only fixture's
    # explicit default; the guard itself must still enforce backend/precision.
    source = (ROOT / "python/generativeqc/calculator.py").read_text(encoding="utf-8")
    constructor = next(
        node
        for owner in ast.parse(source).body
        if isinstance(owner, ast.ClassDef) and owner.name == "Calculator"
        for node in owner.body
        if isinstance(node, ast.FunctionDef) and node.name == "__init__"
    )
    initializer = next(
        node
        for node in constructor.body
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id == "automatic_cuda_force"
            for target in node.targets
        )
    )
    guard = constructor.body[constructor.body.index(initializer) + 1]
    assert isinstance(guard, ast.If)
    discovery = next(
        node
        for node in guard.body
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id == "automatic_cuda_force"
            for target in node.targets
        )
    )

    def discover(library: object, model: object, **controls: object) -> bool:
        assert library is owner._library and model is owner._ks_options
        if discovery_calls is not None:
            discovery_calls.append(controls)
        return automatic_package

    scope["native_cuda_stationary_model_available"] = discover
    scope["automatic_cuda_force"] = eval(  # noqa: S307
        compile(
            ast.Expression(initializer.value), "<automatic force initializer>", "eval"
        ),
        {"__builtins__": {}, **scope},
    )
    if eval(  # noqa: S307
        compile(ast.Expression(guard.test), "<automatic force guard>", "eval"),
        {"__builtins__": {}, **scope},
    ):
        scope["automatic_cuda_force"] = eval(  # noqa: S307
            compile(
                ast.Expression(discovery.value), "<automatic force discovery>", "eval"
            ),
            {"__builtins__": {}, **scope},
        )
    predicates = (
        [
            (
                "cpu_direct_semilocal_force",
                assignments["cpu_direct_semilocal_force"].value,
            ),
            ("semilocal_force", assignments["semilocal_force"].value),
        ]
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
            {
                "__builtins__": {"all": all, "any": any, "isinstance": isinstance},
                **scope,
            },
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
def test_generated_split_hybrid_force_uses_shared_precision_admission(
    spin: str, method: str
) -> None:
    assert _promoted(precision=_native.PRECISION_FP64, spin=spin, method=method)
    assert _promoted(precision=_native.PRECISION_AUTO, spin=spin, method=method)


@pytest.mark.parametrize("spin", ("unpolarized", "polarized"))
@pytest.mark.parametrize("method", ("PBE0", "B3LYP", "M06-2X", "MN15"))
def test_auto_force_admission_uses_resolved_domain_under_renaming(
    spin: str, method: str
) -> None:
    assert _promoted(
        precision=_native.PRECISION_AUTO, spin=spin, method=method, renamed=True
    )


@pytest.mark.parametrize("precision", (_native.PRECISION_FP64, _native.PRECISION_AUTO))
@pytest.mark.parametrize("method", ("PBE0", "B3LYP", "M06-2X", "MN15"))
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
    precision: int, method: str, blocked: str
) -> None:
    assert not _promoted(precision=precision, method=method, blocked=blocked)


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
    ) is (
        precision == _native.PRECISION_FP64
        and (method not in {"M06-2X", "MN15"} or device == "cuda")
    )


@pytest.mark.parametrize("spin", ("unpolarized", "polarized"))
@pytest.mark.parametrize("backend", ("cpu", "cuda"))
def test_mixed_df_admission_keeps_unqualified_range_graph_closed(
    spin: str, backend: str
) -> None:
    method = resolve_method("CAM-B3LYP", spin=spin)
    assert not mixed_density_fitted_force_eligible(method, backend=backend)


@pytest.mark.parametrize("precision", (_native.PRECISION_FP64, _native.PRECISION_AUTO))
@pytest.mark.parametrize("spin", ("unpolarized", "polarized"))
@pytest.mark.parametrize("device", ("cpu", "cuda"))
@pytest.mark.parametrize("composite", ("R2SCAN-3c", "R2SCAN-D4(BJ-EEQ-ATM)"))
def test_df_composite_force_keeps_its_manifest_and_precision_boundaries(
    precision: int, spin: str, device: str, composite: str
) -> None:
    assert _promoted(
        precision=precision,
        spin=spin,
        method="R2SCAN",
        density_fitting=True,
        include_semilocal=True,
        device=device,
        composite=composite,
    ) is (composite == "R2SCAN-3c" and precision == _native.PRECISION_FP64)


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


@pytest.mark.parametrize("spin", ["unpolarized", "polarized"])
@pytest.mark.parametrize("method", ["LDA_XC_PW", "PBE"])
@pytest.mark.parametrize(
    "blocked",
    [
        None,
        "missing-options",
        "automatic-libxc",
        "nonlocal",
        "dispersion",
        "unsupported-device",
    ],
)
def test_expanded_cpu_branch_uses_its_shared_qualified_context(
    spin: str,
    method: str,
    blocked: str | None,
) -> None:
    assert _promoted(
        precision=_native.PRECISION_FP64,
        spin=spin,
        method=method,
        include_semilocal=True,
        device="cpu",
        blocked=blocked,
    ) is (blocked is None)


@pytest.mark.parametrize("device", ("cpu", "cuda"))
@pytest.mark.parametrize("precision", (_native.PRECISION_FP64, _native.PRECISION_AUTO))
@pytest.mark.parametrize("spin", ("unpolarized", "polarized"))
@pytest.mark.parametrize("name", ("LDA_C_CHACHIYO", "GGA_X_APBE", "MGGA_X_LTA"))
def test_automatic_semilocal_without_linked_package_remains_cpu_fp64_only(
    device: str, precision: int, spin: str, name: str
) -> None:
    assert _promoted(
        precision=precision,
        spin=spin,
        include_semilocal=True,
        device=device,
        automatic_name=name,
    ) is (device == "cpu" and precision == _native.PRECISION_FP64)


@pytest.mark.parametrize("spin", ("unpolarized", "polarized"))
def test_split_hybrid_df_source_qualifies_exact_graph_not_method_name(
    spin: str,
) -> None:
    method = resolve_method("M06-2X", spin=spin)
    renamed = replace(method, identifier="opaque-qualified-split")
    assert split_hybrid_df_force_admitted(
        renamed, backend="cuda", xc_schedule="device_fused"
    )
    assert not split_hybrid_df_force_admitted(
        renamed, backend="cpu", xc_schedule="device_fused"
    )
    assert not split_hybrid_df_force_admitted(
        renamed, backend="cuda", xc_schedule="host_unfused"
    )
    wrong_fraction = replace(
        renamed,
        primitives=tuple(
            replace(node, coefficient=Fraction(3, 7))
            if isinstance(node, ExactExchangePrimitive)
            else node
            for node in renamed.primitives
        ),
    )
    assert wrong_fraction != renamed
    assert not split_hybrid_df_force_admitted(
        wrong_fraction, backend="cuda", xc_schedule="device_fused"
    )
    assert split_hybrid_df_force_admitted(
        resolve_method("MN15", spin=spin),
        backend="cuda",
        xc_schedule="device_fused",
    )


@pytest.mark.parametrize("available", [False, True])
@pytest.mark.parametrize("spin", ["unpolarized", "polarized"])
def test_automatic_cuda_force_depends_on_native_package(
    available: bool, spin: str
) -> None:
    calls = []
    assert (
        _promoted(
            precision=_native.PRECISION_FP64,
            device="cuda",
            spin=spin,
            automatic_name="LDA_C_CHACHIYO",
            include_semilocal=True,
            automatic_package=available,
            discovery_calls=calls,
        )
        is available
    )
    assert len(calls) == 1
    assert calls[0]["precision"] == "fp64"
    assert calls[0]["max_iterations"] == 100


@pytest.mark.parametrize("case", ["cpu", "auto", "df", "host-unfused", "ecp"])
def test_automatic_force_discovery_preserves_native_guard(case: str) -> None:
    calls = []
    _promoted(
        precision=_native.PRECISION_AUTO if case == "auto" else _native.PRECISION_FP64,
        device="cpu" if case == "cpu" else "cuda",
        automatic_name="LDA_C_CHACHIYO",
        include_semilocal=True,
        density_fitting=case == "df",
        blocked="host-unfused"
        if case == "host-unfused"
        else "unqualified-ecp"
        if case == "ecp"
        else None,
        automatic_package=True,
        discovery_calls=calls,
    )
    assert calls == []
