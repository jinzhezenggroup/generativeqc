"""Public global-hybrid admission, detached results, failures and cleanup."""

import typing
from dataclasses import replace
from fractions import Fraction
from pathlib import Path

import generativeqc.composite_rks_hessian as composite
import numpy as np
import pytest
from generativeqc import Calculator, GridSpec, KsOptions
from generativeqc.ks import full_range_stationary_second_order_eligible
from generativeqc.response_solver import GMRESOptions, ResponseSolveError
from generativeqc_compiler.method import resolve_method
from test_integral_autodiff_second import require_native_library  # noqa: F401

ATOMS = [("He", (0.03, -0.02, -0.72)), ("H", (0.11, 0.13, 0.72))]
DIRECTION = np.array([[0.02, -0.03, -0.1], [-0.02, 0.03, 0.1]])
GRID = GridSpec(radial_points=8, angular_polar=4, angular_azimuth=8)
SOLVER = GMRESOptions(atol=1e-12, rtol=1e-11)


@pytest.fixture(params=("B3LYP", "PBE0"))
def calculator(request: pytest.FixtureRequest) -> Calculator:
    """Use a changed physical exchange fraction through the shared public owner."""
    method = resolve_method(request.param)
    method = replace(
        method,
        identifier="user-defined-qualified-hybrid",
        primitives=(
            method.primitives[0],
            replace(method.primitives[1], coefficient=Fraction(37, 100)),
        ),
    )
    return Calculator(
        method=method,
        device="cpu",
        basis_representation="spherical",
        ks_options=KsOptions(grid=GRID),
        energy_tolerance=1e-13,
        density_tolerance=1e-11,
        max_iterations=240,
    )


def test_public_results_are_complete_detached_and_changed_geometry_is_fresh(
    calculator: Calculator, tmp_path: Path
) -> None:
    """The public entry point requalifies geometry and preserves raw covariance."""
    options = {"charge": 1, "cache": tmp_path, "solver_options": SOLVER}
    matrix = calculator.hessian(ATOMS, **options)
    first = calculator.hessian_vector_product(ATOMS, DIRECTION, **options)
    other = np.array([[0.01, 0.04, -0.03], [-0.01, -0.04, 0.03]])
    second = calculator.hessian_vector_product(ATOMS, other, **options)
    for direction, action in ((DIRECTION, first), (other, second)):
        np.testing.assert_allclose(
            matrix.matrix @ direction.ravel(), action.value.ravel(), atol=2e-8, rtol=0
        )
        assert action.diagnostics["public_calculator_endpoint"]
        assert action.diagnostics["nuclear_response_solves"] == 1
        assert action.diagnostics["displaced_scf_evaluations"] == 0
        assert action.diagnostics["response_residual_norm"] < 2e-10
        assert action.diagnostics["scf_fock_builds"] > 0
        assert (
            action.diagnostics["scf_density_sha256"]
            == matrix.diagnostics["scf_density_sha256"]
        )
        assert action.diagnostics["scf_occupied_orbitals"] == (1,)
    assert np.vdot(other, first.value) == pytest.approx(
        np.vdot(DIRECTION, second.value), abs=2e-10
    )
    np.testing.assert_allclose(matrix.matrix, matrix.matrix.T, atol=2e-8, rtol=0)
    np.testing.assert_allclose(
        matrix.matrix @ np.tile(np.eye(3), (2, 1)), 0, atol=2e-8, rtol=0
    )
    assert not matrix.diagnostics["posthoc_symmetrization"]
    shifted = [
        (symbol, np.asarray(position) + 0.8 * motion)
        for (symbol, position), motion in zip(ATOMS, DIRECTION, strict=True)
    ]
    changed = calculator.hessian_vector_product(shifted, DIRECTION, **options)
    assert changed.identity != first.identity
    assert (
        changed.diagnostics["scf_density_sha256"]
        != first.diagnostics["scf_density_sha256"]
    )
    assert (
        abs(changed.diagnostics["scf_energy"] - first.diagnostics["scf_energy"]) > 1e-7
    )
    calculator.clear_cache()
    assert np.isfinite(matrix.matrix).all() and np.isfinite(first.value).all()
    assert not matrix.matrix.flags.writeable and not first.value.flags.writeable


def test_public_failures_close_owners_and_do_not_poison_the_next_request(
    calculator: Calculator, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Reject insufficient resources and injected failures without publishing."""
    options = {"charge": 1, "cache": tmp_path, "solver_options": SOLVER}
    with pytest.raises(ValueError, match="output_budget_bytes"):
        calculator.hessian(ATOMS, output_budget_bytes=1, **options)
    with pytest.raises(MemoryError, match="grid_budget_bytes"):
        calculator.hessian_vector_product(
            ATOMS, DIRECTION, grid_budget_bytes=1, **options
        )
    with pytest.raises(ResponseSolveError):
        calculator.hessian_vector_product(
            ATOMS,
            DIRECTION,
            charge=1,
            cache=tmp_path,
            solver_options=GMRESOptions(max_workspace_bytes=1),
        )
    captured = []

    def fail(
        operator: typing.Any, *args: typing.Any, **kwargs: typing.Any
    ) -> typing.Any:
        """Observe ownership at the public assembly boundary before failing it."""
        captured.append(operator)
        raise FloatingPointError("injected public assembly failure")

    with monkeypatch.context() as context:
        context.setattr(composite, "composite_rks_hvp", fail)
        with pytest.raises(
            FloatingPointError, match="injected public assembly failure"
        ):
            calculator.hessian_vector_product(ATOMS, DIRECTION, **options)
    for owner in captured:
        with pytest.raises(RuntimeError, match="prepared batch is closed"):
            owner.validate_current()
    recovered = calculator.hessian_vector_product(ATOMS, DIRECTION, **options)
    assert recovered.diagnostics["complete_source_coverage"]
    assert recovered.diagnostics["public_calculator_endpoint"]


@pytest.mark.parametrize("method", ("b3lyp-uks",))
def test_public_uks_raw_matrix_and_failure_recovery(
    method: str, tmp_path: Path
) -> None:
    """Exercise nontrivial occupied/virtual spaces in both physical spin blocks."""
    atoms = [
        ("H", (0, 0.1, -1.6)),
        ("H", (0.3, -0.2, 0)),
        ("H", (-0.1, 0.4, 1.6)),
    ]
    direction = np.array(
        [[0.02, -0.03, 0.1], [-0.04, 0.01, -0.02], [0.02, 0.02, -0.08]]
    )
    calculator = Calculator(
        method=method,
        device="cpu",
        basis_representation="spherical",
        ks_options=KsOptions(grid=GRID),
        energy_tolerance=1e-13,
        density_tolerance=1e-11,
        max_iterations=240,
    )
    assert calculator.second_order_capabilities == {"hvp", "hessian"}
    options = {"multiplicity": 2, "cache": tmp_path, "solver_options": SOLVER}
    with pytest.raises(MemoryError, match="grid_budget_bytes"):
        calculator.hessian_vector_product(
            atoms, direction, grid_budget_bytes=1, **options
        )
    with pytest.raises(ResponseSolveError):
        calculator.hessian_vector_product(
            atoms,
            direction,
            multiplicity=2,
            cache=tmp_path,
            solver_options=GMRESOptions(max_workspace_bytes=1),
        )
    matrix = calculator.hessian(atoms, **options)
    action = calculator.hessian_vector_product(atoms, direction, **options)
    np.testing.assert_allclose(
        matrix.matrix @ direction.ravel(), action.value.ravel(), atol=2e-8, rtol=0
    )
    np.testing.assert_allclose(matrix.matrix, matrix.matrix.T, atol=2e-8, rtol=0)
    np.testing.assert_allclose(
        matrix.matrix @ np.tile(np.eye(3), (3, 1)), 0, atol=2e-8, rtol=0
    )
    for result in (matrix, action):
        assert result.diagnostics["public_calculator_endpoint"]
        assert result.diagnostics["spin"] == "polarized"
        assert result.diagnostics["displaced_scf_evaluations"] == 0
        assert result.diagnostics["scf_fock_builds"] > 0
        assert result.diagnostics["scf_occupied_orbitals"] == (2, 1)
        assert result.diagnostics["scf_density_shape"] == (2, 3, 3)
        assert result.diagnostics["scf_physical_residual"] < 2e-10
        assert (
            result.diagnostics["scf_density_sha256"]
            == matrix.diagnostics["scf_density_sha256"]
        )
    calculator.clear_cache()
    assert np.isfinite(matrix.matrix).all() and not matrix.matrix.flags.writeable
    assert np.isfinite(action.value).all() and not action.value.flags.writeable


def test_unqualified_pbe0_uks_stays_closed() -> None:
    """A failed unrestricted point provider cannot inherit RKS admission."""
    calculator = Calculator(
        method="pbe0-uks", device="cpu", ks_options=KsOptions(grid=GRID)
    )
    assert not calculator.second_order_capabilities
    with pytest.raises(NotImplementedError, match="does not expose public hessian"):
        calculator.hessian(ATOMS)


@pytest.mark.parametrize("method", ("b3lyp-rks", "pbe0-rks"))
def test_fitting_does_not_inherit_direct_admission(method: str) -> None:
    """DF has separate response and approximation gates, even on the CPU."""
    calculator = Calculator(
        method=method,
        device="cpu",
        density_fitting="cpu",
        ks_options=KsOptions(grid=GRID),
    )
    assert not calculator.second_order_capabilities
    with pytest.raises(NotImplementedError, match="does not expose public hessian"):
        calculator.hessian(ATOMS)


@pytest.mark.parametrize("method", ("B3LYP", "PBE0"))
@pytest.mark.parametrize("spin", ("unpolarized", "polarized"))
def test_semantic_admission_does_not_trust_an_identifier(
    method: str, spin: str
) -> None:
    """Qualification follows the physical graph, including its resolved spin."""
    graph = resolve_method(method, spin=spin)
    assert full_range_stationary_second_order_eligible(
        replace(graph, identifier="not-a-canonical-method-name")
    ) is (spin == "unpolarized" or method == "B3LYP")
    assert not full_range_stationary_second_order_eligible(
        replace(resolve_method("WB97M-V", spin=spin), identifier=method)
    )


@pytest.mark.parametrize("method", ("b3lyp-rks", "pbe0-rks"))
@pytest.mark.parametrize("domain", ("ecp", "g_shell", "auto_precision"))
def test_unqualified_basis_and_precision_domains_remain_closed(
    method: str, domain: str
) -> None:
    """Angular, core and precision qualification cannot leak from WB97M-V."""
    from generativeqc import Primitive, Shell
    from test_wb97mv_ecp_hessian import synthetic_basis

    options = {}
    if domain == "ecp":
        options["basis"] = synthetic_basis()
    elif domain == "g_shell":
        options["basis"] = (Shell(0, 4, (Primitive(0.7, 1.0),)),)
    else:
        options["precision"] = "auto"
        with pytest.raises(
            NotImplementedError, match="DFT automatic precision currently requires CUDA"
        ):
            Calculator(method=method, device="cpu", **options)
        return
    public = Calculator(method=method, device="cpu", **options)
    assert not public.second_order_capabilities
    with pytest.raises(NotImplementedError, match="does not expose public hessian"):
        public.hessian(ATOMS)
