"""Real Slurm GPU qualification through the actual compiled C++ CLI boundary."""

from __future__ import annotations

import json
import os
import subprocess
from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING, Any, NoReturn

import numpy as np
import pytest
from generativeqc import Atom, Calculator, GridSpec, KsOptions, TargetErrorBudget
from generativeqc.basis import BasisProvenance, BasisSet, BasisShell, ElementBasis
from generativeqc.ks_diagnostics import (
    CUDA_M062X_STATIONARY_DOMAIN_VERSION,
    CUDA_MN15_STATIONARY_DOMAIN_VERSION,
    CUDA_SPLIT_HYBRID_STATIONARY_DOMAINS,
)
from generativeqc.response_solver import GMRESOptions
from generativeqc_compiler.common.resources import ResourceBudget

if TYPE_CHECKING:
    from collections.abc import Callable

pytestmark = pytest.mark.skipif(
    os.environ.get("GENERATIVEQC_WB97MV_HESSIAN_CUDA_TEST") != "1"
    or not os.environ.get("GENERATIVEQC_WB97MV_CLI"),
    reason="requires an explicitly allocated Slurm GPU and a compiled C++ CLI",
)


@pytest.fixture(autouse=True)
def explicit_exact_lr_response(monkeypatch: pytest.MonkeyPatch) -> None:
    """The C++ CLI and Python reference must prepare the same exact LR model."""
    monkeypatch.setenv("GENERATIVEQC_DF_RANGE_EXCHANGE", "direct")


@pytest.mark.parametrize(
    ("method", "domain", "option"),
    (
        ("m06-2x", CUDA_M062X_STATIONARY_DOMAIN_VERSION, "M062X"),
        ("mn15", CUDA_MN15_STATIONARY_DOMAIN_VERSION, "MN15"),
    ),
)
@pytest.mark.parametrize("spin", ("rks", "uks"))
@pytest.mark.parametrize("fitted", (False, True), ids=("direct", "df"))
@pytest.mark.parametrize("ecp", (False, True), ids=("ae", "ecp"))
@pytest.mark.parametrize(
    "orbital_angular", (0, 2, 3), ids=("s", "d-orbital", "f-orbital")
)
def test_cpp_cli_and_python_cuda_split_hybrid_hessian_and_hvp(
    tmp_path: Path,
    spin: str,
    fitted: bool,
    ecp: bool,
    method: str,
    domain: int,
    option: str,
    orbital_angular: int,
) -> None:
    """The two public entry points must replay the same raw stationary result."""
    if os.environ.get(f"GENERATIVEQC_{option}_HESSIAN_CUDA_TEST") != "1":
        pytest.skip(f"requires the opt-in {method} build and s-shell inventory")
    if (
        orbital_angular == 2
        and os.environ.get("GENERATIVEQC_SPLIT_HYBRID_ANGULAR_TEST") != "1"
    ):
        pytest.skip("requires linked d orbital and g auxiliary second derivatives")
    if (
        orbital_angular == 3
        and os.environ.get("GENERATIVEQC_SPLIT_HYBRID_F_TEST") != "1"
    ):
        pytest.skip("requires linked f orbital and g auxiliary second derivatives")
    atoms = [("Na" if ecp else "H", (0.03, -0.02, 0.01)), ("H", (0.15, 0.13, 1.47))]
    direction = np.array([[0.11, 0.02, -0.15], [-0.03, -0.09, 0.06]])
    selector = f"{method}-{spin}"
    charge = int(spin == "uks")
    multiplicity = charge + 1
    orbital: str | BasisSet = "sto-3g"
    orbital_argument = "sto-3g"
    if ecp:
        from test_wb97mv_ecp_hessian import synthetic_basis

        orbital = synthetic_basis()
        if orbital_angular:
            orbital = replace(
                orbital,
                elements=(
                    replace(
                        orbital.elements[0],
                        shells=orbital.elements[0].shells
                        + (BasisShell(orbital_angular, ("0.48",), (("1",),)),),
                    ),
                    orbital.elements[1],
                ),
            )
    elif orbital_angular:
        orbital = BasisSet(
            "Synthetic spherical polarized H basis",
            (
                ElementBasis(
                    1,
                    (
                        BasisShell(0, ("0.73",), (("1",),)),
                        BasisShell(orbital_angular, ("0.56",), (("1",),)),
                    ),
                ),
            ),
            BasisProvenance("test orbital polarization", "1", "fixture", "b" * 64),
            representation="spherical",
        )
    if isinstance(orbital, BasisSet):
        orbital_file = tmp_path / "orbital.json"
        orbital.write(orbital_file)
        orbital_argument = str(orbital_file)
    extra = (
        "--method",
        selector,
        "--basis",
        orbital_argument,
        "--charge",
        str(charge),
        "--multiplicity",
        str(multiplicity),
        "--grid-radial-points",
        "24",
        "--grid-polar-points",
        "8",
        "--grid-azimuth-points",
        "16",
        "--screening-tolerance",
        "1e-14",
    )
    auxiliary = None
    if fitted:
        # The s-only development inventory qualifies these two H JKFIT shells.
        auxiliary = BasisSet(
            "Hydrogen def2-universal-JKFIT s subset",
            (
                ElementBasis(
                    1,
                    (
                        BasisShell(
                            0,
                            ("22.068343", "4.3905712", "1.0540787"),
                            (("0.053033986", "0.3946522022", "0.9172987712"),),
                        ),
                        BasisShell(0, ("0.2717874",), (("1",),)),
                    )
                    + (
                        (BasisShell(4, ("0.43",), (("1",),)),)
                        if orbital_angular
                        else ()
                    ),
                ),
            ),
            BasisProvenance(
                "def2-universal-JKFIT H s subset", "1", "fixture", "a" * 64
            ),
            representation="spherical",
        )
        if ecp:
            auxiliary = replace(
                auxiliary,
                name="NaH synthetic auxiliary fixture",
                elements=(
                    ElementBasis(
                        11,
                        (BasisShell(0, ("0.8",), (("1",),)),)
                        + (
                            (BasisShell(4, ("0.6",), (("1",),)),)
                            if orbital_angular
                            else ()
                        ),
                    ),
                    ElementBasis(1, (BasisShell(0, ("0.65",), (("1",),)),)),
                ),
            )
        auxiliary_file = tmp_path / f"{method} auxiliary.json"
        auxiliary.write(auxiliary_file)
        extra += ("--density-fitting", "cuda", "--auxiliary-basis", str(auxiliary_file))
    settings = {
        "method": selector,
        "device": "cuda",
        "basis": orbital,
        "basis_representation": "spherical",
        "precision": "fp64",
        "density_fitting": "cuda" if fitted else "none",
        "auxiliary_basis": auxiliary,
        "ks_options": KsOptions(
            grid=GridSpec(radial_points=24, angular_polar=8, angular_azimuth=16)
        ),
        "max_iterations": 240,
        "energy_tolerance": 1e-13,
        "density_tolerance": 1e-11,
        "screening_tolerance": 1e-14,
    }
    plan = (
        Calculator(**settings)
        .estimate_resources(
            [atoms],
            charges=[charge],
            multiplicities=[multiplicity],
            properties=("energy",),
        )
        .require_feasible()
    )
    assert plan.requests[0].identity.observables == ("energy",)
    if ecp:
        assert (
            json.loads(plan.requests[0].identity.schedule)["ecp_device_inventory"]
            == "native-ecp-v1"
        )
    budget = ResourceBudget(
        host_bytes=plan.peak_bytes["host"] + 16 * direction.size**2,
        device_bytes=plan.peak_bytes["device"] + (128 << 20),
    )
    calculator = Calculator(**settings, resource_budget=budget)
    assert calculator.second_order_capabilities == {"hessian", "hvp"}
    if ecp:
        assert "forces" not in calculator.capabilities.supported_properties
    state = {"charge": charge, "multiplicity": multiplicity}
    hessian = calculator.hessian(atoms, **state)
    hvp = calculator.hessian_vector_product(atoms, direction, **state)
    cli_hessian = invoke_cli(
        tmp_path, atoms, operation="hessian", default_grid=True, extra=extra
    )
    cli_hvp = invoke_cli(
        tmp_path, atoms, direction=direction, default_grid=True, extra=extra
    )
    for python, cli in ((hessian, cli_hessian), (hvp, cli_hvp)):
        assert (
            python.diagnostics["scf_domain"]
            == CUDA_SPLIT_HYBRID_STATIONARY_DOMAINS[domain]
        )
        assert python.diagnostics["numerical_host_fallbacks"] == 0
        resources = python.diagnostics["resources"]
        ledger = resources["hessian_response"]["device_ledger"]
        assert ledger["peak_bytes"] <= budget.device_bytes
        assert ledger["rejected_allocations"] == 0
        assert (
            ledger["live_bytes"]
            == resources["observation"]["device_ledger"]["live_bytes"]
        )
        assert cli["diagnostics"]["scf_domain_version"] == domain
        assert python.diagnostics["ecp_ad_backend"] == ("cuda" if ecp else None)
        assert cli["diagnostics"]["ecp_ad_backend"] == ("cuda" if ecp else None)
    np.testing.assert_allclose(cli_hessian["matrix"], hessian.matrix, atol=3e-9, rtol=0)
    np.testing.assert_allclose(cli_hvp["value"], hvp.value, atol=3e-9, rtol=0)
    np.testing.assert_allclose(
        hvp.value.ravel(), hessian.matrix @ direction.ravel(), atol=3e-9, rtol=0
    )
    np.testing.assert_allclose(hessian.matrix, hessian.matrix.T, atol=3e-9, rtol=0)
    np.testing.assert_allclose(
        hessian.matrix @ np.tile(np.eye(3), (2, 1)), 0, atol=3e-9, rtol=0
    )


def invoke_cli(
    tmp_path: Path,
    atoms: list,
    *,
    operation: str = "hvp",
    direction: np.ndarray | None = None,
    extra: tuple = (),
    default_grid: bool = False,
    expected_error: str | None = None,
) -> dict:
    """Preserve Slurm visibility and exercise input, process and output contracts."""
    assert os.environ.get("SLURM_JOB_ID"), "real GPU tests require Slurm"
    xyz = tmp_path / "Input With Spaces.xyz"
    xyz.write_text(
        str(len(atoms))
        + "\nBohr\n"
        + "\n".join(
            element + " " + " ".join(str(value) for value in position)
            for element, position in atoms
        )
        + "\n"
    )
    command = [
        os.environ["GENERATIVEQC_WB97MV_CLI"],
        operation,
        str(xyz),
        "--units",
        "bohr",
        "--backend",
        "cuda",
        "--representation",
        "spherical",
        "--json",
        "--max-iterations",
        "240",
        "--energy-tolerance",
        "1e-13",
        "--density-tolerance",
        "1e-11",
        "--response-atol",
        "1e-12",
        "--response-rtol",
        "1e-11",
    ]
    if not default_grid:
        command += [
            "--grid-radial-points",
            "8",
            "--grid-polar-points",
            "4",
            "--grid-azimuth-points",
            "8",
        ]
    if direction is not None:
        source = tmp_path / "Direction With Spaces.json"
        source.write_text(json.dumps(direction.tolist()))
        command += ["--direction", str(source)]
    result = subprocess.run(
        [*command, *extra],
        capture_output=True,
        text=True,
        check=False,
        timeout=600,
        env=dict(
            os.environ,
            PATH="/nonexistent",
            PYTHONHOME="/nonexistent",
            PYTHONPATH="/nonexistent",
        ),
    )
    if expected_error is not None:
        assert result.returncode != 0
        assert expected_error in result.stderr, result.stderr
        assert not result.stdout, "invalid request published a result"
        return {}
    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    (tmp_path / f"{operation}-result.json").write_text(result.stdout)
    assert payload["operation"] == operation
    assert payload["units"] == "hartree/bohr^2"
    assert payload["backend"] == "cuda"
    diagnostics = payload["diagnostics"]
    assert diagnostics["execution_residency"] == "device-resident"
    assert diagnostics["numerical_host_fallbacks"] == 0
    assert diagnostics["intermediate_panel_downloads"] == 0
    assert diagnostics["displaced_scf_evaluations"] == 0
    assert diagnostics["runtime_python"] is False
    assert diagnostics["runtime_compilation"] is False
    assert diagnostics["scf_iterations"] > 0
    for name in (
        "first_integral_backend",
        "second_integral_backend",
        "grid_ad_backend",
        "response_backend",
    ):
        assert diagnostics[name] == "cuda"
    return payload


def qualify_native_frontend(
    monkeypatch: pytest.MonkeyPatch,
    atoms: list,
    direction: np.ndarray,
    expected: object,
    *,
    matrix: object = None,
    multiplicity: int = 1,
    budgeted: bool = False,
    **options: object,
) -> None:
    """Opt-in frontend qualification must not launch any compiler/interpreter."""
    if os.environ.get("GENERATIVEQC_TEST_NATIVE_HESSIAN_FRONTEND") != "1":
        return
    settings = {
        "device": "cuda",
        "basis_representation": "spherical",
        "initial_guess": None,
        "max_iterations": 240,
        "energy_tolerance": 1e-13,
        "density_tolerance": 1e-11,
        "ks_options": KsOptions(
            grid=GridSpec(radial_points=8, angular_polar=4, angular_azimuth=8)
        ),
        **options,
    }
    calculator = Calculator(**settings)
    budget = None
    if budgeted:
        scf = calculator.estimate_resources(
            [atoms], multiplicities=[multiplicity], properties=("energy",)
        ).require_feasible()
        budget = ResourceBudget(
            host_bytes=scf.peak_bytes["host"] + 16 * direction.size**2,
            device_bytes=scf.peak_bytes["device"] + (160 << 20),
        )
        calculator = Calculator(**settings, resource_budget=budget)

    def reject_legacy(*arguments: Any, **keywords: Any) -> NoReturn:
        raise AssertionError(
            "native CUDA endpoint tried Python numerics or a subprocess"
        )

    with monkeypatch.context() as guard:
        guard.setattr(calculator, "_run_public_rks_second_order", reject_legacy)
        guard.setattr(subprocess, "Popen", reject_legacy)
        result = calculator.hessian_vector_product(
            atoms,
            direction,
            multiplicity=multiplicity,
            solver_options=GMRESOptions(atol=1e-12, rtol=1e-11),
        )
        np.testing.assert_allclose(result.value, expected, atol=3e-8, rtol=2e-6)
        assert not result.value.flags.writeable
        assert result.diagnostics["numerical_runtime"] == "native-cpp-cuda"
        assert result.diagnostics["runtime_compilation"] is False
        assert result.diagnostics["intermediate_panel_downloads"] == 0
        if budget is not None:
            ledgers = result.diagnostics["resources"]
            response = ledgers["hessian_response"]["device_ledger"]
            assert response["peak_bytes"] <= budget.device_bytes
            assert (
                response["live_bytes"]
                == ledgers["observation"]["device_ledger"]["live_bytes"]
            )
        if matrix is not None:
            full = calculator.hessian(
                atoms,
                multiplicity=multiplicity,
                solver_options=GMRESOptions(atol=1e-12, rtol=1e-11),
            )
            np.testing.assert_allclose(full.matrix, matrix, atol=3e-8, rtol=2e-6)
            np.testing.assert_allclose(
                full.matrix @ direction.ravel(),
                result.value.ravel(),
                atol=3e-8,
                rtol=2e-6,
            )
            assert full.diagnostics["completed_directions"] == 3 * len(atoms)
            assert full.diagnostics["response_strategy"] == "scalar"
            assert not full.matrix.flags.writeable


@pytest.mark.parametrize("metric_cutoff", (1e-10, 0.03, 0.08))
def test_cpp_cli_cuda_fitted_lr_rks_hessian_and_hvp(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, metric_cutoff: float
) -> None:
    monkeypatch.delenv("GENERATIVEQC_DF_RANGE_EXCHANGE", raising=False)
    atoms = [("H", (0.03, -0.02, 0.01)), ("H", (0.15, 0.13, 1.5))]
    direction = np.array([[0.02, -0.03, 0.1], [-0.02, 0.03, -0.1]])
    fitted = (
        "--density-fitting",
        "cuda",
        "--screening-tolerance",
        "1e-14",
        "--density-fitting-threshold",
        str(metric_cutoff),
    )
    hvp = invoke_cli(tmp_path, atoms, direction=direction, extra=fitted)
    full = invoke_cli(tmp_path, atoms, operation="hessian", extra=fitted)
    matrix = np.asarray(full["matrix"])
    np.testing.assert_allclose(matrix, matrix.T, atol=3e-9, rtol=0)
    np.testing.assert_allclose(
        matrix @ direction.ravel(), np.ravel(hvp["value"]), atol=3e-8
    )
    for payload, directions in ((hvp, 1), (full, 3 * len(atoms))):
        receipt = payload["diagnostics"]
        assert receipt["exact_primitive_components"] == 0
        assert receipt["induced_response_quartets"] == 0
        assert (
            receipt["df_value_passes"]
            == receipt["df_contraction_passes"]
            == 2 * directions
        )
        assert (
            receipt["df_range_value_passes"]
            == receipt["df_range_contraction_passes"]
            == directions
        )
        assert receipt["df_range_workspace_bytes"] > 0
    qualify_native_frontend(
        monkeypatch,
        atoms,
        direction,
        hvp["value"],
        matrix=matrix,
        method="wb97m-v-rks",
        density_fitting="cuda",
        density_fitting_relative_threshold=metric_cutoff,
        screening_tolerance=1e-14,
    )


def test_cpp_cli_cuda_rejects_metric_cutoff_without_publishing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("GENERATIVEQC_DF_RANGE_EXCHANGE", raising=False)
    atoms = [("H", (0, 0, 0)), ("H", (0.15, 0.13, 1.5))]
    invoke_cli(
        tmp_path,
        atoms,
        operation="hessian",
        extra=("--density-fitting", "cuda", "--density-fitting-threshold", "0.0193"),
        expected_error="metric cutoff separation",
    )
    if os.environ.get("GENERATIVEQC_TEST_NATIVE_HESSIAN_FRONTEND") == "1":
        calculator = Calculator(
            method="wb97m-v-rks",
            device="cuda",
            basis_representation="spherical",
            initial_guess=None,
            density_fitting="cuda",
            density_fitting_relative_threshold=0.0193,
            ks_options=KsOptions(
                grid=GridSpec(radial_points=8, angular_polar=4, angular_azimuth=8)
            ),
        )
        with pytest.raises(RuntimeError, match="metric cutoff separation"):
            calculator.hessian(atoms)


def test_wb97mv_separate_vv10_grid_cli_and_python(tmp_path: Path) -> None:
    atoms = [("H", (0.03, -0.02, 0.01)), ("H", (0.15, 0.13, 1.5))]
    direction = np.array([[0.02, -0.03, 0.1], [-0.02, 0.03, -0.1]])
    flags = (
        "--nonlocal-grid-radial-points",
        "6",
        "--nonlocal-grid-polar-points",
        "3",
        "--nonlocal-grid-azimuth-points",
        "6",
    )
    full = invoke_cli(tmp_path, atoms, operation="hessian", extra=flags)
    action = invoke_cli(tmp_path, atoms, direction=direction, extra=flags)
    assert full["diagnostics"]["separate_vv10_grid"]
    assert action["diagnostics"]["separate_vv10_right_directions"] == 2
    assert action["diagnostics"]["vv10_reference_pairs"] > 0
    invoke_cli(
        tmp_path,
        atoms,
        direction=direction,
        extra=(*flags, "--max-nonlocal-pairs", "1"),
        expected_error="max_nonlocal_pairs exceeded",
    )
    calculator = Calculator(
        method="wb97m-v",
        device="cuda",
        basis_representation="spherical",
        initial_guess=None,
        max_iterations=240,
        energy_tolerance=1e-13,
        density_tolerance=1e-11,
        ks_options=KsOptions(
            grid=GridSpec(radial_points=8, angular_polar=4, angular_azimuth=8),
            nonlocal_grid=GridSpec(radial_points=6, angular_polar=3, angular_azimuth=6),
        ),
    )
    assert calculator.second_order_capabilities == {"hvp", "hessian"}
    solve = GMRESOptions(atol=1e-12, rtol=1e-11)
    expected = calculator.hessian_vector_product(atoms, direction, solver_options=solve)
    matrix = calculator.hessian(atoms, solver_options=solve)
    np.testing.assert_allclose(expected.value, action["value"], atol=3e-9, rtol=0)
    np.testing.assert_allclose(matrix.matrix, full["matrix"], atol=3e-9, rtol=0)
    np.testing.assert_allclose(
        matrix.matrix @ direction.ravel(), expected.value.ravel(), atol=3e-9
    )


@pytest.mark.parametrize(
    "vv10_level", [None, 0], ids=("default-level3", "explicit-level0")
)
def test_wb97mv_tight_xc_grid_keeps_independent_vv10(
    tmp_path: Path, vv10_level: int | None
) -> None:
    """The CLI level-4 and Python tight policies must bind the same two grids."""
    atoms = [("H", (0.03, -0.02, 0.01)), ("H", (0.15, 0.13, 1.5))]
    direction = np.array([[0.02, -0.03, 0.1], [-0.02, 0.03, -0.1]])
    flags = ("--nonlocal-grid-level", str(vv10_level)) if vv10_level is not None else ()
    # The standard VV10 mesh for this geometry exceeds the default pair cap.
    # Resource rejection must not write a matrix or silently shrink the grid.
    if vv10_level is None:
        invoke_cli(
            tmp_path,
            atoms,
            direction=direction,
            default_grid=True,
            extra=("--grid-level", "4"),
            expected_error="max_nonlocal_pairs exceeded",
        )
    pair_cap = 350_000_000  # Level 3 has about 3.22e8 ordered pairs for this H₂.
    action = invoke_cli(
        tmp_path,
        atoms,
        direction=direction,
        default_grid=True,
        extra=("--grid-level", "4", "--max-nonlocal-pairs", str(pair_cap), *flags),
    )
    assert action["diagnostics"]["separate_vv10_grid"]
    assert action["diagnostics"]["separate_vv10_right_directions"] == 2
    calculator = Calculator(
        method="wb97m-v",
        device="cuda",
        basis_representation="spherical",
        initial_guess=None,
        max_iterations=240,
        energy_tolerance=1e-13,
        density_tolerance=1e-11,
        ks_options=KsOptions(
            grid_accuracy="tight",
            nonlocal_grid=None if vv10_level is None else GridSpec.treutler(vv10_level),
        ),
    )
    expected = calculator.hessian_vector_product(
        atoms,
        direction,
        max_nonlocal_pairs=pair_cap,
        solver_options=GMRESOptions(atol=1e-12, rtol=1e-11),
    )
    np.testing.assert_allclose(expected.value, action["value"], atol=3e-9, rtol=0)


def test_cpp_cli_cuda_direct_rks_hessian_and_hvp(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    atoms = [("H", (0.03, -0.02, 0.01)), ("H", (0.15, 0.13, 1.5))]
    direction = np.array([[0.02, -0.03, 0.1], [-0.02, 0.03, -0.1]])
    hvp = invoke_cli(tmp_path, atoms, direction=direction)
    matrix = invoke_cli(tmp_path, atoms, operation="hessian")
    assert matrix["diagnostics"]["posthoc_symmetrization"] is False
    matrix = np.asarray(matrix["matrix"])
    np.testing.assert_allclose(
        matrix @ direction.reshape(-1),
        np.asarray(hvp["value"]).reshape(-1),
        atol=2e-8,
        rtol=2e-6,
    )
    np.testing.assert_allclose(matrix, matrix.T, atol=2e-8, rtol=0)
    qualify_native_frontend(
        monkeypatch, atoms, direction, hvp["value"], matrix=matrix, method="wb97m-v"
    )
    cpu = Calculator(
        method="wb97m-v",
        device="cpu",
        basis_representation="spherical",
        initial_guess=None,
        max_iterations=240,
        energy_tolerance=1e-13,
        density_tolerance=1e-11,
        ks_options=KsOptions(
            grid=GridSpec(radial_points=8, angular_polar=4, angular_azimuth=8)
        ),
    )
    energy = cpu.singlepoint(atoms, properties=("energy",)).energy
    np.testing.assert_allclose(hvp["energy_hartree"], energy, atol=5e-9, rtol=0)
    expected = cpu.hessian_vector_product(
        atoms,
        direction,
        cache=tmp_path,
        solver_options=GMRESOptions(atol=1e-12, rtol=1e-11),
    )
    np.testing.assert_allclose(hvp["value"], expected.value, atol=3e-8, rtol=2e-6)


def test_cpp_cli_cuda_physical_csh_selective_ecp_confirmation(tmp_path: Path) -> None:
    """Reproduce a nonlocal curvature miss at unit (not scaled-down) direction."""
    pytest.importorskip("pyscf")
    from test_global_hybrid_ecp_forces import (
        ecp_records,
        independent_solver,
        molecule_atoms,
    )

    _, raw, basis = ecp_records.__wrapped__()
    orbital = tmp_path / "def2-SVP.json"
    basis.write(orbital)
    atoms = molecule_atoms("Cs")
    extra = (
        "--method",
        "b3lyp-rks",
        "--basis",
        str(orbital),
        "--grid-radial-points",
        "16",
        "--grid-polar-points",
        "6",
        "--grid-azimuth-points",
        "12",
    )
    axes = np.eye(6).reshape(6, 2, 3)
    action = invoke_cli(
        tmp_path, atoms, direction=axes[0], extra=extra, default_grid=True
    )
    full = invoke_cli(
        tmp_path, atoms, operation="hessian", extra=extra, default_grid=True
    )
    calculator = Calculator(
        method="b3lyp-rks",
        device="cuda",
        basis=basis,
        basis_representation="spherical",
        density_fitting="none",
        max_iterations=240,
        energy_tolerance=1e-13,
        density_tolerance=1e-11,
        ks_options=KsOptions(
            grid=GridSpec(radial_points=16, angular_polar=6, angular_azimuth=12)
        ),
    )
    public = calculator.hessian_vector_product(
        atoms,
        axes[0],
        cache=tmp_path,
        solver_options=GMRESOptions(atol=1e-12, rtol=1e-11),
    )
    assert public.diagnostics["ecp_ad_backend"] == "cuda"
    assert public.diagnostics["numerical_host_fallbacks"] == 0
    np.testing.assert_allclose(public.value, action["value"], atol=2e-8, rtol=0)
    work = action["diagnostics"]
    assert work["ecp_validation_nonlocal_attempts"] > 0
    assert work["ecp_validation_local_attempts"] == 0
    assert (
        work["ecp_validation_passed_actions"]
        == work["ecp_validation_nonlocal_attempts"]
    )
    assert 0 < work["ecp_validation_affine_terms"] < work["ecp_affine_terms"]
    matrix = np.asarray(full["matrix"])
    assert full["diagnostics"]["posthoc_symmetrization"] is False
    np.testing.assert_allclose(matrix, matrix.T, atol=2e-7, rtol=0)
    np.testing.assert_allclose(
        matrix @ np.tile(np.eye(3), (2, 1)), 0, atol=2e-7, rtol=0
    )
    np.testing.assert_allclose(
        matrix[:, 0], np.ravel(action["value"]), atol=2e-8, rtol=0
    )

    symbols, center = zip(*atoms, strict=True)

    def energy(direction: np.ndarray) -> float:
        moved = list(zip(symbols, np.asarray(center) + direction, strict=True))
        return independent_solver(raw, moved, charge=0, multiplicity=1).e_tot

    origin = energy(np.zeros((2, 3)))
    np.testing.assert_allclose(full["energy_hartree"], origin, atol=2e-9, rtol=0)

    def independent_column(step: float) -> np.ndarray:
        right = step * axes[0]
        diagonal = (energy(right) + energy(-right) - 2 * origin) / step**2
        mixed = [
            (
                energy(left + right)
                + energy(-left - right)
                - energy(left - right)
                - energy(-left + right)
            )
            / (4 * step**2)
            for left in step * axes[1:]
        ]
        return np.array([diagonal, *mixed])

    reference = (4 * independent_column(0.001) - independent_column(0.002)) / 3
    np.testing.assert_allclose(matrix[:, 0], reference, atol=2e-6, rtol=0)


@pytest.mark.parametrize("functional", ("lda", "pbe", "pbe0", "b3lyp"))
@pytest.mark.parametrize("spin", ("rks", "uks"))
@pytest.mark.parametrize("fitted", (False, True), ids=("direct", "df"))
def test_cpp_cli_cuda_semilocal_hessian_and_hvp(
    tmp_path: Path, functional: str, spin: str, fitted: bool
) -> None:
    from generativeqc_compiler.method.spec import resolve_method

    atoms = [("H", (0.03, -0.02, 0.01)), ("H", (0.15, 0.13, 1.5))]
    direction = np.array([[0.02, -0.03, 0.1], [-0.02, 0.03, -0.1]])
    selector = f"{functional}-{spin}"
    extra = ("--method", selector, "--max-nonlocal-pairs", "1")
    if fitted:
        auxiliary = BasisSet(
            "Synthetic hydrogen fitting basis",
            (ElementBasis(1, (BasisShell(0, ("0.65",), (("1",),)),)),),
            BasisProvenance("synthetic test parameters", "1", "fixture", "a" * 64),
            representation="spherical",
        )
        auxiliary_file = tmp_path / "Auxiliary.json"
        auxiliary.write(auxiliary_file)
        extra += ("--density-fitting", "cuda", "--auxiliary-basis", str(auxiliary_file))
    if spin == "uks":
        extra += ("--charge", "1", "--multiplicity", "2")
    hp = invoke_cli(tmp_path, atoms, direction=direction, extra=extra)
    full = invoke_cli(tmp_path, atoms, operation="hessian", extra=extra)
    expected = resolve_method(
        "LDA_XC_PW" if functional == "lda" else functional.upper(),
        spin="polarized" if spin == "uks" else "unpolarized",
    )
    for result in (hp, full):
        assert result["method"] == selector
        assert result["method_ir_identity"] == expected.identity
        assert result["diagnostics"]["df_integral_ad_backend"] == (
            "cuda" if fitted else None
        )
    assert full["diagnostics"]["posthoc_symmetrization"] is False
    matrix = np.asarray(full["matrix"])
    np.testing.assert_allclose(matrix, matrix.T, atol=3e-9, rtol=0)
    np.testing.assert_allclose(
        matrix @ direction.ravel(), np.asarray(hp["value"]).ravel(), atol=3e-9, rtol=0
    )


def test_pbe0_uks_hessian_with_subnormal_spin_densities(tmp_path: Path) -> None:
    """Both occupied spin blocks reach subnormal density on this finite grid."""
    atoms = [
        ("H", (0.13, -0.21, 0.17)),
        ("H", (0.43, 0.19, 2.1)),
        ("H", (0.31, 0.08, 4.0)),
    ]
    direction = np.array(
        [[0.23, -0.11, 0.17], [-0.14, 0.08, -0.32], [0.09, 0.31, 0.13]]
    )
    extra = ("--method", "pbe0-uks", "--basis", "sto-3g", "--multiplicity", "2")
    full = invoke_cli(tmp_path, atoms, operation="hessian", extra=extra)
    hp = invoke_cli(tmp_path, atoms, direction=direction, extra=extra)
    calculator = Calculator(
        method="pbe0-uks",
        device="cuda",
        basis="sto-3g",
        basis_representation="spherical",
        precision="fp64",
        max_iterations=240,
        energy_tolerance=1e-13,
        density_tolerance=1e-11,
        ks_options=KsOptions(
            grid=GridSpec(radial_points=8, angular_polar=4, angular_azimuth=8)
        ),
    )
    controls = {
        "multiplicity": 2,
        "solver_options": GMRESOptions(atol=1e-12, rtol=1e-11),
    }
    python_full = calculator.hessian(atoms, **controls)
    python_hp = calculator.hessian_vector_product(atoms, direction, **controls)
    matrix = np.asarray(full["matrix"])
    assert full["diagnostics"]["posthoc_symmetrization"] is False
    np.testing.assert_allclose(matrix, matrix.T, atol=3e-9, rtol=0)
    np.testing.assert_allclose(
        matrix @ np.tile(np.eye(3), (3, 1)), 0, atol=3e-9, rtol=0
    )
    np.testing.assert_allclose(python_full.matrix, matrix, atol=3e-9, rtol=0)
    np.testing.assert_allclose(python_hp.value, hp["value"], atol=3e-9, rtol=0)
    np.testing.assert_allclose(
        matrix @ direction.ravel(), python_hp.value.ravel(), atol=3e-9, rtol=0
    )


@pytest.mark.parametrize(
    "selector", ("r2scan-rks", "r2scan-uks", "r2scan-3c-rks", "r2scan-3c-uks")
)
@pytest.mark.parametrize("fitted", (False, True), ids=("direct", "df"))
def test_r2scan_cuda_cli_and_python(
    tmp_path: Path, selector: str, fitted: bool
) -> None:
    """Compare full frontend results and the potential-work domain they reuse."""
    from generativeqc.ks_diagnostics import (
        CUDA_R2SCAN_STATIONARY_DOMAIN,
        CUDA_R2SCAN_STATIONARY_DOMAIN_VERSION,
    )

    atoms = [("H", (0.03, -0.02, 0.01)), ("H", (0.15, 0.13, 1.5))]
    direction = np.array([[0.02, -0.03, 0.1], [-0.02, 0.03, -0.1]])
    unrestricted = selector.endswith("-uks")
    composite = "-3c-" in selector
    charge, multiplicity = (1, 2) if unrestricted else (0, 1)
    extra = (
        "--method",
        selector,
        "--max-nonlocal-pairs",
        "1",
        "--screening-tolerance",
        "1e-14",
    )
    if unrestricted:
        extra += ("--charge", "1", "--multiplicity", "2")
    if fitted:
        extra += ("--density-fitting", "cuda")
    kwargs = (
        {} if composite else {"basis": "sto-3g", "basis_representation": "spherical"}
    )
    calculator = Calculator(
        method=selector,
        device="cuda",
        density_fitting="cuda" if fitted else "none",
        ks_options=KsOptions(
            grid=GridSpec(radial_points=8, angular_polar=4, angular_azimuth=8)
        ),
        max_iterations=240,
        energy_tolerance=1e-13,
        density_tolerance=1e-11,
        screening_tolerance=1e-14,
        **kwargs,
    )
    assert calculator.second_order_capabilities == frozenset(("hvp", "hessian"))
    params = {
        "charge": charge,
        "multiplicity": multiplicity,
        "max_nonlocal_pairs": 1,
        "solver_options": GMRESOptions(atol=1e-12, rtol=1e-11),
    }
    hessian = calculator.hessian(atoms, **params)
    hvp = calculator.hessian_vector_product(atoms, direction, **params)
    cli_hessian = invoke_cli(tmp_path, atoms, operation="hessian", extra=extra)
    cli_hvp = invoke_cli(tmp_path, atoms, direction=direction, extra=extra)
    for cli_result, py_result, count in ((cli_hessian, hessian, 6), (cli_hvp, hvp, 1)):
        diagnostics = cli_result["diagnostics"]
        assert (
            diagnostics["scf_domain_version"] == CUDA_R2SCAN_STATIONARY_DOMAIN_VERSION
        )
        assert diagnostics["correction_directions"] == (count if composite else 0)
        assert diagnostics["vv10_reference_pairs"] == 0
        assert cli_result["method_ir_identity"] == calculator.method_ir.identity
        assert py_result.diagnostics["scf_domain"] == CUDA_R2SCAN_STATIONARY_DOMAIN
    np.testing.assert_allclose(cli_hessian["matrix"], hessian.matrix, atol=3e-9, rtol=0)
    np.testing.assert_allclose(cli_hvp["value"], hvp.value, atol=3e-9, rtol=0)
    np.testing.assert_allclose(
        hessian.matrix @ direction.ravel(), hvp.value.ravel(), atol=3e-9
    )


def test_r2scan_cuda_direct_f_orbital_public_domain(tmp_path: Path) -> None:
    """Energy, force and both Hessian frontends bind the same f-shell domain."""
    from generativeqc.ks_diagnostics import (
        CUDA_R2SCAN_STATIONARY_DOMAIN,
        CUDA_R2SCAN_STATIONARY_DOMAIN_VERSION,
    )

    basis = BasisSet(
        "Direct r2scan spherical f-shell fixture",
        (
            ElementBasis(
                1,
                (
                    BasisShell(
                        0,
                        ("3.425250914", "0.6239137298", "0.168855404"),
                        (("0.1543289673", "0.5353281423", "0.4446345422"),),
                    ),
                    BasisShell(3, ("0.48",), (("1",),)),
                ),
            ),
        ),
        BasisProvenance("synthetic test parameters", "1", "fixture", "5" * 64),
        representation="spherical",
    )
    orbital = tmp_path / "f-orbital.json"
    basis.write(orbital)
    atoms = [("H", (0.03, -0.02, 0.01)), ("H", (0.15, 0.13, 1.5))]
    direction = np.array([[0.02, -0.03, 0.1], [-0.02, 0.03, -0.1]])
    calculator = Calculator(
        method="r2scan-rks",
        device="cuda",
        basis=basis,
        basis_representation="spherical",
        density_fitting="none",
        precision="fp64",
        ks_options=KsOptions(
            grid=GridSpec(radial_points=8, angular_polar=4, angular_azimuth=8)
        ),
        max_iterations=240,
        energy_tolerance=1e-13,
        density_tolerance=1e-11,
    )
    with calculator.prepare_batch([atoms]) as batch:
        assert batch._native_dft_force_eligible()
        energy = batch.execute(strict=True, properties=("energy",)).items[0]
        force = batch.execute(strict=True, properties=("energy", "forces")).items[0]
    for state in (energy, force):
        assert state.ks_diagnostic.scf_domain == CUDA_R2SCAN_STATIONARY_DOMAIN
    assert force.energy == pytest.approx(energy.energy, abs=1e-11)
    assert force.forces is not None
    np.testing.assert_allclose(force.forces.sum(axis=0), 0, atol=2e-9, rtol=0)

    action = calculator.hessian_vector_product(
        atoms, direction, solver_options=GMRESOptions(atol=1e-12, rtol=1e-11)
    )
    cli = invoke_cli(
        tmp_path,
        atoms,
        operation="hessian",
        extra=("--method", "r2scan-rks", "--basis", str(orbital)),
    )
    assert action.diagnostics["scf_domain"] == CUDA_R2SCAN_STATIONARY_DOMAIN
    assert (
        cli["diagnostics"]["scf_domain_version"]
        == CUDA_R2SCAN_STATIONARY_DOMAIN_VERSION
    )
    np.testing.assert_allclose(
        np.array(cli["matrix"]) @ direction.ravel(),
        action.value.ravel(),
        atol=3e-9,
        rtol=0,
    )


@pytest.mark.parametrize("unrestricted", (False, True), ids=("rks", "uks"))
@pytest.mark.parametrize("ecp", (False, True), ids=("ae", "ecp"))
def test_r2scan_rvv10_cuda_cli_and_python(
    tmp_path: Path, unrestricted: bool, ecp: bool
) -> None:
    """Check public native ownership, including both ECP and nonlocal reservations."""
    atoms = [("Na" if ecp else "H", (0.03, -0.02, 0.01)), ("H", (0.15, 0.13, 1.5))]
    direction = np.array([[0.02, -0.03, 0.1], [-0.02, 0.03, -0.1]])
    selector = "r2scan+rvv10-uks" if unrestricted else "r2scan+rvv10-rks"
    extra = ("--method", selector, "--screening-tolerance", "1e-14")
    orbital: str | BasisSet = "sto-3g"
    auxiliary = None
    nonlocal_grid = None
    if ecp:
        from test_wb97mv_ecp_hessian import synthetic_basis

        orbital = synthetic_basis()
        orbital_path = tmp_path / "NaH orbital.json"
        orbital.write(orbital_path)
        extra += ("--basis", str(orbital_path))
        if unrestricted:
            auxiliary = BasisSet(
                "NaH rVV10 ECP auxiliary fixture",
                tuple(
                    ElementBasis(element, (BasisShell(0, (exponent,), (("1",),)),))
                    for element, exponent in ((11, "0.8"), (1, "0.65"))
                ),
                BasisProvenance("synthetic fitting fixture", "1", "fixture", "a" * 64),
                representation="spherical",
            )
            auxiliary_path = tmp_path / "NaH auxiliary.json"
            auxiliary.write(auxiliary_path)
            extra += ("--auxiliary-basis", str(auxiliary_path))
    if unrestricted:
        extra += (
            "--charge",
            "1",
            "--multiplicity",
            "2",
            "--density-fitting",
            "cuda",
            "--nonlocal-grid-radial-points",
            "6",
            "--nonlocal-grid-polar-points",
            "3",
            "--nonlocal-grid-azimuth-points",
            "6",
        )
        nonlocal_grid = GridSpec(radial_points=6, angular_polar=3, angular_azimuth=6)
    settings = {
        "method": selector,
        "device": "cuda",
        "basis": orbital,
        "basis_representation": "spherical",
        "density_fitting": "cuda" if unrestricted else "none",
        "auxiliary_basis": auxiliary,
        "ks_options": KsOptions(
            grid=GridSpec(radial_points=8, angular_polar=4, angular_azimuth=8),
            nonlocal_grid=nonlocal_grid,
        ),
        "max_iterations": 240,
        "energy_tolerance": 1e-13,
        "density_tolerance": 1e-11,
        "screening_tolerance": 1e-14,
    }
    calculator = Calculator(**settings)
    assert calculator.second_order_capabilities == frozenset(("hessian", "hvp"))
    assert "forces" not in calculator.capabilities.supported_properties
    params = {
        "charge": int(unrestricted),
        "multiplicity": 2 if unrestricted else 1,
        "solver_options": GMRESOptions(atol=1e-12, rtol=1e-11),
    }
    if ecp:
        plan = calculator.estimate_resources(
            [atoms],
            charges=[params["charge"]],
            multiplicities=[params["multiplicity"]],
            properties=("energy",),
        ).require_feasible()
        assert plan.requests[0].identity.observables == ("energy",)
        assert (
            json.loads(plan.requests[0].identity.schedule)["ecp_device_inventory"]
            == "native-ecp-v1"
        )
        budget = ResourceBudget(
            host_bytes=plan.peak_bytes["host"] + 16 * direction.size**2,
            device_bytes=plan.peak_bytes["device"] + (96 << 20),
        )
        calculator = Calculator(**settings, resource_budget=budget)
    dense = calculator.hessian(atoms, **params)
    hvp = calculator.hessian_vector_product(atoms, direction, **params)
    if ecp:
        for result in (dense, hvp):
            assert result.diagnostics["ecp_ad_backend"] == "cuda"
            assert result.diagnostics["numerical_host_fallbacks"] == 0
            resources = result.diagnostics["resources"]
            ledger = resources["hessian_response"]["device_ledger"]
            assert ledger["peak_bytes"] <= budget.device_bytes
            assert ledger["rejected_allocations"] == 0
            assert (
                ledger["live_bytes"]
                == resources["observation"]["device_ledger"]["live_bytes"]
            )
    matrix, action = dense.matrix, hvp.value
    cli_matrix = invoke_cli(tmp_path, atoms, operation="hessian", extra=extra)
    cli_action = invoke_cli(tmp_path, atoms, direction=direction, extra=extra)
    for result in (cli_matrix, cli_action):
        assert result["method_ir_identity"] == calculator.method_ir.identity
        assert result["diagnostics"]["separate_vv10_grid"] == unrestricted
        assert result["diagnostics"]["vv10_reference_pairs"] > 0
        assert result["diagnostics"]["ecp_ad_backend"] == ("cuda" if ecp else None)
    np.testing.assert_allclose(cli_matrix["matrix"], matrix, atol=3e-9, rtol=0)
    np.testing.assert_allclose(cli_action["value"], action, atol=3e-9, rtol=0)
    np.testing.assert_allclose(
        matrix @ direction.ravel(), action.ravel(), atol=3e-9, rtol=0
    )


@pytest.mark.parametrize("method", ("lda", "pbe", "pbe0", "b3lyp", "wb97m-v"))
@pytest.mark.parametrize("spin", ("rks", "uks"))
def test_auto_scf_cuda_hessian_frontends(
    tmp_path: Path, method: str, spin: str
) -> None:
    atoms = [("H", (0.03, -0.02, 0.01)), ("H", (0.15, 0.13, 1.5))]
    direction = np.array([[0.02, -0.03, 0.1], [-0.02, 0.03, -0.1]])
    selector = f"{method}-{spin}"
    unrestricted = spin == "uks"
    charge, multiplicity = (1, 2) if unrestricted else (0, 1)
    common = {
        "method": selector,
        "device": "cuda",
        "basis": "sto-3g",
        "basis_representation": "spherical",
        "ks_options": KsOptions(
            grid=GridSpec(radial_points=8, angular_polar=4, angular_azimuth=8)
        ),
        "max_iterations": 240,
        "energy_tolerance": 1e-13,
        "density_tolerance": 1e-11,
        "screening_tolerance": 1e-14,
    }
    auto = Calculator(precision="auto", **common)
    strict = Calculator(precision="fp64", **common)
    assert auto.second_order_capabilities == frozenset(("hessian", "hvp"))
    params = {
        "charge": charge,
        "multiplicity": multiplicity,
        "solver_options": GMRESOptions(atol=1e-12, rtol=1e-11),
    }
    expected = strict.hessian(atoms, **params).matrix
    matrix = auto.hessian(atoms, **params)
    hvp = auto.hessian_vector_product(atoms, direction, **params)
    extra = (
        "--method",
        selector,
        "--precision",
        "auto",
        "--screening-tolerance",
        "1e-14",
    )
    if unrestricted:
        extra += ("--charge", "1", "--multiplicity", "2")
    cli = invoke_cli(tmp_path, atoms, direction=direction, extra=extra)
    assert cli["diagnostics"]["scf_strict_physical_fock"]
    assert cli["diagnostics"]["scf_requested_precision"] == "auto"
    assert matrix.diagnostics["scf_requested_precision"] == "auto"
    if method in ("pbe0", "b3lyp", "wb97m-v"):
        assert cli["diagnostics"]["scf_mixed_fock_builds"] > 0
        assert matrix.diagnostics["scf_mixed_fock_builds"] > 0
    np.testing.assert_allclose(matrix.matrix, expected, atol=3e-9, rtol=0)
    np.testing.assert_allclose(hvp.value, cli["value"], atol=3e-9, rtol=0)
    np.testing.assert_allclose(
        hvp.value.ravel(), expected @ direction.ravel(), atol=3e-9, rtol=0
    )


@pytest.mark.parametrize("warmup", (True, False), ids=("mixed-xc", "strict-fallback"))
def test_auto_scf_fitted_pbe_hessian_frontends(tmp_path: Path, warmup: bool) -> None:
    atoms = [("H", (0.13, -0.21, 0.17)), ("H", (0.43, 0.19, 2.1))]
    direction = np.array([[0.02, -0.03, 0.1], [-0.02, 0.03, -0.1]])
    provenance = BasisProvenance("synthetic test parameters", "1", "fixture", "a" * 64)
    shell_specs = ((0, "0.7"), (0, "0.28"), (1, "0.38"), (1, "1.3"), (2, "0.72"))
    orbital_shells = tuple(
        BasisShell(angular, (exponent,), (("1",),))
        for angular, exponent in (shell_specs if warmup else shell_specs[:1])
    )
    orbital = BasisSet(
        "Synthetic orbital",
        (ElementBasis(1, orbital_shells),),
        provenance,
        representation="spherical",
    )
    auxiliary = BasisSet(
        "Synthetic fitting",
        (ElementBasis(1, (BasisShell(0, ("0.8",), (("1",),)),)),),
        provenance,
        representation="spherical",
    )
    orbital_file, auxiliary_file = (
        tmp_path / "Orbital.json",
        tmp_path / "Auxiliary.json",
    )
    orbital.write(orbital_file)
    auxiliary.write(auxiliary_file)
    common = {
        "method": "pbe-rks",
        "device": "cuda",
        "basis": orbital,
        "auxiliary_basis": auxiliary,
        "density_fitting": "cuda",
        "max_iterations": 240,
        "energy_tolerance": 1e-13,
        "density_tolerance": 1e-11,
        "screening_tolerance": 1e-14,
    }
    options = GMRESOptions(atol=1e-12, rtol=1e-11)
    auto = Calculator(precision="auto", **common)
    strict = Calculator(precision="fp64", **common)
    assert auto.second_order_capabilities == frozenset(("hessian", "hvp"))
    matrix = auto.hessian(atoms, solver_options=options)
    expected = strict.hessian(atoms, solver_options=options)
    cli = invoke_cli(
        tmp_path,
        atoms,
        direction=direction,
        default_grid=True,
        extra=(
            "--method",
            "pbe-rks",
            "--precision",
            "auto",
            "--density-fitting",
            "cuda",
            "--basis",
            str(orbital_file),
            "--auxiliary-basis",
            str(auxiliary_file),
            "--screening-tolerance",
            "1e-14",
        ),
    )
    for diagnostics in (matrix.diagnostics, cli["diagnostics"]):
        assert diagnostics["scf_requested_precision"] == "auto"
        assert (diagnostics["scf_mixed_fock_builds"] > 0) == warmup
        assert diagnostics["df_integral_ad_backend"] == "cuda"
    np.testing.assert_allclose(matrix.matrix, expected.matrix, atol=3e-9, rtol=0)
    hvp = auto.hessian_vector_product(atoms, direction, solver_options=options).value
    np.testing.assert_allclose(
        hvp.ravel(), matrix.matrix @ direction.ravel(), atol=3e-9, rtol=0
    )
    np.testing.assert_allclose(hvp, cli["value"], atol=3e-9, rtol=0)

    if warmup:
        from generativeqc._native_hessian import run_native_cuda

        relaxed = Calculator(
            precision="auto",
            _application_precision_budget=TargetErrorBudget(1e-5, 1e-4, 3e-5),
            **{**common, "energy_tolerance": 1e-6, "density_tolerance": 1e-4},
        )
        assert not relaxed.second_order_capabilities
        # Exercise the C bridge directly rather than only the Python policy.
        with pytest.raises(
            NotImplementedError, match="relaxed application stationarity"
        ):
            run_native_cuda(
                relaxed,
                tuple(Atom.from_value(atom) for atom in atoms),
                charge=0,
                multiplicity=1,
                direction=direction,
            )


@pytest.mark.parametrize(
    "extra",
    (("--method", "r2scan-rks"), ("--method", "pbe0-rks", "--density-fitting", "cuda")),
)
def test_auto_scf_hessian_rejects_unqualified_model(
    tmp_path: Path, extra: tuple
) -> None:
    atoms = [("H", (0.0, 0.0, 0.0)), ("H", (0.0, 0.0, 1.5))]
    invoke_cli(
        tmp_path,
        atoms,
        operation="hessian",
        extra=("--precision", "auto", *extra),
        expected_error="AUTO requires a qualified electronic method and provider",
    )


def test_fp32_cpks_proposal_is_audited_by_both_frontends(tmp_path: Path) -> None:
    atoms = [
        ("O", (0.0, 0.0, 0.0)),
        ("H", (1.43233673, 0.0, 1.10715266)),
        ("H", (-1.43233673, 0.0, 1.10715266)),
    ]
    args = ("--method", "wb97m-v-rks", "--screening-tolerance", "1e-14")
    first = invoke_cli(tmp_path, atoms, operation="hessian", extra=args)
    mixed = invoke_cli(
        tmp_path,
        atoms,
        operation="hessian",
        extra=(*args, "--response-cache-precision", "fp32-proposal"),
    )
    diagnostics = mixed["diagnostics"]
    assert diagnostics["cpks_cache_precision"] == "fp32-proposal"
    assert (
        diagnostics["cpks_operator_bytes"] * 2
        == first["diagnostics"]["cpks_operator_bytes"]
    )
    assert diagnostics["cpks_proposal_seeds"] > 0
    assert diagnostics["cpks_physical_checks"] == 3 * len(atoms)
    np.testing.assert_allclose(mixed["matrix"], first["matrix"], atol=3e-9, rtol=0)
    calculator = Calculator(
        method="wb97m-v-rks",
        device="cuda",
        basis="sto-3g",
        basis_representation="spherical",
        precision="fp64",
        max_iterations=240,
        energy_tolerance=1e-13,
        density_tolerance=1e-11,
        screening_tolerance=1e-14,
        ks_options=KsOptions(
            grid=GridSpec(radial_points=8, angular_polar=4, angular_azimuth=8)
        ),
    )
    python_matrix = calculator.hessian(
        atoms,
        response_cache_precision="fp32-proposal",
        solver_options=GMRESOptions(atol=1e-12, rtol=1e-11),
    )
    for field in (
        "cpks_cache_precision",
        "cpks_operator_bytes",
        "cpks_physical_checks",
    ):
        assert python_matrix.diagnostics[field] == diagnostics[field]
    assert python_matrix.diagnostics["cpks_proposal_seeds"] > 0
    np.testing.assert_allclose(python_matrix.matrix, mixed["matrix"], atol=3e-9, rtol=0)
    invoke_cli(
        tmp_path,
        atoms,
        direction=np.ones((3, 3)),
        extra=(*args, "--response-cache-precision", "fp32-proposal"),
        expected_error="requires a dense hessian",
    )


def test_r2scan_cuda_point_domain_routing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The same library retains incumbent AUTO and batches the strict RKS domain."""
    from generativeqc._dft_gradient import (
        StationaryDerivativeContract,
        StationaryKsState,
    )
    from generativeqc.ks_diagnostics import CUDA_R2SCAN_STATIONARY_DOMAIN
    from generativeqc_compiler.dft import NativeAO

    assert os.environ.get("SLURM_JOB_ID"), "real GPU tests require Slurm"
    atoms = [("H", (0.03, -0.02, 0.01)), ("H", (0.15, 0.13, 1.5))]
    options = {
        "method": "r2scan-rks",
        "device": "cuda",
        "basis": "sto-3g",
        "ks_options": KsOptions(
            grid=GridSpec(radial_points=8, angular_polar=4, angular_azimuth=8),
            tile_points=37,
        ),
        "energy_tolerance": 1e-12,
        "density_tolerance": 1e-10,
    }
    strict_results = []
    monkeypatch.setenv("GENERATIVEQC_CUDA_XC_EQUAL_SPIN", "1")
    for tiles in (1, 4):
        trace = tmp_path / f"xc-tiles-{tiles}.jsonl"
        monkeypatch.setenv("GENERATIVEQC_DF_TRACE", str(trace))
        monkeypatch.setenv("GENERATIVEQC_CUDA_XC_BATCH_TILES", str(tiles))
        calculator = Calculator(precision="fp64", **options)
        with calculator.prepare_batch([atoms]) as batch:
            assert batch._native_dft_force_eligible()
            item = batch.execute(strict=True, properties=("energy", "forces")).items[0]
            if tiles == 1:
                with NativeAO(atoms, basis="sto-3g") as basis:
                    state = StationaryKsState.from_native(
                        batch, basis, allow_reference_domain=True
                    )
                    try:
                        assert state._source.scf_domain == CUDA_R2SCAN_STATIONARY_DOMAIN
                        with pytest.raises(
                            NotImplementedError, match="variational domain"
                        ):
                            StationaryDerivativeContract(state.identity).validate(state)
                    finally:
                        state._source.close()
        assert item.ks_diagnostic.scf_domain == CUDA_R2SCAN_STATIONARY_DOMAIN
        counts = [
            event["counters"]["xc_point_batch_tiles"]
            for event in map(json.loads, trace.read_text().splitlines())
            if "xc_point_batch_tiles" in event["counters"]
        ]
        assert counts and max(counts) == tiles
        strict_results.append(item)
    single, batched = strict_results
    assert single.energy == pytest.approx(batched.energy, abs=1e-11)
    np.testing.assert_allclose(single.forces, batched.forces, atol=1e-10, rtol=0)

    monkeypatch.delenv("GENERATIVEQC_DF_TRACE")
    incumbent = Calculator(precision="auto", **options)
    assert not incumbent.second_order_capabilities
    result = incumbent.singlepoint(atoms, properties=("energy",))
    assert result.ks_diagnostic.scf_domain == incumbent.ks_options.scf_domain
    assert result.ks_diagnostic.scf_domain != CUDA_R2SCAN_STATIONARY_DOMAIN

    # The compiled native force requires three partition iterations. A
    # different grid retains the incumbent energy before any force is requested.
    other_grid = replace(options["ks_options"].grid, partition_iterations=2)
    unsupported = Calculator(**{**options, "ks_options": KsOptions(grid=other_grid)})
    assert not unsupported.second_order_capabilities
    result = unsupported.singlepoint(atoms, properties=("energy",))
    assert result.ks_diagnostic.scf_domain == unsupported.ks_options.scf_domain


@pytest.mark.parametrize(
    "default_grid", (False, True), ids=("small-grid", "default-grid")
)
def test_r2scan3c_fitted_water_hvp(tmp_path: Path, default_grid: bool) -> None:
    """Cover cancellation in the full-rank metric with the actual method basis."""
    elements = ("O", "H", "H")
    positions = np.array(
        [[0.0, 0.0, 0.0], [1.43233673, 0.0, 1.10715266], [-1.40233673, 0.1, 1.10715266]]
    )
    motion = np.array([[0.01, -0.03, 0.10], [-0.04, 0.02, 0.01], [0.03, 0.01, -0.11]])
    result = invoke_cli(
        tmp_path,
        list(zip(elements, positions, strict=True)),
        direction=motion,
        default_grid=default_grid,
        extra=(
            "--method",
            "r2scan-3c-rks",
            "--density-fitting",
            "cuda",
            "--screening-tolerance",
            "1e-14",
        ),
    )
    diagnostics = result["diagnostics"]
    assert diagnostics["correction_directions"] == 1
    assert diagnostics["df_integral_ad_backend"] == "cuda"
    assert diagnostics["exact_primitive_components"] == 0
    calculator = Calculator(
        method="r2scan-3c-rks",
        device="cuda",
        density_fitting="cuda",
        ks_options=KsOptions(
            grid=None
            if default_grid
            else GridSpec(radial_points=8, angular_polar=4, angular_azimuth=8)
        ),
        max_iterations=240,
        energy_tolerance=1e-13,
        density_tolerance=1e-11,
        screening_tolerance=1e-14,
    )

    def gradient(shift: float) -> np.ndarray:
        atoms = list(zip(elements, positions + shift * motion, strict=True))
        with calculator.prepare_batch([atoms], warm_start=False) as batch:
            item = batch.execute(strict=True, properties=("energy", "forces")).items[0]
            assert item.forces is not None and item.dispersion is not None
            return -item.forces

    coarse = (gradient(0.002) - gradient(-0.002)) / 0.004
    fine = (gradient(0.001) - gradient(-0.001)) / 0.002
    actual = np.asarray(result["value"])
    np.testing.assert_allclose(actual, (4 * fine - coarse) / 3, atol=8e-7, rtol=0)
    np.testing.assert_allclose(actual.sum(axis=0), 0, atol=3e-9, rtol=0)


@pytest.mark.parametrize("fitted", (False, True), ids=("direct", "df"))
def test_pbe_d4_cli_and_python(tmp_path: Path, fitted: bool) -> None:
    """Both the catalog owner and the legacy intrinsic owner publish total response."""
    from generativeqc_compiler.method import resolve_method

    atoms = [
        ("O", (0.0, 0.0, 0.0)),
        ("H", (1.43233673, 0.0, 1.10715266)),
        ("H", (-1.40233673, 0.1, 1.10715266)),
    ]
    direction = np.array(
        [[0.01, -0.03, 0.10], [-0.04, 0.02, 0.01], [0.03, 0.01, -0.11]]
    )
    selector = "pbe-d4(bj-eeq-atm)-rks" if fitted else "pbe-d4-rks"
    common = (
        "--basis",
        "sto-3g",
        "--grid-radial-points",
        "16",
        "--grid-polar-points",
        "8",
        "--grid-azimuth-points",
        "16",
        "--screening-tolerance",
        "1e-14",
        "--density-fitting",
        "cuda" if fitted else "none",
    )
    full = invoke_cli(
        tmp_path,
        atoms,
        operation="hessian",
        default_grid=True,
        extra=("--method", selector, *common),
    )
    bare = invoke_cli(
        tmp_path,
        atoms,
        direction=direction,
        default_grid=True,
        extra=("--method", "pbe-rks", *common),
    )
    graph = resolve_method("PBE-D4(BJ-EEQ-ATM)", spin="unpolarized")
    assert full["method_ir_identity"] == graph.identity
    assert full["diagnostics"]["correction_directions"] == direction.size
    assert bare["diagnostics"]["correction_directions"] == 0
    assert abs(full["energy_hartree"] - bare["energy_hartree"]) > 1e-5
    matrix = np.asarray(full["matrix"]).reshape(direction.size, direction.size)
    expected = matrix @ direction.ravel()
    assert np.max(np.abs(expected - np.ravel(bare["value"]))) > 1e-7
    action = invoke_cli(
        tmp_path,
        atoms,
        direction=direction,
        default_grid=True,
        extra=("--method", selector, *common),
    )
    assert action["diagnostics"]["correction_directions"] == 1
    np.testing.assert_allclose(np.ravel(action["value"]), expected, atol=2e-9, rtol=0)
    options = {
        "device": "cuda",
        "basis": "sto-3g",
        "basis_representation": "spherical",
        "density_fitting": "cuda" if fitted else "none",
        "max_iterations": 240,
        "energy_tolerance": 1e-13,
        "density_tolerance": 1e-11,
        "screening_tolerance": 1e-14,
        "ks_options": KsOptions(
            grid=GridSpec(radial_points=16, angular_polar=8, angular_azimuth=16)
        ),
    }
    solver = GMRESOptions(atol=1e-12, rtol=1e-11)
    for method in ("pbe-d4-rks", graph):
        calculator = Calculator(method=method, **options)
        assert calculator.second_order_capabilities == {"hvp", "hessian"}
        result = calculator.hessian_vector_product(
            atoms, direction, solver_options=solver
        )
        assert result.diagnostics["df_integral_ad_backend"] == (
            "cuda" if fitted else None
        )
        np.testing.assert_allclose(result.value.ravel(), expected, atol=2e-9, rtol=0)
        assert result.diagnostics["completed_directions"] == 1
        if method is graph:
            dense = calculator.hessian(atoms, solver_options=solver)
            np.testing.assert_allclose(dense.matrix, matrix, atol=2e-9, rtol=0)
        else:
            plan = calculator.estimate_resources([atoms], properties=("energy",))
            plan.require_feasible()
            assert {request.name for request in plan.requests} == {"ks", "d4"}
            bounded = Calculator(
                method=method,
                resource_budget=ResourceBudget(
                    host_bytes=plan.peak_bytes["host"] + 16 * direction.size**2,
                    device_bytes=plan.peak_bytes["device"] + (128 << 20),
                ),
                **options,
            )
            dense = bounded.hessian(atoms, solver_options=solver)
            action = bounded.hessian_vector_product(
                atoms, direction, solver_options=solver
            )
            for result, count in ((dense, direction.size), (action, 1)):
                assert result.diagnostics["completed_directions"] == count
                resources = result.diagnostics["resources"]
                ledger = resources["hessian_response"]["device_ledger"]
                assert ledger["rejected_allocations"] == 0
                assert (
                    ledger["live_bytes"]
                    == resources["observation"]["device_ledger"]["live_bytes"]
                )
            np.testing.assert_allclose(dense.matrix, matrix, atol=2e-9, rtol=0)
            np.testing.assert_allclose(
                action.value.ravel(), expected, atol=2e-9, rtol=0
            )


@pytest.mark.parametrize("functional", ("pbe", "pbe0", "b3lyp", "r2scan"))
@pytest.mark.parametrize("spin", ("rks", "uks"))
@pytest.mark.parametrize("fitted", (False, True), ids=("direct", "df"))
def test_named_d4_cuda_complete_hessian(
    tmp_path: Path,
    functional: str,
    spin: str,
    fitted: bool,
    record_property: Callable[[str, float], None],
) -> None:
    """Gate the total endpoint against the same charged D4 correction in isolation."""
    from generativeqc.ks_diagnostics import CUDA_R2SCAN_STATIONARY_DOMAIN
    from generativeqc_compiler.method import resolve_method

    interface = pytest.importorskip("dftd4.interface")
    numbers = np.array([8, 1, 1], dtype=np.int32)
    positions = np.array(
        [[0.0, 0.0, 0.0], [1.43233673, 0.0, 1.10715266], [-1.40233673, 0.1, 1.10715266]]
    )
    atoms = list(zip(("O", "H", "H"), positions, strict=True))
    direction = np.array(
        [[0.02, -0.01, 0.03], [-0.01, 0.02, -0.04], [-0.01, -0.01, 0.01]]
    )
    charge, multiplicity = (1, 2) if spin == "uks" else (0, 1)
    damping = interface.DampingParam(method=functional, atm=True)
    axes = np.eye(positions.size).reshape(positions.size, *positions.shape)

    def oracle(xyz: np.ndarray, molecular_charge: int) -> dict:
        model = interface.DispersionModel(numbers, xyz, charge=float(molecular_charge))
        return model.get_dispersion(damping, grad=True)

    def stencil(step: float) -> np.ndarray:
        return np.column_stack(
            [
                np.ravel(
                    oracle(positions + step * axis, charge)["gradient"]
                    - oracle(positions - step * axis, charge)["gradient"]
                )
                / (2 * step)
                for axis in axes
            ]
        )

    expected = (4 * stencil(0.0002) - stencil(0.0004)) / 3
    reference_energy = oracle(positions, charge)["energy"]
    if charge:
        # Reject an ineffective charged fixture (e.g. a change in spin only).
        assert abs(reference_energy - oracle(positions, 0)["energy"]) > 1e-6
    assert np.max(np.abs(expected)) > 1e-6
    fitting = "cuda" if fitted else "none"
    common = (
        "--basis",
        "sto-3g",
        "--screening-tolerance",
        "1e-14",
        "--density-fitting",
        fitting,
        "--charge",
        str(charge),
        "--multiplicity",
        str(multiplicity),
        "--grid-radial-points",
        "16",
        "--grid-polar-points",
        "8",
        "--grid-azimuth-points",
        "16",
    )
    graph = resolve_method(
        f"{functional.upper()}-D4(BJ-EEQ-ATM)",
        spin="polarized" if spin == "uks" else "unpolarized",
    )
    alias = {("pbe0", "rks", False): "pbe1pbe", ("b3lyp", "uks", True): "b3lypg"}
    selector = (
        f"{alias.get((functional, spin, fitted), functional)}-d4(bj-eeq-atm)-{spin}"
    )
    total = invoke_cli(
        tmp_path,
        atoms,
        operation="hessian",
        default_grid=True,
        extra=("--method", selector, *common),
    )
    bare = invoke_cli(
        tmp_path,
        atoms,
        operation="hessian",
        default_grid=True,
        extra=("--method", f"{functional}-{spin}", *common),
    )
    assert total["method_ir_identity"] == graph.identity
    assert total["diagnostics"]["correction_directions"] == positions.size
    assert bare["diagnostics"]["correction_directions"] == 0
    np.testing.assert_allclose(
        total["energy_hartree"] - bare["energy_hartree"],
        reference_energy,
        atol=2e-11,
        rtol=0,
    )
    matrix = np.asarray(total["matrix"])
    delta = matrix - np.asarray(bare["matrix"])
    record_property("max_d4_error_hartree_per_bohr2", np.max(np.abs(delta - expected)))
    np.testing.assert_allclose(delta, expected, atol=2e-10, rtol=0)
    np.testing.assert_allclose(matrix, matrix.T, atol=4e-9, rtol=0)
    np.testing.assert_allclose(
        matrix @ np.tile(np.eye(3), (len(atoms), 1)), 0, atol=4e-9, rtol=0
    )
    settings = {
        "method": graph,
        "device": "cuda",
        "basis": "sto-3g",
        "basis_representation": "spherical",
        "density_fitting": fitting,
        "max_iterations": 240,
        "energy_tolerance": 1e-13,
        "density_tolerance": 1e-11,
        "screening_tolerance": 1e-14,
        "ks_options": KsOptions(
            grid=GridSpec(radial_points=16, angular_polar=8, angular_azimuth=16)
        ),
    }
    plan = (
        Calculator(**settings)
        .estimate_resources(
            [atoms],
            charges=[charge],
            multiplicities=[multiplicity],
            properties=("energy",),
        )
        .require_feasible()
    )
    requests = {request.name: request for request in plan.requests}
    assert requests["ks"].identity.observables == ("energy",)
    assert any(
        row.name == "D4 retained CUDA owner" and row.bytes for row in plan.estimates
    )
    budget = ResourceBudget(
        host_bytes=plan.peak_bytes["host"] + 16 * positions.size**2,
        device_bytes=plan.peak_bytes["device"] + (128 << 20),
    )
    calculator = Calculator(**settings, resource_budget=budget)
    assert calculator.second_order_capabilities == {"hvp", "hessian"}
    endpoint = {
        "charge": charge,
        "multiplicity": multiplicity,
        "solver_options": GMRESOptions(atol=1e-12, rtol=1e-11),
    }
    dense = calculator.hessian(atoms, **endpoint)
    action = calculator.hessian_vector_product(
        atoms,
        direction,
        **endpoint,
    )
    for result, count in ((dense, positions.size), (action, 1)):
        assert result.diagnostics["completed_directions"] == count
        assert result.diagnostics["numerical_host_fallbacks"] == 0
        resources = result.diagnostics["resources"]
        ledger = resources["hessian_response"]["device_ledger"]
        assert ledger["peak_bytes"] <= budget.device_bytes
        assert ledger["rejected_allocations"] == 0
        assert (
            ledger["live_bytes"]
            == resources["observation"]["device_ledger"]["live_bytes"]
        )
    assert action.diagnostics["df_integral_ad_backend"] == ("cuda" if fitted else None)
    if functional == "r2scan":
        assert action.diagnostics["scf_domain"] == CUDA_R2SCAN_STATIONARY_DOMAIN
    np.testing.assert_allclose(dense.matrix, matrix, atol=3e-9, rtol=0)
    np.testing.assert_allclose(
        action.value.ravel(), matrix @ direction.ravel(), atol=3e-9, rtol=0
    )


def test_d4_hessian_admission_does_not_follow_the_correction_alone(
    tmp_path: Path,
) -> None:
    """Require both a qualified electronic program and an exact correction graph."""
    from generativeqc_compiler.method import (
        DispersionCorrectionPrimitive,
        resolve_method,
    )

    invoke_cli(
        tmp_path,
        [("H", (0.0, 0.0, 0.0)), ("H", (0.0, 0.0, 1.5))],
        operation="hessian",
        extra=("--method", "wb97x-d4(bj-eeq-atm)-rks", "--basis", "sto-3g"),
        expected_error="method has no qualified native CUDA hessian/hvp",
    )
    graph = resolve_method("PBE0-D4(BJ-EEQ-ATM)", spin="unpolarized")
    modified = replace(
        graph,
        primitives=tuple(
            replace(
                node,
                specification=replace(
                    node.specification, s8=node.specification.s8 + 0.1
                ),
            )
            if isinstance(node, DispersionCorrectionPrimitive)
            else node
            for node in graph.primitives
        ),
    )
    calculator = Calculator(method=modified, device="cuda", basis="sto-3g")
    assert not calculator.second_order_capabilities


@pytest.mark.parametrize(
    ("functional", "variant", "spin"),
    (
        ("PBE", "D3(BJ)", "rks"),
        ("PBE", "D3(BJ)-ATM", "rks"),
        ("PBE", "D3(0)", "rks"),
        ("PBE0", "D3(BJ)", "uks"),
        ("PBE0", "D3(0)", "uks"),
    ),
)
@pytest.mark.parametrize("fitted", (False, True), ids=("direct", "df"))
def test_d3_cuda_complete_hessian(
    tmp_path: Path,
    functional: str,
    variant: str,
    spin: str,
    fitted: bool,
    record_property: Callable[[str, float], None],
) -> None:
    """Isolate the analytic GPU correction using independent dftd3 gradients."""
    from generativeqc_compiler.method import (
        DispersionCorrectionPrimitive,
        resolve_method,
    )

    interface = pytest.importorskip("dftd3.interface")
    elements = ("O", "H", "H")
    positions = np.array(
        [[0.0, 0.0, 0.0], [1.43233673, 0.0, 1.10715266], [-1.40233673, 0.1, 1.10715266]]
    )
    direction = np.array(
        [[0.01, -0.03, 0.10], [-0.04, 0.02, 0.01], [0.03, 0.01, -0.11]]
    )
    atoms = list(zip(elements, positions, strict=True))
    charge, multiplicity = (1, 2) if spin == "uks" else (0, 1)
    graph = resolve_method(
        f"{functional}-{variant}",
        spin="polarized" if spin == "uks" else "unpolarized",
    )
    spec = next(
        node.specification
        for node in graph.primitives
        if isinstance(node, DispersionCorrectionPrimitive)
    )
    damping = (
        interface.RationalDampingParam(
            s6=spec.s6, s8=spec.s8, s9=spec.s9, a1=spec.a1, a2=spec.a2
        )
        if spec.damping == "bj"
        else interface.ZeroDampingParam(
            s6=spec.s6, s8=spec.s8, s9=spec.s9, rs6=spec.rs6, rs8=spec.rs8, alp=spec.alp
        )
    )

    def gradient(xyz: np.ndarray) -> np.ndarray:
        model = interface.DispersionModel(np.array([8, 1, 1], dtype=np.int32), xyz)
        return np.ravel(model.get_dispersion(damping, grad=True)["gradient"])

    common = (
        "--basis",
        "sto-3g",
        "--grid-radial-points",
        "16",
        "--grid-polar-points",
        "8",
        "--grid-azimuth-points",
        "16",
        "--screening-tolerance",
        "1e-14",
        "--density-fitting",
        "cuda" if fitted else "none",
        "--charge",
        str(charge),
        "--multiplicity",
        str(multiplicity),
    )
    selector = f"{functional.lower()}-{variant.lower()}-{spin}"
    bare = invoke_cli(
        tmp_path,
        atoms,
        operation="hessian",
        default_grid=True,
        extra=("--method", f"{functional.lower()}-{spin}", *common),
    )
    total = invoke_cli(
        tmp_path,
        atoms,
        operation="hessian",
        default_grid=True,
        extra=("--method", selector, *common),
    )
    assert total["method_ir_identity"] == graph.identity
    assert total["diagnostics"]["correction_directions"] == positions.size
    assert total["diagnostics"]["completed_directions"] == positions.size
    assert bare["diagnostics"]["correction_directions"] == 0
    actual = np.asarray(total["matrix"])
    delta = actual - bare["matrix"]
    axes = np.eye(positions.size).reshape(positions.size, *positions.shape)

    def stencil(step: float) -> np.ndarray:
        return np.column_stack(
            [
                (gradient(positions + step * axis) - gradient(positions - step * axis))
                / (2 * step)
                for axis in axes
            ]
        )

    expected = (4 * stencil(0.0002) - stencil(0.0004)) / 3
    if spec.s9:
        # ATM is a separately detectable contribution, not merely a catalog label.
        original = damping
        damping = interface.RationalDampingParam(
            s6=spec.s6, s8=spec.s8, s9=0, a1=spec.a1, a2=spec.a2
        )
        assert np.max(np.abs(expected - stencil(0.0002))) > 1e-9
        damping = original
    assert np.max(np.abs(expected)) > 1e-6
    record_property("max_d3_error_hartree_per_bohr2", np.max(np.abs(delta - expected)))
    np.testing.assert_allclose(delta, expected, atol=2e-10, rtol=0)
    np.testing.assert_allclose(actual, actual.T, atol=3e-9, rtol=0)
    np.testing.assert_allclose(
        actual @ np.tile(np.eye(3), (len(atoms), 1)), 0, atol=3e-9, rtol=0
    )
    settings = {
        "method": graph,
        "device": "cuda",
        "basis": "sto-3g",
        "basis_representation": "spherical",
        "density_fitting": "cuda" if fitted else "none",
        "max_iterations": 240,
        "energy_tolerance": 1e-13,
        "density_tolerance": 1e-11,
        "screening_tolerance": 1e-14,
        "ks_options": KsOptions(
            grid=GridSpec(radial_points=16, angular_polar=8, angular_azimuth=16)
        ),
    }
    plan = (
        Calculator(**settings)
        .estimate_resources(
            [atoms],
            charges=[charge],
            multiplicities=[multiplicity],
            properties=("energy",),
        )
        .require_feasible()
    )
    requests = {request.name: request for request in plan.requests}
    assert requests["ks"].identity.observables == ("energy",)
    assert (
        json.loads(requests["d3"].identity.schedule)["inventory_owner"]
        == "native-ks-d3-layout-v1"
    )
    budget = ResourceBudget(
        host_bytes=plan.peak_bytes["host"] + 16 * positions.size**2,
        device_bytes=plan.peak_bytes["device"] + (128 << 20),
    )
    calculator = Calculator(**settings, resource_budget=budget)
    assert calculator.second_order_capabilities == {"hvp", "hessian"}
    endpoint = {
        "charge": charge,
        "multiplicity": multiplicity,
        "solver_options": GMRESOptions(atol=1e-12, rtol=1e-11),
    }
    dense = calculator.hessian(atoms, **endpoint)
    action = calculator.hessian_vector_product(atoms, direction, **endpoint)
    for result, count in ((dense, positions.size), (action, 1)):
        assert result.diagnostics["completed_directions"] == count
        resources = result.diagnostics["resources"]
        ledger = resources["hessian_response"]["device_ledger"]
        assert ledger["peak_bytes"] <= budget.device_bytes
        assert ledger["rejected_allocations"] == 0
        assert (
            ledger["live_bytes"]
            == resources["observation"]["device_ledger"]["live_bytes"]
        )
    np.testing.assert_allclose(dense.matrix, actual, atol=2e-9, rtol=0)
    np.testing.assert_allclose(
        action.value.ravel(), actual @ direction.ravel(), atol=2e-9, rtol=0
    )


@pytest.mark.parametrize(
    ("method", "fitted", "charge"),
    (
        pytest.param("pbe-rks", False, 0, id="direct-rks"),
        pytest.param("pbe-rks", True, 0, id="fitted-rks"),
        pytest.param("pbe-uks", True, 1, id="fitted-uks"),
        pytest.param("pbe0-rks", False, 0, id="direct-f-pbe0-rks"),
        pytest.param("pbe0-uks", False, 1, id="direct-f-pbe0-uks"),
        pytest.param("b3lyp-rks", False, 0, id="direct-f-b3lyp-rks"),
        pytest.param("b3lyp-uks", False, 1, id="direct-f-b3lyp-uks"),
        pytest.param("pbe0-rks", True, 0, id="fitted-pbe0-rks"),
        pytest.param("pbe0-uks", True, 0, id="fitted-pbe0-uks"),
        pytest.param("b3lyp-rks", True, 0, id="fitted-b3lyp-rks"),
        pytest.param("b3lyp-uks", True, 0, id="fitted-b3lyp-uks"),
        pytest.param("r2scan+rvv10-rks", True, 0, id="fitted-rvv10-shared"),
        pytest.param("r2scan+rvv10-uks", True, 1, id="fitted-rvv10-independent"),
        pytest.param("r2scan+rvv10-rks", False, 0, id="direct-rvv10-shared"),
        pytest.param("r2scan+rvv10-uks", False, 1, id="direct-rvv10-independent"),
    ),
)
def test_public_cuda_hessian_and_hvp_share_global_resource_budget(
    monkeypatch: pytest.MonkeyPatch,
    method: str,
    fitted: bool,
    charge: int,
) -> None:
    """Both calls use a real resource ledger without changing the SCF model."""
    assert os.environ.get("SLURM_JOB_ID"), "real GPU tests require Slurm"
    atoms = [("H", (0.03, -0.02, 0.01)), ("H", (0.15, 0.13, 1.5))]
    unrestricted = method.endswith("-uks")
    rvv10 = method.startswith("r2scan+rvv10-")
    direct_hybrid = not fitted and method.startswith(("pbe0-", "b3lyp-"))
    independent_grid = rvv10 and unrestricted
    direction = np.array([[0.02, -0.03, 0.1], [-0.02, 0.03, -0.1]])
    if unrestricted and not charge:
        # The hybrid doublet has occupied alpha and beta channels.
        atoms.append(("H", (1.9, 0.3, 0.65)))
        direction = np.array(
            [[0.01, -0.03, 0.1], [-0.02, 0.02, -0.08], [0.01, 0.01, -0.02]]
        )
    settings = {
        "method": method,
        "device": "cuda",
        "basis": "sto-3g",
        "basis_representation": "spherical",
        "initial_guess": None,
        "max_iterations": 240,
        "energy_tolerance": 1e-13,
        "density_tolerance": 1e-11,
        "ks_options": KsOptions(
            grid=GridSpec(radial_points=8, angular_polar=4, angular_azimuth=8),
            nonlocal_grid=GridSpec(radial_points=6, angular_polar=3, angular_azimuth=6)
            if independent_grid
            else None,
        ),
    }
    if direct_hybrid:
        settings["basis"] = BasisSet(
            "Budget regression H orbital f basis",
            (
                ElementBasis(
                    1,
                    (
                        BasisShell(0, ("0.73",), (("1",),)),
                        BasisShell(3, ("0.56",), (("1",),)),
                    ),
                ),
            ),
            BasisProvenance("test high-angular orbital", "1", "fixture", "b" * 64),
            representation="spherical",
        )
    if fitted:
        settings["density_fitting"] = "cuda"
        if unrestricted:
            settings["density_fitting_memory_budget_bytes"] = 32 << 20
            settings["auxiliary_basis"] = BasisSet(
                "Budget regression H fitting basis",
                (
                    ElementBasis(
                        1,
                        (
                            BasisShell(0, ("0.8",), (("1",),)),
                            BasisShell(1, ("0.4",), (("1",),)),
                        ),
                    ),
                ),
                BasisProvenance(
                    "test explicit auxiliary topology", "1", "fixture", "a" * 64
                ),
                representation="spherical",
            )
    multiplicity = 2 if unrestricted else 1
    electronic = {"charge": charge, "multiplicity": multiplicity}
    reference = Calculator(**settings)
    matrix = reference.hessian(atoms, **electronic).matrix
    scf_plan = reference.estimate_resources(
        [atoms],
        charges=[charge],
        multiplicities=[multiplicity],
        properties=("energy",),
    ).require_feasible()
    if direct_hybrid:
        assert scf_plan.requests[0].identity.observables == ("energy",)
        assert any(
            estimate.name == "all KS radial Fock host envelopes" and estimate.bytes > 0
            for estimate in scf_plan.estimates
        )
    if rvv10:
        inventory = json.loads(scf_plan.requests[0].identity.topology)["items"][0]
        assert (
            inventory["grid_points"] != inventory["nonlocal_grid_points"]
        ) is independent_grid
        assert any(
            estimate.name == "all KS nonlocal host envelopes" and estimate.bytes > 0
            for estimate in scf_plan.estimates
        )
    if fitted:
        assert scf_plan.requests[0].identity.provider == "native-ks-fitted-v1"
        assert any(
            estimate.name == "all KS fitted Fock host envelopes" and estimate.bytes > 0
            for estimate in scf_plan.estimates
        )
        if unrestricted:
            inventory = json.loads(scf_plan.requests[0].identity.topology)["items"][0]
            assert inventory["auxiliary"]["nbf"] == 4 * len(atoms)
            assert inventory["orbital"]["nbf"] == len(atoms)
            assert any(
                estimate.name == "all KS device coulomb" and estimate.bytes == 32 << 20
                for estimate in scf_plan.estimates
            )
    budget = ResourceBudget(
        host_bytes=scf_plan.peak_bytes["host"] + 2 * direction.size**2 * 8,
        device_bytes=scf_plan.peak_bytes["device"] + (96 << 20),
    )
    calculator = Calculator(**settings, resource_budget=budget)
    hessian = calculator.hessian(atoms, **electronic)
    action = calculator.hessian_vector_product(atoms, direction, **electronic)
    np.testing.assert_allclose(matrix, hessian.matrix, atol=3e-9, rtol=0)
    np.testing.assert_allclose(
        matrix @ direction.ravel(), action.value.ravel(), atol=3e-9, rtol=0
    )
    if direct_hybrid and method == "pbe0-uks":
        # The empty beta channel triggers the cancellation-sensitive AO adjoint.
        # Check the physical endpoint independently, not just budgeted parity.
        path = Path(__file__).parents[1] / "data/xc/pbe0-uks-direct-f-hessian.json"
        oracle = np.asarray(
            json.loads(path.read_text())["matrix_hartree_per_bohr_squared"]
        )
        np.testing.assert_allclose(hessian.matrix, oracle, atol=3e-8, rtol=0)
        np.testing.assert_allclose(
            action.value.ravel(), oracle @ direction.ravel(), atol=3e-8, rtol=0
        )
    for result in (hessian, action):
        resources = result.diagnostics["resources"]
        assert resources["plan"]["peak_bytes"]["host"] <= budget.host_bytes
        response = resources["hessian_response"]["device_ledger"]
        scf = resources["observation"]["device_ledger"]
        assert response["live_bytes"] == scf["live_bytes"]
        assert (
            response["peak_bytes"]
            >= response["live_bytes"] + result.diagnostics["device_bytes"]
        )
        assert response["peak_bytes"] <= budget.device_bytes
        assert bool(result.diagnostics["df_workspace_bytes"]) == fitted
        if rvv10:
            assert result.diagnostics["vv10_reference_pairs"] > 0
    tight = Calculator(
        **settings,
        resource_budget=replace(
            budget, device_bytes=scf_plan.peak_bytes["device"] + 2048
        ),
    )
    with pytest.raises(MemoryError, match="aggregate device budget") as failure:
        tight.hessian(atoms, **electronic)
    resources = failure.value.resource_diagnostics
    assert resources["phase"] == "hessian_response"
    before = resources["observation"]["device_ledger"]
    after = resources["hessian_response"]["device_ledger"]
    assert before["live_bytes"] == after["live_bytes"]

    def unexpected_prepare(*_args: object, **_kwargs: object) -> None:
        pytest.fail("an infeasible Hessian must not prepare SCF")

    monkeypatch.setattr(Calculator, "prepare_batch", unexpected_prepare)
    if rvv10:
        with pytest.raises(MemoryError, match="nonlocal_memory_budget_bytes"):
            Calculator(
                **{
                    **settings,
                    "ks_options": replace(
                        settings["ks_options"], nonlocal_memory_budget_bytes=1
                    ),
                },
                resource_budget=budget,
            ).hessian(atoms, **electronic)
    for too_small in (
        replace(budget, host_bytes=budget.host_bytes - 1),
        replace(budget, device_bytes=0),
    ):
        with pytest.raises(MemoryError, match="no supported plan fits"):
            Calculator(**settings, resource_budget=too_small).hessian(
                atoms, **electronic
            )


@pytest.mark.parametrize(
    ("fitted", "mode", "fitted_range", "separate", "spin"),
    (
        (False, "direct", False, True, "rks"),
        (True, "direct", False, False, "uks"),
        (True, "df", True, True, "uks"),
        (True, "auto", True, False, "rks"),
        (True, "auto", False, True, "rks"),
    ),
    ids=("direct", "mixed", "explicit-ri", "auto-ri", "auto-exact-retry"),
)
def test_wb97mv_public_hessian_radial_resource_budget(
    monkeypatch: pytest.MonkeyPatch,
    fitted: bool,
    mode: str,
    fitted_range: bool,
    separate: bool,
    spin: str,
) -> None:
    """Budgeting preserves the accepted radial operator, including an auto retry."""
    assert os.environ.get("SLURM_JOB_ID"), "real GPU tests require Slurm"
    monkeypatch.setenv("GENERATIVEQC_DF_RANGE_EXCHANGE", mode)
    atoms = [("H", (0.03, -0.02, 0.01)), ("H", (0.15, 0.13, 1.5))]
    direction = np.array([[0.02, -0.03, 0.1], [-0.02, 0.03, -0.1]])
    charge = int(spin == "uks")
    multiplicity = charge + 1
    settings = {
        "method": f"wb97m-v-{spin}",
        "device": "cuda",
        "basis": "sto-3g",
        "basis_representation": "spherical",
        "initial_guess": None,
        "max_iterations": 240,
        "energy_tolerance": 1e-13,
        "density_tolerance": 1e-11,
        "screening_tolerance": 1e-14,
        "density_fitting": "cuda" if fitted else "none",
        "ks_options": KsOptions(
            grid=GridSpec(radial_points=8, angular_polar=4, angular_azimuth=8),
            nonlocal_grid=GridSpec(radial_points=6, angular_polar=3, angular_azimuth=6)
            if separate
            else None,
        ),
    }
    if fitted:
        settings["density_fitting_memory_budget_bytes"] = (32 << 20) + 1
    if fitted and mode == "auto" and not fitted_range:
        from test_lr_df_admission import _boundary_fixture

        atoms, basis, threshold = _boundary_fixture()
        settings.update(
            basis=basis,
            basis_representation=basis.representation,
            auxiliary_basis=basis,
            density_fitting_relative_threshold=threshold,
        )
    electronic = {"charge": charge, "multiplicity": multiplicity}
    reference = Calculator(**settings)
    matrix = reference.hessian(atoms, **electronic).matrix
    scf = reference.estimate_resources(
        [atoms], charges=[charge], multiplicities=[multiplicity], properties=("energy",)
    ).require_feasible()
    assert (
        json.loads(scf.requests[0].identity.schedule)["wb97mv_range_exchange"] == mode
    )
    assert scf.requests[0].identity.observables == ("energy",)
    budget = ResourceBudget(
        host_bytes=scf.peak_bytes["host"] + 2 * direction.size**2 * 8,
        device_bytes=scf.peak_bytes["device"] + (96 << 20),
    )
    owner = Calculator(**settings, resource_budget=budget)
    dense = owner.hessian(atoms, **electronic)
    action = owner.hessian_vector_product(atoms, direction, **electronic)
    np.testing.assert_allclose(dense.matrix, matrix, atol=3e-9, rtol=0)
    np.testing.assert_allclose(
        action.value.ravel(), matrix @ direction.ravel(), atol=3e-9, rtol=0
    )
    for output in (dense, action):
        receipt = output.diagnostics
        assert receipt["vv10_reference_pairs"] > 0
        assert bool(receipt["exact_primitive_components"]) is not fitted_range
        assert (
            receipt["df_value_passes"]
            == (fitted + fitted_range) * receipt["completed_directions"]
        )
        resources = receipt["resources"]
        response = resources["hessian_response"]["device_ledger"]
        assert response["peak_bytes"] <= budget.device_bytes
        assert (
            response["live_bytes"]
            == resources["observation"]["device_ledger"]["live_bytes"]
        )
    tight = Calculator(
        **settings,
        resource_budget=replace(budget, device_bytes=scf.peak_bytes["device"] + 2048),
    )
    with pytest.raises(MemoryError, match="aggregate device budget") as failure:
        tight.hessian(atoms, **electronic)
    resources = failure.value.resource_diagnostics
    assert resources["phase"] == "hessian_response"
    assert (
        resources["observation"]["device_ledger"]["live_bytes"]
        == (resources["hessian_response"]["device_ledger"]["live_bytes"])
    )


def test_r2scan3c_default_grid_replays_water(tmp_path: Path) -> None:
    """The default arena retains some jets and replays the rest on device."""
    atoms = [
        ("O", (0.0, 0.0, 0.0)),
        ("H", (1.43233673, 0.0, 1.10715266)),
        ("H", (-1.40233673, 0.1, 1.10715266)),
    ]
    direction = np.array(
        [[0.01, -0.03, 0.10], [-0.04, 0.02, 0.01], [0.03, 0.01, -0.11]]
    )
    common = (
        "--method",
        "r2scan-3c-rks",
        "--density-fitting",
        "cuda",
        "--screening-tolerance",
        "1e-14",
    )
    full = invoke_cli(
        tmp_path, atoms, operation="hessian", extra=common, default_grid=True
    )
    resident = invoke_cli(
        tmp_path,
        atoms,
        direction=direction,
        default_grid=True,
        extra=(*common, "--grid-budget-bytes", str(512 << 20)),
    )
    replay = full["diagnostics"]
    cached = resident["diagnostics"]
    assert 0 < replay["grid_ao_replayed_primitive_term_bound"] <= 1 << 33
    assert (
        0
        < replay["grid_ao_prepared_primitive_term_bound"]
        < cached["grid_ao_prepared_primitive_term_bound"]
    )
    assert 0 < replay["grid_geometry_workspace_bytes"] <= 128 << 20
    assert (
        replay["completed_directions"]
        == replay["correction_directions"]
        == direction.size
    )
    assert cached["grid_ao_replayed_primitive_term_bound"] == 0
    matrix = np.asarray(full["matrix"]).reshape(direction.size, direction.size)
    np.testing.assert_allclose(matrix, matrix.T, atol=2e-7, rtol=0)
    translations = np.tile(np.eye(3), (len(atoms), 1))
    np.testing.assert_allclose(matrix @ translations, 0, atol=1e-8, rtol=0)
    np.testing.assert_allclose(
        matrix @ direction.ravel(), np.ravel(resident["value"]), atol=1e-8, rtol=0
    )


@pytest.mark.parametrize("functional", ("lda", "pbe", "pbe0", "b3lyp"))
@pytest.mark.parametrize("spin", ("rks", "uks"))
@pytest.mark.parametrize("fitted", (False, True), ids=("direct", "df"))
def test_public_cuda_semilocal_ecp_hvp(
    tmp_path: Path, functional: str, spin: str, fitted: bool
) -> None:
    """Both public endpoints must use the native residual scalar ECP source."""
    from test_wb97mv_ecp_hessian import synthetic_basis

    atoms = [("Na", (0.13, -0.21, 0.17)), ("H", (0.43, 0.19, 2.1))]
    direction = np.array([[0.02, -0.03, 0.1], [-0.02, 0.03, -0.1]])
    basis = synthetic_basis()
    orbital = tmp_path / "Orbital ECP.json"
    basis.write(orbital)
    method = f"{functional}-{spin}"
    charge, multiplicity = (0, 1) if spin == "rks" else (1, 2)
    extra = (
        "--method",
        method,
        "--basis",
        str(orbital),
        "--charge",
        str(charge),
        "--multiplicity",
        str(multiplicity),
    )
    auxiliary = None
    if fitted:
        auxiliary = BasisSet(
            "Synthetic Na and H auxiliary fitting basis",
            (
                ElementBasis(11, (BasisShell(0, ("0.8",), (("1",),)),)),
                ElementBasis(1, (BasisShell(0, ("0.65",), (("1",),)),)),
            ),
            BasisProvenance("synthetic test parameters", "1", "fixture", "a" * 64),
            representation="spherical",
        )
        auxiliary_file = tmp_path / "Auxiliary.json"
        auxiliary.write(auxiliary_file)
        extra += ("--density-fitting", "cuda", "--auxiliary-basis", str(auxiliary_file))
    cli = invoke_cli(tmp_path, atoms, direction=direction, extra=extra)
    settings = {
        "method": method,
        "device": "cuda",
        "basis": basis,
        "basis_representation": "spherical",
        "density_fitting": "cuda" if fitted else "none",
        "auxiliary_basis": auxiliary,
        "max_iterations": 240,
        "energy_tolerance": 1e-13,
        "density_tolerance": 1e-11,
        "ks_options": KsOptions(
            grid=GridSpec(radial_points=8, angular_polar=4, angular_azimuth=8)
        ),
    }
    calculator = Calculator(**settings)
    budget = None
    if fitted:
        scf = calculator.estimate_resources(
            [atoms],
            charges=[charge],
            multiplicities=[multiplicity],
            properties=("energy",),
        ).require_feasible()
        assert scf.requests[0].identity.observables == ("energy",)
        assert (
            json.loads(scf.requests[0].identity.schedule)["ecp_device_inventory"]
            == "native-ecp-v1"
        )
        budget = ResourceBudget(
            host_bytes=scf.peak_bytes["host"] + 16 * direction.size**2,
            device_bytes=scf.peak_bytes["device"] + (96 << 20),
        )
        calculator = Calculator(**settings, resource_budget=budget)
    result = calculator.hessian_vector_product(
        atoms,
        direction,
        charge=charge,
        multiplicity=multiplicity,
        solver_options=GMRESOptions(atol=1e-12, rtol=1e-11),
    )
    assert cli["diagnostics"]["ecp_ad_backend"] == "cuda"
    assert result.diagnostics["ecp_ad_backend"] == "cuda"
    df_backend = "cuda" if fitted else None
    assert cli["diagnostics"]["df_integral_ad_backend"] == df_backend
    assert result.diagnostics["df_integral_ad_backend"] == df_backend
    np.testing.assert_allclose(result.value, cli["value"], atol=2e-8, rtol=0)
    np.testing.assert_allclose(result.value.sum(axis=0), 0, atol=2e-8, rtol=0)
    if budget is not None:
        resources = result.diagnostics["resources"]
        response = resources["hessian_response"]["device_ledger"]
        assert response["peak_bytes"] <= budget.device_bytes
        assert (
            response["live_bytes"]
            == resources["observation"]["device_ledger"]["live_bytes"]
        )

    # Qualify full-matrix publication on the combined open-shell hybrid path
    # once; the native oracle suite already checks all eight fitted ECP matrices.
    if fitted and functional == "b3lyp" and spin == "uks":
        assert budget is not None
        with pytest.raises(NotImplementedError, match="does not cover forces"):
            calculator.estimate_resources(
                [atoms],
                charges=[charge],
                multiplicities=[multiplicity],
                properties=("energy", "forces"),
            )
        tight = Calculator(
            **settings,
            resource_budget=replace(
                budget, device_bytes=scf.peak_bytes["device"] + 2048
            ),
        )
        with pytest.raises(MemoryError, match="aggregate device budget") as failure:
            tight.hessian(atoms, charge=charge, multiplicity=multiplicity)
        resources = failure.value.resource_diagnostics
        assert resources["phase"] == "hessian_response"
        assert (
            resources["hessian_response"]["device_ledger"]["live_bytes"]
            == resources["observation"]["device_ledger"]["live_bytes"]
        )
        cli_full = invoke_cli(tmp_path, atoms, operation="hessian", extra=extra)
        full = calculator.hessian(
            atoms,
            charge=charge,
            multiplicity=multiplicity,
            solver_options=GMRESOptions(atol=1e-12, rtol=1e-11),
        )
        for diagnostics in (cli_full["diagnostics"], full.diagnostics):
            assert diagnostics["completed_directions"] == direction.size
            assert diagnostics["posthoc_symmetrization"] is False
            assert diagnostics["ecp_ad_backend"] == "cuda"
            assert diagnostics["df_integral_ad_backend"] == "cuda"
        np.testing.assert_allclose(full.matrix, cli_full["matrix"], atol=2e-8, rtol=0)
        np.testing.assert_allclose(full.matrix, full.matrix.T, atol=2e-8, rtol=0)
        np.testing.assert_allclose(
            full.matrix @ direction.ravel(), result.value.ravel(), atol=2e-8, rtol=0
        )


@pytest.mark.parametrize(
    "auxiliary_order,orbital_angular,range_policy",
    (
        (None, 0, "direct"),
        (2, 0, "direct"),
        (2, 0, "df"),
        (4, 0, "direct"),
        (4, 3, "direct"),
    ),
    ids=(
        "direct",
        "d-auxiliary",
        "d-auxiliary-radial-df",
        "g-auxiliary",
        "f-orbital-g-auxiliary",
    ),
)
def test_cpp_cli_and_budgeted_wb97mv_uks_ecp_hvp(
    tmp_path: Path,
    auxiliary_order: int | None,
    orbital_angular: int,
    range_policy: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from test_wb97mv_ecp_hessian import synthetic_basis

    monkeypatch.setenv("GENERATIVEQC_DF_RANGE_EXCHANGE", range_policy)
    atoms = [
        ("Na", (0.13, -0.21, 0.17)),
        ("H", (0.43, 0.19, 2.1)),
        ("H", (-0.3, 0.4, -1.6)),
    ]
    direction = np.array(
        [[0.02, -0.03, 0.1], [-0.04, 0.01, -0.02], [0.02, 0.02, -0.08]]
    )
    orbital = synthetic_basis()
    if orbital_angular:
        orbital = replace(
            orbital,
            elements=tuple(
                replace(
                    element,
                    shells=element.shells
                    + (BasisShell(orbital_angular, ("0.48",), (("1",),)),),
                )
                if element.atomic_number == 11
                else element
                for element in orbital.elements
            ),
        )
    auxiliary = (
        None
        if auxiliary_order is None
        else BasisSet(
            "synthetic spherical auxiliary CLI fixture",
            (
                ElementBasis(
                    11,
                    (
                        BasisShell(0, ("0.8",), (("1",),)),
                        BasisShell(auxiliary_order, ("0.6",), (("1",),)),
                    ),
                ),
                ElementBasis(1, (BasisShell(0, ("0.65",), (("1",),)),)),
            ),
            BasisProvenance(
                "synthetic test parameters", "1", "test-only fixture", "a" * 64
            ),
            representation="spherical",
        )
    )
    orbital_file, auxiliary_file = (
        tmp_path / "Orbital ECP.json",
        tmp_path / "Auxiliary.json",
    )
    orbital.write(orbital_file)
    fitted_options = ()
    if auxiliary is not None:
        auxiliary.write(auxiliary_file)
        fitted_options = (
            "--density-fitting",
            "cuda",
            "--auxiliary-basis",
            str(auxiliary_file),
        )
    extra = (
        "--method",
        "wb97m-v-uks",
        "--multiplicity",
        "2",
        "--basis",
        str(orbital_file),
        *fitted_options,
    )
    result = invoke_cli(tmp_path, atoms, direction=direction, extra=extra)
    matrix = (
        invoke_cli(tmp_path, atoms, operation="hessian", extra=extra)["matrix"]
        if range_policy == "df"
        else None
    )
    assert result["diagnostics"]["response_iterations"] > 0
    assert result["diagnostics"]["ecp_ad_backend"] == "cuda"
    assert result["diagnostics"]["df_integral_ad_backend"] == (
        "cuda" if auxiliary is not None else None
    )
    qualify_native_frontend(
        monkeypatch,
        atoms,
        direction,
        result["value"],
        matrix=matrix,
        multiplicity=2,
        budgeted=True,
        method="wb97m-v-uks",
        basis=orbital,
        auxiliary_basis=auxiliary,
        density_fitting="cuda" if auxiliary is not None else "none",
    )
    if auxiliary_order == 4 or range_policy == "df":
        from test_native_wb97mv_stationary import _native, _qualify

        reference = _native(
            2,
            True,
            True,
            True,
            False,
            auxiliary_d=auxiliary_order == 2,
            auxiliary_g=auxiliary_order == 4,
            orbital_angular=orbital_angular,
            fitted_range=range_policy == "df",
        )
        np.testing.assert_array_equal(
            np.asarray(reference["motion"]).reshape(direction.shape), direction
        )
        reference["hvp"] = np.asarray(result["value"]).ravel().tolist()
        reference["oracle_consumer"] = "cpp-cli-and-native-frontend"
        _qualify(reference, fitted=True, spherical=True, ecp=True, full=False)
        return
    cpu = Calculator(
        method="wb97m-v-uks",
        device="cpu",
        basis=orbital,
        auxiliary_basis=auxiliary,
        density_fitting="cpu" if auxiliary is not None else "none",
        basis_representation="spherical",
        initial_guess=None,
        max_iterations=240,
        energy_tolerance=1e-13,
        density_tolerance=1e-11,
        ks_options=KsOptions(
            grid=GridSpec(radial_points=8, angular_polar=4, angular_azimuth=8)
        ),
    )
    energy = cpu.singlepoint(atoms, multiplicity=2, properties=("energy",)).energy
    np.testing.assert_allclose(result["energy_hartree"], energy, atol=5e-9, rtol=0)
    expected = cpu.hessian_vector_product(
        atoms,
        direction,
        multiplicity=2,
        cache=tmp_path,
        solver_options=GMRESOptions(atol=1e-12, rtol=1e-11),
    )
    np.testing.assert_allclose(result["value"], expected.value, atol=3e-8, rtol=2e-6)
