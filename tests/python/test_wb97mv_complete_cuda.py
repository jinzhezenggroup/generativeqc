"""Complete public CUDA energy/force gates against independent GPU4PySCF.

Run GENERATIVEQC_TEST_WB97MV_CUDA=1 through Slurm main with one 5090 and a finite
time limit. Grid motion, partition response, SR/LR exchange and self-consistent
VV10 are all included in both engines; no component-only success promotes API.
"""

import ast
import os
import typing
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest


def test_native_wb97mv_pairs_stationary_one_electron_sources() -> None:
    source = (
        Path(__file__).resolve().parents[2] / "src/methods/dft_method.cpp"
    ).read_text(encoding="utf-8")
    begin = source.index("generativeqc_status cuda_integral_gradient(")
    end = source.index("Result execute(bool compute_forces)", begin)
    bridge = source[begin:end]
    assert "resident_final_stationary_weights" in bridge
    assert "resident_weights.density" in bridge
    assert "resident_weights.weighted_density" in bridge
    assert "execute_prepared_cuda_stationary_one_electron_pair(" in bridge
    assert "fock_.cuda_direct_source()" in bridge
    assert "execute_cuda_stationary_one_electron_pair(" in bridge  # bounded fallback
    assert "auto density = (*cached_density)[0]" not in bridge
    assert "execute_cuda_one_electron_gradient(" not in bridge


def test_wb97mv_geometry_layout_admits_f_without_spdf_integral_schedule() -> None:
    """WB97M-V geometry composition must not require the generic SPD inventory."""
    from generativeqc._stationary_cuda import _layout

    packed = np.zeros(3 + 2 + 16, dtype=np.float64)
    packed[3:5] = (1.0, 1.0)
    ao = packed[5:].reshape(1, 16)[0]
    ao[0:4] = (0, 0, 1, 1)
    ao[4:7] = (3, 0, 0)
    ao[7] = 1.0
    basis = SimpleNamespace(
        natom=1,
        nprimitive=1,
        packed=packed,
        shells=(SimpleNamespace(angular_momentum=3),),
    )

    _, _, expansions, requests = _layout(basis, integral_derivatives=False)
    assert expansions == ((("xxx", 1.0),),)
    assert requests == (("nuclear", ()),)
    with pytest.raises(NotImplementedError, match="s/p/d"):
        _layout(basis)


@pytest.mark.parametrize(
    "basis,expected",
    [("sto-3g", True), ("def2-svp", True), ("def2-tzvp", True), ("def2-tzvpd", False)],
)
def test_wb97mv_cuda_named_force_basis_domain(basis: str, expected: bool) -> None:
    """A non-bundled diffuse name must still require an explicit local snapshot."""
    from generativeqc.ks import cuda_wb97mv_force_basis_eligible

    assert cuda_wb97mv_force_basis_eligible(basis) is expected


def test_wb97mv_cuda_local_force_basis_admits_f_but_not_g() -> None:
    """Geometry-only f admission must not expand the generic GPU angular domain."""
    from generativeqc import load_basis
    from generativeqc.ks import cuda_wb97mv_force_basis_eligible

    basis = load_basis(
        Path(__file__).resolve().parents[2]
        / "benchmarks/results/omol25-wb97mv-20261001/def2-tzvpd-ho.json"
    )
    assert cuda_wb97mv_force_basis_eligible(basis)
    higher = replace(
        basis,
        elements=tuple(
            replace(
                element,
                shells=(
                    *element.shells,
                    replace(element.shells[-1], angular_momentum=4),
                ),
            )
            if element.atomic_number == 8
            else element
            for element in basis.elements
        ),
    )
    assert not cuda_wb97mv_force_basis_eligible(higher)


def test_wb97mv_cuda_force_gate_treats_auto_as_scf_component_policy() -> None:
    """AUTO changes SCF components, not the strict stationary derivative owner."""
    source = (
        Path(__file__).resolve().parents[2] / "python/generativeqc/calculator.py"
    ).read_text(encoding="utf-8")
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
    assignment = next(
        node
        for node in ast.walk(constructor)
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id == "cuda_wb97mv_force"
            for target in node.targets
        )
    )
    segment = ast.get_source_segment(source, assignment)
    assert segment is not None
    assert "_precision_mode" not in segment
    for guard in (
        'self._device_name == "cuda"',
        'self._method_name.startswith("wb97m-v")',
        "not basis_has_ecp",
        "self._ks_options is not None",
        "cuda_wb97mv_force_basis_eligible(self._basis)",
    ):
        assert guard in segment


def test_cuda_geometry_only_f_nuclear_pair(tmp_path: Path) -> None:
    """Qualify nuclear dispatch independently of costly SCF and grid composition."""
    if os.environ.get("GENERATIVEQC_TEST_WB97MV_CUDA") != "1":
        pytest.skip("set GENERATIVEQC_TEST_WB97MV_CUDA=1 inside Slurm")
    assert os.environ.get("SLURM_JOB_ID"), "real GPU tests require Slurm"
    from generativeqc._stationary_cuda import _CudaSources
    from generativeqc.profiles import find_nvcc
    from generativeqc_compiler.common.cuda_adapter import CudaCompilerAdapter
    from generativeqc_compiler.common.cuda_target import cuda_target_info
    from generativeqc_compiler.dft import NativeAO
    from generativeqc_compiler.integral.first_derivative_native import (
        emit_first_derivative_cuda,
    )
    from generativeqc_compiler.method import resolve_method
    from generativeqc_compiler.method.stationary_cuda import compile_stationary_cuda
    from generativeqc_compiler.method.stationary_gradient import (
        StationaryGradientPlan,
        StationaryMeanField,
    )

    nvcc = find_nvcc()
    assert nvcc is not None, "set CUDACXX or CUDA_PATH for generated force kernels"
    compiler = CudaCompilerAdapter(Path(nvcc), cuda_target_info("sm_120"))
    plan = StationaryGradientPlan(
        resolve_method("WB97M-V"),
        StationaryMeanField("libxc-7.0/work-mgga-v1/smooth-lr-a1.35-order16"),
    )
    artifact = compile_stationary_cuda(
        emit_first_derivative_cuda((("nuclear", ()),)),
        functional=4,
        plan=plan,
        iterations=3,
        compiler=compiler,
        cache=tmp_path,
    )
    atoms = [("O", (0.02, -0.03, 0.04)), ("H", (0.1, 1.43, 1.1))]
    with (
        NativeAO(
            atoms, "def2-tzvp", representation="spherical", multiplicity=2
        ) as basis,
        _CudaSources(
            basis,
            artifact,
            compiler,
            0,
            1,
            1,
            16 << 20,
            integral_derivatives=False,
        ) as sources,
    ):
        assert not sources.component_mode
        sources.reset_geometry(1e-12)
        sources.nuclear_all(np.array([8.0, 1.0]))
        actual = sources.finish()["nuclear"]
    separation = np.asarray(atoms[0][1]) - atoms[1][1]
    first = -8.0 * separation / np.linalg.norm(separation) ** 3
    np.testing.assert_allclose(actual, [first, -first], atol=1e-12, rtol=0)


def _assert_wb97mv_auto_component_precision(result: typing.Any) -> None:
    precision = result.precision
    assert precision is not None
    assert precision["requested_mode"] == "auto"
    assert precision["effective_bits"] == 32
    assert precision["strict_refinement_applied"] is True
    assert precision["refinement_iterations"] >= 1
    assert precision["complete"] is True
    assert precision["operator_inventory_complete"] is True
    assert precision["operator_work_counters_valid"] is True

    operators = precision["operators"]
    mixed_names = {
        row["name"]
        for row in operators
        if row["arithmetic_mode"] == "mixed" and row["count"] > 0
    }
    assert mixed_names == {"coulomb_j", "coulomb_recurrence"}
    assert any(
        row["name"] == "coulomb_recurrence"
        and row["compute"] == "fp32"
        and row["accumulation"] == "fp64"
        and row["reduction"] == "fp64"
        and row["count"] > 0
        for row in operators
    )
    for name in ("exchange_k", "xc", "nonlocal_correlation"):
        rows = [row for row in operators if row["name"] == name]
        assert rows
        assert all(
            row["storage"] == "fp64"
            and row["compute"] == "fp64"
            and row["accumulation"] == "fp64"
            and row["reduction"] == "fp64"
            and row["arithmetic_mode"] == "strict"
            for row in rows
        )
    timeline = precision["scf_fock_timeline"]
    assert timeline and timeline[-1]["kind"] == "final_audit"
    assert timeline[-1]["state"] == precision["returned_state_identity"]


@pytest.mark.parametrize(
    "method,spin,atoms",
    [
        ("wb97m-v", 0, [("H", (0.0, 0.0, 0.0)), ("H", (0.15, 0.13, 1.5))]),
        (
            "wb97m-v-uks",
            1,
            [
                ("H", (0.0, 0.0, 0.0)),
                ("H", (0.15, 0.13, 1.5)),
                ("H", (1.8, -0.1, -0.3)),
            ],
        ),
    ],
)
def test_wb97mv_auto_matches_fp64_cold_warm_and_moved(
    method: str, spin: int, atoms: typing.Any
) -> None:
    """AUTO lowers only Direct J and returns the same strict physical E/F state."""
    if os.environ.get("GENERATIVEQC_TEST_WB97MV_CUDA") != "1":
        pytest.skip("set GENERATIVEQC_TEST_WB97MV_CUDA=1 inside Slurm")
    assert os.environ.get("SLURM_JOB_ID"), "real GPU tests require Slurm"
    from generativeqc import Calculator, GridSpec, KsOptions

    common = dict(
        method=method,
        basis="sto-3g",
        basis_representation="spherical",
        device="cuda",
        ks_options=KsOptions(
            grid=GridSpec(radial_points=12, angular_polar=4, angular_azimuth=8)
        ),
        energy_tolerance=1e-12,
        density_tolerance=1e-10,
        screening_tolerance=1e-14,
        max_iterations=200,
    )
    automatic = Calculator(**common, precision="auto")
    strict = Calculator(**common, precision="fp64")
    assert automatic.capabilities.supported_properties == frozenset(
        {"energy", "forces"}
    )

    with (
        automatic.prepare_batch(
            [atoms], multiplicities=[spin + 1], warm_start=True
        ) as automatic_batch,
        strict.prepare_batch(
            [atoms], multiplicities=[spin + 1], warm_start=True
        ) as strict_batch,
    ):
        auto_cold = automatic_batch.execute(strict=True).items[0]
        fp64_cold = strict_batch.execute(strict=True).items[0]
        auto_warm = automatic_batch.execute(strict=True).items[0]
        fp64_warm = strict_batch.execute(strict=True).items[0]

        moved = np.asarray([xyz for _, xyz in atoms], dtype=float)
        moved[-1, 0] += 0.02
        auto_moved = automatic_batch.execute(coordinates=[moved], strict=True).items[0]
        fp64_moved = strict_batch.execute(coordinates=[moved], strict=True).items[0]

    for automatic_result, fp64_result in (
        (auto_cold, fp64_cold),
        (auto_warm, fp64_warm),
        (auto_moved, fp64_moved),
    ):
        _assert_wb97mv_auto_component_precision(automatic_result)
        assert automatic_result.energy == pytest.approx(fp64_result.energy, abs=2e-8)
        np.testing.assert_allclose(
            automatic_result.forces, fp64_result.forces, atol=2e-7, rtol=0
        )


@pytest.mark.parametrize(
    "method,spin,atoms,basis",
    [
        ("wb97m-v", 0, [("H", (0.0, 0.0, 0.0)), ("H", (0.15, 0.13, 1.5))], "sto-3g"),
        (
            "wb97m-v-uks",
            1,
            [
                ("H", (0.0, 0.0, 0.0)),
                ("H", (0.15, 0.13, 1.5)),
                ("H", (1.8, -0.1, -0.3)),
            ],
            "sto-3g",
        ),
        (
            "wb97m-v",
            0,
            [
                ("O", (0.02, -0.03, 0.04)),
                ("H", (0.1, 1.43, 1.1)),
                ("H", (-0.15, -1.45, 1.12)),
            ],
            "def2-svp",
        ),
        (
            "wb97m-v",
            0,
            [
                ("O", (0.02, -0.03, 0.04)),
                ("H", (0.1, 1.43, 1.1)),
                ("H", (-0.15, -1.45, 1.12)),
            ],
            "def2-tzvp",
        ),
        (
            "wb97m-v",
            0,
            [
                ("O", (0.02, -0.03, 0.04)),
                ("H", (0.1, 1.43, 1.1)),
                ("H", (-0.15, -1.45, 1.12)),
            ],
            "local-def2-tzvpd",
        ),
        pytest.param(
            "wb97m-v-uks",
            1,
            [
                ("N", (0.02, -0.03, 0.04)),
                ("H", (0.1, 1.51, 1.12)),
                ("H", (-0.15, -1.55, 1.13)),
            ],
            "def2-tzvp",
            id="wb97m-v-uks-nh2-def2-tzvp",
        ),
    ],
)
def test_complete_cuda_force_matches_independent_engine(
    method: str, spin: int, atoms: typing.Any, basis: str
) -> None:
    """Reconverge every displaced energy, reusing only native engine-local seeds."""
    if os.environ.get("GENERATIVEQC_TEST_WB97MV_CUDA") != "1":
        pytest.skip("set GENERATIVEQC_TEST_WB97MV_CUDA=1 inside Slurm")
    assert os.environ.get("SLURM_JOB_ID"), "real GPU tests require Slurm"
    import cupy as cp
    from generativeqc import Calculator, GridSpec, KsOptions

    from benchmarks.readme_wb97mv import (
        reference_engine,
        reference_sample,
        reference_vv10_domain,
    )

    native_basis = reference_basis = basis
    if basis == "local-def2-tzvpd":
        from benchmarks.compare_gpu4pyscf_batch import load_comparison_basis

        native_basis, reference_basis = load_comparison_basis(
            Path(__file__).resolve().parents[2]
            / "benchmarks/results/omol25-wb97mv-20261001/def2-tzvpd-ho.json",
            SimpleNamespace(atoms=atoms, basis_representation="spherical"),
            role="orbital",
            compute_forces=True,
        )
    grid = GridSpec(radial_points=12, angular_polar=4, angular_azimuth=8)
    calc = Calculator(
        method=method,
        basis=native_basis,
        basis_representation="spherical",
        device="cuda",
        ks_options=KsOptions(grid=grid),
        energy_tolerance=1e-12,
        density_tolerance=1e-10,
        screening_tolerance=1e-14,
        max_iterations=200,
    )
    assert calc.capabilities.supported_properties == frozenset({"energy", "forces"})
    with calc.prepare_batch(
        [atoms], multiplicities=[spin + 1], warm_start=True
    ) as batch:
        assert batch.capabilities == calc.capabilities
        cold = batch.execute(strict=True).items[0]
        warm = batch.execute(strict=True).items[0]
        if spin == 0:
            assert warm.iterations == 1
        work = batch._stationary_cuda_execution.last_work
        assert work["prepared_execution_reused"]
        assert work["ao_collocation_point_visits"] == work["grid_points"]
        assert work["geometry_point_visits"] == 2 * work["grid_points"]
        assert work["nonlocal_execution"] == "resident-full-grid-device-seeds"
        assert work["nonlocal_feature_source"] == "exact-final-scf-device-binding"
        assert work["nonlocal_feature_d2h_bytes"] == 0
        assert work["nonlocal_seed_h2d_bytes"] == 0
        assert work["nonlocal_dense_pair_capacity"] == work["grid_points"] ** 2
        assert work["nonlocal_seed_generation"] > 0
        assert "nonlocal_pair_evaluations" not in work
        assert len(work["source_names"]) == 12
        assert work["two_electron_quartet_traversals"] is None
        assert work["symmetry_unique_quartets_per_integral_source"] is None
        assert work["maximum_center_dual3_evaluations_total"] is None
        assert work["two_electron_shell_traversals"] is None
        assert work["range_recurrences_per_participating_center"] is None
        assert work["two_electron_radial_operators"] == [
            "full-range",
            "short-range",
            "long-range",
        ]
        assert (
            "native execution route is not exported" in work["two_electron_work_scope"]
        )
        native = work["native_integral_resources"]
        assert native["final_state_export_d2h_bytes"] == 0
        assert native["final_state_export_reads"] == 0
        assert native["final_state_export_synchronizations"] == 0
        assert native["one_electron_h2d_bytes"] == 0
        assert (
            native["one_electron_host_peak_bytes"]
            <= work["additional_host_numeric_bound"]
        )
        energy_only = batch.execute(strict=True, properties=("energy",)).items[0]
        assert cold.executed_backend == warm.executed_backend == "cuda"
        assert energy_only.forces is None
        np.testing.assert_allclose(warm.forces, cold.forces, atol=2e-8, rtol=0)
        assert abs(energy_only.energy - cold.energy) < 1e-9
        with reference_vv10_domain(1e-8):
            oracle = reference_sample(
                reference_engine(atoms, reference_basis, grid, spin=spin), cp
            )
        assert abs(cold.energy - oracle["energies_hartree"][0]) < 1e-8
        np.testing.assert_allclose(
            cold.forces, oracle["forces_hartree_per_bohr"][0], atol=1e-7, rtol=0
        )
        np.testing.assert_allclose(cold.forces.sum(axis=0), 0, atol=2e-8, rtol=0)
        direction = np.random.default_rng(13421389).normal(size=(len(atoms), 3))
        direction -= direction.mean(axis=0)
        direction /= np.linalg.norm(direction)
        analytic = -np.vdot(cold.forces, direction)
        coordinates = np.asarray([xyz for _, xyz in atoms])
        errors = []
        for step in (1e-3, 3e-4, 1e-4):
            energies = []
            for sign in (-1, 1):
                displaced = coordinates + sign * step * direction
                result = batch.execute(
                    [displaced], strict=True, properties=("energy",)
                ).items[0]
                assert result.converged
                energies.append(result.energy)
            errors.append(abs((energies[1] - energies[0]) / (2 * step) - analytic))
        assert errors[0] < 1e-5 and max(errors[1:]) < 2e-6, errors


def test_cuda_force_rebuild_failure_isolation_and_stale_snapshot() -> None:
    """Rebuild real owners, reject old SCF generations, then recover cleanly."""
    if os.environ.get("GENERATIVEQC_TEST_WB97MV_CUDA") != "1":
        pytest.skip("set GENERATIVEQC_TEST_WB97MV_CUDA=1 inside Slurm")
    assert os.environ.get("SLURM_JOB_ID"), "real GPU tests require Slurm"
    from generativeqc import Calculator, GridSpec, KsOptions
    from generativeqc._dft_gradient import StationaryKsState
    from generativeqc_compiler.dft import NativeAO

    atoms = [("H", (0.0, 0.0, 0.0)), ("H", (0.15, 0.13, 1.5))]
    calc = Calculator(
        method="wb97m-v",
        basis="sto-3g",
        device="cuda",
        ks_options=KsOptions(
            grid=GridSpec(radial_points=12, angular_polar=4, angular_azimuth=8)
        ),
        energy_tolerance=1e-12,
        density_tolerance=1e-10,
        max_iterations=200,
    )
    with calc.prepare_batch([atoms, atoms], warm_start=True) as batch:
        cold = batch.execute(strict=True)
        xyz = np.asarray([p for _, p in atoms])
        displaced = xyz.copy()
        displaced[1, 0] += 0.02
        changed = batch.execute(coordinates=[displaced, xyz], strict=True)
        fresh = calc.singlepoint([("H", p) for p in displaced])
        np.testing.assert_allclose(
            changed.items[0].forces, fresh.forces, atol=2e-8, rtol=0
        )
        np.testing.assert_allclose(
            changed.items[1].forces, cold.items[1].forces, atol=2e-8, rtol=0
        )
        failed = batch.execute(coordinates=[[0.0], xyz], strict=False)
        assert not failed.items[0].succeeded and failed.items[1].succeeded
        assert np.isfinite(failed.items[1].forces).all()
        batch.execute(coordinates=[xyz, xyz], strict=True)
        with NativeAO(atoms, basis="sto-3g", representation="cartesian") as basis:
            state = StationaryKsState.from_native(batch, basis, index=0)
            try:
                batch.execute(strict=True, properties=("energy",))
                # An old snapshot must fail before returning any derivative;
                # the next public call must rebuild the discarded scratch.
                with pytest.raises(
                    (RuntimeError, ValueError), match="stale|current|generation"
                ):
                    batch._stationary_cuda_execution.execute(
                        state,
                        basis,
                        compiler=batch._stationary_cuda_compiler(),
                        cache=".cache/stationary-cuda",
                        library=calc._library._name,
                    )
            finally:
                state._source.close()
        recovered = batch.execute(strict=True)
        np.testing.assert_allclose(
            recovered.items[0].forces, cold.items[0].forces, atol=2e-8, rtol=0
        )
