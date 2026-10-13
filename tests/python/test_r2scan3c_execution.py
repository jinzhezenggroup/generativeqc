"""End-to-end canonical r2SCAN-3c composition gates for #172."""

import json
import os
import typing
from pathlib import Path

import numpy as np
import pytest
from generativeqc import (
    Calculator,
    GridSpec,
    KsOptions,
    R2SCAN3CCorrectionBatch,
    evaluate_r2scan3c_correction,
    evaluate_r2scan3c_gcp,
    load_r2scan3c_basis,
)
from generativeqc_compiler.method import UnsupportedMethod, resolve_method

H2 = [("H", (0.0, 0.0, -0.7)), ("H", (0.0, 0.0, 0.7))]
WATER = [
    ("O", (0.0, 0.0, 0.0)),
    ("H", (0.75, 0.58, 0.0)),
    ("H", (-0.75, 0.58, 0.0)),
]
HF = [("H", (0.0, 0.0, 0.0)), ("F", (0.0, 0.0, 0.92))]
OH = [("O", (0.0, 0.0, 0.0)), ("H", (0.0, 0.0, 1.8))]
AUDIT_H2 = [("H", (0.1, -0.1, -0.7)), ("H", (0.0, 0.1, 0.8))]
H3_CATION = [*AUDIT_H2, ("H", (1.6, 0.2, 0.0))]
H2_DIMER = [
    *AUDIT_H2,
    ("H", (4.5, 0.2, -0.8)),
    ("H", (4.4, 0.3, 0.7)),
]
GRID = GridSpec(radial_points=12, angular_polar=4, angular_azimuth=8)


def _numbers_and_coordinates(atoms: typing.Any) -> tuple[list[int], np.ndarray]:
    from generativeqc.calculator import Atom

    normalized = tuple(Atom.from_value(atom) for atom in atoms)
    return (
        [atom.atomic_number for atom in normalized],
        np.asarray([atom.position for atom in normalized], dtype=np.float64),
    )


def _cuda_evidence(name: str, payload: typing.Any) -> None:
    directory = os.environ.get("GENERATIVEQC_R2SCAN3C_CUDA_EVIDENCE")
    if directory:
        path = Path(directory)
        path.mkdir(parents=True, exist_ok=True)
        (path / f"{name}.json").write_text(
            json.dumps(payload, indent=2, default=str) + "\n", encoding="utf-8"
        )


def _real_cuda_allocation() -> str:
    allocation = os.environ.get("SLURM_JOB_ID") or os.environ.get(
        "GENERATIVEQC_QZ_WORKLOAD"
    )
    assert allocation, "real GPU tests require a Slurm allocation or qz workload name"
    return allocation


def test_native_gcp_public_wrapper_matches_independent_reference() -> None:
    from tools.generativeqc_gcp.reference import evaluate_r2scan3c_gcp as reference

    graph = resolve_method("R2SCAN-3c")
    numbers, xyz = _numbers_and_coordinates(H2)
    native = evaluate_r2scan3c_gcp(graph, numbers, xyz)
    oracle = reference(tuple(numbers), xyz)

    assert native.backend == "cpu"
    assert native.provider_identity == "r2scan3c-gcp-cpu-v1"
    assert native.energy == pytest.approx(oracle.energy, abs=2e-14)
    np.testing.assert_allclose(native.gradient, oracle.gradient, atol=2e-13, rtol=0)


def test_r2scan3c_correction_batch_composes_d4_and_gcp_once_on_replay() -> None:
    graph = resolve_method("R2SCAN-3c")
    numbers, xyz = _numbers_and_coordinates(H2)
    moved = xyz.copy()
    moved[1, 2] += 0.05

    with R2SCAN3CCorrectionBatch(graph, [(numbers, xyz, 0.0)], device="cpu") as batch:
        first = batch.execute(gradients=True)[0]
        replay = batch.execute([moved], gradients=True)[0]
        diagnostic = batch.diagnostic()

    direct = evaluate_r2scan3c_correction(
        graph, numbers, moved, total_charge=0.0, device="cpu"
    )
    assert first.ok and replay.ok
    assert replay.energy == pytest.approx(
        replay.d4.energy + replay.gcp.energy, abs=2e-15
    )
    np.testing.assert_allclose(
        replay.gradient, replay.d4.gradient + replay.gcp.gradient, atol=2e-14, rtol=0
    )
    assert replay.energy == pytest.approx(direct.energy, abs=2e-15)
    np.testing.assert_allclose(replay.gradient, direct.gradient, atol=2e-14, rtol=0)
    assert first.energy != replay.energy
    assert diagnostic.method_ir_identity == graph.identity
    assert diagnostic.d4.profile == "r2scan3c"
    assert diagnostic.gcp_backend == "cpu"


def test_calculator_composes_canonical_r2scan3c_energy_and_binds_basis() -> None:
    graph = resolve_method("R2SCAN-3c")
    basis = load_r2scan3c_basis()
    plain = Calculator(
        method="r2scan-rks",
        basis=basis,
        ks_options=KsOptions(grid=GRID),
        max_iterations=200,
    ).singlepoint(H2, properties=("energy",))
    calculator = Calculator(
        method=graph,
        ks_options=KsOptions(grid=GRID),
        max_iterations=200,
    )
    corrected = calculator.singlepoint(H2, properties=("energy",))
    numbers, xyz = _numbers_and_coordinates(H2)
    correction = evaluate_r2scan3c_correction(
        graph, numbers, xyz, total_charge=0.0, device="cpu", gradients=False
    )

    assert calculator.method_ir == graph
    assert calculator._basis.identity == basis.identity
    assert calculator._basis.representation == "spherical"
    assert corrected.dispersion is not None
    assert corrected.dispersion.energy == pytest.approx(correction.energy, abs=2e-15)
    assert corrected.dispersion.d4.energy == pytest.approx(
        correction.d4.energy, abs=2e-15
    )
    assert corrected.dispersion.gcp.energy == pytest.approx(
        correction.gcp.energy, abs=2e-15
    )
    assert corrected.energy == pytest.approx(
        plain.energy + correction.energy, abs=2e-11
    )


def test_r2scan3c_rejects_wrong_basis_and_out_of_domain_element() -> None:
    graph = resolve_method("R2SCAN-3c")
    with pytest.raises(ValueError, match="expected basis"):
        Calculator(method=graph, basis="sto-3g", ks_options=KsOptions(grid=GRID))

    calculator = Calculator(method=graph, ks_options=KsOptions(grid=GRID))
    with pytest.raises(UnsupportedMethod, match="does not support atomic numbers"):
        calculator.singlepoint([("K", (0.0, 0.0, 0.0))], properties=("energy",))


@pytest.mark.skipif(
    os.environ.get("GENERATIVEQC_DFT_CUDA_TEST") != "1",
    reason="explicit real-device CUDA gate",
)
@pytest.mark.parametrize("atoms", (H2, WATER), ids=("h2-sp", "water-spd"))
def test_cuda_r2scan3c_public_force_is_electronic_plus_d4_plus_gcp(
    atoms: list[tuple[str, tuple[float, float, float]]],
) -> None:
    allocation = _real_cuda_allocation()
    graph = resolve_method("R2SCAN-3c")
    basis = load_r2scan3c_basis()
    controls = {
        "device": "cuda",
        "ks_options": KsOptions(grid=GRID),
        "max_iterations": 200,
        "energy_tolerance": 1e-12,
        "density_tolerance": 1e-10,
    }
    plain = Calculator(method="r2scan-rks", basis=basis, **controls).singlepoint(
        atoms, properties=("energy", "forces")
    )
    corrected = Calculator(method=graph, **controls).singlepoint(
        atoms, properties=("energy", "forces")
    )
    numbers, xyz = _numbers_and_coordinates(atoms)
    correction = evaluate_r2scan3c_correction(
        graph, numbers, xyz, total_charge=0.0, device="cuda"
    )

    assert corrected.dispersion is not None
    assert corrected.dispersion.backend == "cuda+cpu"
    assert corrected.energy == pytest.approx(plain.energy + correction.energy, abs=2e-9)
    np.testing.assert_allclose(
        corrected.forces,
        plain.forces - correction.gradient,
        atol=2e-7,
        rtol=0,
    )
    _cuda_evidence(
        f"composition-{len(atoms)}-atoms",
        {
            "atomic_numbers": numbers,
            "allocation": allocation,
            "basis_identity": basis.identity,
            "dispersion_backend": corrected.dispersion.backend,
            "energy_composition_error": abs(
                corrected.energy - plain.energy - correction.energy
            ),
            "force_composition_max_error": float(
                np.max(np.abs(corrected.forces - plain.forces + correction.gradient))
            ),
        },
    )


@pytest.mark.skipif(
    os.environ.get("GENERATIVEQC_DFT_CUDA_TEST") != "1",
    reason="explicit real-device CUDA gate",
)
def test_cuda_r2scan3c_water_total_force_matches_reconverged_directional_fd() -> None:
    allocation = _real_cuda_allocation()
    calculator = Calculator(
        method="r2scan-3c-rks",
        device="cuda",
        ks_options=KsOptions(grid=GRID),
        max_iterations=200,
        energy_tolerance=1e-12,
        density_tolerance=1e-10,
    )
    result = calculator.singlepoint(WATER, properties=("energy", "forces"))
    xyz = np.asarray([position for _, position in WATER], dtype=np.float64)
    direction = np.asarray(
        [[0.13, -0.07, 0.11], [-0.05, 0.17, 0.03], [0.09, 0.02, -0.14]]
    )
    estimates = []
    for step in (3e-4, 1e-4):
        energies = []
        for sign in (1, -1):
            moved = [
                (atom[0], position)
                for atom, position in zip(
                    WATER, xyz + sign * step * direction, strict=True
                )
            ]
            energies.append(
                calculator.singlepoint(moved, properties=("energy",)).energy
            )
        estimates.append((energies[0] - energies[1]) / (2 * step))
    analytic = float(np.sum(-result.forces * direction))

    _cuda_evidence(
        "water-total-force-directional-fd",
        {
            "steps": [3e-4, 1e-4],
            "allocation": allocation,
            "estimates": estimates,
            "analytic": analytic,
            "step_convergence": abs(estimates[-1] - estimates[-2]),
            "analytic_error": abs(estimates[-1] - analytic),
        },
    )
    assert abs(estimates[-1] - estimates[-2]) < 5e-6
    assert abs(estimates[-1] - analytic) < 5e-6


@pytest.mark.skipif(
    os.environ.get("GENERATIVEQC_DFT_CUDA_TEST") != "1",
    reason="explicit real-device CUDA gate",
)
def test_cuda_r2scan3c_spd_force_reports_bounded_resource_work() -> None:
    allocation = _real_cuda_allocation()
    from generativeqc._dft_gradient import StationaryKsState
    from generativeqc._stationary_cuda import complete_rks_cuda_gradient_diagnostic
    from generativeqc_compiler.dft import NativeAO

    calculator = Calculator(
        method="r2scan-3c-rks",
        device="cuda",
        ks_options=KsOptions(grid=GRID),
        max_iterations=200,
        energy_tolerance=1e-12,
        density_tolerance=1e-10,
    )
    with calculator.prepare_batch([WATER], warm_start=False) as batch:
        batch.execute(strict=True, properties=("energy",))
        with NativeAO(
            WATER, basis=calculator._basis, representation="spherical"
        ) as basis:
            state = StationaryKsState.from_native(batch, basis)
            try:
                with pytest.raises(ValueError, match="primitive page work budget"):
                    complete_rks_cuda_gradient_diagnostic(
                        state,
                        basis,
                        compiler=batch._stationary_cuda_compiler(),
                        target=batch._stationary_cuda_target(),
                        cache=Path(
                            os.environ.get(
                                "GENERATIVEQC_STATIONARY_CACHE",
                                ".cache/stationary-cuda",
                            )
                        ),
                        aot_directory=None,
                        native_grid_library=Path(str(batch._library._name)).resolve(),
                        max_primitive_records=1,
                    )
            finally:
                state._source.close()
        force, work = batch._public_dft_cuda_force(0, WATER)

    assert force.shape == (3, 3)
    assert work["primitive_records"] == 17_408
    assert work["full_range_derivative_route"] == "prepared-direct-shell"
    assert work["full_range_ao_task_domain_elided"] is True
    assert work["primitive_record_page_budget"] == 16_000_000
    assert (
        0 < work["primitive_page_peak_records"] <= work["primitive_record_page_budget"]
    )
    assert work["primitive_pages"] > 0
    assert work["task_descriptors"] > 0
    assert work["task_batches"] > 0
    assert work["additional_device_peak_bound"] <= work["additional_device_budget"]
    _cuda_evidence(
        "water-spd-resource-work",
        {**work, "allocation": allocation, "rejected_page_budget": 1},
    )


@pytest.mark.skipif(
    os.environ.get("GENERATIVEQC_DFT_CUDA_TEST") != "1",
    reason="explicit real-device CUDA gate",
)
def test_cuda_r2scan3c_ragged_force_replay_matches_fresh_changed_geometry() -> None:
    allocation = _real_cuda_allocation()
    calculator = Calculator(
        method="r2scan-3c-rks",
        device="cuda",
        ks_options=KsOptions(grid=GRID),
        max_iterations=200,
        energy_tolerance=1e-12,
        density_tolerance=1e-10,
    )
    systems = (HF, WATER)
    moved = [
        np.asarray([position for _, position in HF], dtype=np.float64),
        np.asarray([position for _, position in WATER], dtype=np.float64),
    ]
    moved[0][1] += [0.01, -0.02, 0.03]
    moved[1][1] += [-0.02, 0.01, 0.04]
    moved_atoms = tuple(
        [
            (atom[0], tuple(position))
            for atom, position in zip(system, coordinates, strict=True)
        ]
        for system, coordinates in zip(systems, moved, strict=True)
    )

    with calculator.prepare_batch(systems, warm_start=False) as batch:
        first = batch.execute(strict=True, properties=("energy", "forces"))
        replay = batch.execute(
            coordinates=moved,
            strict=True,
            properties=("energy", "forces"),
        )

    fresh = tuple(
        calculator.singlepoint(atoms, properties=("energy", "forces"))
        for atoms in moved_atoms
    )
    assert all(item.succeeded for item in first.items)
    energy_errors = []
    force_errors = []
    for item, direct in zip(replay.items, fresh, strict=True):
        energy_errors.append(abs(item.energy - direct.energy))
        force_errors.append(float(np.max(np.abs(item.forces - direct.forces))))
        assert item.energy == pytest.approx(direct.energy, abs=2e-9)
        np.testing.assert_allclose(item.forces, direct.forces, atol=2e-7, rtol=0)
    _cuda_evidence(
        "ragged-force-changed-geometry",
        {
            "atom_counts": [len(system) for system in systems],
            "allocation": allocation,
            "energy_errors": energy_errors,
            "force_max_errors": force_errors,
        },
    )


@pytest.mark.skipif(
    os.environ.get("GENERATIVEQC_DFT_CUDA_TEST") != "1",
    reason="explicit real-device CUDA gate",
)
def test_cuda_r2scan3c_charged_ragged_failure_isolation() -> None:
    """A malformed item must not poison a charged neighbor's retained force."""

    allocation = _real_cuda_allocation()
    calculator = Calculator(
        method="r2scan-3c-rks",
        device="cuda",
        ks_options=KsOptions(),
        max_iterations=200,
        energy_tolerance=1e-12,
        density_tolerance=1e-10,
    )
    with calculator.prepare_batch(
        [AUDIT_H2, H3_CATION],
        charges=[0, 1],
        multiplicities=[1, 1],
        warm_start=False,
    ) as batch:
        first = batch.execute(strict=True, properties=("energy", "forces"))
        isolated = batch.execute(
            coordinates=[[0.0], None], properties=("energy", "forces")
        )

    assert all(item.succeeded for item in first.items)
    assert not isolated.items[0].succeeded
    assert isolated.items[0].forces is None
    assert isolated.items[1].succeeded
    assert isolated.items[1].energy == pytest.approx(first.items[1].energy, abs=2e-9)
    np.testing.assert_allclose(
        isolated.items[1].forces, first.items[1].forces, atol=2e-7, rtol=0
    )
    _cuda_evidence(
        "charged-ragged-failure-isolation",
        {
            "allocation": allocation,
            "atom_counts": [len(AUDIT_H2), len(H3_CATION)],
            "charges": [0, 1],
            "first_statuses": [item.status for item in first.items],
            "isolated_statuses": [item.status for item in isolated.items],
            "surviving_energy_error": abs(
                isolated.items[1].energy - first.items[1].energy
            ),
            "surviving_force_max_error": float(
                np.max(np.abs(isolated.items[1].forces - first.items[1].forces))
            ),
        },
    )


@pytest.mark.skipif(
    os.environ.get("GENERATIVEQC_DFT_CUDA_TEST") != "1",
    reason="explicit real-device CUDA gate",
)
@pytest.mark.parametrize(
    ("atoms", "charge", "multiplicity", "spin", "initial_guess"),
    [
        (WATER, 0, 1, "rks", "default"),
        (H2_DIMER, 0, 1, "rks", "default"),
        (OH, 0, 2, "uks", "native"),
        (WATER, 1, 2, "uks", "default"),
    ],
    ids=(
        "water-rks-spd",
        "h2-dimer-rks-sp",
        "oh-uks-spd-native-seed",
        "water-cation-uks-spd",
    ),
)
def test_cuda_r2scan3c_spd_total_matches_independent_analytic_oracle(
    atoms: list[tuple[str, tuple[float, float, float]]],
    charge: int,
    multiplicity: int,
    spin: str,
    initial_guess: str,
) -> None:
    """Compare every canonical component on the production grid, including d AOs."""

    allocation = _real_cuda_allocation()
    dftd4 = pytest.importorskip("dftd4")
    pyscf = pytest.importorskip("pyscf")
    from dftd4.interface import DampingParam, DispersionModel
    from generativeqc._dft_gradient import StationaryKsState
    from generativeqc_compiler.dft import NativeAO
    from test_dft_complete_cpu import independent_semilocal_total_gradient

    from tools.generativeqc_gcp.reference import evaluate_r2scan3c_gcp as reference_gcp

    basis_spec = load_r2scan3c_basis()
    calculator = Calculator(
        method=f"r2scan-3c-{spin}",
        device="cuda",
        ks_options=KsOptions(),
        max_iterations=200,
        energy_tolerance=1e-12,
        density_tolerance=1e-10,
    )
    with (
        calculator.prepare_batch(
            [atoms], charges=[charge], multiplicities=[multiplicity], warm_start=False
        ) as batch,
        NativeAO(
            atoms,
            basis=basis_spec,
            representation="spherical",
            charge=charge,
            multiplicity=multiplicity,
        ) as basis,
    ):
        item = batch.execute(strict=True, properties=("energy", "forces")).items[0]
        state = StationaryKsState.from_native(batch, basis)
        try:
            electronic_energy, electronic_gradient = (
                independent_semilocal_total_gradient(
                    basis,
                    state,
                    f"r2scan-{spin}",
                    cart=False,
                    initial_guess=initial_guess,
                )
            )
            grid_points = len(state.grid.points)
        finally:
            state._source.close()

    numbers, coordinates = _numbers_and_coordinates(atoms)
    damping = DampingParam(s6=1.0, s8=0.0, s9=2.0, a1=0.42, a2=5.65)
    d4 = DispersionModel(
        np.asarray(numbers, dtype=np.int32),
        coordinates,
        charge=float(charge),
        ga=2.0,
        gc=1.0,
    ).get_dispersion(damping, grad=True)
    gcp = reference_gcp(tuple(numbers), coordinates)
    reference_energy = electronic_energy + float(d4["energy"]) + gcp.energy
    reference_force = -electronic_gradient - d4["gradient"] - gcp.gradient

    assert item.dispersion is not None
    energy_error = abs(item.energy - reference_energy)
    force_error = float(np.max(np.abs(item.forces - reference_force)))
    d4_error = abs(item.dispersion.d4.energy - float(d4["energy"]))
    gcp_error = abs(item.dispersion.gcp.energy - gcp.energy)
    _cuda_evidence(
        f"{spin}-{len(atoms)}-charge-{charge}-production-grid-independent-oracle",
        {
            "allocation": allocation,
            "source_tree": os.environ.get("GENERATIVEQC_SOURCE_TREE"),
            "basis_identity": basis_spec.identity,
            "atomic_numbers": numbers,
            "charge": charge,
            "multiplicity": multiplicity,
            "grid_points": grid_points,
            "pyscf_version": pyscf.__version__,
            "pyscf_scf_solver": "diis",
            "pyscf_initial_guess": initial_guess,
            "dftd4_version": dftd4.__version__,
            "energy_error": energy_error,
            "force_max_error": force_error,
            "d4_energy_error": d4_error,
            "gcp_energy_error": gcp_error,
            "physical_residual_rms": item.physical_residual_rms,
        },
    )
    assert energy_error < 2e-9
    assert force_error < 1e-7
    assert d4_error < 2e-9
    assert gcp_error < 2e-9


def test_df_canonical_r2scan3c_cpu_force_composes_and_replays() -> None:
    """DF electronic force and external D4/gCP gradients each appear once."""
    controls = {
        "device": "cpu",
        "density_fitting": "cpu",
        "auxiliary_basis": "def2-svp",
        "ks_options": KsOptions(grid=GRID),
        "max_iterations": 200,
        "energy_tolerance": 1e-12,
        "density_tolerance": 1e-10,
    }
    graph = resolve_method("R2SCAN-3c")
    electronic = Calculator(
        method="r2scan-rks", basis=load_r2scan3c_basis(), **controls
    )
    composite = Calculator(method="r2scan-3c-rks", **controls)
    assert "forces" in electronic.capabilities.supported_properties
    assert "forces" in composite.capabilities.supported_properties
    assert composite._basis.identity == load_r2scan3c_basis().identity

    xyz = np.asarray([position for _, position in H2], dtype=np.float64)
    moved = xyz.copy()
    moved[1] += [0.02, -0.01, 0.03]
    moved_atoms = [
        (symbol, tuple(row))
        for (symbol, _), row in zip(H2, moved, strict=True)
    ]
    with composite.prepare_batch([H2], warm_start=True) as batch:
        cold = batch.execute(strict=True, properties=("energy", "forces")).items[0]
        warm = batch.execute(strict=True, properties=("energy", "forces")).items[0]
        changed = batch.execute(
            coordinates=[moved], strict=True, properties=("energy", "forces")
        ).items[0]
        moved_warm = batch.execute(
            strict=True, properties=("energy", "forces")
        ).items[0]
    fresh = composite.singlepoint(moved_atoms, properties=("energy", "forces"))

    for row, atoms in ((cold, H2), (changed, moved_atoms)):
        electronic_result = electronic.singlepoint(
            atoms, properties=("energy", "forces")
        )
        numbers, coordinates = _numbers_and_coordinates(atoms)
        correction = evaluate_r2scan3c_correction(
            graph, numbers, coordinates, device="cpu"
        )
        assert row.converged and row.dispersion is not None
        assert row.energy == pytest.approx(
            electronic_result.energy + correction.energy, abs=1e-8
        )
        np.testing.assert_allclose(
            row.forces,
            electronic_result.forces - correction.gradient,
            atol=3e-7,
            rtol=0,
        )
        assert row.dispersion.d4.energy == pytest.approx(
            correction.d4.energy, abs=2e-13
        )
        assert row.dispersion.gcp.energy == pytest.approx(
            correction.gcp.energy, abs=2e-13
        )
        np.testing.assert_allclose(
            row.dispersion.gradient, correction.gradient, atol=2e-11, rtol=0
        )
        np.testing.assert_allclose(row.forces.sum(axis=0), 0, atol=2e-8, rtol=0)

    assert warm.energy == pytest.approx(cold.energy, abs=1e-9)
    np.testing.assert_allclose(warm.forces, cold.forces, atol=3e-7, rtol=0)
    assert moved_warm.energy == pytest.approx(changed.energy, abs=1e-9)
    np.testing.assert_allclose(moved_warm.forces, changed.forces, atol=3e-7, rtol=0)
    assert changed.energy == pytest.approx(fresh.energy, abs=1e-8)
    np.testing.assert_allclose(changed.forces, fresh.forces, atol=3e-7, rtol=0)


def test_df_noncanonical_d4_composite_remains_energy_only() -> None:
    """Adding a D4 force provider must not widen unrelated DF compositions."""
    calc = Calculator(
        method="r2scan-d4-rks",
        basis="sto-3g",
        device="cpu",
        density_fitting="cpu",
        auxiliary_basis="def2-svp",
        ks_options=KsOptions(grid=GRID),
    )
    assert not calc._density_fitted_composite_force_eligible()
    assert "forces" not in calc.capabilities.supported_properties
    with pytest.raises(ValueError, match="does not support properties: forces"):
        calc.singlepoint(H2, properties=("energy", "forces"))


@pytest.mark.skipif(
    os.environ.get("GENERATIVEQC_DFT_CUDA_TEST") != "1",
    reason="independent CUDA DF qualification requires a Slurm allocation",
)
def test_cuda_r2scan3c_df_spherical_total_force_matches_independent_reference() -> None:
    """3-atom canonical DF E/F: cold, five warm, moved and five moved-warm.

    The PySCF force differentiates both auxiliary integrals and the metric
    under the exact unpruned moving grid. D4 and gCP have independent sources.
    """
    allocation = _real_cuda_allocation()
    pyscf = pytest.importorskip("pyscf")
    dftd4 = pytest.importorskip("dftd4")
    from dftd4.interface import DampingParam, DispersionModel
    from generativeqc._dft_gradient import StationaryKsState
    from generativeqc.basis import load_basis
    from generativeqc_compiler.dft import NativeAO
    from test_dft_df_public import pyscf_energy_gradient
    from tools.generativeqc_gcp.reference import evaluate_r2scan3c_gcp as reference_gcp

    auxiliary = load_basis(
        Path(__file__).resolve().parents[2]
        / "benchmarks/results/issue206-practical-auxiliary/identity/cc-pvdz-jkfit.json"
    )
    basis_spec = load_r2scan3c_basis()
    calc = Calculator(
        method="r2scan-3c-rks",
        device="cuda",
        density_fitting="cuda",
        auxiliary_basis=auxiliary,
        ks_options=KsOptions(grid=GridSpec(12, 8, 16)),
        max_iterations=200,
        energy_tolerance=1e-12,
        density_tolerance=1e-10,
    )
    assert calc._basis.identity == basis_spec.identity
    assert calc._representation_name == "spherical"
    assert "forces" in calc.capabilities.supported_properties

    def reference(batch: typing.Any, atoms: typing.Any) -> tuple[float, np.ndarray]:
        with NativeAO(
            atoms, basis=basis_spec, representation="spherical"
        ) as basis:
            state = StationaryKsState.from_native(batch, basis)
            try:
                coulomb, exchange, threshold = state._source.fock_provider_proof()
                assert coulomb == "density-fitted" and exchange is None
                assert threshold == calc._density_fitting_relative_threshold
                electronic_energy, electronic_gradient = pyscf_energy_gradient(
                    calc, state, atoms, 1, "R2SCAN"
                )
            finally:
                state._source.close()
        numbers, xyz = _numbers_and_coordinates(atoms)
        damping = DampingParam(s6=1.0, s8=0.0, s9=2.0, a1=0.42, a2=5.65)
        d4 = DispersionModel(
            np.asarray(numbers, dtype=np.int32),
            xyz,
            charge=0.0,
            ga=2.0,
            gc=1.0,
        ).get_dispersion(damping, grad=True)
        gcp = reference_gcp(tuple(numbers), xyz)
        return (
            electronic_energy + float(d4["energy"]) + gcp.energy,
            -electronic_gradient - d4["gradient"] - gcp.gradient,
        )

    moved = np.asarray([xyz for _, xyz in WATER], dtype=np.float64)
    moved[1] += [0.025, -0.01, 0.035]
    moved_atoms = [
        (symbol, tuple(row))
        for (symbol, _), row in zip(WATER, moved, strict=True)
    ]
    errors: dict[str, list[dict[str, float]]] = {"fixed": [], "moved": []}

    with calc.prepare_batch([WATER], warm_start=True) as batch:
        first = batch.execute(strict=True, properties=("energy", "forces")).items[0]
        first_reference = reference(batch, WATER)
        for repeat in range(6):
            row = (
                first
                if repeat == 0
                else batch.execute(
                    strict=True, properties=("energy", "forces")
                ).items[0]
            )
            assert row.converged and row.dispersion is not None
            errors["fixed"].append(
                {
                    "energy": abs(row.energy - first_reference[0]),
                    "force": float(
                        np.max(np.abs(row.forces - first_reference[1]))
                    ),
                }
            )
        first_moved = batch.execute(
            coordinates=[moved], strict=True, properties=("energy", "forces")
        ).items[0]
        moved_reference = reference(batch, moved_atoms)
        for repeat in range(6):
            row = (
                first_moved
                if repeat == 0
                else batch.execute(
                    strict=True, properties=("energy", "forces")
                ).items[0]
            )
            assert row.converged and row.dispersion is not None
            errors["moved"].append(
                {
                    "energy": abs(row.energy - moved_reference[0]),
                    "force": float(
                        np.max(np.abs(row.forces - moved_reference[1]))
                    ),
                }
            )
        correction_diagnostic = batch.dispersion_diagnostic
        metrics = batch.last_density_fitting_metric_diagnostics()

    direction = np.asarray(
        [[0.13, -0.07, 0.11], [-0.05, 0.17, 0.03], [0.09, 0.02, -0.14]]
    )
    step = 1e-4
    displaced_energies = []
    for sign in (1, -1):
        displaced = [
            (symbol, tuple(row))
            for (symbol, _), row in zip(
                WATER,
                np.asarray([xyz for _, xyz in WATER]) + sign * step * direction,
                strict=True,
            )
        ]
        displaced_energies.append(
            calc.singlepoint(displaced, properties=("energy",)).energy
        )
    finite_difference = (displaced_energies[0] - displaced_energies[1]) / (2 * step)
    projected = float(-np.sum(first.forces * direction))
    max_energy_error = max(entry["energy"] for rows in errors.values() for entry in rows)
    max_force_error = max(entry["force"] for rows in errors.values() for entry in rows)
    translation_error = max(
        float(np.max(np.abs(row.sum(axis=0))))
        for row in (first.forces, first_moved.forces)
    )
    _cuda_evidence(
        "canonical-3atom-r2scan3c-df-independent-force",
        {
            "allocation": allocation,
            "method": "r2scan-3c-rks",
            "basis_identity": basis_spec.identity,
            "auxiliary_basis_identity": auxiliary.identity,
            "pyscf_version": pyscf.__version__,
            "dftd4_version": dftd4.__version__,
            "errors": errors,
            "metric_diagnostics": [str(value) for value in metrics],
            "correction_identity": correction_diagnostic.method_ir_identity,
            "max_energy_error": max_energy_error,
            "max_force_error": max_force_error,
            "translation_error": translation_error,
            "directional_fd_error": abs(finite_difference - projected),
        },
    )
    assert max_energy_error < 1e-8
    assert max_force_error < 1e-7
    assert translation_error < 2e-8
    assert abs(finite_difference - projected) < 5e-6
