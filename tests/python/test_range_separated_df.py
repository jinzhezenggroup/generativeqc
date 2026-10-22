"""Full RI-J/K endpoints: independent LR fitting, moving-grid forces and replay."""

import os
from contextlib import closing
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from generativeqc import Calculator, GridSpec, KsOptions
from generativeqc._ks_snapshot import NativeKsSnapshot
from generativeqc.fock import FockBuildSpec, FockPlan, FockTerm
from generativeqc_compiler.dft import NativeAO
from test_mixed_rsh_df_forces import _mixed_pyscf_energy

from benchmarks._retained_basis import load_retained_basis


def _practical_auxiliary() -> Any:
    """Reuse checksum-verified primitive tables without a network basis lookup."""
    return load_retained_basis(
        Path(__file__).resolve().parents[2]
        / "benchmarks/results/issue206-practical-auxiliary/identity/cc-pvdz-jkfit.json"
    )


@pytest.mark.parametrize("mapping", ("auto", "auxiliary"))
def test_cuda_packed_lr_values_preserve_the_radial_operator(
    mapping: str, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Prepared full-range raw values must not replace the separate LR source.

    Packed-single storage with automatic mapping admits prepared Hermite values
    for the primary J/K owner. The LR owner must retain its omega-aware fallback.
    Compare the complete energy and a nuclear derivative with an independently
    fitted operator; the auxiliary mapping also exercises the generic fallback.
    """
    if os.environ.get("GENERATIVEQC_DFT_CUDA_TEST") != "1":
        pytest.skip("requires an explicitly Slurm-allocated CUDA device")
    assert os.environ.get("SLURM_JOB_ID")
    from benchmarks.df_component_ledger import read_trace

    monkeypatch.setenv("GENERATIVEQC_DF_RANGE_EXCHANGE", "df")
    monkeypatch.setenv("GENERATIVEQC_DF_VALUE_STORAGE", "packed-single")
    monkeypatch.setenv("GENERATIVEQC_DF_VALUE_MAPPING", mapping)
    trace = tmp_path / "packed-lr.jsonl"
    monkeypatch.setenv("GENERATIVEQC_DF_TRACE", str(trace))
    atoms = [
        ("O", (0.0, 0.0, 0.0)),
        ("H", (0.1, 1.43, 1.1)),
        ("H", (-0.15, -1.45, 1.12)),
    ]
    calculator = Calculator(
        method="wb97m-v-rks",
        basis="def2-svp",
        basis_representation="spherical",
        auxiliary_basis="def2-svp",
        device="cuda",
        density_fitting="auto",
        initial_guess=None,
        ks_options=KsOptions(
            grid=GridSpec(radial_points=12, angular_polar=4, angular_azimuth=8)
        ),
        energy_tolerance=1e-12,
        density_tolerance=1e-10,
        screening_tolerance=1e-14,
        max_iterations=200,
    )
    with calculator.prepare_batch([atoms], warm_start=True) as batch:
        cold = batch.execute(strict=True, properties=("energy", "forces")).items[0]
        warm = batch.execute(strict=True, properties=("energy", "forces")).items[0]
    assert cold.executed_backend == warm.executed_backend == "cuda"
    assert cold.energy == pytest.approx(
        _mixed_pyscf_energy(calculator, atoms, 0, fitted_range=True), abs=2e-8
    )
    assert warm.energy == pytest.approx(cold.energy, abs=2e-10)
    np.testing.assert_allclose(warm.forces, cold.forces, atol=2e-8, rtol=0)
    energies = []
    step = 2e-4
    for sign in (1, -1):
        moved = list(atoms)
        moved[1] = (
            atoms[1][0],
            tuple(np.asarray(atoms[1][1]) + (0, 0, sign * step)),
        )
        energies.append(_mixed_pyscf_energy(calculator, moved, 0, fitted_range=True))
    assert -cold.forces[1][2] == pytest.approx(
        (energies[0] - energies[1]) / (2 * step), abs=3e-7
    )
    prepared = sum(
        record["counters"].get("raw_hermite_values", 0) for record in read_trace(trace)
    )
    assert (prepared > 0) == (mapping == "auto")


@pytest.mark.parametrize("device", ("cpu", "cuda"))
@pytest.mark.parametrize("spin", (0, 1))
@pytest.mark.parametrize(
    "selector", (None, "direct", "df"), ids=("default", "direct", "df")
)
def test_range_exchange_selection_is_frozen_for_all_endpoints(
    device: str,
    spin: int,
    selector: str | None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Opt-in LR fitting cannot change with properties, replay or geometry."""
    if device == "cuda" and os.environ.get("GENERATIVEQC_DFT_CUDA_TEST") != "1":
        pytest.skip("requires an explicitly Slurm-allocated CUDA device")
    if device == "cuda":
        assert os.environ.get("SLURM_JOB_ID")
    if selector is None:
        monkeypatch.delenv("GENERATIVEQC_DF_RANGE_EXCHANGE", raising=False)
    else:
        monkeypatch.setenv("GENERATIVEQC_DF_RANGE_EXCHANGE", selector)
    fitted_range = selector == "df"
    atoms = [("H", (0.0, 0.0, 0.0)), ("H", (0.15, 0.13, 1.5))]
    if spin:
        atoms.append(("H", (1.8, -0.1, -0.3)))
    calculator = Calculator(
        method="wb97m-v-uks" if spin else "wb97m-v-rks",
        basis="sto-3g",
        auxiliary_basis="sto-3g",
        device=device,
        density_fitting="auto",
        initial_guess=None,
        ks_options=KsOptions(
            grid=GridSpec(radial_points=12, angular_polar=4, angular_azimuth=8)
        ),
        energy_tolerance=1e-12,
        density_tolerance=1e-10,
        screening_tolerance=1e-14,
        max_iterations=200,
    )
    with calculator.prepare_batch(
        [atoms], multiplicities=[spin + 1], warm_start=True
    ) as batch:
        monkeypatch.setenv(
            "GENERATIVEQC_DF_RANGE_EXCHANGE", "direct" if fitted_range else "df"
        )
        cold = batch.execute(strict=True, properties=("energy",)).items[0]
        with closing(NativeKsSnapshot(batch, 0)) as snapshot:
            assert snapshot.range_density_fitted == fitted_range
            warm = batch.execute(strict=True, properties=("energy", "forces")).items[0]
            with pytest.raises(ValueError, match="stale"):
                _ = snapshot.range_density_fitted
        with closing(NativeKsSnapshot(batch, 0)) as snapshot:
            assert snapshot.range_density_fitted == fitted_range
        moved = list(atoms)
        moved[1] = (
            atoms[1][0],
            tuple(np.asarray(atoms[1][1]) + np.array((0.03, -0.02, 0.04))),
        )
        rebound = batch.execute(
            coordinates=([position for _, position in moved],),
            strict=True,
            properties=("energy", "forces"),
        ).items[0]
        with closing(NativeKsSnapshot(batch, 0)) as snapshot:
            assert snapshot.range_density_fitted == fitted_range
    assert cold.converged and warm.converged and rebound.converged
    assert warm.energy == pytest.approx(cold.energy, abs=2e-10)
    assert cold.energy == pytest.approx(
        _mixed_pyscf_energy(calculator, atoms, spin, fitted_range=fitted_range),
        abs=2e-8,
    )
    assert rebound.energy == pytest.approx(
        _mixed_pyscf_energy(calculator, moved, spin, fitted_range=fitted_range),
        abs=2e-8,
    )
    step = 2e-4
    energies = []
    for sign in (1, -1):
        displaced = list(moved)
        displaced[1] = (
            moved[1][0],
            tuple(np.asarray(moved[1][1]) + np.array((0.0, 0.0, sign * step))),
        )
        energies.append(
            _mixed_pyscf_energy(calculator, displaced, spin, fitted_range=fitted_range)
        )
    assert -rebound.forces[1][2] == pytest.approx(
        (energies[0] - energies[1]) / (2 * step), abs=3e-7
    )


@pytest.mark.parametrize("device", ("cpu", "cuda"))
@pytest.mark.parametrize(
    "basis,spin,materialized",
    (
        ("def2-svp", 0, False),
        ("def2-tzvp", 0, False),
        ("def2-svp", 1, False),
        ("def2-svp", 0, True),
    ),
)
def test_practical_lr_auxiliary_stationary_force_and_replay(
    device: str,
    basis: str,
    spin: int,
    materialized: bool,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Ill-conditioned retained LR modes must not amplify warm-density roundoff."""
    if device == "cuda" and os.environ.get("GENERATIVEQC_DFT_CUDA_TEST") != "1":
        pytest.skip("requires an explicitly Slurm-allocated CUDA device")
    if device == "cuda":
        assert os.environ.get("SLURM_JOB_ID")
        if materialized:
            pytest.skip("materialized derivative storage is a CPU diagnostic")
    monkeypatch.setenv("GENERATIVEQC_DF_RANGE_EXCHANGE", "df")
    monkeypatch.setenv(
        "GENERATIVEQC_CPU_DF_MATERIALIZED_DERIVATIVES", str(int(materialized))
    )
    atoms = [
        ("N" if spin else "O", (0.0, 0.0, 0.0)),
        ("H", (0.1, 1.43, 1.1)),
        ("H", (-0.15, -1.45, 1.12)),
    ]
    calculator = Calculator(
        method="wb97m-v-uks" if spin else "wb97m-v-rks",
        basis=basis,
        basis_representation="spherical",
        auxiliary_basis=_practical_auxiliary(),
        device=device,
        density_fitting="auto",
        initial_guess=None,
        ks_options=KsOptions(
            grid=GridSpec(radial_points=24, angular_polar=6, angular_azimuth=12)
        ),
        energy_tolerance=1e-10,
        density_tolerance=1e-8,
        screening_tolerance=1e-14,
        max_iterations=200,
    )
    with calculator.prepare_batch(
        [atoms], multiplicities=[spin + 1], warm_start=True
    ) as batch:
        cold = batch.execute(strict=True, properties=("energy", "forces")).items[0]
        for _ in range(3):
            warm = batch.execute(strict=True, properties=("energy", "forces")).items[0]
            assert warm.converged and warm.executed_backend == cold.executed_backend
            assert warm.energy == pytest.approx(cold.energy, abs=2e-10)
            np.testing.assert_allclose(warm.forces, cold.forces, atol=2e-8, rtol=0)
    assert cold.energy == pytest.approx(
        _mixed_pyscf_energy(calculator, atoms, spin, fitted_range=True), abs=2e-8
    )
    direction = np.array(
        ((0.17, -0.11, 0.07), (-0.09, 0.06, 0.13), (0.04, -0.03, 0.08))
    )[: len(atoms)]
    for displacement in (np.eye(len(atoms) * 3)[5].reshape(-1, 3), direction):
        step = 2e-4
        energies = []
        for sign in (1, -1):
            moved = [
                (symbol, tuple(np.asarray(position) + sign * step * shift))
                for (symbol, position), shift in zip(atoms, displacement, strict=True)
            ]
            energies.append(
                _mixed_pyscf_energy(calculator, moved, spin, fitted_range=True)
            )
        assert -np.sum(np.asarray(cold.forces) * displacement) == pytest.approx(
            (energies[0] - energies[1]) / (2 * step), abs=3e-7
        )


@pytest.mark.parametrize("device", ("cpu", "cuda"))
@pytest.mark.parametrize(
    "spin,auxiliary,basis,representation",
    [
        (spin, auxiliary, "sto-3g", "cartesian")
        for spin in (0, 1)
        for auxiliary in ("sto-3g", "def2-svp")
    ]
    + [
        (0, "def2-svp", "def2-svp", representation)
        for representation in ("cartesian", "spherical")
    ]
    + [(0, "def2-svp", "def2-tzvp", "spherical")],
)
def test_complete_ri_jk_vv10_energy_and_force(
    device: str,
    spin: int,
    auxiliary: str,
    basis: str,
    representation: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The oracle fits its own omega metric, including displaced auxiliary centers."""
    if device == "cuda" and os.environ.get("GENERATIVEQC_DFT_CUDA_TEST") != "1":
        pytest.skip("requires an explicitly Slurm-allocated CUDA device")
    if device == "cuda":
        assert os.environ.get("SLURM_JOB_ID")
    monkeypatch.setenv("GENERATIVEQC_DF_RANGE_EXCHANGE", "df")
    atoms = [("H", (0.0, 0.0, 0.0)), ("H", (0.15, 0.13, 1.5))]
    if spin:
        atoms.append(("H", (1.8, -0.1, -0.3)))
    if basis != "sto-3g":
        atoms = [
            ("O", (0.0, 0.0, 0.0)),
            ("H", (0.1, 1.43, 1.1)),
            ("H", (-0.15, -1.45, 1.12)),
        ]
    calculator = Calculator(
        method="wb97m-v-uks" if spin else "wb97m-v-rks",
        basis=basis,
        basis_representation=representation,
        auxiliary_basis=auxiliary,
        device=device,
        density_fitting="auto",
        initial_guess=None,
        ks_options=KsOptions(
            grid=GridSpec(radial_points=12, angular_polar=4, angular_azimuth=8)
        ),
        energy_tolerance=1e-12,
        density_tolerance=1e-10,
        screening_tolerance=1e-14,
        max_iterations=200,
    )
    with calculator.prepare_batch(
        [atoms], multiplicities=[spin + 1], warm_start=True
    ) as batch:
        cpu_work = []
        if device == "cpu":
            original_force = batch._public_dft_cpu_force

            def collect_force(index: int, geometry: Any) -> Any:
                result = original_force(index, geometry)
                cpu_work.append(result[1])
                return result

            batch._public_dft_cpu_force = collect_force
        cold = batch.execute(strict=True, properties=("energy", "forces")).items[0]
        with closing(NativeKsSnapshot(batch, 0)) as snapshot:
            assert snapshot.density_fitted
            assert snapshot.range_density_fitted
            with pytest.raises(ValueError, match="owns its"):
                snapshot.mixed_density_fitted_integral_derivatives(
                    len(atoms), 256 << 20, lr_correction=np.zeros((len(atoms), 3))
                )
            monkeypatch.setenv("GENERATIVEQC_DF_RANGE_EXCHANGE", "direct")
            assert snapshot.range_density_fitted
            warm = batch.execute(strict=True, properties=("energy", "forces")).items[0]
            with pytest.raises(ValueError, match="stale"):
                _ = snapshot.range_density_fitted
        work = (
            batch._stationary_cuda_execution.last_work
            if device == "cuda"
            else cpu_work[-1]
        )
        if device == "cpu":
            assert work["stationary_integral_derivative_route"] == "prepared-full-ri-jk"
            assert work["integral_quartet_passes"] == 0
            assert work["range_exchange_primitive_records"] == 0
        assert not work["native_integral_resources"][
            "additional_device_peak_includes_lazy_lr"
        ]
        moved = list(atoms)
        moved[1] = (
            atoms[1][0],
            tuple(np.asarray(atoms[1][1]) + np.array((0.03, -0.02, 0.04))),
        )
        rebound = batch.execute(
            coordinates=([position for _, position in moved],),
            strict=True,
            properties=("energy", "forces"),
        ).items[0]
        with closing(NativeKsSnapshot(batch, 0)) as snapshot:
            assert snapshot.range_density_fitted
    monkeypatch.setenv("GENERATIVEQC_DF_RANGE_EXCHANGE", "df")
    fresh = calculator.singlepoint(
        moved, multiplicity=spin + 1, properties=("energy", "forces")
    )
    assert rebound.energy == pytest.approx(fresh.energy, abs=2e-10)
    np.testing.assert_allclose(rebound.forces, fresh.forces, atol=2e-8, rtol=0)
    assert cold.converged and warm.converged
    assert (
        cold.executed_backend
        == warm.executed_backend
        == ("cpu_reference" if device == "cpu" else device)
    )
    assert warm.energy == pytest.approx(cold.energy, abs=2e-10)
    np.testing.assert_allclose(warm.forces, cold.forces, atol=2e-8, rtol=0)
    assert cold.energy == pytest.approx(
        _mixed_pyscf_energy(calculator, atoms, spin, fitted_range=True), abs=2e-8
    )
    if auxiliary == "sto-3g":
        assert abs(cold.energy - _mixed_pyscf_energy(calculator, atoms, spin)) > 1e-6
    direction = np.array(
        ((0.17, -0.11, 0.07), (-0.09, 0.06, 0.13), (0.04, -0.03, 0.08))
    )[: len(atoms)]
    step = 2e-4
    energies = []
    for sign in (1, -1):
        displaced = [
            (symbol, tuple(np.asarray(position) + sign * step * shift))
            for (symbol, position), shift in zip(atoms, direction, strict=True)
        ]
        energies.append(
            _mixed_pyscf_energy(calculator, displaced, spin, fitted_range=True)
        )
    assert -np.sum(np.asarray(cold.forces) * direction) == pytest.approx(
        (energies[0] - energies[1]) / (2 * step), abs=3e-7
    )


def test_invalid_range_mode_is_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    """An unsupported selection must never become a silent Direct fallback."""
    monkeypatch.setenv("GENERATIVEQC_DF_RANGE_EXCHANGE", "typo")
    with pytest.raises((ValueError, RuntimeError), match="DF_RANGE_EXCHANGE"):
        Calculator(
            method="wb97m-v",
            device="cpu",
            density_fitting="auto",
            ks_options=KsOptions(
                grid=GridSpec(radial_points=3, angular_polar=2, angular_azimuth=4)
            ),
        ).singlepoint([("H", (0, 0, 0)), ("H", (0, 0, 1.4))])


@pytest.mark.parametrize("coulomb_approximation", ("exact", "density_fitted"))
def test_lr_fitting_requires_an_operator_matched_owner(
    coulomb_approximation: str,
) -> None:
    """A Coulomb J must not share an erf-metric owner with fitted LR exchange."""
    specification = FockBuildSpec(
        coulomb=FockTerm(approximation=coulomb_approximation),
        exchange=FockTerm(
            coefficient=-0.5,
            operator="long_range",
            omega=0.3,
            approximation="density_fitted",
        ),
    )
    with (
        NativeAO([("H", (0, 0, 0)), ("H", (0, 0, 1.4))]) as basis,
        pytest.raises((ValueError, RuntimeError), match="separate metric owner"),
    ):
        FockPlan(basis, specification)


@pytest.mark.parametrize("omega", (0.03, 0.3, 3.0))
@pytest.mark.parametrize("materialized", (False, True))
def test_fixed_density_lr_metric_response(
    omega: float, materialized: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Both response storage choices differentiate the same operator-matched factors."""
    monkeypatch.setenv(
        "GENERATIVEQC_CPU_DF_MATERIALIZED_DERIVATIVES", "1" if materialized else "0"
    )
    atoms = [("H", (0.0, 0.0, -0.7)), ("H", (0.15, 0.13, 0.8))]
    density = np.array(((0.8, 0.13), (0.13, 0.6)))
    specification = FockBuildSpec(
        coulomb=FockTerm(present=False),
        exchange=FockTerm(
            coefficient=-0.5,
            operator="long_range",
            omega=omega,
            approximation="density_fitted",
        ),
    )
    with NativeAO(atoms) as basis, FockPlan(basis, specification) as plan:
        result = plan.evaluate(density, derivative=True)
        assert result.coulomb is None
        assert np.isfinite(result.exchange).all()
    step = 2e-5
    energies = []
    for sign in (1, -1):
        displaced = list(atoms)
        position = list(displaced[1][1])
        position[2] += sign * step
        displaced[1] = ("H", tuple(position))
        with NativeAO(displaced) as basis, FockPlan(basis, specification) as plan:
            energies.append(plan.evaluate(density).energy_two_electron)
    assert result.gradient[5] == pytest.approx(
        (energies[0] - energies[1]) / (2 * step), abs=2e-8
    )
    np.testing.assert_allclose(result.gradient[:3] + result.gradient[3:], 0, atol=2e-11)
