"""Complete R1 force routes: independent molecular oracles and native ownership."""

from __future__ import annotations

import os
from typing import Any

import numpy as np
import pytest
from generativeqc import Calculator, GridSpec, KsOptions
from test_treutler_endpoints import WATER, reference

METHODS = [
    ("lda", "LDA_X,LDA_C_PW"),
    ("pbe", "PBE"),
    ("pbe0", "PBE0"),
    ("pbe50", "0.5*HF+0.5*GGA_X_PBE,GGA_C_PBE"),
    ("b3lyp", "B3LYP"),
    ("r2scan", "R2SCAN"),
    ("wb97m-v", "WB97M_V"),
    ("m06-2x", "M06-2X"),
    ("mn15", "MN15"),
]
H2 = [("H", (0.12, -0.09, -0.71)), ("H", (-0.08, 0.13, 0.79))]


def require_backend(device: str) -> None:
    if device == "cuda" and not (
        os.environ.get("SLURM_JOB_ID")
        and os.environ.get("GENERATIVEQC_DFT_CUDA_TEST") == "1"
    ):
        pytest.skip("requires an explicitly allocated Slurm GPU")


def forbid_python_force(monkeypatch: pytest.MonkeyPatch) -> None:
    from generativeqc.batch import PreparedBatch

    def forbidden(*args: Any, **kwargs: Any) -> None:
        pytest.fail("qualified native force escaped to Python scientific orchestration")

    monkeypatch.setattr(PreparedBatch, "_public_dft_cpu_force", forbidden)
    monkeypatch.setattr(PreparedBatch, "_public_dft_cuda_force", forbidden)


def assert_final_state(
    calc: Calculator,
    batch: Any,
    atoms: list,
    spin: int,
    reference_state: Any,
    index: int,
) -> None:
    """Compare D/W after an independent AO transform, outside force execution."""
    from generativeqc._dft_gradient import StationaryKsState
    from generativeqc_compiler.dft import NativeAO

    with NativeAO(
        atoms,
        basis=calc._basis,
        representation="spherical",
        charge=spin,
        multiplicity=spin + 1,
    ) as basis:
        state = StationaryKsState.from_native(batch, basis, index=index)
        points = np.random.default_rng(2227).normal(size=(3 * basis.nao + 20, 3))
        actual_ao = basis.evaluate(points, order=0)[0]
        expected_ao = reference_state.mol.eval_gto("GTOval_sph", points)
        transform, _, rank, _ = np.linalg.lstsq(actual_ao, expected_ao, rcond=1e-13)
        assert rank == basis.nao
        np.testing.assert_allclose(
            actual_ao @ transform, expected_ao, atol=2e-11, rtol=2e-11
        )
        density = np.asarray(reference_state.make_rdm1()).reshape(
            (-1, basis.nao, basis.nao)
        )
        weighted = np.asarray(
            reference_state.nuc_grad_method().make_rdm1e(
                reference_state.mo_energy,
                reference_state.mo_coeff,
                reference_state.mo_occ,
            )
        ).reshape(density.shape)
        for native, oracle in (
            (state.density, density),
            (state.weighted_density, weighted),
        ):
            for actual, expected in zip(native, oracle, strict=True):
                np.testing.assert_allclose(
                    actual, transform @ expected @ transform.T, atol=4e-7, rtol=0
                )
        state._source.close()


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("fitted", [False, True])
@pytest.mark.parametrize("spin", [0, 1])
@pytest.mark.parametrize("method,functional", METHODS)
def test_all_curated_native_force_routes(
    device: str,
    fitted: bool,
    spin: int,
    method: str,
    functional: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Both spins and approximations keep the same independent Hamiltonian.

    Treutler level zero is an identical, independent quadrature prescription,
    not a grid-convergence claim. WB97M-V matches the declared molecular density
    domain, including its independent exact LR owner in the mixed DF model.
    """
    require_backend(device)
    if device == "cpu" and method in {"m06-2x", "mn15"}:
        pytest.skip("split-hybrid SCF programs currently have CUDA providers only")
    if device == "cuda" and method == "pbe50":
        pytest.skip("PBE50 SCF currently has a CPU provider only")
    forbid_python_force(monkeypatch)
    from tools._pyscf_vv10_domain import vv10_density_domain

    calc = Calculator(
        method=f"{method}-{'uks' if spin else 'rks'}",
        basis="sto-3g",
        basis_representation="spherical",
        device=device,
        density_fitting=device if fitted else "none",
        ks_options=KsOptions(
            grid=GridSpec.treutler(0),
            nonlocal_grid=GridSpec.treutler(0) if method == "wb97m-v" else None,
        ),
        max_iterations=200,
        energy_tolerance=1e-12,
        density_tolerance=1e-10,
        screening_tolerance=1e-13,
    )
    with calc.prepare_batch(
        [WATER, H2], charges=[spin, spin], multiplicities=[spin + 1, spin + 1]
    ) as batch:
        assert all(batch._native_dft_force_eligible(i) for i in range(2))
        actual = batch.execute(strict=True, properties=("energy", "forces"))
        warm = batch.execute(strict=True, properties=("energy", "forces"))
        for index, (atoms, result, replay) in enumerate(
            zip((WATER, H2), actual.items, warm.items, strict=True)
        ):
            final_states = []
            with vv10_density_domain(1e-10):
                energy, forces = reference(
                    calc,
                    fitted=fitted,
                    functional=functional,
                    spin=spin,
                    grid_level=0,
                    nonlocal_level=0,
                    atom_values=atoms,
                    final_states=final_states,
                )
            assert_final_state(calc, batch, atoms, spin, final_states[0], index)
            assert result.energy == pytest.approx(energy, abs=2e-8)
            np.testing.assert_allclose(result.forces, forces, atol=3e-7, rtol=0)
            np.testing.assert_allclose(replay.forces, result.forces, atol=3e-8, rtol=0)
            np.testing.assert_allclose(result.forces.sum(axis=0), 0, atol=3e-10, rtol=0)
        failed = batch.execute(
            [np.full((3, 3), np.nan), None], properties=("energy", "forces")
        )
        assert not failed.items[0].succeeded and failed.items[0].forces is None
        assert failed.items[1].succeeded
        recovered = batch.execute(strict=True, properties=("energy", "forces"))
        np.testing.assert_allclose(
            recovered.items[0].forces, actual.items[0].forces, atol=3e-8, rtol=0
        )
        moved = np.array([position for _, position in WATER])
        # Change both O-H distances while preserving the cation's orbital symmetry.
        moved[0, 2] += 0.02
        moved_atoms = [
            (element, tuple(position))
            for (element, _), position in zip(WATER, moved, strict=True)
        ]
        replay = batch.execute(
            [moved, None], strict=True, properties=("energy", "forces")
        ).items[0]
        with vv10_density_domain(1e-10):
            energy, forces = reference(
                calc,
                fitted=fitted,
                functional=functional,
                spin=spin,
                grid_level=0,
                nonlocal_level=0,
                atom_values=moved_atoms,
            )
        assert replay.energy == pytest.approx(energy, abs=2e-8)
        np.testing.assert_allclose(replay.forces, forces, atol=3e-7, rtol=0)


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize(
    "method", ["lda", "pbe", "pbe0", "b3lyp", "r2scan", "wb97m-v", "m06-2x", "mn15"]
)
@pytest.mark.parametrize("spin", [0, 1])
def test_native_ecp_force(
    device: str, method: str, spin: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    """ECP residuals and ionic nuclear charges must both reach the native sum."""
    require_backend(device)
    if device == "cpu" and method in {"m06-2x", "mn15"}:
        pytest.skip("split-hybrid SCF programs currently have CUDA providers only")
    forbid_python_force(monkeypatch)
    from generativeqc._dft_gradient import StationaryKsState
    from generativeqc_compiler.dft import NativeAO
    from test_ecp import fixture

    atoms, basis_record, _ = fixture(spin=spin)
    calc = Calculator(
        method=f"{method}-{'uks' if spin else 'rks'}",
        basis=basis_record,
        basis_representation="spherical",
        device=device,
        ks_options=KsOptions(
            grid=GridSpec(24, 8, 16),
            nonlocal_grid=GridSpec(24, 8, 16) if method == "wb97m-v" else None,
        ),
        max_iterations=200,
        energy_tolerance=1e-12,
        density_tolerance=1e-10,
    )
    with (
        calc.prepare_batch([atoms], charges=[spin], multiplicities=[spin + 1]) as batch,
        NativeAO(
            atoms,
            basis=basis_record,
            representation="spherical",
            charge=spin,
            multiplicity=spin + 1,
        ) as basis,
    ):
        assert batch._native_dft_force_eligible()
        actual = batch.execute(strict=True, properties=("energy", "forces")).items[0]
        # The ECP fixture's independent libcint Hamiltonian is used below.
        from test_ecp_stationary_cpu import reference as ecp_reference

        state = StationaryKsState.from_native(batch, basis)
        _, _, mol = fixture(spin=spin)
        if method in {"lda", "pbe"}:
            energy, gradient = ecp_reference(
                mol, state, f"{method}-{'uks' if spin else 'rks'}"
            )
            assert actual.energy == pytest.approx(energy, abs=2e-8)
            np.testing.assert_allclose(actual.forces, -gradient, atol=3e-7, rtol=0)
        # All electronic families also differentiate the full ECP energy.
        position = np.asarray([p for _, p in atoms])
        for step in (2e-4, 1e-4):
            values = []
            for sign in (1, -1):
                moved = position.copy()
                moved[1, 2] += sign * step
                values.append(
                    batch.execute([moved], strict=True, properties=("energy",))
                    .items[0]
                    .energy
                )
            assert actual.forces[1, 2] == pytest.approx(
                -(values[0] - values[1]) / (2 * step), abs=2e-6
            )


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("fitted", [False, True])
def test_wb97mv_force_above_old_atom_ceiling(
    device: str, fitted: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A nine-atom molecule must be governed by derivative providers and bytes."""
    require_backend(device)
    forbid_python_force(monkeypatch)
    from tools._pyscf_vv10_domain import vv10_density_domain

    atoms = [
        (element, (position[0] + 6.5 * copy, position[1], position[2]))
        for copy in range(3)
        for element, position in WATER
    ]
    calc = Calculator(
        method="wb97m-v-rks",
        basis="sto-3g",
        basis_representation="spherical",
        device=device,
        density_fitting=device if fitted else "none",
        ks_options=KsOptions(
            grid=GridSpec.treutler(0), nonlocal_grid=GridSpec.treutler(0)
        ),
        max_iterations=200,
        energy_tolerance=1e-12,
        density_tolerance=1e-10,
        screening_tolerance=1e-13,
    )
    with calc.prepare_batch([atoms]) as batch:
        assert batch._native_dft_force_eligible()
        result = batch.execute(strict=True, properties=("energy", "forces")).items[0]
        with vv10_density_domain(1e-10):
            energy, forces = reference(
                calc,
                fitted=fitted,
                functional="WB97M_V",
                grid_level=0,
                nonlocal_level=0,
                atom_values=atoms,
            )
        assert result.energy == pytest.approx(energy, abs=6e-8)
        np.testing.assert_allclose(result.forces, forces, atol=5e-7, rtol=0)


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("spin", [0, 1])
def test_native_fitted_long_range_force(
    device: str, spin: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The explicit all-DF Hamiltonian differentiates its own LR metric response."""
    require_backend(device)
    forbid_python_force(monkeypatch)
    monkeypatch.setenv("GENERATIVEQC_DF_RANGE_EXCHANGE", "df")
    from tools._pyscf_vv10_domain import vv10_density_domain

    calc = Calculator(
        method=f"wb97m-v-{'uks' if spin else 'rks'}",
        basis="sto-3g",
        basis_representation="spherical",
        device=device,
        density_fitting=device,
        ks_options=KsOptions(
            grid=GridSpec.treutler(0), nonlocal_grid=GridSpec.treutler(0)
        ),
        max_iterations=200,
        energy_tolerance=1e-12,
        density_tolerance=1e-10,
        screening_tolerance=1e-13,
    )
    with calc.prepare_batch(
        [WATER], charges=[spin], multiplicities=[spin + 1]
    ) as batch:
        assert batch._native_dft_force_eligible()
        actual = batch.execute(strict=True, properties=("energy", "forces")).items[0]
        final_states = []
        with vv10_density_domain(1e-10):
            energy, forces = reference(
                calc,
                fitted=True,
                functional="WB97M_V",
                spin=spin,
                grid_level=0,
                nonlocal_level=0,
                fitted_range=True,
                final_states=final_states,
            )
        assert_final_state(calc, batch, WATER, spin, final_states[0], 0)
        assert actual.energy == pytest.approx(energy, abs=2e-8)
        np.testing.assert_allclose(actual.forces, forces, atol=3e-7, rtol=0)
