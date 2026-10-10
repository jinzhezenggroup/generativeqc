"""Qualify fitted force masks independently of DF integral/source storage."""

from __future__ import annotations

import os
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from generativeqc import Calculator, GridSpec, KsOptions
from generativeqc import _force_active_ao as policy

from benchmarks._cases import benchmark_cases
from benchmarks._retained_basis import load_retained_comparison_basis
from benchmarks.compare_gpu4pyscf_batch import load_comparison_basis
from benchmarks.readme_wb97mv import reference_engine

pytestmark = pytest.mark.skipif(
    os.environ.get("GENERATIVEQC_DFT_CUDA_TEST") != "1",
    reason="requires a finite Slurm GPU allocation",
)
ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize(
    "method,xc", [("lda", "LDA_X,LDA_C_PW"), ("pbe", "PBE"), ("pbe0", "PBE0")]
)
@pytest.mark.parametrize("spin", ["rks", "uks"])
def test_fitted_exact_force_maps_oracle_replay_rebinding_and_fallback(
    monkeypatch: pytest.MonkeyPatch, method: str, xc: str, spin: str
) -> None:
    """Exercise all AO jets, unequal spins, two geometries and dense fallback.

    Test-only work/occupancy admission exercises the production consumer on a
    small fixture; it does not qualify a small-system performance default.
    The water cation keeps the open-shell reference away from the problematic
    neutral-OH convergence case. Both engines use the same unpruned quadrature.
    """
    cupy = pytest.importorskip("cupy")
    pytest.importorskip("gpu4pyscf")
    from gpu4pyscf.dft import uks
    from test_dft_complete_cuda import no_cpu_derivatives

    assert os.environ.get("SLURM_JOB_ID")
    case = benchmark_cases()["water-tetramer-def2-svp-spherical"]
    if spin == "uks":
        case = replace(case, atoms=case.atoms[:3])
    basis, reference_basis = load_comparison_basis(
        ROOT / "benchmarks/results/pbe0-def2-svp-20261003/def2-svp-ho.json",
        case,
        role="orbital",
        compute_forces=True,
    )
    auxiliary, reference_auxiliary = load_retained_comparison_basis(
        ROOT
        / "benchmarks/results/issue206-practical-auxiliary/identity/cc-pvdz-jkfit.json",
        case,
        role="auxiliary",
        compute_forces=True,
    )
    grid = GridSpec(radial_points=32, angular_polar=10, angular_azimuth=20)
    coordinates = np.asarray([position for _, position in case.atoms])
    moved = coordinates.copy()
    moved[1, 0] += 0.002
    geometries = (coordinates, moved)
    charge = int(spin == "uks")
    references = []
    for points in geometries:
        atoms = [(element, point) for (element, _), point in zip(case.atoms, points)]
        reference = reference_engine(atoms, reference_basis, grid, xc=xc)
        if charge:
            reference.mol.build(charge=charge, spin=1)
            unrestricted = uks.UKS(reference.mol)
            unrestricted.xc = xc
            unrestricted.grids = reference.grids
            reference = unrestricted
        reference = reference.density_fit(auxbasis=reference_auxiliary)
        reference.conv_tol = 1e-12
        reference.conv_tol_grad = 1e-10
        reference.direct_scf_tol = 1e-14
        reference.max_cycle = 200
        energy = float(reference.kernel())
        assert reference.converged
        gradient = reference.nuc_grad_method()
        gradient.grid_response = True
        references.append((energy, cupy.asnumpy(-gradient.kernel())))

    calculator = Calculator(
        method=f"{method}-{spin}",
        basis=basis,
        auxiliary_basis=auxiliary,
        basis_representation="spherical",
        device="cuda",
        density_fitting="cuda",
        ks_options=KsOptions(grid=grid),
        energy_tolerance=1e-12,
        density_tolerance=1e-10,
        screening_tolerance=1e-12,
        max_iterations=200,
    )
    profile = next(
        profile
        for profile in policy.QUALIFIED_FORCE_ACTIVE_AO_PROFILES
        if profile.profile_id == "fitted-exact-jet-bitmask-v1-budget-auto"
    )
    profile = replace(
        profile, min_dense_point_ao_square_work=1, max_active_fraction=1.0
    )
    monkeypatch.setattr(policy, "QUALIFIED_FORCE_ACTIVE_AO_PROFILES", ())
    dense = []
    with calculator.prepare_batch(
        [case.atoms],
        charges=[charge],
        multiplicities=[charge + 1],
        warm_start=True,
    ) as batch:
        for points in geometries:
            with no_cpu_derivatives():
                result = batch.execute(
                    coordinates=(points,),
                    strict=True,
                    properties=("energy", "forces"),
                ).items[0]
            dense.append(result.forces.copy())

    monkeypatch.setattr(policy, "QUALIFIED_FORCE_ACTIVE_AO_PROFILES", (profile,))
    with calculator.prepare_batch(
        [case.atoms],
        charges=[charge],
        multiplicities=[charge + 1],
        warm_start=True,
    ) as batch:
        records = []
        original_force = batch._public_dft_cuda_force

        def observed_force(
            *arguments: Any, **keywords: Any
        ) -> tuple[np.ndarray, dict[str, Any]]:
            """Observe actual public work, without replacing any consumer."""
            forces, work = original_force(*arguments, **keywords)
            records.append(work)
            return forces, work

        monkeypatch.setattr(batch, "_public_dft_cuda_force", observed_force)
        previous = None
        for call, geometry in enumerate((0, 0, 1, 1)):
            with no_cpu_derivatives():
                result = batch.execute(
                    coordinates=None if geometry == 0 else (moved,),
                    strict=True,
                    properties=("energy", "forces"),
                ).items[0]
            energy, forces = references[geometry]
            assert result.executed_backend == "cuda" and result.converged
            assert result.physical_residual_rms <= 1e-9
            assert abs(result.energy - energy) <= 1e-8
            np.testing.assert_allclose(result.forces, forces, atol=3e-7, rtol=0)
            np.testing.assert_allclose(
                result.forces, dense[geometry], atol=1e-9, rtol=0
            )
            owner = batch._stationary_cuda_execution._resident_ao_maps
            assert owner.domain.derivative_order == (1 if method == "lda" else 2)
            work = records[-1]["resident_ao_selection"]["work"]
            assert work["discovery_producer"] == "exact-jets-native-bitmask"
            assert work["point_ao_square_sum"] < work["dense_point_ao_square_sum"]
            assert work["discovery_ao_panel_write_bytes"] == 0
            if call % 2:
                assert owner is previous
                assert work["discoveries"] == 0
                assert work["cache_hits"] == work["tile_count"] > 0
            else:
                assert owner is not previous and work["discoveries"] == 1
            previous = owner

        for fallback in (
            replace(profile, cache_bytes=0),
            replace(profile, max_active_fraction=0.0001),
        ):
            monkeypatch.setattr(
                policy, "QUALIFIED_FORCE_ACTIVE_AO_PROFILES", (fallback,)
            )
            with no_cpu_derivatives():
                result = batch.execute(
                    coordinates=(moved,),
                    strict=True,
                    properties=("energy", "forces"),
                ).items[0]
            np.testing.assert_allclose(
                result.forces, references[1][1], atol=3e-7, rtol=0
            )
            np.testing.assert_allclose(result.forces, dense[1], atol=1e-9, rtol=0)
            work = records[-1]["resident_ao_selection"]["work"]
            assert work["point_ao_square_sum"] == work["dense_point_ao_square_sum"]
            assert (
                work[
                    "dense_budget_tiles"
                    if not fallback.cache_bytes
                    else "dense_occupancy_tiles"
                ]
                == work["tile_count"]
                > 0
            )
