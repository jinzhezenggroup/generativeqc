"""Cross-functional force-map acceptance at the explicit stationary CUDA endpoint.

Small, explicitly admitted fixtures qualify shared consumers and map lifetimes,
not a performance crossover or a production default for composite functionals.
Run with GENERATIVEQC_DFT_CUDA_TEST=1 inside a finite Slurm allocation.
"""

from __future__ import annotations

import json
import os
from dataclasses import replace
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
import pytest
from generativeqc import Calculator, GridSpec, KsOptions
from generativeqc import _force_active_ao as policy
from generativeqc_compiler.dft.nonlocal_policy import MOLECULAR_VV10_DENSITY_THRESHOLD

from benchmarks._cases import benchmark_cases
from benchmarks.compare_gpu4pyscf_batch import load_comparison_basis
from benchmarks.readme_wb97mv import reference_engine, reference_vv10_domain

pytestmark = pytest.mark.skipif(
    os.environ.get("GENERATIVEQC_DFT_CUDA_TEST") != "1",
    reason="requires a finite Slurm GPU allocation",
)
ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize("spin", ["rks", "uks"])
@pytest.mark.parametrize(
    "method,xc,producer,split_grid",
    [
        ("lda", "LDA_X,LDA_C_PW", "exact-jets-native-bitmask", False),
        ("pbe", "PBE", "exact-jets-native-bitmask", False),
        ("pbe0", "PBE0", "exact-jets-native-bitmask", False),
        ("r2scan", "R2SCAN", "exact-jets-native-bitmask", False),
        ("wb97m-v", "WB97M_V", "exact-jets-native-bitmask", False),
        ("wb97m-v", "WB97M_V", "exact-jets-native-bitmask", True),
        ("wb97m-v", "WB97M_V", "sampled-jets", False),
    ],
)
def test_cross_functional_force_maps(
    monkeypatch: pytest.MonkeyPatch,
    record_property: Any,
    spin: str,
    method: str,
    xc: str,
    producer: str,
    split_grid: bool,
) -> None:
    """Gate E/F, executed AO work, warm reuse, geometry changes and fallback.

    The oracle reconverges independently on matching atomic quadrature rules,
    including moving-grid response. Water cation provides unequal spin blocks
    without the neutral-OH convergence ambiguity. Independent XC/NLC grids must
    never share maps, even though the force arena is shared. Cold denotes the
    first prepared call in this process, including preparation time; these are
    numerical acceptance samples, not fresh-process performance qualification.
    """
    assert os.environ.get("SLURM_JOB_ID"), "real GPU tests require Slurm"
    cp = pytest.importorskip("cupy")
    pytest.importorskip("gpu4pyscf")
    from test_dft_complete_cuda import no_cpu_derivatives

    case = benchmark_cases()["water-tetramer-def2-svp-spherical"]
    if spin == "uks":
        case = replace(case, atoms=case.atoms[:3])
    basis, reference_basis = load_comparison_basis(
        ROOT / "benchmarks/results/pbe0-def2-svp-20261003/def2-svp-ho.json",
        case,
        role="orbital",
        compute_forces=True,
    )
    grid = GridSpec(radial_points=32, angular_polar=10, angular_azimuth=20)
    nonlocal_grid = replace(grid, radial_points=24) if split_grid else None
    coordinates = np.asarray([position for _, position in case.atoms])
    moved = coordinates.copy()
    moved[1, 0] += 0.002
    geometries = (coordinates, moved)
    charge = int(spin == "uks")
    references = []
    # Match only the public VV10 density domain; all reference algebra and SCF
    # iterations remain GPU4PySCF-owned and outside native endpoint timing.
    with reference_vv10_domain(float(MOLECULAR_VV10_DENSITY_THRESHOLD)):
        for points in geometries:
            atoms = [
                (element, point)
                for (element, _), point in zip(case.atoms, points, strict=True)
            ]
            reference = reference_engine(
                atoms,
                reference_basis,
                grid,
                xc=xc,
                spin=charge,
                charge=charge,
                nonlocal_spec=nonlocal_grid,
                full_fock=True,
            )
            reference.conv_tol = 1e-12
            reference.conv_tol_grad = 1e-10
            reference.direct_scf_tol = 1e-14
            reference.max_cycle = 200
            energy = float(reference.kernel())
            assert reference.converged
            gradient = reference.nuc_grad_method()
            gradient.grid_response = True
            references.append((energy, cp.asnumpy(-gradient.kernel())))
    record_property(
        "force_map_reference",
        json.dumps(
            [
                {"energy": energy, "forces": forces.tolist()}
                for energy, forces in references
            ]
        ),
    )

    calculator = Calculator(
        method=f"{method}-{spin}",
        basis=basis,
        basis_representation="spherical",
        device="cuda",
        ks_options=KsOptions(grid=grid, nonlocal_grid=nonlocal_grid),
        energy_tolerance=1e-12,
        density_tolerance=1e-10,
        screening_tolerance=1e-12,
        max_iterations=200,
    )
    # Admit by capability in this test only. Composite production still has no
    # qualified profile; small-fixture timings must not promote a new default.
    profile = policy.QualifiedForceActiveAoProfile(
        profile_id="test-cross-functional-force-maps",
        evidence=("test-only-independent-gpu4pyscf-endpoints",),
        compositions=("ordinary", "composite"),
        derivative_orders=(1, 2),
        spin_blocks=(1, 2),
        density_fitted=False,
        min_dense_point_ao_square_work=1,
        tile_policy="budget-auto",
        tile_points=None,
        min_device_bytes=1,
        min_host_bytes=1,
        cutoff=1e-16,
        cache_bytes=64 << 20,
        producer=producer,
    )
    samples = []
    native_results = []
    with calculator.prepare_batch(
        [case.atoms], charges=[charge], multiplicities=[charge + 1], warm_start=True
    ) as batch:

        def forbidden_python_force(*args: Any, **kwargs: Any) -> Any:
            pytest.fail("public native-force control entered Python orchestration")

        monkeypatch.setattr(batch, "_public_dft_cuda_force", forbidden_python_force)
        for points, (energy, forces) in zip(geometries, references, strict=True):
            with no_cpu_derivatives():
                native = batch.execute(
                    coordinates=(points,), strict=True, properties=("energy", "forces")
                ).items[0]
            assert native.converged and native.executed_backend == "cuda"
            assert abs(native.energy - energy) <= 1e-8
            np.testing.assert_allclose(native.forces, forces, atol=3e-7, rtol=0)
            native_results.append(native)
    record_property(
        "force_map_native_control",
        json.dumps(
            [
                {"energy": result.energy, "forces": result.forces.tolist()}
                for result in native_results
            ]
        ),
    )

    dense_results = []
    for mode, profiles in (("dense", ()), ("selected", (profile,))):
        monkeypatch.setattr(policy, "QUALIFIED_FORCE_ACTIVE_AO_PROFILES", profiles)
        prepare_started = perf_counter()
        with calculator.prepare_batch(
            [case.atoms],
            charges=[charge],
            multiplicities=[charge + 1],
            warm_start=True,
        ) as batch:
            preparation_seconds = perf_counter() - prepare_started
            previous = None
            for call, (stage, geometry) in enumerate(
                (
                    ("cold", 0),
                    ("warm", 0),
                    ("moved", 1),
                    ("moved-warm", 1),
                    ("zero-cache", 1),
                )
            ):
                if stage == "zero-cache":
                    monkeypatch.setattr(
                        policy,
                        "QUALIFIED_FORCE_ACTIVE_AO_PROFILES",
                        (replace(profile, cache_bytes=0),) if profiles else (),
                    )
                with no_cpu_derivatives():
                    started = perf_counter()
                    result = batch.execute(
                        coordinates=None if geometry == 0 else (moved,),
                        strict=True,
                        properties=("energy",),
                    ).items[0]
                    # Current public native forces bypass the experimental
                    # selector. Invoke its retained stationary consumer
                    # explicitly after a real SCF; never spoof native
                    # capability or count a dense native call as map work.
                    atoms = [
                        (element, point)
                        for (element, _), point in zip(
                            case.atoms, geometries[geometry], strict=True
                        )
                    ]
                    actual_forces, force_work = batch._public_dft_cuda_force(0, atoms)
                    seconds = perf_counter() - started
                energy, forces = references[geometry]
                energy_error = abs(result.energy - energy)
                force_error = float(np.max(np.abs(actual_forces - forces)))
                samples.append(
                    {
                        "mode": mode,
                        "stage": stage,
                        "execute_seconds": seconds,
                        "preparation_seconds": preparation_seconds
                        if call == 0
                        else 0.0,
                        "endpoint_seconds": seconds
                        + (preparation_seconds if call == 0 else 0.0),
                        "energy_error": energy_error,
                        "force_error": force_error,
                        "energy": result.energy,
                        "forces": actual_forces.tolist(),
                        "physical_residual_rms": result.physical_residual_rms,
                        "iterations": result.iterations,
                        "fock_builds": result.fock_builds,
                        "route": "explicit-stationary",
                        "work": force_work,
                    }
                )
                # Preserve diagnostic work and all measured stages in JUnit,
                # including a numerical failure; a policy label is insufficient.
                record_property("force_map_endpoints", json.dumps(samples))
                assert result.executed_backend == "cuda" and result.converged
                assert result.physical_residual_rms <= 1e-9
                assert energy_error <= 1e-8
                assert force_error <= 3e-7
                np.testing.assert_allclose(
                    actual_forces, native_results[geometry].forces, atol=1e-9, rtol=0
                )
                owner = batch._stationary_cuda_execution._resident_ao_maps
                if mode == "dense":
                    dense_results.append((result.energy, actual_forces.copy()))
                    assert owner is None
                    continue
                baseline = dense_results[call]
                assert result.energy == pytest.approx(baseline[0], abs=1e-9)
                np.testing.assert_allclose(
                    actual_forces, baseline[1], atol=1e-9, rtol=0
                )
                assert owner.domain.derivative_order == (1 if method == "lda" else 2)
                work = owner.work
                assert work["discovery_producer"] == producer
                if stage == "zero-cache":
                    assert owner is not previous
                    assert work["dense_budget_tiles"] == work["tile_count"] > 0
                    assert (
                        work["point_ao_square_sum"] == work["dense_point_ao_square_sum"]
                    )
                    assert work["discoveries"] == 0
                    assert (
                        force_work["force_active_ao_policy"]["actual_mode"]
                        == "dense-identity"
                    )
                else:
                    assert (
                        0
                        < work["point_ao_square_sum"]
                        < work["dense_point_ao_square_sum"]
                    )
                    assert (
                        force_work["force_active_ao_policy"]["actual_mode"]
                        == "selected"
                    )
                    if stage in ("warm", "moved-warm"):
                        assert owner is previous
                        assert work["discoveries"] == 0
                        assert work["cache_hits"] == work["tile_count"] > 0
                    else:
                        assert owner is not previous and work["discoveries"] > 0
                previous = owner
