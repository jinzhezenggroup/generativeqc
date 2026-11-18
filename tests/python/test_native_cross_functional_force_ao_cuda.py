"""Public native force-map adoption, not test-profile/compatibility execution.

Run explicitly in Slurm with GENERATIVEQC_DFT_CUDA_TEST=1. The production
work crossover and occupancy guard are unchanged. Timings include complete
E/F calls and first preparation, but are not a multi-device promotion campaign.
"""

from __future__ import annotations

import ctypes
import gc
import hashlib
import json
import os
from dataclasses import replace
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
import pytest
from generativeqc import Calculator, GridSpec, KsOptions, _native
from generativeqc_compiler.dft.nonlocal_policy import MOLECULAR_VV10_DENSITY_THRESHOLD
from test_cross_functional_force_ao_cuda import (
    _assert_force_endpoint,
    _assert_physical_residual,
)

from benchmarks._cases import benchmark_cases
from benchmarks._retained_basis import load_retained_comparison_basis
from benchmarks.compare_gpu4pyscf_batch import load_comparison_basis
from benchmarks.readme_wb97mv import (
    configure_reference_full_fock,
    reference_engine,
    reference_vv10_domain,
)

pytestmark = pytest.mark.skipif(
    os.environ.get("GENERATIVEQC_DFT_CUDA_TEST") != "1",
    reason="requires an explicit finite Slurm GPU allocation",
)
ROOT = Path(__file__).resolve().parents[2]


def _minimum_force_capacity(batch: Any) -> dict[str, list[int]]:
    """Decline all optional force storage without changing SCF or dense bounds."""
    query = batch._library.generativeqc_ks_batch_stationary_cuda_capacity_v1
    query.argtypes = [
        ctypes.c_void_p,
        ctypes.c_uint32,
        ctypes.POINTER(ctypes.c_uint64),
        ctypes.c_size_t,
    ]
    query.restype = ctypes.c_int
    capacity = (ctypes.c_uint64 * 9)()
    _native.check(batch._library, query(batch._batch, 0, capacity, len(capacity)))
    automatic = list(capacity)
    limit = batch._library.generativeqc_ks_batch_stationary_cuda_limit_v1
    limit.argtypes = [ctypes.c_void_p, ctypes.c_uint32, ctypes.c_uint64]
    limit.restype = ctypes.c_int
    # An automatic tile may retire to the bounded tile when the limit shrinks.
    # Query that new envelope before testing its exact minimum boundary.
    assert limit(batch._batch, 0, 1) == _native.STATUS_OUT_OF_MEMORY
    _native.check(batch._library, query(batch._batch, 0, capacity, len(capacity)))
    assert limit(batch._batch, 0, capacity[4] - 1) == _native.STATUS_OUT_OF_MEMORY
    _native.check(batch._library, limit(batch._batch, 0, capacity[4]))
    return {"automatic": automatic, "bounded": list(capacity)}


@pytest.mark.parametrize("spin", ("rks", "uks"))
@pytest.mark.parametrize(
    "method,xc,fitted",
    (
        ("lda", "LDA_X,LDA_C_PW", True),
        ("pbe", "PBE", True),
        ("pbe0", "PBE0", True),
        ("r2scan", "R2SCAN", True),
        ("wb97m-v", "WB97M_V", True),
        ("pbe0", "PBE0", False),
        ("wb97m-v", "WB97M_V", False),
    ),
)
def test_public_native_force_map_adoption(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    record_property: Any,
    method: str,
    xc: str,
    fitted: bool,
    spin: str,
) -> None:
    """Independent moved E/F, actual sparse work and bounded native replay.

    Forty-eight water atoms cross the existing 2**35 work guard without artificial
    profiles. Separate, smaller VV10 quadrature keeps the nonlocal domain
    tractable and must never borrow the primary grid's masks. Both engines
    reconverge at each geometry; the force calls forbid CPU and Python force
    orchestration. Dense ablation preserves the selected arm's geometry cache.
    UKS adds a distant hydrogen radical, avoiding competing hole-localization
    solutions of a large water cation without borrowing a native SCF guess.
    WB97M-V UKS uses separated, fully spin-polarized hydrogen atoms: the
    low-spin water/H2/OH reference candidates fail strict independent SCF
    convergence. The high-spin fixture also exercises an empty beta channel;
    fitted range exchange independently verifies its full-rank metric model.
    """
    assert os.environ.get("SLURM_JOB_ID"), "real GPU tests require Slurm"
    assert os.environ.get("GENERATIVEQC_LIBRARY"), (
        "select a source-matched native library explicitly"
    )
    cupy = pytest.importorskip("cupy")
    pytest.importorskip("gpu4pyscf")
    from test_dft_complete_cuda import no_cpu_derivatives

    base = benchmark_cases()["water-32mer-4s4-def2-svp-spherical"]
    case = replace(base, atoms=base.atoms[:48], expected_ao_count=384)
    if spin == "uks":
        outer = np.max([position for _, position in case.atoms], axis=0) + [15, 12, 10]
        case = replace(
            case,
            atoms=(*case.atoms, ("H", tuple(outer))),
            expected_ao_count=389,
            multiplicity=2,
        )
        if method == "wb97m-v":
            case = replace(
                case,
                atoms=tuple(
                    (
                        "H",
                        (
                            10.0 * (atom % 5),
                            10.0 * ((atom // 5) % 4),
                            10.0 * (atom // 20),
                        ),
                    )
                    for atom in range(61)
                ),
                expected_ao_count=305,
                multiplicity=62,
            )
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
    basis_path = "benchmarks/results/pbe0-def2-svp-20261003/def2-svp-ho.json"
    auxiliary_path = (
        "benchmarks/results/issue206-practical-auxiliary/identity/cc-pvdz-jkfit.json"
    )
    auxiliary_retained = True
    if method == "wb97m-v" and fitted:
        # A large JKFIT LR metric is rank truncated under the native relative
        # cutoff, unlike GPU4PySCF's absolute cutoff. Use a verified full-rank
        # auxiliary fixture so both independent force oracles define one model.
        auxiliary, reference_auxiliary = load_comparison_basis(
            ROOT / basis_path, case, role="auxiliary", compute_forces=True
        )
        auxiliary_path, auxiliary_retained = basis_path, False
    grid = GridSpec(32, 10, 20)
    nonlocal_grid = GridSpec(12, 4, 8) if method == "wb97m-v" else None
    charge, multiplicity = case.charge, case.multiplicity
    if method == "wb97m-v" and fitted:
        monkeypatch.setenv("GENERATIVEQC_DF_RANGE_EXCHANGE", "df")
    coordinates = np.asarray([position for _, position in case.atoms])
    moved = coordinates.copy()
    moved[1, 0] += 0.002
    references = []
    reference_metrics = []
    with reference_vv10_domain(float(MOLECULAR_VV10_DENSITY_THRESHOLD)):
        for geometry in (coordinates, moved):
            atoms = [
                (element, point)
                for (element, _), point in zip(case.atoms, geometry, strict=True)
            ]
            reference = reference_engine(
                atoms,
                reference_basis,
                grid,
                xc=xc,
                spin=multiplicity - 1,
                charge=charge,
                nonlocal_spec=nonlocal_grid,
            )
            if fitted:
                reference = reference.density_fit(auxbasis=reference_auxiliary)
            if method == "wb97m-v" and fitted:
                from pyscf.df.addons import make_auxmol

                independent_auxiliary = make_auxmol(reference.mol, reference_auxiliary)
                scales = 1 / np.sqrt(np.diag(independent_auxiliary.intor("int1e_ovlp")))
                omega = reference._numint.rsh_and_hybrid_coeff(
                    xc, spin=multiplicity - 1
                )[0]
                for radial in (0.0, omega):
                    with independent_auxiliary.with_range_coulomb(radial):
                        metric = (
                            independent_auxiliary.intor("int2c2e")
                            * scales[:, None]
                            * scales[None, :]
                        )
                    spectrum = np.linalg.eigvalsh(metric)
                    assert np.all(np.isfinite(spectrum))
                    assert spectrum[0] > spectrum[-1] * 1e-10
                    reference_metrics.append(
                        {
                            "omega": radial,
                            "dimension": len(spectrum),
                            "minimum": float(spectrum[0]),
                            "maximum": float(spectrum[-1]),
                            "relative_threshold": 1e-10,
                        }
                    )
            configure_reference_full_fock(reference)
            reference.conv_tol, reference.conv_tol_grad = 1e-12, 1e-10
            reference.direct_scf_tol, reference.max_cycle = 1e-14, 200
            energy = float(reference.kernel())
            assert reference.converged
            gradient = reference.nuc_grad_method()
            gradient.grid_response = True
            if fitted:
                gradient.auxbasis_response = True
            forces = cupy.asnumpy(-gradient.kernel())
            _assert_force_endpoint(energy, forces, len(case.atoms))
            references.append((energy, forces))
    del reference, gradient
    gc.collect()
    cupy.get_default_memory_pool().free_all_blocks()
    record_property(
        "native_force_map_references",
        json.dumps(
            [
                {"energy": energy, "forces": forces.tolist()}
                for energy, forces in references
            ]
        ),
    )
    record_property("native_force_map_reference_metrics", json.dumps(reference_metrics))
    record_property(
        "native_force_map_input",
        json.dumps(
            {
                "method": f"{method}-{spin}",
                "fitted": fitted,
                "atoms": case.atoms,
                "moved": moved.tolist(),
                "charge": charge,
                "multiplicity": multiplicity,
                "aos": case.expected_ao_count,
                "grid": [32, 10, 20],
                "nonlocal_grid": [12, 4, 8] if nonlocal_grid is not None else None,
                "force_tile_points": None,
                "basis": basis_path,
                "auxiliary": auxiliary_path,
                "auxiliary_retained": auxiliary_retained,
                "density_fitting_relative_threshold": 1e-10,
                "energy_tolerance": 1e-12,
                "density_tolerance": 1e-10,
                "screening_tolerance": 1e-12,
                "max_iterations": 200,
                "basis_sha256": hashlib.sha256(
                    (ROOT / basis_path).read_bytes()
                ).hexdigest(),
                "auxiliary_sha256": hashlib.sha256(
                    (ROOT / auxiliary_path).read_bytes()
                ).hexdigest(),
                "library_sha256": hashlib.sha256(
                    Path(os.environ["GENERATIVEQC_LIBRARY"]).read_bytes()
                ).hexdigest(),
                "controls": {"GENERATIVEQC_DF_RANGE_EXCHANGE": "df"}
                if method == "wb97m-v" and fitted
                else {},
            }
        ),
    )

    def forbidden(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError(
            "native force acceptance entered Python force orchestration"
        )

    calculator = Calculator(
        f"{method}-{spin}",
        basis=basis,
        auxiliary_basis=auxiliary if fitted else None,
        basis_representation="spherical",
        device="cuda",
        precision="fp64",
        density_fitting="cuda" if fitted else "none",
        density_fitting_relative_threshold=1e-10,
        ks_options=KsOptions(grid=grid, nonlocal_grid=nonlocal_grid),
        energy_tolerance=1e-12,
        density_tolerance=1e-10,
        screening_tolerance=1e-12,
        max_iterations=200,
    )
    assert (
        Path(calculator._library._name).resolve()
        == Path(os.environ["GENERATIVEQC_LIBRARY"]).resolve()
    )
    samples = []
    dense_results = []
    trace = tmp_path / "native-force.jsonl"
    monkeypatch.setenv("GENERATIVEQC_DF_TRACE", str(trace))
    try:
        for mode in ("dense", "auto", "minimum"):
            if mode == "dense":
                monkeypatch.setenv("GENERATIVEQC_CUDA_FORCE_ACTIVE_AO", "0")
            else:
                monkeypatch.delenv("GENERATIVEQC_CUDA_FORCE_ACTIVE_AO", raising=False)
            prepared = perf_counter()
            with calculator.prepare_batch(
                [case.atoms],
                charges=[charge],
                multiplicities=[multiplicity],
                warm_start=True,
            ) as batch:
                preparation_seconds = perf_counter() - prepared
                assert batch._native_dft_force_eligible(0)
                monkeypatch.setattr(batch, "_public_dft_cuda_force", forbidden)
                if mode == "minimum":
                    record_property(
                        "native_force_map_capacity",
                        json.dumps(_minimum_force_capacity(batch)),
                    )
                for call, stage in enumerate(("cold", "warm", "moved", "moved-warm")):
                    trace.write_text("")
                    batch.set_warm_start_updates(stage in ("cold", "moved"))
                    with no_cpu_derivatives():
                        started = perf_counter()
                        result = batch.execute(
                            coordinates=(moved,) if call >= 2 else None,
                            strict=True,
                            properties=("energy", "forces"),
                        ).items[0]
                        seconds = perf_counter() - started
                    rows = [json.loads(line) for line in trace.read_text().splitlines()]
                    work = [
                        row["counters"]
                        for row in rows
                        if row["operation"] == "stationary_force"
                    ][-1]
                    expected_energy, expected_forces = references[int(call >= 2)]
                    _assert_force_endpoint(
                        result.energy, result.forces, len(case.atoms)
                    )
                    _assert_physical_residual(result.physical_residual_rms)
                    sample = {
                        "mode": mode,
                        "stage": stage,
                        "energy": result.energy,
                        "forces": result.forces.tolist(),
                        "energy_error": abs(result.energy - expected_energy),
                        "force_error": float(
                            np.max(np.abs(result.forces - expected_forces))
                        ),
                        "physical_residual_rms": result.physical_residual_rms,
                        "iterations": result.iterations,
                        "fock_builds": result.fock_builds,
                        "execute_seconds": seconds,
                        "endpoint_seconds": seconds
                        + (preparation_seconds if call == 0 else 0),
                        "work": work,
                    }
                    samples.append(sample)
                    record_property("native_force_map_endpoints", json.dumps(samples))
                    assert result.converged and result.executed_backend == "cuda"
                    assert sample["energy_error"] <= 1e-8
                    assert sample["force_error"] <= 3e-7
                    assert bool(work["indexed_grid"]) == (mode == "auto")
                    assert (work["grid_discovery_ao_jet_values"] > 0) == (
                        mode == "auto" and stage in ("cold", "moved")
                    )
                    active, dense = (
                        work["grid_active_point_ao"],
                        work["grid_dense_point_ao"],
                    )
                    assert 0 < active <= dense
                    assert (active < dense) == (mode == "auto")
                    if mode == "dense":
                        dense_results.append(
                            (result.energy, result.forces.copy(), work)
                        )
                    else:
                        energy, forces, dense_work = dense_results[call]
                        assert abs(result.energy - energy) <= 1e-9
                        np.testing.assert_allclose(
                            result.forces, forces, atol=1e-9, rtol=0
                        )
                        assert dense == dense_work["grid_dense_point_ao"]
                        for counter in (
                            "grid_ao_jet_values",
                            "grid_projection_fma_pairs",
                        ):
                            if mode == "auto":
                                assert work[counter] < dense_work[counter]
                            else:
                                assert work[counter] == dense_work[counter]
    finally:
        calculator.clear_cache()
