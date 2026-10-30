"""Public KS producer/force consumer hit gate; ordinary parity alone is not a hit."""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

import numpy as np
import pytest
from generativeqc import Calculator, KsOptions
from generativeqc_compiler.dft.grid import GridSpec

from benchmarks.df_component_ledger import read_trace

if TYPE_CHECKING:
    from pathlib import Path
    from typing import Any

pytestmark = pytest.mark.skipif(
    os.environ.get("GENERATIVEQC_DFT_CUDA_TEST") != "1",
    reason="requires an allocated native CUDA library/device",
)


@pytest.mark.parametrize(
    "method,functional", [("pbe0", "PBE0"), ("b3lyp", "B3LYP"), ("m06-2x", "M06-2X")]
)
@pytest.mark.parametrize("policy", ["auto", "candidate"])
def test_combined_df_hybrid_forces_against_independent_moving_grid(
    method: str, functional: str, policy: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Combined J/K cotangents retain full forces across cold, replay and motion.

    PySCF owns SCF, three-center/metric derivatives and grid response. This
    complements projection-hit checks, which only compare native execution
    variants and cannot independently validate the summed response weights.
    """
    from test_treutler_endpoints import WATER, reference

    assert os.environ.get("SLURM_JOB_ID") or os.environ.get("LSB_JOBID")
    monkeypatch.setenv("GENERATIVEQC_DF_SHELL_POLICY", policy)
    calc = Calculator(
        method=f"{method}-rks",
        basis="def2-svp",
        auxiliary_basis="def2-svp",
        basis_representation="spherical",
        device="cuda",
        density_fitting="cuda",
        energy_tolerance=1e-12,
        density_tolerance=1e-10,
        screening_tolerance=1e-13,
        max_iterations=200,
    )
    initial = np.array([xyz for _, xyz in WATER])
    moved = initial.copy()
    moved[1] += [0.02, -0.01, 0.03]
    with calc.prepare_batch([WATER], warm_start=True) as batch:
        for positions in (initial, moved):
            atoms = [
                (symbol, xyz) for (symbol, _), xyz in zip(WATER, positions, strict=True)
            ]
            energy, forces = reference(
                calc, fitted=True, functional=functional, atom_values=atoms
            )
            for _ in range(2):
                result = batch.execute(
                    [positions], strict=True, properties=("energy", "forces")
                ).items[0]
                assert result.energy == pytest.approx(energy, abs=2e-8, rel=0)
                np.testing.assert_allclose(result.forces, forces, atol=3e-7, rtol=0)
                np.testing.assert_allclose(
                    result.forces.sum(axis=0), 0, atol=2e-9, rtol=0
                )


def _assert_projection_reuse(
    rows: list[dict[str, Any]], expected_hit: bool | None, coordinates: int
) -> None:
    admitted = sum(
        r["counters"].get("response_reused_final_fitted_projection", 0) for r in rows
    )
    arithmetic = sum(
        r["counters"].get("response_final_fitted_projection_reused", 0) for r in rows
    )
    completed = sum(
        r["counters"].get("response_final_fitted_projection_reused", 0)
        for r in rows
        if r["counters"].get("atom_coordinates", 0) == coordinates
    )
    # Admission precedes allocation/OOM fallback and the force_response trace
    # scope, so its counter alone cannot establish reuse (and may be absent).
    # Arithmetic can also precede a later failure. Its SAME response row must
    # reach result D2H/synchronization (atom_coordinates), and the public call
    # must succeed. A failed attempt plus successful ordinary retry is no hit.
    if expected_hit is not None:
        assert completed == int(expected_hit)
        if not expected_hit:
            assert admitted == 0 and arithmetic == 0
    assert admitted <= 1 and arithmetic <= 1, "a one-shot projection was consumed twice"


def test_public_pbe0_packed_projection_hit_and_fallback(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    atoms = [
        ("O", (0.0, 0.0, 0.0)),
        ("H", (0.0, -1.43233673, 1.10715266)),
        ("H", (0.0, 1.43233673, 1.10715266)),
    ]
    for key, value in {
        "GENERATIVEQC_DF_EXCHANGE": "occupied",
        "GENERATIVEQC_DF_RESIDENT_EXCHANGE": "full",
        "GENERATIVEQC_DF_RESPONSE_SPACE": "auto",
        "GENERATIVEQC_DF_RESPONSE_STORAGE": "auto",
        "GENERATIVEQC_DF_OCCUPIED_RESPONSE_SOURCE": "auto",
    }.items():
        monkeypatch.setenv(key, value)
    answers = []
    # Cold owners ensure final occupied K is actually submitted; a one-step
    # warm solve is permitted to have no new projection and is not a hit gate.
    cases = [
        ("packed-single", "off", 1 << 30, False),
        ("packed-single", "auto", 1 << 30, True),
        ("dense", "auto", 1 << 30, False),
        ("packed-single", "auto", 8 << 20, None),
    ]
    for storage, reuse, budget, expected_hit in cases:
        trace = tmp_path / f"{storage}-{reuse}-{budget}.jsonl"
        monkeypatch.setenv("GENERATIVEQC_DF_VALUE_STORAGE", storage)
        monkeypatch.setenv("GENERATIVEQC_DF_FINAL_PROJECTION", reuse)
        monkeypatch.setenv("GENERATIVEQC_DF_TRACE", str(trace))
        calc = Calculator(
            method="pbe0-rks",
            basis="def2-svp",
            auxiliary_basis="def2-svp",
            basis_representation="spherical",
            device="cuda",
            density_fitting="auto",
            density_fitting_memory_budget_bytes=budget,
            ks_options=KsOptions(grid=GridSpec()),
            max_iterations=200,
            energy_tolerance=1e-12,
            density_tolerance=1e-10,
        )
        with calc.prepare_batch([atoms], warm_start=False) as batch:
            result = batch.execute(strict=True, properties=("energy", "forces")).items[
                0
            ]
        assert result.converged and result.executed_backend == "cuda"
        rows = [r for r in read_trace(trace) if r["operation"] == "force_response"]
        assert rows, "the public force endpoint did not reach DF response"
        _assert_projection_reuse(rows, expected_hit, 3 * len(atoms))
        # The actual public 8-MiB budget must complete, whether U remains
        # admitted or the existing bounded ordinary response is selected.
        np.testing.assert_allclose(result.forces.sum(axis=0), 0, atol=2e-9, rtol=0)
        answers.append(result)
    for result in answers[1:]:
        assert result.energy == pytest.approx(answers[0].energy, abs=1e-8, rel=0)
        np.testing.assert_allclose(result.forces, answers[0].forces, atol=3e-7, rtol=0)


@pytest.mark.parametrize("budget_mib", [24, 64])
def test_projection_scratch_preserves_warm_and_moved_geometry(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    budget_mib: int,
) -> None:
    """Exercise borrowed/owned scratch and compare all calls to bounded response.

    The 24-MiB public cap admits final U but cannot hold the preferred independent
    response panel. Trace completion proves the recycled interval was actually
    used; the 64-MiB control keeps independent scratch. A smaller explicit grid
    bounds this resource/lifetime test, without making a grid-accuracy claim.
    """
    from benchmarks.readme_hf_scaling import scaling_cases

    assert os.environ.get("SLURM_JOB_ID") or os.environ.get("LSB_JOBID")
    atoms = scaling_cases()["water-12"].atoms
    moved = np.array([xyz for _, xyz in atoms])
    moved[1, 0] += 0.02
    populations = []
    monkeypatch.setenv("GENERATIVEQC_DF_VALUE_STORAGE", "packed-single")
    for policy in ("off", "auto"):
        trace = tmp_path / f"{budget_mib}-{policy}.jsonl"
        monkeypatch.setenv("GENERATIVEQC_DF_FINAL_PROJECTION", policy)
        monkeypatch.setenv("GENERATIVEQC_DF_TRACE", str(trace))
        calculator = Calculator(
            method="pbe0-rks",
            basis="def2-svp",
            auxiliary_basis="def2-svp",
            basis_representation="spherical",
            device="cuda",
            density_fitting="auto",
            density_fitting_memory_budget_bytes=budget_mib << 20,
            ks_options=KsOptions(
                grid=GridSpec(radial_points=16, angular_polar=8, angular_azimuth=16)
            ),
            max_iterations=200,
            energy_tolerance=1e-12,
            density_tolerance=1e-10,
        )
        with calculator.prepare_batch([atoms], warm_start=True) as batch:
            cold = batch.execute(strict=True, properties=("energy", "forces")).items[0]
            warm = batch.execute(strict=True, properties=("energy", "forces")).items[0]
            changed = batch.execute(
                coordinates=[moved], strict=True, properties=("energy", "forces")
            ).items[0]
        populations.append((cold, warm, changed))
        recycled = [
            row
            for row in read_trace(trace)
            if row["operation"] == "force_response"
            and row["counters"].get(
                "response_recycled_final_projection_scratch_bytes", 0
            )
            and row["counters"].get("atom_coordinates") == 3 * len(atoms)
        ]
        assert len(recycled) == (3 if policy == "auto" and budget_mib == 24 else 0)
    for baseline, candidate in zip(*populations, strict=True):
        assert candidate.energy == pytest.approx(baseline.energy, abs=1e-9, rel=0)
        np.testing.assert_allclose(candidate.forces, baseline.forces, atol=3e-8, rtol=0)
