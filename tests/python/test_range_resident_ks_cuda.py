"""Opt-in complete range-source replay and bounded-refusal qualification."""

import json
import os
from pathlib import Path

import pytest


@pytest.mark.parametrize("policy", ["full", "partial", "refused"])
def test_range_source_preserves_complete_geometry_and_warm_results(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, policy: str
) -> None:
    """Exercise real source consumption, derivatives, retirement and new geometry.

    The separate molecular suite provides the independent engine and energy
    derivative gates. This comparison isolates cache selection and capacity
    refusal while requiring a fully converged physical result at every call.
    """
    if os.environ.get("GENERATIVEQC_TEST_WB97MV_CUDA") != "1":
        pytest.skip("requires an explicitly Slurm-allocated GPU")
    assert os.environ.get("SLURM_JOB_ID")
    import numpy as np
    from generativeqc import Calculator, GridSpec, KsOptions

    for name in (
        "GENERATIVEQC_KS_INCREMENTAL_DIRECT_JK",
        "GENERATIVEQC_VV10_INCREMENTAL_SEED",
        "GENERATIVEQC_VV10_FROZEN_RANGE_SEED",
    ):
        monkeypatch.setenv(name, "0")
    monkeypatch.setenv("GENERATIVEQC_CUDA_KS_CHUNK", "1")
    atoms = [
        ("O", (0.02, -0.03, 0.04)),
        ("H", (0.1, 1.43, 1.1)),
        ("H", (-0.15, -1.45, 1.12)),
    ]
    moved = np.array([[xyz for _, xyz in atoms]])
    moved[0, -1, 0] += 0.001
    populations = []
    evidence = []
    for enabled in (False, True):
        journal = tmp_path / f"{policy}-{enabled}.jsonl"
        monkeypatch.setenv("GENERATIVEQC_DF_PROGRESS_TRACE", str(journal))
        monkeypatch.setenv("GENERATIVEQC_KS_RANGE_RESIDENT_VALUES", str(int(enabled)))
        monkeypatch.setenv(
            "GENERATIVEQC_KS_RESIDENT_BUCKET_MASK",
            "0x1555" if policy == "partial" else "0xfffffff",
        )
        monkeypatch.setenv(
            "GENERATIVEQC_KS_RESIDENT_MAX_BYTES",
            "1" if policy == "refused" else str(8 << 30),
        )
        calculator = Calculator(
            method="wb97m-v",
            basis="def2-svp",
            basis_representation="spherical",
            device="cuda",
            precision="fp64",
            max_iterations=100,
            energy_tolerance=1e-12,
            density_tolerance=1e-10,
            screening_tolerance=1e-14,
            ks_options=KsOptions(grid=GridSpec(12, 4, 8)),
        )
        with calculator.prepare_batch([atoms], warm_start=True) as owner:
            cold = owner.execute(strict=True).items[0]
            owner.set_warm_start_updates(False)
            warm = owner.execute(strict=True).items[0]
            owner.set_warm_start_updates(True)
            displaced = owner.execute(coordinates=moved, strict=True).items[0]
            owner.set_warm_start_updates(False)
            moved_warm = owner.execute(coordinates=moved, strict=True).items[0]
        population = (cold, warm, displaced, moved_warm)
        assert all(result.converged for result in population)
        for stationary, replay in ((cold, warm), (displaced, moved_warm)):
            assert abs(stationary.energy - replay.energy) < 1e-10
            assert replay.iterations == 1
        rows = [json.loads(line) for line in journal.read_text().splitlines()]
        built = [r["value"] for r in rows if r.get("key") == "source_values_built"]
        consumed = [
            r["value"] for r in rows if r.get("key") == "phase_resident_fock_actions"
        ]
        if enabled and policy != "refused":
            assert any(built) and any(consumed), "optional source was never consumed"
        else:
            assert not any(built) and not any(consumed)
        populations.append(population)
        evidence.append(
            {
                "enabled": enabled,
                "policy": policy,
                "source_values_built": built,
                "source_actions": consumed,
                "results": [
                    {"energy": r.energy, "diagnostic": r.ks_diagnostic.to_payload()}
                    for r in population
                ],
            }
        )
    (tmp_path / "results.json").write_text(json.dumps(evidence, indent=2) + "\n")
    for ordinary, cached in zip(*populations, strict=True):
        assert abs(ordinary.energy - cached.energy) < 1e-8
        np.testing.assert_allclose(ordinary.forces, cached.forces, atol=1e-7, rtol=0)
