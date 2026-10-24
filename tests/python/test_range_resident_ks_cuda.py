"""Complete range-source selection, replay and bounded-refusal qualification."""

import json
import os
from pathlib import Path
from typing import Any

import pytest


def _auxiliary_basis(fitted: bool) -> Any:
    """Load the same pinned JK-fit inputs used by independent fitted E/F gates.

    The public runtime deliberately bundles only a small set of basis names;
    a reference-engine name does not make that auxiliary table available there.
    """
    if not fitted:
        return None
    from test_df_fast_public_cuda import auxiliary_bases

    from benchmarks._cases import benchmark_cases

    return auxiliary_bases(benchmark_cases()["water-def2-svp-spherical"])[0]


@pytest.mark.parametrize(
    "policy", ["full", "partial", "refused", "auto", "auto-refused"]
)
@pytest.mark.parametrize("fitted", (False, True))
def test_range_source_preserves_complete_geometry_and_warm_results(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, policy: str, fitted: bool
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
        if policy.startswith("auto"):
            if enabled:
                monkeypatch.delenv("GENERATIVEQC_KS_RANGE_RESIDENT_VALUES")
            monkeypatch.delenv("GENERATIVEQC_KS_RANGE_RESIDENT_REPLAY", raising=False)
            monkeypatch.delenv("GENERATIVEQC_KS_RESIDENT_BUCKET_MASK")
            if policy == "auto-refused":
                monkeypatch.setenv("GENERATIVEQC_KS_RESIDENT_MAX_BYTES", "1")
            else:
                monkeypatch.delenv("GENERATIVEQC_KS_RESIDENT_MAX_BYTES")
        calculator = Calculator(
            method="wb97m-v",
            basis="def2-svp",
            basis_representation="spherical",
            device="cuda",
            precision="fp64",
            density_fitting="cuda" if fitted else None,
            auxiliary_basis=_auxiliary_basis(fitted),
            max_iterations=100,
            energy_tolerance=1e-12,
            density_tolerance=1e-10,
            screening_tolerance=1e-14,
            ks_options=KsOptions(
                grid=GridSpec(48, 16, 32)
                if policy.startswith("auto")
                else GridSpec(12, 4, 8)
            ),
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
        admitted = enabled and policy not in {"refused", "auto-refused"}
        # Fitted primary plus exact LR is an explicit composition candidate;
        # automatic selection must retain its existing fallback.
        if admitted:
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


@pytest.mark.parametrize(
    ("policy", "fitted"),
    [
        ("full", False),
        ("partial", False),
        ("auto", False),
        ("full", True),
        ("partial", True),
        ("auto", True),
    ],
)
@pytest.mark.parametrize("invalidate", ["selection", "capacity", "buckets", "replay"])
def test_range_replay_reuses_values_and_falls_back_after_policy_change(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    policy: str,
    fitted: bool,
    invalidate: str,
) -> None:
    """Retained and retired sources must both certify the same warm density.

    Per-call work receipts distinguish actual cache consumption from numerical
    parity obtained by silently using the ordinary provider throughout.
    """
    if os.environ.get("GENERATIVEQC_TEST_WB97MV_CUDA") != "1":
        pytest.skip("requires an explicitly Slurm-allocated GPU")
    assert os.environ.get("SLURM_JOB_ID")
    import numpy as np
    from generativeqc import (
        BatchItemResult,
        Calculator,
        GridSpec,
        KsOptions,
        PreparedBatch,
    )

    monkeypatch.setenv("GENERATIVEQC_KS_RANGE_RESIDENT_VALUES", "1")
    monkeypatch.setenv("GENERATIVEQC_KS_RANGE_RESIDENT_REPLAY", "1")
    monkeypatch.setenv("GENERATIVEQC_KS_INCREMENTAL_DIRECT_JK", "0")
    monkeypatch.setenv("GENERATIVEQC_CUDA_KS_CHUNK", "1")
    monkeypatch.setenv(
        "GENERATIVEQC_KS_RESIDENT_BUCKET_MASK",
        "0x1555" if policy == "partial" else "0xfffffff",
    )
    monkeypatch.setenv("GENERATIVEQC_KS_RESIDENT_MAX_BYTES", str(8 << 30))
    if policy == "auto":
        for name in (
            "GENERATIVEQC_KS_RANGE_RESIDENT_VALUES",
            "GENERATIVEQC_KS_RANGE_RESIDENT_REPLAY",
            "GENERATIVEQC_KS_RESIDENT_BUCKET_MASK",
            "GENERATIVEQC_KS_RESIDENT_MAX_BYTES",
        ):
            monkeypatch.delenv(name)
    journal = tmp_path / "replay.jsonl"
    monkeypatch.setenv("GENERATIVEQC_DF_PROGRESS_TRACE", str(journal))
    atoms = [
        ("O", (0.02, -0.03, 0.04)),
        ("H", (0.1, 1.43, 1.1)),
        ("H", (-0.15, -1.45, 1.12)),
    ]
    calculator = Calculator(
        method="wb97m-v",
        basis="def2-svp",
        basis_representation="spherical",
        device="cuda",
        precision="fp64",
        density_fitting="cuda" if fitted else None,
        auxiliary_basis=_auxiliary_basis(fitted),
        max_iterations=100,
        energy_tolerance=1e-12,
        density_tolerance=1e-10,
        screening_tolerance=1e-14,
        ks_options=KsOptions(
            grid=GridSpec(48, 16, 32) if policy == "auto" else GridSpec(12, 4, 8)
        ),
    )
    evidence = []
    offset = 0

    def execute(owner: PreparedBatch) -> tuple[BatchItemResult, int, int]:
        nonlocal offset
        result = owner.execute(strict=True).items[0]
        lines = journal.read_text().splitlines()
        rows = [json.loads(line) for line in lines[offset:]]
        offset = len(lines)
        built = sum(r["value"] for r in rows if r.get("key") == "source_values_built")
        actions = sum(
            r["value"] for r in rows if r.get("key") == "phase_resident_fock_actions"
        )
        evidence.append(
            {
                "built": built,
                "actions": actions,
                "energy": result.energy,
                "iterations": result.iterations,
            }
        )
        assert result.converged
        return result, built, actions

    with calculator.prepare_batch([atoms], warm_start=True) as owner:
        cold, built, actions = execute(owner)
        assert built > 0 and actions > 0
        owner.set_warm_start_updates(False)
        for _ in range(2):
            warm, built, actions = execute(owner)
            assert built == 0 and actions > 0
            assert warm.iterations == 1 and abs(warm.energy - cold.energy) < 1e-10
            np.testing.assert_allclose(warm.forces, cold.forces, atol=1e-7, rtol=0)
        key, value = {
            "selection": ("GENERATIVEQC_KS_RANGE_RESIDENT_VALUES", "0"),
            "capacity": ("GENERATIVEQC_KS_RESIDENT_MAX_BYTES", "1"),
            "buckets": ("GENERATIVEQC_KS_RESIDENT_BUCKET_MASK", "0x1"),
            "replay": ("GENERATIVEQC_KS_RANGE_RESIDENT_REPLAY", "0"),
        }[invalidate]
        monkeypatch.setenv(key, value)
        fallback, built, actions = execute(owner)
        assert built == 0 and actions == 0
        assert fallback.iterations == 1 and abs(fallback.energy - cold.energy) < 1e-10
        np.testing.assert_allclose(fallback.forces, cold.forces, atol=1e-7, rtol=0)
    (tmp_path / "results.json").write_text(json.dumps(evidence, indent=2) + "\n")
