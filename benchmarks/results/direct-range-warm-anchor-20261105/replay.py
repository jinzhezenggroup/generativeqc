"""Recompute the retained Direct trial gates without a GPU or reference engine."""

from __future__ import annotations

import itertools
import json
import lzma
from pathlib import Path
from typing import Any

import numpy as np


def observable_errors(
    sample: dict[str, Any], target: dict[str, Any], atoms: int
) -> tuple[float, float, float]:
    """Use full force arrays and per-atom energy, including every repeat pair."""
    difference = np.asarray(sample["forces"]) - np.asarray(target["forces"])
    assert difference.shape == (atoms, 3) and np.isfinite(difference).all()
    energy = abs(sample["energy"] - target["energy"]) / atoms
    assert np.isfinite(energy)
    return (
        energy,
        float(np.abs(difference).max()),
        float(np.sqrt((difference**2).mean())),
    )


def replay() -> dict[str, Any]:
    """Validate work, common science gates, policy parity and the declared guard."""
    root = Path(__file__).resolve().parent
    publication = json.loads(
        lzma.decompress((root / "publication.json.xz").read_bytes())
    )
    assert publication["schema"] == "generativeqc.direct_range_warm_anchor_trial.v1"
    records = publication["records"]
    atoms = publication["cohort"]["atoms"]
    reference = records["A100/reference"]
    assert reference["status"] == "complete" and len(reference["samples"]) == 2
    all_samples = []
    summary = {
        "devices": {},
        "independent_pairs": 0,
        "worst_reference_errors": [0.0] * 3,
    }
    for device in ("A100", "RTX5090"):
        identity = publication["identities"][device]
        assert identity["source_identity"] == publication["source_identity"]
        assert identity["library_sha256"] == publication["core_sha256"]
        baseline, candidate = [
            records[device + "/" + selector] for selector in ("default", "1")
        ]
        for record in (baseline, candidate):
            assert record["status"] == "complete"
            assert record["source_revision"] == publication["source_identity"]
            samples = record["samples"]
            assert [sample["phase"] for sample in samples] == [
                "cold",
                "warm",
                "warm",
                "warm",
            ]
            assert not samples[0]["warm_start_used"]
            for sample in samples:
                assert sample["converged"]
                precision = sample["precision"]
                assert (
                    precision["complete"] and precision["operator_inventory_complete"]
                )
                assert precision["operator_work_counters_valid"]
                assert (
                    sum(
                        event["count"]
                        for event in precision["scf_fock_timeline"]
                        if event["kind"]
                        in ("mixed_fock", "strict_fock", "post_scf_fock")
                    )
                    == sample["fock_builds"]
                )
            assert all(
                sample["warm_start_used"]
                and sample["iterations"] == sample["fock_builds"] == 1
                for sample in samples[1:]
            )
            all_samples.extend(samples)
        initial, opted = baseline["samples"], candidate["samples"]
        assert initial[0]["iterations"] == opted[0]["iterations"] == 18
        assert initial[0]["fock_builds"] - opted[0]["fock_builds"] == 1
        ratios = []
        for index, (sample, target) in enumerate(zip(initial, opted, strict=True)):
            errors = observable_errors(sample, target, atoms)
            assert all(
                error <= gate
                for error, gate in zip(errors, (1e-9, 1e-7, 3e-8), strict=True)
            )
            left, right = sample["ks_diagnostic"], target["ks_diagnostic"]
            assert left["nonlocal_work"] == right["nonlocal_work"]
            assert left["coulomb_exchange_work"]["complete"]
            assert right["coulomb_exchange_work"]["complete"]
            for counter, saved in (
                ("coulomb_applications", 1),
                ("exchange_applications", 2),
            ):
                assert (
                    left["coulomb_exchange_work"][counter]
                    - right["coulomb_exchange_work"][counter]
                ) == (saved if index == 0 else 0)
            ratios.append(target["seconds"] / sample["seconds"])
        assert all(ratio <= 1.05 for ratio in ratios[1:])
        summary["devices"][device] = {
            "cold_seconds": [initial[0]["seconds"], opted[0]["seconds"]],
            "cold_reduction_percent": 100 * (1 - ratios[0]),
            "warm_opt_in_over_eager": ratios[1:],
            "lifecycle_reduction_percent": 100
            * (
                1
                - sum(sample["seconds"] for sample in opted)
                / sum(sample["seconds"] for sample in initial)
            ),
        }
    for sample, target in itertools.product(all_samples, reference["samples"]):
        errors = observable_errors(sample, target, atoms)
        assert all(
            error <= gate
            for error, gate in zip(errors, (1e-5, 1e-4, 3e-5), strict=True)
        )
        summary["independent_pairs"] += 1
        summary["worst_reference_errors"] = [
            max(previous, current)
            for previous, current in zip(
                summary["worst_reference_errors"], errors, strict=True
            )
        ]
    assert summary["independent_pairs"] == 32
    return summary


if __name__ == "__main__":
    print(json.dumps(replay(), indent=2))
