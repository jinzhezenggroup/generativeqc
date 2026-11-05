"""Replay promoted-default science/work gates without GPU or reference imports."""

from __future__ import annotations

import hashlib
import itertools
import json
import lzma
from pathlib import Path
from typing import Any

import numpy as np


def observable_errors(
    sample: dict[str, Any], target: dict[str, Any], atoms: int
) -> tuple[float, float, float]:
    """Return common E/atom, force max and force RMS absolute errors."""
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
    """Check the new image against the separately retained opt-in qualification.

    The large independent reference is reused, not rerun. Original matched
    timings remain owned by that trial; these are single default observations.
    """
    root = Path(__file__).resolve().parent
    publication = json.loads(
        lzma.decompress((root / "publication.json.xz").read_bytes())
    )
    assert publication["schema"] == "generativeqc.direct_range_warm_anchor_default.v1"
    original_bytes = (
        root.parent / "direct-range-warm-anchor-20261105/publication.json.xz"
    ).read_bytes()
    assert (
        hashlib.sha256(original_bytes).hexdigest()
        == publication["qualification_publication_sha256"]
    )
    original = json.loads(lzma.decompress(original_bytes))
    assert publication["source_identity"] != original["source_identity"]
    assert publication["core_sha256"] != original["core_sha256"]
    assert publication["selection"] == "unset"
    atoms = original["cohort"]["atoms"]
    reference = original["records"]["A100/reference"]["samples"]
    summary: dict[str, Any] = {
        "source_identity": publication["source_identity"],
        "devices": {},
        "independent_pairs": 0,
        "worst_reference_errors": [0.0, 0.0, 0.0],
        "worst_policy_errors": [0.0, 0.0, 0.0],
    }
    all_samples = []
    for device in ("A100", "RTX5090"):
        identity = publication["identities"][device]
        assert identity["source_identity"] == publication["source_identity"]
        assert identity["library_sha256"] == publication["core_sha256"]
        record = publication["records"][device]
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
        opted = original["records"][device + "/1"]
        eager = original["records"][device + "/default"]
        assert record["occupied_xc_env"] == opted["occupied_xc_env"]
        assert record["protocol"] == opted["protocol"]
        assert record["thread_controls"] == opted["thread_controls"]
        for index, (sample, target) in enumerate(
            zip(samples, opted["samples"], strict=True)
        ):
            assert sample["converged"]
            assert np.isfinite(sample["seconds"]) and sample["seconds"] > 0
            assert sample["settings"] == target["settings"]
            assert (
                sample["iterations"]
                == sample["fock_builds"]
                == (18 if index == 0 else 1)
            )
            assert sample["warm_start_used"] == (index != 0)
            precision = sample["precision"]
            assert precision["complete"] and precision["operator_inventory_complete"]
            assert precision["operator_work_counters_valid"]
            assert (
                sum(
                    event["count"]
                    for event in precision["scf_fock_timeline"]
                    if event["kind"] in ("mixed_fock", "strict_fock", "post_scf_fock")
                )
                == sample["fock_builds"]
            )
            assert all(
                event["phase"] != "preparation"
                for event in precision["scf_fock_timeline"]
            )
            work, previous = sample["ks_diagnostic"], target["ks_diagnostic"]
            assert work["nonlocal_work"] == previous["nonlocal_work"]
            assert work["coulomb_exchange_work"]["complete"]
            for counter in ("coulomb_applications", "exchange_applications"):
                assert (
                    work["coulomb_exchange_work"][counter]
                    == previous["coulomb_exchange_work"][counter]
                )
            errors = observable_errors(sample, target, atoms)
            assert all(
                error <= gate
                for error, gate in zip(errors, (1e-9, 1e-7, 3e-8), strict=True)
            )
            summary["worst_policy_errors"] = [
                max(previous_error, current)
                for previous_error, current in zip(
                    summary["worst_policy_errors"], errors, strict=True
                )
            ]
        ratios = [
            sample["seconds"] / baseline["seconds"]
            for sample, baseline in zip(samples[1:], eager["samples"][1:], strict=True)
        ]
        assert all(ratio <= 1.05 for ratio in ratios)
        trace = publication["traces"][device]
        assert trace["open_scopes"] == 0
        regions = trace["diagnostic_per_execute_region_seconds"]
        assert len(regions) == 4
        assert all(
            region.get("cuda_ks_warm_energy_rebase", 0) == 0 for region in regions
        )
        summary["devices"][device] = {
            "default_seconds": [sample["seconds"] for sample in samples],
            "warm_default_over_trial_eager": ratios,
            "cold_focks_j_k": [
                samples[0]["fock_builds"],
                samples[0]["ks_diagnostic"]["coulomb_exchange_work"][
                    "coulomb_applications"
                ],
                samples[0]["ks_diagnostic"]["coulomb_exchange_work"][
                    "exchange_applications"
                ],
            ],
        }
        all_samples.extend(samples)
    for sample, target in itertools.product(all_samples, reference):
        errors = observable_errors(sample, target, atoms)
        assert all(
            error <= gate
            for error, gate in zip(errors, (1e-5, 1e-4, 3e-5), strict=True)
        )
        summary["independent_pairs"] += 1
        summary["worst_reference_errors"] = [
            max(previous_error, current)
            for previous_error, current in zip(
                summary["worst_reference_errors"], errors, strict=True
            )
        ]
    assert summary["independent_pairs"] == 16
    return summary


if __name__ == "__main__":
    print(json.dumps(replay(), indent=2))
