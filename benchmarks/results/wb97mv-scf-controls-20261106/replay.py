"""Audit the rejected fixed SCF controls without a GPU or reference import."""

from __future__ import annotations

import hashlib
import itertools
import json
import lzma
import math
import re
import statistics
from pathlib import Path
from typing import Any

GATES = (1e-5, 1e-4, 3e-5)


def errors(sample: dict[str, Any], reference: dict[str, Any]) -> list[float]:
    """Evaluate every force component and the intensive energy gate."""
    actual, expected = sample["forces"], reference["forces"]
    assert len(actual) == len(expected) == 96
    assert all(len(row) == 3 for row in actual + expected)
    differences = [
        value - target
        for row, targets in zip(actual, expected, strict=True)
        for value, target in zip(row, targets, strict=True)
    ]
    assert all(math.isfinite(value) for value in differences)
    values = [
        abs(sample["energy"] - reference["energy"]) / 96,
        max(map(abs, differences)),
        math.sqrt(math.fsum(value * value for value in differences) / 288),
    ]
    assert all(math.isfinite(value) for value in values)
    return values


def junit(text: str) -> dict[str, int]:
    """Count pytest's emitted tags without parsing or expanding XML entities."""
    return {
        name: len(re.findall(rf"<{tag}(?:\s|/?>)", text))
        for name, tag in (
            ("cases", "testcase"),
            ("failures", "failure"),
            ("errors", "error"),
            ("skips", "skipped"),
        )
    }


def replay() -> dict[str, Any]:
    """Recompute settings/work/identity gates and all independent repeat pairs."""
    directory = Path(__file__).resolve().parent
    evidence = json.loads(
        lzma.decompress((directory / "evidence.json.xz").read_bytes())
    )
    assert evidence["schema"] == "generativeqc.wb97mv-scf-controls.v1"
    assert evidence["decision"] == "reject-both-controls"
    assert not evidence["production_changed"]
    reference_path = directory.parents[2] / evidence["reference"]["path"]
    reference_bytes = reference_path.read_bytes()
    assert (
        hashlib.sha256(reference_bytes).hexdigest() == evidence["reference"]["sha256"]
    )
    reference = json.loads(lzma.decompress(reference_bytes))
    targets = reference["records"]["A100/reference"]["samples"]
    assert len(targets) == 2 and all(sample["converged"] for sample in targets)
    records = evidence["records"]
    summary: dict[str, Any] = {
        "decision": evidence["decision"],
        "production_changed": False,
        "fresh_gpu4pyscf_ranking": False,
        "devices": {},
        "independent_pairs": 0,
        "worst_reference_errors": [0.0, 0.0, 0.0],
        "paired_errors": {},
        "qualification": {},
    }
    for device, arms in (
        ("A100", ("baseline", "minao", "orthogonal", "combined")),
        ("RTX5090", ("baseline", "orthogonal")),
    ):
        baseline = records[f"{device}/baseline"]
        assert evidence["jobs"][device]["exit_code"] == 0
        summary["devices"][device] = {}
        for arm in arms:
            key = f"{device}/{arm}"
            record = records[key]
            identity = evidence["identities"][key]
            assert record["status"] == "complete" and record["arm"] == "standard"
            assert record["method"] == "wb97m-v" and record["mode"] == "direct"
            assert record["atoms"] == 96 and record["spin"] == record["charge"] == 0
            assert record["geometry_bohr"] == reference["cohort"]["geometry_bohr"]
            assert record["basis_representation"] == "spherical"
            assert record["orbital_basis"] == "def2-SVP"
            assert record["source_revision"] == evidence["image"]["source_identity"]
            for field in ("source_identity", "library_sha256"):
                assert (
                    record["native_build"][field]
                    == identity[field]
                    == evidence["image"][field]
                )
            scheduler = record["scheduler"]
            assert scheduler == identity["scheduler"]
            job_field = "LSB_JOBID" if device == "A100" else "SLURM_JOB_ID"
            assert scheduler[job_field] == evidence["jobs"][device]["id"]
            assert scheduler["CUDA_VISIBLE_DEVICES"]
            assert device.replace("RTX", "RTX ") in record["accelerator"]["name"]
            experiment = record["solver_experiment"]
            assert experiment["guess_override"] == (
                "minao" if arm in ("minao", "combined") else "default"
            )
            assert experiment["orthogonal_diis"] == (
                "1" if arm in ("orthogonal", "combined") else "0"
            )
            samples = record["samples"]
            assert len(samples) == (2 if device == "A100" else 4)
            assert samples[0]["phase"] == "cold" and not samples[0]["warm_start_used"]
            for sample in samples:
                assert sample["converged"] and not sample["warm_start_fallback"]
                assert math.isfinite(sample["seconds"]) and sample["seconds"] > 0
                assert sample["settings"]["energy_tolerance"] == 1e-9
                assert sample["settings"]["density_tolerance"] == 1e-5
                assert sample["settings"]["screening_tolerance"] == 1e-12
                assert sample["settings"]["diis_history"] == 8
                controls = dict(sample["settings"])
                base_controls = dict(baseline["samples"][0]["settings"])
                controls.pop("initial_guess")
                base_controls.pop("initial_guess")
                assert controls == base_controls
                precision = sample["precision"]
                assert (
                    precision["complete"] and precision["operator_inventory_complete"]
                )
                assert precision["native_provenance"]["effective_bits"] == 64
                assert precision["native_provenance"]["mixed_stage_fock_builds"] == 0
                diagnostic = sample["ks_diagnostic"]
                assert diagnostic["coulomb_exchange_work"]["complete"]
                work = diagnostic["nonlocal_work"]
                if sample["phase"] == "cold":
                    assert sample["initial_guess"]["kind"] == (
                        "minao" if arm in ("minao", "combined") else "huckel"
                    )
                    assert sample["initial_guess"]["outcome"] == "used"
                    assert sample["iterations"] == sample["fock_builds"] == 18
                    assert work == {
                        "provisional_focks": 13,
                        "target_focks": 5,
                        "pair_cache_probes": 4,
                        "pair_cache_hits": 1,
                        "pair_sweeps": 4,
                    }
                    assert (
                        diagnostic["nonlocal_history_scopes"]
                        == ["provisional-semilocal-rsh"] * 9
                        + ["full-target"]
                        + ["provisional-frozen-vv10"] * 4
                        + ["full-target"] * 4
                    )
                else:
                    assert sample["phase"] == "warm" and sample["warm_start_used"]
                    assert sample["iterations"] == sample["fock_builds"] == 1
                    assert work["pair_sweeps"] == 0 and work["pair_cache_hits"] == 1
                for target in targets:
                    values = errors(sample, target)
                    assert all(
                        value <= gate for value, gate in zip(values, GATES, strict=True)
                    )
                    summary["worst_reference_errors"] = list(
                        map(
                            max,
                            zip(summary["worst_reference_errors"], values, strict=True),
                        )
                    )
                    summary["independent_pairs"] += 1
            if arm != "baseline":
                comparisons = [
                    errors(sample, other)
                    for sample, other in itertools.product(samples, baseline["samples"])
                ]
                summary["paired_errors"][key] = [
                    max(values[index] for values in comparisons) for index in range(3)
                ]
            cold = samples[0]
            warm = statistics.median(sample["seconds"] for sample in samples[1:])
            summary["devices"][device][arm] = {
                "cold_seconds": cold["seconds"],
                "warm_median_seconds": warm,
                "cold_percent_change": 100
                * (cold["seconds"] / baseline["samples"][0]["seconds"] - 1),
                "cold_iterations": cold["iterations"],
                "cold_fock_builds": cold["fock_builds"],
                "nonlocal_work": cold["ks_diagnostic"]["nonlocal_work"],
            }
    for name, text in evidence["qualification"]["junit"].items():
        result = junit(text)
        summary["qualification"][name] = result
        if name in ("host_repaired", "metric_strict", "independent_default"):
            assert result["failures"] == result["errors"] == 0
        if name in ("metric_strict", "independent_default"):
            assert result["skips"] == 0
    assert summary["qualification"]["metric_initial"]["failures"] == 2
    assert summary["independent_pairs"] == 32
    assert evidence["restoration"]["exit_code"] == 0
    assert evidence["restoration"]["policy"]["policy_version"] == 2
    assert (
        evidence["restoration"]["identity"]["source_identity"]
        == "ff6117e3ba5b481e6124f3bf80f4613fa27586d77bff88f5187f9290abbf5afa"
    )
    return summary


if __name__ == "__main__":
    print(json.dumps(replay(), indent=2, allow_nan=False))
