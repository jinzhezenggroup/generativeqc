"""Replay staged-solver work and independent E/F gates without a GPU."""

from __future__ import annotations

import hashlib
import json
import lzma
import math
import statistics
from collections import Counter
from pathlib import Path
from typing import Any

GATES = (1e-5, 1e-4, 3e-5)


def replay() -> dict[str, Any]:
    """Distinguish complete endpoints from explicitly timeboxed physical prefixes."""
    directory = Path(__file__).resolve().parent
    evidence = json.loads(
        lzma.decompress((directory / "evidence.json.xz").read_bytes())
    )
    assert evidence["schema"] == "generativeqc.wb97mv-stage-handoff.v1"
    reference_path = directory.parents[2] / evidence["reference"]["path"]
    reference_bytes = reference_path.read_bytes()
    assert (
        hashlib.sha256(reference_bytes).hexdigest() == evidence["reference"]["sha256"]
    )
    reference = json.loads(lzma.decompress(reference_bytes))
    targets = reference["records"]["A100/reference"]["samples"]
    summary: dict[str, Any] = {
        "decision": evidence["decision"],
        "production_changed": evidence["production_changed"],
        "fresh_gpu4pyscf_ranking": False,
        "independent_pairs": 0,
        "worst_reference_errors": [0.0, 0.0, 0.0],
        "records": {},
        "clean_paired": {},
    }
    for key, record in evidence["records"].items():
        journal = evidence["journals"].get(key, [])
        if record["status"] != "complete":
            exit_code = evidence["terminations"][key]
            assert exit_code != 0
            summary["records"][key] = {
                "complete": False,
                "status": record["status"],
                "process_exit_code": exit_code,
                "timeboxed": exit_code == 124,
                "completed_physical_observations": len(journal) if journal else None,
                "last_observation": journal[-1] if journal else None,
                "accuracy_qualified": False,
            }
            continue
        assert record["geometry_bohr"] == reference["cohort"]["geometry_bohr"]
        assert record["method"] == "wb97m-v" and record["mode"] == "direct"
        assert record["atoms"] == 96 and record["basis_representation"] == "spherical"
        assert record["orbital_basis"] == "def2-SVP"
        assert evidence["terminations"][key] == 0
        assert (
            record["protocol"]["cold"] == "constructor+prepare+SCF+forces+host-export"
        )
        assert not record["protocol"]["grid_overrides"]
        assert ("A100" if key.startswith("A100") else "5090") in record["accelerator"][
            "name"
        ]
        image = evidence["images"][record["source_revision"]]
        assert record["native_build"]["source_identity"] == record["source_revision"]
        assert record["native_build"]["library_sha256"] == image["library_sha256"]
        identity = evidence["identities"][key]
        for field in ("source_identity", "library_sha256"):
            assert identity[field] == record["native_build"][field]
        assert record["scheduler"] == identity["scheduler"]
        assert record["scheduler"]["CUDA_VISIBLE_DEVICES"]
        device = key.split("/")[0]
        baseline_key = evidence.get("baseline_links", {}).get(
            device, f"{device}/baseline"
        )
        baseline = evidence["records"][baseline_key]
        samples = record["samples"]
        assert samples[0]["phase"] == "cold" and not samples[0]["warm_start_used"]
        worst = [0.0, 0.0, 0.0]
        for sample in samples:
            assert sample["converged"] and not sample["warm_start_fallback"]
            assert sample["settings"] == baseline["samples"][0]["settings"]
            assert sample["settings"]["energy_tolerance"] == 1e-9
            assert sample["settings"]["density_tolerance"] == 1e-5
            assert sample["settings"]["screening_tolerance"] == 1e-12
            assert sample["settings"]["force_grid_response"]
            assert sample["precision"]["effective_bits"] == 64
            assert sample["precision"]["mixed_stage_fock_builds"] == 0
            assert sample["precision"]["final_residual_audits"] == 1
            diagnostic = sample["ks_diagnostic"]
            work = diagnostic["coulomb_exchange_work"]
            assert work["complete"]
            assert work["coulomb_applications"] == sample["fock_builds"]
            assert work["exchange_applications"] == 2 * sample["fock_builds"]
            nonlocal_work = diagnostic["nonlocal_work"]
            assert (
                nonlocal_work["provisional_focks"] + nonlocal_work["target_focks"]
                == sample["fock_builds"]
            )
            assert sample["iterations"] == sample["fock_builds"]
            if sample["phase"] == "warm":
                assert sample["warm_start_used"]
            for target in targets:
                differences = [
                    value - expected
                    for row, other in zip(
                        sample["forces"], target["forces"], strict=True
                    )
                    for value, expected in zip(row, other, strict=True)
                ]
                assert len(differences) == 288
                errors = (
                    abs(sample["energy"] - target["energy"]) / 96,
                    max(map(abs, differences)),
                    math.sqrt(math.fsum(value * value for value in differences) / 288),
                )
                for index, (value, gate) in enumerate(zip(errors, GATES, strict=True)):
                    assert math.isfinite(value) and value < gate, (key, errors)
                    worst[index] = max(worst[index], value)
                    summary["worst_reference_errors"][index] = max(
                        summary["worst_reference_errors"][index], value
                    )
                summary["independent_pairs"] += 1
        cold = samples[0]
        state = cold["precision"]["returned_state_identity"].split(":")
        stages = [
            row
            for row in journal
            if row["owner"] == int(state[1]) and row["epoch"] == int(state[2])
        ]
        if stages:
            assert len(stages) == cold["fock_builds"]
            scopes = Counter(row["scope"] for row in stages)
            assert scopes[0] == cold["ks_diagnostic"]["nonlocal_work"]["target_focks"]
            assert (
                scopes[1] + scopes[2]
                == cold["ks_diagnostic"]["nonlocal_work"]["provisional_focks"]
            )
            sequence = [row["scope"] for row in stages]
        else:
            scope_codes = {
                "full-target": 0,
                "provisional-semilocal-rsh": 1,
                "provisional-frozen-vv10": 2,
            }
            sequence = [
                scope_codes[value]
                for value in cold["ks_diagnostic"]["nonlocal_history_scopes"]
            ]
            assert len(sequence) == cold["fock_builds"]
        warm = [sample["seconds"] for sample in samples if sample["phase"] == "warm"]
        summary["records"][key] = {
            "complete": True,
            "cold_seconds": cold["seconds"],
            "warm_seconds": warm,
            "warm_median_seconds": statistics.median(warm) if warm else None,
            "focks": cold["fock_builds"],
            "nonlocal_work": cold["ks_diagnostic"]["nonlocal_work"],
            "scopes": sequence,
            "worst_reference_errors": worst,
            "accuracy_qualified": True,
        }
    for device in ("A100-clean", "RTX5090-clean"):
        baseline = evidence["records"][f"{device}/baseline"]
        candidate = evidence["records"][f"{device}/default"]
        baseline_samples = baseline["samples"]
        candidate_samples = candidate["samples"]
        assert len(baseline_samples) == len(candidate_samples) == 4
        assert baseline["stage_experiment"]["rms_handoff_override"] == "0"
        assert candidate["stage_experiment"]["rms_handoff_override"] is None
        assert baseline["stage_experiment"]["clean"]
        assert candidate["stage_experiment"]["clean"]
        ratios = []
        for original, changed in zip(baseline_samples, candidate_samples, strict=True):
            assert original["phase"] == changed["phase"]
            if changed["phase"] == "warm":
                assert changed["fock_builds"] == original["fock_builds"] == 1
                assert (
                    changed["ks_diagnostic"]["nonlocal_work"]["provisional_focks"] == 0
                )
            ratio = changed["seconds"] / original["seconds"]
            assert ratio < evidence["qualification"]["paired_warm_guard"], (
                device,
                ratio,
            )
            ratios.append(ratio)
        summary["clean_paired"][device] = {
            "cold_improvement_percent": (1 - ratios[0]) * 100,
            "paired_call_ratios": ratios,
            "within_declared_guard": True,
        }
    baseline_stages = evidence["journals"]["A100/baseline"][:18]
    assert [row["scope"] for row in baseline_stages[:9]] == [1] * 9
    assert baseline_stages[6]["ao_rms"] < 1e-5 < baseline_stages[6]["ao_max"]
    assert baseline_stages[7]["ao_max"] > 1e-5 > baseline_stages[8]["ao_max"]
    assert baseline_stages[15]["mo_gradient"] > math.sqrt(1e-9)
    assert baseline_stages[16]["mo_gradient"] < math.sqrt(1e-9)
    assert baseline_stages[16]["energy_change"] > 1e-9
    assert baseline_stages[17]["energy_change"] < 1e-9
    baseline_work = summary["records"]["A100/baseline"]
    rms_work = summary["records"]["A100/rms"]
    assert baseline_work["focks"] - rms_work["focks"] == 2
    assert rms_work["scopes"].count(1) == 7
    assert baseline_work["scopes"].count(2) == rms_work["scopes"].count(2) == 4
    assert (
        baseline_work["nonlocal_work"]["pair_sweeps"]
        == rms_work["nonlocal_work"]["pair_sweeps"]
        == 4
    )
    assert summary["records"]["A100/history"]["focks"] == 23
    summary["causal_initializer_focks_saved"] = 2
    return summary


if __name__ == "__main__":
    print(json.dumps(replay(), indent=2, allow_nan=False))
