"""Replay source-matched complete endpoints, work and independent E/F off device."""

from __future__ import annotations

import hashlib
import json
import lzma
import math
import statistics
from pathlib import Path
from typing import Any
from xml.etree import ElementTree

GATES = (1e-5, 1e-4, 3e-5)


def replay() -> dict[str, Any]:
    """Retained reference timing is never a fresh GPU4PySCF performance ranking."""
    directory = Path(__file__).resolve().parent
    evidence = json.loads(
        lzma.decompress((directory / "evidence.json.xz").read_bytes())
    )
    assert evidence["schema"] == "generativeqc.shared-force-grid.v1"
    reference_bytes = (
        directory.parents[2] / evidence["reference"]["path"]
    ).read_bytes()
    assert (
        hashlib.sha256(reference_bytes).hexdigest() == evidence["reference"]["sha256"]
    )
    reference = json.loads(lzma.decompress(reference_bytes))
    targets = reference["records"]["A100/reference"]["samples"]
    summary: dict[str, Any] = {
        "decision": "default strict shared-panel consumption; retain bounded two-pass fallback",
        "source_identity": evidence["image"]["source_identity"],
        "library_sha256": evidence["image"]["library_sha256"],
        "base_commit": evidence["image"]["base_commit"],
        "fresh_gpu4pyscf_ranking": False,
        "independent_pairs": 0,
        "worst_reference_errors": [0.0, 0.0, 0.0],
        "records": {},
        "paired": {},
        "tests": {},
    }
    for name, population in evidence["populations"].items():
        record, identity = population["record"], population["identity"]
        assert population["exit_code"] == 0 and record["status"] == "complete"
        assert record["geometry_bohr"] == reference["cohort"]["geometry_bohr"]
        assert record["method"] == "wb97m-v" and record["mode"] == "direct"
        assert record["atoms"] == 96 and record["orbital_basis"] == "def2-SVP"
        assert record["basis_representation"] == "spherical"
        assert (
            record["protocol"]["cold"] == "constructor+prepare+SCF+forces+host-export"
        )
        assert not record["protocol"]["grid_overrides"]
        assert name.split("/")[0] in record["accelerator"]["name"]
        for field in ("source_identity", "library_sha256"):
            assert (
                record["native_build"][field]
                == identity[field]
                == evidence["image"][field]
            )
        assert record["source_revision"] == evidence["image"]["source_identity"]
        assert record["scheduler"] == identity["scheduler"]
        assert record["scheduler"]["CUDA_VISIBLE_DEVICES"]
        assert record["shared_grid_experiment"]["clean"]
        arm = name.split("/")[1]
        assert record["shared_grid_experiment"]["shared_grid_override"] == (
            "0" if arm == "baseline" else None
        )
        samples = record["samples"]
        assert len(samples) == 4
        diagnostics = record["separate_work_diagnostics"]
        assert len(diagnostics) == 1
        diagnostic_sample = diagnostics[0]
        assert diagnostic_sample["fock_builds"] == 1
        force_work = diagnostic_sample["force_work"]
        assert bool(force_work["shared_nonlocal_grid"]) == (arm == "default")
        assert force_work["nonlocal_pair_cache_hits"] == 1
        assert force_work["nonlocal_pair_sweeps"] == 0
        assert force_work["grid_discovery_passes"] == 0
        for index, sample in enumerate(samples):
            assert sample["converged"] and not sample["warm_start_fallback"]
            assert sample["warm_start_used"] == bool(index)
            assert sample["phase"] == ("warm" if index else "cold")
            assert math.isfinite(sample["seconds"]) and sample["seconds"] > 0
            settings = sample["settings"]
            assert settings["energy_tolerance"] == 1e-9
            assert settings["density_tolerance"] == 1e-5
            assert settings["screening_tolerance"] == 1e-12
            assert settings["force_grid_response"]
            assert sample["precision"]["effective_bits"] == 64
            assert sample["precision"]["mixed_stage_fock_builds"] == 0
            assert (
                sum(
                    solve["force_evaluations"]
                    for solve in sample["precision_validation"]["solves"]
                )
                == 1
            )
            diagnostic = sample["ks_diagnostic"]
            assert diagnostic["coulomb_exchange_work"]["complete"]
            assert (
                diagnostic["coulomb_exchange_work"]["coulomb_applications"]
                == sample["fock_builds"]
            )
            assert (
                diagnostic["coulomb_exchange_work"]["exchange_applications"]
                == 2 * sample["fock_builds"]
            )
            if index:
                assert sample["fock_builds"] == 1
        for sample in [*samples, diagnostic_sample]:
            for target in targets:
                differences = [
                    observed - expected
                    for row, other in zip(
                        sample["forces"], target["forces"], strict=True
                    )
                    for observed, expected in zip(row, other, strict=True)
                ]
                assert len(differences) == 288
                errors = (
                    abs(sample["energy"] - target["energy"]) / 96,
                    max(map(abs, differences)),
                    math.sqrt(sum(value * value for value in differences) / 288),
                )
                assert all(
                    math.isfinite(value) and value <= gate
                    for value, gate in zip(errors, GATES, strict=True)
                )
                summary["independent_pairs"] += 1
                summary["worst_reference_errors"] = [
                    max(old, value)
                    for old, value in zip(
                        summary["worst_reference_errors"], errors, strict=True
                    )
                ]
        summary["records"][name] = {
            "cold_seconds": samples[0]["seconds"],
            "warm_seconds": [sample["seconds"] for sample in samples[1:]],
            "warm_median_seconds": statistics.median(
                sample["seconds"] for sample in samples[1:]
            ),
            "cold_plus_three_warm_seconds": sum(
                sample["seconds"] for sample in samples
            ),
            "cold_focks": samples[0]["fock_builds"],
            "cold_scf_nonlocal_work": samples[0]["ks_diagnostic"]["nonlocal_work"],
            "separate_diagnostic_force_work": force_work,
            "scheduler": record["scheduler"],
        }
    for device in ("A100", "5090"):
        baseline = evidence["populations"][f"{device}/baseline"]["record"]
        candidate = evidence["populations"][f"{device}/default"]["record"]
        base_summary = summary["records"][f"{device}/baseline"]
        new_summary = summary["records"][f"{device}/default"]
        for index in range(4):
            for field in ("settings", "fock_builds", "iterations"):
                assert (
                    candidate["samples"][index][field]
                    == baseline["samples"][index][field]
                )
        base_diagnostic = baseline["separate_work_diagnostics"][0]
        new_diagnostic = candidate["separate_work_diagnostics"][0]
        for field in (
            "grid_evaluation_points",
            "grid_evaluation_tiles",
            "grid_active_point_ao",
            "grid_ao_jet_values",
            "grid_projection_fma_pairs",
        ):
            assert (
                base_diagnostic["force_work"][field]
                == 2 * new_diagnostic["force_work"][field]
            )
        assert new_diagnostic["force_work"]["grid_evaluation_points"] == 1078336
        warm_ratios = [
            new / old
            for new, old in zip(
                new_summary["warm_seconds"], base_summary["warm_seconds"], strict=True
            )
        ]
        assert max(warm_ratios) <= 1.05
        summary["paired"][device] = {
            "cold_ratio": new_summary["cold_seconds"] / base_summary["cold_seconds"],
            "warm_median_ratio": new_summary["warm_median_seconds"]
            / base_summary["warm_median_seconds"],
            "warm_per_call_ratios": warm_ratios,
            "cold_plus_three_warm_ratio": new_summary["cold_plus_three_warm_seconds"]
            / base_summary["cold_plus_three_warm_seconds"],
        }
    first = ElementTree.fromstring(evidence["tests"]["initial_xml"])  # noqa: S314 -- trusted retained local JUnit
    cases = first.findall(".//testcase")
    failures = [case for case in cases if case.find("failure") is not None]
    assert len(cases) == 12 and len(failures) == 1
    assert failures[0].get("name", "").endswith("[failure]")
    assert (
        "undefined symbol: ks_cuda_fail_next_runtime_for_test_v1"
        in failures[0].find("failure").text
    )
    hook = ElementTree.fromstring(evidence["tests"]["hook_xml"])  # noqa: S314 -- trusted retained local JUnit
    assert len(hook.findall(".//testcase")) == 1
    assert not hook.findall(".//failure") and not hook.findall(".//error")
    assert evidence["tests"]["hook_exit_code"] == 0
    assert (
        evidence["tests"]["hook_identity"]["source_identity"]
        == evidence["image"]["source_identity"]
    )
    assert (
        evidence["tests"]["hook_identity"]["library_sha256"]
        != evidence["image"]["library_sha256"]
    )
    assert "271 passed" in evidence["tests"]["host_log"]
    summary["tests"] = {
        "production_image_passed": 11,
        "initial_missing_hook_failures": 1,
        "separate_hook_image_passed": 1,
        "host_passed": 271,
    }
    return summary


if __name__ == "__main__":
    print(json.dumps(replay(), indent=2, sort_keys=True))
