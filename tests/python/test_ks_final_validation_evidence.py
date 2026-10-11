"""Authenticate scoped KS proof evidence without launching another GPU cohort."""

from __future__ import annotations

import gzip
import hashlib
import json
import lzma
import statistics
from pathlib import Path

import pytest
from generativeqc_compiler.common.evidence import block_error
from generativeqc_compiler.common.performance import assess_comparison

from tools.generativeqc_validation.publication import validate_publication
from tools.generativeqc_validation.record import load_publication_record

ROOT = Path(__file__).resolve().parents[2]
BUNDLE = ROOT / "benchmarks/results/ks-resident-final-validation-20261010"


def evidence() -> dict:
    """Load only the publication-authenticated standard validation envelope."""
    return load_publication_record(BUNDLE)


def test_original_publication_keeps_its_exact_source_and_numerical_scope() -> None:
    """A later policy edit must not relabel source-frozen historical evidence."""
    publication = json.loads((BUNDLE / "publication.json").read_text())
    files = {
        entry["path"]: (BUNDLE / entry["path"]).read_bytes()
        for entry in publication["files"]
    }
    validate_publication(publication, files)
    record = evidence()
    assert publication["decision"]["scope"] == "numerical"
    assert record["settings"]["selector_default"] == 0
    assert not record["default_promoted"]
    assert record["performance"]["status"] == "not-run"
    assert record["stages"]["production"]["status"] == "not-run"
    assert record["revision"] == "c820ad08de1ceb33bc19c79115e0af2d52b01784"
    assert record["gates"] == {
        "energy_eh": 1e-8,
        "force_eh_per_bohr": 1e-7,
        "relative_floor": 0.02,
        "relative_mad_multiplier": 2,
    }
    for source in record["sources"].values():
        patch = gzip.decompress(files[source["patch_member"]])
        assert hashlib.sha256(patch).hexdigest() == source["patch_sha256"]
    assert record["sources"]["submission"]["patch_sha256"] == (
        "d44b766d753414cf7c2fdc1895b6dbac999901ebf1ada7069971c0c1eb5ee5ad"
    )


@pytest.mark.parametrize("cohort", ["prototype", "submission"])
@pytest.mark.parametrize("atoms", ["48", "96"])
def test_all_raw_samples_keep_numerical_work_and_timing_gates(
    cohort: str,
    atoms: str,
) -> None:
    """Recompute 80 comparisons, including the historical below-gate cohort."""
    record = evidence()
    case = record["cases"][cohort][atoms]
    assert case["status"] == "measured" and case["stage"] == "complete"
    assert case["selectors"] == {"baseline": 0, "candidate": 1}
    assert len(case["samples"]) == 20
    expected_library = record["sources"][cohort]["library_sha256"]
    assert all(
        arm["native_build"]["library_sha256"] == expected_library
        for arm in case["arms"].values()
    )
    work_by_geometry = {}
    for sample in case["samples"]:
        diagnostic = sample["diagnostics"]
        oracle = next(
            row
            for row in record["references"][atoms]["records"]
            if row["geometry"] == sample["geometry"]
            and row["phase"] in ("cold", "moved")
        )
        assert block_error(
            diagnostic["energy"],
            oracle["energy"],
            atol=1e-8,
            rtol=0,
        )["passed"]
        assert block_error(
            diagnostic["forces"],
            oracle["forces"],
            atol=1e-7,
            rtol=0,
        )["passed"]
        assert diagnostic["status"] == 0 and diagnostic["converged"]
        assert diagnostic["iterations"] == diagnostic["fock_builds"] == 1
        assert diagnostic["warm_start_used"] and not diagnostic["warm_start_fallback"]
        force = diagnostic["native_force_components"]
        bounds = force["resource_bounds"]
        assert (
            bounds["additional_device_peak_bound"] <= bounds["additional_device_budget"]
        )
        assert (
            bounds["additional_host_numeric_bound"] <= bounds["additional_host_budget"]
        )
        semantic_work = {
            "scf_ao": {
                key: value
                for key, value in diagnostic["native_scf_ao_work"].items()
                if key != "discovery_seconds"
            },
            "force_ao": {
                key: value
                for key, value in force["resident_ao_selection"]["work"].items()
                if key != "discovery_seconds"
            },
            "force_counts": force["work_counts"],
            "grid": force["grid_work_plan"],
            "point": diagnostic["actual_point_selection"],
        }
        assert semantic_work == work_by_geometry.setdefault(
            sample["geometry"], semantic_work
        )
    for phase in ("warm", "moved-warm"):
        samples = [row for row in case["samples"] if row["workload"] == phase]
        assessment = case["assessments"][phase]
        assert assess_comparison(samples) == assessment
        summary = assessment["workloads"][phase]
        spread = {}
        for side in ("baseline", "candidate"):
            timings = [row["seconds"] for row in samples if row["selection"] == side]
            assert len(timings) == 5
            assert timings == summary[side]["raw_seconds"]
            middle = statistics.median(timings)
            assert middle == summary[side]["median_seconds"]
            spread[side] = (
                statistics.median(abs(value - middle) for value in timings) / middle
            )
        noise = max(0.02, 2 * (spread["baseline"] + spread["candidate"]))
        reduction = (
            1
            - summary["candidate"]["median_seconds"]
            / summary["baseline"]["median_seconds"]
        )
        assert summary["noise_floor"] == noise
        assert summary["relative_improvement"] == reduction
        assert summary["significant"] == (reduction > noise)
        assert summary["significant"] == (
            not (cohort == "prototype" and atoms == "48" and phase == "warm")
        )


@pytest.mark.parametrize("cohort", ["prototype", "submission"])
@pytest.mark.parametrize("atoms", ["48", "96"])
def test_checkpoint_receipts_and_published_weight_lease_fallback(
    cohort: str,
    atoms: str,
) -> None:
    """Receipt equality is not an assertion that NPZ files can be reconstructed."""
    case = evidence()["cases"][cohort][atoms]
    checkpoints = {
        (row["selection"], row["phase"], row["label"]): row["blob_sha256"]
        for row in case["checkpoints"]
    }
    assert len(checkpoints) == 8
    for side in case["selectors"]:
        for phase in ("warm", "moved-warm"):
            assert (
                checkpoints[side, phase, "before"] == checkpoints[side, phase, "after"]
            )
    for sample in case["samples"]:
        assert (
            sample["frozen_seed_blob_sha256"]
            == checkpoints[sample["selection"], sample["workload"], "before"]
        )
        exported = sample["diagnostics"]["native_snapshot_export"]
        legacy_bytes = exported["spins"] * (
            (3 * exported["nao"] ** 2 + exported["nao"]) * 8 + 4
        )
        assert exported["reads"] == 1
        assert exported["selector"] == case["selectors"][sample["selection"]]
        assert exported["d2h_bytes"] - legacy_bytes == (
            112 if sample["selection"] == "candidate" else 0
        )
        assert exported["synchronizations"] == (
            3 if sample["selection"] == "candidate" else 1
        )
    assert len(case["setup"]) == 4
    for setup in case["setup"]:
        original = setup["diagnostics"]["native_snapshot_export"]
        repeated = setup["diagnostics"]["repeated_lease_export"]
        assert repeated["identity"] == original["identity"]
        assert repeated["reads"] == repeated["synchronizations"] == 1
        assert repeated["d2h_bytes"] == repeated["spins"] * (
            (3 * repeated["nao"] ** 2 + repeated["nao"]) * 8 + 4
        )


def test_raw_receipts_keep_failures_sanitizers_and_original_recipes() -> None:
    """Authenticate each uncompressed receipt as well as its storage envelope."""
    record = evidence()
    storage = record["raw_receipts"]
    raw = (BUNDLE / storage["path"]).read_bytes()
    assert len(raw) == storage["bytes"]
    assert hashlib.sha256(raw).hexdigest() == storage["sha256"]
    files = json.loads(lzma.decompress(raw))["files"]
    for receipt in files.values():
        decoded = receipt["utf8"].encode()
        assert len(decoded) == receipt["bytes"]
        assert hashlib.sha256(decoded).hexdigest() == receipt["sha256"]
    assert "ERROR SUMMARY: 0 errors" in files["submission-pr-v1/memcheck.log"]["utf8"]
    assert (
        "0 hazards displayed (0 errors, 0 warnings)"
        in files["submission-pr-v1/racecheck.log"]["utf8"]
    )
    for selector in (0, 1):
        assert (
            "PBE0 density provider admission route"
            in files[f"submission-pr-v1/ks-selector-{selector}.log"]["utf8"]
        )
        assert (
            int(
                files[f"submission-pr-v1/ks-selector-{selector}-exit-status.txt"][
                    "utf8"
                ]
            )
            != 0
        )
    assert record["broader_ks_suite"]["status"] == "fail"
    assert not record["broader_ks_suite"]["pristine_baseline_tested"]
    assert not record["broader_ks_suite"]["device_proof_admitted"]
    assert {
        "recipes/retained-owner.py",
        "recipes/analyze-endpoints.py",
        "endpoint-v4/paired-48.json",
        "endpoint-v4/paired-96.json",
    } <= files.keys()
    for name in ("prototype", "submission"):
        folder = "endpoint-v5" if name == "prototype" else "submission-pr-v1"
        assert files[f"{folder}/exit-status.txt"]["utf8"].strip() == "0"
        for atoms in ("48", "96"):
            assert (
                json.loads(files[f"{folder}/paired-{atoms}.json"]["utf8"])
                == record["cases"][name][atoms]
            )
