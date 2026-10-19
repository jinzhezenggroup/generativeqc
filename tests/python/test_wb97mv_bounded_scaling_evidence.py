"""Audit retained larger-system observations without requiring a speed win."""

from __future__ import annotations

import hashlib
import itertools
import lzma
from pathlib import Path

import numpy as np
import pytest
from generativeqc_compiler.common.evidence import block_error

from benchmarks.compare_wb97mv_hessian_gpu4pyscf import _summarize
from tools.generativeqc_validation.publication import validate_publication
from tools.generativeqc_validation.record import load_json, load_publication_record

DIRECTORY = (
    Path(__file__).resolve().parents[2]
    / "benchmarks/results/wb97mv-hessian-bounded-scaling-20261016"
)


def test_bounded_campaign_retains_authenticated_driver_and_binary_provenance() -> None:
    """Historical execution bytes are integrity inputs, not production-body pins."""
    manifest = load_json(DIRECTORY / "publication.json")
    files = {
        entry["path"]: (DIRECTORY / entry["path"]).read_bytes()
        for entry in manifest["files"]
    }
    validate_publication(manifest, files)
    evidence = load_publication_record(DIRECTORY)
    samples = load_publication_record(DIRECTORY, role="samples")
    drivers = {
        hashlib.sha256(lzma.decompress(files[f"{arm}-driver.py.xz"])).hexdigest()
        for arm in ("pilot", "followup", "final")
    }
    assert manifest["decision"]["status"] == "inconclusive"
    assert manifest["decision"]["scope"] == "performance"
    assert evidence["performance"]["status"] != "pass"
    assert evidence["hardware"]["exclusive_node"] is False
    binary_identities = set()
    for run in samples["runs"].values():
        provenance = run["provenance"]
        assert provenance["script_sha256"] in drivers
        assert provenance["source_commit"] == evidence["revision"]
        binary_identities.add(
            (provenance["cli_launcher_sha256"], provenance["native_library_sha256"])
        )
        for case in run["cases"].values():
            for row in case["samples"]:
                if row["engine"] == "generativeqc":
                    result = row["result"]
                    assert (
                        result["native_source_identity"] == evidence["hashes"]["source"]
                    )
                    diagnostics = result["diagnostics"]
                    assert not diagnostics["runtime_python"]
                    assert not diagnostics["runtime_compilation"]
                    assert diagnostics["numerical_host_fallbacks"] == 0
    assert len(binary_identities) == 1


def test_every_completed_pair_passes_the_unchanged_full_matrix_gate() -> None:
    """Recompute all pairs and medians, including unfavorable native timings."""
    samples = load_publication_record(DIRECTORY, role="samples")
    paired_cases = set()
    for run in samples["runs"].values():
        for name, case in run["cases"].items():
            native = [row for row in case["samples"] if row["engine"] == "generativeqc"]
            reference = [row for row in case["samples"] if row["engine"] == "gpu4pyscf"]
            size = 3 * len(case["atoms_bohr"])
            for candidate, baseline in itertools.product(native, reference):
                actual = np.asarray(candidate["result"]["matrix"])
                expected = np.asarray(baseline["result"]["matrix"])
                assert actual.shape == expected.shape == (size, size)
                assert block_error(actual, expected, atol=3e-6, rtol=0)["passed"]
                paired_cases.add(name)
            if native and reference:
                calculated = _summarize(case["samples"], 3e-6)
                assert case["summary"] == calculated
                assert calculated["accepted"]
    assert {"h2", "water", "formaldehyde", "water-dimer"} <= paired_cases


def test_memory_records_are_sampled_device_totals_not_owner_arena_bytes() -> None:
    """Keep watchdog semantics and observed footprint separate from exact peaks."""
    evidence = load_publication_record(DIRECTORY)
    samples = load_publication_record(DIRECTORY, role="samples")
    assert evidence["memory"]["peak_bytes"] is None
    assert evidence["settings"]["endpoint_timeout_seconds"] == 180
    for run in samples["runs"].values():
        assert run["workload"]["response_grid_policy"] == "same-as-primary"
        assert not run["workload"]["scientific_grid_convergence_claim"]
        for case in run["cases"].values():
            for observation in case["gpu_memory_observations"].values():
                assert observation["samples"] > 0
                assert (
                    0
                    <= observation["before_bytes"]
                    <= observation["sampled_peak_bytes"]
                )
                assert (
                    observation["sampled_peak_bytes"]
                    <= observation["device_total_bytes"]
                )
                assert observation["limit_bytes"] == 24 * 1024**3
                assert "not an exact peak" in observation["sampling_policy"]


@pytest.mark.parametrize(
    "name", ("pilot2-methanol", "pilot2-ethane", "followup-ethane-reference")
)
def test_unqualified_cases_remain_failed_without_a_speed_ratio(name: str) -> None:
    """A timeout or missing converged reference cannot become a speed win."""
    samples = load_publication_record(DIRECTORY, role="samples")
    run = samples["runs"][name]
    assert run["status"] == "failed"
    for case in run["cases"].values():
        assert case["failure"]
        assert not case.get("summary", {}).get("accepted", False)
        assert (
            case.get("summary", {}).get("generativeqc_over_gpu4pyscf_wall_ratio")
            is None
        )
    assert any("SCF did not converge" in text for text in samples["raw_logs"].values())
