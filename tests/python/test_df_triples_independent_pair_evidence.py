"""Audit canonical independent-pair sources and every retained observation offline."""

import base64
import gzip
import hashlib
import json
import lzma
import shutil
import subprocess
from pathlib import Path

import pytest

from tools.generativeqc_validation.publication import validate_publication
from tools.generativeqc_validation.record import load_publication_record

ROOT = Path(__file__).resolve().parents[2]
BUNDLE = ROOT / "benchmarks/results/df-triples-independent-pair-20261011"


@pytest.fixture(scope="module")
def retained() -> tuple[dict, dict[str, bytes]]:
    """Recover exact receipt bytes; neither GPU nor historical endpoint runs."""
    manifest = json.loads((BUNDLE / "publication.json").read_text())
    files = {
        row["path"]: (BUNDLE / row["path"]).read_bytes() for row in manifest["files"]
    }
    validate_publication(manifest, files)
    assert manifest["decision"]["scope"] == "numerical"
    decoded = json.loads(lzma.decompress(files["raw-receipts.json.xz"]))
    restored = {}
    for name, receipt in decoded.items():
        raw = (
            base64.b64decode(receipt["data"])
            if receipt["encoding"] == "base64"
            else receipt["data"].encode()
        )
        assert len(raw) == receipt["bytes"]
        assert hashlib.sha256(raw).hexdigest() == receipt["sha256"]
        restored[name] = raw
    assert len(restored) == 91
    return load_publication_record(BUNDLE), restored


def test_canonical_publication_binds_current_sources_not_prototype_timing(
    retained: tuple,
) -> None:
    evidence, restored = retained
    assert (
        evidence["performance"]["status"]
        == evidence["stages"]["production"]["status"]
        == "not-run"
    )
    settings = evidence["settings"]
    assert (
        settings["no_samples_pooled"] and not settings["prototype_timing_transferred"]
    )
    assert settings["actual_w_bits"] == [64, 64, 64]
    assert (
        settings["original_typed_descriptors"]
        and settings["finite_audit_values_unchanged"]
    )
    assert (
        settings["original_canonical_energy_addresses"]
        and settings["six_entry_fallback_groups"]
    )
    assert (
        settings["pointer_table_bytes"] == 48
        and settings["aligned_arena_delta_bytes"] == 256
    )
    build = evidence["qualification"]["build"]
    assert (
        build["candidate_sha256"]
        == "b48c6b1a49884b6877692cae95546a754a5ae767935de0f3dfa92781a110eb0e"
    )
    parent = load_publication_record(
        ROOT / "benchmarks/results/df-triples-panel-traversal-20261011"
    )
    assert (
        build["baseline_sha256"] == parent["qualification"]["build"]["candidate_sha256"]
    )
    assert build["recompiled_targets"] == 2 and build["borrowed_objects_unchanged"]
    assert not build["latest_master_full_reference_qualified"]
    for name, expected in build["source_sha256"].items():
        assert hashlib.sha256(restored["remote/" + name]).hexdigest() == expected
        assert (
            hashlib.sha256(
                (ROOT / name.removeprefix("source/")).read_bytes()
            ).hexdigest()
            == expected
        )
    for name, row in build["generated_sha256"].items():
        assert (
            hashlib.sha256(restored["remote/generated/" + name]).hexdigest()
            == row["sha256"]
        )
    for name in (
        "generated_df_occupied_triples_cuda.cuh",
        "generated_df_occupied_triples_cuda.cu",
    ):
        assert build["generated_sha256"][name]["unchanged_from_qualified"]
        assert (
            build["generated_sha256"][name]["sha256"]
            == parent["qualification"]["build"]["unchanged_generated_sha256"][name]
        )
    assert restored["remote/job-status.txt"].strip() == b"exit:1"
    assert restored["remote/followup-status.txt"].strip() == b"exit:0"
    assert (
        b'has no member "release"'
        in restored["remote/objects/pair-compile-initial.log"]
    )
    assert (
        b"provider.release();"
        in restored["initial/source/tests/native/test_native_contraction_pair_cuda.cu"]
    )
    for name, expected in evidence["source_reconstruction"][
        "current_compiled_sources_exact"
    ].items():
        assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == expected

    host = evidence["qualification"]["host"]
    assert host["test_cases"] == 11 and host["exit_code"] == 0
    assert host["actual_grouping_domains"] == 256
    assert host["actual_optional_allocation_domains"] == 8
    assert (
        hashlib.sha256(restored["local/host-gates.log"]).hexdigest()
        == host["output_sha256"]
    )
    for name, expected in host["source_sha256"].items():
        assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == expected


def test_all_api_owner_sanitizer_and_failure_observations_are_retained(
    retained: tuple,
) -> None:
    evidence, restored = retained
    assert (
        b"50 oracle/layout cases and 7 unsupported cases passed"
        in restored["remote/pair-gates.log"]
    )
    observations = evidence["qualification"]["cuda_observations"]
    assert len(observations) == 3
    for index, suite in enumerate(observations):
        assert suite == json.loads(
            restored[f"remote/targeted-observations/{index}/targeted-gpu-summary.json"]
        )
        assert suite["independent_gates_pass"] and len(suite["records"]) == 4
        assert suite["records"][-1]["unpublished_sentinels_preserved"]
        assert suite["records"][2]["paired_groups"] == 0
        for row in suite["records"][:3]:
            assert row["counts"][5] == 3
            assert row["counts"][14] == row["expected_driver_calls"]
            assert row["energy_error"] <= 3e-12
            assert row["counts"][4] <= 96 << 20
            assert row["counts"][2] == row["counts"][3] + (96 << 20) + row["counts"][23]
    for kind in ("pair", "owner"):
        for tool in ("memcheck", "initcheck"):
            assert b"ERROR SUMMARY: 0 errors" in restored[f"remote/{kind}-{tool}.log"]
    assert all(row["passed"] for row in evidence["block_errors"].values())


def test_separate_endpoint_retains_original_gates_work_and_energy_bits(
    retained: tuple,
) -> None:
    evidence, _ = retained
    samples = evidence["retained_samples"]["separate_pair"]
    assert [sample["variant"] for sample in samples] == ["serpentine", "base"]
    for sample in samples:
        gates = sample["gates"]
        assert (
            gates["total_energy_error"] <= 1e-8
            and gates["triples_energy_error"] <= 1e-10
        )
        assert max(gates["physical_r1"], gates["physical_r2"]) <= 1e-10
    before, after = samples[1]["record"], samples[0]["record"]
    assert before["total_energy"] == after["total_energy"]
    assert before["triples_energy"] == after["triples_energy"]
    assert before["triples_fp64_gemms"] == 1556 and after["triples_fp64_gemms"] == 908
    assert before["triples_work"] == after["triples_work"] == 2326012282334
    for key, value in before.items():
        if key.startswith("ccsd_") and not key.endswith("_seconds"):
            assert value == after[key]
    summary = evidence["qualification"]["summary"]
    assert summary["wall_speedup"] > 1 and summary["triples_speedup"] > 1
    assert summary["ccsd_capacity_delta"] == 0


def test_dirty_patch_restores_exact_measured_scientific_sources(
    retained: tuple, tmp_path: Path
) -> None:
    """Replay the retained patch without needing historical Git objects or a GPU."""
    executable = shutil.which("git")
    if not executable:
        pytest.skip("Git patch application is required for source reconstruction")
    evidence, restored = retained
    reconstruction = evidence["source_reconstruction"]
    original_hashes = reconstruction["patch_base_source_sha256"]
    measured_hashes = reconstruction["current_compiled_sources_exact"]
    assert len(original_hashes) == 3
    source = tmp_path / "patch-base"
    source.mkdir()
    for relative, expected in original_hashes.items():
        original = restored["patch-base-source/" + relative]
        assert hashlib.sha256(original).hexdigest() == expected
        destination = source / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(original)
    patch = tmp_path / "production.patch"
    patch.write_bytes(
        gzip.decompress((BUNDLE / reconstruction["patch_file"]).read_bytes())
    )
    subprocess.run(
        [executable, "apply", "--check", "--whitespace=error-all", str(patch)],
        cwd=source,
        check=True,
        capture_output=True,
        text=True,
    )
    subprocess.run(
        [executable, "apply", "--whitespace=error-all", str(patch)],
        cwd=source,
        check=True,
        capture_output=True,
        text=True,
    )
    for relative in original_hashes:
        assert (
            hashlib.sha256((source / relative).read_bytes()).hexdigest()
            == measured_hashes[relative]
        )


def test_historical_report_compaction_preserves_every_original_byte() -> None:
    """Space for this evidence comes from storage only, not discarded old gates."""
    raw = gzip.decompress(
        (ROOT / "benchmarks/results/local-spaces-182/report.json.gz").read_bytes()
    )
    assert len(raw) == 276817
    assert (
        hashlib.sha256(raw).hexdigest()
        == "3caf68447d26bfa614c9bb43ff123d03f34259285b05955c9025a95d0e16ec7f"
    )
