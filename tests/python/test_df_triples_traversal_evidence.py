"""Source-bound traversal gates and endpoint work can be verified offline."""

import base64
import hashlib
import json
import lzma
from pathlib import Path

import numpy as np

from tools.generativeqc_validation.publication import validate_publication
from tools.generativeqc_validation.record import load_publication_record

ROOT = Path(__file__).resolve().parents[2]
BUNDLE = ROOT / "benchmarks/results/df-triples-panel-traversal-20261011"


def test_traversal_publication_binds_original_receipts_and_reviewed_sources() -> None:
    manifest = json.loads((BUNDLE / "publication.json").read_text())
    files = {
        entry["path"]: (BUNDLE / entry["path"]).read_bytes()
        for entry in manifest["files"]
    }
    validate_publication(manifest, files)
    evidence = load_publication_record(BUNDLE)
    assert manifest["decision"]["scope"] == "numerical"
    assert (
        evidence["performance"]["status"]
        == evidence["stages"]["production"]["status"]
        == "not-run"
    )
    settings = evidence["settings"]
    assert settings["scope"] == "cuda-df-triples-energy-only-fp64-three-panels"
    assert settings["same_numeric_storage"] and settings["constant_host_storage"]
    assert settings["original_canonical_energy_addresses"]
    assert settings["original_fp32_one_two_panel_and_response_traversal"]
    assert settings["no_samples_pooled"] and settings["baseline_fixture_matrix_reused"]
    receipts = json.loads(lzma.decompress(files["raw-receipts.json.xz"]))
    restored = {}
    for name, receipt in receipts.items():
        raw = (
            base64.b64decode(receipt["data"])
            if receipt["encoding"] == "base64"
            else receipt["data"].encode()
        )
        assert (
            len(raw) == receipt["bytes"]
            and hashlib.sha256(raw).hexdigest() == receipt["sha256"]
        )
        restored[name] = raw
    assert len(restored) == 65
    build = evidence["qualification"]["build"]
    assert build["recompiled_targets"] == 1 and build["borrowed_objects_unchanged"]
    assert build["no_public_or_result_abi_change"]
    identity = evidence["source_reconstruction"]["reviewed_source_identity"]
    assert not identity["gpu_repeat_required"]
    for relative, row in identity["files"].items():
        assert row["formatter_output_exactly_matches_reviewed"]
        assert (
            hashlib.sha256(restored["qualified-source/" + relative]).hexdigest()
            == row["compiled_sha256"]
        )
        assert (
            hashlib.sha256(restored["reviewed-source/" + relative]).hexdigest()
            == row["reviewed_sha256"]
        )
        assert (
            hashlib.sha256((ROOT / relative).read_bytes()).hexdigest()
            == row["reviewed_sha256"]
        )
    header = identity["files"]["src/cc/df_triples_traversal.hpp"]
    assert header["compiled_sha256"] == header["reviewed_sha256"]
    parent = load_publication_record(
        ROOT / "benchmarks/results/df-triples-distinct-moments-20261011"
    )
    assert (
        build["baseline_sha256"] == parent["qualification"]["build"]["candidate_sha256"]
    )
    assert (
        build["unchanged_generated_sha256"]
        == parent["qualification"]["build"]["generated_sha256"]
    )
    assert evidence["hashes"]["equation"] == parent["hashes"]["equation"]
    assert evidence["hashes"]["ir"] == parent["hashes"]["ir"]
    assert all(row["passed"] for row in evidence["block_errors"].values())
    for name in ("memcheck", "initcheck"):
        assert b"ERROR SUMMARY: 0 errors" in restored["remote/" + name + ".log"]
    annex = json.loads(lzma.decompress(files["targeted-receipts.json.xz"]))
    for receipt in annex.values():
        raw = receipt["data"].encode()
        assert (
            len(raw) == receipt["bytes"]
            and hashlib.sha256(raw).hexdigest() == receipt["sha256"]
        )
    for name in ("memcheck", "initcheck"):
        assert "ERROR SUMMARY: 0 errors" in annex["targeted-" + name + ".log"]["data"]


def test_actual_visitor_and_candidate_gates_preserve_every_tile_and_failure() -> None:
    qualification = load_publication_record(BUNDLE)["qualification"]
    cpu = qualification["cpu"]
    assert cpu["independent_cpu_gates_pass"] and len(cpu["coverage"]) == 32
    assert len(cpu["cases"]) == 6
    for case in cpu["cases"]:
        assert case["all_canonical_tile_bits_equal"]
        np.testing.assert_allclose(
            case["energy"], case["independent_oracle"], atol=3e-12, rtol=3e-12
        )
    suites = qualification["cuda"]
    assert [
        len(suites[name]["records"])
        for name in ("candidate", "baseline_reused", "memcheck", "initcheck")
    ] == [12, 11, 2, 2]
    for suite in suites.values():
        assert (
            suite["independent_gates_pass"]
            and suite["slurm_job"]
            and suite["visibility"]
        )
        for case in suite["records"]:
            if case["status"]:
                assert case["unpublished_sentinels_preserved"]
                assert case["overflow"] or case["refusal"]
                continue
            occupied, virtuals, auxiliary = case["shape"]
            tiles = occupied * (occupied + 1) * (occupied + 2) // 6
            counts = case["counts"]
            assert counts[1] == tiles and counts[7] == 2 * occupied**3
            assert counts[8] == tiles and counts[9] == tiles + 1
            assert counts[10] == tiles * virtuals**3
            assert counts[11] == counts[6] * auxiliary * virtuals**3 + occupied**3 * (
                virtuals**4 + occupied * virtuals**3
            )
            np.testing.assert_allclose(
                case["values"][0], case["oracle_energy"], atol=3e-12, rtol=3e-12
            )
    candidate = suites["candidate"]["records"]
    assert any(row["panels_requested"] == 2 and row["status"] == 0 for row in candidate)
    assert any(
        row["status"] == 0 and row["counts"][20] == row["counts"][22] == 1
        for row in candidate
    )
    assert any(row["refusal"] for row in candidate) and any(
        row["overflow"] for row in candidate
    )
    for record in qualification["targeted_new_domains"].values():
        assert record["independent_gates_pass"]
        generic, overflow = record["records"]
        assert generic["status"] == 0 and generic["counts"][5:7] == [3, 6]
        assert generic["counts"][20] == generic["counts"][22] == 1
        np.testing.assert_allclose(
            generic["values"][0], generic["independent_oracle"], atol=3e-12, rtol=3e-12
        )
        assert overflow["requested_panels"] == 3 and overflow["shape"][0] == 3
        assert overflow["status"] != 0 and overflow["unpublished_sentinels_preserved"]


def test_complete_pair_reduces_only_panel_work_with_fixed_capacity() -> None:
    evidence = load_publication_record(BUNDLE)
    pair = evidence["retained_samples"]["separate_pair"]
    assert [row["variant"] for row in pair] == ["serpentine", "base"]
    after, before = pair
    assert after["wall_seconds"] < before["wall_seconds"]
    for sample in pair:
        record = sample["record"]
        assert record["ccsd_evaluations"] == record["ccsd_iterations"] == 19
        assert (
            record["ccsd_device_bytes"] == 8583749632
            and record["numeric_capacity_bytes"] == 8945677650
        )
        for name, expected in sample["predicted_work"].items():
            assert record[name] == expected
        for name, limit in (
            ("total_energy_error", 1e-8),
            ("triples_energy_error", 1e-10),
            ("physical_r1", 1e-10),
            ("physical_r2", 1e-10),
        ):
            assert 0 <= sample["gates"][name] <= limit
    old, new = before["record"], after["record"]
    assert (
        old["total_energy"] == new["total_energy"]
        and old["triples_energy"] == new["triples_energy"]
    )
    assert old["triples_fp64_gemms"] == 1610 and new["triples_fp64_gemms"] == 1556
    assert old["triples_work"] == 2610452107406 and new["triples_work"] == 2326012282334
    for name in old:
        if name.startswith("ccsd_") and "seconds" not in name:
            assert old[name] == new[name], name
    summary = evidence["qualification"]["summary"]
    assert (
        summary["independent_scientific_gates_pass"]
        and summary["exact_work_predictions_pass"]
    )
    assert (
        summary["triples_work_delta"] == -284439825072
        and summary["ccsd_capacity_delta"] == 0
    )
