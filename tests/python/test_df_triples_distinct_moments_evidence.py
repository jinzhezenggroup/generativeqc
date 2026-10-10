"""Audit original gates and complete work offline, without repeating GPU runs."""

import base64
import hashlib
import json
import lzma
from pathlib import Path

import numpy as np
import pytest

from tools.generativeqc_validation.publication import validate_publication
from tools.generativeqc_validation.record import load_publication_record

ROOT = Path(__file__).resolve().parents[2]
BUNDLE = ROOT / "benchmarks/results/df-triples-distinct-moments-20261011"


def _assert_native_owner_layout(
    qualified: bytes, current: bytes, identity: dict
) -> None:
    """Bind the measured source to exactly the three reviewed space insertions."""
    layout = identity["native_owner_layout_delta"]
    assert layout["schema"] == "generativeqc.source-leading-space-delta.v1"
    assert layout["path"] == "src/cc/df_triples_cuda.cu"
    assert hashlib.sha256(qualified).hexdigest() == layout["qualified_sha256"]
    assert hashlib.sha256(current).hexdigest() == layout["current_sha256"]
    lines = qualified.splitlines(keepends=True)
    edits = layout["edits"]
    assert len(edits) == 3
    assert len({entry["line"] for entry in edits}) == 3
    for entry in edits:
        before, after = entry["before"].encode(), entry["after"].encode()
        assert after == b" " + before
        index = entry["line"] - 1
        assert 0 <= index < len(lines) and lines[index] == before
        lines[index] = after
    assert b"".join(lines) == current


def test_publication_retains_hash_bound_inputs_sources_and_unpooled_scope() -> None:
    manifest = json.loads((BUNDLE / "publication.json").read_text())
    files = {
        entry["path"]: (BUNDLE / entry["path"]).read_bytes()
        for entry in manifest["files"]
    }
    validate_publication(manifest, files)
    evidence = load_publication_record(BUNDLE)
    assert manifest["decision"]["scope"] == "numerical"
    assert evidence["performance"]["status"] == "not-run"
    assert evidence["stages"]["production"]["status"] == "not-run"
    settings = evidence["settings"]
    assert settings["scope"] == "cuda-df-energy-only-fp64"
    assert settings["actual_w_storage_compute_accumulation_bits"] == [64, 64, 64]
    assert settings["all_six_energy_contributions"] and settings["response_unchanged"]
    assert settings["fp32_original_path"] and settings["all_distinct_original_kernel"]
    assert (
        settings["six_cube_layout_unchanged"] and settings["no_numeric_scratch_growth"]
    )
    assert not settings["samples_pooled"]
    assert all(gate["passed"] for gate in evidence["block_errors"].values())
    receipts = json.loads(lzma.decompress(files["raw-receipts.json.xz"]))
    restored = {}
    for name, receipt in receipts.items():
        raw = (
            base64.b64decode(receipt["data"])
            if receipt["encoding"] == "base64"
            else receipt["data"].encode()
        )
        assert len(raw) == receipt["bytes"]
        assert hashlib.sha256(raw).hexdigest() == receipt["sha256"]
        restored[name] = raw
    assert len(restored) == 75
    build = evidence["qualification"]["build"]
    assert len(build["commands"]) == 2 and build["borrowed_objects_unchanged"]
    assert not build["latest_master_full_reference_qualified"]
    for name, expected in build["generated_sha256"].items():
        assert hashlib.sha256(restored["generated/" + name]).hexdigest() == expected
    assert "generated/generated_df_occupied_triples.hpp" in restored
    for name, expected in build["source_sha256"].items():
        assert (
            hashlib.sha256(
                restored["qualified-source/" + name.removeprefix("source/")]
            ).hexdigest()
            == expected
        )
    owner = ROOT / "src/cc/df_triples_cuda.cu"
    qualified_owner = restored["qualified-source/src/cc/df_triples_cuda.cu"]
    current_owner = owner.read_bytes()
    reconstruction = evidence["source_reconstruction"]
    assert reconstruction["production_objects_recompiled"] == 2
    identity = reconstruction["master_source_assessment"]
    assert not identity["native_owner_byte_identical_to_qualified"]
    _assert_native_owner_layout(qualified_owner, current_owner, identity)
    assert (
        identity["native_owner_layout_delta"]["qualified_sha256"]
        == build["source_sha256"]["source/src/cc/df_triples_cuda.cu"]
    )
    assert (
        identity["generated_byte_identical_to_qualified"] == build["generated_sha256"]
    )
    assert not identity["full_latest_master_HF_or_force_qualification"]
    assert not identity["gpu_repeat_required"]
    for name in ("memcheck.log", "initcheck.log", "memcheck-receipt-rerun.log"):
        assert b"ERROR SUMMARY: 0 errors" in restored["remote/" + name]


def test_original_cpu_gpu_gates_work_fallbacks_and_failures_are_preserved() -> None:
    qualification = load_publication_record(BUNDLE)["qualification"]
    cpu = qualification["cpu"]
    assert cpu["independent_cpu_gates_pass"] and len(cpu["cases"]) == 5
    assert cpu["old_kernel_byte_identical"] and cpu["response_source_byte_identical"]
    assert cpu["old_cpp_types_and_functions_byte_identical"]
    for case in cpu["cases"]:
        occupied = case["o"]
        assert case["original_moment_evaluations"] == occupied * (occupied + 1) * (
            occupied + 2
        )
        assert case["unique_moment_evaluations"] == occupied**3
        assert case["unwritten_duplicate_poison_checks"] > 0
        np.testing.assert_allclose(
            [case["original_energy"], case["alias_energy"]],
            case["independent_oracle"],
            atol=3e-12,
            rtol=3e-12,
        )
    suites = qualification["cuda"]
    assert [
        len(suites[name]["records"])
        for name in ("base", "distinct", "memcheck", "initcheck")
    ] == [7, 11, 2, 2]
    for name, suite in suites.items():
        assert (
            suite["independent_gates_pass"]
            and suite["slurm_job"]
            and suite["visibility"]
        )
        for case in suite["records"]:
            if case["status"]:
                assert case["unpublished_sentinels_preserved"]
                assert (case["overflow"] and "nonfinite" in case["error"]) or (
                    case["refusal"] and "budget" in case["error"]
                )
                continue
            occupied, virtuals, auxiliary = case["shape"]
            tiles = occupied * (occupied + 1) * (occupied + 2) // 6
            moments = 6 * tiles if name == "base" else occupied**3
            counts = case["counts"]
            assert counts[1] == tiles and counts[7] == 2 * moments
            assert counts[8] == tiles and counts[9] == tiles + 1
            assert counts[10] == tiles * virtuals**3
            work = counts[6] * auxiliary * virtuals**3 + moments * (
                virtuals**4 + occupied * virtuals**3
            )
            assert counts[11] == case["predicted_work"] == work
            assert counts[17:20] == [64, 64, 64]
            np.testing.assert_allclose(
                case["values"][0], case["oracle_energy"], atol=3e-12, rtol=3e-12
            )
    candidate = suites["distinct"]["records"]
    assert any(
        case["status"] == 0 and case["panels_requested"] == 1 and case["counts"][5] == 1
        for case in candidate
    )
    assert any(
        case["status"] == 0 and case["counts"][20] == case["counts"][22] == 1
        for case in candidate
    )
    assert any(case["overflow"] and case["shape"][0] == 1 for case in candidate)
    assert any(case["refusal"] for case in candidate)
    for name in ("memcheck", "initcheck"):
        assert all(case["shape"] == [3, 7, 5] for case in suites[name]["records"])


def test_complete_endpoint_gain_reduces_only_declared_triples_work() -> None:
    evidence = load_publication_record(BUNDLE)
    samples = evidence["retained_samples"]["separate_pair"]
    assert [sample["variant"] for sample in samples] == ["distinct", "base"]
    after, before = samples
    assert after["wall_seconds"] < before["wall_seconds"]
    for sample in samples:
        record = sample["record"]
        assert record["ccsd_iterations"] == record["ccsd_evaluations"] == 19
        assert record["ccsd_device_bytes"] == 8583749632
        assert record["numeric_capacity_bytes"] == 8945677650
        for name, expected in sample["predicted_work"].items():
            assert record[name] == expected
        for name, tolerance in (
            ("total_energy_error", 1e-8),
            ("triples_energy_error", 1e-10),
            ("physical_r1", 1e-10),
            ("physical_r2", 1e-10),
        ):
            assert 0 <= sample["gates"][name] <= tolerance
    old, new = before["record"], after["record"]
    assert old["total_energy"] == new["total_energy"]
    assert old["triples_energy"] == new["triples_energy"]
    for name in old:
        if name.startswith("ccsd_") and "seconds" not in name:
            assert old[name] == new[name], name
    assert old["triples_fp64_gemms"] == 2132 and new["triples_fp64_gemms"] == 1610
    assert old["triples_work"] == 3258407583236 and new["triples_work"] == 2610452107406
    assert new["triples_work"] - old["triples_work"] == -647955475830
    summary = evidence["qualification"]["summary"]
    assert (
        summary["independent_scientific_gates_pass"]
        and summary["exact_work_predictions_pass"]
    )
    assert summary["ccsd_capacity_delta"] == 0
    assert summary["whole_process_peak_rss_delta"] == 1212416


@pytest.mark.parametrize(
    "mutation", ("extra-indent", "token", "line-ending", "trailing-line")
)
def test_native_layout_binding_rejects_unrecorded_drift(mutation: str) -> None:
    evidence = load_publication_record(BUNDLE)
    identity = evidence["source_reconstruction"]["master_source_assessment"]
    receipts = json.loads(
        lzma.decompress((BUNDLE / "raw-receipts.json.xz").read_bytes())
    )
    receipt = receipts["qualified-source/src/cc/df_triples_cuda.cu"]
    assert receipt["encoding"] == "utf-8"
    qualified = receipt["data"].encode()
    current = (ROOT / "src/cc/df_triples_cuda.cu").read_bytes()
    _assert_native_owner_layout(qualified, current, identity)
    altered = {
        "extra-indent": b" " + current,
        "token": current.replace(b" == 64", b" == 32", 1),
        "line-ending": current.replace(b"\n", b"\r\n", 1),
        "trailing-line": current + b"\n",
    }[mutation]
    assert altered != current
    with pytest.raises(AssertionError):
        _assert_native_owner_layout(qualified, altered, identity)
