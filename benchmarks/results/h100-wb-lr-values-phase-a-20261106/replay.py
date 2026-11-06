"""Audit retained standalone LR reports; no array reconstruction, oracle or GPU."""

from __future__ import annotations

import hashlib
import json
import lzma
import math
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
REVISION = "1fa3ba2f9d579e868103bfd5877c83cc5d225f5d"
TREE = "4cb25a54dd8cad13bb03947103e30a7bb84a7144"


def check(condition: object, message: str) -> None:
    if not condition:
        raise ValueError(message)


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def audit(cohort: dict[str, Any], records: dict[str, bytes]) -> dict[str, Any]:
    check(
        cohort["source"]["revision"] == REVISION
        and cohort["source"]["tree"] == TREE
        and cohort["source"]["clean"] is True,
        "measured source differs",
    )
    check(
        cohort["scope"]
        == {
            "standalone_fixture": "PASS",
            "molecular_source_replay": "NOT_QUALIFIED",
            "native_integration": "NOT_ADDED",
            "endpoint_performance": "NOT_QUALIFIED",
            "production_default": "NOT_PROMOTED",
            "all30": "OPEN",
        },
        "scope promotion refused",
    )
    reduction, qualified, fixture, review, binding, commit, selection = [
        json.loads(records[n])
        for n in (
            "reduction.json",
            "qualification.json",
            "fixture-receipt.json",
            "source-review.json",
            "source-binding.json",
            "commit-binding.json",
            "sanitizer-selection.json",
        )
    ]
    check(
        reduction["status"] == "STANDALONE_FIXTURE_REDUCTION_PASS"
        and qualified["status"] == "STANDALONE_WB_LR_CUDA_CORRECTNESS_PASS",
        "original qualification failed",
    )
    check(
        reduction["commit"] == binding["commit"] == commit["commit"] == REVISION
        and reduction["tree"] == binding["tree"] == commit["tree"] == TREE,
        "source/tree bindings differ",
    )
    check(
        binding["committed_blob_hashes_match_all_five_reviewed_files"] is True
        and commit["committed_file_hashes_match_review"] is True
        and len(review["files"]) == 5,
        "five measured blob bindings required",
    )
    check(
        binding["reduction_sha256"] == digest(records["reduction.json"])
        and binding["commit_binding_sha256"] == digest(records["commit-binding.json"])
        and binding["source_review_manifest_sha256"]
        == digest(records["source-review.json"]),
        "reviewed result provenance differs",
    )
    bound = reduction["binding"]
    check(
        bound["qualification_sha256"] == digest(records["qualification.json"])
        and bound["reference_receipt_sha256"] == digest(records["fixture-receipt.json"])
        and bound["actual_input_manifest_equals_local"] is True,
        "qualified input hashes differ",
    )
    check(
        bound["source_sha256"]
        == review["files"]["tests/native/test_materialized_lr_values.cu"]["sha256"]
        and qualified["object_sha256"] == bound["object_sha256"]
        and qualified["executable_sha256"] == bound["executable_sha256"]
        and qualified["bundle_manifest_sha256"]
        == bound["bundle_manifest_sha256"]
        == commit["bundle_manifest_sha256"],
        "TU/object/executable/bundle differs",
    )
    check(
        qualified["device"] == reduction["device"]
        and "H100" in qualified["device"]["name"]
        and qualified["device"]["compute_capability"] == "9.0"
        and qualified["device"]["mig"] == "Disabled"
        and qualified["device"]["uuid"].startswith("GPU-"),
        "actual H100 identity missing",
    )
    check(
        fixture["cases"] == 3120
        and fixture["compositions"] == 51
        and fixture["orders"] == [3, 4, 5]
        and fixture["pyscf"] == "2.14.0"
        and fixture["numpy"] == "2.2.6",
        "independent fixture population differs",
    )
    check(
        fixture["emitted_gate"] == {"atol": 2e-12, "rtol": 2e-10}
        and fixture["component_gate"] == {"atol": 3e-11, "rtol": 3e-11},
        "original numerical gates differ",
    )
    files = {r["path"]: r for r in qualified["input_manifest"]["files"]}
    check(
        files["fixtures-full.txt"]["sha256"]
        == fixture["files"]["fixtures.txt"]["sha256"]
        and files["sanitizer-selection.json"]["sha256"]
        == digest(records["sanitizer-selection.json"]),
        "full/sanitizer frozen inputs differ",
    )
    check(
        selection["count"] == 264
        and len(set(selection["selected_case_ids"])) == 264
        and selection["input_sha256"] == files["fixtures-sanitizer.txt"]["sha256"],
        "pre-data sanitizer slice differs",
    )
    metric_names = {
        "candidate_emitted_values",
        "candidate_complete_values",
        "scalar_emitted_values",
        "scalar_complete_values",
        "candidate_scalar_parity",
        "reference_raw_K_recomputed",
        "candidate_raw_K_recomputed",
        "candidate_raw_K_stdout_vs_recompute",
        "scalar_raw_K_recomputed",
        "scalar_raw_K_stdout_vs_recompute",
        "ordinary_fock_raw_K",
    }
    check(
        set(reduction["metrics"]) == metric_names, "complete metric inventory required"
    )
    check(
        bound["reference_json_sha256"] == fixture["files"]["fixtures.json"]["sha256"],
        "frozen reference array identity differs",
    )
    for name, row in reduction["metrics"].items():
        check(
            row["compared_elements"] == (558960 if "raw_K" in name else 126792),
            "metric coverage differs",
        )
        check(
            all(
                math.isfinite(row[k]) and row[k] >= 0
                for k in ("max_abs_error", "max_gate_fraction")
            )
            and row["max_gate_fraction"] <= 1,
            "reported numerical gate failed",
        )
        check(
            (row["atol"], row["rtol"])
            == ((2e-12, 2e-10) if "emitted_values" in name else (3e-11, 3e-11)),
            "metric bound relaxed",
        )
    work = reduction["ordinary_actual_work_one_execution_per_case"]
    check(
        work == [10128, 46704, 46704, 1654656, 126792]
        and reduction["zero_live_cases"]
        == reduction["nonempty_LR0_cases_exact_zero_publication"]
        == 624,
        "work/zero publication inventory differs",
    )
    check(
        set(reduction["reports"])
        == {"ordinary-graph", "memcheck", "racecheck", "synccheck"},
        "report populations missing",
    )
    for name, row in reduction["reports"].items():
        count = 3120 if name == "ordinary-graph" else 264
        check(
            (
                row["cases"],
                row["ordinary_executions"],
                row["graph_replay_executions"],
                row["reports"],
            )
            == (count, count, 2 * count, 3 * count),
            "ordinary/graph reports omitted",
        )
        check(
            math.isfinite(row["max_emitted_gate_fraction"])
            and 0 <= row["max_emitted_gate_fraction"] <= 1
            and math.isfinite(row["max_complete_gate_fraction"])
            and 0 <= row["max_complete_gate_fraction"] <= 1,
            "execution gate failed",
        )
    check(
        reduction["reports"]["ordinary-graph"]["summed_actual_work_all_executions"]
        == [3 * n for n in work],
        "ordinary/graph actual work differs",
    )
    for tool in ("memcheck", "racecheck", "synccheck"):
        summary = reduction["sanitizer"][tool]["summary"]
        check(
            summary["status"] == "PASS" and len(summary["summaries"]) == 1,
            "missing sanitizer terminal summary",
        )
        row = summary["summaries"][0]
        check(
            row["terminal"] is True
            and row["kind"] == ("racecheck" if tool == "racecheck" else "errors")
            and all(row.get(k, 0) == 0 for k in ("errors", "hazards", "warnings")),
            "sanitizer errors/hazards/warnings",
        )
    resources = reduction["resources"]
    check(
        len(resources) == 9
        and {(r["order"], r["symbol"]) for r in resources}
        == {(o, s) for o in (3, 4, 5) for s in ("candidate", "scalar", "fock")},
        "resource population missing",
    )
    for row in resources:
        check(
            row["registers"] == row["ptxas"]["registers"] == row["linked"]["REG"]
            and row["local_bytes"]
            == row["ptxas"]["stack_bytes"]
            == row["linked"]["STACK"],
            "resource views differ",
        )
        check(
            row["ptxas"]["spill_store_bytes"] == row["ptxas"]["spill_load_bytes"] == 0,
            "observed zero-spill result differs",
        )
        if row["symbol"] == "candidate":
            check(
                row["shared_bytes"] <= 16384 and row["linked"]["SHARED"] <= 16384,
                "candidate static shared exceeds16KiB",
            )
    expected_stages = {
        "cache-version",
        "nvcc-version",
        "gcc-version",
        "sanitizer-version",
        "gpu-identity",
        "cache-before",
        "compile-ptxas",
        "link",
        "resource-usage",
        "ordinary-graph",
        "memcheck",
        "racecheck",
        "synccheck",
        "cache-after",
    }
    check(
        {r["name"] for r in qualified["stages"]} == expected_stages,
        "finite stage inventory differs",
    )
    compiler = next(
        r["argv"] for r in qualified["stages"] if r["name"] == "compile-ptxas"
    )
    check(
        compiler[:2]
        == [
            qualified["tool_binaries"]["cache"]["path"],
            qualified["tool_binaries"]["nvcc"]["path"],
        ]
        and "arch=compute_90,code=sm_90" in compiler
        and not any(
            x in compiler
            for x in ("--use_fast_math", "-use_fast_math", "-ffast-math", "--ftz=true")
        ),
        "verified cache/strict sm90 compiler command differs",
    )
    check(
        len(qualified["stages"]) == 14
        and all(
            r["status"] == "PASS" and r["returncode"] == 0 for r in qualified["stages"]
        ),
        "finite stage missing/failed",
    )
    check(
        reduction["new_libcint_gpu_build_calls"] == 0
        and len(reduction["raw_files"]) == 31
        and digest(records["link-warning.txt"])
        == reduction["raw_files"]["link.stderr"]["sha256"],
        "retained raw/reduction boundary differs",
    )
    for name, row in cohort["compressed_sources"].items():
        check(0 < row["decoded_bytes"] < 65536, "unbounded retained producer source")
        decoder = lzma.LZMADecompressor(memlimit=96 << 20)
        data = decoder.decompress(records[name], max_length=row["decoded_bytes"] + 1)
        check(
            decoder.eof
            and not decoder.unused_data
            and len(data) == row["decoded_bytes"]
            and digest(data) == row["sha256"],
            "retained producer source differs",
        )
        if name == "reducer.py.xz":
            check(
                digest(data) == reduction["reducer_sha256"],
                "actual reducer identity differs",
            )
    return {
        "status": "RETAINED_STANDALONE_REPORTS_PASS",
        "measured_source": REVISION,
        "ordinary_cases": 3120,
        "graph_replays": 6240,
        "sanitizer_cases_each": 264,
        "resource_rows": 9,
        "native_integration": False,
        "endpoint_performance_claim": False,
        "raw_array_or_oracle_recomputed": False,
    }


def replay() -> dict[str, Any]:
    cohort = json.loads((HERE / "cohort.json").read_text())
    records = {}
    for row in cohort["files"]:
        relative = Path(row["path"])
        check(
            len(relative.parts) == 1
            and not relative.is_absolute()
            and relative.name not in records,
            "unsafe/duplicate evidence member",
        )
        check(0 <= row["bytes"] <= 131072, "unbounded compact evidence member")
        data = (HERE / relative).read_bytes()
        check(
            len(data) == row["bytes"] and digest(data) == row["sha256"],
            "retained evidence hash differs: " + str(relative),
        )
        records[str(relative)] = data
    return audit(cohort, records)


if __name__ == "__main__":
    print(json.dumps(replay(), sort_keys=True))
