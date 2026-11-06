"""Check retained synthetic CUDA reports without chemistry, native code or GPU."""

from __future__ import annotations

import hashlib
import itertools
import json
import lzma
import math
from pathlib import Path

HERE = Path(__file__).resolve().parent


def check(condition: object, message: str) -> None:
    if not condition:
        raise ValueError(message)


def replay() -> dict:
    cohort = json.loads((HERE / "cohort.json").read_text())
    check(cohort["decision"]["production_selection"] == "REJECTED", "Promotion differs")
    check(
        cohort["decision"]["endpoint_performance"] == "NOT_QUALIFIED", "Scope differs"
    )
    records = {}
    for item in cohort["files"]:
        path = Path(item["path"])
        check(len(path.parts) == 1, "Evidence member escapes its directory")
        data = (HERE / path).read_bytes()
        check(
            len(data) == item["bytes"]
            and hashlib.sha256(data).hexdigest() == item["sha256"],
            f"Evidence bytes differ: {path}",
        )
        records[str(path)] = data
    runner = cohort["runner"]
    check(0 < runner["decoded_bytes"] <= 65536, "Unbounded qualifier source")
    decoder = lzma.LZMADecompressor(memlimit=96 << 20)
    source = decoder.decompress(
        records[runner["stored_path"]], max_length=runner["decoded_bytes"] + 1
    )
    check(
        decoder.eof
        and not decoder.unused_data
        and len(source) == runner["decoded_bytes"]
        and hashlib.sha256(source).hexdigest() == runner["sha256"],
        "Frozen qualifier source differs",
    )
    qualified = json.loads(records["qualification.json"])
    failed = json.loads(records["failed-v1.json"])
    fixture = json.loads(records["fixture-receipt.json"])
    samples = json.loads(records["samples.json"])
    freeze = cohort["freeze"]
    check(
        qualified["status"] == "SYNTHETIC_CUDA_CORRECTNESS_PASS"
        and qualified["input_manifest"]["code_revision"] == cohort["source"]["revision"]
        and qualified["input_manifest"]["code_tree"] == cohort["source"]["tree"]
        and qualified["bundle_manifest_sha256"] == freeze["bundle_manifest_sha256"],
        "Successful attempt source/bundle differs",
    )
    check(
        failed["status"] == "FAILED"
        and "No such file or directory" in failed["failure"]
        and "compute-sanitizer" in failed["failure"]
        and not any(s["name"] == "compile-ptxas" for s in failed["stages"])
        and hashlib.sha256(records["failed-v1.json"]).hexdigest()
        == freeze["preserved_v1_failure"]["sha256"],
        "Original missing-tool failure differs",
    )
    original_files = {row["path"]: row for row in failed["input_manifest"]["files"]}
    repaired_files = {row["path"]: row for row in qualified["input_manifest"]["files"]}
    for name in (
        "shared_rys_xy_fixture.cu",
        "receipt.json",
        "headers-manifest.json",
        "phase-a.patch",
    ):
        check(original_files[name] == repaired_files[name], "Scientific input changed")
    check(
        hashlib.sha256(records["fixture-receipt.json"]).hexdigest()
        == repaired_files["receipt.json"]["sha256"]
        and fixture["source_sha256"] == freeze["source_sha256"]
        and fixture["fock_tolerances"] == {"atol": 2e-11, "rtol": 2e-10},
        "Original independent fixture/gates differ",
    )
    check(
        qualified["device"]["compute_capability"] == "9.0"
        and qualified["device"]["mig"] == "Disabled"
        and "H100" in qualified["device"]["name"]
        and all(
            s["status"] == "PASS" and s["returncode"] == 0 for s in qualified["stages"]
        ),
        "Device/stage qualification differs",
    )
    expected = {
        (f"{shell}_{geometry}_{kind}", graph, repeat)
        for shell, geometry, kind, graph, repeat in itertools.product(
            ("dppp", "pppp"),
            ("cartesian", "coincident", "reversed_pairs"),
            ("private", "shared"),
            (0, 1),
            (0, 1),
        )
    }
    maxima = {}
    for run in ("ordinary-graph", "memcheck", "racecheck", "synccheck"):
        record = samples["runs"][run]
        rows = record["rows"]
        check(
            record["all_synthetic_gates_pass"]
            and len(rows) == len(expected)
            and {(r["case"], r["graph"], r["replay"]) for r in rows} == expected,
            f"Incomplete ordinary/graph/full-matrix reports: {run}",
        )
        check(
            all(
                math.isfinite(r["max_abs"]) and 0 <= r["max_abs"] <= 2e-11 for r in rows
            ),
            f"Reported independent maximum error fails absolute gate: {run}",
        )
        if run != "ordinary-graph":
            expected_summary = (
                "========= RACECHECK SUMMARY: 0 hazards displayed (0 errors, 0 warnings)"
                if run == "racecheck"
                else "========= ERROR SUMMARY: 0 errors"
            )
            check(
                record["sanitizer_summary"] == expected_summary,
                "Sanitizer outcome differs",
            )
        maxima[run] = max(r["max_abs"] for r in rows)
    deltas = {}
    for shell, extra, threads in (("dppp", 576, 192), ("pppp", 432, 96)):
        suffix = f"{shell}_shell_class_fock_rhf_persistent_kernel"
        private = samples["runtime_resources"][f"private_{suffix}"]
        shared = samples["runtime_resources"][f"shared_{suffix}"]
        check(
            shared["shared_bytes"] - private["shared_bytes"] == extra
            and shared["registers"] == private["registers"] == 166
            and shared["local_bytes"] == private["local_bytes"]
            and shared["max_threads"] == private["max_threads"] == threads,
            "Runtime resource admission differs",
        )
        for kind in ("private", "shared"):
            ptx = samples["ptxas_resources"][f"{kind}_{suffix}"]
            check(
                ptx["spill_store_bytes"] == ptx["spill_load_bytes"] == 0,
                "PTX spill found",
            )
        for report in ("ptxas_resources", "cuobjdump_resources"):
            rows = samples[report]
            check(
                rows[f"shared_{suffix}"]["shared_bytes"]
                - rows[f"private_{suffix}"]["shared_bytes"]
                == extra,
                "Static resource delta differs",
            )
        deltas[shell] = extra
    return {
        "status": "PASS_RETAINED_SYNTHETIC_REPORTS",
        "source_revision": cohort["source"]["revision"],
        "device_uuid": qualified["device"]["uuid"],
        "full_matrix_gate_reports": len(expected) * 4,
        "maximum_absolute_errors": maxima,
        "extra_shared_bytes": deltas,
        "original_missing_tool_attempt": "FAILED_BEFORE_COMPILE_PRESERVED",
        "production_selection": "REJECTED",
        "endpoint_performance": "NOT_QUALIFIED",
        "full30_goal": "UNFULFILLED",
    }


if __name__ == "__main__":
    print(json.dumps(replay(), indent=2, allow_nan=False))
