"""NEW-run orchestration using the unchanged published worker and gate functions.

Source admission belongs to reproduce.py. Native build receipts and the current
Python/header source snapshot are recorded separately, without inventing a build.
"""

from __future__ import annotations

import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path


def run_campaign(plan: dict, original: dict, bundle: Path) -> int:
    source = Path(plan["source_root"])
    library = Path(plan["library_path"])
    output = Path(plan["output"])
    sys.path[:0] = [str(source / "python"), str(source)]
    from benchmarks._retention import raw_output_path

    raw_output_path(output, repository_root=source)
    raw_output_path(output, repository_root=bundle.parents[2])
    if output.exists():
        raise FileExistsError("use a new output directory")
    actual = original["source_snapshot"](source)
    if actual["source_hash"] != plan["verified_production_source_hash"]:
        raise ValueError("source changed after reviewed-map verification")
    if original["sha"](library) != plan["reproduction_library_sha256"]:
        raise ValueError("library changed after input verification")
    output.mkdir(parents=True)
    (output / "attempts").mkdir()
    fixture = json.loads((bundle / "inputs.json").read_text())
    dump = original["dump"]
    dump(output / "inputs.json", fixture)
    paths = ("src", "include", "python", "cmake", "CMakeLists.txt")
    (output / "source-tracked.patch").write_bytes(
        original["git"](source, "diff", "HEAD", "--binary", "--", *paths)
    )
    for name in actual["untracked_source_files"]:
        target = output / "source-untracked" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((source / name).read_bytes())
    environment = os.environ | {
        "GENERATIVEQC_LIBRARY": str(library),
        "PYTHONPATH": str(source / "python"),
        "OMP_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "NUMEXPR_NUM_THREADS": "1",
        "PYTHONHASHSEED": "0",
    }
    schedule = []
    for repeat in range(3):
        cases = (
            original["CASES"] if repeat % 2 == 0 else tuple(reversed(original["CASES"]))
        )
        modes = original["MODES"]
        for case in cases:
            schedule.extend(
                (repeat, case, mode) for mode in modes[repeat:] + modes[:repeat]
            )
    model_lines = Path("/proc/cpuinfo").read_text().splitlines()
    models = sorted(
        {
            line.split(":", 1)[1].strip()
            for line in model_lines
            if line.startswith("model name")
        }
    )
    receipt = plan["native_library_build_receipt"]
    provenance = {
        "scope": "NEW reproduction; historical measurements are not reused or requalified",
        "source": actual,
        "source_mapping_id": plan["source_mapping_id"],
        "source_mapping_kind": plan["source_mapping_kind"],
        "source_mapping_review_scope": plan["source_mapping_review_scope"],
        "native_library_build_receipt": receipt,
        "caller_declared_build_source": plan["caller_declared_build_source"],
        "build_source_revision": None
        if receipt is None
        else receipt.get("build_source_revision"),
        "build_source_tree": None
        if receipt is None
        else receipt.get("build_source_tree"),
        "production_sources_match_build": None
        if receipt is None
        else actual["source_hash"] == receipt.get("production_source_sha256"),
        "build_provenance_limit": "No native build is performed or inferred by this run. A reviewed compatible older native binary may execute newer Python/header source; unknown rebuilt binaries retain actual hashes without an invented build receipt.",
        "library_path": str(library),
        "library_sha256": plan["reproduction_library_sha256"],
        "harness_sha256": original["sha"](bundle / "run.py"),
        "orchestration_sha256": original["sha"](bundle / "new_campaign.py"),
        "fixture_sha256": original["sha"](bundle / "inputs.json"),
        "mapping_sha256": plan["mapping_sha256"],
        "publication_sha256": plan["publication_sha256"],
        "schedule": schedule,
        "time_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "device": {
            "kind": "cpu",
            "models": models,
            "platform": platform.platform(),
            "affinity": sorted(os.sched_getaffinity(0)),
            "affinity_isolated": False,
        },
        "runtime_environment": {
            k: v
            for k, v in environment.items()
            if k.startswith(("GENERATIVEQC_", "OMP_", "OPENBLAS_", "MKL_", "NUMEXPR_"))
        },
        "endpoint_boundary": plan["endpoint_boundary"],
        "checkpoint_boundary": plan["checkpoint_boundary"],
        "status": "running",
    }
    dump(output / "provenance.json", provenance)
    dump(
        output / "reproduction-context.json",
        {**plan, "completed": False, "status": "running"},
    )
    rows = []
    with (output / "records.jsonl").open("w") as stream:
        for index, (repeat, case, mode) in enumerate(schedule):
            record = output / "attempts" / f"{index:02d}-{case}-{mode}.json"
            started = time.perf_counter()
            command = [
                sys.executable,
                str(bundle / "run.py"),
                "--worker",
                "--source-root",
                str(source),
                "--case",
                case,
                "--mode",
                mode,
                "--repeat",
                str(repeat),
                "--record",
                str(record),
                "--launch-start-perf",
                repr(started),
            ]
            try:
                child = subprocess.run(
                    command,
                    env=environment,
                    capture_output=True,
                    text=True,
                    timeout=plan["timeout_seconds"],
                    check=False,
                )
                row = (
                    json.loads(record.read_text())
                    if record.exists()
                    else {
                        "case": case,
                        "mode": mode,
                        "repeat": repeat,
                        "status": "failed",
                        "error": "worker returned no record",
                    }
                )
                row.update(
                    returncode=child.returncode,
                    stdout=child.stdout,
                    stderr=child.stderr,
                )
            except subprocess.TimeoutExpired as error:
                row = {
                    "case": case,
                    "mode": mode,
                    "repeat": repeat,
                    "status": "failed",
                    "error": "process timeout",
                    "timeout_s": plan["timeout_seconds"],
                    "stdout": error.stdout.decode(errors="replace")
                    if isinstance(error.stdout, bytes)
                    else error.stdout,
                    "stderr": error.stderr.decode(errors="replace")
                    if isinstance(error.stderr, bytes)
                    else error.stderr,
                }
            row["process_wall_s"] = time.perf_counter() - started
            if "endpoint_total_s" in row:
                row["process_nonendpoint_s"] = (
                    row["process_wall_s"] - row["endpoint_total_s"]
                )
            rows.append(row)
            stream.write(json.dumps(row, allow_nan=False) + "\n")
            stream.flush()
            print(
                case,
                mode,
                repeat,
                row["status"],
                row.get("endpoint_total_s"),
                flush=True,
            )
    provenance.update(
        status="completed",
        source_unchanged=original["source_snapshot"](source)["source_hash"]
        == actual["source_hash"],
        library_unchanged=original["sha"](library)
        == plan["reproduction_library_sha256"],
    )
    dump(output / "provenance.json", provenance)
    numerical_passed = original["finish_evidence"](output, rows, provenance, fixture)
    code = int(
        not numerical_passed
        or any(not original["accepted"](row) for row in rows)
        or not provenance["source_unchanged"]
        or not provenance["library_unchanged"]
    )
    dump(
        output / "reproduction-context.json",
        {
            **plan,
            "completed": code == 0,
            "status": "completed",
            "runner_returncode": code,
            "source_and_library_match_verified_plan": provenance["source_unchanged"]
            and provenance["library_unchanged"],
            "observed_source": {
                k: actual[k] for k in ("revision", "tree", "dirty", "source_hash")
            },
            "observed_library_sha256": provenance["library_sha256"],
        },
    )
    return code
