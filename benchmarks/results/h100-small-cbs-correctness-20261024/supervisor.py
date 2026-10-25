"""One root-released qz batch; no scheduling API and no automatic retry."""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from identity import verify
from native_worker import helper
from transport import (
    bounded,
    can_continue_scientific_failure,
    checked,
    phase_deadline,
    read,
    release,
    sha,
    write_new,
)

sys.dont_write_bytecode = True


def static(plan_path: Path, plan: dict, output: Path) -> None:
    with phase_deadline(plan["caps"]["metadata_phase_seconds"]):
        static_unbounded(plan_path, plan, output)


def static_unbounded(plan_path: Path, plan: dict, output: Path) -> None:
    if os.environ.get("PYTHONDONTWRITEBYTECODE") != "1" or not sys.dont_write_bytecode:
        raise ValueError(
            "outer PYTHONDONTWRITEBYTECODE=1 and -B are required before admission"
        )
    for name, expected in plan["task_files"].items():
        checked(plan_path.parent / name, expected)
    source = plan["science_source"]
    adapter = helper(plan)
    record = {
        "source": adapter.checked_source_tree(
            Path(source["root"]), Path(source["inventory"]), source["inventory_sha256"]
        ),
        "library_sha256": sha(
            checked(Path(source["library"]), source["library_sha256"])
        ),
        "environment": adapter.environment_observation(),
    }
    bundle = read(
        checked(
            plan_path.parent / plan["input_bundle_file"], plan["input_bundle_sha256"]
        )
    )
    additions_spec = plan["environment_additions"]
    additions = read(
        checked(plan_path.parent / additions_spec["file"], additions_spec["sha256"])
    )
    proof_path = checked(
        plan_path.parent / additions["derivation_proof_file"],
        additions["derivation_proof_sha256"],
    )
    derivation_proof = read(proof_path)
    record["whole_original_inputs"] = verify(
        Path(plan["public_readonly_runtime"]),
        read(
            checked(
                Path(plan["environment_inventory"]["path"]),
                plan["environment_inventory"]["sha256"],
            )
        ),
        bundle["external_dependencies"],
        additions=additions,
        base_manifest_sha256=plan["environment_inventory"]["sha256"],
        derivation_proof=derivation_proof,
        derivation_proof_sha256=sha(proof_path),
    )
    launcher = shutil.which("sccache")
    if launcher is None:
        record["compiler_cache"] = {"status": "UNAVAILABLE"}
    else:
        observation = subprocess.run(
            [launcher, "--version"],
            capture_output=True,
            text=True,
            timeout=15,
            check=False,
        )
        record["compiler_cache"] = {
            "path": launcher,
            "returncode": observation.returncode,
            "stdout": observation.stdout,
            "stderr": observation.stderr,
        }
    write_new(output, record)
    if (
        launcher is None
        or observation.returncode != 0
        or not observation.stdout.strip().startswith("sccache 0.16.")
    ):
        raise ValueError("verified sccache 0.16 launcher prerequisite missing")
    if (
        record["environment"]["executable"] != plan["expected_python_executable"]
        or record["environment"]["python"] != plan["expected_python_version"]
    ):
        raise ValueError("interpreter identity differs from frozen public environment")
    packages = record["environment"]["packages"]
    if any(packages.get(name) != version for name, version in plan["packages"].items()):
        raise ValueError("runtime package closure differs from pinned environment")
    for item in bundle["conventional"].items():
        relative, identity = item
        checked(Path(source["root"]) / relative, identity["sha256"])


def probe(
    plan_path: Path, plan: dict, folder: Path, output: Path, approval: Path | None
) -> None:
    command = [
        sys.executable,
        str(plan_path.parent / "device_probe.py"),
        "--plan",
        str(plan_path),
        "--output",
        str(output),
    ]
    if approval is not None:
        command += ["--release", str(approval.resolve())]
    outcome = bounded(
        command,
        folder,
        folder.parent,
        Path(plan["runtime_controls"]["GENERATIVEQC_JIT_SCCACHE_ROOT"]),
        plan["caps"]["device_probe_seconds"],
        plan["caps"]["all_output_max_bytes"],
        plan["caps"]["new_JIT_slots_max_bytes"],
        plan["caps"]["all_task_stop_bytes"],
        tmp_cache_cap=plan["caps"]["new_tmp_cache_stop_bytes"],
    )
    if outcome["status"] != "RETURNED":
        raise RuntimeError("bounded zero-science device observation failed")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--release", type=Path)
    parser.add_argument(
        "--wait-release",
        action="store_true",
        help="publish zero-endpoint preflight then wait at most 300 seconds",
    )
    parser.add_argument(
        "--preflight",
        type=Path,
        help="new directory for root inspection; no science or release marker",
    )
    args = parser.parse_args()
    plan_path = args.plan.resolve(strict=True)
    plan = read(plan_path)
    if args.preflight is not None:
        if args.release is not None and not args.wait_release:
            parser.error("preflight and release execution are distinct")
        args.preflight.mkdir(exist_ok=False)
        static(plan_path, plan, args.preflight / "static.json")
        # Root obtains this one actual observation before authoring allocation proof.
        probe(
            plan_path,
            plan,
            args.preflight / "device-process",
            args.preflight / "device.json",
            None,
        )
        boot = read(args.preflight / "device.json")
        write_new(
            args.preflight / "preflight.json",
            {
                "schema": "generativeqc.h100-environment-gate-preflight.v2",
                "scope": "issue2727 zero-endpoint allocation admission only",
                "job_name": plan["expected_job_name"],
                "device": boot,
                "endpoints_started": 0,
                "wait_cap_seconds": 300,
                "observed_utc": datetime.now(timezone.utc).isoformat(),
                "prepared_plan_sha256": sha(plan_path),
                "static_identity": {
                    "path": str((args.preflight / "static.json").resolve()),
                    "sha256": sha(args.preflight / "static.json"),
                },
                "observed_unix_seconds": time.time(),
                "expected_job_name": plan["expected_job_name"],
                "hostname": __import__("socket").gethostname(),
                "new_scientific_calls": 0,
                "release_status": "ABSENT",
            },
        )
        if not args.wait_release:
            return
        if args.release is None:
            parser.error("--wait-release requires an external root --release path")
        deadline = time.monotonic() + 300
        while not args.release.exists():
            if time.monotonic() >= deadline:
                write_new(
                    args.preflight / "gate-outcome.json",
                    {"status": "RELEASE_TIMEOUT", "calls": 0},
                )
                raise TimeoutError("root release window expired; no scientific call")
            time.sleep(0.5)
    if args.release is None:
        parser.error("execution requires an external root --release")
    plan, proof = release(plan_path, args.release.resolve(strict=True))
    approval = read(args.release)
    if time.time() - approval["released_unix_seconds"] >= 300:
        raise ValueError("batch start release window expired")
    if os.name != "posix":
        raise RuntimeError("qz batch requires real POSIX process groups")
    task = Path(plan["task_root"])
    out = Path(plan["output_root"])
    jit = Path(plan["runtime_controls"]["GENERATIVEQC_JIT_SCCACHE_ROOT"])
    if (
        out.parent != task
        or jit.parent != task
        or task.resolve() != task
        or task.is_symlink()
    ):
        raise ValueError("task output/JIT lease escapes task root")
    if shutil.disk_usage(task).free < plan["caps"]["required_free_storage_bytes"]:
        raise ValueError("insufficient reserved task storage")
    out.mkdir(exist_ok=False)
    jit.mkdir(exist_ok=False)
    write_new(
        out / "exclusive-batch-lease.json",
        {
            "release_sha256": sha(args.release),
            "prepared_plan_sha256": sha(plan_path),
            "pid": os.getpid(),
            "job": plan["expected_job_name"],
            "created_at": proof["platform_status"]["created_at"],
        },
    )
    status = {
        "status": "PREFLIGHT",
        "processes": [],
        "science_acceptance": "INCOMPLETE",
        "numeric_failures": [],
    }
    started = time.monotonic()
    try:
        for name, value in plan["runtime_controls"].items():
            if name in (
                "PYTHONDONTWRITEBYTECODE",
                "OMP_NUM_THREADS",
                "OPENBLAS_NUM_THREADS",
                "MKL_NUM_THREADS",
                "GENERATIVEQC_LIBRARY",
                "GENERATIVEQC_JIT_SCCACHE_ROOT",
            ):
                os.environ[name] = value
        if args.preflight is not None and args.wait_release:
            before = read(args.preflight / "preflight.json")["static_identity"]
            checked(Path(before["path"]), before["sha256"])
            write_new(out / "static-before-binding.json", before)
        else:
            static(plan_path, plan, out / "static-before.json")
        write_new(out / "device-before-binding.json", proof["runtime_preflight"])
        worker = plan_path.parent / "native_worker.py"
        for process in plan["processes"]:
            remaining = int(
                plan["caps"]["job_seconds"] - (time.monotonic() - started) - 60
            )
            if remaining <= 0:
                raise TimeoutError("batch hard wall cap reached before next process")
            record = bounded(
                [
                    sys.executable,
                    str(worker),
                    "--plan",
                    str(plan_path),
                    "--release",
                    str(args.release.resolve()),
                    "--process",
                    process["id"],
                ],
                out / (process["id"] + "-process"),
                out,
                jit,
                min(remaining, plan["caps"]["per_process_seconds"]),
                plan["caps"]["all_output_max_bytes"],
                plan["caps"]["new_JIT_slots_max_bytes"],
                plan["caps"]["all_task_stop_bytes"],
                tmp_cache_cap=plan["caps"]["new_tmp_cache_stop_bytes"],
            )
            status["processes"].append(record)
            if record["status"] != "RETURNED":
                outcome_path = out / process["id"] / "worker-outcome.json"
                if (
                    record["status"] == "FAILED"
                    and outcome_path.is_file()
                    and can_continue_scientific_failure(record, read(outcome_path))
                ):
                    status["numeric_failures"].append(process["id"])
                    continue
                raise RuntimeError(
                    "finite attempt failed; preserve all data and do not retry"
                )
        status["status"] = (
            "FAILED" if status["numeric_failures"] else "BATCH_NUMERIC_RETURNED"
        )
    except BaseException as error:
        status.update(status="FAILED", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        # Each closure failure is separate from returned originals and other postchecks.
        for name, action in (
            (
                "static-after",
                lambda: static(plan_path, plan, out / "static-after.json"),
            ),
            (
                "device-after",
                lambda: probe(
                    plan_path,
                    plan,
                    out / "device-after-process",
                    out / "device-after.json",
                    args.release,
                ),
            ),
        ):
            try:
                action()
            except BaseException as error:  # noqa: BLE001 -- retain every independent closure failure before reporting failure
                status[name] = f"{type(error).__name__}: {error}"
                status["status"] = "FAILED"
        status["elapsed_seconds"] = time.monotonic() - started
        write_new(out / "batch-outcome.json", status)
    if status["status"] == "FAILED":
        raise RuntimeError("batch postcheck failed; retained raw remains available")


if __name__ == "__main__":
    main()
