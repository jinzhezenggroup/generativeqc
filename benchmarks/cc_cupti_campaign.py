"""Fresh-process CUPTI experiment runner; preserves independent raw receipts."""

from __future__ import annotations

import hashlib
import json
import stat
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from benchmarks.cc_cupti_clock import CallerClock
from benchmarks.cc_tile_environment import (
    NativeSupervisor,
    atomic_json,
    canonical_hash,
    execution_environment,
    file_identity,
    observe_native,
    process_identity,
    validate_child,
    validate_execution,
)
from benchmarks.cc_tile_pipeline_campaign import CLOCK_SCHEMA, CONTROLS, check_science
from benchmarks.cc_trace_reader import open_trace


def digest(path: Path) -> str:
    """Read actual immutable input bytes for the frozen campaign identity."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


INPUTS = ("executable", "library", "input", "oracle", "backend", "cupti")


def frozen_inputs(spec: dict[str, Any]) -> dict[str, dict[str, int]]:
    """Bind both exact bytes and file identities before and after the child."""
    identities = {}
    for name in INPUTS:
        path = Path(spec[name])
        if not path.is_absolute() or not stat.S_ISREG(path.lstat().st_mode):
            raise ValueError(f"frozen {name} must be an absolute regular file")
        before = file_identity(path)
        if digest(path) != spec["sha256"][name] or file_identity(path) != before:
            raise ValueError(f"frozen {name} input changed")
        identities[name] = before
    return identities


def run(spec: dict[str, Any], output: Path, *, diagnostic: bool) -> dict[str, Any]:
    """Run one source-bound native child; actual science/capture statuses differ."""
    output.mkdir(parents=True, exist_ok=False)
    inputs_before = frozen_inputs(spec)
    environment = execution_environment(spec["environment"])
    native_environment = validate_execution(
        environment, canonical_hash(environment["profile"])
    )
    if any(
        key in native_environment
        for key in (
            "LD_PRELOAD",
            "CUDA_INJECTION64_PATH",
            "NVTX_INJECTION64_PATH",
            "NSYS_PROFILING_SESSION_ID",
        )
    ):
        raise ValueError("CUPTI recipe forbids injection and another profiling session")
    # The controller must have verified the live allocation before issuing this
    # immutable child request. Report consumption additionally revalidates raw
    # allocation/process/device evidence; this marker alone never proves it.
    allocation = json.loads(Path(spec["allocation_receipt"]).read_text())
    if canonical_hash(allocation) != spec["allocation_receipt_sha256"]:
        raise ValueError("allocation receipt changed")
    oracle = json.loads(Path(spec["oracle"]).read_text())
    loader = subprocess.run(
        ["ldd", spec["executable"]],
        env=native_environment,
        check=True,
        capture_output=True,
        text=True,
    )
    libraries: dict[str, dict[str, Any]] = {}
    for line in loader.stdout.splitlines():
        if "=>" not in line:
            continue
        name, selected = line.split("=>", 1)
        path = Path(selected.strip().split(" (", 1)[0]).resolve()
        before = file_identity(path)
        libraries[name.strip()] = {
            "path": str(path),
            "sha256": digest(path),
            "file_identity": before,
        }
        if file_identity(path) != before:
            raise ValueError("loader input changed during hash")
    expected_libraries = spec["mapped_libraries"]
    if set(libraries) != set(expected_libraries):
        raise ValueError(
            "native loader dependency inventory differs from frozen manifest"
        )
    for name, item in libraries.items():
        expected = expected_libraries[name]
        if item["path"] != expected["path"] or item["sha256"] != expected["sha256"]:
            raise ValueError("selected loader dependency differs from frozen identity")
        if "/stubs/" in item["path"]:
            raise ValueError("driver stub is not a runtime dependency")
    for path, expected in spec["driver_libraries"].items():
        driver = Path(path)
        before = file_identity(driver)
        if digest(driver) != expected["sha256"] or file_identity(driver) != before:
            raise ValueError("allocated node driver identity changed")
        libraries["actual_driver:" + path] = {
            "path": path,
            "sha256": expected["sha256"],
            "file_identity": before,
        }
    selected = [
        item["path"] for name, item in libraries.items() if "libgenerativeqc" in name
    ]
    if selected != [str(Path(spec["library"]).resolve())]:
        raise ValueError("native loader selected a different scientific library")
    if any(
        item["sha256"] != spec["sha256"]["library"]
        for name, item in libraries.items()
        if "libgenerativeqc" in name
    ):
        raise ValueError("selected scientific library differs from frozen digest")
    if diagnostic:
        for name in ("backend", "cupti"):
            libraries["observer_" + name] = {
                "path": str(Path(spec[name]).resolve()),
                "sha256": spec["sha256"][name],
                "file_identity": inputs_before[name],
            }
    if frozen_inputs(spec) != inputs_before:
        raise ValueError("frozen inputs replaced before native launch")
    command = [
        spec["executable"],
        spec["input"],
        str(output / "endpoint.json"),
        *CONTROLS,
    ]
    if diagnostic:
        command += ["--cc-cupti-v1", spec["backend"], str(output / "capture")]
    for name in ("cc_tile_child.py", "cc_tile_environment.py"):
        (output / name).write_bytes(Path(__file__).with_name(name).read_bytes())
    plan = {
        "argv": command,
        "environment": environment,
        "profile_sha256": canonical_hash(environment["profile"]),
    }
    atomic_json(output / "execution_plan.json", plan)
    child_receipt = output / "child_environment.json"
    launch = [
        sys.executable,
        "-B",
        str(output / "cc_tile_child.py"),
        str(output / "execution_plan.json"),
        str(child_receipt),
    ]
    clock = CallerClock(Path(spec["cupti"]), spec["sha256"]["cupti"])
    # Parent-only initialization is retained separately from the child interval.
    clock_initialization = clock.sample()
    parent_before = clock.sample()
    started = time.perf_counter_ns()
    receipt: dict[str, Any] = {
        "schema": "generativeqc.cc-cupti-run.v1",
        "diagnostic": diagnostic,
        "command": command,
        "science": "INCOMPLETE",
        "capture": "INCOMPLETE",
        "clock_schema": CLOCK_SCHEMA,
        "clock_initialization": clock_initialization,
        "parent_before": parent_before,
        "allocation_receipt_sha256": spec["allocation_receipt_sha256"],
        "inputs_before": inputs_before,
    }
    process = None
    supervisor = None
    try:
        with (output / "native.log").open("wb") as log:
            deadline = time.monotonic() + 1800
            process = subprocess.Popen(
                launch, stdout=log, stderr=subprocess.STDOUT, env=native_environment
            )
            supervisor = NativeSupervisor(process, deadline)
            supervisor.start()
            native = process_identity(process.pid)
            receipt["native_process"] = native
            atomic_json(output / "receipt.json", receipt)
            receipt["native_bindings"] = observe_native(
                process.pid, Path(spec["executable"]), libraries
            )
            supervisor.join(max(0, deadline - time.monotonic()) + 10)
            if not supervisor.finished.is_set() or supervisor.failure is not None:
                raise RuntimeError("native supervisor did not complete cleanly")
            receipt["native_exit"] = supervisor.returncode
            receipt["supervised_child_wall_ns"] = supervisor.ended_ns - started
        receipt["parent_after"] = clock.sample()
        if supervisor.returncode != 0:
            raise ValueError("native endpoint did not complete with exit zero")
        child = json.loads(child_receipt.read_text())
        validate_child(child, plan, process.pid, native)
        receipt["inputs_after"] = frozen_inputs(spec)
        if receipt["inputs_after"] != inputs_before:
            raise ValueError("frozen inputs replaced during native execution")
        record = json.loads((output / "endpoint.json").read_text())
        receipt["science"] = check_science(record, oracle)
        receipt["semantic_record"] = record
        if diagnostic:
            receipt["raw_artifacts"] = {
                kind: {
                    "sha256": trace.sha256,
                    "records": trace.count,
                    "metadata": list(trace.metadata),
                    "opened_identity": trace.identity,
                }
                for kind in ("activity", "source", "graph", "legacy")
                for trace in [open_trace(output / f"capture.{kind}.bin")]
            }
        receipt["capture"] = (
            "RAW_EXPORTED_NOT_QUALIFIED"
            if diagnostic and supervisor.returncode == 0
            else "DISABLED"
            if not diagnostic
            else "INCOMPLETE"
        )
        return receipt
    except BaseException as error:
        receipt["failure_type"] = type(error).__name__
        raise
    finally:
        if process is not None and process.poll() is None:
            process.kill()
            process.wait(timeout=10)
        if supervisor is not None:
            supervisor.join(10)
            receipt["cleanup_errors"] = [
                {"stage": stage, "type": type(error).__name__}
                for stage, error in supervisor.cleanup_errors
            ]
        try:
            receipt["inputs_after"] = frozen_inputs(spec)
            receipt["frozen_inputs_unchanged"] = (
                receipt["inputs_after"] == inputs_before
            )
        except (OSError, ValueError) as error:
            receipt["input_verification_error"] = type(error).__name__
            receipt["frozen_inputs_unchanged"] = False
        if not receipt["frozen_inputs_unchanged"]:
            receipt["science"] = "INCOMPLETE_INPUT_CHANGED"
            receipt["capture"] = "INCOMPLETE"
        atomic_json(output / "receipt.json", receipt)
