"""Fresh-process strict-FP64 energy runs through the existing native endpoint.

Run twice clean, then once under Nsight CUDA/NVTX in independent directories.
The diagnostic's parent NVTX scope includes native child startup and teardown.
This adapter neither supplies scientific state nor replaces the native owner.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import subprocess
import sys
import time
from contextlib import nullcontext
from pathlib import Path

try:
    from benchmarks._retention import raw_output_path
    from benchmarks.cc_tile_capture_diagnostics import validate_diagnostics_policy
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
        validate_observation,
        validate_observation_policy,
        validate_profile,
    )
except ModuleNotFoundError:
    from _retention import raw_output_path
    from cc_tile_capture_diagnostics import validate_diagnostics_policy
    from cc_tile_environment import (
        NativeSupervisor,
        atomic_json,
        canonical_hash,
        execution_environment,
        file_identity,
        observe_native,
        process_identity,
        validate_child,
        validate_execution,
        validate_observation,
        validate_observation_policy,
        validate_profile,
    )

# Positional contract of benchmarks/df_ccsdt_force_endpoint.cpp, args 3..31.
# Explicitly disable forces, lifecycle repetitions and all mixed precision.
CONTROLS = [
    "1",
    "1",
    "0",
    "1",
    "8",
    "8",
    "8",
    "0",
    "0",
    "2",
    "1",
    "30",
    "0",
    "0",
    "0",
    "auto",
    "1",
    "0",
    "auto",
    "auto",
    "0",
    "30",
    "1",
    "1",
    "1",
    "auto",
    "0",
    "0",
    "0",
]
WORK = (
    "ccsd_iterations",
    "ccsd_evaluations",
    "ccsd_q_batch_size",
    "ccsd_q_slices",
    "ccsd_q_tiles",
    "ccsd_gemm_calls",
    "ccsd_gemm_summands",
    "ccsd_contraction_terms",
    "ccsd_capacity",
    "ccsd_accumulation_calls",
    "ccsd_accumulation_bytes",
    "ccsd_q_operations",
    "reference_iterations",
)
CLOCK_SCHEMA = "generativeqc.supervised-child-wall.v1"


def checksum(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def check_identity(identity: dict) -> None:
    """Require all source, oracle and execution identities used by the campaign."""
    if not isinstance(identity, dict):
        raise TypeError("campaign identity must be an object")
    for key, digits in (
        ("source_commit", 40),
        *(
            (name, 64)
            for name in (
                "library_sha256",
                "executable_sha256",
                "input_sha256",
                "oracle_sha256",
                "environment_sha256",
            )
        ),
    ):
        value = identity.get(key)
        if (
            not isinstance(value, str)
            or re.fullmatch(f"[0-9a-f]{{{digits}}}", value) is None
        ):
            raise ValueError(f"missing/invalid campaign identity: {key}")
    for key in ("device_uuid", "run_id"):
        if not isinstance(identity.get(key), str) or not identity[key].strip():
            raise ValueError(f"missing campaign identity: {key}")
    if identity.get("oracle_input_sha256") != identity["input_sha256"]:
        raise ValueError("same-Hamiltonian independent oracle input receipt required")
    validate_profile(
        identity.get("environment_profile"), identity["environment_sha256"]
    )
    validate_observation_policy(
        identity.get("observation_policy"), identity.get("observation_policy_sha256")
    )
    warning_policy = identity.get("capture_diagnostics_policy")
    if warning_policy is not None:
        validate_diagnostics_policy(
            warning_policy,
            identity.get("capture_diagnostics_policy_sha256"),
            identity["observation_policy"]["session_control"]["profiler"],
        )
    elif identity.get("capture_diagnostics_policy_sha256") is not None:
        raise ValueError("capture diagnostics policy missing for hash")


def check_science(record: dict, oracle: dict) -> dict:
    """Keep the existing independent energy/(T)/physical replay gates exact."""
    if not isinstance(record, dict) or not isinstance(oracle, dict):
        raise TypeError("endpoint and independent oracle must be objects")
    for item, keys in ((record, ("nbf", "naux")), (oracle, ("nocc", "nvir", "naux"))):
        if any(type(item.get(key)) is not int or item[key] <= 0 for key in keys):
            raise ValueError("positive integer scientific dimensions required")
    if (
        record.get("nbf") != oracle["nocc"] + oracle["nvir"]
        or record.get("naux") != oracle["naux"]
    ):
        raise ValueError("independent oracle dimensions mismatch")
    if (
        type(record.get("forces_requested")) is not int
        or record["forces_requested"] != 0
    ):
        raise ValueError("energy-only contract violated")
    for key in ("ccsd_mixed_evaluations", "ccsd_fp32_gemm_calls", "triples_fp32_gemms"):
        if type(record.get(key)) is not int or record[key] != 0:
            raise ValueError(f"strict FP64 execution not established: {key}")
    for item, keys in (
        (record, ("total_energy", "triples_energy")),
        (oracle, ("total_energy", "triples_energy")),
    ):
        for key in keys:
            if type(item.get(key)) not in (float, int) or not math.isfinite(item[key]):
                raise ValueError(f"nonfinite/untyped independent energy: {key}")
    gates = {}
    for name, value, limit in (
        (
            "total_energy_error",
            abs(record["total_energy"] - oracle["total_energy"]),
            1e-8,
        ),
        (
            "triples_energy_error",
            abs(record["triples_energy"] - oracle["triples_energy"]),
            1e-10,
        ),
        ("physical_replay_r1", record.get("ccsd_replay_r1_max"), 1e-10),
        ("physical_replay_r2", record.get("ccsd_replay_r2_max"), 1e-10),
    ):
        if type(value) not in (float, int) or not math.isfinite(value) or value < 0:
            raise ValueError(f"missing/nonfinite scientific observation: {name}")
        gates[name] = {"value": value, "limit": limit, "passed": value <= limit}
    for key in WORK:
        value = record.get(key)
        if type(value) is not int or value < 0:
            raise ValueError(f"unavailable semantic work: {key}")
    return {
        "status": "PASS" if all(g["passed"] for g in gates.values()) else "FAIL",
        "gates": gates,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--executable", type=Path, required=True)
    parser.add_argument("--library", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--oracle", type=Path, required=True)
    parser.add_argument(
        "--identity",
        type=Path,
        required=True,
        help="frozen source/environment/device/input/oracle receipts",
    )
    parser.add_argument("--output", type=raw_output_path, required=True)
    parser.add_argument("--diagnostic", action="store_true")
    parser.add_argument(
        "--nsys-binding",
        type=Path,
        help="sealed owned outer-launch and actual profiler-parent session evidence",
    )
    args = parser.parse_args()
    identity = json.loads(args.identity.read_text())
    check_identity(identity)
    # These are the only environment values recorded; never serialize credentials.
    environment = execution_environment(os.environ)
    native_environment = validate_execution(environment, identity["environment_sha256"])
    binding_bytes = args.nsys_binding.read_bytes() if args.nsys_binding else None
    binding = json.loads(binding_bytes) if binding_bytes is not None else None
    binding_hash = canonical_hash(binding) if binding is not None else None
    parent_process = process_identity(os.getpid()) if binding is not None else None
    validate_observation(
        environment,
        identity["observation_policy"],
        identity["observation_policy_sha256"],
        args.diagnostic,
        str(args.output.resolve()),
        binding,
        binding_hash,
        parent_process,
    )
    if any(
        native_environment.get(key) != "1"
        for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")
    ):
        raise ValueError("one numerical-library thread required")
    environment_sha = canonical_hash(environment["profile"])
    if identity.get("environment_sha256") != environment_sha:
        raise ValueError("execution environment differs from frozen receipt")
    for key, path in (
        ("executable_sha256", args.executable),
        ("library_sha256", args.library),
        ("input_sha256", args.input),
        ("oracle_sha256", args.oracle),
    ):
        if identity.get(key) != checksum(path):
            raise ValueError(f"frozen {key} mismatch")
    oracle_bytes = args.oracle.read_bytes()
    if hashlib.sha256(oracle_bytes).hexdigest() != identity["oracle_sha256"]:
        raise ValueError("independent oracle changed during preflight")
    oracle = json.loads(oracle_bytes)
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "oracle.json").write_bytes(oracle_bytes)
    if binding_bytes is not None:
        (args.output / "nsys_binding.json").write_bytes(binding_bytes)
    warning_policy = identity.get("capture_diagnostics_policy")
    if warning_policy is not None:
        parent_source = Path(warning_policy["profiler"]["argv"][-1])
        parent_bytes = parent_source.read_bytes()
        if (
            hashlib.sha256(parent_bytes).hexdigest()
            != warning_policy["parent_source_sha256"]
        ):
            raise ValueError("predeclared NVTX-only parent source changed")
        (args.output / "nsys_parent.py").write_bytes(parent_bytes)
    # Resolve the actual loader choice before timing, using this frozen executable.
    loader = subprocess.run(
        ["ldd", str(args.executable.resolve())],
        capture_output=True,
        text=True,
        check=True,
        env=native_environment,
    )
    matches = [
        line.split("=>", 1)[1].strip().split(" (", 1)[0]
        for line in loader.stdout.splitlines()
        if "libgenerativeqc.so" in line and "=>" in line
    ]
    if len(matches) != 1 or Path(matches[0]).resolve() != args.library.resolve():
        raise ValueError("native loader does not select the frozen library")
    loader_libraries = {}
    for line in loader.stdout.splitlines():
        if "=>" not in line or not any(
            name in line
            for name in ("libgenerativeqc", "libcublas", "libcusolver", "libcudart")
        ):
            continue
        name, selected = line.split("=>", 1)
        path = Path(selected.strip().split(" (", 1)[0]).resolve()
        before = file_identity(path)
        digest = checksum(path)
        if file_identity(path) != before:
            raise ValueError("loader library changed during preflight hashing")
        loader_libraries[name.strip()] = {
            "path": str(path),
            "sha256": digest,
            "file_identity": before,
        }
    output = args.output.resolve() / "endpoint.json"
    command = [
        str(args.executable.resolve()),
        str(args.input.resolve()),
        str(output),
        *CONTROLS,
    ]
    receipt = {
        "schema": "generativeqc.cc-tile-campaign.v2",
        "identity": identity,
        "command": command,
        "diagnostic": args.diagnostic,
        "outcome": "INCOMPLETE",
        "clock": "supervised child completion; includes witness/exec startup, excludes parent attestation/finalization tail; separate from Nsight clock",
        "clock_schema": CLOCK_SCHEMA,
        "forces_requested": False,
        "loader_library": str(args.library.resolve()),
        "loader_libraries": loader_libraries,
        "execution_environment": environment,
        "execution_environment_sha256": canonical_hash(environment),
        "profiler_session_binding": binding,
        "profiler_session_binding_sha256": binding_hash,
        "parent_process": parent_process,
    }
    # Freeze the launcher dependencies in this immutable run directory. Its
    # Python imports resolve beside the copied script, not an ambient package.
    for name in ("cc_tile_child.py", "cc_tile_environment.py"):
        (args.output / name).write_bytes(Path(__file__).with_name(name).read_bytes())
    receipt["launcher_sha256"] = {
        name: checksum(args.output / name)
        for name in ("cc_tile_child.py", "cc_tile_environment.py")
    }
    plan = {
        "argv": command,
        "environment": environment,
        "profile_sha256": environment_sha,
    }
    plan_path = args.output / "execution_plan.json"
    child_path = args.output / "child_environment.json"
    atomic_json(plan_path, plan)
    launch_command = [
        sys.executable,
        str(args.output.resolve() / "cc_tile_child.py"),
        str(plan_path.resolve()),
        str(child_path.resolve()),
    ]
    receipt_path = args.output / "receipt.json"
    pending_receipt_path = args.output / "receipt.json.tmp"

    def write_receipt() -> None:
        # A failed write must not truncate the last durable receipt.
        pending_receipt_path.write_text(json.dumps(receipt, indent=2) + "\n")
        pending_receipt_path.replace(receipt_path)

    write_receipt()
    process = None
    supervisor = None
    attestation_started = False
    finished = False
    failure = None
    log = None
    started = time.perf_counter_ns()
    finalization_errors = []
    finalization_failure = None

    def retain_error(stage: str, error: BaseException) -> None:
        nonlocal finalization_failure
        if finalization_failure is None:
            finalization_failure = error
        finalization_errors.append(
            {"stage": stage, "type": type(error).__name__, "message": str(error)}
        )

    try:
        if args.diagnostic:
            import nvtx  # Explicit diagnostic-only dependency; never a production import.

            scope = nvtx.annotate("cc1765/complete-process", domain="cc1765")
        else:
            scope = nullcontext()
        log = (args.output / "native.log").open("wb")
        started = time.perf_counter_ns()
        with scope:
            try:
                deadline = time.monotonic() + 1800
                process = subprocess.Popen(
                    launch_command,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    env=native_environment,
                )
                supervisor = NativeSupervisor(process, deadline)
                supervisor.start()
                receipt["native_pid"] = process.pid
                if binding is not None:
                    receipt["native_process"] = process_identity(process.pid)
                write_receipt()
                attestation_started = True
                attestation_start = time.perf_counter_ns()
                receipt["native_bindings"] = observe_native(
                    process.pid, args.executable, loader_libraries
                )
                receipt["attestation_wall_ns"] = (
                    time.perf_counter_ns() - attestation_start
                )
                supervisor.join(max(0, deadline - time.monotonic()) + 10)
                if not supervisor.finished.is_set():
                    raise subprocess.TimeoutExpired(launch_command, 1800)
                if supervisor.failure is not None:
                    raise supervisor.failure
                rc = supervisor.returncode
                finished = True
            except BaseException as error:
                # Remember the original error before diagnostic scope teardown.
                if (
                    attestation_started
                    and supervisor is not None
                    and supervisor.failure is not None
                    and error is not supervisor.failure
                ):
                    retain_error("parent attestation", error)
                    failure = supervisor.failure
                    raise failure from error
                failure = error
                raise
            finally:
                # Kill/reap only our child, also on interrupts and receipt errors.
                # Keep teardown inside the diagnostic complete-process interval.
                if (
                    process is not None
                    and not finished
                    and not (supervisor is not None and supervisor.cleanup_attempted)
                ):
                    for stage, action in (
                        ("child kill", process.kill),
                        ("child reap", lambda: process.wait(timeout=10)),
                    ):
                        try:
                            action()
                        except BaseException as error:  # noqa: BLE001
                            # Cleanup must continue without replacing the failure.
                            retain_error(stage, error)
                if supervisor is not None and not supervisor.join(10):
                    retain_error(
                        "child supervisor",
                        TimeoutError("child wait did not finish after bounded cleanup"),
                    )
                if supervisor is not None:
                    for stage, error in supervisor.cleanup_errors:
                        retain_error(stage, error)
                    if (
                        supervisor.failure is not None
                        and failure is not supervisor.failure
                    ):
                        retain_error("child wait", supervisor.failure)
        receipt["exit_code"] = rc
        receipt["process_wall_ns"] = supervisor.ended_ns - started
        if rc != 0:
            receipt["outcome"] = "FAIL"
            return 2
        child = json.loads(child_path.read_text())
        validate_child(child, plan, process.pid, receipt.get("native_process"))
        receipt["child_environment_verified"] = True
        record = json.loads(output.read_text())
        receipt["science"] = check_science(record, oracle)
        receipt["outcome"] = receipt["science"]["status"]
        receipt["work"] = {key: record[key] for key in WORK}
        return 0 if receipt["outcome"] == "PASS" else 2
    except BaseException as error:
        if failure is not None and error is not failure:
            retain_error("diagnostic scope", error)
            raise failure from error
        failure = error
        raise
    finally:
        # Close before hashing; a close failure must not mask a wait failure.
        if log is not None:
            try:
                log.close()
            except BaseException as error:  # noqa: BLE001
                # Preserve interrupts too, while retaining the original failure.
                retain_error("native log close", error)
        if failure is not None:
            receipt["outcome"] = (
                "TIMEOUT" if isinstance(failure, subprocess.TimeoutExpired) else "FAIL"
            )
            receipt["failure"] = {
                "type": type(failure).__name__,
                "message": str(failure),
            }
        for name, expected in receipt["launcher_sha256"].items():
            try:
                if checksum(args.output / name) != expected:
                    raise ValueError("child launcher changed during endpoint")
            except BaseException as error:  # noqa: BLE001
                retain_error(f"launcher {name}", error)
        for name, item in loader_libraries.items():
            try:
                if (
                    checksum(Path(item["path"])) != item["sha256"]
                    or file_identity(Path(item["path"])) != item["file_identity"]
                ):
                    raise ValueError(
                        "prehashed provider binding changed during endpoint"
                    )
            except BaseException as error:  # noqa: BLE001
                retain_error(f"provider {name}", error)
        identity_matches = {}
        for key, path in (
            ("library_sha256", args.library),
            ("executable_sha256", args.executable),
            ("input_sha256", args.input),
            ("oracle_sha256", args.oracle),
        ):
            try:
                identity_matches[key] = checksum(path) == identity[key]
            except BaseException as error:  # noqa: BLE001
                # Unavailable evidence fails closed; still inventory other files.
                identity_matches[key] = False
                retain_error(key, error)
        receipt["identity_matches"] = identity_matches
        receipt["binary_identity_unchanged"] = all(
            identity_matches[key] for key in ("library_sha256", "executable_sha256")
        )
        receipt["input_identity_unchanged"] = all(
            identity_matches[key] for key in ("input_sha256", "oracle_sha256")
        )
        receipt["elapsed_wall_ns"] = time.perf_counter_ns() - started
        receipt["artifacts"] = {}
        try:
            for path in args.output.iterdir():
                if path in (receipt_path, pending_receipt_path):
                    continue
                try:
                    if path.is_symlink() or not path.is_file():
                        raise ValueError("campaign artifact must be a regular file")
                    receipt["artifacts"][path.name] = {
                        "sha256": checksum(path),
                        "size": path.stat().st_size,
                    }
                except BaseException as error:  # noqa: BLE001
                    # One unreadable artifact must not hide the other artifacts.
                    retain_error(f"artifact {path.name}", error)
        except BaseException as error:  # noqa: BLE001
            # Directory failure still permits an attempted terminal receipt.
            retain_error("artifact inventory", error)
        if receipt["outcome"] == "PASS":
            required = {
                "endpoint.json",
                "oracle.json",
                "native.log",
                "execution_plan.json",
                "child_environment.json",
                "cc_tile_child.py",
                "cc_tile_environment.py",
            }
            if binding is not None:
                required.add("nsys_binding.json")
            if warning_policy is not None:
                required.add("nsys_parent.py")
            for name in required - receipt["artifacts"].keys():
                retain_error(
                    f"artifact {name}",
                    ValueError("required campaign artifact unavailable"),
                )
        changed = not all(identity_matches.values())
        if (changed or finalization_errors) and receipt["outcome"] != "TIMEOUT":
            receipt["outcome"] = "FAIL"
        if finalization_errors:
            receipt["finalization_errors"] = finalization_errors
        try:
            write_receipt()
        except BaseException as error:
            original = failure if failure is not None else finalization_failure
            if original is None:
                raise
            original.add_note(
                f"receipt finalization failed: {type(error).__name__}: {error}"
            )
        if failure is None:
            if finalization_failure is not None:
                raise finalization_failure
            if changed:
                raise ValueError(
                    "binary/input/oracle changed during endpoint execution"
                )
        elif finalization_errors:
            failure.add_note(
                f"campaign finalization errors: {json.dumps(finalization_errors)}"
            )


if __name__ == "__main__":
    raise SystemExit(main())
