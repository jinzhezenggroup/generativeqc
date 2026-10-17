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
import time
from contextlib import nullcontext
from pathlib import Path

try:
    from benchmarks._retention import raw_output_path
except ModuleNotFoundError:
    from _retention import raw_output_path

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
    args = parser.parse_args()
    identity = json.loads(args.identity.read_text())
    check_identity(identity)
    # These are the only environment values recorded; never serialize credentials.
    environment = {
        key: os.environ.get(key)
        for key in (
            "OMP_NUM_THREADS",
            "OPENBLAS_NUM_THREADS",
            "MKL_NUM_THREADS",
            "CUDA_VISIBLE_DEVICES",
            "LD_LIBRARY_PATH",
        )
    }
    if any(
        environment[key] != "1"
        for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")
    ):
        raise ValueError("one numerical-library thread required")
    environment_sha = hashlib.sha256(
        json.dumps(environment, sort_keys=True).encode()
    ).hexdigest()
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
    # Resolve the actual loader choice before timing, using this frozen executable.
    loader = subprocess.run(
        ["ldd", str(args.executable.resolve())],
        capture_output=True,
        text=True,
        check=True,
    )
    matches = [
        line.split("=>", 1)[1].strip().split(" (", 1)[0]
        for line in loader.stdout.splitlines()
        if "libgenerativeqc.so" in line and "=>" in line
    ]
    if len(matches) != 1 or Path(matches[0]).resolve() != args.library.resolve():
        raise ValueError("native loader does not select the frozen library")
    output = args.output.resolve() / "endpoint.json"
    command = [
        str(args.executable.resolve()),
        str(args.input.resolve()),
        str(output),
        *CONTROLS,
    ]
    receipt = {
        "schema": "generativeqc.cc-tile-campaign.v1",
        "identity": identity,
        "command": command,
        "diagnostic": args.diagnostic,
        "outcome": "INCOMPLETE",
        "clock": "perf_counter_ns, separate from Nsight ns clock",
        "forces_requested": False,
        "loader_library": str(args.library.resolve()),
    }
    receipt_path = args.output / "receipt.json"
    pending_receipt_path = args.output / "receipt.json.tmp"

    def write_receipt() -> None:
        # A failed write must not truncate the last durable receipt.
        pending_receipt_path.write_text(json.dumps(receipt, indent=2) + "\n")
        pending_receipt_path.replace(receipt_path)

    write_receipt()
    process = None
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
                process = subprocess.Popen(
                    command, stdout=log, stderr=subprocess.STDOUT
                )
                receipt["native_pid"] = process.pid
                write_receipt()
                rc = process.wait(timeout=1800)
                finished = True
            except BaseException as error:
                # Remember the original error before diagnostic scope teardown.
                failure = error
                raise
            finally:
                # Kill/reap only our child, also on interrupts and receipt errors.
                # Keep teardown inside the diagnostic complete-process interval.
                if process is not None and not finished:
                    for stage, action in (
                        ("child kill", process.kill),
                        ("child reap", lambda: process.wait(timeout=10)),
                    ):
                        try:
                            action()
                        except BaseException as error:  # noqa: BLE001
                            # Cleanup must continue without replacing the failure.
                            retain_error(stage, error)
        receipt["exit_code"] = rc
        receipt["process_wall_ns"] = time.perf_counter_ns() - started
        if rc != 0:
            receipt["outcome"] = "FAIL"
            return 2
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
            for name in {"endpoint.json", "oracle.json", "native.log"} - receipt[
                "artifacts"
            ].keys():
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
