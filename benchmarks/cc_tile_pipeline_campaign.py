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


def check_science(record: dict, oracle: dict) -> dict:
    """Keep the existing independent energy/(T)/physical replay gates exact."""
    if (
        record.get("nbf") != oracle["nocc"] + oracle["nvir"]
        or record.get("naux") != oracle["naux"]
    ):
        raise ValueError("independent oracle dimensions mismatch")
    if record.get("forces_requested") != 0:
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
        ("physical_replay_r1", record["ccsd_replay_r1_max"], 1e-10),
        ("physical_replay_r2", record["ccsd_replay_r2_max"], 1e-10),
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
    if identity.get("oracle_input_sha256") != identity.get("input_sha256"):
        raise ValueError("same-Hamiltonian independent oracle input receipt required")
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
    args.output.mkdir(parents=True, exist_ok=False)
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
    receipt_path.write_text(json.dumps(receipt, indent=2) + "\n")
    if args.diagnostic:
        import nvtx  # Explicit diagnostic-only dependency; never a production import.

        scope = nvtx.annotate("cc1765/complete-process", domain="cc1765")
    else:
        scope = nullcontext()
    with (args.output / "native.log").open("wb") as log:
        started = time.perf_counter_ns()
        try:
            with scope:
                process = subprocess.Popen(
                    command, stdout=log, stderr=subprocess.STDOUT
                )
                receipt["native_pid"] = process.pid
                receipt_path.write_text(json.dumps(receipt, indent=2) + "\n")
                # Finite watchdog; a timed-out native process is killed and reaped,
                # never reclassified as successfully drained CUDA execution.
                try:
                    rc = process.wait(timeout=1800)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
                    receipt["outcome"] = "TIMEOUT"
                    raise
            receipt["exit_code"] = rc
            receipt["process_wall_ns"] = time.perf_counter_ns() - started
            if rc != 0:
                receipt["outcome"] = "FAIL"
                return 2
            record = json.loads(output.read_text())
            receipt["science"] = check_science(
                record, json.loads(args.oracle.read_text())
            )
            receipt["outcome"] = receipt["science"]["status"]
            receipt["work"] = {key: record[key] for key in WORK}
            return 0 if receipt["outcome"] == "PASS" else 2
        finally:
            receipt["binary_identity_unchanged"] = (
                checksum(args.library) == identity["library_sha256"]
                and checksum(args.executable) == identity["executable_sha256"]
            )
            if not receipt["binary_identity_unchanged"]:
                receipt["outcome"] = "FAIL"
            receipt["elapsed_wall_ns"] = time.perf_counter_ns() - started
            receipt["artifacts"] = {
                p.name: {"sha256": checksum(p), "size": p.stat().st_size}
                for p in args.output.iterdir()
                if p.is_file() and p != receipt_path
            }
            receipt_path.write_text(json.dumps(receipt, indent=2) + "\n")
            if not receipt["binary_identity_unchanged"]:
                raise ValueError("binary changed during endpoint execution")


if __name__ == "__main__":
    raise SystemExit(main())
