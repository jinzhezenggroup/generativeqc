"""Run one fresh conventional CCSD(T) endpoint using the #3174 protocol."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import resource
import signal
import subprocess
import sys
import tempfile
import time
from contextlib import suppress
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from types import ModuleType

PROTOCOL_SHA = {
    "run.py": "0b9b8d6d1fb9d3ea13b875e6ffa29eec5209301523dd5cf881aeca14333d23fd",
    "fixtures.json": "559cb9a7c952654fadd79559d7f52973d6d5ab6faca6fcc40ba408b304f45695",
}
LIBRARY_SHA = "f6cb1d381f9fb97009f963cc51944084cbd15574745f3f358cf67337e40a6003"
Record = dict[str, Any]


def digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def protocol(directory: Path) -> ModuleType:
    for name, expected in PROTOCOL_SHA.items():
        if digest(directory / name) != expected:
            raise ValueError(f"#3174 protocol checksum differs for {name}")
    spec = importlib.util.spec_from_file_location(
        "ccsdt_protocol", directory / "run.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def write(path: Path, record: Record) -> None:
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(record, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def cgroup_memory() -> dict[str, str]:
    paths = (
        "memory.max",
        "memory.peak",
        "memory.events",
        "memory/memory.limit_in_bytes",
        "memory/memory.max_usage_in_bytes",
        "memory/memory.failcnt",
    )
    root = Path("/sys/fs/cgroup")
    return {
        key: (root / key).read_text().strip() for key in paths if (root / key).exists()
    }


def device(args: argparse.Namespace, field: str) -> str:
    return subprocess.check_output(
        [
            "nvidia-smi",
            "-i",
            str(args.gpu_index),
            f"--query-gpu={field}",
            "--format=csv,noheader,nounits",
        ],
        text=True,
        timeout=10,
    ).strip()


def hardware(args: argparse.Namespace, module: ModuleType) -> Record:
    cpu = module.cpu_model()
    if args.expected_cpu and args.expected_cpu not in cpu:
        raise RuntimeError(f"Expected CPU {args.expected_cpu}, found {cpu}")
    affinity = sorted(os.sched_getaffinity(0))
    if args.threads > len(affinity):
        raise RuntimeError(
            f"Requested {args.threads} threads with {len(affinity)} CPUs in affinity"
        )
    scratch = os.statvfs(args.scratch)
    info: Record = {
        "cpu_model": cpu,
        "affinity": affinity,
        "host_topology": json.loads(subprocess.check_output(["lscpu", "-J"])),
        "cgroup_memory": cgroup_memory(),
        "threads": args.threads,
        "process_id": os.getpid(),
        "qz_job_name": os.environ.get("GQC_QZ_JOB_NAME"),
        "qz_project": os.environ.get("GQC_QZ_PROJECT"),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "slurm_cpus_per_task": os.environ.get("SLURM_CPUS_PER_TASK"),
        "scratch_free_bytes": scratch.f_bavail * scratch.f_frsize,
        "python": sys.version.split()[0],
        "driver_sha256": digest(Path(__file__)),
        "protocol_sha256": PROTOCOL_SHA,
        "arm": args.arm,
        "trial": args.label,
        "wall_limit_seconds": args.timeout,
        "application_memory_mib": args.memory_mib,
    }
    if args.arm == "native-conv":
        info.update(module.native_provenance(args.source, args.source_archive))
        if info["library_sha256"] != LIBRARY_SHA:
            raise ValueError("Native library differs from the #3174 H100 binary")
        name, total, driver, uuid = device(
            args, "name,memory.total,driver_version,uuid"
        ).split(", ")
        if "H100" not in name:
            raise ValueError(f"Expected H100, found {name}")
        info["gpu"] = {
            "model": name,
            "total_mib": int(total),
            "driver": driver,
            "uuid": uuid,
        }
        info["correlation_budget_gib"] = args.correlation_budget_gib
        info["cublas_workspace_config"] = os.environ.get("CUBLAS_WORKSPACE_CONFIG")
    return info


def sample(args: argparse.Namespace, pid: int) -> dict[str, int]:
    result = {}
    status = Path(f"/proc/{pid}/status")
    with suppress(FileNotFoundError):
        for line in status.read_text().splitlines():
            if line.startswith("VmHWM:"):
                result["rss_high_water_kib"] = int(line.split()[1])
    if args.arm == "native-conv":
        with suppress(ValueError, subprocess.SubprocessError):
            result["gpu_device_used_mib"] = int(device(args, "memory.used"))
    return result


def acquire(args: argparse.Namespace, module: ModuleType) -> int:
    args.output.mkdir(parents=True)
    label_record = args.output / f"{args.label}.json"
    try:
        info = hardware(args, module)
    except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as exc:
        write(label_record, {"status": "preflight_error", "error": str(exc)})
        raise
    write(args.output / "hardware.json", info)
    env = {
        **os.environ,
        **dict.fromkeys(
            (
                "OMP_NUM_THREADS",
                "OPENBLAS_NUM_THREADS",
                "MKL_NUM_THREADS",
                "NUMEXPR_NUM_THREADS",
            ),
            str(args.threads),
        ),
    }
    if "gpu" in info:
        env["CUDA_VISIBLE_DEVICES"] = info["gpu"]["uuid"]
    peak: dict[str, int] = {}
    samples = 0
    started = time.monotonic()
    timed_out = False
    with (
        tempfile.TemporaryDirectory(
            prefix="gqc-3167-conv-", dir=args.scratch
        ) as scratch,
        (args.output / "worker.log").open("x") as log,
    ):
        process = subprocess.Popen(
            [
                sys.executable,
                "-B",
                "-u",
                str(Path(__file__).resolve()),
                *sys.argv[1:],
                "--worker",
            ],
            env={**env, "TMPDIR": scratch, "PYSCF_TMPDIR": scratch},
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        try:
            deadline = started + args.timeout
            while process.poll() is None:
                for key, value in sample(args, process.pid).items():
                    peak[key] = max(peak.get(key, 0), value)
                samples += 1
                remaining = deadline - time.monotonic()
                write(
                    args.output / "progress.json",
                    {
                        "elapsed_seconds": time.monotonic() - started,
                        "samples": samples,
                        "peak": peak,
                    },
                )
                if remaining <= 0:
                    timed_out = True
                    break
                with suppress(subprocess.TimeoutExpired):
                    process.wait(timeout=min(30, remaining))
        finally:
            if process.returncode is None:
                with suppress(ProcessLookupError):
                    os.killpg(process.pid, signal.SIGKILL)
                process.wait()
        if not label_record.exists():
            write(
                label_record,
                {
                    "status": "timeout" if timed_out else "worker_exit",
                    "returncode": process.returncode,
                    "timeout_seconds": args.timeout,
                },
            )
    write(
        args.output / "resource.json",
        {
            "duration_seconds": time.monotonic() - started,
            "sample_interval_seconds": 30,
            "samples": samples,
            "sampled_peak": peak,
            "cgroup_memory_after": cgroup_memory(),
        },
    )
    return int(json.loads(label_record.read_text())["status"] != "completed")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--label", required=True, choices=("C5.1", "C6.0", "C6.1", "C7.0", "C7.1")
    )
    parser.add_argument(
        "--arm", required=True, choices=("native-conv", "psi4-conv", "pyscf-conv")
    )
    parser.add_argument("--threads", type=int, required=True)
    parser.add_argument("--memory-mib", type=int, default=0)
    parser.add_argument("--correlation-budget-gib", type=int)
    parser.add_argument("--timeout", type=int, required=True)
    parser.add_argument("--scratch", type=Path, required=True)
    parser.add_argument("--expected-cpu")
    parser.add_argument("--source", type=Path)
    parser.add_argument("--source-archive", type=Path)
    parser.add_argument("--gpu-index", type=int, default=0)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.threads <= 0 or args.timeout <= 0:
        parser.error("--threads and --timeout must be positive.")
    if args.arm == "native-conv" and not (args.source and args.correlation_budget_gib):
        parser.error(
            "Native measurements require --source and --correlation-budget-gib."
        )
    if args.arm != "native-conv" and args.memory_mib <= 0:
        parser.error("CPU references require --memory-mib.")
    module = protocol(args.protocol_dir)
    if args.worker:
        module.worker(args, args.label)
        write(
            args.output / "worker-memory.json",
            {
                "process_rss_high_water_kib": resource.getrusage(
                    resource.RUSAGE_SELF
                ).ru_maxrss,
            },
        )
        return 0
    return acquire(args, module)


if __name__ == "__main__":
    raise SystemExit(main())
