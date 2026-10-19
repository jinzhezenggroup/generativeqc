"""Scoped, reproducible qualification of the opt-in MD-J reference engine.

Host-only (no native library or CUDA execution):
  python benchmarks/md_j_engine_acceptance.py --output .artifacts/md-j-host.json

Real integration, from a prepared CUDA build with PySCF and CuPy installed:
  srun --gres=gpu:1 --ntasks=1 --cpus-per-task=4 --time=00:30:00 \\
    python benchmarks/md_j_engine_acceptance.py --gpu --timeout 1500 \\
    --output .artifacts/md-j-gpu.json

Supply GENERATIVEQC_LIBRARY, CUDA_PATH, PYTHONPATH=python:. and a verified
compiler-cache executable on PATH. Optional --upstream-root selects a clean
checkout of the exact pinned GPU4PySCF commit, with its extension built there.
Raw stdout/stderr, JUnit properties, timings and gate status are retained next
to the output. Passing host tests or collecting/skipping CUDA tests is never
reported as CUDA integration, complete performance acceptance, or permission
to change the default engine. This runner never creates a Slurm allocation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import time
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

try:
    from benchmarks._retention import raw_output_path
except ModuleNotFoundError:
    from _retention import raw_output_path

ROOT = Path(__file__).resolve().parents[1]
UPSTREAM_COMMIT = "9af748a71ca8f4881c604332c2f0cc1632e033ac"
REQUIRED_GATES = ("host", "raw_j_cuda", "endpoints_cuda", "cuda_graph_replay")


def junit_gate(path: Path, *, returncode: int) -> dict[str, Any]:
    """A successful pytest process containing skips is not a passed gate."""
    result: dict[str, Any] = {
        "status": "failed",
        "returncode": returncode,
        "junit": str(path),
        "passed": 0,
        "skipped": 0,
        "failed": 0,
        "tests": [],
    }
    try:
        root = ET.parse(path).getroot()  # noqa: S314 - own subprocess-produced JUnit only
    except (OSError, ET.ParseError) as error:
        result["reason"] = f"missing or invalid test evidence: {error}"
        return result
    for case in root.iter("testcase"):
        failed = case.find("failure") is not None or case.find("error") is not None
        skipped = case.find("skipped")
        status = "failed" if failed else "skipped" if skipped is not None else "passed"
        result[status] += 1
        result["tests"].append(
            {
                "name": case.get("name"),
                "classname": case.get("classname"),
                "status": status,
                "seconds": float(case.get("time", "0")),
                "skip_reason": skipped.get("message", "")
                if skipped is not None
                else None,
                "properties": [
                    {"name": value.get("name"), "value": value.get("value")}
                    for value in case.findall("./properties/property")
                ],
            }
        )
    if returncode or result["failed"]:
        result["status"] = "failed"
    elif result["skipped"]:
        result["status"] = "skipped"
    elif result["passed"]:
        result["status"] = "passed"
    else:
        result["status"] = "not-run"
        result["reason"] = "no test cases executed"
    return result


def summarize(gates: dict[str, Any]) -> dict[str, Any]:
    """Separate science qualification from incomplete/default-promotion gates."""
    return {
        "schema": "md-j-engine-acceptance-v1",
        "gates": gates,
        "correctness_qualified": all(
            gates.get(name, {}).get("status") == "passed" for name in REQUIRED_GATES
        )
        and not any(gate.get("status") == "failed" for gate in gates.values()),
        "default_promotion_qualified": False,
        "default_promotion_reason": "This scoped correctness run does not independently qualify full-size cold performance or change the production default",
        "gpu_evidence": "actual executed resident tests required; host scalar tests and skipped tests do not qualify CUDA",
        "timing_policy": "JUnit properties retain per-endpoint and resident diagnostic timings; no speedup is inferred from instrumented correctness runs",
    }


def verify_upstream(root: Path) -> str:
    """Pin both imported source and a clean tracked checkout, never just a label."""

    def git(*arguments: str) -> str:
        return subprocess.run(
            ["git", "-C", str(root), *arguments],
            check=True,
            capture_output=True,
            text=True,
            timeout=15,
        ).stdout.strip()

    commit = git("rev-parse", "HEAD")
    if commit != UPSTREAM_COMMIT:
        raise ValueError(
            f"GPU4PySCF must be pinned to {UPSTREAM_COMMIT}; found {commit}"
        )
    if git("status", "--porcelain", "--untracked-files=no"):
        raise ValueError("GPU4PySCF checkout contains tracked modifications")
    if not (root / "gpu4pyscf/scf/j_engine.py").is_file():
        raise ValueError("pinned GPU4PySCF MD J entrypoint is absent")
    return commit


def _finite_slurm() -> dict[str, str]:
    job = os.environ.get("SLURM_JOB_ID")
    if not job:
        raise RuntimeError("--gpu requires an existing finite Slurm allocation")
    end = os.environ.get("SLURM_JOB_END_TIME")
    if end and end.isdigit() and time.time() < int(end) < time.time() + 31 * 86400:
        return {"job_id": job, "end_time": end}
    command = shutil.which("scontrol")
    if command is None:
        raise RuntimeError(
            "cannot verify finite Slurm limit: SLURM_JOB_END_TIME and scontrol unavailable"
        )
    details = subprocess.run(
        [command, "show", "job", job, "--oneliner"],
        check=True,
        capture_output=True,
        text=True,
        timeout=15,
    ).stdout
    match = re.search(r"\bTimeLimit=([^\s]+)", details)
    if not match or match.group(1) in ("UNLIMITED", "NOT_SET", "Partition_Limit", "0"):
        raise RuntimeError("the Slurm job must have an explicit finite time limit")
    return {"job_id": job, "time_limit": match.group(1)}


def _cuda_preflight() -> dict[str, Any]:
    record: dict[str, Any] = _finite_slurm()
    for name in ("GENERATIVEQC_LIBRARY", "CUDA_PATH"):
        value = os.environ.get(name)
        if not value or not Path(value).exists():
            raise RuntimeError(f"{name} must name an existing CUDA build dependency")
    # The complete acceptance suite's NativeCxx host fixtures require ccache.
    # The resident bridge uses the repository's sccache-first/ccache resolver.
    cache = shutil.which("ccache")
    if cache is None:
        raise RuntimeError("the resident test bridge requires verified ccache on PATH")
    record["compiler_cache"] = subprocess.run(
        [cache, "--version"], check=True, capture_output=True, text=True, timeout=15
    ).stdout.splitlines()[0]
    library = Path(os.environ["GENERATIVEQC_LIBRARY"])
    digest = hashlib.sha256()
    with library.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    record.update(
        library=str(library.resolve()),
        library_sha256=digest.hexdigest(),
        cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES", ""),
    )
    return record


def _run_gate(
    name: str,
    selection: list[str],
    output: Path,
    environment: dict[str, str],
    timeout: int,
) -> dict[str, Any]:
    stem = output.with_name(f"{output.stem}-{name}")
    xml = Path(str(stem) + ".xml")
    xml.unlink(missing_ok=True)
    command = [
        sys.executable,
        "-m",
        "pytest",
        "-q",
        "-o",
        "junit_family=legacy",
        "--junitxml",
        str(xml),
        *selection,
    ]
    started = time.perf_counter()
    timeout_reason = None
    try:
        run = subprocess.run(
            command,
            cwd=ROOT,
            env=environment,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
        stdout, stderr, returncode = run.stdout, run.stderr, run.returncode
    except subprocess.TimeoutExpired as error:
        stdout = error.stdout or ""
        stderr = error.stderr or ""
        stdout = (
            stdout.decode(errors="replace") if isinstance(stdout, bytes) else stdout
        )
        stderr = (
            stderr.decode(errors="replace") if isinstance(stderr, bytes) else stderr
        )
        returncode, timeout_reason = 124, f"gate exceeded explicit {timeout}s timeout"
    Path(str(stem) + ".stdout.txt").write_text(stdout)
    Path(str(stem) + ".stderr.txt").write_text(stderr)
    result = junit_gate(xml, returncode=returncode)
    result.update(
        command=command,
        wall_seconds=time.perf_counter() - started,
        stdout=str(Path(str(stem) + ".stdout.txt")),
        stderr=str(Path(str(stem) + ".stderr.txt")),
    )
    if timeout_reason:
        result["reason"] = timeout_reason
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--output", type=raw_output_path, required=True)
    parser.add_argument(
        "--gpu",
        action="store_true",
        help="execute resident and complete endpoint CUDA gates in this Slurm allocation",
    )
    parser.add_argument(
        "--upstream-root",
        type=Path,
        help=f"optional clean built GPU4PySCF checkout at {UPSTREAM_COMMIT}",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=1500,
        help="finite timeout in seconds for each gate subprocess",
    )
    args = parser.parse_args()
    if args.timeout <= 0 or args.timeout > 86400:
        parser.error("--timeout must be between 1 and 86400 seconds")
    if args.upstream_root and not args.gpu:
        parser.error("--upstream-root requires --gpu")
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    environment = dict(os.environ)
    # Qualify the explicitly supplied binary, not a compatible local profile
    # selected from an unrelated on-disk tuning cache.
    environment["GENERATIVEQC_PROFILE"] = "off"
    environment.setdefault("OMP_NUM_THREADS", "1")
    environment.setdefault("OPENBLAS_NUM_THREADS", "1")
    environment["PYTHONPATH"] = os.pathsep.join(
        [str(ROOT / "python"), str(ROOT), environment.get("PYTHONPATH", "")]
    )
    environment.pop("GENERATIVEQC_TEST_MD_J_ENGINE", None)
    gates: dict[str, Any] = {
        "raw_j_cuda": {"status": "not-run", "reason": "--gpu was not requested"},
        "endpoints_cuda": {"status": "not-run", "reason": "--gpu was not requested"},
        "upstream_gpu4pyscf": {
            "status": "not-run",
            "reason": "optional pinned upstream was not supplied",
        },
        "cuda_graph_replay": {
            "status": "not-run",
            "reason": "--gpu was not requested",
        },
        "complete_cold_performance": {
            "status": "not-run",
            "reason": "run the separate bounded benchmarks/md_j_engine_benchmark.py driver for full-size cold/warm J endpoint evidence",
        },
    }
    provenance: dict[str, Any] = {
        "python": sys.version,
        "source_root": str(ROOT),
        "source_revision": os.environ.get(
            "GENERATIVEQC_BENCHMARK_SOURCE", "unrecorded"
        ),
        "upstream_pin": UPSTREAM_COMMIT,
        "local_profile_policy": "off: use the explicitly supplied native library",
    }

    evidence_sources = (
        "src/scf/cuda/direct_jk.cpp",
        "src/scf/cuda/direct_md_j.cu",
        "src/scf/cuda/direct_md_j.hpp",
        "src/scf/cuda/md_j_engine.cuh",
        "src/scf/cuda/md_j_engine_contract.hpp",
        "src/scf/md_transform_layout.hpp",
        "python/generativeqc_compiler/integral/md_j_engine_cuda.py",
        "tests/python/test_md_j_engine_cuda.py",
        "tests/python/test_md_j_engine_math.py",
        "tests/python/test_md_j_engine_contract.py",
        "tests/python/test_md_j_engine_runtime.py",
        "tests/python/test_md_j_resources.py",
        "tests/python/test_md_j_engine_reference.py",
        "tests/python/_md_j_engine_oracle.py",
        "tests/python/_md_j_engine_pipeline_probe.py",
        "tests/native/md_j_resident_probe.cpp",
        "tests/python/md_j_resident_probe.py",
        "tests/python/test_md_j_resident_probe.py",
        "benchmarks/md_j_engine_benchmark.py",
        "tests/python/test_md_j_engine_benchmark.py",
        "benchmarks/_md_j_engine_endpoints.py",
        "tests/python/test_md_j_engine_endpoints.py",
        "benchmarks/md_j_engine_acceptance.py",
    )
    provenance["source_sha256"] = {
        name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
        for name in evidence_sources
        if (ROOT / name).is_file()
    }

    def save() -> None:
        current = {
            name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
            for name in evidence_sources
            if (ROOT / name).is_file()
        }
        changed = sorted(
            name
            for name in set(current) | set(provenance["source_sha256"])
            if current.get(name) != provenance["source_sha256"].get(name)
        )
        provenance["source_sha256_after"] = current
        gates["source_provenance"] = {
            "status": "failed" if changed else "passed",
            "changed_during_run": changed,
        }
        output.write_text(
            json.dumps(
                {**summarize(gates), "provenance": provenance},
                indent=2,
                allow_nan=False,
            )
            + "\n"
        )

    host_tests = [
        "tests/python/test_md_j_engine_math.py",
        "tests/python/test_md_j_engine_contract.py",
        "tests/python/test_md_j_engine_reference.py",
        "tests/python/test_md_j_engine_runtime.py",
        "tests/python/test_md_j_resources.py",
        "tests/python/test_md_j_resident_probe.py",
        "tests/python/test_md_j_engine_benchmark.py",
        "tests/python/test_md_j_engine_endpoints.py",
    ]
    gates["host"] = _run_gate(
        "host", [*host_tests, "-k", "not asan_ubsan"], output, environment, args.timeout
    )
    save()
    gates["host_address_sanitizer"] = _run_gate(
        "host_address_sanitizer",
        ["tests/python/test_md_j_engine_runtime.py", "-k", "asan_ubsan"],
        output,
        environment,
        args.timeout,
    )
    save()
    if args.gpu:
        try:
            provenance["cuda"] = _cuda_preflight()
            if args.upstream_root:
                provenance["upstream_commit"] = verify_upstream(args.upstream_root)
        except (RuntimeError, ValueError, OSError, subprocess.SubprocessError) as error:
            gates["cuda_preflight"] = {"status": "failed", "reason": str(error)}
            for name in ("raw_j_cuda", "endpoints_cuda", "cuda_graph_replay"):
                gates[name] = {"status": "not-run", "reason": "CUDA preflight failed"}
            save()
            print(f"CUDA qualification blocked: {error}", file=sys.stderr)
            raise SystemExit(1) from error
        environment["GENERATIVEQC_TEST_MD_J_ENGINE"] = "1"
        source = "tests/python/test_md_j_engine_cuda.py"
        gates["raw_j_cuda"] = _run_gate(
            "raw_j_cuda",
            [
                source,
                "-k",
                "not complete_energy_force_endpoint and not pinned_gpu4pyscf and not graph_capture",
            ],
            output,
            environment,
            args.timeout,
        )
        save()
        gates["endpoints_cuda"] = _run_gate(
            "endpoints_cuda",
            [source, "-k", "complete_energy_force_endpoint"],
            output,
            environment,
            args.timeout,
        )
        save()
        gates["cuda_graph_replay"] = _run_gate(
            "cuda_graph_replay",
            [source, "-k", "graph_capture"],
            output,
            environment,
            args.timeout,
        )
        save()
        if args.upstream_root:
            environment["GENERATIVEQC_MD_J_UPSTREAM_ROOT"] = str(
                args.upstream_root.resolve()
            )
            environment["PYTHONPATH"] = (
                str(args.upstream_root.resolve())
                + os.pathsep
                + environment["PYTHONPATH"]
            )
            gates["upstream_gpu4pyscf"] = _run_gate(
                "upstream_gpu4pyscf",
                [source, "-k", "pinned_gpu4pyscf"],
                output,
                environment,
                args.timeout,
            )
            save()
    report = summarize(gates)
    print(
        json.dumps(
            {
                "output": str(output),
                "gates": {key: value["status"] for key, value in gates.items()},
                "correctness_qualified": report["correctness_qualified"],
                "default_promotion_qualified": False,
            },
            sort_keys=True,
        )
    )
    requested = REQUIRED_GATES if args.gpu else ("host",)
    if (
        any(gate.get("status") == "failed" for gate in gates.values())
        or any(gates[name]["status"] != "passed" for name in requested)
        or (args.upstream_root and gates["upstream_gpu4pyscf"]["status"] != "passed")
    ):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
