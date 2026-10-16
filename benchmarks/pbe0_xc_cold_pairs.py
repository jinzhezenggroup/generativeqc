"""Fresh-process PBE-family cold E+F observations inside one GPU allocation.

The primary parent wallclock includes synchronized E/F, provenance collection
and process cleanup. The earlier E/F delivery window is retained separately.
Each child creates one Calculator/owner without a supplied or retained density.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
import queue
import signal
import subprocess
import sys
import threading
import uuid
from pathlib import Path
from time import perf_counter
from typing import Any, TextIO

if not __debug__:
    raise RuntimeError("cold evidence collection requires Python without -O/-OO")

ENDPOINT = "PBE_COLD_ENDPOINT="
RESULT = "PBE_COLD_RESULT="
_RETENTION_FILE = Path(__file__).resolve().parent / "_retention.py"


def raw_output_path(path: str | Path, *, repository_root: Path | None = None) -> Path:
    """Apply the stdlib guard without importing a shadowing SDK/science package."""
    spec = importlib.util.spec_from_file_location("cold_retention", _RETENTION_FILE)
    assert spec is not None and spec.loader is not None
    retention = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(retention)
    root = (
        Path(__file__).resolve().parents[1]
        if repository_root is None
        else repository_root
    )
    try:
        return retention.raw_output_path(path, repository_root=root)
    except ValueError as error:
        raise argparse.ArgumentTypeError(str(error)) from error


def digest(path: Path) -> str:
    """Hash large SDK artifacts with bounded memory on supported Python 3.10+."""
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            result.update(chunk)
    return result.hexdigest()


def native_inventory(source: Path) -> tuple[int, str]:
    """Use the frozen SDK's complete production identity inventory."""
    spec = importlib.util.spec_from_file_location(
        "cold_sdk_identity", source / "tools/generate_build_identity.py"
    )
    assert spec is not None and spec.loader is not None
    identity = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(identity)
    inputs = identity._inventory(
        source, source / "cmake/GenerativeQCSourceIdentity.json"
    )
    return len(inputs), identity._source_identity(source, inputs)


def observe_process(
    command: list[str], *, nonce: str, timeout: float
) -> dict[str, Any]:
    """Retain complete child wallclock and a separate E/F delivery window."""
    started = perf_counter()
    row: dict[str, Any] = {
        "command": command,
        "nonce": nonce,
        "parent_pid": os.getpid(),
        "launch_started": started,
        "child_pid": None,
        "timeout": False,
        "endpoint": None,
        "worker": None,
        "synchronized": False,
        "diagnostics": [],
    }
    try:
        process = subprocess.Popen(
            command,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            start_new_session=True,
        )
    except (OSError, ValueError) as error:
        row.update(
            spawn_error=type(error).__name__,
            exit_code=None,
            process_completed=perf_counter(),
        )
        row["seconds"] = row["process_seconds"] = row["process_completed"] - started
        return row
    row["child_pid"] = process.pid
    row["launcher_pid"] = process.pid
    messages: queue.Queue[tuple[float, Any]] = queue.Queue()

    def read() -> None:
        try:
            assert process.stdout is not None
            for line in process.stdout:
                messages.put((perf_counter(), line))
        except (OSError, UnicodeError) as error:
            messages.put((perf_counter(), {"reader_error": type(error).__name__}))
        finally:
            messages.put((perf_counter(), None))

    reader = threading.Thread(target=read, daemon=True)
    reader.start()
    failed = False
    try:
        while True:
            remaining = timeout - (perf_counter() - started)
            if remaining <= 0:
                row["timeout"] = failed = True
                break
            try:
                received, line = messages.get(timeout=min(remaining, 0.1))
            except queue.Empty:
                continue
            if line is None:
                break
            if isinstance(line, dict):
                row.update(line)
                failed = True
                break
            if line.startswith((ENDPOINT, RESULT)):
                is_endpoint = line.startswith(ENDPOINT)
                key, prefix = (
                    ("endpoint", ENDPOINT) if is_endpoint else ("worker", RESULT)
                )
                if row[key] is not None:
                    row["duplicate_" + key] = True
                try:
                    payload = json.loads(line.removeprefix(prefix))
                    # Reject nonfinite/overflowed JSON before it can poison a
                    # failed-observation journal with unserializable values.
                    json.dumps(payload, allow_nan=False)
                    row[key] = payload
                except (ValueError, RecursionError):
                    row["parse_error"] = "malformed " + key
                    failed = True
                    break
                if is_endpoint:
                    row["endpoint_received"] = received
                    row["delivery_seconds"] = received - started
            elif len(row["diagnostics"]) < 20 and "://" not in line:
                row["diagnostics"].append(line.rstrip()[:500])
        if not failed:
            try:
                process.wait(timeout=max(0.01, timeout - (perf_counter() - started)))
            except subprocess.TimeoutExpired:
                row["timeout"] = failed = True
    except (OSError, TypeError, ValueError) as error:
        row["collector_error"] = type(error).__name__
        failed = True
    finally:
        if failed or process.poll() is None:
            # Only this child's new process group is owned by the collector.
            if hasattr(os, "killpg"):
                try:
                    os.killpg(process.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
            elif process.poll() is None:
                subprocess.run(
                    ["taskkill", "/PID", str(process.pid), "/T", "/F"],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    timeout=5,
                    check=False,
                )
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                pass
            if hasattr(os, "killpg"):
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
            elif process.poll() is None:
                process.kill()
        row["exit_code"] = process.wait(timeout=5)
        row["process_completed"] = perf_counter()
        row["seconds"] = row["process_seconds"] = row["process_completed"] - started
        endpoint = row["endpoint"]
        lineage = None
        if isinstance(endpoint, dict) and type(endpoint.get("pid")) is int:
            if endpoint["pid"] == process.pid and endpoint.get("ppid") == os.getpid():
                lineage = "direct"
            elif os.name == "nt" and endpoint.get("ppid") == process.pid:
                # Windows venv's executable is a launcher; retain both real
                # process identities instead of accepting an arbitrary PID.
                lineage = "windows-launcher"
            if lineage is not None:
                row["child_pid"] = endpoint["pid"]
        row["process_lineage"] = lineage
        row["synchronized"] = (
            not failed
            and row["exit_code"] == 0
            and isinstance(endpoint, dict)
            and endpoint.get("nonce") == nonce
            and lineage is not None
            and endpoint.get("synchronized") is True
            and isinstance(row["worker"], dict)
            and row["worker"].get("status") == "complete"
            and row["worker"].get("endpoint") == endpoint
        )
        reader.join(timeout=1)
        if process.stdout is not None and not reader.is_alive():
            process.stdout.close()
    return row


def worker(args: argparse.Namespace) -> dict[str, Any]:
    """Use the frozen SDK's production modules, without an oracle solve."""
    sys.path[:0] = [str(args.sdk_source / "python"), str(args.sdk_source)]
    os.environ.update(
        GENERATIVEQC_LIBRARY=str(args.library),
        GENERATIVEQC_CUDA_XC_PBE_POINT_SPECIALIZATION=(
            "1" if args.selection == "candidate" else "0"
        ),
        GENERATIVEQC_CUDA_XC_EQUAL_SPIN="0",
        GENERATIVEQC_CUDA_XC_BATCH_TILES="1",
        GENERATIVEQC_CUDA_XC_BATCH_BYTES=str(32 << 20),
        GENERATIVEQC_CUDA_XC_COMPACT_BATCH="0",
        GENERATIVEQC_STATIONARY_BECKE_PRIMITIVE="off",
    )
    import cupy as cp
    from generativeqc import Calculator, GridSpec, KsOptions

    from benchmarks._support import cuda_accelerator_metadata, environment_metadata
    from benchmarks.compare_gpu4pyscf_batch import native_build_metadata
    from benchmarks.dft_force_components import normalize_force_work
    from benchmarks.ks_preliminary_density import read_ao_work
    from benchmarks.readme_hf_scaling import scaling_cases
    from benchmarks.readme_omol25 import load_comparison_basis, protocol
    from benchmarks.readme_pbe0 import PBE0

    case = scaling_cases()[f"water-{args.atoms}"]
    basis, _ = load_comparison_basis(
        args.basis_file, case, role="orbital", compute_forces=True
    )
    grid = GridSpec(radial_points=48, angular_polar=16, angular_azimuth=32)
    scientific = protocol(args.atoms, basis, grid, 5, benchmark=PBE0)
    reference = json.loads(args.reference.read_text(encoding="utf-8"))
    assert scientific == reference["protocol"] and reference["stage"] == "complete"
    captured = []
    owner = None
    try:
        construct_start = perf_counter()
        calculator = Calculator(
            method="pbe0-rks",
            basis=basis,
            device="cuda",
            basis_representation="spherical",
            ks_options=KsOptions(grid=grid, tile_points=256),
            energy_tolerance=1e-12,
            density_tolerance=1e-10,
            screening_tolerance=1e-12,
            max_iterations=100,
            precision="fp64",
        )
        construct_end = perf_counter()
        owner = calculator.prepare_batch(
            [scientific["geometries_bohr"][args.geometry]], warm_start=False
        )
        prepare_end = perf_counter()
        original = owner._public_dft_cuda_force

        def force(*values: Any) -> Any:
            result = original(*values, tile_points=256)
            captured.append(result[1])
            return result

        owner._public_dft_cuda_force = force
        item = owner.execute(strict=False, properties=("energy", "forces")).items[0]
        cp.cuda.Stream.null.synchronize()
        executed = perf_counter()
        if item.status != 0 or not item.converged or item.forces is None:
            raise RuntimeError(f"cold E+F failed: {item.status_message}")
        if len(captured) != 1:
            raise RuntimeError("selected route did not publish one fixed-force receipt")
        raw = captured[0]
        if (
            type(raw.get("grid_tile_points_requested")) is not int
            or raw["grid_tile_points_requested"] != 256
            or raw.get("grid_tile_schedule") != "explicit"
        ):
            raise RuntimeError("fixed force request was not honored")
        endpoint = {
            "nonce": args.nonce,
            "pid": os.getpid(),
            "ppid": os.getppid(),
            "selection": args.selection,
            "geometry": args.geometry,
            "energy": item.energy,
            "forces": item.forces.tolist(),
            "synchronized": True,
        }
        print(ENDPOINT + json.dumps(endpoint, allow_nan=False), flush=True)
        # Expensive binary hashing, device metadata and independent validation
        # follow result delivery; none is subtracted from a measured interval.
        return {
            "status": "complete",
            "endpoint": endpoint,
            "protocol": scientific,
            "native_build": native_build_metadata(calculator),
            "sdk_source_head": subprocess.check_output(
                ["git", "-C", str(args.sdk_source), "rev-parse", "HEAD"], text=True
            ).strip(),
            "worker_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "environment": environment_metadata(
                accelerator=cuda_accelerator_metadata(cp)
            ),
            "constructor_interval": [construct_start, construct_end],
            "preparation_interval": [construct_end, prepare_end],
            "execution_interval": [prepare_end, executed],
            "warm_start_requested": False,
            "item_status": item.status,
            "converged": item.converged,
            "warm_start_used": item.warm_start_used,
            "warm_start_fallback": item.warm_start_fallback,
            "iterations": item.iterations,
            "fock_builds": item.fock_builds,
            "precision": item.precision,
            "initial_guess": item.initial_guess,
            "restart_origin": item.restart_origin,
            "native_ks_diagnostic": item.ks_diagnostic.to_payload(),
            "native_scf_ao_work": read_ao_work(owner),
            "native_force_work_raw": raw,
            "native_force_components": normalize_force_work(raw),
            "allocation": {
                "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
                "inspire_job_name": os.environ.get("INSPIRE_JOB_NAME"),
                "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
                "nvidia_visible_devices": os.environ.get("NVIDIA_VISIBLE_DEVICES"),
            },
        }
    finally:
        if owner is not None:
            owner.close()


def load_consumer() -> Any:
    """Load this collector's auditor without a frozen SDK package shadowing it."""
    spec = importlib.util.spec_from_file_location(
        "cold_consumer",
        Path(__file__).resolve().parents[1]
        / "tools/generativeqc_validation/pbe0_xc_cold.py",
    )
    assert spec is not None and spec.loader is not None
    consumer = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(consumer)
    return consumer


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--atoms", type=int, choices=(48, 96), required=True)
    parser.add_argument("--basis-file", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--sdk-source", type=Path, required=True)
    parser.add_argument("--library", type=Path, required=True)
    parser.add_argument("--expected-native-identity", required=True)
    parser.add_argument("--output", type=raw_output_path)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--feasibility", action="store_true")
    parser.add_argument("--timeout", type=float, default=1200)
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--selection", choices=("baseline", "candidate"))
    parser.add_argument("--geometry", type=int, choices=(0, 1))
    parser.add_argument("--nonce")
    args = parser.parse_args()
    if args.output is not None:
        # The SDK may be a different tree and --output may appear before it.
        # Finish CLI admission for both roots before allocation or any imports.
        try:
            args.output = raw_output_path(args.output, repository_root=args.sdk_source)
        except argparse.ArgumentTypeError as error:
            parser.error(f"argument --output: {error}")
    if not math.isfinite(args.timeout) or args.timeout <= 0:
        parser.error("timeout must be positive and finite")
    if not (
        os.environ.get("SLURM_JOB_ID") or os.environ.get("INSPIRE_JOB_NAME")
    ) or not (
        os.environ.get("CUDA_VISIBLE_DEVICES")
        or os.environ.get("NVIDIA_VISIBLE_DEVICES")
    ):
        parser.error("a real allocation and assigned visibility are required")
    if args.worker:
        if args.selection is None or args.geometry is None or not args.nonce:
            parser.error("worker identity is required")
        try:
            data = worker(args)
        except Exception as error:  # noqa: BLE001 - journal worker failures, then exit nonzero
            data = {
                "status": "failed",
                "error_type": type(error).__name__,
                "error": str(error)[:1000],
            }
        print(RESULT + json.dumps(data, allow_nan=False), flush=True)
        raise SystemExit(0 if data["status"] == "complete" else 1)
    if args.output is None or not (
        args.repeats >= 5 or (args.feasibility and args.repeats == 1)
    ):
        parser.error(
            "output and five pairs (or one explicit feasibility pair) required"
        )
    sys.path[:0] = [str(args.sdk_source / "python"), str(args.sdk_source)]

    # The collector/consumer version may be newer than a frozen SDK with an
    # unchanged native identity. Runtime and scientific helpers come from the
    # explicit SDK tree; this independent consumer comes with this collector.
    consumer = load_consumer()

    output = args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    # Windows' exclusive CRT open can follow a dangling symlink and create its
    # target. Reject that existing directory entry before reserving a new file.
    if output.is_symlink():
        parser.error("output must be new, including symlinks")
    try:
        journal = output.open("x+", encoding="utf-8")
    except FileExistsError:
        parser.error("output must be new, including symlinks")
    try:
        collect(args, journal, consumer)
    except BaseException as error:
        try:
            journal.close()
        except Exception as close_error:
            raise error from close_error
        raise
    else:
        journal.close()


def collect(args: argparse.Namespace, output: TextIO, consumer: Any) -> None:
    """Keep campaign updates on one exclusively reserved evidence file."""
    from generativeqc_compiler.common.evidence import canonical_hash
    from generativeqc_compiler.common.performance import assess_comparison

    reference = json.loads(args.reference.read_text(encoding="utf-8"))
    source_head = subprocess.check_output(
        ["git", "-C", str(args.sdk_source), "rev-parse", "HEAD"], text=True
    ).strip()
    assert not subprocess.check_output(
        [
            "git",
            "-C",
            str(args.sdk_source),
            "status",
            "--porcelain",
            "--untracked-files=no",
        ],
        text=True,
    ).strip()

    native_count, native_identity = native_inventory(args.sdk_source)
    assert native_identity == args.expected_native_identity
    manifest = {
        "worker_sha256": digest(Path(__file__)),
        "consumer_sha256": digest(Path(consumer.__file__)),
        "retention_guard_sha256": digest(_RETENTION_FILE),
        "sdk_source_head": source_head,
        "native_identity": args.expected_native_identity,
        "native_input_count": native_count,
        "library_path": str(args.library.resolve()),
        "library_sha256": digest(args.library),
        "basis_sha256": digest(args.basis_file),
        "reference_sha256": digest(args.reference),
        "controls": {
            "scf_tile_points": 256,
            "force_tile_points": 256,
            "equal_spin": 0,
            "point_batch_tiles": 1,
            "point_batch_bytes": 32 << 20,
            "compact_batch": 0,
            "becke_primitive": "off",
        },
    }
    record: dict[str, Any] = {
        "schema": "generativeqc.pbe0-xc-point-family-cold.v1",
        "protocol": reference["protocol"],
        "reference_sha256": hashlib.sha256(args.reference.read_bytes()).hexdigest(),
        "expected_native_identity": args.expected_native_identity,
        "manifest": manifest,
        "feasibility": args.feasibility,
        "repeats": args.repeats,
        "measurement": "fresh child launch through successful completion, including synchronized E/F, provenance and cleanup; delivery window separate",
        "stage": "running",
        "observations": [],
    }

    def save() -> None:
        serialized = json.dumps(record, indent=2, allow_nan=False) + "\n"
        output.seek(0)
        output.write(serialized)
        output.truncate()
        output.flush()

    def retain_failure(error: Exception) -> None:
        record.update(
            stage="incomplete",
            failure_type=type(error).__name__,
            failure=str(error)[:1000],
        )
        try:
            save()
        except Exception as journal_error:
            raise error from journal_error

    save()
    for geometry in (0, 1):
        for pair in range(args.repeats):
            for arm in (
                ("baseline", "candidate")
                if pair % 2 == 0
                else ("candidate", "baseline")
            ):
                nonce = uuid.uuid4().hex
                command = [sys.executable, str(Path(__file__).resolve()), "--worker"]
                for name in (
                    "atoms",
                    "basis_file",
                    "reference",
                    "sdk_source",
                    "library",
                    "expected_native_identity",
                ):
                    command += ["--" + name.replace("_", "-"), str(getattr(args, name))]
                command += [
                    "--selection",
                    arm,
                    "--geometry",
                    str(geometry),
                    "--nonce",
                    nonce,
                ]
                row = observe_process(command, nonce=nonce, timeout=args.timeout)
                row.update(
                    selection=arm,
                    geometry=geometry,
                    pair=pair,
                    workload="energy-plus-force",
                )
                row["inputs_hash"] = canonical_hash(
                    {
                        "protocol": record["protocol"],
                        "geometry": geometry,
                        "density_scope": "fresh process, owner and density",
                    }
                )
                record["observations"].append(row)
                try:
                    consumer.verify_observation(row, reference, manifest)
                except Exception as error:
                    retain_failure(error)
                    raise
                save()
    record["assessments"] = {
        str(g): assess_comparison(
            [r for r in record["observations"] if r["geometry"] == g]
        )
        for g in (0, 1)
    }
    record["stage"] = "complete"
    try:
        assert digest(Path(__file__)) == manifest["worker_sha256"]
        assert digest(Path(consumer.__file__)) == manifest["consumer_sha256"]
        assert digest(_RETENTION_FILE) == manifest["retention_guard_sha256"]
        assert digest(args.library) == manifest["library_sha256"]
        assert digest(args.basis_file) == manifest["basis_sha256"]
        assert digest(args.reference) == manifest["reference_sha256"]
        assert native_inventory(args.sdk_source) == (native_count, native_identity)
        assert (
            subprocess.check_output(
                ["git", "-C", str(args.sdk_source), "rev-parse", "HEAD"], text=True
            ).strip()
            == source_head
        )
        assert not subprocess.check_output(
            [
                "git",
                "-C",
                str(args.sdk_source),
                "status",
                "--porcelain",
                "--untracked-files=no",
            ],
            text=True,
        ).strip()
        consumer.verify_pairs(record, reference, allow_feasibility=args.feasibility)
    except Exception as error:
        retain_failure(error)
        raise
    record["qualification"] = (
        "numerical-work-feasibility" if args.feasibility else "numerical-work-complete"
    )
    record["cold_profitability_clearance"] = not args.feasibility and all(
        x["status"] == "pass" for x in record["assessments"].values()
    )
    record["dynamic_local_traffic"] = "not-run"
    save()


if __name__ == "__main__":
    main()
