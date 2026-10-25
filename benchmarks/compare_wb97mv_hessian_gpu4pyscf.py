"""Compare complete C++ CLI and GPU4PySCF WB97M-V Hessian endpoints.

The GenerativeQC child is the linked Python-free native CUDA CLI, with Python
and compiler discovery unavailable. Python drives the independent comparator
only. Partial linked closure can fail a case; no runtime compilation is allowed.

Run under a finite Slurm GPU allocation. Both engines start in fresh processes,
perform a fresh direct SCF, include moving-grid/VV10/orbital response, and return
the full host Hessian. Timings include process startup and result serialization.
Persistent compilation caches are reused, not purged: sample zero is first
observed use, not a claimed empty-cache cold start. Later samples are interleaved
artifact-primed fresh-process runs, not converged-density or plan reuse.
Exactly one device must be visible in the benchmark step: GPU4PySCF otherwise
automatically distributes work over every visible GPU. For an exclusive-node
allocation, use a nested single-GPU Slurm step rather than editing visibility.

The default coarse grid is a latency/implementation diagnostic, not a chemically
grid-converged benchmark. DF is deliberately not enabled: the engines' default
range-separated DF approximations differ. No reference scientific kernel is
replaced. Ratios are published only after every Hessian passes the common gate.
Larger cases are opt-in via --cases; defaults retain the tiny qualification set.
--gpu-memory-limit-mib samples total-device usage, including comparator contexts.
It is a conservative watchdog, not allocator enforcement or an exact peak;
shorter-than-polling allocation spikes can be missed. Use an unshared GPU for
attributable memory observations and retain failed/time-limited attempts.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import signal
import statistics
import subprocess
import time
from dataclasses import asdict
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from benchmarks._retention import raw_output_path
from benchmarks._support import (
    cuda_accelerator_metadata,
    environment_metadata,
)

if TYPE_CHECKING:
    from collections.abc import Callable
    from typing import TextIO

CASES = {
    "h2": {
        "atoms": [("H", (0.03, -0.02, 0.01)), ("H", (0.15, 0.13, 1.5))],
        "spin": 0,
    },
    "water": {
        "atoms": [
            ("O", (0.0, 0.0, 0.0)),
            ("H", (1.43233673, 0.0, 1.10715266)),
            ("H", (-1.43233673, 0.0, 1.10715266)),
        ],
        "spin": 0,
    },
    "oh": {
        "atoms": [("O", (0.02, -0.03, 0.04)), ("H", (0.21, 0.31, 1.78))],
        "spin": 1,
    },
    "formaldehyde": {
        "atoms": [
            ("C", (0.0, 0.0, 0.0)),
            ("O", (0.0, 0.0, 2.286)),
            ("H", (1.765, 0.0, -1.1)),
            ("H", (-1.765, 0.0, -1.1)),
        ],
        "spin": 0,
    },
    "methanol": {
        "atoms": [
            ("C", (0.0, 0.0, 0.0)),
            ("O", (2.69, 0.0, 0.0)),
            ("H", (3.333, 1.696, 0.0)),
            ("H", (-0.673, 1.903, 0.0)),
            ("H", (-0.673, -0.9515, 1.648)),
            ("H", (-0.673, -0.9515, -1.648)),
        ],
        "spin": 0,
    },
    "ethane": {
        "atoms": [
            ("C", (-1.455, 0.0, 0.0)),
            ("C", (1.455, 0.0, 0.0)),
            ("H", (-2.142, 1.943, 0.0)),
            ("H", (-2.142, -0.9715, 1.682687)),
            ("H", (-2.142, -0.9715, -1.682687)),
            ("H", (2.142, -1.943, 0.0)),
            ("H", (2.142, 0.9715, 1.682687)),
            ("H", (2.142, 0.9715, -1.682687)),
        ],
        "spin": 0,
    },
    "water-dimer": {
        "atoms": [
            ("O", (0.0, 0.0, 0.0)),
            ("H", (1.43233673, 0.0, 1.10715266)),
            ("H", (-1.43233673, 0.0, 1.10715266)),
            ("O", (6.5, 0.0, 0.4)),
            ("H", (7.93233673, 0.0, 1.50715266)),
            ("H", (5.06766327, 0.0, 1.50715266)),
        ],
        "spin": 0,
    },
}

CASES["water-tetramer"] = {
    "atoms": [
        (
            symbol,
            tuple(
                coordinate + shift
                for coordinate, shift in zip(position, translation, strict=True)
            ),
        )
        for translation in (
            (0.0, 0.0, 0.0),
            (6.5, 0.0, 0.4),
            (0.0, 6.5, 0.8),
            (6.5, 6.5, 1.2),
        )
        for symbol, position in CASES["water"]["atoms"]
    ],
    "spin": 0,
}


def _digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _grid(arguments: argparse.Namespace) -> Any:
    from generativeqc import GridSpec

    return GridSpec(
        radial_points=arguments.grid[0],
        angular_polar=arguments.grid[1],
        angular_azimuth=arguments.grid[2],
    )


def _reference_worker(arguments: argparse.Namespace) -> dict[str, Any]:
    import cupy as cp
    from gpu4pyscf.dft import numint
    from gpu4pyscf.grad import rks as gradient_rks
    from gpu4pyscf.hessian import rks as hessian_rks

    from benchmarks._gpu4pyscf_grid import preserve_reference_grid_order
    from benchmarks.readme_wb97mv import reference_engine

    case = CASES[arguments.worker]
    for owner in (numint, gradient_rks, hessian_rks):
        owner.NLC_REMOVE_ZERO_RHO_GRID_THRESHOLD = 1e-8
    engine = reference_engine(
        case["atoms"], arguments.basis, _grid(arguments), spin=case["spin"]
    )
    preserve_reference_grid_order(engine)
    engine.conv_tol = 1e-13
    engine.conv_tol_grad = 1e-11
    engine.conv_tol_cpscf = 1e-11
    engine.direct_scf_tol = 1e-14
    engine.max_cycle = 240
    work = {"scf_cycles": 0, "response_operator_actions": None}

    def cycle(environment: dict[str, Any]) -> None:
        work["scf_cycles"] = max(work["scf_cycles"], int(environment["cycle"]) + 1)

    engine.callback = cycle
    cp.cuda.Stream.null.synchronize()
    started = time.perf_counter()
    energy = float(engine.kernel())
    cp.cuda.Stream.null.synchronize()
    scf_seconds = time.perf_counter() - started
    if not engine.converged:
        raise RuntimeError("GPU4PySCF SCF did not converge")
    hessian = engine.Hessian()
    hessian.grid_response = True
    started = time.perf_counter()
    result = hessian.kernel()
    host = result.get() if hasattr(result, "get") else np.asarray(result)
    cp.cuda.Stream.null.synchronize()
    hessian_seconds = time.perf_counter() - started
    matrix = host.transpose(0, 2, 1, 3).reshape(3 * engine.mol.natm, -1)
    if not np.isfinite(matrix).all():
        raise RuntimeError("GPU4PySCF returned a nonfinite Hessian")
    return {
        "engine": "gpu4pyscf",
        "matrix": matrix.tolist(),
        "units": "hartree/bohr^2",
        "energy_hartree": energy,
        "nao": engine.mol.nao_nr(),
        "scf_seconds": scf_seconds,
        "hessian_seconds": hessian_seconds,
        "grid_response": hessian.grid_response,
        "vv10": engine.do_nlc(),
        "vv10_density_threshold": 1e-8,
        "semilocal_padded_grid_points": len(engine.grids.weights),
        "nlc_padded_grid_points": len(engine.nlcgrids.weights),
        "response_grid_points": len(engine.cphf_grids.weights),
        "response_grid_policy": "same-as-primary",
        "raw_symmetry_error": float(np.max(abs(matrix - matrix.T))),
        "work": work,
    }


def _kill_child(process: subprocess.Popen) -> None:
    """Stop this endpoint's children before its private process group.

    Some Slurm installations track process groups rather than cgroups. Killing
    only the session can leave a comparator's separately grouped child alive
    after allocation cancellation. Direct children of this private session are
    owned processes, never a pre-existing shared compiler-cache server.
    """
    children = Path(f"/proc/{process.pid}/task/{process.pid}/children")
    try:
        children_ids = [int(value) for value in children.read_text().split()]
    except FileNotFoundError:
        # Linux builds without CONFIG_CHECKPOINT_RESTORE omit task/children.
        # PPid is still available; inspect only immediate children of this
        # private endpoint, never shared cache or unrelated process groups.
        children_ids = []
        for candidate in Path("/proc").iterdir():
            if not candidate.name.isdecimal():
                continue
            try:
                status = (candidate / "status").read_text()
            except (FileNotFoundError, PermissionError, ProcessLookupError):
                continue
            if f"\nPPid:\t{process.pid}\n" in status:
                children_ids.append(int(candidate.name))
    for child_id in children_ids:
        try:
            if os.getpgid(child_id) == child_id:
                os.killpg(child_id, signal.SIGKILL)
            else:
                os.kill(child_id, signal.SIGKILL)
        except ProcessLookupError:
            pass
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass


def _run_child(
    command: list[str],
    prefix: Path,
    timeout: float,
    *,
    resource_check: Callable[[], None] | None = None,
) -> dict[str, Any]:
    """Time process lifetime, with owned nested-worker cleanup on cancellation."""
    # Reserve both logs before launch; never truncate a previous run or follow a
    # pre-existing symlink. Children write directly to these owned descriptors,
    # so inherited pipes cannot make timeout cleanup wait indefinitely.
    with (
        prefix.with_suffix(".stdout").open("x+", encoding="utf-8") as out,
        prefix.with_suffix(".stderr").open("x+", encoding="utf-8") as err,
    ):
        started = time.perf_counter()
        process = subprocess.Popen(
            command,
            stdout=out,
            stderr=err,
            text=True,
            start_new_session=True,
            cwd=Path(__file__).resolve().parents[1],
            env=(
                dict(
                    os.environ,
                    PATH="/nonexistent",
                    PYTHONHOME="/nonexistent",
                    PYTHONPATH="/nonexistent",
                )
                if len(command) > 1 and command[1] == "hessian"
                else None
            ),
        )

        def interrupted(signum: int, frame: Any) -> None:
            raise SystemExit(128 + signum)

        previous_handler = signal.signal(signal.SIGTERM, interrupted)
        try:
            if resource_check is None:
                process.wait(timeout=timeout)
            else:
                deadline = started + timeout
                while True:
                    resource_check()
                    remaining = deadline - time.perf_counter()
                    if remaining <= 0:
                        raise subprocess.TimeoutExpired(command, timeout)
                    try:
                        process.wait(timeout=min(0.05, remaining))
                    except subprocess.TimeoutExpired:
                        continue
                    resource_check()
                    break
        except subprocess.TimeoutExpired:
            _kill_child(process)
            process.wait(timeout=5)
            raise RuntimeError(
                f"endpoint timed out after {timeout}s: {command}"
            ) from None
        except BaseException as error:
            _kill_child(process)
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                raise RuntimeError(
                    "endpoint cleanup exceeded its finite deadline"
                ) from error
            raise
        finally:
            signal.signal(signal.SIGTERM, previous_handler)
        elapsed = time.perf_counter() - started
        out.seek(0)
        err.seek(0)
        stdout, stderr = out.read(), err.read()
    if process.returncode:
        raise RuntimeError(f"endpoint exited {process.returncode}: {stderr[-4000:]}")
    payload = json.loads(stdout)
    matrix = np.asarray(payload["matrix"], dtype=float)
    if not np.isfinite(matrix).all() or payload["units"] != "hartree/bohr^2":
        raise RuntimeError("invalid Hessian output")
    return {"wall_seconds": elapsed, "command": command, "result": payload}


def _command(
    arguments: argparse.Namespace, engine: str, case_name: str, xyz: Path
) -> list[str]:
    if engine == "gpu4pyscf":
        return [
            str(arguments.compiler_python),
            "-m",
            "benchmarks.compare_wb97mv_hessian_gpu4pyscf",
            "--worker",
            case_name,
            "--basis",
            arguments.basis,
            "--grid",
            *(str(value) for value in arguments.grid),
        ]
    spin = CASES[case_name]["spin"]
    return [
        str(arguments.cli),
        "hessian",
        str(xyz),
        "--method",
        "wb97m-v-uks" if spin else "wb97m-v-rks",
        "--basis",
        arguments.basis,
        "--representation",
        "spherical",
        "--backend",
        "cuda",
        "--units",
        "bohr",
        "--multiplicity",
        str(spin + 1),
        "--grid-radial-points",
        str(arguments.grid[0]),
        "--grid-polar-points",
        str(arguments.grid[1]),
        "--grid-azimuth-points",
        str(arguments.grid[2]),
        "--max-iterations",
        "240",
        "--energy-tolerance",
        "1e-13",
        "--density-tolerance",
        "1e-11",
        "--screening-tolerance",
        "1e-14",
        "--response-atol",
        "1e-12",
        "--response-rtol",
        "1e-11",
        "--json",
    ]


def _summarize(rows: list[dict[str, Any]], tolerance: float) -> dict[str, Any]:
    """Gate all repeat pairs, never selecting only the fastest accurate sample."""
    if not math.isfinite(tolerance) or tolerance <= 0:
        raise ValueError("Hessian tolerance must be finite and positive")
    shapes = set()
    for row in rows:
        matrix = np.asarray(row["result"]["matrix"])
        if (
            matrix.ndim != 2
            or not matrix.size
            or matrix.shape[0] != matrix.shape[1]
            or not np.issubdtype(matrix.dtype, np.number)
            or np.iscomplexobj(matrix)
            or not np.isfinite(matrix).all()
        ):
            raise ValueError(
                "benchmark matrix must be finite real nonempty square data"
            )
        shapes.add(matrix.shape)
    if len(shapes) > 1:
        raise ValueError(
            "benchmark matrix shapes differ; broadcasting is not an accuracy gate"
        )
    groups = {
        engine: [row for row in rows if row["engine"] == engine]
        for engine in ("generativeqc", "gpu4pyscf")
    }
    errors = [
        float(
            np.max(
                abs(
                    np.asarray(native["result"]["matrix"])
                    - np.asarray(reference["result"]["matrix"])
                )
            )
        )
        for native in groups["generativeqc"]
        for reference in groups["gpu4pyscf"]
    ]
    medians = {
        engine: statistics.median(row["wall_seconds"] for row in samples[1:])
        if len(samples) > 1
        else None
        for engine, samples in groups.items()
    }
    accepted = bool(errors) and max(errors) <= tolerance
    return {
        "maximum_hessian_error_hartree_per_bohr2": max(errors) if errors else None,
        "hessian_absolute_tolerance": tolerance,
        "accepted": accepted,
        "artifact_primed_fresh_process_median_seconds": medians,
        "generativeqc_over_gpu4pyscf_wall_ratio": (
            medians["generativeqc"] / medians["gpu4pyscf"]
            if accepted and all(value is not None for value in medians.values())
            else None
        ),
    }


def main() -> None:
    """Execute synchronized, matched-science comparisons on an allocated GPU."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", choices=CASES, help=argparse.SUPPRESS)
    parser.add_argument(
        "--cases", nargs="+", choices=CASES, default=["h2", "water", "oh"]
    )
    parser.add_argument(
        "--engines",
        nargs="+",
        choices=("generativeqc", "gpu4pyscf"),
        default=["generativeqc", "gpu4pyscf"],
    )
    parser.add_argument("--basis", default="sto-3g")
    parser.add_argument("--grid", type=int, nargs=3, default=(8, 4, 8))
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--timeout-seconds", type=int, default=1200)
    parser.add_argument(
        "--gpu-memory-limit-mib",
        type=int,
        help="Abort above sampled total-device usage; includes parent/other contexts, not a hard allocation cap",
    )
    parser.add_argument("--hessian-atol", type=float, default=3e-6)
    parser.add_argument("--cli", type=Path)
    parser.add_argument(
        "--reference-python",
        "--compiler-python",
        dest="compiler_python",
        type=Path,
        help="Python for the GPU4PySCF comparator only; never passed to native CLI",
    )
    parser.add_argument("--cache", type=Path, default=Path(".artifacts"))
    parser.add_argument("--source-commit")
    parser.add_argument(
        "--output",
        type=raw_output_path,
        default=".artifacts/wb97mv-hessian-gpu4pyscf.json",
    )
    arguments = parser.parse_args()
    if not os.environ.get("SLURM_JOB_ID"):
        parser.error("real-GPU benchmarks must run in a finite Slurm allocation")
    if any(value <= 0 for value in arguments.grid):
        parser.error("grid dimensions must be positive")
    if not math.isfinite(arguments.hessian_atol) or arguments.hessian_atol <= 0:
        parser.error("Hessian tolerance must be finite and positive")
    if len(set(arguments.cases)) != len(arguments.cases) or len(
        set(arguments.engines)
    ) != len(arguments.engines):
        parser.error("cases and engines must not contain duplicates")
    if arguments.worker:
        print(json.dumps(_reference_worker(arguments)))
        return
    if arguments.repeats < 0 or arguments.timeout_seconds <= 0:
        parser.error("repeats must be nonnegative and timeout must be positive")
    if (
        arguments.gpu_memory_limit_mib is not None
        and arguments.gpu_memory_limit_mib <= 0
    ):
        parser.error("GPU memory limit must be positive")
    for name in ("cli", "compiler_python"):
        path = getattr(arguments, name)
        if path is None or not path.is_absolute() or not os.access(path, os.X_OK):
            parser.error(f"--{name.replace('_', '-')} requires an absolute executable")
    try:
        output = raw_output_path(arguments.output).absolute()
        logs = raw_output_path(output.parent / (output.stem + "-samples"))
    except ValueError as error:
        parser.error(str(error))
    output.parent.mkdir(parents=True, exist_ok=True)
    try:
        reserved = output.open("x+", encoding="utf-8")
    except FileExistsError:
        parser.error("output must be new, including symlinks")
    with reserved:
        try:
            logs.mkdir()
        except FileExistsError:
            parser.error("sample directory must be new, including symlinks")
        _run(arguments, reserved, logs)


def _portable_report(payload: dict[str, Any]) -> dict[str, Any]:
    """Preserve rejected nonfinite values as null plus explicit value tags.

    This conversion is only for the journal. Qualification still sees the
    original payload, and exact child stdout remains in the reserved raw log.
    """
    nonfinite: dict[str, str] = {}

    def convert(value: Any, path: str) -> Any:
        if isinstance(value, dict):
            return {
                key: convert(child, f"{path}.{key}") for key, child in value.items()
            }
        if isinstance(value, (tuple, list)):
            return [
                convert(child, f"{path}[{index}]") for index, child in enumerate(value)
            ]
        if isinstance(value, float) and not math.isfinite(value):
            nonfinite[path] = repr(value)
            return None
        return value

    portable = convert(payload, "$report")
    if nonfinite:
        portable["nonfinite_values"] = nonfinite
    return portable


def _run(arguments: argparse.Namespace, output: TextIO, logs: Path) -> None:
    """Update only the reserved evidence descriptor, including failed attempts."""
    payload: dict[str, Any] = {
        "schema": "generativeqc.wb97mv-hessian-comparison/v1",
        "status": "running",
        "cases": {},
    }

    def save() -> None:
        serialized = (
            json.dumps(_portable_report(payload), indent=2, allow_nan=False) + "\n"
        )
        output.seek(0)
        output.write(serialized)
        output.truncate()
        output.flush()

    save()
    try:
        _measure(arguments, payload, logs, save)
    except BaseException as error:
        payload.update(status="failed", failure=str(error))
        save()
        raise
    payload["status"] = "completed"
    save()


def _measure(
    arguments: argparse.Namespace,
    payload: dict[str, Any],
    logs: Path,
    save: Callable[[], None],
) -> None:
    import cupy as cp

    if cp.cuda.runtime.getDeviceCount() != 1:
        raise RuntimeError(
            "matched endpoints require exactly one Slurm-visible GPU; "
            "use a single-GPU srun step and preserve CUDA_VISIBLE_DEVICES"
        )
    library = Path(os.environ["GENERATIVEQC_LIBRARY"])
    accelerator = cuda_accelerator_metadata(cp)
    if accelerator.get("nvidia_smi"):
        accelerator["nvidia_smi"]["sampling_point"] = "before benchmark measurements"
    payload.update(
        {
            "environment": environment_metadata(
                distributions={
                    "gpu4pyscf": ("gpu4pyscf-cuda12x", "gpu4pyscf"),
                    "pyscf": ("pyscf",),
                    "cupy": ("cupy-cuda12x",),
                },
                accelerator=accelerator,
            ),
            "provenance": {
                "source_commit": arguments.source_commit,
                "script_sha256": _digest(Path(__file__)),
                "native_library_sha256": _digest(library),
                "cli_launcher_sha256": _digest(arguments.cli),
            },
            "slurm_job_id": os.environ["SLURM_JOB_ID"],
            "workload": {
                "basis": arguments.basis,
                "representation": "spherical",
                "visible_gpu_count": 1,
                "grid": asdict(_grid(arguments)),
                "density_fitting": False,
                "ecp": False,
                "moving_grid_response": True,
                "response_grid_policy": "same-as-primary",
                "vv10_density_threshold": 1e-8,
                "units": "hartree/bohr^2",
                "timing_boundary": "fresh process, fresh SCF, full host Hessian and JSON",
                "runtime_contract": "linked C++ CUDA compiler-AD endpoint; no Python/JIT in native child",
                "cache_policy": "reuse persistent caches; no empty-cache cold claim",
                "warm_repeats": arguments.repeats,
                "scientific_grid_convergence_claim": False,
                "gpu_memory_limit_mib": getattr(
                    arguments, "gpu_memory_limit_mib", None
                ),
            },
        }
    )
    save()
    for case_name in arguments.cases:
        case = CASES[case_name]
        xyz = logs / f"{case_name}.xyz"
        with xyz.open("x", encoding="utf-8") as coordinates:
            coordinates.write(
                f"{len(case['atoms'])}\nBohr\n"
                + "\n".join(
                    symbol + " " + " ".join(map(str, coordinates))
                    for symbol, coordinates in case["atoms"]
                )
                + "\n"
            )
        rows: list[dict[str, Any]] = []
        record = {
            "atoms_bohr": case["atoms"],
            "spin": case["spin"],
            "logical_grid_points": len(case["atoms"]) * int(np.prod(arguments.grid)),
            "hessian_coordinates": 3 * len(case["atoms"]),
            "samples": rows,
        }
        payload["cases"][case_name] = record
        save()
        for repeat in range(arguments.repeats + 1):
            engines = (
                arguments.engines
                if repeat % 2 == 0
                else list(reversed(arguments.engines))
            )
            for engine in engines:
                cp.cuda.Stream.null.synchronize()
                label = f"{case_name}-{repeat}-{engine}"
                print(f"START {label}", flush=True)
                command = _command(arguments, engine, case_name, xyz)
                # A prior subset cannot qualify this case while another required
                # sample is running or has failed. Keep every raw sample.
                record.pop("summary", None)
                attempt = {
                    "command": command,
                    "status": "running",
                    "endpoint_deadline_seconds": arguments.timeout_seconds,
                }
                record.setdefault("attempts", {})[label] = attempt
                save()
                resource_options: dict[str, Any] = {}
                memory_limit = getattr(arguments, "gpu_memory_limit_mib", None)
                if memory_limit is not None:
                    memory = {
                        "sampling_policy": "total-device used bytes at approximately 50 ms intervals; not an exact peak",
                        "limit_bytes": memory_limit * 1024**2,
                        "samples": 0,
                        "sampled_peak_bytes": 0,
                    }
                    record.setdefault("gpu_memory_observations", {})[label] = memory

                    def check_memory(memory: dict[str, Any] = memory) -> None:
                        """Use device-wide accounting conservatively, including parent contexts."""
                        free, total = cp.cuda.runtime.memGetInfo()
                        used = total - free
                        memory.setdefault("before_bytes", used)
                        memory["device_total_bytes"] = total
                        memory["sampled_peak_bytes"] = max(
                            memory["sampled_peak_bytes"], used
                        )
                        memory["samples"] += 1
                        if used > memory["limit_bytes"]:
                            raise RuntimeError(
                                f"sampled total-device GPU memory {used} bytes exceeds "
                                f"limit {memory['limit_bytes']} bytes"
                            )

                    resource_options["resource_check"] = check_memory
                attempt_started = time.perf_counter()
                try:
                    if resource_options:
                        check_memory()
                    row = _run_child(
                        command,
                        logs / label,
                        arguments.timeout_seconds,
                        **resource_options,
                    )
                except BaseException as error:
                    attempt.update(
                        status="failed",
                        comparator_elapsed_seconds=time.perf_counter()
                        - attempt_started,
                    )
                    record["failure"] = str(error)
                    save()
                    raise
                attempt.update(
                    status="completed",
                    comparator_elapsed_seconds=time.perf_counter() - attempt_started,
                )
                row.update(engine=engine, repeat=repeat)
                rows.append(row)
                save()
                if _portable_report(row).get("nonfinite_values"):
                    raise RuntimeError(
                        "nonfinite endpoint payload; see raw sample logs"
                    )
                size = 3 * len(case["atoms"])
                if np.asarray(row["result"]["matrix"]).shape != (size, size):
                    raise RuntimeError("endpoint Hessian has the wrong molecular shape")
                if engine == "generativeqc":
                    diagnostics = row["result"]["diagnostics"]
                    if diagnostics["numerical_host_fallbacks"] != 0:
                        raise RuntimeError("native CUDA Hessian used a host fallback")
                record["summary"] = _summarize(rows, arguments.hessian_atol)
                save()
                print(f"DONE {label}: {row['wall_seconds']:.6f}s", flush=True)
        print(json.dumps({case_name: record["summary"]}), flush=True)
    if len(arguments.engines) == 2 and not all(
        record["summary"]["accepted"] for record in payload["cases"].values()
    ):
        raise SystemExit("Hessian acceptance failed; no speed ratio is qualified")


if __name__ == "__main__":
    main()
