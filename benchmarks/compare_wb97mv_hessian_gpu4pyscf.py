"""Compare complete C++ CLI and GPU4PySCF WB97M-V Hessian endpoints.

This driver measures the existing Python/CuPy CLI bridge. It is not a benchmark
of the unfinished Python-free native Hessian runtime or its AOT leaf package.

Run under a finite Slurm GPU allocation. Both engines start in fresh processes,
perform a fresh direct SCF, include moving-grid/VV10/orbital response, and return
the full host Hessian. Timings include process startup and result serialization.
Persistent compilation caches are reused, not purged: sample zero is first
observed use, not a claimed empty-cache cold start. Later samples are interleaved
artifact-primed fresh-process runs, not converged-density or plan reuse.

The default coarse grid is a latency/implementation diagnostic, not a chemically
grid-converged benchmark. DF is deliberately not enabled: the engines' default
range-separated DF approximations differ. No reference scientific kernel is
replaced. Ratios are published only after every Hessian passes the common gate.
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
        "raw_symmetry_error": float(np.max(abs(matrix - matrix.T))),
        "work": work,
    }


def _kill_child(process: subprocess.Popen) -> None:
    """Kill the CLI's separately grouped worker before its supervising process.

    Some Slurm installations track process groups rather than cgroups. Killing
    only the timed CLI session can otherwise leave its Python CUDA worker alive
    after allocation cancellation. Direct children of this private session are
    owned endpoints, never a pre-existing shared compiler-cache server.
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


def _run_child(command: list[str], prefix: Path, timeout: float) -> dict[str, Any]:
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
        )

        def interrupted(signum: int, frame: Any) -> None:
            raise SystemExit(128 + signum)

        previous_handler = signal.signal(signal.SIGTERM, interrupted)
        try:
            process.wait(timeout=timeout)
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
        "--compiler-python",
        str(arguments.compiler_python),
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
        "--cache",
        str(arguments.cache),
        "--timeout-seconds",
        str(arguments.timeout_seconds),
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
    parser.add_argument("--cases", nargs="+", choices=CASES, default=list(CASES))
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
    parser.add_argument("--hessian-atol", type=float, default=3e-6)
    parser.add_argument("--cli", type=Path)
    parser.add_argument("--compiler-python", type=Path)
    parser.add_argument("--cache", type=Path, default=Path(".artifacts"))
    parser.add_argument("--source-commit")
    parser.add_argument(
        "--output",
        type=raw_output_path,
        default=Path(".artifacts/wb97mv-hessian-gpu4pyscf.json"),
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
                "grid": asdict(_grid(arguments)),
                "density_fitting": False,
                "ecp": False,
                "moving_grid_response": True,
                "vv10_density_threshold": 1e-8,
                "units": "hartree/bohr^2",
                "timing_boundary": "fresh process, fresh SCF, full host Hessian and JSON",
                "runtime_contract": "explicit C++ CLI Python/CuPy bridge; not Python-free",
                "cache_policy": "reuse persistent caches; no empty-cache cold claim",
                "warm_repeats": arguments.repeats,
                "scientific_grid_convergence_claim": False,
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
                save()
                try:
                    row = _run_child(
                        command,
                        logs / label,
                        arguments.timeout_seconds
                        + (30 if engine == "generativeqc" else 0),
                    )
                except Exception as error:
                    record["failure"] = str(error)
                    save()
                    raise
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
