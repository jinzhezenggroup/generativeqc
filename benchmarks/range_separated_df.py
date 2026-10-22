"""Live range-DF endpoint runner with owned output and loaded-binary provenance.

The archived measurement harness remains immutable. This runner preserves its
scientific workload and timing boundaries, but reserves a new journal before
imports/preparation and records failures without overwriting previous evidence.
"""

from __future__ import annotations

import argparse
import ctypes
import dataclasses
import hashlib
import json
import os
import platform
import resource
import subprocess
import time
import typing
from contextlib import closing
from pathlib import Path

from benchmarks._retention import raw_output_path

WATER = [
    ("O", (0.0, 0.0, 0.0)),
    ("H", (0.1, 1.43, 1.1)),
    ("H", (-0.15, -1.45, 1.12)),
]


def json_default(value: typing.Any) -> typing.Any:
    """Keep diagnostic arrays and NumPy scalars machine-readable."""
    import numpy as np

    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"unsupported evidence value: {type(value).__name__}")


def loaded_library_identity(calculator: typing.Any) -> dict:
    """Hash the actual selected library, including a validated CUDA profile."""
    library = Path(os.fsdecode(calculator._library._name)).resolve(strict=True)
    with library.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    return {"library": str(library), "library_sha256": digest}


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cpu", "cuda"), required=True)
    parser.add_argument("--selector", choices=("direct", "df"), required=True)
    parser.add_argument("--case", choices=("water", "water-dimer"), default="water")
    parser.add_argument("--basis", default="def2-svp")
    parser.add_argument("--auxiliary", default="def2-svp")
    parser.add_argument("--endpoint", choices=("energy", "forces"), required=True)
    parser.add_argument("--grid", nargs=3, type=int, default=(24, 6, 12))
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--geometry-repeats", type=int, default=3)
    parser.add_argument("--build-boundary", required=True)
    parser.add_argument("--source-head")
    parser.add_argument("--output", type=raw_output_path, required=True)
    arguments = parser.parse_args(argv)
    if arguments.repeats < 1 or arguments.geometry_repeats < 0:
        parser.error("repeats must be positive and geometry-repeats nonnegative")
    if arguments.device == "cuda" and not os.environ.get("SLURM_JOB_ID"):
        parser.error("real GPU execution requires a finite Slurm allocation")
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    # O_EXCL rejects files, directories and symlinks before any preparation.
    with arguments.output.open("x", encoding="utf-8") as output:
        payload = {
            "schema": "range-separated-df-endpoint-v1",
            "arguments": {**vars(arguments), "output": str(arguments.output)},
            "samples": [],
        }

        def save(stage: str) -> None:
            payload["stage"] = stage
            output.seek(0)
            output.write(json.dumps(payload, default=json_default, indent=2) + "\n")
            output.truncate()
            output.flush()

        save("initialization")
        previous_selector = os.environ.get("GENERATIVEQC_DF_RANGE_EXCHANGE")
        try:
            # main() is also called in-process. Keep the selected operator local
            # to this campaign, including failed or interrupted measurements.
            os.environ["GENERATIVEQC_DF_RANGE_EXCHANGE"] = arguments.selector
            _measure(arguments, payload, save)
        except BaseException as error:
            payload["failure"] = {
                "stage": payload["stage"],
                "type": type(error).__name__,
                "message": str(error),
            }
            save("failed")
            raise
        finally:
            if previous_selector is None:
                os.environ.pop("GENERATIVEQC_DF_RANGE_EXCHANGE", None)
            else:
                os.environ["GENERATIVEQC_DF_RANGE_EXCHANGE"] = previous_selector


def _measure(arguments: argparse.Namespace, payload: dict, save: typing.Any) -> None:
    """Execute the unchanged endpoint after reserving its output journal."""
    import numpy as np
    from generativeqc import Calculator, GridSpec, KsOptions
    from generativeqc._ks_snapshot import NativeKsSnapshot

    from benchmarks._retained_basis import load_retained_basis

    atoms = list(WATER)
    if arguments.case == "water-dimer":
        atoms += [
            (element, (position[0] + 5.5, position[1] + 0.5, position[2]))
            for element, position in WATER
        ]
    grid = GridSpec(
        radial_points=arguments.grid[0],
        angular_polar=arguments.grid[1],
        angular_azimuth=arguments.grid[2],
    )
    auxiliary = arguments.auxiliary
    if auxiliary == "cc-pvdz-jkfit":
        auxiliary = load_retained_basis(
            Path(
                "benchmarks/results/issue206-practical-auxiliary/identity/cc-pvdz-jkfit.json"
            )
        )
    properties = ("energy", "forces") if arguments.endpoint == "forces" else ("energy",)

    def synchronize() -> None:
        """CPU endpoints already return host values synchronously."""

    if arguments.device == "cuda":
        runtime = ctypes.CDLL("libcudart.so.12")
        runtime.cudaDeviceSynchronize.restype = ctypes.c_int

        def synchronize() -> None:
            status = runtime.cudaDeviceSynchronize()
            if status:
                raise RuntimeError(f"cudaDeviceSynchronize failed: {status}")

    payload.update(
        {
            "schema": "range-separated-df-endpoint-v1",
            "arguments": {**vars(arguments), "output": str(arguments.output)},
            "environment": {
                "hostname": platform.node(),
                "platform": platform.platform(),
                "python": platform.python_version(),
                "job_id": os.environ.get("SLURM_JOB_ID"),
                "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
                "threads": {
                    name: os.environ.get(name)
                    for name in (
                        "OMP_NUM_THREADS",
                        "OPENBLAS_NUM_THREADS",
                        "MKL_NUM_THREADS",
                    )
                },
            },
            "source_head": arguments.source_head
            or subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
            "grid": dataclasses.asdict(grid),
            "scientific_controls": {
                "method": "wb97m-v-rks",
                "basis_representation": "spherical",
                "density_fitting_relative_threshold": 1e-10,
                "energy_tolerance": 1e-10,
                "density_tolerance": 1e-8,
                "screening_tolerance": 1e-14,
                "initial_guess": "hcore",
                "max_iterations": 200,
            },
            "atoms_bohr": atoms,
            "timing_boundary": "Calculator creation + prepare + execute returning host properties; serialization excluded",
            "cold_contract": "fresh prepared owner/process; existing compiler caches retained",
            "samples": [],
        }
    )
    if arguments.device == "cuda":
        gpu_query = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,uuid,driver_version",
                "--format=csv,noheader",
            ],
            text=True,
            capture_output=True,
            check=False,
        )
        payload["gpu_query"] = {
            "returncode": gpu_query.returncode,
            "stdout": gpu_query.stdout.strip(),
            "stderr": gpu_query.stderr.strip(),
        }
    save("calculator-initialization")
    started = time.perf_counter()
    calculator = Calculator(
        method="wb97m-v-rks",
        basis=arguments.basis,
        basis_representation="spherical",
        auxiliary_basis=auxiliary,
        device=arguments.device,
        density_fitting="auto",
        density_fitting_relative_threshold=1e-10,
        initial_guess=None,
        ks_options=KsOptions(grid=grid),
        energy_tolerance=1e-10,
        density_tolerance=1e-8,
        screening_tolerance=1e-14,
        max_iterations=200,
    )
    creation_seconds = time.perf_counter() - started
    payload.update(loaded_library_identity(calculator))
    save("preparation")
    started = time.perf_counter()
    with calculator.prepare_batch([atoms], warm_start=True) as batch:
        synchronize()
        payload["prepare_seconds"] = creation_seconds + time.perf_counter() - started
        if arguments.device == "cuda":
            driver = ctypes.CDLL("libcuda.so.1")
            device = ctypes.c_int()
            name = ctypes.create_string_buffer(256)
            status = driver.cuCtxGetDevice(ctypes.byref(device))
            if status == 0:
                status = driver.cuDeviceGetName(name, len(name), device.value)
            if status:
                raise RuntimeError(f"CUDA driver device query failed: {status}")
            payload["gpu_device"] = name.value.decode()
        cpu_work = []
        if arguments.device == "cpu" and arguments.endpoint == "forces":
            original = batch._public_dft_cpu_force

            def collect_work(index: int, geometry: typing.Any) -> typing.Any:
                result = original(index, geometry)
                cpu_work.append(result[1])
                return result

            batch._public_dft_cpu_force = collect_work

        def execute(label: str, coordinates: typing.Any = None) -> None:
            save(label)
            synchronize()
            started = time.perf_counter()
            keywords = {} if coordinates is None else {"coordinates": (coordinates,)}
            result = batch.execute(
                strict=True, properties=properties, **keywords
            ).items[0]
            synchronize()
            seconds = time.perf_counter() - started
            if not result.converged or not np.isfinite(result.energy):
                raise RuntimeError("benchmark endpoint did not converge")
            expected = "cuda" if arguments.device == "cuda" else "cpu_reference"
            if result.executed_backend != expected:
                raise RuntimeError(f"unexpected backend: {result.executed_backend}")
            with closing(NativeKsSnapshot(batch, 0)) as snapshot:
                if not snapshot.density_fitted:
                    raise RuntimeError("primary operator is not fitted")
                fitted_range = snapshot.range_density_fitted
                if fitted_range != (arguments.selector == "df"):
                    raise RuntimeError("native LR approximation proof disagrees")
                metadata = snapshot.metadata
            sample = {
                "label": label,
                "seconds": seconds,
                "energy": result.energy,
                "iterations": result.iterations,
                "density_rms": result.density_rms,
                "energy_change": result.energy_change,
                "executed_backend": result.executed_backend,
                "range_density_fitted": fitted_range,
                "snapshot_metadata": metadata,
                "basis_metadata": result.basis_metadata,
                "forces": result.forces,
            }
            if arguments.endpoint == "forces":
                sample["force_work"] = (
                    batch._stationary_cuda_execution.last_work
                    if arguments.device == "cuda"
                    else cpu_work[-1]
                )
            payload["samples"].append(sample)
            print(
                json.dumps(
                    {
                        key: sample[key]
                        for key in ("label", "seconds", "energy", "iterations")
                    }
                ),
                flush=True,
            )
            save(label + "-complete")

        execute("cold")
        for sample_index in range(arguments.repeats):
            execute(f"warm-{sample_index}")
        original_coordinates = np.array([position for _, position in atoms])
        displaced = original_coordinates.copy()
        displaced[1] += np.array((0.03, -0.02, 0.04))
        for sample_index in range(arguments.geometry_repeats):
            execute(
                f"geometry-{sample_index}",
                displaced if sample_index % 2 == 0 else original_coordinates,
            )
    payload["process_peak_rss_kib"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    save("complete")


if __name__ == "__main__":
    main()
