"""Compare complete WB97M-V DF endpoints with exact or independently fitted LR.

Run each arm in a fresh scheduled process with the same production library.
Cold includes construction and preparation; each warm call starts from that
owner's frozen cold density. Geometry changes and energy-only consumers remain
separate observations, never pooled into the stationary warm population.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from contextlib import closing
from pathlib import Path
from time import perf_counter
from typing import Any

from benchmarks._retention import raw_output_path
from benchmarks.range_separated_df import json_default


def main(argv: list[str] | None = None) -> None:
    """Own an append-safe campaign and restore preparation-time selectors."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--selector", choices=("default", "auto", "direct", "df"), required=True
    )
    parser.add_argument(
        "--response", choices=("auto", "generic", "shell"), default="auto"
    )
    parser.add_argument("--atoms", type=int, choices=(3, 12, 48, 96), default=3)
    parser.add_argument("--endpoint", choices=("energy", "forces"), default="forces")
    parser.add_argument("--warm-repeats", type=int, default=3)
    parser.add_argument("--moved", action="store_true")
    parser.add_argument("--auxiliary", type=Path, required=True)
    parser.add_argument("--source-revision", required=True)
    parser.add_argument("--output", type=raw_output_path, required=True)
    arguments = parser.parse_args(argv)
    if arguments.warm_repeats < 1:
        parser.error("warm repeats must be positive")
    if not os.environ.get("SLURM_JOB_ID"):
        parser.error("real GPU execution requires a finite Slurm allocation")
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    with arguments.output.open("x", encoding="utf-8") as output:
        record: dict[str, Any] = {
            "schema": "generativeqc.wb97mv-lr-crossover.v1",
            "runner_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "status": "initializing",
            "arguments": {
                **vars(arguments),
                "output": str(arguments.output),
                "auxiliary": str(arguments.auxiliary),
            },
            "samples": [],
        }

        def retain(stage: str) -> None:
            record["status"] = stage
            output.seek(0)
            json.dump(record, output, default=json_default, allow_nan=False, indent=2)
            output.write("\n")
            output.truncate()
            output.flush()

        controls = {
            "GENERATIVEQC_DF_RANGE_EXCHANGE": (
                None if arguments.selector == "default" else arguments.selector
            ),
            "GENERATIVEQC_DF_WEIGHTED_EXECUTION": (
                None if arguments.response == "auto" else arguments.response
            ),
        }
        previous = {name: os.environ.get(name) for name in controls}
        retain("initializing")
        try:
            for name, value in controls.items():
                if value is None:
                    os.environ.pop(name, None)
                else:
                    os.environ[name] = value
            _measure(arguments, record, retain)
            retain("complete")
        except BaseException as error:
            record["failure"] = {
                "stage": record["status"],
                "type": type(error).__name__,
                "message": str(error),
            }
            retain("failed")
            raise
        finally:
            for name, value in previous.items():
                if value is None:
                    os.environ.pop(name, None)
                else:
                    os.environ[name] = value


def _measure(arguments: Any, record: dict, retain: Any) -> None:
    """Measure host-returning endpoints, authenticating the actual radial owner."""
    import cupy as cp
    import numpy as np
    from generativeqc import Calculator
    from generativeqc._ks_snapshot import NativeKsSnapshot

    from benchmarks._support import cuda_accelerator_metadata, environment_metadata
    from benchmarks.readme_defaults import native_force_tile, native_settings
    from benchmarks.readme_hf_scaling import scaling_cases
    from benchmarks.readme_method_endpoints import binary_identity

    atoms = scaling_cases()[f"water-{arguments.atoms}"].atoms
    properties = ("energy", "forces") if arguments.endpoint == "forces" else ("energy",)
    record.update(
        geometry_bohr=atoms,
        protocol={
            "cold": "constructor+prepare+SCF+requested-properties+host-export",
            "warm": "same-owner; frozen owner-local cold density",
            "defaults": "production grid, precision, solver and resource policies",
        },
        scheduler={
            name: os.environ.get(name)
            for name in (
                "SLURM_JOB_ID",
                "SLURM_JOB_NODELIST",
                "CUDA_VISIBLE_DEVICES",
            )
        },
        environment=environment_metadata(),
        accelerator=cuda_accelerator_metadata(cp, sampling_point="before measurements"),
        controls={
            name: value
            for name, value in os.environ.items()
            if name.startswith(("GENERATIVEQC_", "OMP_", "OPENBLAS_", "MKL_"))
        },
    )
    retain("cold")
    cp.cuda.runtime.deviceSynchronize()
    started = perf_counter()
    calculator = Calculator(
        method="wb97m-v-rks",
        device="cuda",
        basis="def2-svp",
        basis_representation="spherical",
        density_fitting="cuda",
        auxiliary_basis=arguments.auxiliary,
    )
    try:
        with calculator.prepare_batch([atoms], warm_start=True) as owner:

            def execute(label: str, coordinates: Any = None) -> None:
                nonlocal started
                if label != "cold":
                    retain(label)
                    cp.cuda.runtime.deviceSynchronize()
                    started = perf_counter()
                keywords = (
                    {} if coordinates is None else {"coordinates": (coordinates,)}
                )
                try:
                    item = owner.execute(
                        strict=True, properties=properties, **keywords
                    ).items[0]
                except BaseException:
                    detail = calculator._library.generativeqc_context_get_last_detail(
                        owner._context
                    )
                    record["native_failure_detail"] = (
                        detail.decode("utf-8", errors="replace") if detail else None
                    )
                    raise
                cp.cuda.runtime.deviceSynchronize()
                seconds = perf_counter() - started
                if not item.converged or item.executed_backend != "cuda":
                    raise RuntimeError("native CUDA endpoint did not converge")
                if not np.isfinite(item.energy):
                    raise RuntimeError("nonfinite endpoint energy")
                if arguments.endpoint == "forces" and (
                    np.asarray(item.forces).shape != (arguments.atoms, 3)
                    or not np.isfinite(item.forces).all()
                ):
                    raise RuntimeError("incomplete/nonfinite endpoint forces")
                with closing(NativeKsSnapshot(owner, 0)) as snapshot:
                    fitted_range = snapshot.range_density_fitted
                    if not snapshot.density_fitted or (
                        arguments.selector in ("direct", "df")
                        and fitted_range != (arguments.selector == "df")
                    ):
                        raise RuntimeError(
                            "executed radial approximation disagrees with request"
                        )
                    if (
                        record.setdefault("resolved_range_density_fitted", fitted_range)
                        != fitted_range
                    ):
                        raise RuntimeError("radial approximation changed during replay")
                    metadata = snapshot.metadata
                sample = {
                    "phase": label,
                    "seconds": seconds,
                    "energy": item.energy,
                    "forces": item.forces,
                    "iterations": item.iterations,
                    "fock_builds": item.fock_builds,
                    "warm_start_used": item.warm_start_used,
                    "range_density_fitted": fitted_range,
                    "snapshot_metadata": metadata,
                    "settings": native_settings(calculator, item),
                    "actual_force_tile": native_force_tile(owner),
                    "force_work": getattr(
                        getattr(owner, "_stationary_cuda_execution", None),
                        "last_work",
                        None,
                    ),
                }
                if not record.get("native_build"):
                    record["native_build"] = binary_identity(calculator)
                record["samples"].append(sample)
                retain(label + "-complete")
                print(
                    json.dumps(
                        {
                            name: sample[name]
                            for name in (
                                "phase",
                                "seconds",
                                "energy",
                                "iterations",
                                "fock_builds",
                            )
                        }
                    ),
                    flush=True,
                )

            execute("cold")
            owner.set_warm_start_updates(False)
            for repeat in range(arguments.warm_repeats):
                execute(f"warm-{repeat + 1}")
            if arguments.moved:
                coordinates = np.array([position for _, position in atoms])
                coordinates[1] += np.array((0.001, -0.002, 0.0015))
                owner.set_warm_start_updates(True)
                execute("moved", coordinates)
                owner.set_warm_start_updates(False)
                execute("moved-warm", coordinates)
    finally:
        calculator.clear_cache()


if __name__ == "__main__":
    main()
