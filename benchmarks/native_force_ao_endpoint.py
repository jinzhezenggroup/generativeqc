"""Replay one retained independent native E/F case in a fresh native process.

Reference acquisition is separate: this process never imports CuPy/GPU4PySCF
or runs an oracle. Use finite Slurm allocations, preserve assigned visibility,
and interleave dense/auto processes before making endpoint timing comparisons.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import lzma
import os
import socket
import sys
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from typing import NoReturn

import numpy as np
from generativeqc import Calculator, GridSpec, KsOptions

from benchmarks._cases import BenchmarkCase
from benchmarks._retained_basis import load_retained_comparison_basis
from benchmarks._retention import raw_output_path
from benchmarks.compare_gpu4pyscf_batch import load_comparison_basis

ROOT = Path(__file__).resolve().parents[1]


def _forbid_python_forces(*args: object, **kwargs: object) -> NoReturn:
    """Qualification must not silently switch to the compatibility consumer."""
    raise AssertionError("public native E/F entered Python force orchestration")


def main() -> None:
    """Time complete native preparation/E/F against a source-bound oracle receipt."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--receipt", required=True, type=Path)
    parser.add_argument("--method", required=True)
    parser.add_argument("--direct", action="store_true")
    parser.add_argument("--mode", required=True, choices=("dense", "auto"))
    parser.add_argument("--warm-repeats", type=int, default=3)
    parser.add_argument("--pair", type=int, default=0)
    parser.add_argument("--device-label", default="local")
    parser.add_argument("--output", required=True, type=raw_output_path)
    arguments = parser.parse_args()
    try:
        trace = raw_output_path(arguments.output.with_suffix(".trace.jsonl"))
    except ValueError as error:
        parser.error(f"trace output: {error}")
    if (
        not os.environ.get("SLURM_JOB_ID")
        or arguments.warm_repeats < 1
        or not os.environ.get("GENERATIVEQC_LIBRARY")
    ):
        parser.error(
            "a finite Slurm allocation, explicit native library and positive repeats are required"
        )
    with lzma.open(arguments.receipt, "rt") as handle:
        receipt = json.load(handle)
    case = next(
        case
        for case in receipt["cases"]
        if case["input"]["method"] == arguments.method
        and case["input"]["fitted"] is not arguments.direct
    )
    inputs = case["input"]
    library = Path(os.environ["GENERATIVEQC_LIBRARY"])
    assert hashlib.sha256(library.read_bytes()).hexdigest() == inputs["library_sha256"]
    for name in ("basis", "auxiliary"):
        assert (
            hashlib.sha256((ROOT / inputs[name]).read_bytes()).hexdigest()
            == inputs[f"{name}_sha256"]
        )
    # Only this scientific operator control may be replayed from the receipt.
    # Slurm visibility and unrelated execution settings are never overridden.
    assert set(inputs["controls"]) <= {"GENERATIVEQC_DF_RANGE_EXCHANGE"}
    os.environ.update(inputs["controls"])
    os.environ["GENERATIVEQC_PROFILE"] = "off"
    if arguments.mode == "dense":
        os.environ["GENERATIVEQC_CUDA_FORCE_ACTIVE_AO"] = "0"
    else:
        os.environ.pop("GENERATIVEQC_CUDA_FORCE_ACTIVE_AO", None)
    fixture = BenchmarkCase(
        description="retained native force AO endpoint",
        atoms=tuple((symbol, tuple(point)) for symbol, point in inputs["atoms"]),
        generativeqc_basis="def2-svp",
        pyscf_basis="def2-svp",
        charge=inputs["charge"],
        multiplicity=inputs["multiplicity"],
        basis_representation="spherical",
        expected_ao_count=inputs["aos"],
    )
    basis, _ = load_comparison_basis(
        ROOT / inputs["basis"], fixture, role="orbital", compute_forces=True
    )
    auxiliary_loader = (
        load_retained_comparison_basis
        if inputs["auxiliary_retained"]
        else load_comparison_basis
    )
    auxiliary, _ = auxiliary_loader(
        ROOT / inputs["auxiliary"], fixture, role="auxiliary", compute_forces=True
    )
    os.environ["GENERATIVEQC_DF_TRACE"] = str(trace)
    samples = []
    started_at = datetime.now(timezone.utc).isoformat()
    started = perf_counter()
    calculator = Calculator(
        inputs["method"],
        basis=basis,
        auxiliary_basis=auxiliary if inputs["fitted"] else None,
        basis_representation="spherical",
        device="cuda",
        precision="fp64",
        density_fitting="cuda" if inputs["fitted"] else "none",
        density_fitting_relative_threshold=inputs["density_fitting_relative_threshold"],
        ks_options=KsOptions(
            grid=GridSpec(*inputs["grid"]),
            nonlocal_grid=GridSpec(*inputs["nonlocal_grid"])
            if inputs["nonlocal_grid"]
            else None,
            force_tile_points=inputs["force_tile_points"],
        ),
        energy_tolerance=inputs["energy_tolerance"],
        density_tolerance=inputs["density_tolerance"],
        screening_tolerance=inputs["screening_tolerance"],
        max_iterations=inputs["max_iterations"],
    )
    with calculator.prepare_batch(
        [fixture.atoms],
        charges=[inputs["charge"]],
        multiplicities=[inputs["multiplicity"]],
        warm_start=True,
    ) as batch:
        assert Path(calculator._library._name).resolve() == library.resolve()
        assert batch._native_dft_force_eligible(0)
        batch._public_dft_cuda_force = _forbid_python_forces
        for geometry, prefix in ((0, ""), (1, "moved-")):
            for repeat in range(arguments.warm_repeats + 1):
                stage = (
                    ("moved" if geometry else "cold")
                    if repeat == 0
                    else f"{prefix}warm"
                )
                batch.set_warm_start_updates(repeat == 0)
                trace.write_text("")
                if samples:
                    started = perf_counter()
                result = batch.execute(
                    coordinates=(np.asarray(inputs["moved"]),) if geometry else None,
                    strict=True,
                    properties=("energy", "forces"),
                ).items[0]
                seconds = perf_counter() - started
                expected = case["references"][geometry]
                assert result.converged and result.executed_backend == "cuda"
                assert np.shape(result.forces) == (len(fixture.atoms), 3)
                assert np.isfinite(result.energy) and np.all(np.isfinite(result.forces))
                assert (
                    np.isfinite(result.physical_residual_rms)
                    and 0 <= result.physical_residual_rms <= 1e-9
                )
                assert abs(result.energy - expected["energy"]) <= 1e-8
                np.testing.assert_allclose(
                    result.forces, expected["forces"], atol=3e-7, rtol=0
                )
                rows = [json.loads(line) for line in trace.read_text().splitlines()]
                work = [
                    row["counters"]
                    for row in rows
                    if row["operation"] == "stationary_force"
                ][-1]
                assert bool(work["indexed_grid"]) == (arguments.mode == "auto")
                samples.append(
                    {
                        "stage": stage,
                        "seconds": seconds,
                        "energy": result.energy,
                        "forces": result.forces.tolist(),
                        "iterations": result.iterations,
                        "fock_builds": result.fock_builds,
                        "physical_residual_rms": result.physical_residual_rms,
                        "work": work,
                    }
                )
    assert not any(
        module.split(".")[0] in {"cupy", "gpu4pyscf", "pyscf"} for module in sys.modules
    ), "native timing process imported a reference/oracle package"
    arguments.output.write_text(
        json.dumps(
            {
                "mode": arguments.mode,
                "pair": arguments.pair,
                "device": arguments.device_label,
                "case": case["id"],
                "input": inputs,
                "samples": samples,
                "job": os.environ["SLURM_JOB_ID"],
                "host": socket.gethostname(),
                "process": os.getpid(),
                "visible_devices": os.environ["CUDA_VISIBLE_DEVICES"],
                "started_at": started_at,
            },
            allow_nan=False,
            sort_keys=True,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
