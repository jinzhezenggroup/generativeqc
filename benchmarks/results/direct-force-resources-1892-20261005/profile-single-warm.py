"""Retain actual warm work and an isolated, explicitly intrusive CUDA trace."""

from __future__ import annotations

import argparse
import ctypes
import json
import time
from contextlib import ExitStack
from dataclasses import asdict
from pathlib import Path
from unittest.mock import patch

import cupy as cp
from generativeqc import Calculator, GridSpec, KsOptions
from generativeqc._ks_snapshot import NativeKsSnapshot

from benchmarks.compare_df_direct_endpoint import check_endpoint
from benchmarks.compare_gpu4pyscf_batch import (
    load_comparison_basis,
    native_build_metadata,
    scaled_geometries,
)
from benchmarks.dft_force_components import normalize_force_work
from benchmarks.readme_hf_scaling import scaling_cases


def main() -> None:
    """Capture one fixed post-cold replay, not SCF setup or reference work."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", choices=("hf", "pbe0"), required=True)
    parser.add_argument("--atoms", type=int, choices=(48, 96), required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    oracle = json.loads(arguments.reference.read_text())
    baseline = next(row for row in oracle["records"] if row["phase"] == "cold")
    case = scaling_cases()[f"water-{arguments.atoms}"]
    geometry = scaled_geometries(case.atoms, 1)[0]
    options = {}
    basis = case.generativeqc_basis
    if arguments.method == "pbe0":
        basis, _ = load_comparison_basis(
            Path("benchmarks/results/pbe0-def2-svp-20261003/def2-svp-ho.json"),
            case,
            role="orbital",
            compute_forces=True,
        )
        options["ks_options"] = KsOptions(
            grid=GridSpec(radial_points=48, angular_polar=16, angular_azimuth=32)
        )
    calculator = Calculator(
        method="rhf" if arguments.method == "hf" else "pbe0-rks",
        basis=basis,
        device="cuda",
        basis_representation="spherical",
        energy_tolerance=1e-12,
        density_tolerance=1e-10,
        screening_tolerance=1e-12,
        max_iterations=100,
        **options,
    )
    record = {
        "scope": "intrusive single-warm CUDA trace; not clean performance timing",
        "native_build": native_build_metadata(calculator),
        "records": [],
    }
    runtime = ctypes.CDLL("libcudart.so.12")
    for name in ("cudaProfilerStart", "cudaProfilerStop"):
        getattr(runtime, name).argtypes = []
        getattr(runtime, name).restype = ctypes.c_int
    original_derivatives = NativeKsSnapshot.cuda_full_range_derivatives

    def observed_derivatives(self: NativeKsSnapshot, atom_count: int) -> object:
        """Label the unchanged shared native derivative producer, not an oracle."""
        cp.cuda.nvtx.RangePush("p0b/full-range-Jprime-Kprime")
        try:
            return original_derivatives(self, atom_count)
        finally:
            cp.cuda.nvtx.RangePop()

    with ExitStack() as stack:
        stack.enter_context(
            patch.object(
                NativeKsSnapshot, "cuda_full_range_derivatives", observed_derivatives
            )
        )
        # HF's opt-in class census is explicitly intrusive. DFT shell profiles
        # do not describe its exported stationary source, so never relabel them.
        owner = stack.enter_context(
            calculator.prepare_batch(
                [geometry],
                warm_start=True,
                shell_class_profiling=arguments.method == "hf",
            )
        )
        force_work = None
        original_force = owner._public_dft_cuda_force

        def observed_force(*values: object) -> object:
            """Keep already-produced component work during the instrumented call."""
            nonlocal force_work
            cp.cuda.nvtx.RangePush("p0b/public-stationary-force")
            try:
                result = original_force(*values)
                force_work = result[1]
                return result
            finally:
                cp.cuda.nvtx.RangePop()

        owner._public_dft_cuda_force = observed_force
        for phase in ("cold", "prime", "profiled-warm"):
            force_work = None
            cp.cuda.Stream.null.synchronize()
            captured = phase == "profiled-warm"
            if captured and runtime.cudaProfilerStart() != 0:
                raise RuntimeError("CUDA profiler start failed")
            cp.cuda.nvtx.RangePush(f"p0b/{phase}")
            started = time.perf_counter()
            try:
                item = owner.execute(
                    strict=True, properties=("energy", "forces")
                ).items[0]
                cp.cuda.Stream.null.synchronize()
            finally:
                seconds = time.perf_counter() - started
                cp.cuda.nvtx.RangePop()
                if captured and runtime.cudaProfilerStop() != 0:
                    raise RuntimeError("CUDA profiler stop failed")
            row = {
                "phase": phase,
                "instrumented_seconds": seconds,
                "iterations": item.iterations,
                "fock_builds": item.fock_builds,
                "precision": item.precision,
                "incremental_direct_jk": item.incremental_direct_jk,
                "native_ks_diagnostic": item.ks_diagnostic.to_payload()
                if item.ks_diagnostic is not None
                else None,
                "physical_residual_rms": item.physical_residual_rms,
                "energy": item.energy,
                "forces": item.forces.tolist(),
                **check_endpoint(item, baseline),
                "native_force_work": normalize_force_work(force_work)
                if force_work is not None
                else None,
            }
            record["records"].append(row)
            arguments.output.write_text(
                json.dumps(record, indent=2, allow_nan=False) + "\n"
            )
            if not row["gate"]:
                raise RuntimeError(f"independent profile endpoint failed: {phase}")
            owner.set_warm_start_updates(False)
        if arguments.method == "hf":
            record["shell_class_profile"] = [
                asdict(entry) for entry in owner.last_shell_class_profile()
            ]
            record["shell_class_profile_scope"] = (
                "intrusive final-density HF admission; not DFT stationary Jprime/Kprime"
            )
        arguments.output.write_text(
            json.dumps(record, indent=2, allow_nan=False) + "\n"
        )


if __name__ == "__main__":
    main()
