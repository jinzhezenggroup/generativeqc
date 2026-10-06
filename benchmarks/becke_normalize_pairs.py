"""Same-binary, frozen-density complete E+F pairs for Becke normalization.

The control changes only the native normalization schedule before topology.
Whole gradient executors are replaced/primed outside timing; cold/moved setup
is retained separately and is not evidence of a cold/moved performance win.
"""

from __future__ import annotations

import argparse
import ctypes as ct
import hashlib
import json
import os
from pathlib import Path
from time import perf_counter
from typing import Any
from unittest.mock import patch

import numpy as np
from generativeqc import Calculator, GridSpec, KsOptions, _stationary_cuda
from generativeqc_compiler.common.evidence import block_error, canonical_hash
from generativeqc_compiler.common.performance import (
    assess_comparison,
    measure_interleaved,
)
from generativeqc_compiler.common.timing import interleaved_selection_order

from benchmarks._support import (
    cuda_accelerator_metadata,
    environment_metadata,
    raw_output_path,
)
from benchmarks.compare_gpu4pyscf_batch import (
    native_build_metadata,
    require_tuned_native_build,
)
from benchmarks.dft_force_components import normalize_force_work
from benchmarks.readme_hf_scaling import scaling_cases
from benchmarks.readme_omol25 import load_comparison_basis, protocol, source_hashes
from benchmarks.readme_pbe0 import PBE0


def main() -> None:
    """Retain every prime, vector, actual solver history and selected schedule."""
    import cupy as cp

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--atoms", type=int, choices=(48, 96), required=True)
    parser.add_argument("--basis-file", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--output", type=raw_output_path, required=True)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--feasibility", action="store_true")
    parser.add_argument("--profile", action="store_true")
    args = parser.parse_args()
    if not os.environ.get("SLURM_JOB_ID") or not os.environ.get("CUDA_VISIBLE_DEVICES"):
        parser.error("real-GPU execution requires a finite Slurm allocation")
    if args.repeats < 5 and not (args.feasibility and args.repeats == 1):
        parser.error("at least five pairs required unless explicitly feasibility-only")
    if args.profile and not args.feasibility:
        parser.error("intrusive profiling is a separate feasibility-only campaign")
    os.environ["GENERATIVEQC_STATIONARY_BECKE_PRIMITIVE"] = "off"
    case = scaling_cases()[f"water-{args.atoms}"]
    basis, _ = load_comparison_basis(
        args.basis_file, case, role="orbital", compute_forces=True
    )
    grid = GridSpec(radial_points=48, angular_polar=16, angular_azimuth=32)
    scientific = protocol(args.atoms, basis, grid, 5, benchmark=PBE0)
    reference = json.loads(args.reference.read_text())
    assert reference["protocol"] == scientific and reference["stage"] == "complete"
    root = Path(__file__).resolve().parents[1]
    schedule_sources = (
        "benchmarks/becke_normalize_pairs.py",
        "python/generativeqc_compiler/method/stationary_becke_phased.py",
        "python/generativeqc_compiler/xc/grid_native.py",
        "src/dft/stationary_gradient_cuda.cuh",
    )
    record = {
        "schema": "generativeqc.becke-normalize-pairs.v1",
        "protocol": scientific,
        "source_file_sha256": source_hashes()
        | {
            name: hashlib.sha256((root / name).read_bytes()).hexdigest()
            for name in schedule_sources
        },
        "reference_sha256": hashlib.sha256(args.reference.read_bytes()).hexdigest(),
        "environment": environment_metadata(accelerator=cuda_accelerator_metadata(cp)),
        "slurm_job_id": os.environ["SLURM_JOB_ID"],
        "feasibility": args.feasibility,
        "profile_intrusive": args.profile,
        "scope": "shared native post-solve snapshot, frozen using public warm-start API; primes/setup excluded and retained, not cold/moved acceleration; no opaque density byte digest",
        "samples": [],
        "priming": [],
        "setup": [],
        "assessments": {},
    }
    selection = "candidate"
    sources = []
    previous_counts: dict[int, tuple[int, int]] = {}
    force_work = None
    original_init = _stationary_cuda._CudaSources.__init__

    def configured(source: Any, *values: Any, **options: Any) -> None:
        options["becke_normalize"] = selection == "candidate"
        options["profile_device"] = args.profile
        original_init(source, *values, **options)
        sources.append(source)
        previous_counts[id(source)] = (0, 0)

    def save(stage: str) -> None:
        record["stage"] = stage
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(record, indent=2) + "\n")

    def normalize_metrics() -> list[list[int]]:
        metrics = []
        for source in sources:
            if not source.handle:
                continue
            function = source.library.stationary_becke_normalize_metrics_v1
            function.argtypes = [ct.c_void_p, ct.POINTER(ct.c_uint64), ct.c_size_t]
            output = (ct.c_uint64 * 4)()
            assert function(source.handle, output, 4) == 0
            assert output[0] == 1 and output[1] == int(selection == "candidate")
            before = previous_counts[id(source)]
            previous_counts[id(source)] = (output[2], output[3])
            delta = [output[0], output[1], output[2] - before[0], output[3] - before[1]]
            assert delta[3] == args.atoms * 48 * 16 * 32
            metrics.append(delta)
        assert metrics
        return metrics

    def diagnostics(
        item: Any, geometry: int, metrics: list[list[int]], observed_work: Any
    ) -> dict[str, Any]:
        oracle = next(
            row
            for row in reference["records"]
            if row["geometry"] == geometry and row["phase"] in ("cold", "moved")
        )
        errors = {
            "energy": block_error([item.energy], [oracle["energy"]], atol=1e-8, rtol=0),
            "forces": block_error(item.forces, oracle["forces"], atol=1e-7, rtol=0),
        }
        assert item.status == 0 and item.converged
        assert all(error["passed"] for error in errors.values())
        return {
            "energy": item.energy,
            "forces": item.forces.tolist(),
            "errors": errors,
            "iterations": item.iterations,
            "fock_builds": item.fock_builds,
            "warm_start_used": item.warm_start_used,
            "warm_start_fallback": item.warm_start_fallback,
            "native_ks_diagnostic": item.ks_diagnostic.to_payload(),
            "native_force_components": normalize_force_work(observed_work),
            "normalize_metrics": metrics,
        }

    with patch.object(_stationary_cuda._CudaSources, "__init__", configured):
        calculator = Calculator(
            method="pbe0-rks",
            basis=basis,
            device="cuda",
            basis_representation="spherical",
            ks_options=KsOptions(grid=grid),
            energy_tolerance=1e-12,
            density_tolerance=1e-10,
            screening_tolerance=1e-12,
            max_iterations=100,
        )
        record["native_build"] = native_build_metadata(calculator)
        require_tuned_native_build(record["native_build"])
        started = perf_counter()
        owner = calculator.prepare_batch(
            [scientific["geometries_bohr"][0]], warm_start=True
        )
        record["prepare_seconds"] = perf_counter() - started
        original_force = owner._public_dft_cuda_force

        def observed_force(*values: Any) -> Any:
            nonlocal force_work
            result = original_force(*values)
            force_work = result[1]
            return result

        owner._public_dft_cuda_force = observed_force
        try:
            for geometry in (0, 1):
                phase = "warm" if geometry == 0 else "moved-warm"
                coords = (
                    None
                    if geometry == 0
                    else [
                        np.asarray([xyz for _, xyz in scientific["geometries_bohr"][1]])
                    ]
                )
                owner.set_warm_start_updates(True)
                cp.cuda.Stream.null.synchronize()
                started = perf_counter()
                item = owner.execute(
                    coords, strict=False, properties=("energy", "forces")
                ).items[0]
                cp.cuda.Stream.null.synchronize()
                record["setup"].append(
                    {
                        "geometry": geometry,
                        "seconds": perf_counter() - started,
                        "diagnostics": diagnostics(
                            item, geometry, normalize_metrics(), force_work
                        ),
                    }
                )
                owner.set_warm_start_updates(False)
                completed = []

                def prepare(
                    arm: str,
                    *,
                    replay_coords: Any = coords,
                    replay_phase: str = phase,
                    replay_geometry: int = geometry,
                ) -> None:
                    nonlocal selection
                    selection = arm
                    assert owner._warm_updates is False
                    previous = owner._stationary_cuda_execution
                    if previous is not None:
                        previous.close()
                    owner._stationary_cuda_execution = None
                    sources.clear()
                    previous_counts.clear()
                    cp.cuda.Stream.null.synchronize()
                    started = perf_counter()
                    primed = owner.execute(
                        replay_coords, strict=False, properties=("energy", "forces")
                    ).items[0]
                    cp.cuda.Stream.null.synchronize()
                    record["priming"].append(
                        {
                            "phase": replay_phase,
                            "selection": arm,
                            "seconds": perf_counter() - started,
                            "diagnostics": diagnostics(
                                primed, replay_geometry, normalize_metrics(), force_work
                            ),
                        }
                    )
                    save(f"{replay_phase}/prime-{len(record['priming'])}")

                def evaluate(
                    arm: str,
                    *,
                    replay_coords: Any = coords,
                    results: list[tuple[Any, Any, list[list[int]]]] = completed,
                ) -> dict[str, int]:
                    nonlocal force_work
                    item = owner.execute(
                        replay_coords, strict=False, properties=("energy", "forces")
                    ).items[0]
                    results.append((item, force_work, normalize_metrics()))
                    return {"call_index": len(results) - 1}

                inputs_hash = canonical_hash(
                    {
                        "protocol": scientific,
                        "geometry": geometry,
                        "density": "one publicly frozen native snapshot",
                    }
                )
                if args.feasibility:
                    samples = []
                    for arm in interleaved_selection_order(1):
                        prepare(arm)
                        cp.cuda.Stream.null.synchronize()
                        started = perf_counter()
                        details = evaluate(arm)
                        cp.cuda.Stream.null.synchronize()
                        samples.append(
                            {
                                "selection": arm,
                                "seconds": perf_counter() - started,
                                "inputs_hash": inputs_hash,
                                "workload": "energy-plus-force",
                                "synchronized": True,
                                "diagnostics": details,
                            }
                        )
                else:
                    samples = measure_interleaved(
                        evaluate,
                        cp.cuda.Stream.null.synchronize,
                        workload="energy-plus-force",
                        inputs_hash=inputs_hash,
                        repeats=args.repeats,
                        prepare=prepare,
                    )
                for sample in samples:
                    selection = sample["selection"]
                    item, sample_force_work, metrics = completed[
                        sample["diagnostics"]["call_index"]
                    ]
                    sample["diagnostics"] = diagnostics(
                        item, geometry, metrics, sample_force_work
                    )
                    assert item.iterations == item.fock_builds == 1
                    assert item.warm_start_used and not item.warm_start_fallback
                    sample["phase"] = phase
                record["samples"].extend(samples)
                record["assessments"][phase] = assess_comparison(samples)
                save(phase)
        finally:
            owner.close()
    save("complete")


if __name__ == "__main__":
    main()
