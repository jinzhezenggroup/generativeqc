"""Emit or run an isolated, same-allocation ABBA Becke schedule experiment.

Run only through a finite GPU Slurm allocation. No endpoint speedup can be inferred
from these CUDA event times: AO/XC work, SCF, and integral derivatives are absent.
"""

from __future__ import annotations

import argparse
import ctypes as ct
import hashlib
import json
import os
from pathlib import Path

import numpy as np
from generativeqc_compiler.dft.grid import MolecularGrid
from generativeqc_compiler.xc.becke_coefficients import (
    emit_becke_pair_coefficients,
    plan_becke_pair_coefficients,
)
from generativeqc_compiler.xc.becke_partition import (
    emit_becke_partition_derivative,
    plan_becke_partition_derivative,
    recognize_becke_partition_derivative,
)
from generativeqc_compiler.xc.grid_native import emit_grid_adjoint, emit_grid_partials
from generativeqc_compiler.xc.grid_phased import emit_phased_becke, plan_phased_becke
from generativeqc_compiler.xc.grid_response_ir import grid_response_program

from benchmarks._support import raw_output_path
from benchmarks.readme_hf_scaling import scaling_cases


def main() -> None:
    """Keep emission host-only; require Slurm before loading the GPU probe."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--emit", type=Path)
    parser.add_argument("--library", type=Path)
    parser.add_argument("--output", type=raw_output_path)
    parser.add_argument("--atoms", type=int, nargs="+", default=[24, 48, 96])
    parser.add_argument("--tiles", type=int, default=32)
    parser.add_argument("--tile-points", type=int, nargs="+", default=[256, 1024])
    routes = parser.add_mutually_exclusive_group()
    routes.add_argument("--partition-derivative", action="store_true")
    routes.add_argument("--pair-coefficients", action="store_true")
    arguments = parser.parse_args()
    operation = recognize_becke_partition_derivative(
        ratio=grid_response_program("ratio"),
        logarithm=grid_response_program("log"),
        switch=grid_response_program("becke"),
    )
    assert operation is not None
    source = (
        (
            "#define GENERATIVEQC_BECKE_PARTITION_PROBE 1\n"
            if arguments.partition_derivative
            else "#define GENERATIVEQC_BECKE_COEFFICIENT_PROBE 1\n"
            if arguments.pair_coefficients
            else ""
        )
        + emit_grid_adjoint()
        + emit_grid_partials(3, device=True)
        + (
            emit_becke_pair_coefficients(operation)
            if arguments.pair_coefficients
            else emit_becke_partition_derivative(operation)
            if arguments.partition_derivative
            else emit_phased_becke()
        )
        + Path(__file__).with_suffix(".cu").read_text()
    )
    if arguments.emit:
        arguments.emit.write_text(source)
        return
    if not os.environ.get("SLURM_JOB_ID"):
        parser.error("real-GPU execution requires a finite Slurm GPU allocation")
    if not arguments.library or not arguments.output or arguments.tiles < 1:
        parser.error("--library, --output and positive --tiles required")
    if arguments.library.with_suffix(".cu").read_text() != source:
        parser.error("emitted source beside the library does not match this checkout")
    helper = ct.CDLL(str(arguments.library.resolve()))
    pointer = ct.POINTER(ct.c_double)
    probe = (
        helper.probe_coefficients
        if arguments.pair_coefficients
        else helper.probe_partition
        if arguments.partition_derivative
        else helper.probe
    )
    probe.argtypes = [
        ct.c_size_t,
        ct.c_size_t,
        ct.c_size_t,
        pointer,
        pointer,
        ct.POINTER(ct.c_int64),
        pointer,
        pointer,
        pointer,
        pointer,
    ]
    if arguments.partition_derivative:
        probe.argtypes += [ct.POINTER(ct.c_size_t)]
    probe.restype = ct.c_int
    report = {
        "schema": "becke-phased-isolated-v1",
        "job": os.environ["SLURM_JOB_ID"],
        "node": os.uname().nodename,
        "source_sha256": hashlib.sha256(source.encode()).hexdigest(),
        "library_sha256": hashlib.sha256(arguments.library.read_bytes()).hexdigest(),
        "timing_scope": "sampled tiles, Becke-only plus per-point publication; CUDA events; NOT endpoint",
        "order": "ABBAABBA; A=phased, B=pair-coefficients"
        if arguments.pair_coefficients
        else "ABBAABBA; A=phased, B=partition-derivative"
        if arguments.partition_derivative
        else "ABBAABBA; A=bounded cooperative, B=phased",
        "primitive_identity": operation.identity
        if arguments.partition_derivative or arguments.pair_coefficients
        else None,
        "cases": [],
    }
    for atoms in arguments.atoms:
        grid = MolecularGrid(scaling_cases()[f"water-{atoms}"].atoms)
        centers = np.ascontiguousarray(grid.centers)
        for tile_points in arguments.tile_points:
            selected = set(
                np.linspace(
                    0, (grid.npoint - 1) // tile_points, arguments.tiles, dtype=int
                )
            )
            tiles = [
                tile
                for index, tile in enumerate(grid._raw_tiles(tile_points))
                if index in selected
            ]
            points = np.ascontiguousarray(
                np.concatenate([tile.points for tile in tiles])
            )
            owners = np.ascontiguousarray(
                np.concatenate([tile.owners for tile in tiles]), dtype=np.int64
            )
            seeds = np.random.default_rng(20261004).normal(size=len(points))
            old = np.full((len(points), atoms, 3), np.nan)
            new = np.full_like(old, np.nan)
            times = np.empty(8)
            as_pointer = lambda array: array.ctypes.data_as(pointer)
            values = [
                atoms,
                len(points),
                tile_points,
                as_pointer(centers),
                as_pointer(points),
                owners.ctypes.data_as(ct.POINTER(ct.c_int64)),
                as_pointer(seeds),
                as_pointer(old),
                as_pointer(new),
                as_pointer(times),
            ]
            live_counts = np.zeros(len(points), dtype=np.uintp)
            if arguments.partition_derivative:
                values.append(live_counts.ctypes.data_as(ct.POINTER(ct.c_size_t)))
            status = probe(*values)
            if status:
                raise RuntimeError(f"probe failed: {status}")
            np.testing.assert_allclose(new, old, rtol=5e-12, atol=2e-12)
            plan = plan_phased_becke(
                atoms=atoms, points=tile_points, budget_bytes=1 << 30
            )
            assert plan is not None
            case = {
                "atoms": atoms,
                "full_grid_points": grid.npoint,
                "sample_points": len(points),
                "tile_points": tile_points,
                "tile_indices": sorted(map(int, selected)),
                "seeds": "normal(seed=20261004), NOT endpoint XC seeds",
                "milliseconds": times.tolist(),
                "baseline_median_ms": float(np.median(times[[0, 3, 4, 7]])),
                "phased_median_ms": float(np.median(times[[1, 2, 5, 6]])),
                "max_abs_difference": float(np.max(abs(new - old))),
                "phased_scratch_bytes": plan.scratch_bytes,
                "pair_evaluations_model_old": 2 * len(points) * plan.pairs
                if atoms > 32
                else len(points) * plan.pairs,
                "pair_evaluations_model_new": len(points) * plan.pairs,
                "sample_sha256": hashlib.sha256(
                    points.tobytes()
                    + centers.tobytes()
                    + owners.tobytes()
                    + seeds.tobytes()
                ).hexdigest(),
            }
            if arguments.partition_derivative:
                primitive_plan = plan_becke_partition_derivative(
                    operation=operation,
                    atoms=atoms,
                    points=tile_points,
                    budget_bytes=1 << 30,
                )
                assert primitive_plan is not None
                live = live_counts.astype(np.int64)
                reverse_visits = int(np.sum(live * atoms - live * (live + 1) // 2))
                case.update(
                    {
                        "candidate_median_ms": case.pop("phased_median_ms"),
                        "primitive_scratch_bytes": primitive_plan.scratch_bytes,
                        "pair_evaluations_model_old": len(points) * plan.pairs,
                        "reverse_pair_visits": reverse_visits,
                        "gather_pair_visits": 2 * reverse_visits,
                        "dense_reverse_pair_visits": len(points) * plan.pairs,
                        "live_atom_counts": {
                            str(int(value)): int(count)
                            for value, count in zip(
                                *np.unique(live, return_counts=True), strict=True
                            )
                        },
                        "counter_scope": "GPU-observed normalization membership; reverse/gather visits derived from exact generated domains; separate post-timing pass",
                        "memory_traffic": "logical scratch/visit model only; cache-level hardware bytes not measured",
                    }
                )
            if arguments.pair_coefficients:
                coefficient_plan = plan_becke_pair_coefficients(
                    operation=operation,
                    atoms=atoms,
                    points=tile_points,
                    budget_bytes=1 << 30,
                    cached_geometry=True,
                )
                assert coefficient_plan is not None
                reverse_visits = len(points) * plan.pairs
                case["candidate_median_ms"] = case.pop("phased_median_ms")
                case.update(
                    {
                        "coefficient_scratch_bytes": coefficient_plan.scratch_bytes,
                        "pair_evaluations_model_old": reverse_visits,
                        "pair_evaluations_model_new": reverse_visits,
                        "reverse_pair_visits": reverse_visits,
                        "gather_pair_visits": 2 * reverse_visits,
                        "reverse_write_bytes_model_old": 4 * 8 * reverse_visits,
                        "reverse_write_bytes_model_new": 2 * 8 * reverse_visits,
                        "gather_pair_read_bytes_model_old": 2 * 4 * 8 * reverse_visits,
                        "gather_pair_read_bytes_model_new": 2 * 2 * 8 * reverse_visits,
                        "gather_geometry_read_bytes_model_new": 2
                        * 3
                        * 8
                        * reverse_visits,
                        "counter_scope": "dense generated pair/ordered-gather domains; work counts derived from sampled points, not measured hardware counters",
                        "memory_traffic": "logical scratch/visit model; additional gather center metadata included separately; cache-level hardware bytes not measured",
                    }
                )
            report["cases"].append(case)
            arguments.output.write_text(json.dumps(report, indent=2) + "\n")
            print(json.dumps(case), flush=True)


if __name__ == "__main__":
    main()
