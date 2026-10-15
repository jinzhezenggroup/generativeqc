"""Bounded meta-GGA occupied-feature crossover experiment (not a KS benchmark).

Compare the existing CUDA grid density-matrix and occupied-orbital feature
producers at the same AO, density, points, and active-AO map. This does NOT
install an occupied route in CudaKsPlan, or benchmark full M06-2X E+F.

On a CUDA build node with a compiler cache, inside a finite Slurm allocation:
  PYTHONPATH=.:python python benchmarks/mgga_occupied_crossover.py \
    --replicas 1 16 32 --output .artifacts/mgga-occupied-crossover.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import median
from time import perf_counter

import numpy as np


def projection_work(active: int, occupied: int, points: int) -> dict[str, int]:
    """Arithmetic counts, not GPU timing estimates; Vxc and setup excluded."""
    if any(type(x) is not int or x < 0 for x in (active, occupied, points)):
        raise ValueError("nonnegative integral AO, rank and point counts required")
    return {
        "four_density_jets": 4 * points * active * active,
        "one_density_three_occupied": points
        * (active * active + 3 * active * occupied),
        "four_occupied_jets": 4 * points * active * occupied,
    }


def hybrid_features(jets, density, factor, ids=None):
    """Independent FP64 D*phi + three B^T*d(phi) prototype for one spin.

    B includes sqrt(occupations); D=B@B.T is required. The density witness is
    verified here, outside performance timing. XC Vxc still uses four AO jets.
    """
    jets = np.asarray(jets, dtype=np.float64)
    d = np.asarray(density, dtype=np.float64)
    b = np.asarray(factor, dtype=np.float64)
    if d.ndim != 2 or d.shape[0] != d.shape[1] or b.ndim != 2 or b.shape[0] != len(d):
        raise ValueError("invalid density or occupied-factor dimensions")
    if jets.ndim != 3 or jets.shape[0] != 4:
        raise ValueError("four AO jets required")
    if ids is None:
        ids = np.arange(len(d))
    else:
        ids = np.asarray(ids)
        if (
            ids.ndim != 1
            or ids.dtype.kind not in "iu"
            or np.any(ids < 0)
            or np.any(ids >= len(d))
            or np.any(ids[1:] <= ids[:-1])
        ):
            raise ValueError("AO map must be sorted, unique and in bounds")
    if jets.shape[2] != len(ids):
        raise ValueError("AO map and jets disagree")
    if not all(np.isfinite(x).all() for x in (jets, d, b)):
        raise ValueError("nonfinite XC input")
    if not np.allclose(d, b @ b.T, rtol=1e-12, atol=1e-13):
        raise ValueError("stale/incompatible occupied density witness")
    phi = jets[0]
    dphi = jets[1:4]
    dphi0 = phi @ d[np.ix_(ids, ids)]
    projected = np.einsum("kpi,io->kpo", dphi, b[ids], optimize=True)
    return {
        "rho": np.einsum("pi,pi->p", phi, dphi0),
        "gradient": 2 * np.einsum("kpi,pi->pk", dphi, dphi0),
        "tau": 0.5 * np.einsum("kpo,kpo->p", projected, projected),
    }


def water_atoms(replicas: int):
    if type(replicas) is not int or not (1 <= replicas <= 64):
        raise ValueError("replicas must be 1..64")
    local = ((8, (0.0, 0.0, 0.0)), (1, (1.4, 0.1, 1.1)), (1, (-1.2, 0.2, 1.3)))
    return [
        (z, (x + 12 * (i % 8), y + 12 * (i // 8), h))
        for i in range(replicas)
        for z, (x, y, h) in local
    ]


def active_ids(ao, cutoff=1e-16):
    if ao.ndim != 3 or ao.shape[0] < 4 or not np.isfinite(ao).all():
        raise ValueError("finite four-jet AO tile required")
    return np.flatnonzero(np.any(np.abs(ao[:4]) > cutoff, axis=(0, 1)))


def benchmark(args):
    from generativeqc_compiler.common.cuda_adapter import CudaCompilerAdapter
    from generativeqc_compiler.common.cuda_target import cuda_target_info
    from generativeqc_compiler.common.provenance import find_nvcc
    from generativeqc_compiler.dft import DensitySource, NativeAO
    from generativeqc_compiler.dft.cuda import CudaGrid, compile_cuda

    nvcc = find_nvcc()
    if nvcc is None:
        raise RuntimeError("CUDA compiler unavailable: requires a GPU build node")
    artifact = compile_cuda(
        CudaCompilerAdapter(nvcc, cuda_target_info(args.arch)), Path(args.cache)
    )
    rng = np.random.default_rng(20261015)
    report = {
        "schema": "generativeqc.mgga-occupied-feature-crossover.v1",
        "scope": "fixed-density features only, NOT full E+F or the proposed hybrid GPU kernel",
        "cuda_candidate": "existing four-occupied-jet grid path",
        "cuda_arch": args.arch,
        "artifact_sha256": artifact.metadata["binary_sha256"],
        "cases": [],
    }
    for replicas in args.replicas:
        atoms = water_atoms(replicas)
        with NativeAO(
            atoms, basis="def2-svp", representation="real_spherical"
        ) as basis:
            n, rank = basis.nao, 5 * replicas
            if rank > n:
                raise RuntimeError("occupied rank exceeds AO count")
            # Synthetic PSD density with a guaranteed factor witness. It is not
            # a converged molecular SCF state or a physical force reference.
            factor = rng.normal(size=(n, rank)) / np.sqrt(n)
            d = factor @ factor.T
            source = DensitySource(
                np.stack((d, d)), basis_identity=basis.identity
            ).with_orbitals(
                (factor, factor), (np.ones(rank), np.ones(rank)), validation_rows=3
            )
            if source.source_kind != "orbitals":
                raise RuntimeError("factor witness rejected by existing source API")
            center = np.repeat(
                np.asarray([xyz for _, xyz in atoms]), args.points_per_atom, axis=0
            )
            points = np.ascontiguousarray(
                center + rng.normal(scale=0.7, size=center.shape)
            )
            tiles = [
                points[i : i + args.tile_points]
                for i in range(0, len(points), args.tile_points)
            ]
            selected = [active_ids(basis.evaluate(tile, order=1)) for tile in tiles]
            for mode in ("dense", "indexed"):
                maps = [None] * len(tiles) if mode == "dense" else selected
                counts = [n if ids is None else len(ids) for ids in maps]
                samples = {"density_matrix": [], "orbitals": []}
                max_error = {"rho": 0.0, "gradient": 0.0, "tau": 0.0}
                source_stats = {}
                with CudaGrid(
                    basis,
                    artifact,
                    order=1,
                    tile_points=args.tile_points,
                    active_ao_capacity=n if mode == "indexed" else None,
                    orbital_capacity=(rank, rank),
                    orbital_tile=args.orbital_tile,
                    features_only=True,
                    ingredients=("rho", "gradient", "tau"),
                ) as cuda:
                    for repeat in range(args.warmups + args.repeats):
                        # Alternating route order; uploads and AO-map
                        # preparation are outside the measured feature calls.
                        order = (
                            ("density_matrix", "orbitals")
                            if repeat % 2 == 0
                            else ("orbitals", "density_matrix")
                        )
                        for route in order:
                            cuda.set_source(source, stamp=source.stamp, route=route)
                            if cuda.source_kind != route:
                                raise RuntimeError("requested CUDA source not executed")
                            source_stats[route] = cuda.source_statistics
                            begin = perf_counter()
                            outputs = [
                                cuda.evaluate(tile, ao_ids=ids, stamp=source.stamp)
                                for tile, ids in zip(tiles, maps, strict=True)
                            ]
                            elapsed = perf_counter() - begin
                            if repeat == 0:
                                # Check the exact active maps against the hybrid
                                # expression; this work is outside all timers.
                                for tile, ids, output in zip(
                                    tiles, maps, outputs, strict=True
                                ):
                                    ao = basis.evaluate(tile, order=1)
                                    if ids is not None:
                                        ao = ao[:, :, ids]
                                    reference = hybrid_features(ao, d, factor, ids)
                                    for spin in range(2):
                                        for key, expected in reference.items():
                                            actual = output[key][spin]
                                            err = float(
                                                np.max(
                                                    np.abs(actual - expected),
                                                    initial=0.0,
                                                )
                                            )
                                            max_error[key] = max(max_error[key], err)
                                            np.testing.assert_allclose(
                                                actual, expected, rtol=2e-9, atol=2e-10
                                            )
                            if repeat >= args.warmups:
                                samples[route].append(elapsed)
                    times = {route: median(values) for route, values in samples.items()}
                    work = [
                        projection_work(a, rank, len(tile))
                        for a, tile in zip(counts, tiles, strict=True)
                    ]
                    report["cases"].append(
                        {
                            "atoms": len(atoms),
                            "ao": n,
                            "occupied_per_spin": rank,
                            "points": len(points),
                            "mode": mode,
                            "active_ao_per_tile": counts,
                            "timings_seconds": samples,
                            "medians_seconds": times,
                            "orbital_over_density_ratio": times["orbitals"]
                            / times["density_matrix"],
                            "max_abs_hybrid_reference_error": max_error,
                            "product_counts": {
                                key: sum(v[key] for v in work) for key in work[0]
                            },
                            "source_uploads_excluded_from_timing": source_stats,
                            "cuda_versions": {
                                key: cuda.metrics()[key]
                                for key in (
                                    "runtime_version",
                                    "driver_version",
                                    "cublas_version",
                                )
                            },
                        }
                    )
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replicas", type=int, nargs="+", default=[1, 16, 32])
    parser.add_argument("--arch", default="sm_120")
    parser.add_argument("--tile-points", type=int, default=64)
    parser.add_argument("--points-per-atom", type=int, default=16)
    parser.add_argument("--orbital-tile", type=int, default=32)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--cache", default=".artifacts/mgga-occupied-grid-cache")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if (
        min(args.tile_points, args.points_per_atom, args.orbital_tile, args.warmups) < 1
        or args.repeats < 2
    ):
        parser.error("positive tile/point/priming dimensions and >=2 repeats required")
    result = json.dumps(benchmark(args), indent=2, sort_keys=True)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(result + "\n", encoding="utf-8")
    else:
        print(result)


if __name__ == "__main__":
    main()
