"""Small CPU scientific screen of CCSD-history Lambda preconditioning.

Run with PYTHONPATH=python:. and one BLAS thread. This uses pinned molecular
Hamiltonians, the generated physical CCSD equations, and the existing GMRES.
It is NOT CUDA DF/CBS E+F qualification: no Q-batch or GPU performance claim is
possible here. No native endpoint option is changed by running this experiment.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
from generativeqc.response_solver import GMRESOptions, solve
from generativeqc_compiler.cc.equations import amplitude_layouts
from generativeqc_compiler.cc.lambda_equations import build_lambda_programs
from generativeqc_compiler.cc.pair_coordinates import RestrictedPairCoordinates
from generativeqc_compiler.tensor import execute

from benchmarks._support import raw_output_path
from benchmarks.cc_lambda_multisecant_native import NativeGMRES
from tools.cc_endpoint_fixtures import load
from tools.generativeqc_cc.multisecant import SecantHistory, fit_multisecant
from tools.generativeqc_cc.oracle import dense_feeds
from tools.generativeqc_cc.solver import _DIIS


class Coordinates:
    """Small compiler-enumerated sqrt-weighted Lambda coordinates.

    This map deliberately does not infer the CUDA DIIS ring's ordering. A
    future native exporter must explicitly map its orbit slots to this order.
    """

    def __init__(self, occupied: int, virtual: int) -> None:
        self.layouts = amplitude_layouts(occupied, virtual)
        self.weights = np.sqrt(
            np.concatenate([np.asarray(p.weights) for p in self.layouts])
        )
        self.dimension = len(self.weights)
        self.diis_slots = np.arange(self.dimension)
        pair_coordinates = RestrictedPairCoordinates(occupied, virtual)
        singles = self.layouts[0].size
        for k, flat in enumerate(self.layouts[1].representatives):
            self.diis_slots[singles + k] = singles + pair_coordinates.orbit(flat)[1]

    def pack(self, one: np.ndarray, two: np.ndarray) -> np.ndarray:
        """Pack symmetric amplitudes or physical residuals isometrically."""
        return self.weights * np.concatenate(
            [p.pack(t) for p, t in zip(self.layouts, (one, two), strict=True)]
        )

    def unpack(self, values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Restore full pair-symmetric tensors from Euclidean coordinates."""
        raw = values / self.weights
        split = self.layouts[0].size
        return self.layouts[0].unpack(raw[:split]), self.layouts[1].unpack(raw[split:])

    def project_cotangent(self, one: np.ndarray, two: np.ndarray) -> np.ndarray:
        """Project an ordered dense cotangent, preserving its action on t."""
        return self.weights * np.concatenate(
            [
                p.unpack_transpose(t)
                for p, t in zip(self.layouts, (one, two), strict=True)
            ]
        )

    def from_diis_packed(self, values: np.ndarray) -> np.ndarray:
        """Convert one detached native unweighted ring row to Lambda ordering.

        CUDA DIIS triangularizes x=i*v+a,y=j*v+b; Lambda enumerates dense ijab
        representatives. Orbit multiplicity alone does not fix this permutation.
        This small reference adapter is not a device exporter or a large-system
        planner; it reuses the compiler's coordinate owner without rederiving it.
        """
        if (
            values.shape != (self.dimension,)
            or values.dtype != np.float64
            or not np.isfinite(values).all()
        ):
            raise ValueError("invalid native packed DIIS row")
        return self.weights * values[self.diis_slots]


def converge(
    feeds: dict[str, np.ndarray],
    denominators: np.ndarray,
    coordinates: Coordinates,
    programs: Any,
    *,
    epoch: str,
    damping: float = 0.0,
    history_rank: int = 6,
) -> tuple[dict[str, np.ndarray], SecantHistory, dict[str, Any]]:
    """Screen the native Anderson pairing with generated FP64 equations.

    Only numerical iteration control is reproduced here: G=t+(1-d)R/D and
    DIIS(G,R). The scientific residual/energy always comes from TensorIR.
    Sampling adds no residual evaluations. The final physical point supplies
    the seventh state needed for rank six with a six-entry mixing ring.
    """
    history = SecantHistory(coordinates.dimension, history_rank, max_bytes=64 << 20)
    mixer = _DIIS(6)
    current = coordinates.pack(feeds["t1"], feeds["t2"])
    previous_energy = None
    capture_seconds = 0.0
    started = time.perf_counter()
    for iteration in range(151):
        feeds = {
            **feeds,
            **dict(zip(("t1", "t2"), coordinates.unpack(current), strict=True)),
        }
        output = execute(programs.primal, feeds).outputs
        energy = float(output["correlation_energy"])
        residual = coordinates.pack(
            output["singles_residual"], output["doubles_residual"]
        )
        physical_max = float(np.max(np.abs(residual / coordinates.weights)))
        if (
            previous_energy is not None
            and abs(energy - previous_energy) < 1e-13
            and physical_max < 1e-11
        ):
            clock = time.perf_counter()
            history.append_pair(current, residual, epoch=epoch, precision="fp64")
            capture_seconds += time.perf_counter() - clock
            return (
                feeds,
                history,
                {
                    "iterations": iteration,
                    "residual_evaluations": iteration + 1,
                    "physical_residual_max": physical_max,
                    "correlation_energy": energy,
                    "seconds_including_capture": time.perf_counter() - started,
                    "capture_seconds": capture_seconds,
                    "history_samples": history.samples,
                    "history_retained_bytes": history.retained_bytes,
                },
            )
        image = current + (1 - damping) * residual / denominators
        clock = time.perf_counter()
        history.append_jacobi(
            image, residual, denominators, damping, epoch=epoch, precision="fp64"
        )
        capture_seconds += time.perf_counter() - clock
        current = mixer.update(image, residual)
        previous_energy = energy
    raise RuntimeError("screening CCSD did not converge")


def transpose_action(
    programs: Any,
    feeds: dict[str, np.ndarray],
    coordinates: Coordinates,
    vector: np.ndarray,
) -> np.ndarray:
    """Execute the real generated J.T action, independent of the inverse fit."""
    one, two = coordinates.unpack(vector)
    result = execute(
        programs.residual_vjp.program,
        {
            **feeds,
            "bar_singles_residual": one,
            "bar_doubles_residual": two,
        },
    ).outputs
    return coordinates.pack(result["bar_t1"], result["bar_t2"])


def lambda_rhs(
    programs: Any,
    feeds: dict[str, np.ndarray],
    coordinates: Coordinates,
) -> np.ndarray:
    """Return -dE_CCSD/dt in the same weighted coordinates as J.T."""
    result = execute(
        programs.energy_vjp.program, {**feeds, "bar_correlation_energy": np.array(-1.0)}
    ).outputs
    return coordinates.pack(result["bar_t1"], result["bar_t2"])


def screen_case(
    name: str,
    *,
    repetitions: int = 2,
    triples: bool = False,
    native_gmres: NativeGMRES | None = None,
) -> dict[str, Any]:
    """A/B rank 0/4/6 at the same accepted CC root and unchanged residual gates."""
    meta, arrays = load(name)
    o, v = arrays["t1"].shape
    coordinates = Coordinates(o, v)
    eps = arrays["eps"]
    d1 = eps[:o, None] - eps[None, o:]
    d2 = d1[:, None, :, None] + d1[None, :, None, :]
    # Denominators are diagonal entries, so their coordinate weights CANCEL.
    denominators = coordinates.pack(d1, d2) / coordinates.weights
    fock = arrays["C"].T @ arrays["F"] @ arrays["C"]
    initial1 = fock[:o, o:] / d1
    initial2 = arrays["g"][:o, o:, :o, o:].transpose(0, 2, 1, 3) / d2
    feeds = dense_feeds(fock, arrays["g"], initial1, initial2)
    generation_started = time.perf_counter()
    shared = build_lambda_programs(o, v)
    independent = build_lambda_programs(o, v, form="expanded")
    generation_seconds = time.perf_counter() - generation_started
    epoch = meta["arrays_hash"] + ":fp64:damping0:lambda-sqrt-pairs"
    feeds, history, cc = converge(feeds, denominators, coordinates, shared, epoch=epoch)
    errors = {
        "energy": abs(cc["correlation_energy"] - meta["correlation_energy"]),
        "t1_max": float(np.max(np.abs(feeds["t1"] - arrays["t1"]))),
        "t2_max": float(np.max(np.abs(feeds["t2"] - arrays["t2"]))),
    }
    if max(errors.values()) > 1e-8:
        raise RuntimeError(f"pinned independent CCSD oracle failed: {errors}")
    source_started = time.perf_counter()
    triples_source = np.zeros(coordinates.dimension)
    if triples:
        from tools.generativeqc_cc.triples_response import accumulate_tile_triples_vjp

        sources = accumulate_tile_triples_vjp(
            o,
            v,
            *(feeds[key] for key in ("ovvv", "ovoo", "ovov", "fov", "t1", "t2")),
            eps[:o],
            eps[o:],
            inputs=("t1", "t2"),
        )
        # Cotangents of the ordered (T) graph are not necessarily symmetric.
        # Project using the compiler layout's adjoint, exactly as the existing
        # corrected-Lambda tooling does; ordinary pack would discard terms.
        triples_source = coordinates.project_cotangent(sources["t1"], sources["t2"])
    source_seconds = time.perf_counter() - source_started
    observations = []
    baseline = None
    for repetition in range(repetitions):
        for rank in (0, 4, 6) if repetition % 2 == 0 else (6, 4, 0):
            started = time.perf_counter()
            audit = execute(independent.primal, feeds).outputs
            primal_max = max(
                float(np.max(np.abs(audit[key])))
                for key in ("singles_residual", "doubles_residual")
            )
            if primal_max > 1e-9:
                raise RuntimeError("independent physical CC residual failed")
            rhs = lambda_rhs(shared, feeds, coordinates) - triples_source
            steps, changes = history.differences(epoch=epoch, rank=rank)
            preconditioner = fit_multisecant(
                -1.0 / denominators,
                steps,
                changes,
                max_rank=rank,
                max_bytes=64 << 20,
            )
            operator = SimpleNamespace(
                dimension=coordinates.dimension,
                apply=lambda vector: transpose_action(
                    shared, feeds, coordinates, vector
                ),
            )
            if native_gmres is not None:
                result = native_gmres.solve(operator.apply, preconditioner.apply, rhs)
            else:
                # The Python controller has no predicted-convergence trigger
                # between periodic checks. Check each step in this fallback so
                # a 30-step interval cannot conceal a preconditioner's benefit.
                result = solve(
                    operator,
                    rhs,
                    options=GMRESOptions(
                        atol=1e-12,
                        rtol=0.0,
                        max_iterations=150,
                        true_residual_every=1,
                    ),
                    preconditioner=preconditioner,
                    raise_on_failure=True,
                    collect_basis=False,
                )
            residual = transpose_action(
                independent, feeds, coordinates, result.solution
            )
            residual -= lambda_rhs(independent, feeds, coordinates) - triples_source
            independent_norm = float(np.linalg.norm(residual))
            if independent_norm > 1e-9:
                raise RuntimeError("independent physical Lambda residual failed")
            if baseline is None:
                baseline = result.solution.copy()
            solution_error = float(np.max(np.abs(result.solution - baseline)))
            if solution_error > 1e-9:
                raise RuntimeError(
                    "preconditioning changed the accepted Lambda solution"
                )
            lambda_seconds = time.perf_counter() - started
            # Q is the range of Y. For H.T, correction.T probes the input and
            # Q supplies the output. RHS projection alone cannot certify that
            # the history covers the difficult Krylov subspace.
            q = preconditioner.basis
            observations.append(
                {
                    "repetition": repetition,
                    "requested_rank": rank,
                    "accepted_rank": preconditioner.rank,
                    "fit_reason": preconditioner.reason,
                    "iterations": result.iterations,
                    "gmres_operator_actions": result.operator_actions,
                    "total_transpose_actions": result.operator_actions + 1,
                    "preconditioner_actions": result.preconditioner_actions,
                    "true_residual_norm": result.residual_norm,
                    "independent_residual_norm": independent_norm,
                    "solution_max_change": solution_error,
                    "lambda_seconds_including_fit_and_audits": lambda_seconds,
                    "lambda_plus_capture_seconds": lambda_seconds
                    + (cc["capture_seconds"] if rank else 0.0),
                    "fit_seconds": preconditioner.build_seconds,
                    "fit_workspace_bound_bytes": preconditioner.workspace_bound_bytes,
                    "preconditioner_retained_host_bytes": preconditioner.retained_bytes,
                    "additional_device_bytes": 0,
                    "rhs_projection_on_y_fraction": float(
                        np.linalg.norm(q.T @ rhs) / np.linalg.norm(rhs)
                    ),
                    "singular_values": preconditioner.singular_values,
                }
            )
    return {
        "case": name,
        "method": "CCSD(T)" if triples else "CCSD",
        "nocc": o,
        "nvir": v,
        "dimension": coordinates.dimension,
        "fixture_arrays_hash": meta["arrays_hash"],
        "gmres_controller": "native" if native_gmres is not None else "python",
        "true_residual_every": 30 if native_gmres is not None else 1,
        "program_generation_seconds": generation_seconds,
        "program_hashes": {
            "primal": shared.primal.logical_hash,
            "rhs": shared.energy_vjp.program.logical_hash,
            "transpose": shared.residual_vjp.program.logical_hash,
            "independent_primal": independent.primal.logical_hash,
            "independent_transpose": independent.residual_vjp.program.logical_hash,
        },
        "triples_source_seconds": source_seconds,
        "cc": cc,
        "pinned_oracle_errors": errors,
        "observations": observations,
    }


def main() -> None:
    """Write reproducible screen evidence without claiming native qualification."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cases",
        nargs="+",
        choices=("h2", "h2o", "nh3", "ch4"),
        default=["h2o", "nh3", "ch4"],
    )
    parser.add_argument("--repetitions", type=int, default=2)
    parser.add_argument("--triples", action="store_true")
    parser.add_argument(
        "--native-gmres", type=Path, help="prebuilt cc_lambda_multisecant_gmres bridge"
    )
    parser.add_argument("--output", type=raw_output_path, required=True)
    args = parser.parse_args()
    if args.repetitions < 1:
        parser.error("repetitions must be positive")
    rows = []
    native = NativeGMRES(args.native_gmres) if args.native_gmres else None
    for case in args.cases:
        row = screen_case(
            case,
            repetitions=args.repetitions,
            triples=args.triples,
            native_gmres=native,
        )
        rows.append(row)
        print(
            case,
            [
                (x["requested_rank"], x["accepted_rank"], x["gmres_operator_actions"])
                for x in row["observations"]
            ],
            flush=True,
        )
    root = Path(__file__).resolve().parents[1]
    sources = [
        Path(__file__),
        root / "tools/generativeqc_cc/multisecant.py",
        root / "benchmarks/cc_lambda_multisecant_native.py",
        root / "benchmarks/cc_lambda_multisecant_gmres.cpp",
        root / "src/response/native_gmres.cpp",
        root / "src/response/native_gmres.hpp",
    ]
    payload = {
        "schema": "generativeqc.cc-lambda-multisecant.screen.v1",
        "backend": "numpy-fp64-generated-tensorir",
        "scope": "supplied-Hamiltonian CPU numerical screen",
        "python_version": platform.python_version(),
        "numpy_version": np.__version__,
        "platform": platform.platform(),
        "thread_environment": {
            key: os.environ.get(key)
            for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")
        },
        "native_gpu_qualified": False,
        "cbs_energy_force_qualified": False,
        "q_batch": None,
        "performance_promotion": False,
        "source_sha256": {
            str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sources
        },
        "native_gmres_sha256": hashlib.sha256(
            args.native_gmres.read_bytes()
        ).hexdigest()
        if args.native_gmres
        else None,
        "cases": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
