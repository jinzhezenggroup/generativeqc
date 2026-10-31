"""Compare CPU HF transfer endpoints using separately built source checkouts."""

import argparse
import hashlib
import json
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path


def worker(samples: int) -> None:
    import numpy as np
    from generativeqc import Calculator, ObservableTarget, TargetAccuracy
    from generativeqc.progressive_controller import (
        TargetProblem,
        make_deterministic_hf_plan,
        run_progressive_hf,
    )

    accuracy = TargetAccuracy(
        (
            ObservableTarget("energy", "absolute", "Eh", absolute=1e-8),
            ObservableTarget("forces", "max_abs", "Eh/bohr", absolute=1e-7),
        )
    )
    records = []
    for method, charge, multiplicity in [("rhf", 0, 1), ("uhf", 1, 2)]:
        for fitted in [False, True]:
            for sample in range(-3, samples):
                # Both geometries are timed, with fresh owners and no reused seed.
                atoms = (
                    ("H", (0.0, 0.0, -0.7)),
                    ("H", (0.1, 0.0, 0.7 + 0.02 * (sample % 2))),
                )
                started = time.perf_counter()
                source = Calculator(
                    basis="sto-3g",
                    method=method,
                    max_iterations=20,
                    energy_tolerance=1e-7,
                    density_tolerance=1e-6,
                )
                target = Calculator(
                    basis="def2-svp",
                    method=method,
                    density_fitting="cpu" if fitted else "none",
                    max_iterations=100,
                    energy_tolerance=1e-12,
                    density_tolerance=1e-10,
                    screening_tolerance=1e-14,
                    target_accuracy=accuracy,
                )
                problem = TargetProblem.from_calculator(
                    target, atoms, charge=charge, multiplicity=multiplicity
                )
                plan = make_deterministic_hf_plan(
                    problem,
                    source,
                    target,
                    atoms,
                    charge=charge,
                    multiplicity=multiplicity,
                )
                result = run_progressive_hf(
                    plan,
                    source,
                    target,
                    atoms,
                    charge=charge,
                    multiplicity=multiplicity,
                )
                seconds = time.perf_counter() - started
                assert (
                    result.verification.target_established and result.target.converged
                )
                assert result.diagnostics["execution_owner"] == "native-v1"
                signature = {
                    "source_energy": result.source.energy,
                    "energy": result.target.energy,
                    "forces": np.asarray(result.target.forces).tolist(),
                    "density": np.asarray(result.target_density).tolist(),
                    "iterations": [x.iterations for x in result.executions],
                    "fock_builds": [x.fock_builds for x in result.executions],
                    "projection": result.diagnostics["projection"]["status"],
                    "restart": result.target.restart_origin,
                    "residual": result.verification.physical_residual_max,
                    "status": result.verification.status,
                }
                if sample >= 0:
                    records.append(
                        {
                            "case": f"{method}-{fitted}",
                            "sample": sample,
                            "seconds": seconds,
                            "signature": signature,
                        }
                    )
    for fitted in [False, True]:
        for sample in range(-3, samples):
            started = time.perf_counter()
            signature = fleet_endpoint(fitted)
            seconds = time.perf_counter() - started
            if sample >= 0:
                records.append(dict(case=f"ragged-rhf-{fitted}", sample=sample,
                                    seconds=seconds, signature=signature))
    print(json.dumps(records))


def fleet_endpoint(fitted: bool) -> dict:
    """Time setup, projection, cold/warm/moved E/F, density receipts and close."""
    import ctypes as ct
    import numpy as np
    from generativeqc import Calculator, _native
    from generativeqc.checkpoint import _descriptor, _pointer

    helium = [("He", (0., 0., 0.))]
    hydrogen = [("H", (0., 0., -.7)), ("H", (.1, 0., .7))]
    systems = [hydrogen, helium]
    options = dict(max_iterations=100, energy_tolerance=1e-12,
                   density_tolerance=1e-10, screening_tolerance=1e-14)

    def receipt(batch, result):
        rows = []
        for index, item in enumerate(result.items):
            assert item.succeeded
            state = _descriptor()
            get = batch._library.generativeqc_batch_get_hf_warm_state
            _native.check(batch._library, get(batch._batch, index, ct.byref(state)))
            density, xyz = np.empty(state.density_count), np.empty(state.coordinate_count)
            state.density, state.coordinates = _pointer(density), _pointer(xyz)
            _native.check(batch._library, get(batch._batch, index, ct.byref(state)))
            rows.append(dict(energy=item.energy, forces=np.asarray(item.forces).tolist(),
                             density=density.tolist(), coordinates=xyz.tolist(),
                             energy_change=item.energy_change, density_rms=item.density_rms,
                             physical_residual_rms=item.physical_residual_rms,
                             iterations=item.iterations, fock_builds=item.fock_builds,
                             restart=item.restart_origin))
        return rows

    with Calculator(**options).prepare_batch(systems) as source, Calculator(
        basis="def2-svp", density_fitting="cpu" if fitted else "none", **options
    ).prepare_batch(systems) as target:
        source_cold = receipt(source, source.execute(strict=True))
        projection = target.initialize_from(source)
        assert all(row["accepted"] for row in projection["items"])
        cold = receipt(target, target.execute(strict=True))
        source_warm = receipt(source, source.execute(strict=True))
        warm = receipt(target, target.execute(strict=True))
        moved = [np.array([atom[1] for atom in atoms]) for atoms in systems]
        for xyz in moved:
            xyz[-1, 2] += 0.035
        source_moved = receipt(source, source.execute(coordinates=moved, strict=True))
        target_moved = receipt(target, target.execute(coordinates=moved, strict=True))
        return dict(source_cold=source_cold, target_cold=cold,
                    source_warm=source_warm, target_warm=warm,
                    source_moved=source_moved, target_moved=target_moved)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--samples", type=int, default=21)
    parser.add_argument("--baseline", type=Path)
    parser.add_argument("--candidate", type=Path)
    parser.add_argument("--baseline-library", type=Path)
    parser.add_argument("--candidate-library", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.worker:
        worker(args.samples)
        return
    records = []
    for order in ("ABBA", "BAAB"):
        for block, variant in enumerate(order):
            checkout = args.baseline if variant == "A" else args.candidate
            env = {
                **os.environ,
                "PYTHONPATH": str(checkout / "python") + os.pathsep + str(checkout),
                "GENERATIVEQC_LIBRARY": str(args.baseline_library if variant == "A" else args.candidate_library),
                "OMP_NUM_THREADS": "1",
                "OPENBLAS_NUM_THREADS": "1",
            }
            result = subprocess.run(
                [
                    sys.executable,
                    str(Path(__file__).resolve()),
                    "--worker",
                    "--samples",
                    str(args.samples),
                ],
                env=env,
                cwd=checkout,
                capture_output=True,
                text=True,
                check=True,
                timeout=1800,
            )
            for row in json.loads(result.stdout):
                records.append(dict(order=order, block=block, variant=variant, **row))
            args.output.with_suffix(".partial.json").write_text(json.dumps(records))
            print(order, block, variant, "complete", flush=True)
    summary = []
    signature_failures = []
    for order in ("ABBA", "BAAB"):
        for case in sorted({row["case"] for row in records}):
            group = [
                row for row in records if row["order"] == order and row["case"] == case
            ]
            # Match geometry/sample identities and all published scientific/work outputs.
            for index in range(args.samples):
                signatures = [
                    row["signature"] for row in group if row["sample"] == index
                ]
                if not all(item == signatures[0] for item in signatures):
                    signature_failures.append(dict(order=order, case=case, sample=index))
            medians = {
                variant: statistics.median(
                    row["seconds"] for row in group if row["variant"] == variant
                )
                for variant in ("A", "B")
            }
            summary.append(
                {
                    "order": order,
                    "case": case,
                    "baseline": medians["A"],
                    "candidate": medians["B"],
                    "ratio": medians["B"] / medians["A"],
                    "variability": {
                        variant: {
                            "minimum": min(row["seconds"] for row in group if row["variant"] == variant),
                            "maximum": max(row["seconds"] for row in group if row["variant"] == variant),
                            "relative_mad": statistics.median(abs(row["seconds"] - medians[variant]) for row in group if row["variant"] == variant) / medians[variant],
                        }
                        for variant in ("A", "B")
                    },
                }
            )
    source_paths = [
        "python/generativeqc/progressive.py",
        "python/generativeqc/_fleet_projection_native.py",
        "src/api/c_api_fleet_projection.cpp",
        "src/scf/fleet_projection.cpp", "src/scf/fleet_projection.hpp",
        "src/scf/fleet.cpp", "src/scf/fleet.hpp",
        "src/methods/hf_method.cpp", "src/methods/batch_capabilities.hpp",
        "include/generativeqc/generativeqc.h",
    ]
    output = {
        "baseline_sha": subprocess.check_output(
            ["git", "-C", str(args.baseline), "rev-parse", "HEAD"], text=True
        ).strip(),
        "candidate_base": subprocess.check_output(
            ["git", "-C", str(args.candidate), "rev-parse", "HEAD"], text=True
        ).strip(),
        "source_sha256": {
            name: hashlib.sha256((args.candidate / name).read_bytes()).hexdigest()
            for name in source_paths
        },
        "library_sha256": {
            variant: hashlib.sha256(path.read_bytes()).hexdigest()
            for variant, path in [("baseline", args.baseline_library),
                                  ("candidate", args.candidate_library)]
        },
        "python": sys.version,
        "scope": "separate matched Release CPU libraries; progressive source/projection/target E+F/audit/cleanup and ragged H2+He cold/warm/moved E+F+density/cleanup; imports excluded; no GPU",
        "signature_failures": signature_failures,
        "samples": records,
        "summary": summary,
    }
    args.output.write_text(json.dumps(output))
    print(json.dumps(summary, indent=2))
    assert not signature_failures, signature_failures
    assert max(row["ratio"] for row in summary) <= 1.05


if __name__ == "__main__":
    main()
