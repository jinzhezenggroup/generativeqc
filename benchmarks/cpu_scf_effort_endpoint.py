"""Retain complete CPU force-aware SCF trajectories with matching source/library evidence.

Without --calibration, write independently measured calibration from the existing
research harness. With that receipt, time construction through strict cleanup
and teardown using an identical frozen empirical envelope for both revisions.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import os
import platform
import statistics
import time
from pathlib import Path
from typing import Any

from _retention import raw_output_path


def endpoint(case: Any, estimator: dict[str, Any], steps: int) -> dict[str, Any]:
    """Include typed policy construction, preparation, all solves, final audit and closure."""
    from generativeqc import ForceAwareScfPolicy, ScfForceErrorEstimator

    from tools.generativeqc_numerics import scf_effort_geomopt_benchmark as harness

    started = time.perf_counter()
    policy = ForceAwareScfPolicy(
        harness.levels(),
        ScfForceErrorEstimator(**estimator),
        optimizer_force_tolerance=5e-4,
        intermediate_energy_error=2e-5,
        intermediate_force_fraction=0.15,
        max_intermediate_force_error=5e-3,
        near_stationary_factor=5.0,
        relax_after=2,
    )
    result = harness.fire_optimize(
        case,
        device="cpu",
        policy=policy,
        adaptive=True,
        precision="fp64",
        max_steps=steps,
    )
    result["seconds"] = time.perf_counter() - started
    return result


def main() -> None:
    """Validate retention before importing scientific packages or constructing an owner."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=raw_output_path, required=True)
    parser.add_argument("--source-sha", required=True)
    parser.add_argument("--calibration", type=Path)
    parser.add_argument("--repeats", type=int, default=21)
    parser.add_argument("--warmups", type=int, default=3)
    parser.add_argument("--steps", type=int, default=6)
    parser.add_argument(
        "--cases", nargs="+", default=["h2", "lih", "water", "water-def2-svp"]
    )
    args = parser.parse_args()
    if args.repeats < 1 or args.warmups < 0 or args.steps < 1:
        parser.error("repeats/steps must be positive and warmups nonnegative")
    if args.output.exists():
        parser.error("refusing to overwrite a preserved receipt")

    import generativeqc
    from generativeqc import ScfForceErrorEstimator, force_aware_scf

    from tools.generativeqc_numerics import scf_effort_geomopt_benchmark as harness

    cases = harness.cases()
    if not set(args.cases) <= cases.keys():
        parser.error("unknown CPU trajectory fixture")
    library = Path(os.environ["GENERATIVEQC_LIBRARY"]).resolve(strict=True)
    publication = {
        "source_sha": args.source_sha,
        "python_package": str(Path(generativeqc.__file__).resolve()),
        "policy_source": str(Path(force_aware_scf.__file__).resolve()),
        "harness_source": str(Path(harness.__file__).resolve()),
        "harness_sha256": hashlib.sha256(
            Path(harness.__file__).read_bytes()
        ).hexdigest(),
        "runner_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "library": str(library),
        "library_sha256": hashlib.sha256(library.read_bytes()).hexdigest(),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "threads": {
            name: os.environ.get(name)
            for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS")
        },
    }
    if args.calibration is None:
        samples, rows = harness.calibration_samples((cases["h2"], cases["lih"]), "cpu")
        fitted = ScfForceErrorEstimator.fit(samples, safety_factor=1.5)
        publication.update(
            kind="measured-scf-effort-calibration-v1",
            estimator=dataclasses.asdict(fitted),
            calibration_id=fitted.identity,
            measured_rows=rows,
        )
    else:
        calibration_bytes = args.calibration.read_bytes()
        calibration = json.loads(calibration_bytes)
        if calibration["kind"] != "measured-scf-effort-calibration-v1":
            parser.error("expected measured empirical SCF calibration receipt")
        estimator = calibration["estimator"]
        if (
            ScfForceErrorEstimator(**estimator).identity
            != calibration["calibration_id"]
        ):
            parser.error("calibration identity does not match fitted metadata")
        publication.update(
            kind="complete-cpu-scf-effort-endpoint-v1",
            calibration_source=str(args.calibration.resolve()),
            calibration_sha256=hashlib.sha256(calibration_bytes).hexdigest(),
            calibration_id=calibration["calibration_id"],
            timing_scope="construct-policy+prepare+all-adaptive-SCF-force-steps+strict-cleanup+verify+close",
            first_use_scope="per-case first endpoint; shared process and persistent compiler cache",
            max_steps=args.steps,
            cases={},
        )
        for name in args.cases:
            for moved in (False, True):
                case = cases[name]
                if moved:
                    atoms = list(case.atoms)
                    symbol, position = atoms[-1]
                    atoms[-1] = (
                        symbol,
                        tuple(
                            value + delta
                            for value, delta in zip(
                                position, (0.017, -0.009, 0.013), strict=True
                            )
                        ),
                    )
                    case = dataclasses.replace(case, atoms=tuple(atoms))
                first = endpoint(case, estimator, args.steps)
                for _ in range(args.warmups):
                    endpoint(case, estimator, args.steps)
                samples = [
                    endpoint(case, estimator, args.steps) for _ in range(args.repeats)
                ]
                publication["cases"][f"{name}-{'moved' if moved else 'original'}"] = {
                    "atoms_bohr": case.atoms,
                    "first_use": first,
                    "warmups": args.warmups,
                    "samples": samples,
                    "median_seconds": statistics.median(
                        sample["seconds"] for sample in samples
                    ),
                }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(publication, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
