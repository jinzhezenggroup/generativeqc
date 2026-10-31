#!/usr/bin/env python3
"""Compare complete CPU CBS endpoints in independent baseline/candidate processes."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import statistics
import subprocess
import sys
from pathlib import Path
from time import perf_counter

CASES = ("standard", "tight", "custom", "explicit")


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def worker(args: argparse.Namespace) -> dict:
    """Include facade construction, component preparation, E/F and ordinary cleanup."""
    from dataclasses import asdict

    import numpy as np
    from generativeqc import (
        CBSBasis,
        CCSDTCBSCalculator,
        PrecisionPolicy,
        TargetErrorBudget,
        load_basis,
    )

    if args.worker == "candidate":
        from generativeqc import _correlation_convergence as policy

        def forbidden(*args: object, **kwargs: object) -> None:
            raise AssertionError("admitted endpoint used Python policy arithmetic")

        policy._compatibility_scale = policy._compatibility_limits = forbidden
    # Reuse pinned algebra fixtures with two distinct exponents. These timings
    # qualify complete policy/solver ownership, not a physical CBS basis limit.
    references = Path("tests/reference_data/cbs")
    bases = tuple(
        CBSBasis(load_basis(references / f"water-algebra-{name}.json"), cardinal)
        for name, cardinal in (("low", 2), ("high", 3))
    )
    selected = (
        PrecisionPolicy("fast", budget=TargetErrorBudget(5e-6, 5e-5, None))
        if args.case == "custom"
        else "tight"
        if args.case == "tight"
        else "standard"
    )
    options = (
        {"energy_tolerance": 1e-13, "ccsd_residual_tolerance": 1e-10}
        if args.case == "explicit"
        else {}
    )

    def endpoint(index: int) -> tuple[float, dict]:
        # Cold preparation every time; alternate genuine bond changes.
        bond = 1.4 + 0.04 * (index % 2)
        atoms = (("H", (0.0, 0.0, -bond / 2)), ("H", (0.0, 0.0, bond / 2)))
        start = perf_counter()
        workflow = CCSDTCBSCalculator(
            bases, policy=selected, calculator_options=options
        )
        result = workflow.singlepoint(atoms)
        # All native component owners close in singlepoint before publication.
        del workflow
        seconds = perf_counter() - start
        assert result.converged and result.executed_backend == "cpu_reference"
        components = []
        for component in result.components:
            r = component.result
            correlation = asdict(r.correlation) if r.correlation else None
            if correlation:
                correlation = {
                    key: value
                    for key, value in correlation.items()
                    if not key.endswith("_ms")
                }
            components.append(
                {
                    "roles": component.roles,
                    "method": component.method,
                    "coefficient": component.coefficient,
                    "energy": r.energy,
                    "forces": r.forces.tolist(),
                    "converged": r.converged,
                    "iterations": r.iterations,
                    "energy_change": r.energy_change,
                    "density_rms": r.density_rms,
                    "physical_residual_rms": r.physical_residual_rms,
                    "correlation": correlation,
                    "precision": r.precision,
                }
            )
        signature = {
            "bond": bond,
            "energy": result.energy,
            "forces": result.forces.tolist(),
            "components": components,
            "convergence": result.convergence,
        }
        assert np.isfinite(result.energy) and np.all(np.isfinite(result.forces))
        return seconds, signature

    for index in range(3):
        endpoint(index)
    samples = [endpoint(index) for index in range(21)]
    return {
        "label": args.worker,
        "case": args.case,
        "samples": samples,
        "source_sha": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
        "library_sha256": digest(Path(os.environ["GENERATIVEQC_LIBRARY"])),
        "production": {
            str(path): digest(path)
            for path in (
                Path("python/generativeqc/cbs.py"),
                Path("python/generativeqc/calculator.py"),
                Path("python/generativeqc/_correlation_convergence.py"),
                Path("src/core/correlation_convergence.hpp"),
            )
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path)
    parser.add_argument("--candidate", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--worker", choices=("baseline", "candidate"))
    parser.add_argument("--case", choices=CASES)
    args = parser.parse_args()
    if args.worker:
        print(json.dumps(worker(args), allow_nan=False))
        return
    record = {"warmups": 3, "samples_per_block": 21, "gate": 1.05, "groups": []}
    script = str(Path(__file__).resolve())
    for case in CASES:
        for order in ("ABBA", "BAAB"):
            runs = []
            for letter in order:
                label = "baseline" if letter == "A" else "candidate"
                root = getattr(args, label).resolve()
                env = dict(
                    os.environ,
                    PYTHONPATH=str(root / "python"),
                    GENERATIVEQC_LIBRARY=str(root / "build/libgenerativeqc.so"),
                    OPENBLAS_NUM_THREADS="1",
                    OMP_NUM_THREADS="1",
                )
                result = subprocess.run(
                    [sys.executable, script, "--worker", label, "--case", case],
                    cwd=root,
                    env=env,
                    check=True,
                    capture_output=True,
                    text=True,
                    timeout=300,
                )
                runs.append(json.loads(result.stdout))
            expected = [sample[1] for sample in runs[0]["samples"]]
            for run in runs[1:]:
                if [sample[1] for sample in run["samples"]] != expected:
                    raise AssertionError(f"scientific/work mismatch: {case}/{order}")
            times = {
                label: [
                    sample[0]
                    for run in runs
                    if run["label"] == label
                    for sample in run["samples"]
                ]
                for label in ("baseline", "candidate")
            }
            medians = {
                label: statistics.median(values) for label, values in times.items()
            }
            ratio = medians["candidate"] / medians["baseline"]
            group = {
                "case": case,
                "order": order,
                "runs": runs,
                "median_seconds": medians,
                "ratio": ratio,
                "relative_mad": {
                    label: statistics.median(
                        abs(value - medians[label]) for value in values
                    )
                    / medians[label]
                    for label, values in times.items()
                },
            }
            record["groups"].append(group)
            args.output.write_text(json.dumps(record, indent=2, allow_nan=False) + "\n")
            print(f"{case} {order}: {ratio:.6f}", flush=True)
    assert all(group["ratio"] <= record["gate"] for group in record["groups"]), (
        "complete endpoint regressed"
    )


if __name__ == "__main__":
    main()
