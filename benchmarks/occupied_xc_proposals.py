"""Qualify admission-inclusive occupied-proposal E+F and replay its receipt.

Execution requires a finite Slurm allocation, an explicitly selected native
library and independently converged default-grid RKS/UKS oracle vectors. Cold
cost includes construction/preparation in a shared process/context, not process
startup. Replay uses only the standard library and never loads a native library.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import lzma
import math
import os
import statistics
from dataclasses import asdict
from pathlib import Path
from time import perf_counter


def load(path: Path) -> dict | list:
    """Read a losslessly compacted receipt or independent oracle input."""
    opener = lzma.open if path.suffix == ".xz" else open
    with opener(path, "rt") as stream:
        return json.load(stream)


def save(path: Path, record: dict) -> None:
    """Retain observations, including legitimate infinite history sentinels.

    Like the existing endpoint receipts, diagnostics use Python JSON's Infinity
    extension for a missing previous energy. Replay independently rejects every
    nonfinite physical observable; diagnostic sentinels are not observables.
    """
    serialized = json.dumps(record, separators=(",", ":"))
    opener = lzma.open if path.suffix == ".xz" else open
    with opener(path, "wt") as stream:
        stream.write(serialized + "\n")


def replay(record: dict) -> dict:
    """Recompute all vector gates, paired gates, completeness and work summaries."""
    if record.get("schema") != "generativeqc.occupied-proposals.v1" or not record.get(
        "completed"
    ):
        raise ValueError("incomplete or unsupported receipt")
    if record["energy_gate_eh"] != 2e-9 or record["force_gate_eh_per_bohr"] != 1e-7:
        raise ValueError("scientific acceptance gates changed")
    for name, size in (("source_commit", 40), ("library_sha256", 64)):
        identity = record.get(name, "")
        if len(identity) != size or any(
            character not in "0123456789abcdef" for character in identity
        ):
            raise ValueError(f"invalid {name}")
    if record["repeats"] < 3:
        raise ValueError("insufficient repetitions")
    references = {(row["spin"], row["geometry"]): row for row in record["oracles"]}
    rows = {}
    for row in record["observations"]:
        key = row["spin"], row["repeat"], row["selector"], row["phase"]
        if key in rows:
            raise ValueError(f"duplicate observation: {key}")
        rows[key] = row
        reference = references[row["spin"], int(row["phase"].startswith("moved"))]
        check_vectors(row, reference)
        if row["iterations"] < 1 or row["fock_builds"] < row["iterations"]:
            raise ValueError("invalid semantic work counts")
        if not math.isfinite(row["seconds"]) or row["seconds"] <= 0:
            raise ValueError("invalid endpoint duration")
    summary = {}
    for spin in ("rks", "uks"):
        for phase in ("cold", "warm", "moved", "moved_warm"):
            cohorts = {selector: [] for selector in ("0", "1")}
            work = {selector: [] for selector in ("0", "1")}
            for repetition in range(record["repeats"]):
                baseline, proposal = (
                    rows[spin, repetition, selector, phase] for selector in ("0", "1")
                )
                check_vectors(proposal, baseline)
                for row in (baseline, proposal):
                    cohorts[row["selector"]].append(row["seconds"])
                    work[row["selector"]].append(
                        (row["iterations"], row["fock_builds"])
                    )
            medians = {
                selector: statistics.median(values)
                for selector, values in cohorts.items()
            }
            summary[f"{spin}-{phase}"] = {
                "matrix_seconds": medians["0"],
                "proposal_seconds": medians["1"],
                "speedup": medians["0"] / medians["1"],
                "iterations_fock_builds": work,
            }
    if len(rows) != 16 * record["repeats"]:
        raise ValueError("incomplete or unexpected observation population")
    return summary


def check_vectors(actual: dict, reference: dict) -> None:
    """Use explicit exceptions so optimized Python cannot disable acceptance."""
    energy_error = abs(actual["energy"] - reference["energy"])
    actual_forces, reference_forces = actual["forces"], reference["forces"]
    if len(actual_forces) != len(reference_forces) or not actual_forces:
        raise ValueError("incomplete force vector")
    errors = []
    for actual_row, reference_row in zip(actual_forces, reference_forces, strict=True):
        if len(actual_row) != 3 or len(reference_row) != 3:
            raise ValueError("invalid force vector shape")
        errors.extend(
            abs(left - right)
            for left, right in zip(actual_row, reference_row, strict=True)
        )
    if not math.isfinite(energy_error) or energy_error > 2e-9:
        raise ValueError(f"energy acceptance failed: {energy_error}")
    if any(not math.isfinite(error) or error > 1e-7 for error in errors):
        raise ValueError(f"force acceptance failed: {max(errors)}")


def main() -> None:
    """Alternate complete native matrix/proposal endpoints, or replay offline."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replay", type=Path)
    parser.add_argument("--oracle", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--source-commit")
    parser.add_argument("--repeats", type=int, default=10)
    args = parser.parse_args()
    if args.replay:
        print(json.dumps(replay(load(args.replay)), indent=2))
        return
    if not args.oracle or not args.output or not args.source_commit or args.repeats < 3:
        parser.error(
            "require oracle, fresh output, source commit and at least three repetitions"
        )
    if (
        args.output.exists()
        or not os.environ.get("SLURM_JOB_ID")
        or not os.environ.get("CUDA_VISIBLE_DEVICES")
    ):
        parser.error("require fresh output and a finite Slurm GPU allocation")
    if len(args.source_commit) != 40 or any(
        character not in "0123456789abcdef" for character in args.source_commit
    ):
        parser.error("source commit must be a full hexadecimal Git SHA")

    import numpy as np
    from generativeqc import Calculator
    from test_dft_complete_cuda import no_cpu_derivatives

    from benchmarks.readme_hf_scaling import scaling_cases

    library = Path(os.environ["GENERATIVEQC_LIBRARY"])
    with library.open("rb") as stream:
        library_sha256 = hashlib.file_digest(stream, "sha256").hexdigest()
    record = {
        "schema": "generativeqc.occupied-proposals.v1",
        "completed": False,
        "source_commit": args.source_commit,
        "library_sha256": library_sha256,
        "slurm_job_id": os.environ["SLURM_JOB_ID"],
        "cuda_visible_devices": os.environ["CUDA_VISIBLE_DEVICES"],
        "repeats": args.repeats,
        "energy_gate_eh": 2e-9,
        "force_gate_eh_per_bohr": 1e-7,
        "oracles": load(args.oracle),
        "observations": [],
    }
    atoms = scaling_cases()["water-3"].atoms
    moved = np.asarray([position for _, position in atoms]).copy()
    moved[1, 2] += 0.02
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for spin in ("rks", "uks"):
        charge = int(spin == "uks")
        for repetition in range(args.repeats):
            for selector in ("0", "1") if repetition % 2 == 0 else ("1", "0"):
                os.environ["GENERATIVEQC_CUDA_KS_OCCUPIED_XC"] = selector
                started = perf_counter()
                calculator = Calculator(
                    method=f"r2scan-{spin}",
                    device="cuda",
                    basis="def2-svp",
                    basis_representation="spherical",
                    energy_tolerance=1e-12,
                    density_tolerance=1e-10,
                    max_iterations=200,
                )
                try:
                    with calculator.prepare_batch(
                        [atoms],
                        charges=[charge],
                        multiplicities=[charge + 1],
                        warm_start=True,
                    ) as owner:
                        prepared = perf_counter()
                        for phase in ("cold", "warm", "moved", "moved_warm"):
                            with no_cpu_derivatives():
                                execution_started = perf_counter()
                                result = owner.execute(
                                    [moved] if phase.startswith("moved") else None,
                                    properties=("energy", "forces"),
                                    strict=True,
                                ).items[0]
                                elapsed = perf_counter() - execution_started
                            if (
                                not result.succeeded
                                or not result.converged
                                or result.executed_backend != "cuda"
                            ):
                                raise RuntimeError(
                                    "native endpoint did not converge on CUDA"
                                )
                            record["observations"].append(
                                {
                                    "spin": spin,
                                    "repeat": repetition,
                                    "selector": selector,
                                    "phase": phase,
                                    "seconds": elapsed
                                    + (prepared - started if phase == "cold" else 0),
                                    "energy": result.energy,
                                    "forces": np.asarray(result.forces).tolist(),
                                    "iterations": result.iterations,
                                    "fock_builds": result.fock_builds,
                                    "ks_diagnostic": asdict(result.ks_diagnostic)
                                    if result.ks_diagnostic is not None
                                    else None,
                                    "density_rms": result.density_rms,
                                    "physical_residual_rms": result.physical_residual_rms,
                                }
                            )
                            save(args.output, record)
                finally:
                    calculator.clear_cache()
    record["completed"] = True
    summary = replay(record)
    save(args.output, record)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
