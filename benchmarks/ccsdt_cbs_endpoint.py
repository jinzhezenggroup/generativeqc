"""Complete native MP2-assisted CCSD(T)/CBS E+F timing and qualification.

Caches are released between components. Repeated samples are complete replay
endpoints, not retained prepared-batch warm timings. Pinned independent oracle
comparison and an optional semantic-work replay are outside the timing samples.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import statistics
from dataclasses import asdict
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
from _support import environment_metadata, write_result
from generativeqc import CBSBasis, CCSDTCBSCalculator, load_basis
from mp2_cuda_ef_endpoint import work_journal

if TYPE_CHECKING:
    from generativeqc import CBSResult

ROOT = Path(__file__).resolve().parents[1]
REFERENCES = ROOT / "tests/reference_data/cbs"


def qualify(result: CBSResult, reference: dict) -> dict[str, float]:
    """Gate composite and every native constituent, not just cancellation."""
    energy_error = abs(result.energy - reference["energy"])
    force_error = float(np.max(np.abs(result.forces - reference["forces"])))
    if energy_error > 1e-9 or force_error > 2e-6:
        raise RuntimeError(
            f"composite oracle failure: E={energy_error}, F={force_error}"
        )
    for component in result.components:
        oracle = reference["components"][component.roles[0]]
        if abs(component.result.energy - oracle["energy"]) > 1e-9:
            raise RuntimeError(f"component energy oracle failure: {component.roles}")
        if np.max(np.abs(component.result.forces - oracle["forces"])) > 2e-6:
            raise RuntimeError(f"component force oracle failure: {component.roles}")
    return {
        "energy_error_hartree": energy_error,
        "max_force_error_hartree_per_bohr": force_error,
    }


def result_record(result: CBSResult) -> dict[str, Any]:
    """Retain component E+F, convergence, response, work and resource evidence."""
    return {
        "energy": result.energy,
        "forces": result.forces.tolist(),
        "hf_energy": result.hf_energy,
        "mp2_correlation_energy": result.mp2_correlation_energy,
        "delta_ccsdt_energy": result.delta_ccsdt_energy,
        "hf_weights": result.hf_weights,
        "mp2_weights": result.mp2_weights,
        "executed_backend": result.executed_backend,
        "endpoint_seconds": result.endpoint_seconds,
        "basis_metadata": result.basis_metadata,
        "components": [
            {
                "roles": component.roles,
                "method": component.method,
                "basis_identity": component.basis_identity,
                "coefficient": component.coefficient,
                "elapsed_seconds": component.elapsed_seconds,
                "energy": component.result.energy,
                "forces": component.result.forces.tolist(),
                "converged": component.result.converged,
                "iterations": component.result.iterations,
                "executed_backend": component.result.executed_backend,
                "correlation": None
                if component.result.correlation is None
                else asdict(component.result.correlation),
                "cc_performance": None
                if component.result.cc_performance is None
                else asdict(component.result.cc_performance),
            }
            for component in result.components
        ],
    }


def main() -> None:
    """Execute scheduler-owned full endpoints and save reproducible JSON."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--case", choices=("h2-dz-tz", "water-algebra"), default="h2-dz-tz"
    )
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--budget", type=int, default=256 << 20)
    parser.add_argument("--oracle", action="store_true")
    parser.add_argument("--work", action="store_true")
    parser.add_argument("--revision")
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    if arguments.repeats < 1 or arguments.budget < 1:
        parser.error("repeats and budget must be positive")
    if arguments.device == "cuda" and not os.environ.get("SLURM_JOB_ID"):
        parser.error("real GPU endpoints must run through srun")
    reference = json.loads((REFERENCES / f"{arguments.case}.json").read_text())
    low = load_basis(REFERENCES / reference["low_basis"])
    high = load_basis(REFERENCES / reference["high_basis"])
    workflow = CCSDTCBSCalculator(
        (CBSBasis(low, 2), CBSBasis(high, 3)),
        device=arguments.device,
        calculator_options={"correlation_memory_budget_bytes": arguments.budget},
    )
    results = []
    for index in range(arguments.repeats + 1):
        result = workflow.singlepoint(reference["atoms"])
        record = result_record(result)
        if arguments.oracle:
            record["oracle"] = qualify(result, reference["reference"])
        results.append(record)
        print(
            arguments.case,
            "cold" if index == 0 else "replay",
            result.endpoint_seconds,
            "seconds",
            flush=True,
        )
    work = None
    if arguments.work:
        from generativeqc import Atom

        work = work_journal(
            workflow,
            {
                "atoms": [Atom.from_value(atom) for atom in reference["atoms"]],
                "charge": 0,
            },
        )
    library = os.environ.get("GENERATIVEQC_LIBRARY")
    output = {
        "schema": "generativeqc.ccsdt_cbs_endpoint",
        "version": 1,
        "case": arguments.case,
        "physical_cbs_basis_pair": reference["physical_cbs_basis_pair"],
        "endpoint": "energy+analytic_forces",
        "prepared_plan_retention": False,
        "source_revision": arguments.revision,
        "source_sha256": {
            str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in (
                ROOT / "python/generativeqc/cbs.py",
                Path(__file__),
                REFERENCES / f"{arguments.case}.json",
            )
        },
        "library_sha256": hashlib.sha256(Path(library).read_bytes()).hexdigest()
        if library
        else None,
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "environment": environment_metadata(),
        "budget_bytes": arguments.budget,
        "cold_seconds": results[0]["endpoint_seconds"],
        "replay_seconds": [result["endpoint_seconds"] for result in results[1:]],
        "median_replay_seconds": statistics.median(
            result["endpoint_seconds"] for result in results[1:]
        ),
        "samples": results,
        "work_journal": work,
    }
    write_result(arguments.output, output)


if __name__ == "__main__":
    main()
