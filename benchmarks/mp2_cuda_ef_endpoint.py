"""Complete native CUDA MP2 E+F qualification, never an energy-only speed claim.

Run through srun. Independent PySCF calculations, optional work-journal replay,
and immutable input preparation are outside the measured public endpoints.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import statistics
import subprocess
import tempfile
import time
from pathlib import Path

import numpy as np
from _retained_basis import load_retained_basis
from _support import environment_metadata, raw_output_path, write_result
from generativeqc import Calculator

from tools.generativeqc_posthf.fixtures import load_fixture, source_arguments


def physical_case(name: str) -> tuple[dict, dict]:
    """Use frozen shells, including a correlation-consistent extrapolation case."""
    metadata, _ = load_fixture("h2" if name == "h2-ccpvdz" else name)
    arguments = source_arguments(metadata)
    if name == "h2-ccpvdz":
        basis = load_retained_basis(
            Path(__file__).resolve().parents[1]
            / "benchmarks/results/issue206-practical-auxiliary/identity/cc-pvdz.json"
        )
        arguments["basis"] = basis.shells_for(arguments["atoms"])
        arguments["representation"] = "spherical"
        metadata["inputs"]["basis_representation"] = "real_spherical"
        metadata["inputs"]["shells"] = [
            {
                "atom_index": shell.atom_index,
                "angular_momentum": shell.angular_momentum,
                "primitives": [
                    [primitive.exponent, primitive.coefficient]
                    for primitive in shell.primitives
                ],
            }
            for shell in arguments["basis"]
        ]
        metadata["records"] = {}
        metadata["inputs"]["basis_identity"] = basis.identity
    return metadata, arguments


def independent_oracle(arguments: dict) -> tuple[float, np.ndarray]:
    """Use identical unnormalized physical inputs, not the native force chain."""
    from pyscf import gto, mp, scf
    from pyscf.data.elements import ELEMENTS

    atoms = [
        (f"{ELEMENTS[atom.atomic_number]}{index}", atom.position)
        for index, atom in enumerate(arguments["atoms"])
    ]
    basis = {label: [] for label, _ in atoms}
    for shell in arguments["basis"]:
        basis[atoms[shell.atom_index][0]].append(
            [
                shell.angular_momentum,
                *[
                    [primitive.exponent, primitive.coefficient]
                    for primitive in shell.primitives
                ],
            ]
        )
    molecule = gto.M(
        atom=atoms,
        basis=basis,
        unit="Bohr",
        charge=arguments["charge"],
        cart=arguments["representation"] == "cartesian",
        verbose=0,
    )
    mean_field = scf.RHF(molecule)
    mean_field.conv_tol = 1e-12
    mean_field.conv_tol_grad = 1e-10
    mean_field.kernel()
    if not mean_field.converged:
        raise RuntimeError("independent reference did not converge")
    correlation = mp.MP2(mean_field)
    correlation.kernel()
    return float(correlation.e_tot), -correlation.nuc_grad_method().kernel()


def work_journal(calc: Calculator, arguments: dict) -> dict:
    """Collect actual completed source/transform work in an untimed replay."""
    with tempfile.TemporaryDirectory(prefix="mp2-ef-work-") as directory:
        journal = Path(directory) / "progress.jsonl"
        previous = os.environ.get("GENERATIVEQC_DF_PROGRESS_TRACE")
        os.environ["GENERATIVEQC_DF_PROGRESS_TRACE"] = str(journal)
        try:
            calc.singlepoint(
                arguments["atoms"],
                charge=arguments["charge"],
                properties=("energy", "forces"),
            )
        finally:
            if previous is None:
                os.environ.pop("GENERATIVEQC_DF_PROGRESS_TRACE", None)
            else:
                os.environ["GENERATIVEQC_DF_PROGRESS_TRACE"] = previous
        records = [json.loads(line) for line in journal.read_text().splitlines()]
    begins = {record["id"] for record in records if record["event"] == "BEGIN"}
    ends = {record["id"] for record in records if record["event"] == "END"}
    if begins != ends or not any(
        record["name"] == "mp2_endpoint"
        and record["event"] == "END"
        and record["status"] == "returned"
        for record in records
    ):
        raise RuntimeError("work journal contains incomplete or failed consumers")
    counters = {}
    for record in records:
        if record["event"] == "VALUE" and record["name"] == "native_mo_blocks":
            counters[record["key"]] = counters.get(record["key"], 0) + record["value"]
    routing = {
        record["key"]: record["value"]
        for record in records
        if record["event"] == "VALUE" and record["name"] == "mp2_endpoint"
    }
    discarded = [
        record["name"]
        for record in records
        if record["event"] == "END"
        and (record["status"] == "exception" or record["status"].endswith("_failed"))
    ]
    return {
        "completed_source_work": counters,
        "routing": routing,
        "discarded_scopes": discarded,
    }


def endpoint(name: str, repeats: int, budget: int, oracle: bool, trace: bool) -> dict:
    """Include reference, correlation, Z solve, relaxed weights and derivatives."""
    metadata, arguments = physical_case(name)
    calc = Calculator(
        method="mp2",
        device="cuda",
        basis=arguments["basis"],
        basis_representation=arguments["representation"],
        correlation_memory_budget_bytes=budget,
    )
    started = time.perf_counter()
    result = calc.singlepoint(
        arguments["atoms"], charge=arguments["charge"], properties=("energy", "forces")
    )
    cold = time.perf_counter() - started
    samples = [result]
    prepared_times = []
    with calc.prepare_batch(
        [arguments["atoms"]], charges=[arguments["charge"]]
    ) as batch:
        for _ in range(repeats + 1):
            started = time.perf_counter()
            result = batch.execute(properties=("energy", "forces"), strict=True).items[
                0
            ]
            prepared_times.append(time.perf_counter() - started)
            samples.append(result)
    expected = metadata["records"].get("conventional")
    if expected and any(
        abs(sample.energy - expected["hf_energy"] - expected["correlation_energy"])
        > 1e-9
        for sample in samples
    ):
        raise RuntimeError(f"{name}: independent pinned energy gate failed")
    if any(
        not sample.converged
        or sample.executed_backend != "cuda"
        or sample.forces is None
        or not np.isfinite(sample.forces).all()
        or sample.correlation.force_provenance_flags != 0x7
        or sample.correlation.numeric_capacity_bytes > budget
        for sample in samples
    ):
        raise RuntimeError(f"{name}: incomplete E+F endpoint")
    diag = result.correlation
    record = {
        "case": name,
        "physical_inputs": metadata["inputs"],
        "domain": {
            "basis_functions": sum(
                2 * shell.angular_momentum + 1
                if arguments["representation"] == "spherical"
                else (shell.angular_momentum + 1) * (shell.angular_momentum + 2) // 2
                for shell in arguments["basis"]
            ),
            "occupied_orbitals": (
                sum(atom.atomic_number for atom in arguments["atoms"])
                - arguments["charge"]
            )
            // 2,
            "shells": len(arguments["basis"]),
        },
        "cold_s": cold,
        "prepared_cold_s": prepared_times[0],
        "prepared_warm_s": prepared_times[1:],
        "prepared_warm_median_s": statistics.median(prepared_times[1:]),
        "energy": result.energy,
        "forces": result.forces.tolist(),
        "opposite_spin": diag.opposite_spin_energy,
        "same_spin": diag.same_spin_energy,
        "response_iterations": diag.response_iterations,
        "response_absolute_residual": diag.response_absolute_residual,
        "numeric_capacity_bytes": diag.numeric_capacity_bytes,
        "planned_endpoint_peak_bytes": diag.planned_endpoint_peak_bytes,
        "measured_endpoint_peak_bytes": diag.measured_endpoint_peak_bytes,
        "mo_transfer_bytes": diag.mo_transfer_bytes,
        "reference_execution_plan_reused": diag.reference_execution_plan_reused,
        "force_provenance_flags": diag.force_provenance_flags,
    }
    if oracle:
        energy, forces = independent_oracle(arguments)
        record["oracle_errors"] = {
            "energy_hartree": max(abs(sample.energy - energy) for sample in samples),
            "maximum_force_hartree_per_bohr": max(
                float(np.max(np.abs(sample.forces - forces))) for sample in samples
            ),
        }
        if (
            record["oracle_errors"]["energy_hartree"] > 1e-9
            or record["oracle_errors"]["maximum_force_hartree_per_bohr"] > 2e-6
        ):
            raise RuntimeError(
                f"{name}: independent complete E+F gate failed: {record['oracle_errors']}"
            )
    if trace:
        record["work_journal"] = work_journal(calc, arguments)
    print(
        f"{name}: cold={cold:.6f}s prepared-warm={record['prepared_warm_median_s']:.6f}s",
        flush=True,
    )
    return record


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--case", choices=("h2", "water", "lih", "f_heh", "h2-ccpvdz"), action="append"
    )
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--budget-mib", type=int, default=256)
    parser.add_argument("--oracle", action="store_true")
    parser.add_argument("--trace", action="store_true")
    parser.add_argument("--output", type=raw_output_path, required=True)
    args = parser.parse_args()
    if not os.environ.get("SLURM_JOB_ID"):
        parser.error("real-GPU qualification must run through srun")
    if args.repeats < 1 or args.budget_mib < 1:
        parser.error("repeats and memory budget must be positive")
    if "h2-ccpvdz" in (args.case or ()) and not args.oracle:
        parser.error("the correlation-consistent case requires --oracle")
    cases = args.case or ("h2", "water", "lih")
    library = Path(os.environ["GENERATIVEQC_LIBRARY"])
    payload = {
        "endpoint": "conventional all-electron CUDA RHF-MP2 energy + analytic forces",
        "environment": environment_metadata(
            distributions={"numpy": ("numpy",), "pyscf": ("pyscf",)},
            accelerator={
                "backend": "cuda",
                "slurm_job_id": os.environ["SLURM_JOB_ID"],
                "nvidia_smi": subprocess.check_output(
                    [
                        "nvidia-smi",
                        "--query-gpu=name,uuid,driver_version,memory.total",
                        "--format=csv,noheader",
                    ],
                    text=True,
                ).strip(),
            },
        ),
        "oracle_acceptance": {
            "energy_hartree": 1e-9,
            "maximum_force_hartree_per_bohr": 2e-6,
        },
        "library_sha256": hashlib.sha256(library.read_bytes()).hexdigest(),
        "source_revision": os.environ.get("GENERATIVEQC_SOURCE_REVISION"),
        "native_source_sha256": {
            relative: hashlib.sha256(
                (Path(__file__).resolve().parents[1] / relative).read_bytes()
            ).hexdigest()
            for relative in (
                "src/methods/mp2_method.cpp",
                "src/posthf/mp2_force.cpp",
                "src/posthf/mp2_energy.cpp",
                "src/posthf/native_provider.cpp",
            )
        },
        "controls": {
            name: os.environ.get(name)
            for name in (
                "GENERATIVEQC_MP2_REFERENCE_SOURCE",
                "GENERATIVEQC_MP2_ENERGY_SOURCE_TILING",
                "GENERATIVEQC_MP2_FORCE_SOURCE_TILING",
            )
        },
        "requested_cases": cases,
        "complete": False,
        "records": [],
    }
    write_result(args.output, payload)
    for name in cases:
        payload["records"].append(
            endpoint(name, args.repeats, args.budget_mib << 20, args.oracle, args.trace)
        )
        write_result(args.output, payload)
    payload["complete"] = True
    write_result(args.output, payload)


if __name__ == "__main__":
    main()
