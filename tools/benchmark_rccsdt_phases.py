"""Collect #1501 phase/work evidence for the public CCSD(T) endpoint."""

from __future__ import annotations

import argparse
import ctypes
import json
import time
from pathlib import Path

from benchmarks.readme_hf_scaling import scaling_cases
from vibeqc import Calculator, _native

EXPECTED = {
    3: (-75.01437492827876, -7.626040122204577e-05),
    6: (-150.02895406931086, -0.00015372676875703484),
}


def _sample(atom_count: int, device: str, budget: int) -> dict[str, object]:
    atoms = scaling_cases()[f"water-{atom_count}"].atoms
    calc = Calculator(
        method="ccsd(t)",
        basis="sto-3g",
        basis_representation="spherical",
        device=device,
        max_iterations=200,
        energy_tolerance=1e-13,
        density_tolerance=1e-11,
        ccsd_max_iterations=150,
        ccsd_energy_tolerance=1e-12,
        ccsd_residual_tolerance=1e-10,
        correlation_memory_budget_bytes=budget,
    )
    started = time.perf_counter()
    result = calc.singlepoint(atoms, properties=("energy",))
    wall = time.perf_counter() - started
    diag = result.correlation
    assert diag is not None and result.converged
    expected_energy, expected_triples = EXPECTED[atom_count]
    assert abs(result.energy - expected_energy) <= 1e-8
    assert abs(diag.ccsd_t_triples_energy - expected_triples) <= 1e-10
    phases = {
        "reference": diag.ccsd_reference_seconds,
        "problem": diag.ccsd_problem_seconds,
        "provider": diag.ccsd_provider_seconds,
        "source": diag.ccsd_source_seconds,
        "solver": diag.ccsd_solver_seconds,
        "iteration": diag.ccsd_iteration_seconds,
        "replay": diag.ccsd_replay_seconds,
        "update": diag.ccsd_update_seconds,
        "diis": diag.ccsd_diis_seconds,
        "triples": diag.ccsd_t_seconds,
    }
    work = {
        "source_scans": diag.ccsd_source_scans,
        "source_reads": diag.ccsd_source_reads,
        "source_values": diag.ccsd_source_values,
        "transform_fmas": diag.ccsd_transform_fmas,
        "mo_blocks": diag.ccsd_mo_blocks,
        "cuda_transform_calls": diag.ccsd_cuda_transform_calls,
        "cuda_batch_calls": diag.ccsd_cuda_batch_calls,
        "iteration_graph_calls": diag.ccsd_iteration_graph_calls,
        "replay_graph_calls": diag.ccsd_replay_graph_calls,
        "update_calls": diag.ccsd_update_calls,
        "generated_error_checks": diag.ccsd_generated_error_checks,
        "diis_gram_calls": diag.ccsd_diis_gram_calls,
        "diis_coefficient_calls": diag.ccsd_diis_coefficient_calls,
        "diis_combine_calls": diag.ccsd_diis_combine_calls,
        "synchronizations": diag.ccsd_synchronizations,
        "setup_h2d_bytes": diag.ccsd_setup_h2d_bytes,
        "scalar_d2h_bytes": diag.ccsd_scalar_d2h_bytes,
        "amplitude_d2h_bytes": diag.ccsd_amplitude_d2h_bytes,
        "mo_transfer_bytes": diag.mo_transfer_bytes,
    }
    return {
        "atoms": atom_count,
        "wall_seconds": wall,
        "energy": result.energy,
        "triples_energy": diag.ccsd_t_triples_energy,
        "ccsd_iterations": diag.ccsd_iterations,
        "virtual_triples": diag.ccsd_t_virtual_triples,
        "phases_seconds": phases,
        "work": work,
        "numeric_capacity_bytes": diag.numeric_capacity_bytes,
        "device_owned_bytes": diag.correlation_owned_device_bytes,
        "provider_retained_bytes": diag.correlation_provider_retained_bytes,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--atoms", type=int, choices=(3, 6), action="append")
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--budget", type=int, default=8 << 30)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    atoms = args.atoms or [3, 6]
    library = _native.load_library(device=args.device)
    library.vibeqc_get_source_identity.argtypes = []
    library.vibeqc_get_source_identity.restype = ctypes.c_char_p
    rows = [
        _sample(atom_count, args.device, args.budget)
        for atom_count in atoms
        for _ in range(args.repeats)
    ]
    record = {
        "schema": "vibeqc.ccsdt.phase-work/1",
        "issue": 1501,
        "source_identity": library.vibeqc_get_source_identity().decode(),
        "device": args.device,
        "budget_bytes": args.budget,
        "results": rows,
        "note": "Instrumentation evidence only; no performance-leadership claim.",
        "agent": "ChatGPT",
        "model": "GPT-5.6 Sol",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
