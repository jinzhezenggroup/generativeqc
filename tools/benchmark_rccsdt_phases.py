"""Collect #1501 phase/work evidence for the public CCSD(T) endpoint."""

from __future__ import annotations

import argparse
import ctypes
import json
import time
from pathlib import Path

from generativeqc import Calculator, _native

from benchmarks.readme_hf_scaling import scaling_cases

EXPECTED = {
    3: (-75.01437492827876, -7.626040122204577e-05),
    6: (-150.02895406931086, -0.00015372676875703484),
}


def _sample(atom_count: int, device: str, budget: int) -> dict[str, object]:
    atoms = scaling_cases()[f"water-{atom_count}"].atoms
    started = time.perf_counter()
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
    result = calc.singlepoint(atoms, properties=("energy",))
    wall = time.perf_counter() - started
    diag = result.correlation
    perf = result.cc_performance
    assert diag is not None and perf is not None and result.converged
    expected_energy, expected_triples = EXPECTED[atom_count]
    assert abs(result.energy - expected_energy) <= 1e-8
    assert abs(diag.ccsd_t_triples_energy - expected_triples) <= 1e-10
    phases = {
        "reference": perf.reference_seconds,
        "problem": perf.problem_seconds,
        "provider": perf.provider_seconds,
        "source": perf.source_seconds,
        "solver": perf.solver_seconds,
        "iteration": perf.iteration_seconds,
        "replay": perf.replay_seconds,
        "update": perf.update_seconds,
        "diis": perf.diis_seconds,
        "triples": perf.triples_seconds,
    }
    attributed = (
        perf.reference_seconds
        + perf.problem_seconds
        + perf.solver_seconds
        + perf.triples_seconds
    )
    phases["unattributed_endpoint"] = wall - attributed
    work = {
        "source_scans": perf.source_scans,
        "source_reads": perf.source_reads,
        "source_values": perf.source_values,
        "transform_fmas": perf.transform_fmas,
        "transform_stages": perf.transform_stages,
        "mo_blocks": perf.mo_blocks,
        "cuda_transform_calls": perf.cuda_transform_calls,
        "cuda_batch_calls": perf.cuda_batch_calls,
        "iteration_graph_calls": perf.iteration_graph_calls,
        "replay_graph_calls": perf.replay_graph_calls,
        "update_calls": perf.update_calls,
        "generated_error_checks": perf.generated_error_checks,
        "diis_gram_calls": perf.diis_gram_calls,
        "diis_coefficient_calls": perf.diis_coefficient_calls,
        "diis_combine_calls": perf.diis_combine_calls,
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
    library.generativeqc_get_source_identity.argtypes = []
    library.generativeqc_get_source_identity.restype = ctypes.c_char_p
    rows = [
        _sample(atom_count, args.device, args.budget)
        for atom_count in atoms
        for _ in range(args.repeats)
    ]
    record = {
        "schema": "generativeqc.ccsdt.phase-work/1",
        "issue": 1501,
        "source_identity": library.generativeqc_get_source_identity().decode(),
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
