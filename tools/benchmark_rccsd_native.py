"""Record compact public native RCCSD endpoint/resource qualification for #149 C."""

from __future__ import annotations

import argparse
import ctypes
import json
import time
from pathlib import Path

from vibeqc import Calculator, _native

ROOT = Path(__file__).resolve().parents[1]
CASES = {
    "h2": ROOT / "tests/reference_data/cc/gradients/h2_shifted.json",
    "h2o": ROOT / "tests/reference_data/cc/gradients/h2o.json",
}


def _orbital_shape(inputs: dict[str, object]) -> tuple[int, int]:
    atomic_numbers = inputs["atomic_numbers"]
    shells = inputs["shells"]
    assert isinstance(atomic_numbers, list)
    assert isinstance(shells, list)
    charge = int(inputs["charge"])
    electron_count = sum(int(value) for value in atomic_numbers) - charge
    if electron_count <= 0 or electron_count % 2:
        raise ValueError("RCCSD qualification case must be a closed-shell system")
    representation = str(inputs["basis_representation"])
    nbf = 0
    for shell in shells:
        assert isinstance(shell, dict)
        angular_momentum = int(shell["angular_momentum"])
        nbf += (
            (angular_momentum + 1) * (angular_momentum + 2) // 2
            if representation == "cartesian"
            else 2 * angular_momentum + 1
        )
    nocc = electron_count // 2
    return nocc, nbf - nocc


def run_case(name: str, device: str, budget: int) -> dict:
    reference = json.loads(CASES[name].read_text())
    inputs = reference["inputs"]
    nocc, nvir = _orbital_shape(inputs)
    atoms = list(zip(inputs["atomic_numbers"], inputs["coordinates"], strict=True))
    calculator = Calculator(
        method="rccsd",
        basis="sto-3g",
        device=device,
        max_iterations=200,
        energy_tolerance=1e-13,
        density_tolerance=1e-11,
        ccsd_max_iterations=150,
        ccsd_energy_tolerance=1e-13,
        ccsd_residual_tolerance=1e-11,
        correlation_memory_budget_bytes=budget,
    )
    started = time.perf_counter()
    try:
        result = calculator.singlepoint(atoms, properties=("energy",))
    except (MemoryError, RuntimeError, ValueError) as error:
        return {
            "case": name,
            "device": device,
            "budget_bytes": budget,
            "nocc": nocc,
            "nvir": nvir,
            "status": "rejected",
            "elapsed_seconds": time.perf_counter() - started,
            "reason": f"{type(error).__name__}: {error}",
        }
    elapsed = time.perf_counter() - started
    diagnostic = result.correlation
    performance = result.cc_performance
    assert diagnostic is not None and performance is not None
    return {
        "case": name,
        "device": device,
        "budget_bytes": budget,
        "nocc": nocc,
        "nvir": nvir,
        "status": "success",
        "elapsed_seconds": elapsed,
        "energy": result.energy,
        "reference_energy": reference["total_energy"],
        "energy_absolute_error": abs(result.energy - reference["total_energy"]),
        "correlation_energy": diagnostic.ccsd_correlation_energy,
        "correlation_absolute_error": abs(
            diagnostic.ccsd_correlation_energy - reference["correlation_energy"]
        ),
        "iterations": diagnostic.ccsd_iterations,
        "energy_change": diagnostic.ccsd_energy_change,
        "r1_max": diagnostic.ccsd_singles_residual_max,
        "r2_max": diagnostic.ccsd_doubles_residual_max,
        "replay_r1_max": diagnostic.ccsd_replay_singles_residual_max,
        "replay_r2_max": diagnostic.ccsd_replay_doubles_residual_max,
        "numeric_capacity_bytes": diagnostic.numeric_capacity_bytes,
        "device_owned_bytes": diagnostic.correlation_owned_device_bytes,
        "provider_retained_bytes": diagnostic.correlation_provider_retained_bytes,
        "setup_h2d_bytes": diagnostic.ccsd_setup_h2d_bytes,
        "scalar_d2h_bytes": diagnostic.ccsd_scalar_d2h_bytes,
        "amplitude_d2h_bytes": diagnostic.ccsd_amplitude_d2h_bytes,
        "synchronizations": diagnostic.ccsd_synchronizations,
        "reference_seconds": performance.reference_seconds,
        "problem_seconds": performance.problem_seconds,
        "provider_seconds": performance.provider_seconds,
        "source_seconds": performance.source_seconds,
        "solver_seconds": performance.solver_seconds,
        "iteration_seconds": performance.iteration_seconds,
        "replay_seconds": performance.replay_seconds,
        "update_seconds": performance.update_seconds,
        "diis_seconds": performance.diis_seconds,
        "source_scans": performance.source_scans,
        "source_reads": performance.source_reads,
        "source_values": performance.source_values,
        "transform_fmas": performance.transform_fmas,
        "transform_stages": performance.transform_stages,
        "mo_blocks": performance.mo_blocks,
        "cuda_transform_calls": performance.cuda_transform_calls,
        "cuda_batch_calls": performance.cuda_batch_calls,
        "iteration_graph_calls": performance.iteration_graph_calls,
        "replay_graph_calls": performance.replay_graph_calls,
        "update_calls": performance.update_calls,
        "generated_error_checks": performance.generated_error_checks,
        "diis_gram_calls": performance.diis_gram_calls,
        "diis_coefficient_calls": performance.diis_coefficient_calls,
        "diis_combine_calls": performance.diis_combine_calls,
        "equation_hash": diagnostic.equation_hash,
        "replay_equation_hash": diagnostic.ccsd_replay_equation_hash,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cpu", "cuda"), required=True)
    parser.add_argument("--budget", type=int, action="append", required=True)
    parser.add_argument("--case", choices=tuple(CASES), action="append")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    cases = tuple(args.case or CASES)
    rows = [
        run_case(case, args.device, budget) for budget in args.budget for case in cases
    ]
    library = _native.load_library(device=args.device)
    library.vibeqc_get_source_identity.argtypes = []
    library.vibeqc_get_source_identity.restype = ctypes.c_char_p
    record = {
        "source_identity": library.vibeqc_get_source_identity().decode(),
        "schema": "vibeqc.rccsd.native-public-qualification/1",
        "issue": 149,
        "device": args.device,
        "cases": list(cases),
        "budgets_bytes": args.budget,
        "results": rows,
        "note": "Endpoint timing is observational; this artifact makes no performance-leadership claim.",
        "agent": "ChatGPT",
        "model": "GPT-5.6 Sol",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
