"""Optimize water's Gaussian exponents and contractions through the public Torch API.

Run with a built library, PYTHONPATH=python:., and the torch extra installed.
The optional --reference flag verifies the resulting raw basis with PySCF;
PySCF never supplies optimizer energies or gradients. CUDA runs require srun.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from generativeqc.experimental.integral_autodiff import IntegralHF
from generativeqc.torch import integral_energy

COORDINATES = np.array([[0.0, 0.0, 0.0], [0.0, -1.43, 1.11], [0.0, 1.43, 1.11]])


def reference_energy(
    owner: IntegralHF, alpha: np.ndarray, coefficients: np.ndarray
) -> float:
    """Evaluate an independent converged RHF using exactly the optimized raw basis."""
    from pyscf import gto, scf

    labels = ("O0", "H1", "H2")
    basis: dict[str, list[Any]] = {label: [] for label in labels}
    begin = 0
    for shell in owner.shells:
        size = len(shell.primitives)
        basis[labels[shell.atom_index]].append(
            [
                shell.angular_momentum,
                *[
                    [float(alpha[i]), float(coefficients[i])]
                    for i in range(begin, begin + size)
                ],
            ]
        )
        begin += size
    molecule = gto.M(
        atom=list(zip(labels, COORDINATES, strict=True)),
        basis=basis,
        unit="Bohr",
        cart=True,
        verbose=0,
    )
    solver = scf.RHF(molecule)
    solver.conv_tol, solver.conv_tol_grad = 1e-13, 1e-11
    energy = solver.kernel()
    if not solver.converged:
        raise RuntimeError("independent optimized-basis reference did not converge")
    return float(energy)


def optimize_basis(
    *,
    steps: int = 6,
    backend: str = "cpu",
    storage: str = "dense",
    cache: str = ".artifacts/basis-optimization",
    reference: bool = False,
) -> dict[str, Any]:
    """Run bounded Armijo descent and return all accepted/trial endpoint evidence.

    Logarithmic exponents keep positivity in the differentiable graph. Signed
    raw contractions retain the solver's differentiated normalization. A finite
    step limit describes a demonstration, not a certificate of optimality.
    """
    if type(steps) is not int or steps < 1 or steps > 100:
        raise ValueError("steps must be between 1 and 100")
    start = time.perf_counter()
    owner = IntegralHF(
        ("O", "H", "H"),
        derivative_order=2,
        backend=backend,
        integral_storage=storage,
        cache=cache,
    )
    setup_seconds = time.perf_counter() - start
    records: list[dict[str, Any]] = []
    initial_alpha, initial_coefficients = (
        owner.exponents.copy(),
        owner.coefficients.copy(),
    )
    parameters = (np.log(initial_alpha), initial_coefficients.copy())
    coordinates = torch.tensor(COORDINATES, dtype=torch.float64)

    def evaluate(
        values: tuple[np.ndarray, np.ndarray],
    ) -> tuple[float, tuple[np.ndarray, np.ndarray]]:
        tensors = tuple(
            torch.tensor(value, dtype=torch.float64, requires_grad=True)
            for value in values
        )
        started = time.perf_counter()
        energy = integral_energy(coordinates, tensors[0].exp(), tensors[1], owner)
        gradients = torch.autograd.grad(energy, tensors)
        records.append(
            {
                "energy": float(energy.detach()),
                "seconds": time.perf_counter() - started,
                "work": owner.diagnostics(),
            }
        )
        return float(energy.detach()), tuple(
            value.detach().numpy() for value in gradients
        )

    try:
        energy, gradients = evaluate(parameters)
        initial_energy = energy
        accepted = [energy]
        failures = []
        for step in range(steps):
            norm = float(np.sqrt(sum(np.sum(value**2) for value in gradients)))
            if norm < 1e-6:
                break
            scale = max(1.0, norm)
            direction = tuple(-value / scale for value in gradients)
            slope = -(norm**2) / scale
            length = 0.1
            for trial in range(12):
                candidate = tuple(
                    value + length * delta
                    for value, delta in zip(parameters, direction, strict=True)
                )
                try:
                    candidate_energy, candidate_gradient = evaluate(candidate)
                except (ArithmeticError, RuntimeError, ValueError) as error:
                    failures.append(
                        {"step": step, "trial": trial, "reason": str(error)}
                    )
                else:
                    if candidate_energy <= energy + 1e-4 * length * slope:
                        parameters, energy, gradients = (
                            candidate,
                            candidate_energy,
                            candidate_gradient,
                        )
                        accepted.append(energy)
                        break
                length *= 0.5
            else:
                raise RuntimeError(
                    "basis line search exhausted its finite trial budget"
                )
        result = {
            "method": "rhf",
            "backend": backend,
            "integral_storage": storage,
            "coordinates_bohr": COORDINATES.tolist(),
            "initial_energy_hartree": initial_energy,
            "final_energy_hartree": energy,
            "accepted_energies_hartree": accepted,
            "exponents": np.exp(parameters[0]).tolist(),
            "coefficients": parameters[1].tolist(),
            "setup_seconds": setup_seconds,
            "total_seconds": time.perf_counter() - start,
            "endpoint_evaluations": records,
            "rejected_execution_trials": failures,
            "gradient_norm": float(
                np.sqrt(sum(np.sum(value**2) for value in gradients))
            ),
            "numeric_peak_bytes_bound": owner.numeric_peak_bytes,
            "compiler_cache_policy": "reuse existing artifacts and verified compiler cache",
        }
        if reference:
            oracle_initial = reference_energy(
                owner, initial_alpha, initial_coefficients
            )
            oracle_final = reference_energy(owner, np.exp(parameters[0]), parameters[1])
            errors = [abs(initial_energy - oracle_initial), abs(energy - oracle_final)]
            if max(errors) > 5e-8:
                raise AssertionError(f"independent basis-energy gate failed: {errors}")
            result["independent_reference"] = {
                "engine": "PySCF",
                "absolute_errors": errors,
                "acceptance_hartree": 5e-8,
            }
        return result
    finally:
        owner.close()


def main() -> None:
    """Write reproducible endpoint evidence for the requested optimization run."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=6)
    parser.add_argument("--backend", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--storage", choices=("dense", "direct"), default="dense")
    parser.add_argument("--reference", action="store_true")
    parser.add_argument("--output", type=Path)
    arguments = parser.parse_args()
    report = optimize_basis(
        steps=arguments.steps,
        backend=arguments.backend,
        storage=arguments.storage,
        reference=arguments.reference,
    )
    text = json.dumps(report, indent=2) + "\n"
    if arguments.output:
        arguments.output.parent.mkdir(parents=True, exist_ok=True)
        arguments.output.write_text(text)
    else:
        print(text, end="")


if __name__ == "__main__":
    main()
