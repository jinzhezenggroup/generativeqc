"""Check complete HF endpoints after composing PR #2530 with current master.

Run inside a finite Slurm GPU allocation with GENERATIVEQC_LIBRARY set. Small
RHF/UHF cases use live CPU PySCF; the 384-AO regression uses the independent
reference arrays retained with the original qualification. No oracle density
is passed to the native solver. This is a correctness check, not a benchmark.
"""

from __future__ import annotations

import ctypes
import hashlib
import json
import lzma
import os
from pathlib import Path

import numpy as np
from generativeqc import Calculator
from pyscf import gto, lib, scf


def emit(value: dict) -> None:
    """Retain full returned force arrays as well as scalar gates and work."""
    print(json.dumps(value, allow_nan=False), flush=True)


def qualify(
    method: str,
    atoms: list,
    moved: list,
    multiplicity: int,
    references: list[dict],
    guess: str | None,
    label: str,
) -> None:
    """Check OFF/on cold, own-density warm, and moved endpoints at strict gates."""
    for incremental in (False, True):
        os.environ["GENERATIVEQC_INCREMENTAL_DIRECT_JK"] = str(int(incremental))
        os.environ["GENERATIVEQC_INCREMENTAL_DIRECT_JK_REBUILD_INTERVAL"] = "8"
        calculator = Calculator(
            method=method,
            basis="def2-svp",
            basis_representation="spherical",
            device="cuda",
            precision="fp64",
            initial_guess=guess,
            energy_tolerance=1e-12,
            density_tolerance=1e-9,
            screening_tolerance=1e-12,
            max_iterations=100,
        )
        try:
            with calculator.prepare_batch(
                [atoms], multiplicities=[multiplicity], warm_start=True
            ) as owner:
                for phase, geometry, reference in zip(
                    ("cold", "warm", "moved"),
                    (atoms, atoms, moved),
                    (references[0], references[0], references[1]),
                    strict=True,
                ):
                    coordinates = np.asarray([position for _, position in geometry])
                    result = owner.execute(
                        [coordinates], strict=True, properties=("energy", "forces")
                    ).items[0]
                    energy_error = abs(result.energy - reference["energy"])
                    force_error = float(
                        np.max(np.abs(result.forces - reference["forces"]))
                    )
                    attempts = result.initial_guess
                    # An automatic preliminary guess may retry the target; the
                    # retained large regression must close within one attempt.
                    single_attempt = (attempts is None and guess is None) or (
                        attempts is not None
                        and attempts["target_attempts"] == 1
                        and attempts["discarded_target_iterations"] == 0
                    )
                    gate = bool(
                        result.converged
                        and result.executed_backend == "cuda"
                        and energy_error <= 1e-8
                        and force_error <= 1e-7
                        and not result.warm_start_fallback
                        and (phase == "cold" or result.warm_start_used)
                        and single_attempt
                    )
                    emit(
                        {
                            "label": label,
                            "incremental": incremental,
                            "phase": phase,
                            "energy": float(result.energy),
                            "forces": result.forces.tolist(),
                            "energy_error": energy_error,
                            "force_error": force_error,
                            "iterations": result.iterations,
                            "warm_used": result.warm_start_used,
                            "fallback": result.warm_start_fallback,
                            "work": result.incremental_direct_jk,
                            "precision": result.precision,
                            "initial_guess": result.initial_guess,
                            "gate": gate,
                        }
                    )
                    assert gate, (label, incremental, phase)
        finally:
            calculator.clear_cache()


def main() -> None:
    """Run independent small oracles and the retained large moved regression."""
    assert os.environ.get("SLURM_JOB_ID") and "CUDA_VISIBLE_DEVICES" in os.environ
    library_path = Path(os.environ["GENERATIVEQC_LIBRARY"])
    library = ctypes.CDLL(str(library_path.resolve()))
    library.generativeqc_get_source_identity.restype = ctypes.c_char_p
    emit(
        {
            "job": os.environ["SLURM_JOB_ID"],
            "visibility": os.environ["CUDA_VISIBLE_DEVICES"],
            "source_identity": library.generativeqc_get_source_identity().decode(),
            "library_sha256": hashlib.sha256(library_path.read_bytes()).hexdigest(),
        }
    )
    os.environ["GENERATIVEQC_FINAL_FOCK_REBUILD"] = "0"
    lib.num_threads(4)
    for method, atoms, multiplicity in (
        (
            "rhf",
            [
                ["O", [0.0, 0.0, 0.0]],
                ["H", [0.0, 1.43, 1.11]],
                ["H", [0.0, -1.43, 1.11]],
            ],
            1,
        ),
        ("uhf", [["O", [0.0, 0.0, 0.0]], ["H", [0.0, 0.0, 1.8]]], 2),
    ):
        moved = [[symbol, list(position)] for symbol, position in atoms]
        moved[-1][1][2] += 0.03
        references = []
        for geometry in (atoms, moved):
            molecule = gto.M(
                atom=geometry,
                basis="def2-svp",
                unit="Bohr",
                spin=multiplicity - 1,
                verbose=0,
            )
            oracle = scf.RHF(molecule) if method == "rhf" else scf.UHF(molecule)
            oracle.conv_tol, oracle.conv_tol_grad = 1e-13, 1e-10
            oracle.max_cycle, oracle.direct_scf_tol = 100, 1e-14
            oracle.kernel()
            assert oracle.converged
            references.append(
                {
                    "energy": float(oracle.e_tot),
                    "forces": (-oracle.nuc_grad_method().kernel()).tolist(),
                }
            )
        emit({"label": method, "independent_references": references})
        qualify(method, atoms, moved, multiplicity, references, None, method)

    reference_path = Path(__file__).resolve().parents[1] / (
        "incremental-hf-full-energy-refinement-20261015/validation.json.xz"
    )
    retained = json.loads(lzma.decompress(reference_path.read_bytes()))["references"][
        "384"
    ]["data"]
    emit(
        {
            "reference_file_sha256": hashlib.sha256(
                reference_path.read_bytes()
            ).hexdigest()
        }
    )
    qualify(
        "rhf",
        retained["atoms"],
        retained["moved"],
        1,
        retained["reference"],
        "auto",
        "rhf-384",
    )


if __name__ == "__main__":
    main()
