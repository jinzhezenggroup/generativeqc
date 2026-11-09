"""Reproduce the retained DZ root scan inside a finite Slurm GPU allocation."""

import os

from generativeqc import Calculator
from tools.generativeqc_validation.correlation_df_oracle import basis_snapshot, energies

assert os.environ.get("SLURM_JOB_ID"), "run this GPU probe through finite srun"

for distance in (1.6, 2.2, 2.4, 2.8):
    atoms = ((6, (0., 0., 0.)), (1, (0.3, -0.4, distance)))
    owner = Calculator(
        method="mp2", basis=basis_snapshot("aug-cc-pvdz", [6, 1]),
        auxiliary_basis=basis_snapshot("aug-cc-pvdz-jkfit", [6, 1]),
        device="cuda", density_fitting="full_cuda", precision="fp64",
        energy_tolerance=1e-12, density_tolerance=1e-11, max_iterations=200,
        initial_guess=None, correlation_memory_budget_bytes=2 << 30,
    )
    try:
        result = owner.singlepoint_components(atoms, charge=1, companions=("rhf",), properties=("energy",))
        expected = energies(atoms, "aug-cc-pvdz", "aug-cc-pvdz-jkfit", df_reference=True, charge=1)
        print("DISTANCE", distance, "NATIVE", {name: value.energy for name, value in result.items()}, "ORACLE", expected, flush=True)
    except Exception as error:
        print("DISTANCE", distance, type(error).__name__, str(error), flush=True)
    finally:
        owner.clear_cache()
