"""Native Fast cache handoff, moving-grid forces and frozen density lifetime."""

from __future__ import annotations

import os

import numpy as np
import pytest
from generativeqc import PrecisionCalculator

from benchmarks._cases import benchmark_cases

pytestmark = pytest.mark.skipif(
    os.environ.get("GENERATIVEQC_RESOURCE_CUDA_TEST") != "1"
    or not os.environ.get("SLURM_JOB_ID"),
    reason="requires a finite Slurm GPU allocation",
)


@pytest.mark.parametrize("functional", ("pbe0", "b3lyp"))
def test_fast_strict_tail_preserves_complete_endpoints(
    functional: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Compare automatic, forced and streamed sources to an independent solve.

    Native Fast defaults exercise the relaxed application-budget strict tail.
    An independently displaced geometry exercises retirement of both the mixed
    lease and any forced strict lease. Each warm replay keeps its own frozen
    post-cold/post-move density; no reference density enters the native owner.
    """
    pyscf = pytest.importorskip("pyscf")
    dft = pytest.importorskip("pyscf.dft")
    pyscf.lib.num_threads(1)
    atoms = benchmark_cases()["water-tetramer-def2-svp-spherical"].atoms
    moved = tuple(
        (
            symbol,
            tuple(np.asarray(position) + ([0.003, -0.002, 0.001] if i == 1 else 0)),
        )
        for i, (symbol, position) in enumerate(atoms)
    )
    references = []
    for geometry in (atoms, moved):
        molecule = pyscf.gto.M(
            atom=geometry, basis="def2-svp", unit="Bohr", cart=False, verbose=0
        )
        reference = dft.RKS(molecule)
        reference.xc = functional
        reference.conv_tol = 1e-12
        reference.conv_tol_grad = 1e-8
        energy = reference.kernel()
        assert reference.converged
        gradient = reference.nuc_grad_method()
        gradient.grid_response = True
        references.append((energy, -gradient.kernel()))

    for selection in ("auto", "1", "0"):
        monkeypatch.setenv("GENERATIVEQC_KS_RESIDENT_VALUES", selection)
        calculator = PrecisionCalculator(
            policy="fast",
            method=f"{functional}-rks",
            device="cuda",
            basis="def2-svp",
            basis_representation="spherical",
        )
        try:
            with calculator.prepare_batch([atoms], warm_start=True) as prepared:
                for index, geometry in enumerate((atoms, moved)):
                    prepared.set_warm_start_updates(True)
                    coordinates = (
                        None
                        if index == 0
                        else [np.asarray([position for _, position in geometry])]
                    )
                    for repeat in range(2):
                        result = prepared.execute(
                            coordinates, strict=True, properties=("energy", "forces")
                        ).items[0]
                        assert result.succeeded and result.converged
                        assert result.executed_backend == "cuda"
                        energy, forces = references[index]
                        assert abs(result.energy - energy) / len(atoms) < 1e-5
                        difference = result.forces - forces
                        assert np.max(np.abs(difference)) < 1e-4, (
                            functional,
                            selection,
                            index,
                            repeat,
                        )
                        assert np.sqrt(np.mean(difference**2)) < 3e-5
                        assert result.precision_validation.solve_count == 1
                        assert result.precision["final_residual_audits"] > 0
                        if index == 0 and repeat == 0:
                            assert result.precision["mixed_stage_fock_builds"] > 0
                            assert result.precision["strict_stage_fock_builds"] >= 2
                            assert result.precision["strict_refinement_applied"]
                        if repeat:
                            assert result.warm_start_used
                        else:
                            prepared.set_warm_start_updates(False)
        finally:
            calculator.clear_cache()
