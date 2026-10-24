"""Cold native seeds must acquire a matching SCF determinant before force export."""

from __future__ import annotations

import os
from typing import Any

import numpy as np
import pytest

pytestmark = pytest.mark.skipif(
    os.environ.get("GENERATIVEQC_RESOURCE_CUDA_TEST") != "1"
    or not os.environ.get("SLURM_JOB_ID"),
    reason="requires a finite Slurm GPU allocation",
)

WATER = [("O", (0.0, 0.0, 0.0)), ("H", (0.0, 0.0, 1.8)), ("H", (1.7, 0.0, -0.6))]


@pytest.mark.parametrize("representation", ("spherical", "cartesian"))
@pytest.mark.parametrize("tight", (False, True), ids=("default", "strict"))
def test_native_cold_frame_matches_independent_forces(
    representation: str, tight: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Exercise cold acquisition, frozen replay, moved fallback and rollback.

    The independent CPU oracle supplies no density to production. Public work
    counters check the removed post-SCF rebuild, while the energy/force checks
    protect the determinant consumed by every derivative and the returned seed.
    """
    from generativeqc import Calculator
    from pyscf import gto, lib, scf

    lib.num_threads(1)
    moved = [
        (s, tuple(np.array(x) + ([0.025, 0, 0] if i == 2 else [0, 0, 0])))
        for i, (s, x) in enumerate(WATER)
    ]
    oracles = []
    for atoms in (WATER, moved):
        mol = gto.M(
            atom=atoms,
            basis="def2-svp",
            unit="Bohr",
            cart=representation == "cartesian",
            verbose=0,
        )
        reference = scf.RHF(mol)
        reference.conv_tol, reference.conv_tol_grad = 1e-12, 1e-10
        reference.kernel()
        assert reference.converged
        oracles.append((reference.e_tot, -reference.nuc_grad_method().kernel()))

    options: dict[str, Any] = {
        "method": "rhf",
        "basis": "def2-svp",
        "basis_representation": representation,
        "device": "cuda",
        "initial_guess": "auto",
    }
    if tight:
        options.update(energy_tolerance=1e-12, density_tolerance=1e-10)

    def compare(item: Any, expected: Any) -> None:
        assert item.converged and item.executed_backend == "cuda"
        assert abs(item.energy - expected[0]) < 1e-7
        np.testing.assert_allclose(
            item.forces, expected[1], atol=1e-6 if tight else 2.5e-5, rtol=0
        )

    results = []
    for rebuild in (False, True):
        monkeypatch.setenv("GENERATIVEQC_FINAL_FOCK_REBUILD", "1" if rebuild else "0")
        calculator = Calculator(**options)
        try:
            with calculator.prepare_batch([WATER], warm_start=True) as owner:
                cold = owner.execute(
                    properties=("energy", "forces"), strict=True
                ).items[0]
                compare(cold, oracles[0])
                assert cold.initial_guess["kind"] == "huckel"
                assert cold.initial_guess["outcome"] == "used"
                assert cold.iterations > 1
                assert cold.precision["operator_work_counters_valid"]
                if not rebuild:
                    assert cold.precision["post_scf_fock_builds"] == 0
                results.append(cold)
                owner.set_warm_start_updates(False)
                for _ in range(2):
                    compare(
                        owner.execute(
                            properties=("energy", "forces"), strict=True
                        ).items[0],
                        oracles[0],
                    )
                coordinates = [[x for _, x in moved]]
                compare(
                    owner.execute(
                        coordinates, properties=("energy", "forces"), strict=True
                    ).items[0],
                    oracles[1],
                )
        finally:
            calculator.clear_cache()
    np.testing.assert_allclose(
        results[0].forces, results[1].forces, atol=1e-6 if tight else 2.5e-5, rtol=0
    )
    assert results[0].fock_builds < results[1].fock_builds


def test_unproven_cold_guess_cannot_publish_after_one_iteration() -> None:
    """Without an energy baseline the first proposal cannot license forces."""
    from generativeqc import Calculator

    calculator = Calculator(
        method="rhf",
        basis="def2-svp",
        device="cuda",
        initial_guess="auto",
        max_iterations=1,
    )
    try:
        with calculator.prepare_batch([WATER], warm_start=True) as owner:
            item = owner.execute(properties=("energy", "forces"), strict=False).items[0]
            assert not item.converged
            assert item.forces is None
            assert not item.warm_start_used
    finally:
        calculator.clear_cache()
