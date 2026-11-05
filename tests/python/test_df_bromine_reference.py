"""Issue #2979: heavy-element DF reference and complete MP2 force regressions."""

from __future__ import annotations

import os

import numpy as np
import pytest
from generativeqc import Calculator

from tools.generativeqc_validation.correlation_df_oracle import basis_snapshot, energies

CASES = (
    (
        (
            (8, (46.221228696702, 49.405870357657854, 16.93559376767291)),
            (1, (45.94139649162088, 47.72233202096395, 16.32037345384337)),
            (1, (47.90680710485766, 50.07233935713916, 16.86000479477521)),
            (1, (45.021724347057415, 50.65800206315583, 16.402255650720253)),
            (35, (41.32001467971134, 50.86279135687009, 13.134881162384028)),
        ),
        (-2648.259276582224, -2649.0447529563503),
    ),
    (
        (
            (8, (-1.082877298321221, 0.7346556172335305, 0.004185743559358396)),
            (15, (0.33711958448427715, -1.477988859549035, -0.0030178927214709133)),
            (35, (5.425628623481428, 0.07240296630818996, 0.0019218513631336527)),
        ),
        (-2987.2890259509, -2988.099554597799),
    ),
)


@pytest.fixture(scope="module", autouse=True)
def allocated_cuda() -> None:
    """Never run a real-device regression on an unallocated development host."""
    if os.environ.get("GENERATIVEQC_FULL_DF_CUDA_TEST") != "1":
        pytest.skip("requires allocated full-DF CUDA qualification")
    assert os.environ.get("SLURM_JOB_ID"), "real GPU tests must run through srun"
    pyscf = pytest.importorskip("pyscf")
    assert pyscf.__version__ == "2.14.0"


@pytest.mark.parametrize("case", (0, 1))
@pytest.mark.parametrize("cardinal", (2, 3))
def test_bromine_df_reference_complete_mp2_force(case: int, cardinal: int) -> None:
    """Use native CUDA throughout and independently reconverge displaced energies.

    Coordinates and HF values are from the pinned public #2979 evidence bundle
    (Gist 9d0e6972b39d3b742de629ec488e434f, revision
    fc579b7f85d9a1c5f260d2801d136076941bc2a2). Keep the original translation:
    recentering the first case would change its historical numerical input.
    """
    atoms, reference_energies = CASES[case]
    orbital = "cc-pvdz" if cardinal == 2 else "cc-pvtz"
    auxiliary = orbital + "-jkfit"
    elements = [number for number, _ in atoms]
    calc = Calculator(
        method="mp2",
        basis=basis_snapshot(orbital, elements),
        auxiliary_basis=basis_snapshot(auxiliary, elements),
        density_fitting="full_cuda",
        device="cuda",
        initial_guess=None,
        correlation_memory_budget_bytes=32 << 30,
        max_iterations=100,
    )
    try:
        actual = calc.singlepoint_components(
            atoms, companions=("rhf",), properties=("energy", "forces")
        )
        for result in actual.values():
            assert result.converged and result.executed_backend == "cuda"
            assert result.forces is not None and np.isfinite(result.forces).all()
            np.testing.assert_allclose(np.sum(result.forces, axis=0), 0, atol=2e-7)
        np.testing.assert_allclose(
            actual["rhf"].energy, reference_energies[cardinal - 2], atol=2e-8, rtol=0
        )
        expected = energies(atoms, orbital, auxiliary, df_reference=True)
        np.testing.assert_allclose(
            actual["mp2"].energy, expected["mp2"], atol=5e-8, rtol=0
        )
        direction = np.random.default_rng(2979 + case).normal(size=(len(atoms), 3))
        direction -= direction.mean(axis=0)
        direction /= np.linalg.norm(direction)
        for step in (1e-3, 3e-4):
            displaced = []
            for sign in (-1, 1):
                moved = [
                    (number, np.asarray(position) + sign * step * shift)
                    for (number, position), shift in zip(atoms, direction, strict=True)
                ]
                displaced.append(energies(moved, orbital, auxiliary, df_reference=True))
            for method in ("rhf", "mp2"):
                derivative = (displaced[1][method] - displaced[0][method]) / (2 * step)
                np.testing.assert_allclose(
                    -np.vdot(actual[method].forces, direction),
                    derivative,
                    atol=5e-6,
                    rtol=0,
                )
    finally:
        calc.clear_cache()
