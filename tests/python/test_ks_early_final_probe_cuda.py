"""Independent forces and warm-state lifetime for final-state-aware stopping."""

import os

import numpy as np
import pytest
from generativeqc import Calculator

pytestmark = pytest.mark.skipif(
    os.environ.get("GENERATIVEQC_DFT_CUDA_TEST") != "1",
    reason="allocated CUDA KS early-final-state qualification",
)


@pytest.mark.parametrize("representation", ["cartesian", "spherical"])
@pytest.mark.parametrize("legacy", [False, True], ids=["default", "rollback"])
@pytest.mark.parametrize("tight", [False, True], ids=["native-tolerances", "tight"])
@pytest.mark.parametrize(
    "functional,spin", [("pbe", 0), ("r2scan", 0), ("pbe0", 0), ("pbe", 1)]
)
def test_physical_export_frozen_warm_and_geometry(
    representation: str,
    functional: str,
    spin: int,
    legacy: bool,
    tight: bool,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Compare independent physical endpoints; UKS retains its ordinary fallback."""
    import cupy as cp
    from gpu4pyscf.dft import rks, uks
    from pyscf import gto, lib

    lib.num_threads(1)
    if legacy:
        monkeypatch.setenv("GENERATIVEQC_KS_EARLY_FINAL_PROBE", "0")
    else:
        monkeypatch.delenv("GENERATIVEQC_KS_EARLY_FINAL_PROBE", raising=False)
    atoms = [("O", (0.0, 0.0, 0.0)), ("H", (0.0, 0.0, 1.8)), ("H", (1.7, 0.0, -0.6))]
    moved = np.array([atom[1] for atom in atoms])
    moved[-1, 0] += 0.02
    displaced = [(atom[0], tuple(xyz)) for atom, xyz in zip(atoms, moved, strict=True)]
    references = []
    for geometry in (atoms, displaced):
        molecule = gto.M(
            atom=geometry,
            unit="Bohr",
            basis="def2-svp",
            cart=representation == "cartesian",
            charge=spin,
            spin=spin,
            verbose=0,
        )
        reference = (uks.UKS if spin else rks.RKS)(molecule)
        reference.xc = functional
        # This independent numerical oracle is outside the production endpoint
        # and timing comparison. Keep stock grids, resolve the equations tightly
        # with full F[D], and never lend its density to the native solver.
        reference.conv_tol = 1e-12
        reference.conv_tol_grad = 1e-9
        reference.max_cycle = 150
        reference.direct_scf = False
        reference.kernel()
        assert reference.converged
        gradient = reference.nuc_grad_method()
        gradient.grid_response = True
        references.append((float(reference.e_tot), -cp.asnumpy(gradient.kernel())))

    tolerances = (
        {"energy_tolerance": 1e-12, "density_tolerance": 1e-10} if tight else {}
    )
    calculator = Calculator(
        method=f"{functional}-{'uks' if spin else 'rks'}",
        basis="def2-svp",
        basis_representation=representation,
        device="cuda",
        **tolerances,
    )
    try:
        with calculator.prepare_batch(
            [atoms], charges=[spin], multiplicities=[spin + 1], warm_start=True
        ) as owner:
            for geometry in range(2):
                owner.set_warm_start_updates(True)
                for repeat in range(2):
                    result = owner.execute(
                        None if not geometry else [moved],
                        properties=("energy", "forces"),
                        strict=True,
                    ).items[0]
                    energy, forces = references[geometry]
                    assert result.converged and result.executed_backend == "cuda"
                    assert abs(result.energy - energy) < 1e-7 * len(atoms)
                    assert np.max(np.abs(result.forces - forces)) < 1e-6
                    if tight:
                        assert result.ks_diagnostic.density_change_max < 1e-10
                    if repeat:
                        assert result.warm_start_used
                    else:
                        owner.set_warm_start_updates(False)
    finally:
        calculator.clear_cache()
