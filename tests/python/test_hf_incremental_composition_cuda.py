"""Qualify delta J/K with ordinary convergence, phase sources and finalization."""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

import numpy as np
import pytest

if TYPE_CHECKING:
    from numpy.typing import NDArray

Geometry = list[tuple[str, tuple[float, float, float]]]
References = tuple[Geometry, Geometry, list[tuple[float, "NDArray[np.float64]"]]]

pytestmark = pytest.mark.skipif(
    os.environ.get("GENERATIVEQC_DFT_CUDA_TEST") != "1",
    reason="requires a finite Slurm GPU allocation and independent CPU PySCF",
)


@pytest.fixture(scope="module")
def references() -> References:
    """Build independent full-density references; never supply oracle SCF seeds."""
    from pyscf import gto, lib, scf

    assert os.environ.get("SLURM_JOB_ID")
    lib.num_threads(1)
    atoms = [
        (symbol, (position[0] + 7.0 * water, position[1], position[2]))
        for water in range(4)
        for symbol, position in (
            ("O", (0.0, 0.0, 0.0)),
            ("H", (0.0, 0.0, 1.8)),
            ("H", (1.7, 0.0, -0.6)),
        )
    ]
    moved = [(symbol, tuple(position)) for symbol, position in atoms]
    symbol, position = moved[2]
    moved[2] = (symbol, (position[0] + 0.02, position[1], position[2]))
    values = []
    for geometry in (atoms, moved):
        molecule = gto.M(
            atom=geometry, unit="Bohr", basis="def2-svp", cart=False, verbose=0
        )
        solver = scf.RHF(molecule)
        solver.direct_scf = False
        solver.conv_tol = 1e-12
        solver.conv_tol_grad = 1e-9
        solver.max_cycle = 150
        solver.kernel()
        assert solver.converged
        values.append((float(solver.e_tot), -solver.nuc_grad_method().kernel()))
    return atoms, moved, values


@pytest.mark.parametrize("resident", ["0", "auto"])
@pytest.mark.parametrize("strict", [False, True])
def test_incremental_complete_endpoints(
    references: References, monkeypatch: pytest.MonkeyPatch, resident: str, strict: bool
) -> None:
    """Full and delta schedules share controls, independent E/F gates and reuse.

    Ninety-six public AOs admit the bounded ordinary resident source. Resource
    refusal remains legal: it must preserve the same numerical endpoint via
    direct recomputation rather than weaken the acceptance contract.
    """
    from generativeqc import Calculator

    atoms, moved, values = references
    monkeypatch.setenv("GENERATIVEQC_RHF_RESIDENT_VALUES", resident)
    monkeypatch.setenv("GENERATIVEQC_FINAL_FOCK_REBUILD", "0")
    monkeypatch.setenv("GENERATIVEQC_INCREMENTAL_DIRECT_JK_REBUILD_INTERVAL", "8")
    controls = {"energy_tolerance": 1e-12, "density_tolerance": 1e-9} if strict else {}
    for selector in ("0", "1", None):
        incremental = selector == "1"
        if selector is None:
            monkeypatch.delenv("GENERATIVEQC_INCREMENTAL_DIRECT_JK", raising=False)
        else:
            monkeypatch.setenv("GENERATIVEQC_INCREMENTAL_DIRECT_JK", selector)
        calculator = Calculator(
            method="rhf",
            basis="def2-svp",
            basis_representation="spherical",
            device="cuda",
            precision="fp64",
            initial_guess="auto",
            **controls,
        )
        assert calculator._energy_tolerance == (1e-12 if strict else 1e-9)
        assert calculator._density_tolerance == (1e-9 if strict else 1e-5)
        with calculator.prepare_batch([atoms], warm_start=True) as owner:
            owner.set_warm_start_updates(True)
            for phase, geometry, reference in (
                ("cold", None, values[0]),
                ("warm", atoms, values[0]),
                ("moved", moved, values[1]),
            ):
                coordinates = (
                    None
                    if geometry is None
                    else [np.asarray([xyz for _, xyz in geometry])]
                )
                point = owner.execute(
                    coordinates, properties=("energy", "forces"), strict=True
                ).items[0]
                assert point.converged and point.executed_backend == "cuda"
                assert point.incremental_direct_jk["requested"] == incremental
                assert abs(point.energy - reference[0]) <= (
                    1e-8 if strict else len(atoms) * 2.5e-6
                )
                error = np.asarray(point.forces) - reference[1]
                assert np.isfinite(error).all()
                assert np.max(np.abs(error)) <= (1e-7 if strict else 2.5e-5)
                assert np.sqrt(np.mean(error**2)) <= (1e-7 if strict else 7.5e-6)
                assert not point.warm_start_fallback
                if phase == "cold":
                    owner.set_warm_start_updates(False)
                    assert point.precision["post_scf_fock_builds"] == 0
                    if incremental:
                        diagnostic = point.incremental_direct_jk
                        assert diagnostic["active"] and diagnostic["delta_builds"] > 0
                        assert diagnostic["anchor_full_builds"] >= 3
                elif phase == "warm" and incremental:
                    assert point.precision["post_scf_fock_builds"] == 0
        calculator.clear_cache()
