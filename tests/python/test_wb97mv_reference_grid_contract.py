"""The reference's SCF and response must integrate the same stated model."""

from __future__ import annotations

import sys
from types import SimpleNamespace

import numpy as np
import pytest
from generativeqc import GridSpec

from benchmarks.readme_wb97mv import reference_engine


@pytest.mark.parametrize("spin", (0, 1))
@pytest.mark.parametrize("nlc_spec", (None, GridSpec(6, 4, 8)))
def test_reference_response_uses_the_primary_moving_quadrature(
    monkeypatch: pytest.MonkeyPatch, spin: int, nlc_spec: GridSpec | None
) -> None:
    """A hidden default SG1 response grid cannot satisfy the common-grid gate."""
    atoms = [("O", (0.0, 0.0, 0.0)), ("H", (0.1, 0.2, 1.5))]
    original_becke = object()
    observed_orders = []

    class Molecule:
        def __init__(self, **settings: object) -> None:
            self.natm = len(atoms)
            self.spin = settings["spin"]

        def atom_symbol(self, index: int) -> str:
            return atoms[index][0]

        def atom_charge(self, index: int) -> int:
            return 8 if index == 0 else 1

    class Engine:
        def __init__(self, molecule: Molecule) -> None:
            self.mol = molecule
            self.grids = SimpleNamespace()
            self.nlcgrids = SimpleNamespace()
            self._numint = SimpleNamespace(
                block_loop=lambda *, strict_grid_order=False: (
                    observed_orders.append(strict_grid_order) or iter(())
                )
            )
            self.cphf_grids = SimpleNamespace(
                prune="SG1",
                gen_atomic_grids=lambda *args: {"H": (np.zeros((1, 3)), np.ones(1))},
            )

    dft = SimpleNamespace(
        gen_grid=SimpleNamespace(original_becke=original_becke),
        rks=SimpleNamespace(RKS=Engine),
        uks=SimpleNamespace(UKS=Engine),
    )
    monkeypatch.setitem(sys.modules, "cupy", np)
    monkeypatch.setitem(sys.modules, "gpu4pyscf", SimpleNamespace(dft=dft))
    monkeypatch.setitem(sys.modules, "gpu4pyscf.dft", dft)
    monkeypatch.setitem(
        sys.modules, "pyscf", SimpleNamespace(gto=SimpleNamespace(M=Molecule))
    )
    engine = reference_engine(
        atoms, "sto-3g", GridSpec(8, 4, 8), spin=spin, nlc_spec=nlc_spec
    )
    assert list(engine._numint.block_loop()) == []
    assert observed_orders and all(observed_orders)
    primary = engine.grids.gen_atomic_grids(engine.mol)
    for grid in (engine.cphf_grids, engine.nlcgrids):
        assert grid.radii_adjust is None
        assert grid.prune is None
        assert grid.becke_scheme is original_becke
        actual = grid.gen_atomic_grids(engine.mol)
        assert set(actual) == set(primary) == {"H", "O"}
        for element in primary:
            if grid is engine.nlcgrids and nlc_spec is not None:
                assert len(actual[element][0]) == 6 * 4 * 8
                assert np.isfinite(actual[element][0]).all()
                assert np.isfinite(actual[element][1]).all()
            else:
                np.testing.assert_array_equal(actual[element][0], primary[element][0])
                np.testing.assert_array_equal(actual[element][1], primary[element][1])
                assert len(actual[element][0]) == 8 * 4 * 8
