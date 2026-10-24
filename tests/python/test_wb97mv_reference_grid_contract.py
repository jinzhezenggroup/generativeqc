"""The reference's SCF and response must integrate the same stated model."""

from __future__ import annotations

import sys
from types import SimpleNamespace

import numpy as np
import pytest
from generativeqc import GridSpec

from benchmarks.readme_wb97mv import reference_engine


@pytest.mark.parametrize("spin", (0, 1))
@pytest.mark.parametrize("pruning", ("none", "radial-v1"))
def test_reference_response_uses_the_primary_moving_quadrature(
    monkeypatch: pytest.MonkeyPatch, spin: int, pruning: str
) -> None:
    """A hidden default SG1 response grid cannot satisfy the common-grid gate."""
    atoms = [("O", (0.0, 0.0, 0.0)), ("H", (0.1, 0.2, 1.5))]
    original_becke = object()

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
    spec = GridSpec(
        8,
        4,
        8,
        ((1, 0.8), (8, 1.4)),
        version=1 if pruning == "none" else 3,
        pruning=pruning,
    )
    engine = reference_engine(atoms, "sto-3g", spec, spin=spin)
    primary = engine.grids.gen_atomic_grids(engine.mol)
    for grid in (engine.cphf_grids, engine.nlcgrids):
        assert grid.radii_adjust is None
        assert grid.prune is None
        assert grid.becke_scheme is original_becke
        actual = grid.gen_atomic_grids(engine.mol)
        assert set(actual) == set(primary) == {"H", "O"}
        for element in primary:
            np.testing.assert_array_equal(actual[element][0], primary[element][0])
            np.testing.assert_array_equal(actual[element][1], primary[element][1])
            assert len(actual[element][0]) == spec.points_per_atom
    from generativeqc_compiler.dft.grid import MolecularGrid

    native_rule = MolecularGrid(atoms, spec)
    raw = list(native_rule._raw_tiles(173))
    points = np.concatenate([tile.points for tile in raw])
    weights = np.concatenate([tile.weights for tile in raw])
    owners = np.concatenate([tile.owners for tile in raw])
    for owner, (element, center) in enumerate(atoms):
        np.testing.assert_allclose(
            primary[element][0], points[owners == owner] - center, atol=2e-13
        )
        np.testing.assert_allclose(
            primary[element][1], weights[owners == owner], rtol=3e-15
        )
