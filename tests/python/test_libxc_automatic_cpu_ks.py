"""CPU ordinary-program final-state and endpoint regression gates.

Independent references run after native execution. Missing CPU operator work is
not interpreted as zero exchange work or as complete precision provenance.
"""

from __future__ import annotations

import os
from pathlib import Path
from time import perf_counter

import numpy as np
import pytest
from generativeqc import Calculator, KsOptions
from generativeqc._ks_snapshot import NativeKsSnapshot
from generativeqc_compiler.common.provenance import atomic_json, file_hash
from generativeqc_compiler.dft.grid import GridSpec
from generativeqc_compiler.method import resolve_bulk_ks
from test_libxc_automatic_point_native import NAMES

GRID = GridSpec(radial_points=12, angular_polar=6, angular_azimuth=12)
ATOMS = [
    ("H", (0.0, 0.0, 0.0)),
    ("H", (0.1, 0.2, 1.4)),
    ("H", (1.6, 0.4, 0.3)),
    ("H", (1.8, 0.3, 1.8)),
]
MOVED = [ATOMS[0], ("H", (0.1, 0.2, 1.47)), *ATOMS[2:]]
BASIS = {
    "H": [
        [
            0,
            [3.425250914, 0.1543289673],
            [0.6239137298, 0.5353281423],
            [0.168855404, 0.4446345422],
        ]
    ]
}


def _state(snapshot: NativeKsSnapshot) -> dict[str, np.ndarray]:
    """Decode the native physical wire, without reevaluating AO or XC data."""
    _, nao, spins, atoms, packed, points = snapshot.metadata[:6]
    offset = 2 + 4 * atoms
    result = {}
    for name in (
        "density",
        "fock",
        "coefficients",
        "orbital_energies",
        "occupations",
        "weighted_density",
        "overlap",
    ):
        shape = (
            (nao, nao)
            if name == "overlap"
            else (spins, nao)
            if name in ("orbital_energies", "occupations")
            else (spins, nao, nao)
        )
        count = int(np.prod(shape))
        result[name] = snapshot.values[offset : offset + count].reshape(shape).copy()
        offset += count
    offset += packed
    result["points"] = (
        snapshot.values[offset : offset + 3 * points].reshape(points, 3).copy()
    )
    offset += 3 * points
    result["weights"] = snapshot.values[offset : offset + points].copy()
    return result


def _reference(
    name: str, spins: int, atoms: list, state: dict[str, np.ndarray]
) -> dict:
    """Reconcile SCF, complete fixed-D Fock and independent legal PSD energy FD."""
    pyscf = pytest.importorskip("pyscf")
    from pyscf import dft, gto

    assert pyscf.__version__ == "2.14.0"
    assert dft.libxc.__version__ == "7.0.0"
    molecule = gto.M(
        atom=atoms,
        basis=BASIS,
        unit="Bohr",
        cart=True,
        charge=spins - 1,
        spin=spins - 1,
        verbose=0,
    )
    reference = dft.RKS(molecule) if spins == 1 else dft.UKS(molecule)
    reference.xc = name
    reference.init_guess = "1e"
    reference.small_rho_cutoff = 0
    reference.grids.coords = state["points"]
    reference.grids.weights = state["weights"]
    reference.conv_tol = 1e-12
    reference.conv_tol_grad = 1e-10
    reference.max_cycle = 1000
    energy = reference.kernel()
    assert reference.converged, (name, spins, atoms)
    np.testing.assert_array_equal(reference.grids.coords, state["points"])
    np.testing.assert_array_equal(reference.grids.weights, state["weights"])
    density = state["density"][0] if spins == 1 else state["density"]
    hcore = reference.get_hcore()
    potential = reference.get_veff(molecule, density)
    fock = hcore + potential
    indices = np.arange(molecule.nao_nr())
    perturbation = 0.02 / (1.0 + np.abs(indices[:, None] - indices))
    perturbation[np.diag_indices(len(indices))] += np.linspace(0.05, 0.11, len(indices))
    direction = perturbation @ density + density @ perturbation
    step = 2e-5
    plus = np.eye(len(indices)) + step * perturbation
    minus = np.eye(len(indices)) - step * perturbation
    derivative = (
        reference.energy_tot(plus @ density @ plus.T)
        - reference.energy_tot(minus @ density @ minus.T)
    ) / (2 * step)
    assert abs(derivative - float(np.sum(fock * direction))) < 5e-8
    return {
        "energy": energy,
        "fixed_energy": reference.energy_tot(density, hcore, potential),
        "density": np.asarray(reference.make_rdm1()),
        "fock": fock.reshape(state["fock"].shape),
        "directional_error": abs(derivative - float(np.sum(state["fock"] * direction))),
    }


@pytest.mark.parametrize("name", NAMES)
@pytest.mark.parametrize("spins", (1, 2))
def test_automatic_cpu_complete_final_state_lifecycle(name: str, spins: int) -> None:
    """Every representative must export the returned, strictly certified state."""
    pytest.importorskip("pyscf")
    resolution = resolve_bulk_ks(
        name, spin="unpolarized" if spins == 1 else "polarized", backend="cpu"
    )
    started = perf_counter()
    calculator = Calculator(
        method=resolution.method,
        basis="sto-3g",
        device="cpu",
        precision="fp64",
        initial_guess=None,
        ks_options=KsOptions(grid=GRID, tile_points=37),
        max_iterations=200,
        energy_tolerance=1e-12,
        density_tolerance=1e-10,
        screening_tolerance=1e-12,
    )
    rows = []
    previous = None
    with calculator.prepare_batch(
        [ATOMS], charges=[spins - 1], multiplicities=[spins], warm_start=True
    ) as batch:
        preparation_seconds = perf_counter() - started
        for phase, atoms in (
            ("cold", ATOMS),
            ("warm-recompute", ATOMS),
            ("changed-geometry", MOVED),
        ):
            started = perf_counter()
            result = batch.execute(
                coordinates=None
                if phase != "changed-geometry"
                else [[atom[1] for atom in atoms]],
                strict=True,
                properties=("energy",),
            ).items[0]
            endpoint_seconds = perf_counter() - started
            assert result.succeeded and result.converged, (name, spins, phase)
            assert result.executed_backend == "cpu_reference"
            assert result.warm_start_used == (phase != "cold")
            assert 0 < result.iterations <= 200
            diagnostic = result.ks_diagnostic
            assert diagnostic is not None
            assert diagnostic.physical_residual_max < 2e-9
            assert diagnostic.fock_builds >= result.iterations
            assert tuple(row.iteration for row in diagnostic.history) == tuple(
                range(1, result.iterations + 1)
            )
            if previous is not None:
                with pytest.raises(ValueError, match="stale"):
                    previous.check_current()
                previous.close()
            snapshot = NativeKsSnapshot(batch, 0)
            assert snapshot.backend == "cpu"
            state = _state(snapshot)
            expected = _reference(name, spins, atoms, state)
            np.testing.assert_allclose(
                result.energy, expected["energy"], rtol=0, atol=3e-9
            )
            np.testing.assert_allclose(
                result.energy, expected["fixed_energy"], rtol=0, atol=3e-10
            )
            np.testing.assert_allclose(
                state["density"][0] if spins == 1 else state["density"],
                expected["density"],
                rtol=0,
                atol=2e-7,
            )
            np.testing.assert_allclose(
                state["fock"], expected["fock"], rtol=0, atol=3e-9
            )
            assert expected["directional_error"] < 5e-8
            work = result.precision
            rows.append(
                {
                    "phase": phase,
                    "endpoint_seconds": endpoint_seconds,
                    "complete_endpoint_seconds": endpoint_seconds
                    + (preparation_seconds if phase == "cold" else 0),
                    "energy_error": abs(result.energy - expected["energy"]),
                    "fixed_energy_error": abs(result.energy - expected["fixed_energy"]),
                    "fock_error": float(
                        np.max(np.abs(state["fock"] - expected["fock"]))
                    ),
                    "directional_error": expected["directional_error"],
                    "iterations": result.iterations,
                    "fock_builds": diagnostic.fock_builds,
                    "operator_inventory_complete": work["operator_inventory_complete"],
                    "precision_complete": work["complete"],
                    "operator_counts": {
                        item["name"]: item["count"] for item in work["operators"]
                    },
                }
            )
            previous = snapshot
        assert previous is not None
        previous.close()
    if directory := os.environ.get("GENERATIVEQC_AUTOMATIC_CPU_KS_EVIDENCE_DIR"):
        atomic_json(
            Path(directory) / f"{name}-spins-{spins}.json",
            {
                "claim": "cpu-energy-and-final-state-no-force-or-work-inventory-promotion",
                "functional": name,
                "spins": spins,
                "native_library_sha256": file_hash(
                    Path(os.environ["GENERATIVEQC_LIBRARY"]).resolve()
                ),
                "rows": rows,
            },
        )


@pytest.mark.parametrize("spins", (1, 2))
@pytest.mark.parametrize("maximum", (1, 4))
def test_automatic_cpu_exhausted_budget_does_not_publish_warm_state(
    spins: int, maximum: int
) -> None:
    """Final closure never obtains a fresh hidden SCF budget or a warm seed."""
    calculator = Calculator(
        method=f"libxc-{'rks' if spins == 1 else 'uks'}:MGGA_X_LTA",
        device="cpu",
        basis="sto-3g",
        initial_guess=None,
        ks_options=KsOptions(grid=GRID, tile_points=37),
        max_iterations=maximum,
        energy_tolerance=1e-12,
        density_tolerance=1e-10,
    )
    with calculator.prepare_batch(
        [ATOMS], charges=[spins - 1], multiplicities=[spins], warm_start=True
    ) as batch:
        for _attempt in range(2):
            result = batch.execute(properties=("energy",), strict=False).items[0]
            assert not result.succeeded and not result.converged
            assert result.iterations <= maximum
            assert not result.warm_start_used
            with pytest.raises((RuntimeError, ValueError)):
                NativeKsSnapshot(batch, 0)
