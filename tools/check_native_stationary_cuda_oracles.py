"""Independently qualify the native CUDA stationary test receipts with PySCF.

The input is the verbose log of an already completed Slurm GPU test, never a
runtime science dependency. The narrow fixture inventory intentionally does not
represent universal CUDA force admission or complete endpoint performance gates.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))

from generativeqc_compiler.dft.grid import GridSpec, MolecularGrid

EXPECTED_CASES = {
    (f"{method}_{spin}", "hydrogen", atoms)
    for method in ("pbe", "pbe0")
    for spin, atoms in (("rks", 2), ("uks", 3), ("uks", 1), ("rks", 18))
} | {
    (f"{method}_{spin}", "lih_sp", 2)
    for method in ("pbe", "pbe0")
    for spin in ("rks", "uks")
}


def read_receipts(text: str) -> list[dict]:
    """Refuse partial/duplicate/nonfinite receipts instead of silently skipping."""
    records = []
    seen = set()
    for line in text.splitlines():
        if "native_stationary_oracle " not in line:
            continue
        record = json.loads(line.split("native_stationary_oracle ", 1)[1])
        if (
            not isinstance(record, dict)
            or not {
                "profile",
                "fixture",
                "atoms",
                "aos",
                "energy",
                "gradient",
                "density",
                "weighted_density",
            }
            <= record.keys()
        ):
            raise ValueError("incomplete native stationary oracle receipt fields")
        if type(record["atoms"]) is not int:
            raise ValueError("native stationary atom count must be an integer")
        key = record["profile"], record["fixture"], record["atoms"]
        if key not in EXPECTED_CASES or key in seen:
            raise ValueError("unexpected or duplicate native stationary oracle receipt")
        aos = 6 if record["fixture"] == "lih_sp" else record["atoms"]
        spins = 1 if record["profile"].endswith("rks") else 2
        if type(record["aos"]) is not int or record["aos"] != aos:
            raise ValueError("invalid native stationary AO extent")
        for name in ("density", "weighted_density"):
            matrix = np.asarray(record[name], dtype=float)
            if matrix.shape != (spins, aos * aos) or not np.all(np.isfinite(matrix)):
                raise ValueError("invalid native stationary final D/W receipt")
        gradient = np.asarray(record["gradient"], dtype=float)
        if (
            gradient.shape != (3 * record["atoms"],)
            or not np.all(np.isfinite(gradient))
            or not np.isfinite(record["energy"])
        ):
            raise ValueError("invalid native stationary energy/gradient receipt")
        seen.add(key)
        records.append(record)
    if seen != EXPECTED_CASES:
        raise ValueError("native stationary oracle fixture inventory is incomplete")
    return records


def independent_reference(
    profile: str, fixture: str, count: int
) -> tuple[float, np.ndarray, np.ndarray, np.ndarray]:
    """libcint/libxc SCF and full-response gradients on the identical finite grid."""
    from pyscf import dft, gto, lib

    lib.num_threads(1)
    coordinates = [
        (0.15 * atom * atom, 0.13 * atom, 1.5 * atom) for atom in range(count)
    ]
    if count == 18:
        coordinates = [
            (5.0 * (atom // 2), 0.0, 0.71 if atom % 2 else -0.71)
            for atom in range(count)
        ]
    labels = [f"H{atom}" for atom in range(count)]
    restricted = profile.endswith("rks")
    shell = [
        0,
        [3.425250914, 0.1543289673],
        [0.6239137298, 0.5353281423],
        [0.168855404, 0.4446345422],
    ]
    if count == 18:
        shell = [0, [0.6, 1.0]]
    elements = ["H"] * count
    basis = {label: [shell] for label in labels}
    charge = 0
    if fixture == "lih_sp":
        coordinates = [(0.0, 0.0, 0.0), (0.21, -0.12, 3.0)]
        labels, elements = ["Li0", "H1"], ["Li", "H"]
        basis = {
            "Li0": [[0, [4.0, 1.0]], [0, [0.5, 1.0]], [1, [0.4, 1.0]]],
            "H1": [[0, [1.0, 1.0]]],
        }
        charge = 0 if restricted else 1
    molecule = gto.M(
        atom=list(zip(labels, coordinates, strict=True)),
        basis=basis,
        unit="Bohr",
        charge=charge,
        spin=0 if restricted else 1,
        cart=True,
        verbose=0,
    )
    mean_field = dft.RKS(molecule) if restricted else dft.UKS(molecule)
    mean_field.xc = "PBE0" if profile.startswith("pbe0") else "PBE"
    grid = MolecularGrid(
        tuple(zip(elements, coordinates, strict=True)),
        GridSpec(radial_points=24, angular_polar=12, angular_azimuth=24),
    )
    tiles = tuple(grid.tiles(257))
    raw_tiles = tuple(grid._raw_tiles(257))
    points = np.concatenate([tile.points for tile in tiles])
    weights = np.concatenate([tile.weights for tile in tiles])
    owners = np.concatenate([tile.owners for tile in tiles])
    raw = np.concatenate([tile.weights for tile in raw_tiles])
    atomic = {
        label: (points[owners == atom] - molecule.atom_coord(atom), raw[owners == atom])
        for atom, label in enumerate(labels)
    }
    mean_field.grids.coords, mean_field.grids.weights = points, weights
    mean_field.grids.radii_adjust = None
    mean_field.grids.gen_atomic_grids = lambda *args, **kwargs: atomic
    mean_field.small_rho_cutoff = 0
    mean_field.conv_tol, mean_field.conv_tol_grad, mean_field.max_cycle = (
        1e-13,
        1e-10,
        200,
    )
    energy = mean_field.kernel()
    if not mean_field.converged:
        raise RuntimeError("independent PySCF stationary oracle did not converge")
    gradient = mean_field.nuc_grad_method()
    gradient.grid_response = True
    density = np.asarray(mean_field.make_rdm1())
    orbitals = np.asarray(mean_field.mo_coeff)
    occupations = np.asarray(mean_field.mo_occ)
    eigenvalues = np.asarray(mean_field.mo_energy)
    if restricted:
        density, orbitals = density[None, ...], orbitals[None, ...]
        occupations, eigenvalues = occupations[None, ...], eigenvalues[None, ...]
    weighted = np.asarray(
        [
            (coefficients * (occupied * energies)[None, :]) @ coefficients.T
            for coefficients, occupied, energies in zip(
                orbitals, occupations, eigenvalues, strict=True
            )
        ]
    )
    return energy, gradient.kernel(), density, weighted


def main() -> None:
    """Every fixture and both independent numerical gates are mandatory."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    args = parser.parse_args()
    records = read_receipts(args.input.read_text())
    from pyscf import __version__

    report = {"pyscf_version": __version__, "cases": []}
    for record in records:
        energy, gradient, density, weighted = independent_reference(
            record["profile"], record["fixture"], record["atoms"]
        )
        energy_error = abs(record["energy"] - energy)
        gradient_error = float(
            np.max(np.abs(np.asarray(record["gradient"]).reshape(-1, 3) - gradient))
        )
        density_error = float(
            np.max(
                np.abs(np.asarray(record["density"]).reshape(density.shape) - density)
            )
        )
        weighted_error = float(
            np.max(
                np.abs(
                    np.asarray(record["weighted_density"]).reshape(weighted.shape)
                    - weighted
                )
            )
        )
        if (
            energy_error > 1e-8
            or gradient_error > 3e-7
            or density_error > 2e-7
            or weighted_error > 2e-7
        ):
            raise AssertionError(
                f"native {record['profile']}/{record['fixture']}/{record['atoms']} oracle failed: "
                f"energy={energy_error} gradient={gradient_error} density={density_error} weighted_density={weighted_error}"
            )
        report["cases"].append(
            {
                "profile": record["profile"],
                "fixture": record["fixture"],
                "atoms": record["atoms"],
                "energy_max_abs": energy_error,
                "gradient_max_abs": gradient_error,
                "density_max_abs": density_error,
                "weighted_density_max_abs": weighted_error,
            }
        )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
