"""Independent libcint/SciPy check of the full- and long-range cutoff margins."""

import argparse
import json
from pathlib import Path

import numpy as np
from pyscf import gto, lib
from scipy.linalg import eigvalsh


def basis_tables(path: Path) -> dict:
    """Translate the exact retained spherical auxiliary basis to PySCF shells."""
    payload = json.loads(path.read_text())
    return {
        gto.mole._symbol(element["atomic_number"]): [
            [
                shell["angular_momentum"],
                *[
                    [float(exponent), *map(float, coefficients)]
                    for exponent, coefficients in zip(
                        shell["exponents"],
                        zip(*shell["coefficients"], strict=True),
                        strict=True,
                    )
                ],
            ]
            for shell in element["shells"]
        ]
        for element in payload["elements"]
    }


def main() -> None:
    """Write one JSON row per geometry and metric to stdout."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("fixture", nargs="+", type=Path)
    args = parser.parse_args()
    lib.num_threads(1)
    for root in args.fixture:
        case = json.loads((root / "frames.json").read_text())
        tables = basis_tables(root / "wb97m-v-auxiliary.json")
        for frame in case["steps"]:
            mol = gto.M(
                atom=frame["geometry_bohr"],
                basis=tables,
                unit="Bohr",
                cart=False,
                verbose=0,
            )
            norm = 1 / np.sqrt(np.diag(mol.intor("int1e_ovlp")))
            for omega in (0.0, 0.3):
                with mol.with_range_coulomb(omega):
                    metric = mol.intor("int2c2e") * norm[:, None] * norm[None, :]
                eigenvalues = eigvalsh(metric)
                largest = float(eigenvalues[-1])
                cutoff = largest * 1e-10
                nearest = float(eigenvalues[np.argmin(abs(eigenvalues - cutoff))])
                resolution = 128 * np.finfo(float).eps * largest
                print(
                    json.dumps(
                        {
                            "system": case["system"],
                            "step": frame["step"],
                            "omega": omega,
                            "geometry_sha256": frame["geometry_sha256"],
                            "naux": mol.nao_nr(),
                            "rank": int(sum(eigenvalues > cutoff)),
                            "largest": largest,
                            "nearest": nearest,
                            "cutoff": cutoff,
                            "resolution": resolution,
                            "margin": (nearest - cutoff) / resolution,
                        }
                    )
                )


if __name__ == "__main__":
    main()
