"""Independent PySCF fixtures for the Direct-J/K policy endpoint qualification."""

import json
from typing import Any

import pyscf
from pyscf import gto, scf


def reference(
    unrestricted: bool, polarized: bool, moved: bool, include_f: bool = False
) -> dict[str, Any]:
    """Use the explicit primitive inputs, spherical AOs and Bohr geometries."""
    basis = {
        "He": [
            [
                0,
                [6.36242139, 0.15432897],
                [1.158923, 0.53532814],
                [0.31364979, 0.44463454],
            ]
        ],
        "H": [
            [
                0,
                [3.42525091, 0.15432897],
                [0.62391373, 0.53532814],
                [0.16885540, 0.44463454],
            ]
        ],
    }
    if polarized:
        basis["He"] += [[1, [0.8, 1.0]], [2, [0.6, 1.0]]]
        basis["H"] += [[1, [0.4, 1.0]], [2, [0.3, 1.0]]]
    if include_f:
        basis["He"] += [[3, [0.5, 1.0]]]
        basis["H"] += [[3, [0.25, 1.0]]]
    molecule = gto.M(
        atom=[["He", (0.0, 0.0, -0.6)], ["H", (0.0, 0.0, 0.85 + 0.03 * moved)]],
        unit="Bohr",
        basis=basis,
        charge=0 if unrestricted else 1,
        spin=1 if unrestricted else 0,
        cart=False,
        verbose=0,
    )
    mean_field = (scf.UHF if unrestricted else scf.RHF)(molecule)
    mean_field.conv_tol = 1e-13
    mean_field.conv_tol_grad = 1e-10
    mean_field.max_cycle = 200
    energy = mean_field.kernel()
    if not mean_field.converged:
        raise RuntimeError("independent oracle did not converge")
    return {
        "unrestricted": unrestricted,
        "moved": moved,
        "nbf": molecule.nao,
        "energy": energy,
        "forces": (-mean_field.nuc_grad_method().kernel()).tolist(),
    }


def main() -> None:
    """Emit independent inputs and results, never a production oracle fallback."""
    fixtures = [
        reference(unrestricted, polarized, moved)
        for polarized in (False, True)
        for unrestricted in (False, True)
        for moved in (False, True)
    ]
    print(
        json.dumps(
            {
                "pyscf_version": pyscf.__version__,
                "energy_tolerance": 1e-13,
                "gradient_tolerance": 1e-10,
                "units": "Bohr",
                "fixtures": fixtures,
                "capacity_fixture": reference(False, True, False, True),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
