"""Reproduce the CPU mixed-DF RSH oracle in test_dft_api.cpp.

Requires optional PySCF 2.14.0 / Libxc 7.0.0. Only full-range J/K are fitted;
get_k(omega=0.3) deliberately uses exact integrals rather than range-separated DF.
The tiny orbital auxiliary basis qualifies composition, not basis-set accuracy.
Run from the checkout with PYTHONPATH=python python tools/verify_mixed_df_rsh_cpu.py.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np


def main() -> None:
    import pyscf
    from generativeqc_compiler.dft.grid import GridSpec, MolecularGrid
    from pyscf import df, dft, gto, lib, scf
    from pyscf.dft import libxc

    assert pyscf.__version__ == "2.14.0"
    assert libxc.libxc_version() == "7.0.0"
    lib.num_threads(1)
    atoms = [("H", (0.0, 0.0, -0.7)), ("H", (0.0, 0.0, 0.7))]
    basis = {
        "H": [
            [
                0,
                [3.425250914, 0.1543289673],
                [0.6239137298, 0.5353281423],
                [0.168855404, 0.4446345422],
            ]
        ]
    }
    grid = MolecularGrid(
        atoms, GridSpec(radial_points=12, angular_polar=4, angular_azimuth=8)
    ).explicit()
    for wb97mv in (False, True):
        for spin_case in range(3):
            mol = gto.M(
                atom=atoms,
                basis=basis,
                unit="Bohr",
                cart=True,
                charge=-int(spin_case == 2),
                spin=int(spin_case == 2),
                verbose=0,
            )
            fitted = df.DF(mol, auxbasis=basis).build()
            mf = dft.UKS(mol) if spin_case else dft.RKS(mol)
            mf.xc = "WB97M_V" if wb97mv else "PBE + 0.2*HF + 0.6*LR_HF(0.3)"

            def get_jk(
                mol_arg: gto.Mole,
                dm: np.ndarray,
                hermi: int = 1,
                with_j: bool = True,
                with_k: bool = True,
                omega: float | None = None,
                *,
                _fitted: df.DF = fitted,
            ) -> tuple[np.ndarray | None, np.ndarray | None]:
                if omega is not None and omega != 0:
                    return scf.hf.get_jk(
                        mol_arg, dm, hermi, with_j=with_j, with_k=with_k, omega=omega
                    )
                return _fitted.get_jk(dm, hermi, with_j, with_k)

            mf.get_jk = get_jk
            for integration_grid in (mf.grids, mf.nlcgrids):
                integration_grid.coords = grid.points.copy()
                integration_grid.weights = grid.weights.copy()
                integration_grid.non0tab = None
            mf.conv_tol = 1e-12
            mf.conv_tol_grad = 1e-10
            mf.max_cycle = 200
            mf.small_rho_cutoff = 0.0
            energy = mf.kernel(dm0=mf.get_init_guess(key="1e"))
            assert mf.converged
            print(
                json.dumps({"wb97mv": wb97mv, "spin_case": spin_case, "energy": energy})
            )


if __name__ == "__main__":
    main()
