"""Reproduce the CPU mixed-DF RSH oracles in test_dft_api.cpp.

Requires optional PySCF 2.14.0 / Libxc 7.0.0. Both modes fit full-range J/K.
The qualified WB97M-V composition fits LR exchange only with the explicit
``df`` selector; the default and ``direct`` keep exact LR integrals. Generic PBE-RSH
keeps exact LR integrals in both modes. The tiny orbital auxiliary basis
qualifies composition, not basis-set accuracy.
Run from the checkout with PYTHONPATH=python python tools/verify_mixed_df_rsh_cpu.py.
"""

from __future__ import annotations

import argparse
import json


def main() -> None:
    import numpy as np
    import pyscf
    from generativeqc_compiler.dft.grid import GridSpec, MolecularGrid
    from pyscf import df, dft, gto, lib, scf
    from pyscf.dft import libxc

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--range-exchange", choices=("direct", "df", "both"), default="both"
    )
    args = parser.parse_args()
    modes = (
        ("direct", "df") if args.range_exchange == "both" else (args.range_exchange,)
    )
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
    for range_mode in modes:
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
                fitted_lr = None
                if wb97mv and range_mode == "df":
                    # Match the declared relative-eigenvalue cutoff in normalized
                    # auxiliary coordinates, independently using libcint/NumPy.
                    auxiliary = fitted.auxmol
                    scales = 1 / np.sqrt(np.diag(auxiliary.intor("int1e_ovlp")))
                    with mol.with_range_coulomb(0.3), auxiliary.with_range_coulomb(0.3):
                        metric = (
                            auxiliary.intor("int2c2e")
                            * scales[:, None]
                            * scales[None, :]
                        )
                        three_center = df.incore.aux_e2(mol, auxiliary, aosym="s2ij")
                    eigenvalues, eigenvectors = np.linalg.eigh(metric)
                    retained = eigenvalues > eigenvalues[-1] * 1e-10
                    whitening = (
                        eigenvectors[:, retained] / np.sqrt(eigenvalues[retained])
                    ).T
                    fitted_lr = df.DF(mol)
                    fitted_lr._cderi = np.ascontiguousarray(
                        whitening @ (three_center * scales).T
                    )
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
                    _fitted_lr: df.DF | None = fitted_lr,
                ) -> tuple[np.ndarray | None, np.ndarray | None]:
                    if omega is not None and omega != 0:
                        if _fitted_lr is not None:
                            assert abs(omega - 0.3) < 1e-14
                            return _fitted_lr.get_jk(dm, hermi, with_j, with_k)
                        return scf.hf.get_jk(
                            mol_arg,
                            dm,
                            hermi,
                            with_j=with_j,
                            with_k=with_k,
                            omega=omega,
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
                    json.dumps(
                        {
                            "range_exchange": range_mode,
                            "wb97mv": wb97mv,
                            "spin_case": spin_case,
                            "fitted_lr": fitted_lr is not None,
                            "energy": energy,
                        }
                    )
                )


if __name__ == "__main__":
    main()
