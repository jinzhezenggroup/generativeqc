"""Independent PySCF correlation-only DF oracle, never a production consumer.

The conventional RHF object is deliberately not density-fitted. Its orbitals,
Fock and reference energy feed PySCF's DF-MP2 and optional DF-CCSD(T). Molecular
derivatives are obtained by reconverging displaced energies, including the
auxiliary-center and metric dependence.
"""

from __future__ import annotations

import hashlib
import json
from typing import TYPE_CHECKING

from generativeqc.basis import BasisProvenance, BasisSet, BasisShell, ElementBasis

if TYPE_CHECKING:
    from collections.abc import Sequence


def basis_snapshot(name: str, elements: Sequence[int]) -> BasisSet:
    """Import the exact local PySCF basis decimals for matched oracle inputs."""
    from pyscf import gto

    records = {
        number: gto.basis.load(name, gto.mole._symbol(number))
        for number in sorted({int(element) for element in elements})
    }
    checksum = hashlib.sha256(json.dumps(records, sort_keys=True).encode()).hexdigest()
    import pyscf

    return BasisSet(
        name=name,
        elements=tuple(
            ElementBasis(
                number,
                tuple(
                    BasisShell(
                        angular_momentum=shell[0],
                        exponents=tuple(str(primitive[0]) for primitive in shell[1:]),
                        coefficients=tuple(
                            tuple(str(primitive[column]) for primitive in shell[1:])
                            for column in range(1, len(shell[1]))
                        ),
                    )
                    for shell in shells
                ),
            )
            for number, shells in records.items()
        ),
        provenance=BasisProvenance(
            "local PySCF basis library", pyscf.__version__, "Apache-2.0", checksum
        ),
        representation="spherical",
    )


def energies(
    atoms: Sequence, orbital: str, auxiliary: str, *, triples: bool = False
) -> dict[str, float]:
    """Return exact RHF plus correlation-only DF total energies in Hartree."""
    from pyscf import cc, gto, lib, scf
    from pyscf.mp import dfmp2

    lib.num_threads(1)
    molecule = gto.M(atom=list(atoms), basis=orbital, unit="Bohr", verbose=0)
    reference = scf.RHF(molecule)
    reference.conv_tol = 1e-13
    reference.conv_tol_grad = 1e-10
    reference.max_cycle = 200
    reference.kernel()
    if not reference.converged:
        raise RuntimeError("independent conventional RHF oracle did not converge")
    mp2 = dfmp2.DFMP2(reference).density_fit(auxbasis=auxiliary)
    mp2.kernel()
    result = {"rhf": float(reference.e_tot), "mp2": float(mp2.e_tot)}
    if triples:
        coupled_cluster = cc.CCSD(reference).density_fit(auxbasis=auxiliary)
        coupled_cluster.conv_tol = 1e-12
        coupled_cluster.conv_tol_normt = 1e-10
        coupled_cluster.max_cycle = 150
        coupled_cluster.kernel()
        if not coupled_cluster.converged:
            raise RuntimeError("independent DF-CCSD oracle did not converge")
        result["df-ccsd(t)"] = float(coupled_cluster.e_tot + coupled_cluster.ccsd_t())
    return result
