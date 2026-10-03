"""Generate independent PySCF CCSD(T) water-cluster force references.

Use PySCF 2.14.0 and the repository's exact STO-3G primitives. Named STO-3G
basis sets have independently rounded coefficients and are not this oracle.
Run with one BLAS/OpenMP thread for reproducible small-cluster qualification.
"""

import argparse
import hashlib
import json
from pathlib import Path

import pyscf
from pyscf import cc, gto, scf
from pyscf.cc import ccsd_t_lambda
from pyscf.grad import ccsd_t as gradients

from benchmarks._retention import raw_output_path
from benchmarks.readme_hf_scaling import scaling_cases


def main() -> None:
    """Keep corrected triples Lambda response separate from production code."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--atoms", nargs="+", type=int, default=[6, 12, 24])
    parser.add_argument("--output", type=raw_output_path, required=True)
    args = parser.parse_args()
    if pyscf.__version__ != "2.14.0":
        raise RuntimeError("reference qualification requires PySCF 2.14.0")
    pack_path = (
        Path(__file__).resolve().parents[1] / "python/generativeqc/data/basis_pack.json"
    )
    pack = json.loads(pack_path.read_text())["bases"]["sto-3g"]["elements"]
    basis = {
        gto.mole._symbol(int(z)): [
            [
                shell["angular_momentum"],
                *[
                    [float(e), float(c)]
                    for e, c in zip(
                        shell["exponents"], shell["coefficients"], strict=True
                    )
                ],
            ]
            for shell in shells
        ]
        for z, shells in pack.items()
        if z in ("1", "8")
    }
    record = {
        "schema": "generativeqc.ccsdt-cluster-oracle.v1",
        "pyscf": pyscf.__version__,
        "basis_pack_sha256": hashlib.sha256(pack_path.read_bytes()).hexdigest(),
        "units": {"coordinates": "bohr", "energy": "hartree", "forces": "hartree/bohr"},
        "rows": [],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for count in args.atoms:
        original = scaling_cases()[f"water-{count}"].atoms
        changed = [
            (
                z,
                [
                    x + (0.01 if i == 1 and axis == 2 else 0)
                    for axis, x in enumerate(xyz)
                ],
            )
            for i, (z, xyz) in enumerate(original)
        ]
        for geometry, atoms in (("original", original), ("changed", changed)):
            mol = gto.M(atom=atoms, basis=basis, unit="Bohr", cart=False, verbose=0)
            hf = scf.RHF(mol)
            hf.conv_tol, hf.conv_tol_grad, hf.max_cycle = 1e-13, 1e-11, 200
            hf.kernel()
            coupled = cc.CCSD(hf)
            coupled.conv_tol, coupled.conv_tol_normt, coupled.max_cycle = (
                1e-13,
                1e-11,
                150,
            )
            coupled.kernel()
            triples = float(coupled.ccsd_t())
            if not hf.converged or not coupled.converged:
                raise RuntimeError("independent reference amplitudes did not converge")
            eris = coupled.ao2mo()
            converged, l1, l2 = ccsd_t_lambda.kernel(
                coupled, eris, coupled.t1, coupled.t2, max_cycle=150, tol=1e-11
            )
            if not converged:
                raise RuntimeError("corrected CCSD(T) Lambda did not converge")
            gradient = gradients.Gradients(coupled)
            gradient.cphf_max_cycle, gradient.cphf_conv_tol = 100, 1e-12
            force = -gradient.kernel(coupled.t1, coupled.t2, l1, l2, eris=eris)
            record["rows"].append(
                {
                    "atoms": count,
                    "geometry": geometry,
                    "inputs": atoms,
                    "energy": float(coupled.e_tot + triples),
                    "triples": triples,
                    "forces": force.tolist(),
                }
            )
            args.output.write_text(json.dumps(record, indent=2) + "\n")


if __name__ == "__main__":
    main()
