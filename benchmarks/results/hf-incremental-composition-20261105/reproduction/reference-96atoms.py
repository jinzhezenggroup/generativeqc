"""Independent full-density GPU4PySCF E/F for the 96-atom promotion gate."""

import argparse
import hashlib
import json
import lzma
import os
from pathlib import Path
import time

import numpy as np
import cupy as cp
from gpu4pyscf.scf import hf
from pyscf import gto, lib


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    assert os.environ.get("SLURM_JOB_ID")
    inputs = json.loads(args.inputs.read_text())
    lib.num_threads(8)
    references = []
    observations = []
    for phase in ("atoms", "moved"):
        molecule = gto.M(atom=inputs[phase], unit="Bohr", basis="def2-svp", cart=False,
                         verbose=4, max_memory=12000)
        assert molecule.nao_nr() == 768 and molecule.natm == 96
        solver = hf.RHF(molecule)
        solver.direct_scf = False
        solver.direct_scf_tol = 1e-14
        solver.conv_tol = 1e-12
        solver.conv_tol_grad = 1e-11
        solver.max_cycle = 150
        started = time.perf_counter()
        solver.kernel()
        density = solver.make_rdm1()
        fock = solver.get_hcore() + solver.get_veff(molecule, density)
        gradient_norm = float(cp.linalg.norm(solver.get_grad(solver.mo_coeff, solver.mo_occ, fock)))
        assert solver.converged and np.isfinite(gradient_norm) and gradient_norm <= 1e-10
        forces = -cp.asnumpy(solver.nuc_grad_method().kernel())
        assert np.isfinite(forces).all()
        reference = {"energy": float(solver.e_tot), "forces": forces.tolist()}
        references.append(reference)
        observations.append({"phase": phase, "converged": bool(solver.converged),
                             "orbital_gradient_norm": gradient_norm,
                             "seconds": time.perf_counter() - started})
        args.output.with_suffix(".progress.json").write_text(json.dumps(observations, indent=2))
    data = {**inputs, "reference": references}
    packet = {"references": {"768": {"data": data}}, "oracle": observations,
              "engine": "GPU4PySCF", "job": os.environ["SLURM_JOB_ID"],
              "inputs_sha256": hashlib.sha256(args.inputs.read_bytes()).hexdigest()}
    args.output.write_bytes(lzma.compress(json.dumps(packet, allow_nan=False).encode()))


if __name__ == "__main__":
    main()
