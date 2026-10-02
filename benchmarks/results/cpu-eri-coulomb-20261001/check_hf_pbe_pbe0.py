#!/usr/bin/env python3
"""Untimed numerical checks: HF/PBE/PBE0, frozen water in two spherical bases.

Independent PySCF/libcint/libxc SCF uses identical frozen shell decimals and,
for DFT, the same explicitly specified quadrature. This is not a quadrature
accuracy study and makes no performance claim. No production files are changed.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys

import matched_cpu_evidence as evidence


def worker(args):
    identity = evidence.identity_worker(args.source, args.library)
    import numpy as np
    import generativeqc_compiler
    from generativeqc import Atom, Calculator, GridSpec, KsOptions
    from generativeqc_compiler.dft.grid import MolecularGrid
    from pyscf import dft, gto, lib, scf
    import pyscf
    from threadpoolctl import threadpool_info

    lib.num_threads(1)
    evidence.frozen_check(evidence.FROZEN)
    cases = json.loads((evidence.FROZEN / "cases.json").read_text())
    grid_spec = GridSpec(radial_points=24, angular_polar=8, angular_azimuth=16)
    rows = []
    for case_name in ["water-sto3g", "water-svp"]:
        case = cases[case_name]
        atoms = tuple(Atom.from_value(a) for a in case["atoms"])
        grid = MolecularGrid(atoms, spec=grid_spec, charge=0, multiplicity=1).explicit()
        for method in ["rhf", "pbe-rks", "pbe0-rks"]:
            options = (
                {} if method == "rhf" else {"ks_options": KsOptions(grid=grid_spec)}
            )
            calculator = Calculator(
                method,
                basis=case["basis"],
                device="cpu",
                basis_representation="spherical",
                density_fitting="none",
                precision="fp64",
                energy_tolerance=1e-10,
                density_tolerance=1e-8,
                diis_history=8,
                max_iterations=200,
                screening_tolerance=1e-13,
                **options,
            )
            native = calculator.singlepoint(
                atoms, charge=0, multiplicity=1, properties=("energy",)
            )
            mol = gto.M(
                atom=case["atoms"],
                basis=case["pyscf_basis"],
                cart=False,
                unit="Bohr",
                charge=0,
                spin=0,
                verbose=0,
                max_memory=4000,
            )
            mf = scf.RHF(mol) if method == "rhf" else dft.RKS(mol)
            if method != "rhf":
                mf.xc = "PBE" if method == "pbe-rks" else "PBE0"
                mf.grids.coords = np.asarray(grid.points)
                mf.grids.weights = np.asarray(grid.weights)
                mf.small_rho_cutoff = 0
            mf.chkfile = None
            mf.init_guess = "1e"
            mf.conv_tol = 1e-10
            mf.conv_tol_grad = 1e-8
            mf.diis_space = 8
            mf.max_cycle = 200
            mf.direct_scf = True
            mf.direct_scf_tol = 1e-13
            checks = []

            def check(state):
                de = abs(state["e_tot"] - state["last_hf_e"])
                dr = state["norm_ddm"] / mol.nao_nr()
                checks.append(
                    {
                        "energy_change": float(de),
                        "density_rms": float(dr),
                        "orbital_gradient_norm": float(state["norm_gorb"]),
                    }
                )
                return de < 1e-10 and dr < 1e-8

            mf.check_convergence = check
            mf.kernel()
            oracle_attempts = [
                {
                    "initial_guess": "core",
                    "converged": bool(mf.converged),
                    "energy": float(mf.e_tot),
                    "iterations": mf.cycles,
                    "checks": list(checks),
                }
            ]
            # This is untimed oracle qualification only. Preserve a failed
            # post-SCF density check and self-refine with unchanged tolerances;
            # no oracle density is ever passed into either native calculation.
            for _ in range(2):
                if mf.converged:
                    break
                density = mf.make_rdm1()
                checks = []
                mf.kernel(dm0=density)
                oracle_attempts.append(
                    {
                        "initial_guess": "previous_oracle_density",
                        "converged": bool(mf.converged),
                        "energy": float(mf.e_tot),
                        "iterations": mf.cycles,
                        "checks": list(checks),
                    }
                )
            row = {
                "case": case_name,
                "method": method,
                "energy": native.energy,
                "converged": native.converged,
                "iterations": native.iterations,
                "energy_change": native.energy_change,
                "density_rms": native.density_rms,
                "backend": native.executed_backend,
                "precision": native.precision,
                "pyscf_energy": float(mf.e_tot),
                "pyscf_converged": bool(mf.converged),
                "pyscf_iterations": mf.cycles,
                "pyscf_eri_incore": mf._eri is not None,
                "pyscf_convergence_checks": checks,
                "pyscf_attempts": oracle_attempts,
                "absolute_oracle_error_hartree": abs(native.energy - mf.e_tot),
                "oracle_gate_hartree": 1e-8,
                "force_error": None,
            }
            rows.append(row)
            print(json.dumps(row), flush=True)
    pools = threadpool_info()
    payload = {
        "identity": identity,
        "rows": rows,
        "threadpool_info": pools,
        "pyscf_version": pyscf.__version__,
        "pyscf_package": pyscf.__file__,
        "compiler_package": generativeqc_compiler.__file__,
        "grid": {
            "radial_points": 24,
            "angular_polar": 8,
            "angular_azimuth": 16,
            "version": 1,
            "pruning": "none",
            "partition": "becke-equal-radius",
        },
        "convergence_contract": {
            "energy_tolerance_hartree": 1e-10,
            "density_rms_tolerance": 1e-8,
            "reason": "Same frozen user-requested SCF gates; extra-tight pilot preserved separately",
        },
        "limits": "Untimed same-quadrature numerical oracle with explicitly retained self-refinement attempts after post-SCF convergence failures; native always uses core guess. Not DFT timing or grid convergence",
    }
    payload["passed"] = all(
        r["converged"]
        and r["pyscf_converged"]
        and r["pyscf_eri_incore"]
        and abs(r["energy_change"]) < 1e-10
        and r["density_rms"] < 1e-8
        and r["absolute_oracle_error_hartree"] < r["oracle_gate_hartree"]
        and r["backend"] == "cpu_reference"
        and r["precision"]["effective_bits"] == 64
        for r in rows
    )
    payload["passed"] &= bool(pools) and all(pool["num_threads"] == 1 for pool in pools)
    evidence.write_json(args.output, payload)
    if not payload["passed"]:
        raise RuntimeError(
            "HF/PBE/PBE0 numerical gate failed; every observation retained"
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="mode", required=True)
    p = sub.add_parser("run")
    for engine in ["baseline", "candidate"]:
        p.add_argument(f"--{engine}-source", type=Path, required=True)
        p.add_argument(f"--{engine}-library", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p = sub.add_parser("_worker")
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--library", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.mode == "_worker":
        worker(args)
        return
    evidence.ensure_quiet()
    out = args.output.resolve()
    if out.exists() and any(out.iterdir()):
        raise RuntimeError("Use a new empty output directory")
    out.mkdir(parents=True, exist_ok=True)
    provenance = {}
    for engine in ["baseline", "candidate"]:
        source, library = (
            getattr(args, f"{engine}_source"),
            getattr(args, f"{engine}_library"),
        )
        provenance[engine] = evidence.inspect(source, library)
        evidence.write_json(out / "provenance.json", provenance)
        with (out / f"{engine}.log").open("w") as log:
            result = subprocess.run(
                [
                    sys.executable,
                    str(Path(__file__).resolve()),
                    "_worker",
                    "--source",
                    str(source.resolve()),
                    "--library",
                    str(library.resolve()),
                    "--output",
                    str(out / f"{engine}.json"),
                ],
                env=evidence.environment(source, library),
                stdout=log,
                stderr=subprocess.STDOUT,
                timeout=1800,
                check=False,
            )
        if result.returncode:
            raise RuntimeError(
                f"{engine} numerical check failed; see {out / engine}.log"
            )
    b, c = [
        json.loads((out / f"{engine}.json").read_text())
        for engine in ["baseline", "candidate"]
    ]
    rows = []
    for baseline, candidate in zip(b["rows"], c["rows"], strict=True):
        assert (baseline["case"], baseline["method"]) == (
            candidate["case"],
            candidate["method"],
        )
        rows.append(
            {
                "case": baseline["case"],
                "method": baseline["method"],
                "baseline_energy": baseline["energy"],
                "candidate_energy": candidate["energy"],
                "absolute_pair_error_hartree": abs(
                    baseline["energy"] - candidate["energy"]
                ),
                "baseline_iterations": baseline["iterations"],
                "candidate_iterations": candidate["iterations"],
                "same_iterations": baseline["iterations"] == candidate["iterations"],
                "baseline_oracle_error": baseline["absolute_oracle_error_hartree"],
                "candidate_oracle_error": candidate["absolute_oracle_error_hartree"],
            }
        )
    passed = (
        b["passed"]
        and c["passed"]
        and all(
            r["absolute_pair_error_hartree"] < 1e-9 and r["same_iterations"]
            for r in rows
        )
    )
    summary = {
        "passed": passed,
        "rows": rows,
        "pair_gate_hartree": 1e-9,
        "oracle_gate_hartree": 1e-8,
        "performance_claim": False,
    }
    evidence.write_json(out / "summary.json", summary)
    print(json.dumps(summary, indent=2))
    if not passed:
        raise RuntimeError("Paired numerical gate failed; see summary")


if __name__ == "__main__":
    main()
