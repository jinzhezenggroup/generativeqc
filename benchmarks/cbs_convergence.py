"""Qualify routine CBS convergence against independent full-force references.

This explicit benchmark uses physical DZ/TZ pairs and intentionally includes
nonstationary, ionic and stretched geometries. It is not a production fallback
or a claim that a single-reference model describes bond dissociation exactly.
All device execution requires Slurm. Reference points never supply native seeds.
"""

from __future__ import annotations

import argparse
import os
import platform
import time
from dataclasses import asdict
from pathlib import Path

from benchmarks._retention import raw_output_path
from benchmarks.readme_cbs import HF_WEIGHTS, MP2_WEIGHTS, checkpoint, digest

WATER = [(8, (0.0, 0.0, 0.0)), (1, (1.4, 0.0, 1.1)), (1, (-1.4, 0.0, 1.1))]
CASES = {
    "water": {"atoms": WATER},
    "water-stretched": {
        "atoms": [WATER[0], (1, (2.1, 0.0, 1.65)), WATER[2]],
    },
    "ammonia": {
        "atoms": [
            (7, (0.0, 0.0, 0.0)),
            (1, (1.77, 0.0, -0.65)),
            (1, (-0.885, 1.533, -0.65)),
            (1, (-0.885, -1.533, -0.65)),
        ],
    },
    "ammonium": {
        "charge": 1,
        "atoms": [
            (7, (0.0, 0.0, 0.0)),
            (1, (1.1, 1.1, 1.1)),
            (1, (1.1, -1.1, -1.1)),
            (1, (-1.1, 1.1, -1.1)),
            (1, (-1.1, -1.1, 1.1)),
        ],
    },
    "hydroxide": {
        "charge": -1,
        "atoms": [(8, (0.0, 0.0, 0.0)), (1, (0.0, 0.0, 1.8))],
    },
    "hydroxide-diffuse": {
        "charge": -1,
        "bases": ("aug-cc-pvdz", "aug-cc-pvtz"),
        "atoms": [(8, (0.0, 0.0, 0.0)), (1, (0.0, 0.0, 1.8))],
    },
    "hydrogen-stretched": {
        "atoms": [(1, (0.0, 0.0, -4.0)), (1, (0.0, 0.0, 4.0))],
    },
    "formaldehyde": {
        "atoms": [
            (6, (0.0, 0.0, 0.0)),
            (8, (0.0, 0.0, 2.27)),
            (1, (1.78, 0.0, -1.1)),
            (1, (-1.78, 0.0, -1.1)),
        ],
    },
    "nitrogen": {"atoms": [(7, (0.0, 0.0, -1.04)), (7, (0.0, 0.0, 1.04))]},
    "nitrogen-stretched": {
        "atoms": [(7, (0.0, 0.0, -1.4)), (7, (0.0, 0.0, 1.4))],
    },
    "water-dimer": {
        "atoms": WATER + [(z, (x, y, a + 5.5)) for z, (x, y, a) in WATER],
    },
}
BASES = ("cc-pvdz", "cc-pvtz")
FITS = tuple(name + "-jkfit" for name in BASES)
STEPS = (1e-4, 3e-5)
# Declare observable gates before collecting candidates. These are deliberately
# stronger than the routine profile's allowance on this qualification cohort.
GATES = {
    "energy": 1e-7,
    "force_max": 3e-6,
    "force_rms": 1e-6,
    "fd_stability": 1e-6,
    # A symmetric root switch can pass two-step gradient stability. Require
    # every displaced component to stay near its central stationary branch.
    "oracle_locality_energy": 1e-2,
}


def inputs(case: dict, model: str, budget: int) -> tuple[dict, tuple, tuple]:
    """Snapshot identical basis decimals, without computing native SCF guesses."""
    from pyscf import df, gto

    from tools.generativeqc_validation.correlation_df_oracle import basis_snapshot

    atoms, charge = case["atoms"], case.get("charge", 0)
    elements = [z for z, _ in atoms]
    names = case.get("bases", BASES)
    fits = tuple(name + "-jkfit" for name in names)
    orbital = tuple(basis_snapshot(name, elements) for name in names)
    auxiliary = tuple(basis_snapshot(name, elements) for name in fits)
    molecules = [
        gto.M(atom=atoms, charge=charge, basis=name, unit="Bohr", verbose=0)
        for name in names
    ]
    return (
        {
            "atoms_bohr": atoms,
            "charge": charge,
            "multiplicity": 1,
            "model": model,
            "orbital_bases": names,
            "auxiliary_bases": fits,
            "orbital_identities": [b.identity for b in orbital],
            "auxiliary_identities": [b.identity for b in auxiliary],
            "nao": [m.nao_nr() for m in molecules],
            "naux": [
                df.make_auxmol(m, b).nao_nr()
                for m, b in zip(molecules, fits, strict=True)
            ],
            "electrons": molecules[0].nelectron,
            "hf_weights": HF_WEIGHTS,
            "mp2_weights": MP2_WEIGHTS,
            "frozen_core": False,
            "budget_bytes": budget,
        },
        orbital,
        auxiliary,
    )


def native(
    case: dict,
    orbital: tuple,
    auxiliary: tuple,
    args: argparse.Namespace,
    policy: str,
    *,
    energy_only: bool = False,
) -> dict:
    """Time fresh owner construction through returned host observables."""
    from generativeqc import CBSBasis, CCSDTCBSCalculator

    started = time.perf_counter()
    owner = CCSDTCBSCalculator(
        tuple(CBSBasis(b, n) for b, n in zip(orbital, (2, 3), strict=True)),
        auxiliary_bases=auxiliary,
        hf_reference="df" if args.model == "full-df" else "exact",
        device="cuda",
        policy=policy,
        preconvergence=args.preconvergence,
        calculator_options={"correlation_memory_budget_bytes": args.budget_mib << 20},
    )
    result = owner.singlepoint(
        case["atoms"],
        charge=case.get("charge", 0),
        properties=("energy",) if energy_only else ("energy", "forces"),
    )
    return {**asdict(result), "complete_seconds": time.perf_counter() - started}


def reference_point(
    atoms: list,
    charge: int,
    model: str,
    bases: tuple = BASES,
    guess: str = "minao",
    cc_solver: str = "diis",
) -> dict:
    """Reconverge independent PySCF RHF/MP2/CCSD(T) without native data.

    Retain gaps and iteration counts to identify small-gap and difficult cases.
    Every displaced geometry recomputes the metric, integrals and orbitals.
    """
    from pyscf import cc, gto, lib, scf
    from pyscf.mp import dfmp2

    lib.num_threads(1)
    values, diagnostics = {}, []
    fits = tuple(name + "-jkfit" for name in bases)
    for index, (basis, auxiliary) in enumerate(zip(bases, fits, strict=True)):
        mol = gto.M(atom=atoms, charge=charge, basis=basis, unit="Bohr", verbose=0)
        mf = scf.RHF(mol)
        if model == "full-df":
            mf = mf.density_fit(auxbasis=auxiliary)
        mf.init_guess = guess
        mf.conv_tol, mf.conv_tol_grad, mf.max_cycle = 1e-13, 1e-10, 200
        mf.kernel()
        if not mf.converged:
            raise RuntimeError("independent RHF did not converge")
        mp = dfmp2.DFMP2(mf).density_fit(auxbasis=auxiliary)
        if model == "full-df":
            mp.with_df = mf.with_df
        mp.kernel()
        label = "low" if index == 0 else "high"
        values[label + "/rhf"] = float(mf.e_tot)
        values[label + "/mp2"] = float(mp.e_tot)
        nocc = mol.nelectron // 2
        row = {
            "basis": basis,
            "scf_cycles": mf.cycles,
            "gap": float(mf.mo_energy[nocc] - mf.mo_energy[nocc - 1]),
        }
        if index == 0:
            coupled = cc.CCSD(mf).density_fit(auxbasis=auxiliary)
            if model == "full-df":
                coupled.with_df = mf.with_df
            coupled.conv_tol, coupled.conv_tol_normt = 1e-12, 1e-10
            coupled.max_cycle = 150
            coupled.kernel()
            row["diis_converged"] = bool(coupled.converged)
            if not coupled.converged and cc_solver == "krylov":
                # Independently solve PySCF's fixed-point equations when DIIS
                # stalls on an unstable RHF branch. This reference-only polish
                # never reads native amplitudes or supplies production seeds.
                import numpy as np
                from scipy.optimize import root

                eris = coupled.ao2mo()
                evaluations = 0

                def residual(vector: np.ndarray) -> np.ndarray:
                    nonlocal evaluations
                    evaluations += 1
                    # root consumes this closure before the basis loop advances.
                    one, two = coupled.vector_to_amplitudes(vector)  # noqa: B023
                    new_one, new_two = coupled.update_amps(one, two, eris)  # noqa: B023
                    return coupled.amplitudes_to_vector(new_one, new_two) - vector  # noqa: B023

                solved = root(
                    residual,
                    coupled.amplitudes_to_vector(coupled.t1, coupled.t2),
                    method="krylov",
                    options={"fatol": 1e-12, "maxiter": 30},
                )
                norm = float(np.linalg.norm(residual(solved.x)))
                coupled.converged = bool(solved.success and norm <= 1e-10)
                coupled.t1, coupled.t2 = coupled.vector_to_amplitudes(solved.x)
                coupled.e_corr = coupled.energy(coupled.t1, coupled.t2, eris)
                row["krylov"] = {
                    "converged": coupled.converged,
                    "update_norm": norm,
                    "evaluations": evaluations,
                    "message": str(solved.message),
                }
            if not coupled.converged:
                raise RuntimeError("independent CCSD did not converge")
            triples = float(coupled.ccsd_t())
            values["low/df-ccsd(t)"] = float(coupled.e_tot + triples)
            row.update(
                ccsd_cycles=coupled.cycles,
                triples=triples,
                t1_diagnostic=float(coupled.get_t1_diagnostic()),
            )
        diagnostics.append(row)
    values["composite"] = float(
        sum(
            w * values[label + "/rhf"]
            for w, label in zip(HF_WEIGHTS, ("low", "high"), strict=True)
        )
        + sum(
            w * (values[label + "/mp2"] - values[label + "/rhf"])
            for w, label in zip(MP2_WEIGHTS, ("low", "high"), strict=True)
        )
        + values["low/df-ccsd(t)"]
        - values["low/mp2"]
    )
    return {"energies": values, "diagnostics": diagnostics}


def run_reference(
    case: dict, record: dict, payload: dict, args: argparse.Namespace
) -> None:
    """Retain every +/- displaced energy for two complete Cartesian gradients."""
    atoms, charge = case["atoms"], case.get("charge", 0)
    record["central"] = reference_point(
        atoms,
        charge,
        args.model,
        case.get("bases", BASES),
        args.reference_guess,
        args.reference_cc_solver,
    )
    record["derivatives"] = []
    checkpoint(args.output, payload)
    for step in STEPS:
        for atom in range(len(atoms)):
            for axis in range(3):
                row = {"step": step, "atom": atom, "axis": axis, "points": []}
                for sign in (-1, 1):
                    moved = [(z, list(xyz)) for z, xyz in atoms]
                    moved[atom][1][axis] += sign * step
                    row["points"].append(
                        reference_point(
                            moved,
                            charge,
                            args.model,
                            case.get("bases", BASES),
                            args.reference_guess,
                            args.reference_cc_solver,
                        )
                    )
                record["derivatives"].append(row)
                checkpoint(args.output, payload)
        print("REFERENCE", payload["stage"], "step", step, flush=True)


def main() -> int:
    """Collect independent references or native populations, preserving failures."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("engine", choices=("native", "reference"))
    parser.add_argument("--cases", nargs="+", choices=tuple(CASES), default=list(CASES))
    parser.add_argument(
        "--model", choices=("full-df", "correlation-df"), default="full-df"
    )
    parser.add_argument("--policies", nargs="+", default=["tight", "standard"])
    parser.add_argument(
        "--preconvergence", choices=("auto", "fp64", "mixed"), default="auto"
    )
    parser.add_argument("--samples", type=int, default=2)
    parser.add_argument("--reference-guess", choices=("minao", "1e"), default="minao")
    parser.add_argument(
        "--reference-cc-solver", choices=("diis", "krylov"), default="diis"
    )
    parser.add_argument("--budget-mib", type=int, default=8192)
    parser.add_argument("--output", required=True, type=raw_output_path)
    parser.add_argument("--source-patch", required=True, type=Path)
    args = parser.parse_args()
    if args.engine == "native" and not os.environ.get("SLURM_JOB_ID"):
        parser.error("real GPU work requires an explicit finite srun allocation")
    import pyscf

    payload = {
        "schema": "generativeqc.cbs-routine-convergence.v1",
        "engine": args.engine,
        "host": platform.node(),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "pyscf": pyscf.__version__,
        "reference_guess": args.reference_guess if args.engine == "reference" else None,
        "reference_cc_solver": args.reference_cc_solver
        if args.engine == "reference"
        else None,
        "source_patch_sha256": digest(args.source_patch),
        "driver_sha256": digest(Path(__file__)),
        "library_sha256": digest(Path(os.environ["GENERATIVEQC_LIBRARY"]))
        if args.engine == "native"
        else None,
        "policies": args.policies,
        "samples_per_policy": args.samples,
        "preconvergence": args.preconvergence,
        "gates": GATES,
        "cases": {},
        "status": "running",
    }
    if args.engine == "native":
        case = CASES["water"]
        _, orbital, auxiliary = inputs(case, args.model, args.budget_mib << 20)
        payload["startup"] = native(case, orbital, auxiliary, args, "tight")
    for name in args.cases:
        payload["stage"] = name
        protocol, orbital, auxiliary = inputs(
            CASES[name], args.model, args.budget_mib << 20
        )
        record = {"protocol": protocol, "samples": [], "status": "running"}
        payload["cases"][name] = record
        checkpoint(args.output, payload)
        try:
            if args.engine == "reference":
                run_reference(CASES[name], record, payload, args)
            else:
                for repeat in range(args.samples):
                    policies = args.policies if repeat % 2 == 0 else args.policies[::-1]
                    for policy in policies:
                        print("START", name, repeat, policy, flush=True)
                        result = native(CASES[name], orbital, auxiliary, args, policy)
                        record["samples"].append(
                            {"policy": policy, "repeat": repeat, **result}
                        )
                        checkpoint(args.output, payload)
                        print(
                            "DONE",
                            name,
                            repeat,
                            policy,
                            result["complete_seconds"],
                            flush=True,
                        )
                record["energy_only"] = native(
                    CASES[name], orbital, auxiliary, args, "standard", energy_only=True
                )
            record["status"] = "completed"
        except Exception as error:  # noqa: BLE001 -- retain failed cases, exit nonzero
            record.update(status="failed", error=f"{type(error).__name__}: {error}")
            print("FAILED", name, record["error"], flush=True)
        checkpoint(args.output, payload)
    payload["status"] = (
        "completed"
        if all(r["status"] == "completed" for r in payload["cases"].values())
        else "failed"
    )
    checkpoint(args.output, payload)
    return int(payload["status"] != "completed")


if __name__ == "__main__":
    raise SystemExit(main())
