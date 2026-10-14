"""Compare complete PBE0 FP64/AUTO/explicit-audit endpoints, not kernel timings.

Run ``--engine reference`` separately for an independent CPU PySCF oracle.
Native execution requires Slurm, an explicit finite allocation, and that oracle.
Every timed call constructs a fresh calculator and releases its workspaces;
only process/compiler caches are warm. No converged density is replayed.
"""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import os
from contextlib import nullcontext
from dataclasses import asdict
from pathlib import Path
from statistics import median
from time import perf_counter
from typing import Any

import numpy as np
from generativeqc import Calculator, GridSpec, KsOptions, PrecisionCalculator
from generativeqc._model_resolution import snapshot_basis
from generativeqc.elements import SYMBOLS

from benchmarks._cases import benchmark_cases
from benchmarks._support import raw_output_path
from benchmarks.compare_gpu4pyscf_batch import (
    native_build_metadata,
    require_tuned_native_build,
)
from tools.dft_mp_v1.precision import precision_from_result

SCHEMA = "generativeqc.pbe0-application-precision.v2"
MODES = ("fp64", "auto", "audit_auto")
CASES = ("water-def2-svp-spherical", "water-tetramer-def2-svp-spherical")


def protocol(case_name: str, spec: GridSpec) -> dict[str, Any]:
    """Bind the exact scientific target shared by the independent processes."""
    case = benchmark_cases()[case_name]
    basis = snapshot_basis(case.generativeqc_basis, case.basis_representation)
    return json.loads(
        json.dumps(
            {
                "case": case_name,
                "atoms_bohr": case.atoms,
                "charge": case.charge,
                "multiplicity": case.multiplicity,
                "method": "pbe0-rks",
                "basis_identity": basis.identity,
                "basis_representation": case.basis_representation,
                "grid": asdict(spec),
                "density_fitting": False,
                "initial_guess": "hcore",
                "properties": ["energy", "forces"],
                "energy_tolerance": 1e-10,
                "density_tolerance": 1e-8,
                "screening_tolerance": 1e-12,
                "max_iterations": 180,
                "energy_gate_Eh": 1e-7,
                "force_gate_Eh_per_bohr": 1e-6,
            }
        )
    )


def reference(
    case_name: str,
    spec: GridSpec,
    threads: int,
    *,
    functional: str = "PBE0",
    atoms: Any = None,
    basis: Any = None,
    auxiliary_basis: Any = None,
    backend: str = "pyscf",
) -> dict[str, Any]:
    """PySCF owns SCF, integrals, XC, partition and analytic moving-grid response.

    Reconstruct atomic quadrature from the public specification rather than
    supplying native final grid weights; derivatives must move the same grid.
    The basis coefficients are common inputs, not a native numerical oracle.
    Other RKS qualification consumers may select the XC functional, an explicit
    moved geometry and shared basis inputs. Nonlocal functionals use the same
    independent atomic rules for their second grid, including its full moving
    response. PySCF's VV10 energy and gradient use the molecular 1e-8 density
    threshold. The default PBE0 protocol keeps the original geometry and basis.
    An explicit PySCF auxiliary basis selects its independent density-fitted
    Hamiltonian, including auxiliary-center and metric response in the forces.
    The optional ``gpu4pyscf`` oracle retains independent grids and complete
    strict SCF/gradient work; it requires a Slurm allocation. Fitted WB97M-V
    uses full-range DF J/K plus exact long-range K and its analytic derivative,
    not the default GPU4PySCF all-fitted range-separated Hamiltonian.
    """
    if backend not in ("pyscf", "gpu4pyscf"):
        raise ValueError("reference backend must be pyscf or gpu4pyscf")
    if backend == "gpu4pyscf" and not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("GPU reference requires a finite Slurm allocation")
    if (
        auxiliary_basis is not None
        and functional.upper().replace("-", "_") == "WB97M_V"
        and backend != "gpu4pyscf"
    ):
        raise NotImplementedError("fitted WB97M-V requires the matched GPU reference")
    import pyscf
    from pyscf import dft, gto, lib

    lib.num_threads(threads)
    case = benchmark_cases()[case_name]
    basis = snapshot_basis(
        case.generativeqc_basis if basis is None else basis, case.basis_representation
    )
    shells = {
        SYMBOLS[element.atomic_number - 1]: [
            [
                shell.angular_momentum,
                *[
                    [
                        float(exponent),
                        *[float(row[index]) for row in shell.coefficients],
                    ]
                    for index, exponent in enumerate(shell.exponents)
                ],
            ]
            for shell in element.shells
        ]
        for element in basis.elements
    }
    molecule = gto.M(
        atom=case.atoms if atoms is None else atoms,
        basis=shells,
        unit="Bohr",
        cart=case.basis_representation == "cartesian",
        charge=case.charge,
        spin=case.multiplicity - 1,
        verbose=0,
    )
    radial_nodes, radial_weights = np.polynomial.legendre.leggauss(spec.radial_points)
    mapped = (radial_nodes + 1) / 2
    radii = mapped / (1 - mapped)
    measures = radial_weights / 2 * mapped**2 / (1 - mapped) ** 4
    polar, polar_weights = np.polynomial.legendre.leggauss(spec.angular_polar)
    azimuth = np.arange(spec.angular_azimuth) * (2 * np.pi / spec.angular_azimuth)
    polar_mesh, azimuth_mesh = np.meshgrid(polar, azimuth, indexing="ij")
    directions = np.stack(
        (
            np.sqrt(1 - polar_mesh**2) * np.cos(azimuth_mesh),
            np.sqrt(1 - polar_mesh**2) * np.sin(azimuth_mesh),
            polar_mesh,
        ),
        axis=-1,
    ).reshape(-1, 3)
    angular_weights = np.repeat(polar_weights * 2 * np.pi / len(azimuth), len(azimuth))
    atomic_rules = {
        molecule.atom_symbol(index): (
            (radii[:, None, None] * directions).reshape(-1, 3),
            (measures[:, None] * angular_weights).ravel(),
        )
        for index in range(molecule.natm)
    }
    if spec.element_radii or spec.partition_iterations != 3 or spec.version != 1:
        raise ValueError(
            "reference adapter requires version-1 unit radii/three Becke passes"
        )
    if backend == "gpu4pyscf":
        import cupy as cp
        import gpu4pyscf
        from gpu4pyscf.dft import gen_grid, rks

        from benchmarks.readme_wb97mv import (
            configure_reference_full_fock,
            configure_reference_mixed_rsh_df,
            reference_vv10_domain,
        )

        engine = rks.RKS(molecule)
        atomic_rules = {
            symbol: tuple(cp.asarray(array) for array in arrays)
            for symbol, arrays in atomic_rules.items()
        }
        becke_scheme = gen_grid.original_becke
    else:
        engine = dft.RKS(molecule)
        becke_scheme = dft.gen_grid.original_becke
    if auxiliary_basis is not None:
        engine = engine.density_fit(auxbasis=auxiliary_basis)
    engine.xc = functional
    if backend == "gpu4pyscf":
        configure_reference_full_fock(engine)
        if auxiliary_basis is not None and engine.do_nlc():
            configure_reference_mixed_rsh_df(engine)
    for grids in (
        (engine.grids, engine.nlcgrids) if engine.do_nlc() else (engine.grids,)
    ):
        grids.radii_adjust = None
        grids.becke_scheme = becke_scheme
        grids.prune = None
        grids.gen_atomic_grids = lambda *args, **kwargs: atomic_rules
    engine.small_rho_cutoff = 0
    engine.conv_tol, engine.conv_tol_grad = 1e-12, 1e-10
    engine.direct_scf_tol, engine.max_cycle = 1e-14, 180
    engine.init_guess = "1e"
    if backend == "gpu4pyscf":
        cp.cuda.Stream.null.synchronize()
    started = perf_counter()
    with (
        reference_vv10_domain(1e-8)
        if backend == "gpu4pyscf" and engine.do_nlc()
        else nullcontext()
    ):
        energy = float(engine.kernel())
        if not engine.converged:
            raise RuntimeError(f"independent {functional} SCF did not converge")
        gradient = engine.nuc_grad_method()
        gradient.grid_response = True
        forces = -gradient.kernel()
        if backend == "gpu4pyscf":
            forces = cp.asnumpy(forces)
            cp.cuda.Stream.null.synchronize()
    return {
        "energy": energy,
        "forces": forces.tolist(),
        "converged": True,
        "seconds": perf_counter() - started,
        "pyscf_version": pyscf.__version__,
        "reference_backend": backend,
        **(
            {"gpu4pyscf_version": gpu4pyscf.__version__}
            if backend == "gpu4pyscf"
            else {}
        ),
        "functional": functional,
        "density_fitting": auxiliary_basis is not None,
        "aos": molecule.nao,
        "grid_points": len(engine.grids.weights),
        "iterations": engine.cycles,
        "threads": threads,
        "conv_tol": engine.conv_tol,
        "conv_tol_grad": engine.conv_tol_grad,
        "direct_scf_tol": engine.direct_scf_tol,
        "grid_response": True,
    }


def errors(observed: dict[str, Any], oracle: dict[str, Any]) -> dict[str, float]:
    """Measure componentwise force errors without hiding invalid endpoints."""
    forces = np.asarray(observed["forces"])
    reference_forces = np.asarray(oracle["forces"])
    if (
        forces.shape != reference_forces.shape
        or forces.ndim != 2
        or forces.shape[1] != 3
        or not forces.size
    ):
        raise ValueError("incompatible force dimensions")
    delta = forces - reference_forces
    energy = abs(observed["energy"] - oracle["energy"])
    if not np.isfinite(delta).all() or not np.isfinite(energy):
        raise ValueError("nonfinite energy/forces")
    return {
        "energy_Eh": float(energy),
        "max_force_Eh_per_bohr": float(np.max(np.abs(delta))),
        "rms_force_Eh_per_bohr": float(np.sqrt(np.mean(delta**2))),
    }


def summarize(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Only publish speed ratios for independently accepted measured endpoints."""
    measured = [record for record in records if record["phase"] == "measured"]
    if not measured or not all(record["oracle_gate"] for record in measured):
        raise ValueError("missing or rejected independently checked endpoints")
    times = {
        mode: median(record["seconds"] for record in measured if record["mode"] == mode)
        for mode in MODES
    }
    return {
        "median_seconds": times,
        "speedup_vs_fp64": {mode: times["fp64"] / times[mode] for mode in MODES},
    }


def native_sample(mode: str, options: dict[str, Any], case_name: str) -> dict[str, Any]:
    """Include construction, SCF, all analytic forces, export and cache release."""
    runtime = ctypes.CDLL("libcudart.so.12")
    if runtime.cudaDeviceSynchronize() != 0:
        raise RuntimeError("CUDA synchronization failed before endpoint")
    case = benchmark_cases()[case_name]
    started = perf_counter()
    calculator = (
        PrecisionCalculator(**options, policy=mode)
        if mode in ("audit_auto", "fast")
        else Calculator(**options, precision=mode)
    )
    try:
        result = calculator.singlepoint(
            case.atoms,
            charge=case.charge,
            multiplicity=case.multiplicity,
            properties=("energy", "forces"),
        )
        forces = np.array(result.forces, copy=True)
    finally:
        calculator.clear_cache()
    if runtime.cudaDeviceSynchronize() != 0:
        raise RuntimeError("CUDA synchronization failed after endpoint")
    seconds = perf_counter() - started
    if not result.converged or result.executed_backend != "cuda":
        raise RuntimeError("endpoint did not converge on CUDA")
    try:
        precision_from_result(result)
        evidence_error = None
    except ValueError as error:
        evidence_error = str(error)
    return {
        "mode": mode,
        "seconds": seconds,
        "energy": result.energy,
        "forces": forces.tolist(),
        "iterations": result.iterations,
        "executed_backend": result.executed_backend,
        "converged": result.converged,
        "precision": result.precision,
        "precision_evidence_error": evidence_error,
        "precision_census_scope": "native SCF census; not a gradient instruction census",
        "ks_diagnostic": result.ks_diagnostic.to_payload()
        if result.ks_diagnostic
        else None,
        "precision_validation": (
            result.precision_validation.to_dict()
            if result.precision_validation
            else None
        ),
    }


def main() -> None:
    """Persist first calls and every repeat before checking numerical acceptance."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--engine", choices=("reference", "native"), required=True)
    parser.add_argument("--cases", nargs="+", choices=CASES, default=CASES)
    parser.add_argument("--grid", type=int, nargs=3, default=(48, 16, 32))
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--reference", type=Path)
    parser.add_argument("--output", type=raw_output_path, required=True)
    args = parser.parse_args()
    if args.repeats < 1 or args.threads < 1:
        parser.error("repeats and threads must be positive")
    if args.engine == "native" and (
        not os.environ.get("SLURM_JOB_ID") or not args.reference
    ):
        parser.error("native execution requires finite srun and --reference")
    spec = GridSpec(
        radial_points=args.grid[0],
        angular_polar=args.grid[1],
        angular_azimuth=args.grid[2],
    )
    report: dict[str, Any] = {
        "schema": SCHEMA,
        "engine": args.engine,
        "status": "running",
        "cases": {},
        "timing_contract": "fresh calculator/SCF + analytic forces + host export + release",
        "warm_contract": "process caches only; hcore initial density on every solve",
        "repeats": args.repeats,
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)

    def save() -> None:
        args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")

    try:
        oracle = None
        if args.engine == "native":
            oracle = json.loads(args.reference.read_text())
            if (
                oracle["schema"] != SCHEMA
                or oracle["engine"] != "reference"
                or oracle["status"] != "measured"
            ):
                raise ValueError("independent reference is incomplete or incompatible")
            report["reference_sha256"] = hashlib.sha256(
                args.reference.read_bytes()
            ).hexdigest()
        for case_name in args.cases:
            scientific = protocol(case_name, spec)
            entry: dict[str, Any] = {"protocol": scientific}
            report["cases"][case_name] = entry
            save()
            if args.engine == "reference":
                entry["reference"] = reference(case_name, spec, args.threads)
                save()
                continue
            if oracle["cases"][case_name]["protocol"] != scientific:
                raise ValueError("reference scientific target differs")
            case = benchmark_cases()[case_name]
            options = {
                "method": "pbe0-rks",
                "basis": case.generativeqc_basis,
                "device": "cuda",
                "basis_representation": case.basis_representation,
                "ks_options": KsOptions(grid=spec),
                "initial_guess": None,
                **{
                    key: scientific[key]
                    for key in (
                        "energy_tolerance",
                        "density_tolerance",
                        "screening_tolerance",
                        "max_iterations",
                    )
                },
            }
            probe = Calculator(**options)
            entry["build"] = native_build_metadata(probe)
            require_tuned_native_build(entry["build"])
            probe.clear_cache()
            entry["records"] = []
            for repeat in range(-1, args.repeats):
                order = MODES if repeat % 2 else tuple(reversed(MODES))
                for mode in order:
                    print(f"{case_name} repeat={repeat} mode={mode}", flush=True)
                    row = native_sample(mode, options, case_name)
                    row.update(
                        repeat=repeat,
                        phase="first-call-after-probe" if repeat < 0 else "measured",
                    )
                    row["oracle_errors"] = errors(
                        row, oracle["cases"][case_name]["reference"]
                    )
                    row["oracle_gate"] = (
                        row["oracle_errors"]["energy_Eh"]
                        <= scientific["energy_gate_Eh"]
                        and row["oracle_errors"]["max_force_Eh_per_bohr"]
                        <= scientific["force_gate_Eh_per_bohr"]
                    )
                    entry["records"].append(row)
                    save()
                    if not row["oracle_gate"]:
                        raise ValueError("independent PBE0 energy/force gate failed")
            entry["summary"] = summarize(entry["records"])
            baseline = next(row for row in entry["records"] if row["mode"] == "fp64")
            for row in entry["records"]:
                row["fp64_errors"] = errors(row, baseline)
            save()
        report["status"] = "measured"
    except Exception as error:
        report.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        save()


if __name__ == "__main__":
    main()
