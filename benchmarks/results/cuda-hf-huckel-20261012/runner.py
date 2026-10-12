"""Frozen CUDA RHF cold-endpoint screen; references are benchmark-only work.

Each measurement runs in a fresh process under finite Slurm, using the same
offline basis and native library. Construction, strict seed admission, target
retry and analytic forces belong to the timed prepare/execute endpoint. Final
density export and independent verification are outside the timer. This screen
does not select or change a production default.
"""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import importlib.metadata
import json
import os
import socket
import subprocess
from pathlib import Path
from statistics import median
from time import perf_counter

import numpy as np


def digest(path: Path) -> str:
    """Stream large library identities without allocating a binary-sized copy."""
    checksum = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            checksum.update(chunk)
    return checksum.hexdigest()


def verify_source(path: Path) -> dict:
    """Check the complete frozen inventory, not a caller-supplied source label."""
    audit = json.loads(path.read_text())
    source = Path(os.environ["SOURCE"])
    for relative, expected in audit["qualified_source_sha256"].items():
        if digest(source / relative) != expected:
            raise RuntimeError(f"qualified source changed: {relative}")
    return audit


def isolated_gpu(bus: str) -> dict:
    """Reject a shared physical GPU without changing Slurm-assigned visibility."""

    def query(fields: str, kind: str) -> list[list[str]]:
        result = subprocess.run(
            ["nvidia-smi", f"--query-{kind}={fields}", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            check=True,
            timeout=10,
        )
        return [
            [value.strip() for value in line.split(",")]
            for line in result.stdout.splitlines()
        ]

    devices = query("uuid,pci.bus_id,name,driver_version", "gpu")
    selected = [
        row
        for row in devices
        if row[1].split(":", 1)[1].lower() == bus.split(":", 1)[1].lower()
    ]
    if len(selected) != 1:
        raise RuntimeError("cannot bind assigned CUDA device to a physical GPU")
    uuid, address, name, driver = selected[0]
    processes = [
        row
        for row in query("gpu_uuid,pid,process_name", "compute-apps")
        if row[0] == uuid
    ]
    if any(int(row[1]) != os.getpid() for row in processes):
        raise RuntimeError(f"assigned GPU has another CUDA process: {processes}")
    return {
        "uuid": uuid,
        "pci_bus_id": address,
        "name": name,
        "driver": driver,
        "processes": processes,
    }


def export_density(owner: object, path: Path) -> dict:
    """Export the returned density after timing, without seeding another target."""
    from generativeqc import _native
    from generativeqc.checkpoint import _descriptor, _pointer

    state = _descriptor()
    query = owner._library.generativeqc_batch_get_hf_warm_state
    _native.check(owner._library, query(owner._batch, 0, ctypes.byref(state)))
    if not state.present or not 0 < state.density_count <= 768**2:
        raise RuntimeError("missing or unbounded final target density")
    density = np.empty(state.density_count, dtype=np.float64)
    coordinates = np.empty(state.coordinate_count, dtype=np.float64)
    state.density, state.coordinates = _pointer(density), _pointer(coordinates)
    _native.check(owner._library, query(owner._batch, 0, ctypes.byref(state)))
    np.save(path, density)
    return {"path": path.name, "sha256": digest(path), "count": density.size}


def numerical_gate(record: dict, oracle: dict) -> dict:
    """Preserve the established independent energy, force and residual gates."""
    energy_error = abs(record["energy"] - oracle["energy"])
    force_error = float(np.max(np.abs(np.asarray(record["forces"]) - oracle["forces"])))
    residual = record["physical_residual_rms"]
    passed = (
        record["status_code"] == 0
        and record["converged"]
        and np.isfinite(energy_error)
        and energy_error <= 1e-8
        and np.isfinite(force_error)
        and force_error <= 1e-7
        and np.isfinite(residual)
        and 0 <= residual <= 1e-10
    )
    return {
        "gate": bool(passed),
        "energy_error": energy_error,
        "force_error": force_error,
        "physical_residual_rms": residual,
    }


def summarize(directory: Path, cases: tuple[int, ...] = (48, 96)) -> dict:
    """Require every paired repeat and numerical gate before reporting gains."""
    records = []
    paths = sorted(
        path
        for arm in ("reference", "core", "huckel")
        for path in directory.glob(f"{arm}-*.json")
    )
    for path in paths:
        record = json.loads(path.read_text())
        if record.get("schema") == "generativeqc.cuda-hf-huckel.v1":
            record["record_sha256"] = digest(path)
            records.append(record)
    summary = {"cases": {}, "default_promoted": False}
    for atoms in cases:
        samples = [record for record in records if record.get("atoms") == atoms]
        oracle = [record for record in samples if record["arm"] == "reference"]
        if len(oracle) != 1 or oracle[0]["status"] != "measured":
            raise ValueError("one successful independent reference is required")
        groups = {
            arm: sorted(
                (record for record in samples if record["arm"] == arm),
                key=lambda record: record["repeat"],
            )
            for arm in ("core", "huckel")
        }
        identities = {
            (
                record["library_sha256"],
                record["source_audit_sha256"],
                record["basis_sha256"],
                record["harness_sha256"],
                json.dumps(record["gpu"], sort_keys=True),
            )
            for group in groups.values()
            for record in group
        }
        if len(identities) != 1:
            raise ValueError("all repeats must use one frozen source/library/device")
        for arm, group in groups.items():
            if [record["repeat"] for record in group] != [0, 1, 2]:
                raise ValueError("three paired fresh-process repeats are required")
            for record in group:
                if record["status"] != "measured" or not record["acceptance"]["gate"]:
                    raise ValueError("a failed endpoint cannot be dropped")
                if record["reference_sha256"] != oracle[0]["record_sha256"]:
                    raise ValueError("independent reference identity changed")
                if arm == "huckel" and (
                    record["initial_guess"]["outcome"] != "used"
                    or record["initial_guess"]["target_attempts"] != 1
                    or not record["initial_guess"]["work_counters_complete"]
                    or record["initial_guess"]["preliminary_fock_builds"] <= 0
                ):
                    raise ValueError(
                        "seeded profitability requires complete work and no retry"
                    )
        paired_errors = []
        for core, seeded in zip(groups["core"], groups["huckel"], strict=True):
            for field in (
                "library_sha256",
                "source_audit_sha256",
                "basis_sha256",
                "gpu",
            ):
                if core[field] != seeded[field]:
                    raise ValueError("mixed source, library, basis or device")
            densities = []
            for record in (core, seeded):
                path = directory / record["final_density"]["path"]
                if digest(path) != record["final_density"]["sha256"]:
                    raise ValueError("final density bytes changed")
                densities.append(np.load(path))
            error = float(np.max(np.abs(densities[0] - densities[1])))
            if not np.isfinite(error) or error > 1e-8:
                raise ValueError("paired final density gate failed")
            paired_errors.append(error)
        times = {
            arm: [record["complete_seconds"] for record in group]
            for arm, group in groups.items()
        }
        reductions = [
            1 - seeded / core
            for core, seeded in zip(times["core"], times["huckel"], strict=True)
        ]
        gain = 1 - median(times["huckel"]) / median(times["core"])
        summary["cases"][str(atoms)] = {
            "complete_seconds": times,
            "median_seconds": {arm: median(values) for arm, values in times.items()},
            "paired_reductions": reductions,
            "median_reduction": gain,
            "strict_five_percent_gate": gain > 0.05 and min(reductions) > 0.05,
            "target_fock_builds": {
                arm: [record["target_fock_builds"] for record in group]
                for arm, group in groups.items()
            },
            "huckel_preparation_seconds": [
                record["initial_guess"]["preparation_seconds"]
                for record in groups["huckel"]
            ],
            "maximum_paired_density_error": max(paired_errors),
        }
    return summary


def measure(args: argparse.Namespace) -> None:
    """Record failed measurements too; never replace assigned GPU visibility."""
    if not os.environ.get("SLURM_JOB_ID") or not os.environ.get("CUDA_VISIBLE_DEVICES"):
        raise RuntimeError("real GPU work requires finite srun")
    import cupy as cp
    from generativeqc import Calculator, InitialGuessSpec

    from benchmarks.compare_gpu4pyscf_batch import (
        _configure_reference_scf,
        load_comparison_basis,
        native_build_metadata,
        require_tuned_native_build,
    )
    from benchmarks.readme_hf_scaling import scaling_cases

    audit = verify_source(args.source_audit)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    case = scaling_cases()[f"water-{args.atoms}"]
    basis, reference_basis = load_comparison_basis(
        args.basis_file, case, role="orbital", compute_forces=True
    )
    record = {
        "schema": "generativeqc.cuda-hf-huckel.v1",
        "atoms": args.atoms,
        "arm": args.arm,
        "repeat": args.repeat,
        "status": "running",
        "method": "exact-fp64-rhf",
        "basis": "offline-def2-svp-spherical",
        "energy_tolerance": 1e-12,
        "density_tolerance": 1e-10,
        "screening_tolerance": 1e-12,
        "endpoint": "fresh process/owner prepare + synchronized energy/analytic forces",
        "harness_sha256": digest(Path(__file__)),
        "basis_sha256": digest(args.basis_file),
        "source_audit_sha256": digest(args.source_audit),
        "slurm_job_id": os.environ["SLURM_JOB_ID"],
        "cuda_visible_devices": os.environ["CUDA_VISIBLE_DEVICES"],
        "host": socket.gethostname(),
        "environment": {
            name: value
            for name, value in os.environ.items()
            if name.startswith(("GENERATIVEQC_", "OMP_", "MKL_", "OPENBLAS_"))
        },
        "versions": {
            name: importlib.metadata.version(name)
            for name in ("numpy", "pyscf", "cupy-cuda12x", "gpu4pyscf-cuda12x")
        },
    }
    owner = None
    try:
        cp.cuda.runtime.deviceSynchronize()
        gpu = cp.cuda.runtime.getDeviceProperties(cp.cuda.runtime.getDevice())
        record["gpu"] = {
            "name": gpu["name"].decode(),
            "pci_bus_id": cp.cuda.Device().pci_bus_id,
        }
        record["gpu_isolation_before"] = isolated_gpu(record["gpu"]["pci_bus_id"])
        if args.arm == "reference":
            from gpu4pyscf.scf import RHF
            from pyscf import gto

            molecule = gto.M(
                atom=case.atoms, basis=reference_basis, unit="Bohr", verbose=0
            )
            engine = RHF(molecule)
            _configure_reference_scf(
                engine,
                energy_tolerance=1e-12,
                gradient_tolerance=1e-10,
                max_iterations=100,
                full_fock=True,
            )
            cp.cuda.runtime.deviceSynchronize()
            started = perf_counter()
            energy = engine.kernel()
            forces = cp.asnumpy(-engine.nuc_grad_method().kernel())
            cp.cuda.runtime.deviceSynchronize()
            record.update(
                energy=float(energy),
                forces=forces.tolist(),
                converged=bool(engine.converged),
                complete_seconds=perf_counter() - started,
            )
            if not engine.converged:
                raise RuntimeError("independent reference did not converge")
        else:
            if args.reference is None:
                raise ValueError("independent reference is required")
            oracle = json.loads(args.reference.read_text())
            if oracle["status"] != "measured" or oracle["atoms"] != args.atoms:
                raise ValueError("invalid independent reference")
            record["reference_sha256"] = digest(args.reference)
            calc = Calculator(
                "rhf",
                basis=basis,
                device="cuda",
                basis_representation="spherical",
                energy_tolerance=1e-12,
                density_tolerance=1e-10,
                screening_tolerance=1e-12,
                max_iterations=100,
                initial_guess=InitialGuessSpec("huckel")
                if args.arm == "huckel"
                else None,
            )
            metadata = native_build_metadata(calc)
            require_tuned_native_build(metadata)
            library = Path(metadata["library_path"])
            record["library_sha256"] = metadata["library_sha256"]
            if record["library_sha256"] != audit["qualified_library_sha256"]:
                raise RuntimeError("loaded library is not the qualified binary")
            record["native_build"] = metadata
            cp.cuda.runtime.deviceSynchronize()
            started = perf_counter()
            owner = calc.prepare_batch([case.atoms], warm_start=True)
            cp.cuda.runtime.deviceSynchronize()
            record["prepare_seconds"] = perf_counter() - started
            item = owner.execute(strict=False, properties=("energy", "forces")).items[0]
            cp.cuda.runtime.deviceSynchronize()
            record["complete_seconds"] = perf_counter() - started
            record.update(
                energy=float(item.energy),
                forces=item.forces.tolist(),
                converged=bool(item.converged),
                status_code=int(item.status),
                iterations=int(item.iterations),
                physical_residual_rms=item.physical_residual_rms,
                initial_guess=item.initial_guess,
                precision=item.precision,
                incremental_direct_jk=item.incremental_direct_jk,
            )
            precision = item.precision
            if not precision or not precision["operator_work_counters_valid"]:
                raise RuntimeError("missing authoritative target Fock census")
            record["target_fock_builds"] = sum(
                precision[name]
                for name in (
                    "mixed_stage_fock_builds",
                    "strict_stage_fock_builds",
                    "post_scf_fock_builds",
                )
            )
            record["final_density"] = export_density(
                owner, args.output.with_suffix(".npy")
            )
            # HF's legacy public scalar residual is absent. Independently
            # rebuild F(D) at the exported target density after the timer.
            from gpu4pyscf.scf import RHF
            from pyscf import gto

            oracle_molecule = gto.M(
                atom=case.atoms, basis=reference_basis, unit="Bohr", verbose=0
            )
            oracle_engine = RHF(oracle_molecule)
            _configure_reference_scf(
                oracle_engine,
                energy_tolerance=1e-12,
                gradient_tolerance=1e-10,
                max_iterations=100,
                full_fock=True,
            )
            density = np.load(args.output.with_suffix(".npy")).reshape(
                oracle_molecule.nao, oracle_molecule.nao
            )
            overlap = cp.asnumpy(oracle_engine.get_ovlp())
            fock = cp.asnumpy(oracle_engine.get_fock(dm=cp.asarray(density)))
            residual = fock @ density @ overlap - overlap @ density @ fock
            record["reported_physical_residual_rms"] = record["physical_residual_rms"]
            record["physical_residual_rms"] = float(np.sqrt(np.mean(residual**2)))
            record["physical_residual_owner"] = "independent GPU4PySCF F(exported D)"
            record["acceptance"] = numerical_gate(record, oracle)
            if not record["acceptance"]["gate"]:
                raise RuntimeError("independent numerical gate failed")
            if digest(library) != record["library_sha256"]:
                raise RuntimeError("native binary changed during measurement")
        record["status"] = "measured"
        record["gpu_isolation_after"] = isolated_gpu(record["gpu"]["pci_bus_id"])
        verify_source(args.source_audit)
    except Exception as error:
        record.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(record, indent=2, allow_nan=False) + "\n")
        if owner is not None:
            owner.close()


def main() -> None:
    """Keep summary validation usable without importing GPU packages."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path)
    parser.add_argument("--atoms", type=int, choices=(3, 48, 96))
    parser.add_argument("--arm", choices=("reference", "core", "huckel"))
    parser.add_argument("--repeat", type=int, default=0)
    parser.add_argument("--basis-file", type=Path)
    parser.add_argument("--source-audit", type=Path)
    parser.add_argument("--reference", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.summary:
        cases = (args.atoms,) if args.atoms is not None else (48, 96)
        print(json.dumps(summarize(args.summary, cases), indent=2, allow_nan=False))
    else:
        if any(
            getattr(args, name) is None
            for name in ("atoms", "arm", "basis_file", "source_audit", "output")
        ):
            parser.error(
                "measurement needs atoms, arm, basis-file, source-audit and output"
            )
        measure(args)


if __name__ == "__main__":
    main()
