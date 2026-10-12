"""Fresh-process CUDA DF RHF E+force timing with an independent DF oracle.

Run each arm/repeat in a separate finite Slurm invocation or process within one
allocation. The timer covers prepare/execute and synchronized analytic forces;
oracle work and final-density export are explicitly outside it. This diagnostic
does not claim to qualify DF KS performance or change the production policy.
"""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import os
import socket
import subprocess
from pathlib import Path
from time import perf_counter

import numpy as np


def digest(path: Path) -> str:
    """Hash source/library provenance without a binary-sized temporary."""
    checksum = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            checksum.update(chunk)
    return checksum.hexdigest()


def measure(args: argparse.Namespace) -> None:
    """Record successful and failed endpoints without altering scheduler visibility."""
    if not os.environ.get("SLURM_JOB_ID") or not os.environ.get("CUDA_VISIBLE_DEVICES"):
        raise RuntimeError("CUDA measurements require finite srun")
    import cupy as cp
    from generativeqc import Calculator, InitialGuessSpec, Primitive, Shell, _native
    from generativeqc.checkpoint import _descriptor, _pointer
    from gpu4pyscf.scf import RHF
    from pyscf import gto

    from benchmarks.pbe0_water_buckets import load_case

    atoms = (
        [
            ("O", (0, 0, 0)),
            ("H", (0, -1.43233673, 1.10715266)),
            ("H", (0, 1.43233673, 1.10715266)),
        ]
        if args.atoms == 3
        else load_case(f"cold{args.atoms}").atoms
    )

    def native_basis(name: str) -> tuple:
        return tuple(
            Shell(
                index, shell[0], tuple(Primitive(row[0], row[1]) for row in shell[1:])
            )
            for index, (element, _) in enumerate(atoms)
            for shell in gto.basis.load(name, element)
        )

    library = Path(_native.load_library(device="cuda")._name).resolve()
    root = Path(__file__).resolve().parents[1]
    source_files = sorted(
        path
        for directory in ("src", "include", "python", "cmake")
        for path in (root / directory).rglob("*")
        if path.is_file()
        and path.suffix in (".cpp", ".hpp", ".h", ".cu", ".cuh", ".py", ".cmake")
    )
    source_identity = {
        str(path.relative_to(root)): digest(path) for path in source_files
    }
    record = {
        "schema": "generativeqc.huckel-df-endpoint.v1",
        "arm": args.arm,
        "repeat": args.repeat,
        "atoms": args.atoms,
        "host": socket.gethostname(),
        "slurm_job_id": os.environ["SLURM_JOB_ID"],
        "cuda_visible_devices": os.environ["CUDA_VISIBLE_DEVICES"],
        "source_identity_sha256": hashlib.sha256(
            json.dumps(source_identity, sort_keys=True).encode()
        ).hexdigest(),
        "library_sha256": digest(library),
        "harness_sha256": digest(Path(__file__)),
        "protocol": {
            "method": "rhf",
            "backend": "cuda",
            "density_fitting": "cuda",
            "basis": "def2-svp",
            "auxiliary_basis": "def2-universal-jkfit",
            "representation": "spherical",
            "precision": "fp64",
            "energy_tolerance": 1e-12,
            "density_tolerance": 1e-10,
            "screening_tolerance": 1e-12,
            "metric_relative_threshold": 1e-10,
            "max_iterations": 100,
            "properties": ["energy", "forces"],
        },
    }
    owner = None
    try:
        device = cp.cuda.runtime.getDevice()
        properties = cp.cuda.runtime.getDeviceProperties(device)
        bus = cp.cuda.runtime.deviceGetPCIBusId(device)
        if isinstance(bus, bytes):
            bus = bus.decode()
        record["gpu"] = {
            "name": properties["name"].decode(),
            "pci_bus_id": bus,
            "driver_version": cp.cuda.runtime.driverGetVersion(),
        }
        record["nvidia_processes"] = subprocess.run(
            [
                "nvidia-smi",
                "--query-compute-apps=gpu_uuid,pid,process_name",
                "--format=csv,noheader",
            ],
            check=True,
            text=True,
            capture_output=True,
            timeout=10,
        ).stdout.splitlines()
        calc = Calculator(
            "rhf",
            native_basis("def2-svp"),
            device="cuda",
            auxiliary_basis=native_basis("def2-universal-jkfit"),
            basis_representation="spherical",
            density_fitting="cuda",
            energy_tolerance=1e-12,
            density_tolerance=1e-10,
            screening_tolerance=1e-12,
            max_iterations=100,
            initial_guess=None if args.arm == "core" else InitialGuessSpec("huckel"),
        )
        cp.cuda.runtime.deviceSynchronize()
        started = perf_counter()
        owner = calc.prepare_batch([atoms])
        result = owner.execute(strict=True, properties=("energy", "forces")).items[0]
        cp.cuda.runtime.deviceSynchronize()
        record["complete_seconds"] = perf_counter() - started
        record.update(
            energy=result.energy,
            forces=result.forces.tolist(),
            iterations=result.iterations,
            target_fock_builds=result.fock_builds,
            initial_guess=result.initial_guess,
        )
        descriptor = _descriptor()
        query = owner._library.generativeqc_batch_get_hf_warm_state
        _native.check(owner._library, query(owner._batch, 0, ctypes.byref(descriptor)))
        density = np.empty(descriptor.density_count)
        coordinates = np.empty(descriptor.coordinate_count)
        descriptor.density, descriptor.coordinates = (
            _pointer(density),
            _pointer(coordinates),
        )
        _native.check(owner._library, query(owner._batch, 0, ctypes.byref(descriptor)))
        molecule = gto.M(atom=atoms, basis="def2-svp", unit="Bohr", verbose=0)
        oracle = RHF(molecule).density_fit(auxbasis="def2-universal-jkfit")
        oracle.conv_tol, oracle.conv_tol_grad, oracle.max_cycle = 1e-13, 1e-10, 100
        oracle.kernel()
        if not oracle.converged:
            raise RuntimeError("independent DF oracle did not converge")
        density = density.reshape(molecule.nao, molecule.nao)
        overlap = cp.asnumpy(oracle.get_ovlp())
        fock = cp.asnumpy(oracle.get_fock(dm=cp.asarray(density)))
        residual = fock @ density @ overlap - overlap @ density @ fock
        errors = {
            "energy": abs(result.energy - oracle.e_tot),
            "force": float(
                np.max(
                    np.abs(
                        np.asarray(result.forces)
                        + cp.asnumpy(oracle.nuc_grad_method().kernel())
                    )
                )
            ),
            "density": float(np.max(np.abs(density - cp.asnumpy(oracle.make_rdm1())))),
            "physical_residual_rms": float(np.sqrt(np.mean(residual**2))),
        }
        gates = {
            "energy": 1e-8,
            "force": 1e-7,
            "density": 1e-8,
            "physical_residual_rms": 1e-8,
        }
        record["acceptance"] = {
            "errors": errors,
            "tolerances": gates,
            "gate": all(errors[key] <= limit for key, limit in gates.items()),
        }
        if not record["acceptance"]["gate"]:
            raise RuntimeError("independent DF endpoint gate failed")
        if digest(library) != record["library_sha256"] or any(
            digest(root / relative) != value
            for relative, value in source_identity.items()
        ):
            raise RuntimeError("measured source/library changed")
        record["status"] = "measured"
    except Exception as error:
        record.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(record, indent=2, allow_nan=False) + "\n")
        if owner is not None:
            owner.close()


def main() -> None:
    """Select one fresh-process sample; defaults never select a production policy."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--atoms", required=True, type=int, choices=(3, 48, 96))
    parser.add_argument("--arm", required=True, choices=("core", "huckel"))
    parser.add_argument("--repeat", type=int, default=0)
    parser.add_argument("--output", required=True, type=Path)
    measure(parser.parse_args())


if __name__ == "__main__":
    main()
