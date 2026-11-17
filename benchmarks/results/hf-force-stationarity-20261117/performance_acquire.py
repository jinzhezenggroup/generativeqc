"""Acquire one fresh-process complete endpoint chain, with untimed oracle checks."""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import importlib.abc
import json
import os
import sys
import time
from pathlib import Path

import numpy as np


class NoOracle(importlib.abc.MetaPathFinder):
    """Prevent reference work from entering the production endpoint process."""

    def find_spec(self, fullname: str, path: object, target: object = None) -> None:
        if fullname == "pyscf" or fullname.startswith("pyscf."):
            raise AssertionError("production must not import the scientific oracle")


def main() -> None:
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("fixture", type=Path)
    parser.add_argument("destination", type=Path)
    parser.add_argument("--arm", required=True)
    parser.add_argument("--profile", required=True)
    parser.add_argument("--commit", required=True)
    args = parser.parse_args()
    assert os.environ.get("SLURM_JOB_ID") and os.environ.get("CUDA_VISIBLE_DEVICES")
    assert "GENERATIVEQC_CUDA_RHF_CONVERGENCE" not in os.environ
    fixture = json.loads(args.fixture.read_text())
    request, reference = fixture["request"], fixture["oracle"]
    sys.meta_path.insert(0, NoOracle())
    started = time.perf_counter_ns()
    from generativeqc import Calculator, _native, profiles

    import_ns = time.perf_counter_ns() - started
    settings = {"density_fitting": "none"}
    if request["mode"] == "df":
        auxiliary = args.destination.with_suffix(".auxiliary.json")
        auxiliary.write_text(json.dumps(reference["auxiliary"]) + "\n")
        settings = {"density_fitting": "cuda", "auxiliary_basis": auxiliary}
    started = time.perf_counter_ns()
    calculator = Calculator(
        method=fixture["method"],
        basis=request["basis"],
        basis_representation="cartesian",
        device="cuda",
        **settings,
    )
    construct_ns = time.perf_counter_ns() - started
    assert (
        calculator._energy_tolerance,
        calculator._density_tolerance,
        calculator._screening_tolerance,
    ) == (1e-9, 1e-5, 1e-12)
    record = {
        "pid": os.getpid(),
        "arm": args.arm,
        "profile": args.profile,
        "commit": args.commit,
        "fixture_sha256": hashlib.sha256(args.fixture.read_bytes()).hexdigest(),
        "slurm_job_id": os.environ["SLURM_JOB_ID"],
        "cuda_visible_devices": os.environ["CUDA_VISIBLE_DEVICES"],
        "import_ns": import_ns,
        "construct_ns": construct_ns,
        "samples": [],
        "timing_scope": "synchronous prepared execute including native work and host result export; cold also reports prepare; oracle checks and JSON outside timing",
    }
    started = time.perf_counter_ns()
    owner = calculator.prepare_batch(
        [request["geometries"]["initial"]],
        multiplicities=[2 if fixture["method"] == "uhf" else 1],
        warm_start=True,
    )
    record["prepare_ns"] = time.perf_counter_ns() - started
    try:
        for phase, geometry, forces, updates in (
            ("cold", "initial", True, True),
            ("warm", "initial", True, False),
            ("moved", "moved", True, True),
            ("moved_warm", "moved", True, False),
            ("energy_only", "initial", False, True),
            ("energy_to_forces", "initial", True, True),
        ):
            coordinates = [position for _, position in request["geometries"][geometry]]
            started = time.perf_counter_ns()
            owner.set_warm_start_updates(updates)
            result = owner.execute(
                [coordinates],
                strict=True,
                properties=("energy", "forces") if forces else ("energy",),
            ).items[0]
            elapsed = time.perf_counter_ns() - started
            expected = reference["records"][geometry]
            errors = {"energy": abs(result.energy - expected["energy"])}
            if forces:
                errors.update(
                    forces=float(np.max(np.abs(result.forces - expected["forces"]))),
                    translation=float(np.max(np.abs(result.forces.sum(axis=0)))),
                )
            valid = bool(
                result.converged
                and result.succeeded
                and result.executed_backend == "cuda"
                and errors["energy"] <= 1e-9
                and (
                    not forces
                    or (errors["forces"] <= 1e-7 and errors["translation"] <= 1e-7)
                )
            )
            record["samples"].append(
                {
                    "phase": phase,
                    "geometry": geometry,
                    "execute_ns": elapsed,
                    "endpoint_ns": elapsed
                    + (record["prepare_ns"] if phase == "cold" else 0),
                    "iterations": result.iterations,
                    "fock_builds": result.fock_builds,
                    "warm_start_used": result.warm_start_used,
                    "warm_start_fallback": result.warm_start_fallback,
                    "converged": result.converged,
                    "status": result.status,
                    "executed_backend": result.executed_backend,
                    "energy": result.energy,
                    "forces": result.forces.tolist() if forces else None,
                    "errors": errors,
                    "scientifically_valid": valid,
                    "incremental_direct_jk": result.incremental_direct_jk,
                    "precision": result.precision,
                    "force_evaluations": None,
                    "force_evaluations_note": "not exposed by the public result; not inferred from requested properties",
                }
            )
    finally:
        started = time.perf_counter_ns()
        owner.close()
        calculator.clear_cache()
        record["cleanup_ns"] = time.perf_counter_ns() - started
    record["first_use_ns"] = (
        record["import_ns"]
        + record["construct_ns"]
        + record["samples"][0]["endpoint_ns"]
    )
    record["native_probe"] = profiles.probe_device(
        _native.load_library(device="cuda", device_id=0)
    )
    from generativeqc.autotune import source_identity

    source = Path(sys.modules["generativeqc"].__file__).resolve().parents[2]
    record["python_source_identity"] = source_identity(source)
    record["recipe_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    assert record["native_probe"]["source_identity"] == record["python_source_identity"]
    device = record["native_probe"]["device"]
    assert device["release_build"] == 1 and device["fast_compile"] == 0
    assert device["official_profile"] == (
        "sm_120" if args.profile == "auto" else "portable_cuda"
    )
    driver = ctypes.CDLL("libcuda.so.1")
    handle, identifier = ctypes.c_int(), (ctypes.c_ubyte * 16)()
    assert driver.cuInit(0) == 0
    assert driver.cuDeviceGet(ctypes.byref(handle), 0) == 0
    assert driver.cuDeviceGetUuid_v2(ctypes.byref(identifier), handle) == 0
    record["device_uuid"] = bytes(identifier).hex()
    library = Path(os.environ["GENERATIVEQC_LIBRARY"])
    with library.open("rb") as stream:
        record["library_sha256"] = hashlib.file_digest(stream, "sha256").hexdigest()
    args.destination.write_text(
        json.dumps(record, separators=(",", ":"), allow_nan=False) + "\n"
    )
    print(
        json.dumps(
            {
                "arm": args.arm,
                "case": args.fixture.stem,
                "phases": [
                    {
                        "phase": item["phase"],
                        "milliseconds": item["endpoint_ns"] / 1e6,
                        "iterations": item["iterations"],
                        "valid": item["scientifically_valid"],
                    }
                    for item in record["samples"]
                ],
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
