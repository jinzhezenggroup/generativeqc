"""Measure dense/occupied first K on one exactly frozen, factor-backed seed.

This is an additional attribution experiment, not a replacement for the
independent baseline/candidate cold and moved qualification populations.
"""

import argparse
import hashlib
import json
import os
from dataclasses import asdict
from pathlib import Path
from time import perf_counter

import numpy as np

from benchmarks.ks_df_warm_factor import (
    check_endpoint,
    load_comparison_basis,
    native_build_metadata,
    require_tuned_native_build,
    raw_output_path,
    scaling_cases,
    source_provenance,
)
from generativeqc_compiler.common.evidence import canonical_hash
from generativeqc_compiler.common.performance import assess_comparison
from generativeqc_compiler.common.timing import interleaved_selection_order


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--functional", choices=["m06-2x", "pbe0", "b3lyp"], required=True)
    parser.add_argument("--atoms", type=int, choices=[48, 96], required=True)
    parser.add_argument("--basis-file", type=Path, required=True)
    parser.add_argument("--auxiliary-file", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output = raw_output_path(args.output)
    if not all(os.environ.get(key) for key in ["SLURM_JOB_ID", "CUDA_VISIBLE_DEVICES"]):
        parser.error("a finite Slurm allocation and its device visibility are required")
    if os.environ.get("GENERATIVEQC_DF_TRACE"):
        parser.error("interleaved timings must not include diagnostic tracing")

    import cupy as cp
    from generativeqc import Calculator, GridSpec, KsOptions

    oracle = json.loads(args.reference.read_text())
    protocol = oracle["protocol"]
    case = scaling_cases()[f"water-{args.atoms}"]
    basis, _ = load_comparison_basis(args.basis_file, case, role="orbital", compute_forces=True)
    auxiliary, _ = load_comparison_basis(
        args.auxiliary_file, case, role="auxiliary", compute_forces=True
    )
    assert oracle["engine"] == "reference" and oracle["status"] == "measured"
    assert protocol["functional"] == args.functional and protocol["atoms"] == args.atoms
    assert protocol["basis_sha256"] == hashlib.sha256(args.basis_file.read_bytes()).hexdigest()
    assert protocol["auxiliary_sha256"] == hashlib.sha256(args.auxiliary_file.read_bytes()).hexdigest()
    grid = GridSpec(radial_points=48, angular_polar=16, angular_azimuth=32)
    assert protocol["grid"] == json.loads(json.dumps(asdict(grid)))
    assert protocol["density_fitting"] and protocol["df_relative_threshold"] == 1e-10
    assert protocol["energy_tolerance"] == 1e-12 and protocol["density_tolerance"] == 1e-10
    assert protocol["screening_tolerance"] == 1e-12 and protocol["max_iterations"] == 200
    record = {
        "source": source_provenance(),
        "runner_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "protocol": protocol,
        "slurm_job_id": os.environ["SLURM_JOB_ID"],
        "cuda_visible_devices": os.environ["CUDA_VISIBLE_DEVICES"],
        "status": "running",
        "records": [],
        "timings": [],
        "seed_policy": "candidate-produced cold/moved seed; one owner; frozen updates; ABBA first-K switch only",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)

    def save():
        args.output.write_text(json.dumps(record, indent=2, allow_nan=False) + "\n")

    os.environ["GENERATIVEQC_CUDA_KS_WARM_FACTOR"] = "1"
    os.environ["GENERATIVEQC_CUDA_KS_WARM_FACTOR_AUDIT"] = "0"
    owner = None
    try:
        save()
        calculator = Calculator(
            method=f"{args.functional}-rks", basis=basis, auxiliary_basis=auxiliary,
            basis_representation="spherical", device="cuda", precision="fp64",
            density_fitting="cuda", density_fitting_relative_threshold=1e-10,
            energy_tolerance=1e-12, density_tolerance=1e-10, screening_tolerance=1e-12,
            max_iterations=200, ks_options=KsOptions(grid=grid),
        )
        record["native_build"] = native_build_metadata(calculator)
        require_tuned_native_build(record["native_build"])
        owner = calculator.prepare_batch([case.atoms], warm_start=True)
        for geometry_index, geometry in enumerate(protocol["geometries_bohr"]):
            expected = next(row for row in oracle["records"] if row["geometry"] == geometry_index)
            coordinates = np.asarray([position for _, position in geometry])
            owner.set_warm_start_updates(True)
            selections = ["seed"] + list(interleaved_selection_order(5))
            for index, selection in enumerate(selections):
                os.environ["GENERATIVEQC_CUDA_KS_WARM_FACTOR"] = str(int(selection != "baseline"))
                cp.cuda.runtime.deviceSynchronize()
                started = perf_counter()
                item = owner.execute(
                    None if geometry_index == 0 else [coordinates],
                    strict=False, properties=("energy", "forces"),
                ).items[0]
                cp.cuda.runtime.deviceSynchronize()
                seconds = perf_counter() - started
                row = {
                    "geometry": geometry_index, "selection": selection,
                    "sequence": index, "seconds": seconds,
                    "energy": item.energy,
                    "forces": None if item.forces is None else item.forces.tolist(),
                    "fock_builds": item.fock_builds, "iterations": item.iterations,
                    "diagnostic": item.ks_diagnostic.to_payload() if item.ks_diagnostic else None,
                    **check_endpoint(item, expected),
                }
                record["records"].append(row)
                save()
                if not row["gate"]:
                    raise RuntimeError("independent DF E/force gate failed; no samples discarded")
                if index == 0:
                    owner.set_warm_start_updates(False)
                else:
                    record["timings"].append({
                        "selection": selection, "seconds": seconds, "synchronized": True,
                        "inputs_hash": canonical_hash({"protocol": protocol, "geometry": geometry_index}),
                        "workload": "unchanged-geometry" if geometry_index == 0 else "energy-plus-force",
                        "fock_builds": item.fock_builds,
                    })
            save()
        record["comparison"] = assess_comparison(record["timings"])
        record["status"] = "measured"
        save()
    except BaseException as error:
        record.update(status="failed", error=f"{type(error).__name__}: {error}")
        save()
        raise
    finally:
        if owner is not None:
            owner.close()


if __name__ == "__main__":
    main()
