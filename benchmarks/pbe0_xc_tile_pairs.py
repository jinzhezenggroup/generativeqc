"""Complete PBE0 E+F A/B pairs changing only SCF XC scheduling.

Both arms use the same binary, scientific inputs and fixed 256-point force
policy. Each prepared owner independently converges and then publicly freezes
its warm snapshot; this is not a claim of identical density bytes. Setup and
priming are retained outside replay timing, and every timed solver call must
perform one physical iteration and one Fock build without warm fallback.
With --point-batch-tiles both arms retain 256-point AO maps/contractions and
only the candidate batches independent point domains within an explicit cap.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from contextlib import contextmanager
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
from generativeqc import Calculator, GridSpec, KsOptions
from generativeqc_compiler.common.evidence import block_error, canonical_hash
from generativeqc_compiler.common.performance import (
    assess_comparison,
    measure_interleaved,
)
from generativeqc_compiler.common.timing import interleaved_selection_order

from benchmarks._support import (
    cuda_accelerator_metadata,
    environment_metadata,
    raw_output_path,
)
from benchmarks.compare_gpu4pyscf_batch import (
    native_build_metadata,
    require_tuned_native_build,
)
from benchmarks.dft_force_components import normalize_force_work
from benchmarks.ks_preliminary_density import read_ao_work
from benchmarks.readme_hf_scaling import scaling_cases
from benchmarks.readme_omol25 import load_comparison_basis, protocol, source_hashes
from benchmarks.readme_pbe0 import PBE0


def main() -> None:
    """Save full vectors, actual solver histories and per-arm work telemetry."""
    import cupy as cp

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--atoms", type=int, choices=(48, 96), required=True)
    parser.add_argument("--basis-file", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--output", type=raw_output_path, required=True)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--feasibility", action="store_true")
    parser.add_argument("--point-batch-tiles", type=int)
    parser.add_argument("--point-batch-bytes", type=int, default=32 * 1024 * 1024)
    args = parser.parse_args()
    if not os.environ.get("SLURM_JOB_ID") or not os.environ.get("CUDA_VISIBLE_DEVICES"):
        parser.error("real-GPU execution requires a finite Slurm allocation")
    if args.repeats < 5 and not (args.feasibility and args.repeats == 1):
        parser.error("at least five pairs required unless explicitly feasibility-only")
    if args.point_batch_tiles is not None and (
        args.point_batch_tiles < 2 or args.point_batch_bytes < 0
    ):
        parser.error(
            "point batching needs at least two tiles and a nonnegative byte cap"
        )
    os.environ["GENERATIVEQC_STATIONARY_BECKE_PRIMITIVE"] = "off"
    case = scaling_cases()[f"water-{args.atoms}"]
    basis, _ = load_comparison_basis(
        args.basis_file, case, role="orbital", compute_forces=True
    )
    grid = GridSpec(radial_points=48, angular_polar=16, angular_azimuth=32)
    scientific = protocol(args.atoms, basis, grid, 5, benchmark=PBE0)
    reference = json.loads(args.reference.read_text())
    assert reference["protocol"] == scientific and reference["stage"] == "complete"
    tiles = {"baseline": 256, "candidate": 256 if args.point_batch_tiles else 512}

    @contextmanager
    def point_batch_selection(arm: str) -> Any:
        """Reapply each arm when moved coordinates rebuild the native owner."""
        values = {
            "GENERATIVEQC_CUDA_XC_BATCH_TILES": str(
                args.point_batch_tiles
                if arm == "candidate" and args.point_batch_tiles
                else 1
            ),
            "GENERATIVEQC_CUDA_XC_BATCH_BYTES": str(args.point_batch_bytes),
        }
        previous = {name: os.environ.get(name) for name in values}
        os.environ.update(values)
        try:
            yield
        finally:
            for name, value in previous.items():
                if value is None:
                    os.environ.pop(name, None)
                else:
                    os.environ[name] = value

    root = Path(__file__).resolve().parents[1]
    record: dict[str, Any] = {
        "schema": (
            "generativeqc.pbe0-xc-point-batch-pairs.v1"
            if args.point_batch_tiles
            else "generativeqc.pbe0-xc-tile-pairs.v1"
        ),
        "protocol": scientific,
        "scf_tiles": tiles,
        "point_batch_request": (
            {"tiles": args.point_batch_tiles, "device_bytes": args.point_batch_bytes}
            if args.point_batch_tiles is not None
            else None
        ),
        "force_tile_points": 256,
        "density_scope": "separate independently converged publicly frozen warm snapshots",
        "scope": "complete E+F replays; setup/prime excluded and retained; not cold/moved acceleration",
        "source_file_sha256": source_hashes()
        | {
            "benchmarks/pbe0_xc_tile_pairs.py": hashlib.sha256(
                Path(__file__).read_bytes()
            ).hexdigest(),
            "python/generativeqc/batch.py": hashlib.sha256(
                (root / "python/generativeqc/batch.py").read_bytes()
            ).hexdigest(),
        },
        "reference_sha256": hashlib.sha256(args.reference.read_bytes()).hexdigest(),
        "environment": environment_metadata(accelerator=cuda_accelerator_metadata(cp)),
        "slurm_job_id": os.environ["SLURM_JOB_ID"],
        "feasibility": args.feasibility,
        "preparation": [],
        "setup": [],
        "priming": [],
        "samples": [],
        "assessments": {},
    }
    owners: dict[str, Any] = {}
    force_work: dict[str, Any] = {}

    def save(stage: str) -> None:
        record["stage"] = stage
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(record, indent=2) + "\n")

    def diagnostics(
        item: Any, arm: str, geometry: int, work: Any, ao_work: Any
    ) -> dict[str, Any]:
        oracle = next(
            row
            for row in reference["records"]
            if row["geometry"] == geometry and row["phase"] in ("cold", "moved")
        )
        errors = {
            "energy": block_error([item.energy], [oracle["energy"]], atol=1e-8, rtol=0),
            "forces": block_error(item.forces, oracle["forces"], atol=1e-7, rtol=0),
        }
        assert item.status == 0 and item.converged
        assert all(error["passed"] for error in errors.values()), errors
        assert item.ks_diagnostic.tile_points == tiles[arm]
        return {
            "energy": item.energy,
            "forces": item.forces.tolist(),
            "errors": errors,
            "iterations": item.iterations,
            "fock_builds": item.fock_builds,
            "warm_start_used": item.warm_start_used,
            "warm_start_fallback": item.warm_start_fallback,
            "native_ks_diagnostic": item.ks_diagnostic.to_payload(),
            "native_scf_ao_work": ao_work,
            "native_force_components": normalize_force_work(work),
            "native_force_work_raw": work,
        }

    def observe_force(arm: str, original: Any) -> Any:
        def observed(*values: Any) -> Any:
            result = original(*values)
            force_work[arm] = result[1]
            return result

        return observed

    try:
        for arm, tile in tiles.items():
            calculator = Calculator(
                method="pbe0-rks",
                basis=basis,
                device="cuda",
                basis_representation="spherical",
                ks_options=KsOptions(grid=grid, tile_points=tile),
                energy_tolerance=1e-12,
                density_tolerance=1e-10,
                screening_tolerance=1e-12,
                max_iterations=100,
            )
            native_build = native_build_metadata(calculator)
            require_tuned_native_build(native_build)
            if "native_build" in record:
                assert native_build == record["native_build"]
            record["native_build"] = native_build
            cp.cuda.Stream.null.synchronize()
            started = perf_counter()
            with point_batch_selection(arm):
                owner = calculator.prepare_batch(
                    [scientific["geometries_bohr"][0]], warm_start=True
                )
            owners[arm] = owner
            cp.cuda.Stream.null.synchronize()
            record["preparation"].append(
                {"selection": arm, "seconds": perf_counter() - started}
            )
            owner._public_dft_cuda_force = observe_force(
                arm, owner._public_dft_cuda_force
            )
            save(f"prepared-{arm}")

        for geometry in (0, 1):
            phase = "warm" if geometry == 0 else "moved-warm"
            coords = (
                None
                if geometry == 0
                else [np.asarray([xyz for _, xyz in scientific["geometries_bohr"][1]])]
            )
            for arm, owner in owners.items():
                owner.set_warm_start_updates(True)
                cp.cuda.Stream.null.synchronize()
                started = perf_counter()
                with point_batch_selection(arm):
                    item = owner.execute(
                        coords, strict=False, properties=("energy", "forces")
                    ).items[0]
                cp.cuda.Stream.null.synchronize()
                record["setup"].append(
                    {
                        "selection": arm,
                        "geometry": geometry,
                        "seconds": perf_counter() - started,
                        "diagnostics": diagnostics(
                            item, arm, geometry, force_work[arm], read_ao_work(owner)
                        ),
                    }
                )
                owner.set_warm_start_updates(False)
                cp.cuda.Stream.null.synchronize()
                started = perf_counter()
                with point_batch_selection(arm):
                    primed = owner.execute(
                        coords, strict=False, properties=("energy", "forces")
                    ).items[0]
                cp.cuda.Stream.null.synchronize()
                record["priming"].append(
                    {
                        "selection": arm,
                        "phase": phase,
                        "seconds": perf_counter() - started,
                        "diagnostics": diagnostics(
                            primed, arm, geometry, force_work[arm], read_ao_work(owner)
                        ),
                    }
                )
                save(f"{phase}/prime-{arm}")

            completed = []

            def evaluate(
                arm: str,
                *,
                replay_coords: Any = coords,
                results: list[tuple[Any, Any, Any]] = completed,
            ) -> dict[str, int]:
                assert owners[arm]._warm_updates is False
                with point_batch_selection(arm):
                    item = (
                        owners[arm]
                        .execute(
                            replay_coords, strict=False, properties=("energy", "forces")
                        )
                        .items[0]
                    )
                results.append((item, force_work[arm], read_ao_work(owners[arm])))
                return {"call_index": len(results) - 1}

            inputs_hash = canonical_hash(
                {
                    "protocol": scientific,
                    "geometry": geometry,
                    "density_scope": record["density_scope"],
                }
            )
            if args.feasibility:
                samples = []
                for arm in interleaved_selection_order(1):
                    cp.cuda.Stream.null.synchronize()
                    started = perf_counter()
                    details = evaluate(arm)
                    cp.cuda.Stream.null.synchronize()
                    samples.append(
                        {
                            "selection": arm,
                            "seconds": perf_counter() - started,
                            "inputs_hash": inputs_hash,
                            "workload": "energy-plus-force",
                            "synchronized": True,
                            "diagnostics": details,
                        }
                    )
            else:
                samples = measure_interleaved(
                    evaluate,
                    cp.cuda.Stream.null.synchronize,
                    workload="energy-plus-force",
                    inputs_hash=inputs_hash,
                    repeats=args.repeats,
                )
            for sample in samples:
                arm = sample["selection"]
                item, work, ao_work = completed[sample["diagnostics"]["call_index"]]
                sample["diagnostics"] = diagnostics(item, arm, geometry, work, ao_work)
                assert item.iterations == item.fock_builds == 1
                assert item.warm_start_used and not item.warm_start_fallback
                sample["phase"] = phase
            record["samples"].extend(samples)
            record["assessments"][phase] = assess_comparison(samples)
            save(phase)
    finally:
        for owner in owners.values():
            owner.close()
    save("complete")


if __name__ == "__main__":
    main()
