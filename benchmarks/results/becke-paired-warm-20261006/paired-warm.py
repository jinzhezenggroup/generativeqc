"""Shared-state warm E+F pairs with legal, untimed gradient-owner reconstruction.

This supplements, rather than replaces, the independent grouped cold/moved
protocol. Only one native SCF batch is resident. Its engine-local post-solve
warm density is frozen through the public API. Each arm closes the entire
private gradient executor and primes a new executor before timing; a live
native Becke configuration is never changed after topology installation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
from generativeqc import Calculator, GridSpec, KsOptions
from generativeqc_compiler.common.evidence import block_error, canonical_hash, outcome
from generativeqc_compiler.common.performance import (
    assess_comparison,
    measure_interleaved,
)
from generativeqc_compiler.common.timing import interleaved_selection_order

from benchmarks._support import cuda_accelerator_metadata, environment_metadata
from benchmarks.compare_gpu4pyscf_batch import (
    native_build_metadata,
    require_tuned_native_build,
)
from benchmarks.dft_force_components import normalize_force_work
from benchmarks.readme_hf_scaling import scaling_cases
from benchmarks.readme_omol25 import load_comparison_basis, protocol, source_hashes
from benchmarks.readme_pbe0 import PBE0


def main() -> None:
    """Validate all complete calls, retaining priming and measured work separately."""
    import cupy as cp

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--atoms", type=int, choices=(48, 96), required=True)
    parser.add_argument("--basis-file", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--feasibility", action="store_true")
    args = parser.parse_args()
    assert os.environ.get("SLURM_JOB_ID") and os.environ.get("CUDA_VISIBLE_DEVICES")
    assert args.repeats >= 5 or (args.feasibility and args.repeats == 1)
    reference = json.loads(args.reference.read_text())
    case = scaling_cases()[f"water-{args.atoms}"]
    basis, _ = load_comparison_basis(
        args.basis_file, case, role="orbital", compute_forces=True
    )
    grid = GridSpec(radial_points=48, angular_polar=16, angular_azimuth=32)
    scientific = protocol(args.atoms, basis, grid, 5, benchmark=PBE0)
    assert reference["protocol"] == scientific and reference["stage"] == "complete"
    record: dict[str, Any] = {
        "schema": "generativeqc.becke-paired-warm.v1",
        "status": "running",
        "protocol": scientific,
        "reference_sha256": hashlib.sha256(args.reference.read_bytes()).hexdigest(),
        "paired_repeats": args.repeats,
        "feasibility": args.feasibility,
        "source_file_sha256": source_hashes(),
        "samples": [],
        "priming": [],
        "setup": [],
        "assessments": {},
        "experiment_scope": "Supplemental shared-state warm/moved-warm E+F, not cold/moved timing. Public frozen native post-solve density, no reference seeds. Entire private gradient executor reconstruction/prime is untimed and separately retained; native configure-once guard is not bypassed.",
        "inputs_hash_scope": "Scientific protocol/geometry and one shared native post-solve warm snapshot frozen by the public API. No independent byte digest of the native retained dm0 is available; identity relies on the inspected frozen-snapshot contract and single unchanged SCF batch.",
        "environment": environment_metadata(),
    }
    owner = None
    original_force = None
    force_work = None

    def save(stage: str) -> None:
        record["stage"] = stage
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(record, indent=2) + "\n")

    def retained(
        item: Any, geometry: int, *, expected_selection: int | None = None
    ) -> dict[str, Any]:
        oracle = next(
            row
            for row in reference["records"]
            if row["geometry"] == geometry and row["phase"] in ("cold", "moved")
        )
        assert item.status == 0 and item.converged
        errors = {
            "energy": block_error([item.energy], [oracle["energy"]], atol=1e-8, rtol=0),
            "forces": block_error(item.forces, oracle["forces"], atol=1e-7, rtol=0),
        }
        assert all(error["passed"] for error in errors.values())
        assert force_work is not None
        work = normalize_force_work(force_work)
        becke = work["becke_owners"]["stationary"]
        assert becke["profile_enabled"] is False
        assert becke["profile_intrusive"] is False
        assert all(value is None for value in becke["profiled_ms"].values())
        expected = (
            int(os.environ["GENERATIVEQC_STATIONARY_BECKE_PRIMITIVE"] == "coefficients")
            if expected_selection is None
            else expected_selection
        )
        assert becke["selection"]["becke_primitive_requested"] == expected
        assert becke["selection"]["becke_primitive_selected"] == expected
        assert becke["work_counters"]["becke_phase_points"] == args.atoms * 48 * 16 * 32
        return {
            "energy": item.energy,
            "forces": item.forces.tolist(),
            "errors": errors,
            "converged": item.converged,
            "status": item.status,
            "iterations": item.iterations,
            "fock_builds": item.fock_builds,
            "warm_start_used": item.warm_start_used,
            "warm_start_fallback": item.warm_start_fallback,
            "native_ks_diagnostic": item.ks_diagnostic.to_payload(),
            "native_force_components": work,
        }

    try:
        os.environ["GENERATIVEQC_STATIONARY_BECKE_PRIMITIVE"] = "off"
        calculator = Calculator(
            method="pbe0-rks",
            basis=basis,
            device="cuda",
            basis_representation="spherical",
            ks_options=KsOptions(grid=grid),
            energy_tolerance=1e-12,
            density_tolerance=1e-10,
            screening_tolerance=1e-12,
            max_iterations=100,
        )
        record["native_build"] = native_build_metadata(calculator)
        require_tuned_native_build(record["native_build"])
        owner = calculator.prepare_batch(
            [scientific["geometries_bohr"][0]], warm_start=True
        )
        original_force = owner._public_dft_cuda_force

        def observed_force(*values: Any) -> Any:
            nonlocal force_work
            result = original_force(*values)
            force_work = result[1]
            return result

        owner._public_dft_cuda_force = observed_force
        for geometry in (0, 1):
            phase = "warm" if geometry == 0 else "moved-warm"
            coords = (
                None
                if geometry == 0
                else [np.asarray([xyz for _, xyz in scientific["geometries_bohr"][1]])]
            )
            owner.set_warm_start_updates(True)
            cp.cuda.Stream.null.synchronize()
            started = perf_counter()
            force_work = None
            item = owner.execute(
                coords, strict=False, properties=("energy", "forces")
            ).items[0]
            cp.cuda.Stream.null.synchronize()
            seconds = perf_counter() - started
            record["setup"].append(
                {
                    "geometry": geometry,
                    "seconds": seconds,
                    "diagnostics": retained(item, geometry),
                }
            )
            owner.set_warm_start_updates(False)
            inputs_hash = canonical_hash(
                {
                    "protocol": scientific,
                    "geometry": geometry,
                    "initial_density": "one unchanged native batch post-cold/post-move warm snapshot, publicly frozen",
                }
            )
            executed = []

            def prepare(
                selection: str,
                *,
                replay_coords: Any = coords,
                replay_geometry: int = geometry,
                replay_phase: str = phase,
            ) -> None:
                """Follow PreparedBatch's whole-executor replacement policy, never toggle native flags."""
                nonlocal force_work
                assert owner._warm_updates is False
                cp.cuda.Stream.null.synchronize()
                started = perf_counter()
                previous = owner._stationary_cuda_execution
                if previous is not None:
                    previous.close()
                owner._stationary_cuda_execution = None
                os.environ["GENERATIVEQC_STATIONARY_BECKE_PRIMITIVE"] = (
                    "coefficients" if selection == "candidate" else "off"
                )
                force_work = None
                primed = owner.execute(
                    replay_coords, strict=False, properties=("energy", "forces")
                ).items[0]
                cp.cuda.Stream.null.synchronize()
                prime_seconds = perf_counter() - started
                diagnostics = retained(primed, replay_geometry)
                assert diagnostics["iterations"] == diagnostics["fock_builds"] == 1
                assert (
                    diagnostics["warm_start_used"]
                    and not diagnostics["warm_start_fallback"]
                )
                record["priming"].append(
                    {
                        "phase": replay_phase,
                        "selection": selection,
                        "seconds": prime_seconds,
                        "diagnostics": diagnostics,
                        "warm_updates": owner._warm_updates,
                    }
                )
                save(f"{replay_phase}/prime-{len(record['priming'])}")

            def evaluate(
                selection: str,
                *,
                replay_coords: Any = coords,
                results: list[tuple[Any, Any]] = executed,
            ) -> dict[str, int]:
                nonlocal force_work
                assert owner._warm_updates is False
                force_work = None
                item = owner.execute(
                    replay_coords, strict=False, properties=("energy", "forces")
                ).items[0]
                results.append((item, force_work))
                return {"call_index": len(results) - 1}

            if args.feasibility:
                samples = []
                for selection in interleaved_selection_order(1):
                    prepare(selection)
                    cp.cuda.Stream.null.synchronize()
                    started = perf_counter()
                    diagnostics = evaluate(selection)
                    cp.cuda.Stream.null.synchronize()
                    samples.append(
                        {
                            "selection": selection,
                            "seconds": perf_counter() - started,
                            "inputs_hash": inputs_hash,
                            "workload": "energy-plus-force",
                            "synchronized": True,
                            "diagnostics": diagnostics,
                        }
                    )
            else:
                samples = measure_interleaved(
                    evaluate,
                    cp.cuda.Stream.null.synchronize,
                    workload="energy-plus-force",
                    inputs_hash=inputs_hash,
                    repeats=args.repeats,
                    prepare=prepare,
                )
            for sample in samples:
                item, force_work = executed[sample["diagnostics"]["call_index"]]
                sample["diagnostics"] = retained(
                    item,
                    geometry,
                    expected_selection=int(sample["selection"] == "candidate"),
                )
                sample["phase"] = phase
                assert (
                    sample["diagnostics"]["iterations"]
                    == sample["diagnostics"]["fock_builds"]
                    == 1
                )
                assert (
                    sample["diagnostics"]["warm_start_used"]
                    and not sample["diagnostics"]["warm_start_fallback"]
                )
            record["samples"].extend(samples)
            record["assessments"][phase] = (
                assess_comparison(samples)
                if not args.feasibility
                else outcome(
                    "not-run",
                    "One pair establishes harness feasibility only, not a performance comparison.",
                )
            )
            save(f"{phase}/complete")
        record["status"] = "measured"
        record["environment"] = environment_metadata(
            accelerator=cuda_accelerator_metadata(cp)
        )
        save("complete")
    except BaseException as error:
        record.update(status="failed", error=f"{type(error).__name__}: {error}")
        save(record.get("stage", "setup"))
        raise
    finally:
        if owner is not None:
            if original_force is not None:
                owner._public_dft_cuda_force = original_force
            owner.close()


if __name__ == "__main__":
    main()
