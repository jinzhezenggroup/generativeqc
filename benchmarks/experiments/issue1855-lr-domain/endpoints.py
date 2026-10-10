"""Qualify isolated LR schedules against an existing, exact-input independent oracle.

The oracle remains immutable and is outside native timers. Intrusive kernel
ledger runs are explicitly separated from the alternating clean population.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
from generativeqc import Calculator, GridSpec, KsOptions
from generativeqc._stationary_composite_cuda import (
    PreparedCompositeStationaryCudaGradient,
)

from benchmarks.compare_gpu4pyscf_batch import (
    load_comparison_basis,
    native_build_metadata,
)
from benchmarks.readme_hf_scaling import scaling_cases
from benchmarks.readme_omol25 import OMOL25, check_record, protocol


def main() -> None:
    """Retain complete returned E/F calls and every same-geometry oracle pairing."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--basis-file", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--repeat-offset", type=int, default=0)
    parser.add_argument("--observe", action="store_true")
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("repeats must be positive")
    if args.repeat_offset < 0:
        parser.error("repeat offset must be nonnegative")
    if not os.environ.get("SLURM_JOB_ID") or not os.environ.get("CUDA_VISIBLE_DEVICES"):
        parser.error("run inside a finite GPU srun with its assigned visibility")
    if not os.environ.get("LD_PRELOAD"):
        parser.error("the isolated LR acceptance interposer must be preloaded")
    reference = json.loads(args.reference.read_text())
    scientific = reference["protocol"]
    basis, _reference_basis = load_comparison_basis(
        args.basis_file,
        scaling_cases()["water-12"],
        role="orbital",
        compute_forces=True,
    )
    grid = GridSpec(**scientific["grid"])
    expected = protocol(
        12,
        basis,
        grid,
        5,
        benchmark=replace(OMOL25, reference_full_fock=True),
    )
    if reference["status"] != "measured" or scientific != expected:
        raise ValueError(
            "independent oracle does not match the exact scientific protocol"
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema": "generativeqc.lr-domain-acceptance.v1",
        "status": "running",
        "protocol": scientific,
        "intrusive": args.observe,
        "clean_timing": not args.observe,
        "slurm_job_id": os.environ["SLURM_JOB_ID"],
        "cuda_visible_devices": os.environ["CUDA_VISIBLE_DEVICES"],
        "reference_sha256": hashlib.sha256(args.reference.read_bytes()).hexdigest(),
        "basis_file_sha256": hashlib.sha256(args.basis_file.read_bytes()).hexdigest(),
        "library_sha256": hashlib.sha256(
            Path(os.environ["GENERATIVEQC_LIBRARY"]).read_bytes()
        ).hexdigest(),
        "interposer_sha256": hashlib.sha256(
            Path(os.environ["LD_PRELOAD"]).read_bytes()
        ).hexdigest(),
        "cold_scope": "prepare_batch through returned host E+F; constructor/import/teardown excluded",
        "replay_scope": "execute through returned host E+F",
        "force_allowances": {"max_device_bytes": 4 << 30, "max_host_bytes": 4 << 30},
        "default_promotion": False,
        "records": [],
        "independent_pairs": [],
    }

    def save() -> None:
        args.output.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")

    original_force = PreparedCompositeStationaryCudaGradient.execute

    def bounded_force(self: Any, *values: Any, **kwargs: Any) -> Any:
        """Match the independently qualified campaign's explicit force allowances."""
        kwargs.update(payload["force_allowances"])
        return original_force(self, *values, **kwargs)

    PreparedCompositeStationaryCudaGradient.execute = bounded_force
    save()
    try:
        for repeat in range(args.repeat_offset, args.repeat_offset + args.repeats):
            modes = (
                ("indexed", "triangular")
                if repeat % 2 == 0
                else ("triangular", "indexed")
            )
            for mode in modes:
                os.environ["GENERATIVEQC_ACCEPTANCE_LR_DOMAIN"] = mode
                if args.observe:
                    os.environ["GENERATIVEQC_ACCEPTANCE_LR_RECEIPTS"] = str(
                        args.output.parent / f"{mode}-work.jsonl"
                    )
                else:
                    os.environ.pop("GENERATIVEQC_ACCEPTANCE_LR_RECEIPTS", None)
                calculator = Calculator(
                    method="wb97m-v-rks",
                    basis=basis,
                    basis_representation="spherical",
                    device="cuda",
                    ks_options=KsOptions(grid=grid),
                    precision="fp64",
                    energy_tolerance=1e-12,
                    density_tolerance=1e-10,
                    screening_tolerance=1e-12,
                    max_iterations=100,
                )
                build = native_build_metadata(calculator)
                if build["library_sha256"] != payload["library_sha256"]:
                    raise RuntimeError(
                        "calculator replaced the explicitly frozen library"
                    )
                payload["native_build"] = build
                started = time.perf_counter()
                with calculator.prepare_batch(
                    [scientific["geometries_bohr"][0]], warm_start=True
                ) as batch:
                    for geometry, atoms in enumerate(scientific["geometries_bohr"]):
                        coordinates = np.asarray([xyz for _, xyz in atoms])
                        for replay in range(6):
                            if geometry or replay:
                                started = time.perf_counter()
                            item = batch.execute(
                                coordinates=[coordinates],
                                properties=("energy", "forces"),
                                strict=True,
                            ).items[0]
                            elapsed = time.perf_counter() - started
                            phase = (
                                ("cold" if geometry == 0 else "moved")
                                if replay == 0
                                else ("warm" if geometry == 0 else "moved-warm")
                            )
                            row = {
                                "arm": mode,
                                "repeat": repeat,
                                "replay": replay,
                                "geometry": geometry,
                                "phase": phase,
                                "seconds": elapsed,
                                "energy": item.energy,
                                "forces": item.forces.tolist()
                                if item.forces is not None
                                else None,
                                "converged": item.converged,
                                "status": item.status,
                                "detail": item.status_message,
                                "iterations": item.iterations,
                                "fock_builds": item.fock_builds,
                                "energy_change": item.energy_change,
                                "density_rms": item.density_rms,
                                "physical_residual_rms": item.physical_residual_rms,
                                "warm_start_used": item.warm_start_used,
                                "warm_start_fallback": item.warm_start_fallback,
                                "native_ks_diagnostic": (
                                    item.ks_diagnostic.to_payload()
                                    if item.ks_diagnostic is not None
                                    else None
                                ),
                            }
                            payload["records"].append(row)
                            save()
                            for oracle in reference["records"]:
                                if oracle["geometry"] == geometry:
                                    gate = check_record(row, oracle)
                                    payload["independent_pairs"].append(
                                        {
                                            "native_row": len(payload["records"]) - 1,
                                            "gate": gate,
                                        }
                                    )
                                    if not gate["gate"]:
                                        raise RuntimeError(
                                            f"independent E/F gate failed: {gate}"
                                        )
                            save()
                            print(mode, repeat, phase, replay, elapsed, flush=True)
        payload["status"] = "PASS"
    except BaseException as error:
        payload["status"] = "failed"
        payload["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        PreparedCompositeStationaryCudaGradient.execute = original_force
        save()


if __name__ == "__main__":
    main()
