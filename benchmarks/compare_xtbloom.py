"""Measure fresh-SCC molecular GFN2 energy/force endpoints against xTBloom.

Run each engine in its own process in the same Slurm GPU allocation. Warm
measurements reuse the public calculator but never reuse a converged SCC state.
Every sample retains energy, forces and iteration counts for subsequent gates;
construction and host-visible calculation times are recorded separately. The
comparison also reports cold_total: construction plus the first calculation.
Cleanup after all samples is recorded separately from the next case's constructor.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import time
from pathlib import Path
from statistics import median

import numpy as np

try:
    from benchmarks._retention import raw_output_path
except ModuleNotFoundError:
    from _retention import raw_output_path

ROOT = Path(__file__).resolve().parents[1]
TIMING_CONTRACT = "calculator-cold-v3-lazy-library-load"


def source_revision(directory: Path) -> str | None:
    """Report checkout provenance when available, including exported runners."""
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=directory,
        text=True,
        capture_output=True,
        check=False,
    )
    return result.stdout.strip() if result.returncode == 0 else None


def geometry_sha256(symbols: list[str], positions: list[list[float]]) -> str:
    """Bind a measurement to the atom identities and coordinates it consumes."""
    return hashlib.sha256(
        json.dumps(
            {"symbols": symbols, "positions": positions}, sort_keys=True
        ).encode()
    ).hexdigest()


def compare_reports(reference: dict, candidate: dict) -> dict:
    """Gate every sample before summarizing any complete endpoint speedup."""
    if not reference["rows"] or not candidate["rows"]:
        raise ValueError("empty measurement report")
    if reference["settings"] != candidate["settings"]:
        raise ValueError("scientific settings differ")
    if reference["device"] != candidate["device"]:
        raise ValueError("execution devices differ")
    # Keep old/old receipts usable as historical evidence, but never silently
    # compare their preceding-calculator cleanup with an isolated constructor.
    timing_contract = reference.get("timing_contract", "legacy-mixed-cleanup")
    if timing_contract != candidate.get("timing_contract", "legacy-mixed-cleanup"):
        raise ValueError("calculator timing contracts differ; remeasure both engines")
    if [row["case"] for row in reference["rows"]] != [
        row["case"] for row in candidate["rows"]
    ]:
        raise ValueError("case inventories differ")
    rows = []
    for expected, actual in zip(reference["rows"], candidate["rows"], strict=True):
        for measured in (expected, actual):
            construction = measured.get("construction_seconds")
            if (
                construction is None
                or not np.isfinite(construction)
                or construction < 0
            ):
                raise ValueError("missing or invalid calculator construction timing")
        if expected["geometry_sha256"] != actual["geometry_sha256"]:
            raise ValueError("input geometries differ")
        errors = []
        for left, right in zip(expected["samples"], actual["samples"], strict=True):
            if (left["mode"], left["repeat"]) != (right["mode"], right["repeat"]):
                raise ValueError("sample inventories differ")
            left_geometry = left.get("geometry_sha256")
            right_geometry = right.get("geometry_sha256")
            if not left_geometry or not right_geometry:
                raise ValueError(
                    "sample geometry identities are missing; remeasure inputs"
                )
            if left_geometry != right_geometry:
                raise ValueError("sample input geometries differ")
            energy_error = abs(left["energy"] - right["energy"])
            left_forces, right_forces = (
                np.asarray(left["forces"]),
                np.asarray(right["forces"]),
            )
            if (
                left_forces.shape != right_forces.shape
                or left_forces.ndim != 2
                or left_forces.shape[1] != 3
            ):
                raise ValueError("force shapes differ or are invalid")
            force_error = float(np.max(np.abs(left_forces - right_forces)))
            if not np.isfinite([energy_error, force_error]).all():
                raise ValueError("nonfinite numerical result")
            errors.append((energy_error, force_error))
        energy_error, force_error = np.max(errors, axis=0).tolist()
        for mode in ("cold", "warm", "changed", "cold_total"):
            sample_mode = "cold" if mode == "cold_total" else mode
            left = [
                sample
                for sample in expected["samples"]
                if sample["mode"] == sample_mode
            ]
            right = [
                sample for sample in actual["samples"] if sample["mode"] == sample_mode
            ]
            if not left or not right:
                raise ValueError("missing endpoint mode")
            if any(
                not np.isfinite(s["seconds"]) or s["seconds"] <= 0 for s in left + right
            ):
                raise ValueError("invalid endpoint timing")
            reference_seconds = median(sample["seconds"] for sample in left)
            candidate_seconds = median(sample["seconds"] for sample in right)
            if mode == "cold_total":
                # The APIs assign setup to different phases. Include both
                # phases so moving work into the constructor cannot hide it.
                reference_seconds += expected["construction_seconds"]
                candidate_seconds += actual["construction_seconds"]
            rows.append(
                {
                    "case": expected["case"],
                    "mode": mode,
                    "max_energy_error": energy_error,
                    "max_force_error": force_error,
                    "accuracy_passed": energy_error <= 5e-7 and force_error <= 5e-7,
                    "iterations_match": [s["iterations"] for s in left]
                    == [s["iterations"] for s in right],
                    "reference_seconds": reference_seconds,
                    "candidate_seconds": candidate_seconds,
                    "speedup": reference_seconds / candidate_seconds,
                }
            )
    return {
        "timing_contract": timing_contract,
        "reference_library_sha256": reference["library_sha256"],
        "candidate_library_sha256": candidate["library_sha256"],
        "rows": rows,
    }


def cases(waters: list[int]) -> list[dict]:
    """Include independent tblite fixtures and deterministic larger clusters."""
    fixtures = json.loads((ROOT / "tests/data/gfn2_native_tblite.json").read_text())
    result = fixtures["cases"]
    water = next(case for case in result if case["name"] == "h2o")
    for count in waters:
        positions = []
        for index in range(count):
            offset = np.array([index % 4, (index // 4) % 4, index // 16]) * 6.0
            positions.extend((np.asarray(water["positions"]) + offset).tolist())
        result.append(
            {
                "name": f"water-{count}",
                "symbols": water["symbols"] * count,
                "positions": positions,
            }
        )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--engine", choices=("generativeqc", "xtbloom"))
    parser.add_argument("--reference", type=Path)
    parser.add_argument("--candidate", type=Path)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--waters", type=int, nargs="*", default=[8, 32])
    parser.add_argument("--case", action="append")
    parser.add_argument("--repeat", type=int, default=5)
    parser.add_argument("--output", type=raw_output_path, required=True)
    args = parser.parse_args()
    if args.reference or args.candidate:
        if not (args.reference and args.candidate) or args.engine:
            parser.error("comparison requires reference and candidate, without engine")
        report = compare_reports(
            json.loads(args.reference.read_text()),
            json.loads(args.candidate.read_text()),
        )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n")
        if not all(row["accuracy_passed"] for row in report["rows"]):
            raise SystemExit("numerical comparison failed")
        return
    if not args.engine:
        parser.error("measurement requires engine")
    if args.device == "cuda" and not os.environ.get("SLURM_JOB_ID"):
        parser.error("CUDA measurements require a Slurm GPU allocation")
    if args.repeat < 1 or any(count < 1 for count in args.waters):
        parser.error("repeat and water counts must be positive")

    if args.engine == "generativeqc":
        import generativeqc as package
        from generativeqc import Calculator
    else:
        import xtbloom as package
        from xtbloom import Calculator

    library = Path(os.environ[f"{args.engine.upper()}_LIBRARY"]).resolve()
    report = {
        "timing_contract": TIMING_CONTRACT,
        "engine": args.engine,
        "device": args.device,
        "library": str(library),
        "library_sha256": hashlib.sha256(library.read_bytes()).hexdigest(),
        "python_source": str(Path(package.__file__).resolve()),
        "python_source_revision": source_revision(Path(package.__file__).parent),
        "benchmark_revision": source_revision(ROOT),
        "benchmark_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "settings": {
            "energy_tolerance": 1e-10,
            "charge_tolerance": 1e-8,
            "max_iterations": 300,
            "temperature_kelvin": 300,
            "fresh_scc": True,
            "mixer": "modified_broyden",
            "mixer_history": 8,
            "mixer_damping": 0.4,
            "properties": ["energy", "forces"],
            "units": "bohr, hartree, hartree/bohr",
        },
        "rows": [],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for case in cases(args.waters):
        if args.case and case["name"] not in args.case:
            continue
        start = time.perf_counter()
        if args.engine == "generativeqc":
            calc = Calculator(
                method="gfn2-xtb",
                device=args.device,
                max_iterations=300,
                energy_tolerance=1e-10,
                density_tolerance=1e-8,
            )
        else:
            calc = Calculator(
                "GFN2-xTB",
                numbers=case["symbols"],
                positions=np.asarray(case["positions"]),
                backend=args.device,
                max_scc_iterations=300,
                energy_tolerance=1e-10,
                charge_tolerance=1e-8,
                warm_start=False,
                cpu_threads=1,
            )
        construction_seconds = time.perf_counter() - start
        base_positions = np.asarray(case["positions"])
        # A non-rigid perturbation invalidates geometry caches without changing
        # topology. The same displacement sequence is used by both engines.
        direction = np.sin(np.arange(base_positions.size)).reshape(base_positions.shape)
        row = {
            "case": case["name"],
            "atoms": len(case["symbols"]),
            "construction_seconds": construction_seconds,
            "geometry_sha256": geometry_sha256(case["symbols"], case["positions"]),
            "samples": [],
        }
        for mode, count in (
            ("cold", 1),
            ("warm", args.repeat),
            ("changed", args.repeat),
        ):
            for repeat in range(count):
                positions = base_positions + (
                    0.001 * (repeat + 1) * direction if mode == "changed" else 0
                )
                sample_geometry = geometry_sha256(case["symbols"], positions.tolist())
                start = time.perf_counter()
                if args.engine == "generativeqc":
                    result = calc.singlepoint(
                        list(zip(case["symbols"], positions, strict=True)),
                        properties=("energy", "forces"),
                    )
                    converged, iterations = result.converged, result.iterations
                else:
                    calc.update(positions=positions)
                    result = calc.singlepoint()
                    converged, iterations = result.scc_converged, result.scc_iterations
                seconds = time.perf_counter() - start
                if not converged:
                    raise RuntimeError(f"{args.engine} {case['name']} did not converge")
                if mode == "cold":
                    # Query identity after the timed first call. xTBloom loads
                    # lazily, so querying beforehand would warm the library
                    # outside either its constructor or calculation timer.
                    if args.engine == "generativeqc":
                        actual_library = Path(calc._library._name).resolve()
                    else:
                        from xtbloom.library import load_library

                        actual_library = Path(load_library()._name).resolve()
                    if actual_library != library:
                        raise RuntimeError(
                            f"loaded {actual_library}, expected {library}"
                        )
                row["samples"].append(
                    {
                        "mode": mode,
                        "repeat": repeat,
                        "geometry_sha256": sample_geometry,
                        "seconds": seconds,
                        "iterations": iterations,
                        "energy": float(result.energy),
                        "forces": np.asarray(result.forces).tolist(),
                    }
                )
                print(
                    f"{args.engine} {case['name']} {mode} {seconds * 1000:.3f} ms {iterations} iterations",
                    flush=True,
                )
        # Assignment to `calc` on the next iteration would otherwise destroy
        # this calculator inside the next case's constructor timer. Release its
        # last result too, and expose cleanup cost separately after all samples.
        start = time.perf_counter()
        del result, calc
        row["cleanup_seconds"] = time.perf_counter() - start
        report["rows"].append(row)
        args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
