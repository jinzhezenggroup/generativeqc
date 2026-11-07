"""Qualify guarded HF defaults against explicit full-density complete endpoints.

Run only in a finite Slurm allocation. Retained independent E/F references never
provide initial densities. Default and off arms own separate frozen cold seeds.
"""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import lzma
import os
import platform
import statistics
import subprocess
import time
import traceback
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from generativeqc import Calculator
from generativeqc.profiles import probe_device

if TYPE_CHECKING:
    from generativeqc._api_types import Result


def digest_file(path: Path) -> str:
    """Hash even large AOT libraries without allocating a binary-sized buffer."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def emit(path: Path, record: dict) -> None:
    """Flush every outcome so failures and scheduler termination remain visible."""
    text = json.dumps(record, allow_nan=False)
    with path.open("a") as stream:
        stream.write(text + "\n")
    print(text, flush=True)


def calculator(incremental: bool, strict: bool) -> Calculator:
    """Exercise the real unset/default path, not an explicit opt-in surrogate."""
    if incremental:
        os.environ.pop("GENERATIVEQC_INCREMENTAL_DIRECT_JK", None)
    else:
        os.environ["GENERATIVEQC_INCREMENTAL_DIRECT_JK"] = "0"
    os.environ["GENERATIVEQC_INCREMENTAL_DIRECT_JK_REBUILD_INTERVAL"] = "8"
    controls = {"energy_tolerance": 1e-12, "density_tolerance": 1e-9} if strict else {}
    return Calculator(
        method="rhf",
        basis="def2-svp",
        basis_representation="spherical",
        device="cuda",
        precision="fp64",
        initial_guess="auto",
        max_iterations=100,
        screening_tolerance=1e-12,
        **controls,
    )


def observe(result: Result, reference: dict, strict: bool, atom_count: int) -> dict:
    """Qualify returned E/F against independently retained tight reference arrays."""
    forces = np.asarray(result.forces)
    differences = forces - np.asarray(reference["forces"])
    energy_error = abs(result.energy - reference["energy"])
    force_maximum = float(np.max(np.abs(differences)))
    force_rms = float(np.sqrt(np.mean(differences * differences)))
    energy_gate = 1e-8 if strict else atom_count * 2.5e-6
    maximum_gate = 1e-7 if strict else 2.5e-5
    rms_gate = 1e-7 if strict else 7.5e-6
    return {
        "status": result.status,
        "converged": result.converged,
        "backend": result.executed_backend,
        "iterations": result.iterations,
        "energy_change": result.energy_change,
        "density_rms": result.density_rms,
        "physical_residual_rms": result.physical_residual_rms,
        "energy": result.energy,
        "forces": forces.tolist(),
        "energy_error": energy_error,
        "force_maximum_error": force_maximum,
        "force_rms_error": force_rms,
        "warm_start_used": result.warm_start_used,
        "warm_start_fallback": result.warm_start_fallback,
        "fock_builds": result.fock_builds,
        "precision": result.precision,
        "incremental_direct_jk": result.incremental_direct_jk,
        "initial_guess": result.initial_guess,
        "gates": {
            "energy": energy_gate,
            "force_maximum": maximum_gate,
            "force_rms": rms_gate,
        },
        "passed": bool(
            result.converged
            and result.executed_backend == "cuda"
            and np.isfinite(forces).all()
            and energy_error <= energy_gate
            and force_maximum <= maximum_gate
            and force_rms <= rms_gate
        ),
    }


def run_arm(
    arguments: argparse.Namespace,
    size: str,
    fixture: dict,
    incremental: bool,
    repeat: int,
    records: list[dict],
) -> None:
    """Time preparation plus cold E/F, frozen warm, then moved-geometry E/F.

    Each independent repetition owns its cold density. Warm and moved calls
    both restart from that same frozen snapshot rather than a rolling seed.
    The public host E/F return completes GPU execution before the timer stops.
    Cleanup is timed separately, never hidden inside the other arm's cold call.
    """
    common = {
        "aos": int(size),
        "atoms": len(fixture["atoms"]),
        "incremental": incremental,
        "repeat": repeat,
        "controls": "strict" if arguments.strict else "standard",
        "selector": "unset" if incremental else "0",
    }
    owner = None
    engine = None
    started = time.perf_counter()
    try:
        engine = calculator(incremental, arguments.strict)
        assert (
            Path(engine._library._name).resolve()
            == Path(os.environ["GENERATIVEQC_LIBRARY"]).resolve()
        )
        owner = engine.prepare_batch(
            [fixture["atoms"]],
            charges=[fixture["charge"]],
            multiplicities=[fixture["multiplicity"]],
            warm_start=True,
        )
        prepared = time.perf_counter()
        result = owner.execute(strict=True, properties=("energy", "forces")).items[0]
        finished = time.perf_counter()
        assert result.incremental_direct_jk["requested"] == incremental
        record = {
            **common,
            "phase": "cold",
            "seconds": finished - started,
            "setup_seconds": prepared - started,
            "execute_seconds": finished - prepared,
            "executed_controls": {
                "energy": engine._energy_tolerance,
                "density": engine._density_tolerance,
                "screening": engine._screening_tolerance,
            },
            **observe(
                result, fixture["reference"][0], arguments.strict, len(fixture["atoms"])
            ),
        }
        records.append(record)
        emit(arguments.journal, record)
        owner.set_warm_start_updates(False)
        for phase, geometry, reference in (
            ("warm", fixture["atoms"], fixture["reference"][0]),
            ("moved", fixture["moved"], fixture["reference"][1]),
        ):
            coordinates = np.asarray([position for _, position in geometry])
            started = time.perf_counter()
            result = owner.execute(
                [coordinates], strict=True, properties=("energy", "forces")
            ).items[0]
            elapsed = time.perf_counter() - started
            assert result.incremental_direct_jk["requested"] == incremental
            record = {
                **common,
                "phase": phase,
                "seconds": elapsed,
                **observe(result, reference, arguments.strict, len(geometry)),
            }
            records.append(record)
            emit(arguments.journal, record)
    except Exception as error:  # noqa: BLE001 -- retain every failed endpoint before qualification.
        record = {
            **common,
            "phase": "failed",
            "seconds": time.perf_counter() - started,
            "passed": False,
            "error": repr(error),
            "traceback": traceback.format_exc(),
        }
        records.append(record)
        emit(arguments.journal, record)
    finally:
        started = time.perf_counter()
        if owner is not None:
            owner.close()
        if engine is not None:
            engine.clear_cache()
        emit(
            arguments.journal,
            {**common, "phase": "cleanup", "seconds": time.perf_counter() - started},
        )


def summarize(records: list[dict], sizes: list[str], repeats: int) -> list[dict]:
    """Rank only complete populations passing every independent numerical gate."""
    summary = []
    for size in sizes:
        for phase in ("cold", "warm", "moved"):
            arms = {}
            for incremental in (False, True):
                samples = [
                    record
                    for record in records
                    if record["aos"] == int(size)
                    and record["phase"] == phase
                    and record["incremental"] == incremental
                ]
                times = [record["seconds"] for record in samples]
                arms[str(incremental)] = {
                    "sample_count": len(samples),
                    "qualified": len(samples) == repeats
                    and all(sample["passed"] for sample in samples),
                    "median_seconds": statistics.median(times) if times else None,
                    "minimum_seconds": min(times) if times else None,
                    "maximum_seconds": max(times) if times else None,
                    "iterations": [sample["iterations"] for sample in samples],
                    "fock_builds": [sample["fock_builds"] for sample in samples],
                }
            off, on = arms["False"], arms["True"]
            summary.append(
                {
                    "aos": int(size),
                    "phase": phase,
                    "off": off,
                    "on": on,
                    "speedup": off["median_seconds"] / on["median_seconds"]
                    if off["qualified"] and on["qualified"]
                    else None,
                }
            )
    return summary


def main() -> None:
    """Keep device/source/reference identity, all samples, and honest failures."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--source-revision", required=True)
    parser.add_argument("--reference-scope", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sizes", nargs="+", default=["96", "192", "384"])
    parser.add_argument("--repeats", type=int, default=4)
    parser.add_argument("--strict", action="store_true")
    arguments = parser.parse_args()
    assert os.environ.get("SLURM_JOB_ID") and "CUDA_VISIBLE_DEVICES" in os.environ
    arguments.output.mkdir(parents=True, exist_ok=True)
    arguments.journal = arguments.output / "samples.jsonl"
    assert not arguments.journal.exists()
    packet = json.loads(lzma.decompress(arguments.reference.read_bytes()))
    fixtures = {size: packet["references"][size]["data"] for size in arguments.sizes}
    library_path = Path(os.environ["GENERATIVEQC_LIBRARY"]).resolve()
    library = ctypes.CDLL(str(library_path))
    library.generativeqc_get_source_identity.restype = ctypes.c_char_p
    metadata = {
        "schema": "generativeqc.hf-incremental-default.v1",
        "host": platform.node(),
        "job": os.environ["SLURM_JOB_ID"],
        "visibility": os.environ["CUDA_VISIBLE_DEVICES"],
        "source_identity": library.generativeqc_get_source_identity().decode(),
        "source_revision": arguments.source_revision,
        "library_sha256": digest_file(library_path),
        "library": str(library_path),
        "harness_sha256": digest_file(Path(__file__)),
        "reference_sha256": digest_file(arguments.reference),
        "reference_scope": arguments.reference_scope,
        "fixtures": fixtures,
        "repeats": arguments.repeats,
        "profile": probe_device(library, 0),
        "gpu": subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=uuid,name,driver_version",
                "--format=csv,noheader",
            ],
            text=True,
            capture_output=True,
            check=True,
        ).stdout,
        "timing_scope": "Cold: Calculator construction + prepare + complete host E/F. "
        "Process imports and common tiny-water bootstrap excluded; cleanup separate. "
        "Warm and moved: each independently replays that run's frozen cold density.",
        "order": "AB/BA/BA/AB, repeated as needed",
        "strict": arguments.strict,
        "limitation": "Restricted all-electron spherical def2-SVP fixtures; not all molecules or providers",
    }
    emit(arguments.journal, {"phase": "metadata", **metadata})
    bootstrap_atoms = next(iter(fixtures.values()))["atoms"][:3]
    for incremental in (False, True):
        engine = calculator(incremental, arguments.strict)
        with engine.prepare_batch([bootstrap_atoms], warm_start=True) as owner:
            result = owner.execute(strict=True, properties=("energy", "forces")).items[
                0
            ]
            emit(
                arguments.journal,
                {
                    "phase": "bootstrap",
                    "incremental": incremental,
                    "converged": result.converged,
                    "iterations": result.iterations,
                },
            )
        engine.clear_cache()
    records = []
    for size, fixture in fixtures.items():
        for repeat in range(arguments.repeats):
            order = (False, True) if repeat % 4 in (0, 3) else (True, False)
            for incremental in order:
                run_arm(arguments, size, fixture, incremental, repeat, records)
    report = {
        **metadata,
        "summary": summarize(records, arguments.sizes, arguments.repeats),
        "records": records,
        "all_passed": all(record["passed"] for record in records),
    }
    (arguments.output / "report.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n"
    )
    for row in report["summary"]:
        print("SUMMARY", json.dumps(row), flush=True)
    if not report["all_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
