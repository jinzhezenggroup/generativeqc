"""Run the pinned GenerativeQC energy endpoints with one CUDA process per point."""

import argparse
import ctypes
import faulthandler
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import subprocess
import sys
import tempfile
import time
import traceback
from dataclasses import asdict
from pathlib import Path
from typing import Any, TextIO

ROOT = Path(__file__).resolve().parent
RECIPES = {"DT_D": (2, 3), "TQ_D": (3, 4)}
ANGSTROM_PER_BOHR = 0.529177210903
BASIS_CARDINAL = {2: "d", 3: "t", 4: "q"}
# Approximate computational order only. Every endpoint remains enrolled.
ATOM_WEIGHT = {"H": 1, "C": 4, "N": 4, "O": 4, "F": 4, "P": 6, "S": 6, "Cl": 6}
SOURCE_REVISION = "22f72d0692a760c2892cd7791022584d9ae1c3e5"


def basis_name(system: dict, cardinal: int) -> str:
    prefix = "aug-" if system["family"] == "aug" else ""
    return f"{prefix}cc-pv{BASIS_CARDINAL[cardinal]}z"


def bohr_geometry(system: dict) -> list[tuple[str, tuple[float, ...]]]:
    return [
        (symbol, tuple(position / ANGSTROM_PER_BOHR for position in xyz))
        for symbol, xyz in system["angstrom"]
    ]


def sha(path: str | Path) -> str:
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def record_json(value: Any) -> Any:
    """Keep diagnostic arrays readable and non-finite diagnostic scalars explicit."""
    if isinstance(value, float) and not math.isfinite(value):
        return str(value)
    if isinstance(value, dict):
        return {key: record_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [record_json(item) for item in value]
    if hasattr(value, "tolist"):
        return record_json(value.tolist())
    return value


def native_endpoint(system: dict, recipe: str) -> dict:
    from generativeqc import CBSBasis, CCSDTCBSCalculator
    from pyscf.gto import charge

    from tools.generativeqc_validation.correlation_df_oracle import basis_snapshot

    elements = [charge(symbol) for symbol, _ in system["angstrom"]]
    low, high = RECIPES[recipe]
    names = {n: basis_name(system, n) for n in (2, low, high)}
    orbital = {n: basis_snapshot(name, elements) for n, name in names.items()}
    aux = {n: basis_snapshot(name + "-jkfit", elements) for n, name in names.items()}
    start = time.perf_counter()
    try:
        owner = CCSDTCBSCalculator(
            (CBSBasis(orbital[low], low), CBSBasis(orbital[high], high)),
            correction_basis=orbital[2],
            auxiliary_bases=(aux[low], aux[high]),
            correction_auxiliary_basis=aux[2],
            hf_reference="df",
            device="cuda",
            policy="fp64",
            preconvergence="fp64",
            calculator_options={"correlation_memory_budget_bytes": 32 << 30},
        )
        result = owner.singlepoint(
            bohr_geometry(system),
            charge=system["charge"],
            multiplicity=system["multiplicity"],
            properties=("energy",),
        )
        elapsed = time.perf_counter() - start
    except Exception as exc:  # noqa: BLE001
        # Preserve every scientific endpoint failure and continue the cohort.
        return {
            "status": "failed",
            "wall_seconds": time.perf_counter() - start,
            "error": str(exc),
            "error_type": type(exc).__name__,
            "traceback": traceback.format_exc(),
            "orbital_bases": names,
        }
    if (
        not result.converged
        or result.executed_backend != "cuda"
        or result.forces is not None
        or not math.isfinite(result.energy)
        or any(not c.result.converged for c in result.components)
    ):
        raise ValueError("invalid CUDA energy-only result")
    return {
        "status": "ok",
        "wall_seconds": elapsed,
        "orbital_bases": names,
        "auxiliary_bases": {n: name + "-jkfit" for n, name in names.items()},
        "result": asdict(result),
    }


def provenance() -> dict:
    import generativeqc
    from cupy.cuda import runtime

    from tools.generate_build_identity import _inventory, _source_identity

    source = Path(generativeqc.__file__).resolve().parents[2]
    identity = _source_identity(
        source, _inventory(source, source / "cmake/GenerativeQCSourceIdentity.json")
    )
    library = Path(os.environ["GENERATIVEQC_LIBRARY"])
    native = ctypes.CDLL(str(library))
    native.generativeqc_get_source_identity.restype = ctypes.c_char_p
    actual = native.generativeqc_get_source_identity().decode()
    binary = sha(library)
    build = json.loads((ROOT / "build-receipts/identity.json").read_text())
    if (
        identity != build["native_source_identity"]
        or actual != identity
        or binary != build["library_sha256"]
        or build["source_revision"] != SOURCE_REVISION
    ):
        raise ValueError(
            f"source/library signature mismatch: {identity} {actual} {binary}"
        )
    device = subprocess.check_output(
        [
            "nvidia-smi",
            "-i",
            runtime.deviceGetPCIBusId(0),
            "--query-gpu=name,uuid,memory.total,compute_cap,driver_version,power.limit",
            "--format=csv,noheader",
        ],
        text=True,
    ).strip()
    assert runtime.getDeviceCount() == 1 and "H100" in device, device
    return {
        "source_revision": SOURCE_REVISION,
        "native_source_identity": identity,
        "reported_library_source_identity": actual,
        "library_sha256": binary,
        "library_build_revision": build["source_revision"],
        "driver_sha256": sha(__file__),
        "inputs_sha256": sha(ROOT / "inputs.json"),
        "gpu": device,
        "cuda_runtime": runtime.runtimeGetVersion(),
        "hostname": platform.node(),
        "system": platform.platform(),
        "affinity": sorted(os.sched_getaffinity(0)),
        "python": platform.python_version(),
        "packages": {
            package: importlib.metadata.version(package)
            for package in ("pyscf", "numpy", "scipy", "cupy-cuda12x")
        },
        "environment": {
            key: os.environ.get(key)
            for key in (
                "CUDA_VISIBLE_DEVICES",
                "CUBLAS_WORKSPACE_CONFIG",
                "OMP_NUM_THREADS",
                "OPENBLAS_NUM_THREADS",
                "GQC_QZ_JOB_NAME",
                "GQC_QZ_PROJECT",
            )
        },
    }


def append(file: TextIO, row: dict) -> None:
    file.write(json.dumps(record_json(row), allow_nan=False) + "\n")
    file.flush()
    os.fsync(file.fileno())


def campaign(dataset: str, systems: dict, output: Path) -> None:
    output.mkdir()
    (output / "provenance.json").write_text(json.dumps(provenance(), indent=2) + "\n")
    ordered = sorted(
        (key for key in systems if key.startswith(dataset + "/")),
        key=lambda key: (
            sum(ATOM_WEIGHT[symbol] for symbol, _ in systems[key]["angstrom"]),
            key,
        ),
    )
    with (
        (output / "records.jsonl").open("x") as rows,
        (output / "process.txt").open("x") as log,
        tempfile.TemporaryDirectory(dir=output) as temporary,
    ):
        result_path = Path(temporary) / "result.json"
        for recipe in RECIPES:
            for key in ordered:
                request = {"system": key, "recipe": recipe}
                append(rows, {**request, "status": "started", "utc_unix": time.time()})
                print(key, recipe, flush=True)
                log.write(f"\n--- {key} {recipe} ---\n")
                log.flush()
                t0 = time.perf_counter()
                try:
                    proc = subprocess.run(
                        [
                            sys.executable,
                            __file__,
                            "point",
                            key,
                            recipe,
                            str(result_path),
                        ],
                        stdout=log,
                        stderr=subprocess.STDOUT,
                        timeout=420,
                        check=False,
                    )
                    if proc.returncode == 0:
                        result = json.loads(result_path.read_text())
                        result_path.unlink()
                    else:
                        result = {
                            "status": "process_failed",
                            "returncode": proc.returncode,
                        }
                except subprocess.TimeoutExpired:
                    result = {"status": "timeout", "limit_seconds": 420}
                append(
                    rows,
                    {
                        **request,
                        **result,
                        "process_wall_seconds": time.perf_counter() - t0,
                    },
                )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    run = subparsers.add_parser("campaign")
    run.add_argument("dataset", choices=("HTBH38", "NHTBH38"))
    run.add_argument("output", type=Path)
    point = subparsers.add_parser("point")
    point.add_argument("system")
    point.add_argument("recipe", choices=RECIPES)
    point.add_argument("output", type=Path)
    options = parser.parse_args()
    systems = json.loads((ROOT / "inputs.json").read_text())["systems"]
    if options.command == "point":
        faulthandler.dump_traceback_later(90, repeat=True)
        try:
            value = native_endpoint(systems[options.system], options.recipe)
        finally:
            faulthandler.cancel_dump_traceback_later()
        options.output.write_text(json.dumps(record_json(value), allow_nan=False))
    else:
        campaign(options.dataset, systems, options.output)
