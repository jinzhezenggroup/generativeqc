"""Acquire fresh-process full-DF CUDA CBS endpoint measurements."""

import argparse
import ctypes
import faulthandler
import importlib.metadata
import json
import math
import os
import platform
import subprocess
import tempfile
import time
import traceback
from dataclasses import asdict
from pathlib import Path

from protocol import (
    MAX_SCALAR_WORK,
    NUMERIC_BYTES,
    RECIPES,
    REVISION,
    ROOT,
    TIME_LIMIT_SECONDS,
    WORK_ENV,
    append,
    basis,
    endpoint,
    load,
    sha,
)


def singlepoint(system: dict, recipe: str) -> dict:
    from generativeqc import CBSBasis, CCSDTCBSCalculator
    from pyscf.gto import charge

    from tools.generativeqc_validation.correlation_df_oracle import basis_snapshot

    low, high = RECIPES[recipe]
    elements = [charge(symbol) for symbol, _ in system["bohr"]]
    names = {cardinal: basis(cardinal) for cardinal in sorted({2, low, high})}
    fit = {cardinal: f"{name}-jkfit" for cardinal, name in names.items()}
    meta = {"orbital_bases": names, "auxiliary_bases": fit}
    start = None
    try:
        orbitals = {n: basis_snapshot(name, elements) for n, name in names.items()}
        aux = {n: basis_snapshot(name, elements) for n, name in fit.items()}
        meta["orbital_input_ids"] = {
            n: record.identity for n, record in orbitals.items()
        }
        meta["auxiliary_input_ids"] = {n: record.identity for n, record in aux.items()}
        start = time.perf_counter()
        owner = CCSDTCBSCalculator(
            (CBSBasis(orbitals[low], low), CBSBasis(orbitals[high], high)),
            correction_basis=orbitals[2],
            auxiliary_bases=(aux[low], aux[high]),
            correction_auxiliary_basis=aux[2],
            hf_reference="df",
            device="cuda",
            policy="fp64",
            preconvergence="fp64",
            calculator_options={"correlation_memory_budget_bytes": NUMERIC_BYTES},
        )
        result = owner.singlepoint(
            system["bohr"],
            charge=system["charge"],
            multiplicity=system["multiplicity"],
            properties=("energy",),
        )
        elapsed = time.perf_counter() - start
        if (
            not result.converged
            or result.executed_backend != "cuda"
            or result.forces is not None
            or not math.isfinite(result.energy)
            or result.convergence["arithmetic"] != "fp64"
            or result.basis_metadata["hf_reference"] != "df"
            or any(
                not c.result.converged or c.result.executed_backend != "cuda"
                for c in result.components
            )
        ):
            raise ValueError("invalid full-DF FP64 CUDA energy result")
        return {
            **meta,
            "status": "ok",
            "wall_seconds": elapsed,
            "result": asdict(result),
        }
    except Exception as exc:  # noqa: BLE001
        return {
            **meta,
            "status": "failed",
            "error": str(exc),
            "error_type": type(exc).__name__,
            "traceback": traceback.format_exc(),
            "wall_seconds": time.perf_counter() - start if start is not None else None,
        }


def provenance() -> dict:
    import generativeqc
    from cupy.cuda import runtime

    from tools.generate_build_identity import _inventory, _source_identity

    source = Path(generativeqc.__file__).resolve().parents[2]
    identity = _source_identity(
        source, _inventory(source, source / "cmake/GenerativeQCSourceIdentity.json")
    )
    library = Path(os.environ["GENERATIVEQC_LIBRARY"]).resolve()
    native = ctypes.CDLL(str(library))
    native.generativeqc_get_source_identity.restype = ctypes.c_char_p
    loaded_identity = native.generativeqc_get_source_identity().decode()
    receipt_path = library.parent.parent / "build-receipts/identity.json"
    receipt = load(receipt_path)
    digest = sha(library)
    if (
        receipt["source_revision"] != REVISION
        or loaded_identity != identity
        or receipt["native_source_identity"] != identity
        or receipt["library_sha256"] != digest
    ):
        raise RuntimeError(
            "source, native library and production build receipt disagree"
        )
    if runtime.getDeviceCount() != 1:
        raise RuntimeError("each worker requires exactly one visible CUDA device")
    nvidia_smi = (
        subprocess.check_output(
            [
                "nvidia-smi",
                "-i",
                runtime.deviceGetPCIBusId(0),
                "--query-gpu=name,uuid,memory.total,compute_cap,driver_version,power.limit",
                "--format=csv,noheader",
            ],
            text=True,
        )
        .strip()
        .split(", ")
    )
    keys = ("model", "uuid", "memory", "compute_capability", "driver", "power_limit")
    card = dict(zip(keys, nvidia_smi, strict=True))
    if "H100" not in card["model"] or card["compute_capability"] != "9.0":
        raise RuntimeError(f"expected production-cohort H100, observed {card}")
    if os.environ[WORK_ENV] != str(MAX_SCALAR_WORK):
        raise RuntimeError("incorrect correlation work policy")
    return {
        "source_revision": REVISION,
        "native_source_identity": identity,
        "library_sha256": digest,
        "build_receipt_sha256": sha(receipt_path),
        "gpu": card,
        "cuda_runtime": runtime.runtimeGetVersion(),
        "hostname": platform.node(),
        "system": platform.platform(),
        "cpu_affinity": sorted(os.sched_getaffinity(0)),
        "python": platform.python_version(),
        "packages": {
            p: importlib.metadata.version(p)
            for p in ("pyscf", "numpy", "scipy", "cupy-cuda12x")
        },
        "inputs_sha256": sha(ROOT / "inputs.json"),
        "driver_sha256": sha(Path(__file__)),
        "protocol_sha256": sha(ROOT / "protocol.py"),
        "correlation_memory_budget_bytes": NUMERIC_BYTES,
        "max_scalar_work": MAX_SCALAR_WORK,
        "endpoint_limit_seconds": TIME_LIMIT_SECONDS,
        "environment": {
            key: os.environ.get(key)
            for key in (
                "GQC_QZ_JOB_NAME",
                "GQC_QZ_PROJECT",
                "CUDA_VISIBLE_DEVICES",
                "CUBLAS_WORKSPACE_CONFIG",
                "OMP_NUM_THREADS",
                "OPENBLAS_NUM_THREADS",
            )
        },
    }


def ordered_systems(data: dict, dataset: str) -> list[str]:
    # Process every dependency for a small complete barrier before larger barriers.
    def size(key: str) -> int:
        return sum(
            1 if symbol == "H" else 4 for symbol, _ in data["systems"][key]["bohr"]
        )

    barriers = sorted(
        (row for row in data["barriers"] if row["set"] == dataset),
        key=lambda row: (max(map(size, row["stoichiometry"])), row["index"]),
    )
    return list(
        dict.fromkeys(
            key for row in barriers for key in sorted(row["stoichiometry"], key=size)
        )
    )


def campaign(data: dict, dataset: str, output: Path, selection: Path | None) -> None:
    ordered = ordered_systems(data, dataset)
    if selection is not None:
        plan = load(selection)
        if plan["dataset"] != dataset or plan["inputs_sha256"] != sha(
            ROOT / "inputs.json"
        ):
            raise ValueError("GPU selection dataset or input digest mismatch")
        selected = plan["systems"]
        if len(selected) != len(set(selected)) or not set(selected) <= set(ordered):
            raise ValueError(
                "selection must contain unique system keys from the dataset"
            )
        ordered = [key for key in ordered if key in selected]
    if not ordered:
        raise ValueError("empty GPU endpoint selection")
    output.mkdir()
    receipt = {**provenance(), "dataset": dataset, "systems": ordered}
    (output / "provenance.json").write_text(json.dumps(receipt, indent=2) + "\n")
    with (
        (output / "records.jsonl").open("x") as journal,
        (output / "process.txt").open("x") as log,
        tempfile.TemporaryDirectory(dir=output) as temporary,
    ):
        point_path = Path(temporary) / "point.json"
        for key in ordered:
            for recipe in RECIPES:
                task = {"system": key, "recipe": recipe}
                append(journal, {**task, "status": "started", "utc_unix": time.time()})
                print(key, recipe, "started", flush=True)
                log.write(f"\n--- {key} {recipe} ---\n")
                log.flush()
                row = endpoint(Path(__file__), [key, recipe], point_path, log)
                append(journal, {**task, **row, "finished_utc_unix": time.time()})
                print(
                    key, recipe, row["status"], row["process_wall_seconds"], flush=True
                )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    run = subparsers.add_parser("campaign")
    run.add_argument("dataset", choices=("CRBH20",))
    run.add_argument("output", type=Path)
    run.add_argument("--selection", type=Path)
    point = subparsers.add_parser("point")
    point.add_argument("system")
    point.add_argument("recipe", choices=tuple(RECIPES))
    point.add_argument("output", type=Path)
    options = parser.parse_args()
    data = load(ROOT / "inputs.json")
    if options.command == "campaign":
        campaign(data, options.dataset, options.output, options.selection)
    else:
        if os.environ[WORK_ENV] != str(MAX_SCALAR_WORK):
            raise RuntimeError("incorrect correlation work policy")
        faulthandler.dump_traceback_later(600, repeat=True)
        try:
            row = singlepoint(data["systems"][options.system], options.recipe)
        finally:
            faulthandler.cancel_dump_traceback_later()
        with options.output.open("x") as stream:
            append(stream, row)


if __name__ == "__main__":
    main()
