"""Offline, receipt-bound accuracy gates for the static README populations.

The dense direct target includes DF approximation error. Missing or unstable
references leave a population unverified; they never turn into a passing gate.
Qualification covers only the measured geometries, not displaced trajectories.
"""

from __future__ import annotations

import hashlib
import json
import re

import numpy as np

from benchmarks.readme_defaults import COMMON_GATES, MEASUREMENT_ARMS, SCHEMA
from benchmarks.readme_defaults_accuracy import errors
from benchmarks.readme_defaults_report import checked_sample, reduce_record

ORACLE_SCHEMA = "generativeqc.readme-defaults-oracle.v1"
REUSE_SCHEMA = "generativeqc.readme-oracle-reuse.v1"
STABILITY_FRACTION = 0.25


def identity(record: dict) -> str:
    """Bind results to the full receipt independently of its JSON whitespace."""
    data = json.dumps(record, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(data.encode()).hexdigest()


def _geometry(record: dict) -> tuple:
    """Validate exact Bohr geometry identity before any numerical subtraction."""
    geometry = record.get("geometry_bohr")
    if not isinstance(geometry, (tuple, list)) or len(geometry) != record["atoms"]:
        raise ValueError("missing or malformed geometry")
    result = []
    for symbol, xyz in geometry:
        if not isinstance(symbol, str) or not symbol or len(xyz) != 3:
            raise ValueError("malformed geometry")
        if not all(type(v) in (int, float) and np.isfinite(v) for v in xyz):
            raise ValueError("nonfinite geometry")
        result.append((symbol, tuple(xyz)))
    return tuple(result)


def _comparison(first: dict, last: dict, atoms: int, *, fraction: float = 1) -> dict:
    """Check shapes before subtraction: NumPy broadcasting is not a force test."""
    for sample in (first, last):
        checked_sample(sample, {"atoms": atoms, "arm": "gpu4pyscf"})
    result = errors(first["energy"], first["forces"], last, atoms)
    result["passed"] = all(
        result[key] <= fraction * threshold for key, threshold in COMMON_GATES.items()
    )
    return result


def _oracle(record: dict) -> None:
    """Validate declared oracle controls without trusting stored pass booleans."""
    _geometry(record)
    if not re.fullmatch(r"[0-9a-f]{40}", record.get("source_revision", "")):
        raise ValueError("missing oracle source revision")
    if not re.fullmatch(r"[0-9a-f]{64}", record.get("driver_sha256", "")):
        raise ValueError("missing oracle driver identity")
    if not record.get("slurm_job_id"):
        raise ValueError("missing oracle allocation")
    if record.get("common_observable_gates") != COMMON_GATES:
        raise ValueError("oracle gates differ from the declared campaign")
    if record.get("grid_stability_gate_fraction") != STABILITY_FRACTION:
        raise ValueError("oracle grid stability margin differs")
    if record["status"] != "complete":
        return
    levels = [sample.get("level") for sample in record["samples"]]
    # A failed grid-stability check requires denser independent references,
    # not a larger error allowance. Preserve the original 4/5 receipts while
    # admitting an explicitly selected, complete pair of finer grid levels.
    valid_levels = (
        levels == [None]
        if record["method"] == "hf"
        else all(type(level) is int and 4 <= level <= 9 for level in levels)
        and (
            len(levels) == 1
            if record["backend"] == "cpu"
            else len(levels) == 2 and levels[1] == levels[0] + 1
        )
    )
    if not valid_levels:
        raise ValueError("incomplete dense-grid reference population")
    nonlocal_levels = []
    for sample in record["samples"]:
        checked_sample(sample, {"atoms": record["atoms"], "arm": "gpu4pyscf"})
        controls = sample.get("controls", {})
        if controls != {
            "energy": 1e-11,
            "orbital_gradient": 1e-7,
            "screening": 1e-14,
            "full_density_fock": True,
            "force_grid_response": None if record["method"] == "hf" else True,
        }:
            raise ValueError("unqualified oracle convergence controls")
        if (
            record["method"] != "hf"
            and (sample.get("xc_grid") or {}).get("level") != sample["level"]
        ):
            raise ValueError("oracle XC grid does not match its declared level")
        if record["method"] == "wb97m-v":
            nonlocal_level = (sample.get("nonlocal_grid") or {}).get("level")
            if type(nonlocal_level) is not int or not 2 <= nonlocal_level <= 9:
                raise ValueError("oracle VV10 grid is missing or too coarse")
            nonlocal_levels.append(nonlocal_level)
    if len(nonlocal_levels) == 2 and nonlocal_levels[1] != nonlocal_levels[0] + 1:
        raise ValueError("incomplete VV10 grid stability pair")


def _reference_contract(record: dict, oracle: dict) -> dict:
    """Bind an audited legacy oracle to its original driver and scientific inputs.

    The original oracle schema did not export basis/spin metadata. Cross-revision
    reuse therefore requires an explicit receipt-bound contract reviewed against
    the producer's source; matching geometry or package versions alone is not
    sufficient. This contract never replaces observable or stability checks.
    """
    keys = ("method", "orbital_basis", "basis_representation", "charge", "spin")
    if any(key not in record for key in keys):
        raise ValueError("reference reuse requires complete scientific inputs")
    return {
        "source_revision": oracle["source_revision"],
        "driver_sha256": oracle["driver_sha256"],
        "scientific_inputs": {key: record[key] for key in keys},
    }


def qualify(
    records: list[dict], oracles: list[dict], *, reference_reuse: dict | None = None
) -> list[dict]:
    """Gate every complete phase, retaining all errors and missing references.

    A method needs a CPU/GPU dense small-system spot check and a stable dense
    reference at the population's own geometry. All warm repeats must pass.
    Production records and their measured samples are never rewritten. Different
    producer revisions require an explicit audited contract for each exact oracle
    receipt. Native timing populations still require a single measured revision.
    """
    reuse = {}
    if reference_reuse is not None:
        if (
            set(reference_reuse) != {"schema", "entries"}
            or reference_reuse["schema"] != REUSE_SCHEMA
            or not isinstance(reference_reuse["entries"], dict)
        ):
            raise ValueError("malformed reference reuse manifest")
        reuse = reference_reuse["entries"]
    indexed = {}
    for oracle in oracles:
        if oracle.get("schema") != ORACLE_SCHEMA or oracle.get("backend") not in (
            "cpu",
            "cuda",
        ):
            raise ValueError("unknown oracle schema or backend")
        _oracle(oracle)
        key = (oracle["method"], oracle["atoms"], oracle["backend"])
        if key in indexed:
            raise ValueError("duplicate oracle population")
        indexed[key] = oracle
    results = []
    seen = set()
    for record in records:
        if record.get("schema") != SCHEMA or record.get("arm") not in MEASUREMENT_ARMS:
            raise ValueError("qualification requires current three-profile receipts")
        reduced = reduce_record(record)
        key = tuple(record[k] for k in ("mode", "method", "arm", "atoms"))
        if key in seen:
            raise ValueError("duplicate measurement population")
        seen.add(key)
        geometry = _geometry(record)
        if record.get("protocol", {}).get("common_observable_gates") != COMMON_GATES:
            raise ValueError("measurement gates differ from the declared campaign")
        if not re.fullmatch(r"[0-9a-f]{40}", record.get("source_revision", "")):
            raise ValueError("missing measurement source revision")
        if not re.fullmatch(r"[0-9a-f]{64}", record.get("runner_sha256", "")):
            raise ValueError("missing measurement driver identity")
        method, atoms = record["method"], record["atoms"]
        target = indexed.get((method, atoms, "cuda"))
        cpu = indexed.get((method, 3, "cpu"))
        gpu = indexed.get((method, 3, "cuda"))
        result = {
            **{k: record[k] for k in ("mode", "method", "arm", "atoms")},
            "receipt_sha256": identity(record),
            "status": "unverified",
            "reasons": [],
            "reference_sha256": [identity(r) for r in (target, cpu, gpu) if r],
            "cold": {"status": "unverified"},
            "warm": {"status": "unverified"},
        }
        results.append(result)
        if target and _geometry(target) != geometry:
            raise ValueError("measurement and reference geometries differ")
        if cpu and gpu and _geometry(cpu) != _geometry(gpu):
            raise ValueError("CPU and GPU reference geometries differ")
        if any(r is None or r["status"] != "complete" for r in (target, cpu, gpu)):
            result["reasons"].append("missing or incomplete dense/CPU reference")
            continue
        for oracle in (target, cpu, gpu):
            if oracle.get("source_revision") != record["source_revision"]:
                receipt_identity = identity(oracle)
                contract = _reference_contract(record, oracle)
                if reuse.get(receipt_identity) != contract:
                    raise ValueError(
                        "different reference revision lacks an audited reuse contract"
                    )
                result.setdefault("reference_reuse", {})[receipt_identity] = contract
            packages = record.get("environment", {}).get("packages")
            if (
                not packages
                or oracle.get("environment", {}).get("packages") != packages
            ):
                raise ValueError("measurement and reference package versions differ")
        # Dense-grid stability is recomputed for both the target and the GPU
        # spot check, never inferred from an implementation-written pass flag.
        stability = []
        for oracle in (target, gpu):
            samples = oracle["samples"]
            if len(samples) == 2:
                stability.append(
                    _comparison(*samples, oracle["atoms"], fraction=STABILITY_FRACTION)
                )
        cpu_sample = cpu["samples"][-1]
        # Compare implementations at identical quadratures. A CPU level-5 spot
        # can still validate the level-5 member of a new GPU 5/6 stability pair.
        gpu_spots = [
            sample
            for sample in gpu["samples"]
            if sample["level"] == cpu_sample["level"]
            and (sample.get("nonlocal_grid") or {}).get("level")
            == (cpu_sample.get("nonlocal_grid") or {}).get("level")
        ]
        if len(gpu_spots) != 1:
            result["reasons"].append("CPU/GPU spot quadratures differ")
            continue
        spot = _comparison(cpu_sample, gpu_spots[0], 3, fraction=STABILITY_FRACTION)
        result.update(grid_stability=stability, cpu_gpu_spot=spot)
        if not spot["passed"] or not all(s["passed"] for s in stability):
            result["reasons"].append(
                "dense reference stability or CPU/GPU spot gate failed"
            )
            continue
        for phase in ("cold", "warm"):
            if reduced[phase] is None:
                result[phase] = {"status": "incomplete"}
                continue
            checks = [
                _comparison(sample, target["samples"][-1], atoms)
                for sample in record["samples"]
                if sample["phase"] == phase
            ]
            result[phase] = {
                "status": "qualified" if all(c["passed"] for c in checks) else "failed",
                "errors": checks,
            }
        statuses = {result[phase]["status"] for phase in ("cold", "warm")}
        result["status"] = (
            "qualified"
            if statuses == {"qualified"}
            else "failed"
            if "failed" in statuses
            else "unverified"
        )
    return results
