"""Reduce retained production endpoint receipts without discarding work or gates."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import tempfile
import uuid
from pathlib import Path
from typing import Any

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from benchmarks._retention import raw_output_path

EXPECTED_ARMS = frozenset(
    (atoms, policy, delta)
    for atoms, policy in ((12, "fast"), (96, "fast"), (12, "fp64"))
    for delta in ("0", "auto")
)
PRECISION_COUNTERS = (
    "mixed_stage_fock_builds",
    "strict_stage_fock_builds",
    "post_scf_fock_builds",
    "execution_retries",
    "fallback_count",
    "final_residual_audits",
)


def digest(path: Path) -> str:
    """Fingerprint raw evidence rather than relying on its local pathname."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def nonfinite_fields(value: Any, location: str = "$") -> list[str]:
    """Find invalid evidence before serialization or numerical comparisons."""
    if isinstance(value, float) and not math.isfinite(value):
        return [location]
    if isinstance(value, dict):
        return [
            field
            for key, item in value.items()
            for field in nonfinite_fields(item, f"{location}.{key}")
        ]
    if isinstance(value, (list, tuple)):
        return [
            field
            for index, item in enumerate(value)
            for field in nonfinite_fields(item, f"{location}[{index}]")
        ]
    return []


def json_evidence(value: Any) -> Any:
    """Retain invalid values in negative diagnostics as explicit strings."""
    if isinstance(value, float) and not math.isfinite(value):
        return str(value)
    if isinstance(value, dict):
        return {str(key): json_evidence(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_evidence(item) for item in value]
    return value


def publish_json(path: Path, value: dict[str, Any]) -> None:
    """Publish a complete receipt atomically, never overwriting evidence."""
    data = json.dumps(value, indent=2, allow_nan=False) + "\n"
    with tempfile.TemporaryDirectory(prefix=".reduction-", dir=path.parent) as staging:
        staged = Path(staging) / path.name
        with staged.open("x") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(staged, path)


def read_evidence(path: Path, errors: list[str], *, structured: bool = False) -> Any:
    """Collect missing/malformed evidence instead of masking an endpoint failure."""
    try:
        text = path.read_text()
        return json.loads(text) if structured else text
    except (OSError, ValueError) as error:
        errors.append(f"{path}: {type(error).__name__}: {error}")
        return None


def finite_number(value: Any) -> bool:
    """Reject missing, boolean and nonfinite numeric evidence."""
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
    )


def valid_digest(value: Any) -> bool:
    """Require complete hexadecimal source and library identities."""
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value.lower())
    )


def bind_samples_to_build(
    reduced: list[tuple[dict[str, Any], dict[str, Any]]],
    authentication: Any,
    errors: list[str],
) -> None:
    """Bind every arm to the source and library authenticated by the build."""
    build = authentication if isinstance(authentication, dict) else {}
    for field in ("source_identity", "library_sha256"):
        expected = build.get(field)
        if not valid_digest(expected):
            errors.append(f"Missing or malformed build {field} binding")
            for sample, _ in reduced:
                sample["gate"] = False
            continue
        for sample, _ in reduced:
            source = sample.get("source_authentication")
            observed = (
                (source.get("native_identity") if isinstance(source, dict) else None)
                if field == "source_identity"
                else sample.get("library_sha256")
            )
            if not valid_digest(observed) or observed != expected:
                error = (
                    f"Source/library binding differs from build: "
                    f"{sample['receipt']}/{sample['receipt_file']} {field}; "
                    f"expected={expected}, observed={observed}"
                )
                errors.append(error)
                sample["diagnostic_errors"].append(error)
                sample["gate"] = False


def reduce_sample(path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    """Retain failed endpoints, including those without precision diagnostics."""
    errors: list[str] = []
    record = read_evidence(path, errors, structured=True)
    if not isinstance(record, dict):
        errors.append("Endpoint receipt is not a JSON object")
        record = {}
    precision = record.get("precision")
    diagnostic = record.get("ks_diagnostic")
    if not isinstance(precision, dict):
        errors.append("Missing precision diagnostics")
        precision = {}
    if not isinstance(diagnostic, dict):
        errors.append("Missing KS diagnostics")
        diagnostic = {}
    before_text = read_evidence(path.parent / "library-before.sha256", errors)
    after_text = read_evidence(path.parent / "library-after.sha256", errors)
    before = before_text.split()[0] if before_text and before_text.split() else None
    after = after_text.split()[0] if after_text and after_text.split() else None
    exit_text = read_evidence(path.parent / "exit-code.txt", errors)
    try:
        exit_code = int(exit_text) if exit_text is not None else None
    except ValueError:
        errors.append("Invalid endpoint exit code")
        exit_code = None
    sample = {
        name: record.get(name)
        for name in (
            "atoms",
            "policy",
            "delta",
            "slurm_job_id",
            "energy",
            "iterations",
            "fock_builds",
            "construction_seconds",
            "prepare_seconds",
            "execute_seconds",
            "export_seconds",
            "teardown_seconds",
            "cold_endpoint_seconds",
            "initial_guess",
            "density_rms",
            "physical_residual_rms",
            "energy_change",
            "warm_start_used",
            "warm_start_fallback",
            "candidate_controls",
            "independent_gate",
            "error",
            "cleanup_error",
            "publication_error",
            "failed_elapsed_seconds",
            "nonfinite_fields",
        )
    }
    authentication = read_evidence(
        path.parent / "source-authentication.json", errors, structured=True
    )
    if not isinstance(authentication, dict) or not authentication.get(
        "native_identity"
    ):
        errors.append("Missing source authentication")
    try:
        receipt_sha256 = digest(path)
    except OSError as error:
        errors.append(f"Cannot fingerprint receipt: {error}")
        receipt_sha256 = None
    sample.update(
        receipt=path.parent.name,
        receipt_file=path.name,
        receipt_sha256=receipt_sha256,
        source_authentication=authentication,
        library_sha256=before,
        library_unchanged=bool(before and before == after),
        exit_code=exit_code,
        nonlocal_work=diagnostic.get("nonlocal_work"),
        coulomb_exchange_work=diagnostic.get("coulomb_exchange_work"),
        nonlocal_history_scopes=diagnostic.get("nonlocal_history_scopes"),
        operator_inventory=precision.get("operators"),
    )
    for name in PRECISION_COUNTERS:
        sample[name] = precision.get(name)
        if not finite_number(sample[name]) or sample[name] < 0:
            errors.append(f"Missing or invalid precision counter: {name}")
    for name in ("nonlocal_work", "coulomb_exchange_work", "nonlocal_history_scopes"):
        if diagnostic.get(name) is None:
            errors.append(f"Missing KS diagnostic: {name}")
    if precision.get("operators") is None:
        errors.append("Missing operator inventory")
    if not finite_number(record.get("energy")):
        errors.append("Missing or nonfinite energy")
    seconds = record.get("cold_endpoint_seconds")
    if not finite_number(seconds) or seconds <= 0:
        errors.append("Missing, nonfinite or nonpositive endpoint time")
    try:
        forces = np.asarray(record.get("forces"), dtype=float)
        if forces.shape != (record.get("atoms"), 3) or not np.isfinite(forces).all():
            errors.append("Missing, nonfinite or malformed forces")
    except (TypeError, ValueError):
        errors.append("Malformed forces")
    invalid = nonfinite_fields(record)
    if invalid:
        errors.append(f"Nonfinite receipt fields: {', '.join(invalid)}")
    gate = record.get("independent_gate")
    sample["gate"] = bool(
        not errors
        and record.get("succeeded") is True
        and record.get("converged") is True
        and record.get("finite_forces") is True
        and record.get("warm_start_used") is False
        and not record.get("error")
        and not record.get("cleanup_error")
        and not record.get("publication_error")
        and not record.get("nonfinite_fields")
        and isinstance(gate, dict)
        and gate.get("gate") is True
        and sample["library_unchanged"]
        and exit_code == 0
        and precision.get("complete") is True
        and precision.get("operator_inventory_complete") is True
    )
    sample["diagnostic_errors"] = errors
    return sample, record


def compare_pair(
    off: tuple[dict[str, Any], dict[str, Any]],
    default: tuple[dict[str, Any], dict[str, Any]],
    root: Path,
) -> dict[str, Any]:
    """Compare identical-controls cold endpoints and their checked final states."""
    off_sample, off_record = off
    on_sample, on_record = default
    if not off_sample["gate"] or not on_sample["gate"]:
        raise ValueError("Pair contains an unqualified endpoint")
    forces = np.asarray(on_record["forces"]) - np.asarray(off_record["forces"])
    densities = []
    for sample, record in (off, default):
        metadata = record["final_density"]
        path = root / sample["receipt"] / Path(metadata["path"]).name
        if path.is_symlink():
            raise ValueError("Density evidence must be a retained file, not a symlink")
        if digest(path) != metadata["sha256"]:
            raise ValueError(f"Density checksum differs: {path}")
        density = np.load(path, allow_pickle=False)
        if (
            not density.size
            or density.size != metadata["count"]
            or not np.isfinite(density).all()
        ):
            raise ValueError(f"Missing, nonfinite or malformed density: {path}")
        densities.append(density)
    if densities[0].shape != densities[1].shape:
        raise ValueError("Paired density shapes differ")
    density_delta = densities[1] - densities[0]
    off_seconds, on_seconds = (
        off_sample["cold_endpoint_seconds"],
        on_sample["cold_endpoint_seconds"],
    )
    return {
        "atoms": off_record["atoms"],
        "policy": off_record["policy"],
        "off_receipt": off_sample["receipt"],
        "default_receipt": on_sample["receipt"],
        "off_seconds": off_seconds,
        "default_seconds": on_seconds,
        "speedup_off_over_default": off_seconds / on_seconds,
        "elapsed_reduction_percent": 100 * (1 - on_seconds / off_seconds),
        "energy_abs_difference_hartree": abs(
            on_record["energy"] - off_record["energy"]
        ),
        "force_max_difference_hartree_per_bohr": float(np.max(np.abs(forces))),
        "density_max_difference": float(np.max(np.abs(density_delta))),
        "same_controls": bool(off_record["candidate_controls"])
        and off_record["candidate_controls"] == on_record["candidate_controls"],
        "same_geometry": bool(off_record["atoms_bohr"])
        and off_record["atoms_bohr"] == on_record["atoms_bohr"],
        "same_basis": bool(off_record["basis_identity"])
        and off_record["basis_identity"] == on_record["basis_identity"],
        "both_independently_qualified": off_sample["gate"] and on_sample["gate"],
    }


def main() -> None:
    """Publish only a complete qualified suite; retain rejected attempts separately."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--build-receipts", type=Path, required=True)
    parser.add_argument("--output", type=raw_output_path, required=True)
    args = parser.parse_args()
    errors: list[str] = []
    paths = []
    for directory in sorted(args.root.glob("*")):
        if directory.is_dir():
            receipts = sorted(directory.glob("result*.json"))
            # A timeout/import/setup failure may leave only the wrapper's logs.
            # It is still a retained attempt, never an absent successful arm.
            paths.extend(receipts or [directory / "result.json"])
    reduced = [reduce_sample(path) for path in paths]
    build = args.build_receipts
    authentication = read_evidence(
        build / "authentication.json", errors, structured=True
    )
    bind_samples_to_build(reduced, authentication, errors)
    arms: dict[tuple[int, str, str], list[Any]] = {}
    for sample, record in reduced:
        key = (record.get("atoms"), record.get("policy"), record.get("delta"))
        if not isinstance(key[0], int) or not all(
            isinstance(item, str) for item in key[1:]
        ):
            errors.append(
                f"Invalid arm identity in {sample['receipt']}/{sample['receipt_file']}"
            )
            continue
        arms.setdefault(key, []).append((sample, record))
    for key in sorted(EXPECTED_ARMS):
        count = len(arms.get(key, []))
        if count != 1:
            errors.append(f"Expected exactly one endpoint for {key}; found {count}")
    for key in arms.keys() - EXPECTED_ARMS:
        errors.append(f"Unexpected endpoint arm: {key}")
    pairs = []
    for atoms, policy in ((12, "fast"), (96, "fast"), (12, "fp64")):
        off, default = (
            arms.get((atoms, policy, "0"), []),
            arms.get((atoms, policy, "auto"), []),
        )
        if len(off) != 1 or len(default) != 1:
            continue
        try:
            pair = compare_pair(off[0], default[0], args.root)
            pairs.append(pair)
            if nonfinite_fields(pair) or not all(
                pair[key]
                for key in (
                    "same_controls",
                    "same_geometry",
                    "same_basis",
                    "both_independently_qualified",
                )
            ):
                errors.append(
                    f"Paired endpoints differ in the qualified scientific contract: {(atoms, policy)}"
                )
        except (OSError, ValueError, TypeError, KeyError) as error:
            errors.append(
                f"Cannot qualify pair {(atoms, policy)}: {type(error).__name__}: {error}"
            )
    if not reduced or not all(sample["gate"] for sample, _ in reduced):
        errors.append("At least one retained endpoint failed qualification")
    summary = {
        "schema": "generativeqc.fast-anchor-default-endpoints.v1",
        "base_revision": "5d1dbbcaf2f04fb83980c8f988795df28b41e392",
        "warning": "One cold observation per arm, not repeated statistics or a universal speed claim; all first-use costs retained.",
        "build": authentication,
        "rebuilt_inputs": read_evidence(
            build / "rebuilt-inputs.json", errors, structured=True
        ),
        "compiler_cache_version": (
            read_evidence(build / "cache-version.txt", errors) or ""
        ).strip(),
        "compiler_cache_before": read_evidence(build / "cache-before.txt", errors),
        "compiler_cache_after": read_evidence(build / "cache-after.txt", errors),
        "samples": [sample for sample, _ in reduced],
        "pairs": pairs,
    }
    if nonfinite_fields(summary):
        errors.append("Reduction contains nonfinite evidence")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if not errors:
        try:
            publish_json(args.output, summary)
        except FileExistsError:
            errors.append(f"Requested summary path is already occupied: {args.output}")
    if errors:
        summary["qualification_errors"] = errors
        summary["raw_records"] = [record for _, record in reduced]
        rejected = args.output.with_name(
            f"{args.output.stem}.rejected-{uuid.uuid4().hex}.json"
        )
        publish_json(rejected, json_evidence(summary))
        raise SystemExit(f"Qualification rejected; diagnostics retained at {rejected}")
    for pair in pairs:
        print(json.dumps(pair, allow_nan=False))


if __name__ == "__main__":
    main()
