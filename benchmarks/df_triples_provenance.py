"""Bind supplied-state triples replays to their source-construction frame.

The replay producer must capture ``frame_identity`` before constructing native
factors and pass that identity to ``write_manifest`` after writing the replay.
There is deliberately no command to certify arbitrary legacy files afterwards.
Hashes establish provenance, not numerical accuracy of the source factors.
"""

from __future__ import annotations

import hashlib
import json
from typing import TYPE_CHECKING

import numpy as np

from benchmarks._retention import raw_output_path

if TYPE_CHECKING:
    from pathlib import Path

SCHEMA = "generativeqc.df-triples.frame.v1"
FRAME_KEYS = (
    "atoms_angstrom",
    "basis",
    "auxiliary_basis",
    "basis_definition",
    "auxiliary_basis_definition",
    "basis_representation",
    "orbital_order",
    "reference_mode",
    "metric_policy",
    "metric_relative_threshold",
    "metric_rank",
    "nocc",
    "nvir",
    "naux",
)
ARRAY_KEYS = ("coefficients", "orbital_energies", "t1", "t2", "metric_whitening")


def sha256(path: Path) -> str:
    """Hash retained input bytes without creating another resident copy."""
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _digest(value: dict) -> str:
    return hashlib.sha256(
        json.dumps(
            value, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
    ).hexdigest()


def frame_identity(state: Path, reference: Path) -> dict:
    """Capture the exact source input frame before source construction.

    The oracle's state digest binds its reference result to the state. Explicit
    basis definitions avoid depending solely on mutable basis names. Array
    digests document the occupied/virtual order and amplitudes being supplied;
    the whole-file digest additionally binds every retained source input.
    """
    record = json.loads(reference.read_text())
    state_digest = sha256(state)
    if record.get("state_sha256") != state_digest:
        raise ValueError("independent state identity mismatch")
    missing = [key for key in FRAME_KEYS if key not in record]
    if missing:
        raise ValueError(f"reference lacks explicit source-frame metadata: {missing}")
    if record["orbital_order"] != "occupied_then_virtual":
        raise ValueError("unsupported supplied orbital order")
    if record["basis_representation"] not in ("spherical", "cartesian"):
        raise ValueError("unsupported basis representation")
    o, v, q = (int(record[key]) for key in ("nocc", "nvir", "naux"))
    if min(o, v, q) <= 0:
        raise ValueError("invalid source-frame dimensions")
    shapes = ((o + v, o + v), (o + v,), (o, v), (o, o, v, v), (q, q))
    arrays = {}
    with np.load(state, allow_pickle=False) as saved:
        for key, shape in zip(ARRAY_KEYS, shapes, strict=True):
            array = np.asarray(saved[key], dtype="<f8")
            if array.shape != shape or not np.isfinite(array).all():
                raise ValueError(f"invalid source-frame {key}")
            arrays[key] = hashlib.sha256(
                memoryview(np.ascontiguousarray(array)).cast("B")
            ).hexdigest()
    payload = {
        "schema": SCHEMA,
        "metadata": {key: record[key] for key in FRAME_KEYS},
        "state_sha256": state_digest,
        "array_sha256": arrays,
    }
    return {**payload, "frame_id": _digest(payload)}


def _check_replay_amplitudes(replay: Path, frame: dict) -> None:
    """Check replay shape and converged amplitudes without numerical reconstruction."""
    with replay.open("rb") as stream:
        header = np.fromfile(stream, dtype="<u8", count=7)
    metadata = frame["metadata"]
    o, v, q = (int(metadata[key]) for key in ("nocc", "nvir", "naux"))
    if len(header) != 7 or list(map(int, header[:3])) != [o, v, q]:
        raise ValueError("replay dimensions do not match source frame")
    # Solver replay: Foo/Fov/Fvv, five retained ERI blocks, D1/D2,
    # converged T1/T2, Bov/Bvv. Empty ovvv/vvvv carry no bytes.
    prefix = o * o + o * v + v * v + 3 * o * o * v * v + o**3 * v + o**4
    prefix += o * v + o * o * v * v
    total = prefix + o * v + o * o * v * v + q * o * v + q * v * v
    if replay.stat().st_size != 56 + 8 * total:
        raise ValueError("replay size does not match source frame")
    mapped = np.memmap(replay, dtype="<f8", mode="r", offset=56)
    for key, size in (("t1", o * v), ("t2", o * o * v * v)):
        digest = hashlib.sha256(
            memoryview(mapped[prefix : prefix + size]).cast("B")
        ).hexdigest()
        if digest != frame["array_sha256"][key]:
            raise ValueError(f"replay {key} does not match source frame")
        prefix += size


def write_manifest(
    path: Path, replay: Path, state: Path, reference: Path, *, source_frame: dict
) -> None:
    """Bind a newly constructed replay using its previously captured frame.

    ``source_frame`` must be the input identity retained by the source producer,
    not an identity reconstructed from unrelated files at qualification time.
    The native producer remains responsible for using those inputs; this does
    not certify the raw integrals or replace the original per-factor gate.
    """
    frame = frame_identity(state, reference)
    if source_frame != frame:
        raise ValueError("source-construction frame identity mismatch")
    _check_replay_amplitudes(replay, frame)
    manifest = {
        "schema": SCHEMA,
        "frame": frame,
        "sha256": {
            "input": sha256(replay),
            "state": sha256(state),
            "reference": sha256(reference),
        },
        "upstream_factor_acceptance": "unqualified",
    }
    raw_output_path(path).write_text(
        json.dumps(manifest, indent=2, allow_nan=False) + "\n"
    )


def validate_manifest(path: Path, replay: Path, state: Path, reference: Path) -> dict:
    """Reject mixed input identities before loading the CUDA library or probe."""
    manifest = json.loads(path.read_text())
    if manifest.get("schema") != SCHEMA:
        raise ValueError("unsupported triples source-frame manifest")
    paths = {"input": replay, "state": state, "reference": reference}
    for key, value in paths.items():
        if manifest.get("sha256", {}).get(key) != sha256(value):
            raise ValueError(f"triples manifest {key} identity mismatch")
    frame = frame_identity(state, reference)
    if manifest.get("frame") != frame:
        raise ValueError("triples source-frame identity mismatch")
    if manifest.get("upstream_factor_acceptance") != "unqualified":
        raise ValueError("triples manifest cannot promote upstream factor acceptance")
    _check_replay_amplitudes(replay, frame)
    return manifest
