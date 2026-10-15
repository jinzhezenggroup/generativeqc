"""Share complete build-generated CPU sources with a same-checkout audit.

This is a local build handoff, not an authenticity claim or a native binary
cache. Missing or stale receipts retain ordinary generation. Present receipts
with incomplete or corrupt outputs fail closed.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from typing import Any

from generativeqc_compiler.common.provenance import atomic_json, file_hash
from generativeqc_compiler.integral.first_derivative_schedule import (
    CPU_AOT_COMPONENTS,
    CPU_AOT_SHARDS,
    MAX_PROGRAM_BYTES,
    MAX_UNIT_BYTES,
    REQUESTS_PER_UNIT,
)

ROOT = Path(__file__).resolve().parents[1]
RECEIPT = ".generativeqc-stationary-cpu-sources.json"
SCHEMA = "generativeqc.stationary-cpu-source-handoff.v1"


def input_identity() -> dict[str, Any]:
    """Hash the complete compiler module/data inventory and the build producer.

    The primitive emitter's transitive Python closure is compiler-owned; it
    emits native include names without reading those headers. Hash the complete
    compiler inventory rather than a hand-maintained list of imported leaves.
    Python identity prevents replay across interpreter-dependent emission.
    """
    package = ROOT / "python/generativeqc_compiler"
    paths = {path for pattern in ("*.py", "*.json") for path in package.rglob(pattern)}
    paths.update(
        ROOT / "tools" / name
        for name in (
            "generate_stationary_cpu_derivative_aot.py",
            "stationary_cpu_source_receipt.py",
        )
    )
    return {
        "inputs": {
            path.relative_to(ROOT).as_posix(): file_hash(path) for path in sorted(paths)
        },
        "python": {
            "implementation": sys.implementation.name,
            "version": list(sys.version_info),
        },
        "components": list(CPU_AOT_COMPONENTS),
        "shards": CPU_AOT_SHARDS,
        "requests_per_unit": REQUESTS_PER_UNIT,
        "max_unit_bytes": MAX_UNIT_BYTES,
        "max_program_bytes": MAX_PROGRAM_BYTES,
    }


def _name(shard: int) -> str:
    return f"generativeqc_stationary_cpu_derivative_{shard}.cpp"


def _snapshot(path: Path, limit: int) -> bytes:
    if path.is_symlink() or not path.is_file():
        raise ValueError("CPU source handoff requires regular files")
    with path.open("rb") as stream:
        data = stream.read(limit + 1)
    if len(data) > limit:
        raise ValueError("CPU source handoff byte budget exceeded")
    return data


def publish(
    directory: Path, identity: dict[str, Any], sources: tuple[str, ...]
) -> None:
    """Publish only after every generated output has its exact emitted bytes."""
    if directory.is_symlink():
        raise ValueError("CPU source handoff directory must not be a symlink")
    if len(sources) != CPU_AOT_SHARDS:
        raise ValueError("CPU source handoff shard inventory mismatch")
    records = []
    total = 0
    for shard, source in enumerate(sources):
        data = _snapshot(directory / _name(shard), MAX_UNIT_BYTES)
        total += len(data)
        if not data or data != source.encode("utf-8") or total > MAX_PROGRAM_BYTES:
            raise ValueError("CPU source handoff differs from generated inventory")
        records.append(
            {
                "name": _name(shard),
                "bytes": len(data),
                "sha256": hashlib.sha256(data).hexdigest(),
            }
        )
    if identity != input_identity():
        raise ValueError("CPU source handoff inputs changed during generation")
    atomic_json(
        directory / RECEIPT,
        {"schema": SCHEMA, "identity": identity, "sources": records},
    )


def load(directory: Path | None) -> tuple[str, ...] | None:
    """Read verified immutable source snapshots, or retain ordinary generation."""
    if directory is None:
        return None
    if directory.is_symlink():
        raise ValueError("CPU source handoff directory must not be a symlink")
    receipt = directory / RECEIPT
    if not receipt.exists() and not receipt.is_symlink():
        return None
    try:
        metadata = json.loads(_snapshot(receipt, 1 << 20).decode("utf-8"))
        if not isinstance(metadata, dict) or metadata.get("schema") != SCHEMA:
            raise ValueError("unsupported receipt schema")
        if not isinstance(metadata.get("identity"), dict):
            raise TypeError("missing receipt input identity")
        # Changed source bytes or an added/deleted module invalidate reuse, even
        # if timestamps and sizes match. Local source-only tests still generate.
        identity = input_identity()
        if metadata["identity"] != identity:
            return None
        records = metadata["sources"]
        if not isinstance(records, list) or len(records) != CPU_AOT_SHARDS:
            raise ValueError("shard inventory mismatch")
        sources = []
        total = 0
        for shard, record in enumerate(records):
            if record["name"] != _name(shard):
                raise ValueError("shard order mismatch")
            data = _snapshot(
                directory / _name(shard), min(MAX_UNIT_BYTES, MAX_PROGRAM_BYTES - total)
            )
            total += len(data)
            if (
                not data
                or type(record["bytes"]) is not int
                or record["bytes"] != len(data)
                or record["sha256"] != hashlib.sha256(data).hexdigest()
            ):
                raise ValueError("source content mismatch")
            sources.append(data.decode("utf-8"))
        if input_identity() != identity:
            return None
        return tuple(sources)
    except (OSError, ValueError, KeyError, TypeError) as error:
        raise ValueError("CPU source handoff integrity failure") from error
