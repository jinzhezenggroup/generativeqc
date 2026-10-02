"""Read frozen pre-rename basis evidence through the current strict loaders."""

from __future__ import annotations

import json
from contextlib import contextmanager
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import TYPE_CHECKING, Any

from generativeqc.basis import SCHEMA, load_basis, read_local_json
from generativeqc.profiles import canonical_hash

if TYPE_CHECKING:
    from collections.abc import Iterator

    from generativeqc.basis import BasisSet


@contextmanager
def retained_basis_record(source: Path) -> Iterator[Path]:
    """Translate only checked legacy schema metadata in a temporary record."""
    payload, _ = read_local_json(source)
    checksum = payload.pop("checksum", None)
    if checksum != canonical_hash(payload):
        raise ValueError("retained basis fixture checksum mismatch")
    if payload.get("schema") != "vibeqc.basis":
        raise ValueError("unsupported retained basis fixture schema")
    # Only the project namespace changed; preserve all decimal tables/provenance.
    payload["schema"] = SCHEMA
    payload["checksum"] = canonical_hash(payload)
    with TemporaryDirectory(prefix="generativeqc-retained-basis-") as directory:
        record = Path(directory) / source.name
        record.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        yield record


def load_retained_basis(source: Path) -> BasisSet:
    """Snapshot a historical fixture without modifying the retained source."""
    with retained_basis_record(source) as record:
        return load_basis(record)


def load_retained_comparison_basis(
    source: Path, case: Any, *, role: str, compute_forces: bool
) -> Any:
    """Keep the comparison loader's capability checks and reference conversion."""
    from benchmarks.compare_gpu4pyscf_batch import load_comparison_basis

    with retained_basis_record(source) as record:
        return load_comparison_basis(
            record, case, role=role, compute_forces=compute_forces
        )
