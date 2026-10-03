"""Read scientific JSON whose large lists are stored in named companion files.

Lists are grouped by workload or observable, with their original ordering and
values preserved. Parts are ordinary reviewable JSON, not byte fragments.
Storage metadata does not change the reconstructed validation schema.
"""

from __future__ import annotations

import gzip
import json
from pathlib import Path, PurePosixPath
from typing import Any

from .retention import digest, safe_relative


def decode_json(data: bytes, *, path: str | Path | None = None) -> Any:
    """Decode ordinary or gzip-compressed JSON."""
    name = str(path) if path is not None else ""
    if name.endswith(".gz") or data.startswith(b"\x1f\x8b"):
        data = gzip.decompress(data)
    return json.loads(data)


def load_json(path: Path) -> Any:
    """Load ordinary or gzip-compressed JSON from the checkout."""
    path = Path(path)
    return decode_json(path.read_bytes(), path=path)


def decode_record(
    data: bytes, files: dict[str, bytes], *, path: str | Path | None = None
) -> dict:
    """Restore list fields from checksum-pinned JSON parts in a publication.

    Part paths are relative to the record's directory. A field must be absent
    from the main record so storage metadata cannot overwrite scientific data.
    Parts cannot recursively include files or redefine other fields.
    """
    record = decode_json(data, path=path)
    parts = record.pop("record_parts", {})
    for field, entries in parts.items():
        if field in record or not entries:
            raise ValueError("record parts overwrite a field or contain no files")
        values = []
        seen = set()
        for entry in entries:
            path = safe_relative(entry["path"])
            if path in seen:
                raise ValueError("duplicate record part")
            seen.add(path)
            raw = files[path]
            if len(raw) != entry["bytes"] or digest(raw) != entry["sha256"]:
                raise ValueError(f"record part checksum/size mismatch: {path}")
            part = decode_json(raw, path=path)
            if not isinstance(part, list):
                raise TypeError("record part must contain a list")
            values.extend(part)
        record[field] = values
    return record


def load_record(path: Path) -> dict:
    """Load a plain or partitioned scientific record from the local checkout."""
    path = Path(path)
    data = path.read_bytes()
    parts = decode_json(data, path=path).get("record_parts", {})
    files = {}
    for entries in parts.values():
        for entry in entries:
            name = safe_relative(entry["path"])
            target = path.parent / name
            if not target.resolve().is_relative_to(path.parent.resolve()):
                raise ValueError("record part escapes the publication directory")
            files[name] = target.read_bytes()
    return decode_record(data, files, path=path)


def load_publication_record(
    directory: Path, *, role: str = "evidence", name: str | None = None
) -> dict:
    """Read a hash-bound record using its declared storage path, not a suffix.

    ``name`` optionally names one logical plain JSON member; its gzip companion
    is accepted only when declared by the publication. Ambiguous roles, duplicate
    paths, escaping links and stale stored identities fail before decoding.
    This verifies storage, not the publication's scientific acceptance decision.
    """
    directory = Path(directory)
    inventory = directory / "publication.json"
    if not inventory.resolve().is_relative_to(directory.resolve()):
        raise ValueError("publication inventory escapes its directory")
    manifest = json.loads(inventory.read_bytes())
    if name is not None:
        name = safe_relative(name)
    files: dict[str, bytes] = {}
    selected = []
    for entry in manifest["files"]:
        path = safe_relative(entry["path"])
        if path in files:
            raise ValueError("duplicate publication member path")
        target = directory / path
        if not target.resolve().is_relative_to(directory.resolve()):
            raise ValueError("publication member escapes its directory")
        data = target.read_bytes()
        if len(data) != entry["bytes"] or digest(data) != entry["sha256"]:
            raise ValueError("publication member checksum/size mismatch")
        files[path] = data
        if entry["role"] == role and (name is None or path in {name, name + ".gz"}):
            selected.append(path)
    if len(selected) != 1:
        raise ValueError("publication record selection must be unique")
    path = selected[0]
    prefix = PurePosixPath(path).parent
    companions = {
        PurePosixPath(member).relative_to(prefix).as_posix(): data
        for member, data in files.items()
        if PurePosixPath(member).is_relative_to(prefix)
    }
    return decode_record(files[path], companions, path=path)
