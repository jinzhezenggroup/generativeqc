"""Inspect consolidated historical measurements without Git history or a GPU.

The collection contains JSON values, not escaped JSON strings or raw archives.
Extraction retains original number/string tokens; exact original formatting and
retired scripts are recovered separately with restore_retained_evidence.py.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from decimal import Decimal
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.compact_evidence_publications import _plain
from tools.generativeqc_validation.retention import safe_relative
from tools.generativeqc_validation.storage_json import decode_storage_json

SCHEMA = "generativeqc.historical-measurements.v1"


def load_measurements(directory: Path) -> dict[str, bytes]:
    """Authenticate a bounded collection and expose lossless JSON token slices."""
    with (directory / "measurements.storage.json").open("rb") as stream:
        metadata = stream.read((64 << 10) + 1)
    if len(metadata) > 64 << 10:
        raise ValueError("oversized measurement storage manifest")
    manifest = decode_storage_json(metadata)
    if not isinstance(manifest, dict) or manifest.get("schema") != SCHEMA:
        raise ValueError("unsupported historical measurement storage")
    entry = manifest["storage"]
    path = directory / safe_relative(entry["path"])
    if not path.resolve().is_relative_to(directory.resolve()):
        raise ValueError("measurement storage escapes directory")
    with path.open("rb") as stream:
        stored = stream.read((1 << 20) + 1)
    if (
        len(stored) > 1 << 20
        or len(stored) != entry["bytes"]
        or hashlib.sha256(stored).hexdigest() != entry["sha256"]
    ):
        raise ValueError("historical measurement storage identity mismatch")
    plain = _plain(stored, path.name)
    value = decode_storage_json(plain)
    if (
        not isinstance(value, dict)
        or not value
        or type(manifest["record_count"]) is not int
        or len(value) != manifest["record_count"]
    ):
        raise ValueError("historical measurement count differs")
    # The validated top-level map is scanned instead of serialized again. Using
    # Decimal during scanning avoids a binary64 round-trip for decimal tokens.
    text = plain.decode("utf-8")
    decoder = json.JSONDecoder(parse_float=Decimal)
    offset = text.index("{") + 1
    records = {}
    for _ in value:
        while text[offset] in " \t\r\n":
            offset += 1
        name, offset = decoder.raw_decode(text, offset)
        safe_relative(name)
        while text[offset] in " \t\r\n:":
            offset += 1
        start = offset
        _, offset = decoder.raw_decode(text, offset)
        records[name] = text[start:offset].encode("utf-8")
        while text[offset] in " \t\r\n":
            offset += 1
        offset += 1  # validated comma or final closing brace
    return records


def main() -> None:
    """List logical paths, or extract one record into a new local file."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("record", nargs="?")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if bool(args.record) != bool(args.output):
        parser.error("record and --output must be specified together")
    records = load_measurements(args.directory)
    if args.record is None:
        print("\n".join(records))
        return
    data = records[args.record]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("xb") as stream:
        stream.write(data + b"\n")


if __name__ == "__main__":
    main()
