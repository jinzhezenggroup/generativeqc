"""Verify compact evidence; optionally restore exact measured text inputs/results."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Iterator

HERE = Path(__file__).resolve().parent


def decoded(entry: dict[str, Any]) -> bytes:
    """Reconstitute the original UTF-8 bytes, including JSON formatting."""
    if entry["encoding"] == "json-indent2-newline":
        return (json.dumps(entry["content"], indent=2, allow_nan=False) + "\n").encode()
    if entry["encoding"] == "utf8":
        return entry["content"].encode()
    raise ValueError(f"Unknown encoding: {entry['encoding']}")


def members(directory: Path = HERE) -> Iterator[tuple[str, bytes]]:
    """Yield hash-checked original records from the deterministic gzip capsules."""
    for path in sorted(directory.glob("*.json.gz")):
        payload = json.loads(gzip.decompress(path.read_bytes()))
        if payload.get("schema") != "generativeqc.cpu-eri-evidence-capsule.v1":
            raise ValueError(f"Unknown capsule schema: {path}")
        for entry in payload["files"]:
            name = entry["path"]
            relative = PurePosixPath(name)
            if relative.is_absolute() or any(
                p in {"", ".", ".."} for p in name.split("/")
            ):
                raise ValueError(f"Unsafe retained path: {name}")
            data = decoded(entry)
            if (
                len(data) != entry["bytes"]
                or hashlib.sha256(data).hexdigest() != entry["sha256"]
            ):
                raise ValueError(f"Changed original record: {name}")
            yield name, data


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--restore", type=Path, help="New directory for original text files"
    )
    args = parser.parse_args()
    manifest = json.loads((HERE / "manifest.json").read_text())
    for entry in manifest["files"]:
        data = (HERE / entry["path"]).read_bytes()
        if (
            len(data) != entry["bytes"]
            or hashlib.sha256(data).hexdigest() != entry["sha256"]
        ):
            raise ValueError(f"Changed published member: {entry['path']}")
    records = dict(members())
    for campaign, expected in [
        ("matched-endpoints-bytecode-normalized", 960),
        ("matched-endpoints-attempt-01", 432),
    ]:
        status = json.loads(records[f"{campaign}/run-status.json"])
        rows = [
            row
            for process in status
            for row in json.loads(records[f"{campaign}/{process['json_file']}"])["rows"]
        ]
        if len(rows) != expected or not all(row["converged"] for row in rows):
            raise ValueError(f"Incomplete or unconverged retained campaign: {campaign}")
    if args.restore:
        args.restore.mkdir(parents=True, exist_ok=False)
        for name, data in records.items():
            path = args.restore / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data)
    print(
        f"Verified {len(manifest['files'])} publication files and {len(records)} exact original text records; 960 normalized and 432 pilot endpoints."
    )


if __name__ == "__main__":
    main()
