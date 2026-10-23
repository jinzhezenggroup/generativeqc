"""Verify and optionally restore lossless endpoint evidence into ignored storage."""

from __future__ import annotations

import argparse
import hashlib
import json
import lzma
from pathlib import Path, PurePosixPath


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--extract", type=Path)
    parser.add_argument("--archive", type=Path, help="verified Git-restored xz bundle")
    arguments = parser.parse_args()
    root = Path(__file__).resolve().parent
    storage = json.loads((root / "storage.json").read_text())
    stored = (arguments.archive or root / storage["path"]).read_bytes()
    if hashlib.sha256(stored).hexdigest() != storage["stored_sha256"]:
        raise ValueError("stored evidence digest mismatch")
    decoder = lzma.LZMADecompressor(memlimit=256 << 20)
    decoded = decoder.decompress(stored, max_length=(16 << 20) + 1)
    if not decoder.eof or decoder.unused_data or len(decoded) > 16 << 20:
        raise ValueError("evidence exceeds the decoded bound or contains trailing data")
    if hashlib.sha256(decoded).hexdigest() != storage["decoded_sha256"]:
        raise ValueError("decoded evidence digest mismatch")
    payload = json.loads(decoded)
    if payload["schema"] != "generativeqc.range-separated-df-evidence.v1":
        raise ValueError("unexpected evidence schema")
    if len(payload["files"]) != storage["file_count"]:
        raise ValueError("evidence member count mismatch")
    for name, text in payload["files"].items():
        path = PurePosixPath(name)
        if (
            path.is_absolute()
            or "\\" in name
            or ":" in name
            or any(part in ("", ".", "..") for part in name.split("/"))
        ):
            raise ValueError("noncanonical evidence member path")
        if arguments.extract is not None:
            target = arguments.extract / path
            data = text.encode("utf-8")
            if target.exists() and target.read_bytes() != data:
                raise ValueError(f"refusing to overwrite different evidence: {target}")
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(data)
    print(f"Verified {storage['file_count']} lossless evidence members")


if __name__ == "__main__":
    main()
