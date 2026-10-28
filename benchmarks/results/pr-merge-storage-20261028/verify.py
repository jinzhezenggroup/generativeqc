"""Opt-in, byte-preserving replay of the historical storage migration."""

from __future__ import annotations

import gzip
import hashlib
import json
import lzma
from pathlib import Path


def verify(repository: Path) -> tuple[int, int]:
    """Authenticate stored and decoded bytes without executing historical code."""
    manifest = json.loads(Path(__file__).with_name("manifest.json").read_text())
    saved = 0
    for row in manifest["members"]:
        path = repository / row["stored_path"]
        if not path.resolve().is_relative_to(repository.resolve()):
            raise ValueError("storage path escapes repository")
        packed = path.read_bytes()
        decoder = {"xz": lzma.decompress, "gzip": gzip.decompress}[row["encoding"]]
        raw = decoder(packed)
        for prefix, data in (("stored", packed), ("original", raw)):
            if (
                len(data) != row[prefix + "_bytes"]
                or hashlib.sha256(data).hexdigest() != row[prefix + "_sha256"]
            ):
                raise ValueError(f"{prefix} identity mismatch: {path}")
        saved += len(raw) - len(packed)
    return len(manifest["members"]), saved


if __name__ == "__main__":
    count, saved = verify(Path(__file__).resolve().parents[3])
    print(f"Verified {count} exact original files; {saved} payload bytes saved")
