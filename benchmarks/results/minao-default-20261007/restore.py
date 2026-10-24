"""Restore the exact historical JSON receipts into an explicit scratch directory.

The bundle preserves original text, including all numerical values and ordering.
Density arrays were never tracked and remain a separate reproduction input.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import lzma
from pathlib import Path


def main() -> None:
    """Check every original byte identity before making scratch copies."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    manifest = json.loads((root / "receipts.json").read_text())
    compressed = (root / manifest["bundle"]).read_bytes()
    if hashlib.sha256(compressed).hexdigest() != manifest["bundle_sha256"]:
        raise ValueError("historical receipt bundle checksum mismatch")
    records = json.loads(lzma.decompress(compressed))
    if records.keys() != manifest["files"].keys():
        raise ValueError("historical receipt population mismatch")
    for name, original in records.items():
        relative = Path(name)
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError(f"invalid receipt path: {name}")
        data = original.encode()
        expected = manifest["files"][name]
        if (
            len(data) != expected["bytes"]
            or hashlib.sha256(data).hexdigest() != expected["sha256"]
        ):
            raise ValueError(f"historical receipt checksum mismatch: {name}")
        destination = args.output / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(data)
    print(f"Restored {len(records)} exact historical receipts to {args.output}")


if __name__ == "__main__":
    main()
