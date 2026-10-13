"""Verify publication integrity and recheck complete E+F gates without a GPU."""

from __future__ import annotations

import hashlib
import argparse
import json
import lzma
import tarfile
from pathlib import Path

from tools.render_readme_cbs import qualification


def main() -> None:
    """Read archive members in memory; never extract unchecked filesystem paths."""
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-archive", type=Path)
    args = parser.parse_args()
    manifest = json.loads((root / "manifest.json").read_text())
    for name, expected in manifest["files"].items():
        data = (root / name).read_bytes()
        assert len(data) == expected["bytes"], name
        assert hashlib.sha256(data).hexdigest() == expected["sha256"], name
    summary = json.loads(lzma.decompress((root / "summary.json.xz").read_bytes()))
    source = lzma.decompress((root / "qualified-source.patch.xz").read_bytes())
    assert hashlib.sha256(source).hexdigest() == manifest["source_patch_sha256"]
    checked = []
    for point in summary["points"]:
        native = point["native"]
        check = qualification(native["samples"][0], native["protocol"], point["reference"])
        assert check["execution_gate"] and check["accuracy_gate"] is not False
        if check["accuracy_gate"]:
            checked.append(point["carbon"])
    assert checked == manifest["independently_checked_native"]
    if args.raw_archive:
        verify_originals(args.raw_archive, manifest, summary)
    print(f"Verified C1–C7 complete E+F; independent directional checks: {checked}")


def verify_originals(path: Path, manifest: dict, summary: dict) -> None:
    """Optionally audit locally retained originals without publishing a full run bundle."""
    assert hashlib.sha256(path.read_bytes()).hexdigest() == manifest["raw_archive"]["sha256"]
    with tarfile.open(path, "r:xz") as archive:
        for name, expected in manifest["raw_archive"]["members"].items():
            data = archive.extractfile(name).read()
            assert len(data) == expected["bytes"], name
            assert hashlib.sha256(data).hexdigest() == expected["sha256"], name
        for point in summary["points"]:
            carbon = point["carbon"]
            raw = json.load(archive.extractfile(f"primary-32-gib/{carbon}/native.json"))
            native = point["native"]
            assert raw["status"] == "complete"
            assert raw["library_sha256"] == manifest["library_sha256"]
            assert native["samples"] == [
                dict(raw["result"], complete_seconds=raw["complete_seconds"])
            ]


if __name__ == "__main__":
    main()
