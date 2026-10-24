"""Verify publication and strict scientific replay from a fresh committed LF view."""

import argparse
import importlib.util
import json
import lzma
import math
import subprocess
import sys
import tempfile
from pathlib import Path

from tools.generativeqc_validation.publication import validate_publication
from tools.generativeqc_validation.retention import safe_relative

PREFIX = "benchmarks/results/cbs-independent-forces-20261024"


def equivalent(actual: object, expected: object) -> bool:
    if isinstance(actual, dict) and isinstance(expected, dict):
        return actual.keys() == expected.keys() and all(
            equivalent(actual[k], expected[k]) for k in actual
        )
    if isinstance(actual, list) and isinstance(expected, list):
        return len(actual) == len(expected) and all(
            equivalent(a, b) for a, b in zip(actual, expected, strict=True)
        )
    if type(actual) is float and type(expected) is float:
        # Dot-product roundoff may differ across host BLAS builds; gates never change.
        return math.isclose(actual, expected, rel_tol=0, abs_tol=1e-14)
    return type(actual) is type(expected) and actual == expected


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--revision", default="HEAD")
    group.add_argument("--working-tree", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).parent

    def read(name: str) -> bytes:
        name = safe_relative(name)
        if args.working_tree:
            path = root / name
            if path.is_symlink() or not path.resolve().is_relative_to(root.resolve()):
                raise ValueError("Publication file escapes the family directory")
            return path.read_bytes()
        return subprocess.check_output(
            ["git", "show", f"{args.revision}:{PREFIX}/{name}"]
        )

    manifest = json.loads(read("publication.json"))
    files = {entry["path"]: read(entry["path"]) for entry in manifest["files"]}
    validate_publication(manifest, files)
    with tempfile.TemporaryDirectory(prefix="cbs-force-replay-") as temporary:
        view = Path(temporary) / "view"
        view.mkdir()
        for name, raw in files.items():
            target = view / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(raw)
        spec = importlib.util.spec_from_file_location("force_pack", view / "pack.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        tasks = json.loads(lzma.decompress(files["inputs.json.xz"]))["tasks"]
        restored = Path(temporary) / "restored"
        if "receipts.json.xz" in files:
            module.restore_collection(
                json.loads(lzma.decompress(files["receipts.json.xz"])),
                restored,
                set(tasks),
            )
        else:
            restored.mkdir()
        output = Path(temporary) / "comparison.json"
        process = subprocess.run(
            [
                sys.executable,
                str(view / "replay.py"),
                "--raw",
                str(restored),
                "--output",
                str(output),
                "--require-pass",
            ],
            check=False,
        )
        if process.returncode != 0:
            raise SystemExit(process.returncode)
        assert equivalent(
            json.loads(output.read_bytes()),
            json.loads(lzma.decompress(files["comparison.json.xz"])),
        )
        assert manifest["decision"]["status"] == "accepted"
    print(
        "Publication and strict C4+C7 two-step scientific replay PASS; no new native qualification"
    )


if __name__ == "__main__":
    main()
