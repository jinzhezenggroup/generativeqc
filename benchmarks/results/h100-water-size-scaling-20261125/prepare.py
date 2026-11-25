"""Reconstruct the frozen source, geometry and DF inputs for this campaign."""

import argparse
import hashlib
import json
import lzma
import os
from pathlib import Path
import shutil
import subprocess
import tarfile

from build import source_identity
from geometry import generate

REVISION = "22f72d0692a760c2892cd7791022584d9ae1c3e5"
COHORT_REVISION = "d154546d2d6629ab8c865ca08f680002d68b4152"
COHORT = "benchmarks/results/paper-hfdft-cold3-20261120/h100"
EXCLUSIONS = ("assets", "benchmarks/results")
METHODS = ("hf", "pbe", "r2scan", "wb97m-v")
HERE = Path(__file__).resolve().parent


def plain(value):
    return (json.dumps(value, indent=2) + "\n").encode()


def digest(data):
    return hashlib.sha256(data).hexdigest()


def published(repository, name):
    object_name = f"{COHORT_REVISION}:{COHORT}/{name}.json.xz"
    packed = subprocess.check_output(["git", "-C", repository, "show", object_name])
    return json.loads(lzma.decompress(packed))


def stage(repository, ergo, root):
    root.mkdir(parents=True, exist_ok=False)
    old = published(repository, "provenance")
    inputs = published(repository, "inputs")["files"]
    (root / "inputs").mkdir()
    for method in METHODS:
        for role in ("auxiliary", "reference"):
            name = f"inputs/{method}-{role}.json"
            raw = plain(inputs[name]["value"])
            if digest(raw) != inputs[name]["sha256"]:
                raise ValueError(f"frozen input mismatch {name}")
            (root / name).write_bytes(raw)
    regenerated = generate(ergo)
    frozen_bytes = lzma.decompress((HERE / "geometries.json.xz").read_bytes())
    frozen = json.loads(frozen_bytes)
    for size, case in regenerated["cases"].items():
        for key in ("source_molecule_indices_zero_based", "geometry_bohr"):
            if case[key] != frozen["cases"][size][key]:
                raise ValueError(f"frozen geometry mismatch {size}: {key}")
    (root / "geometries.json").write_bytes(frozen_bytes)
    output = root / "source.tar.gz"
    subprocess.run(
        [
            "git",
            "-C",
            repository,
            "archive",
            "--format=tar.gz",
            "-o",
            output,
            REVISION,
            "--",
            ".",
            *[f":!{path}" for path in EXCLUSIONS],
        ],
        check=True,
        env={**os.environ, "COPYFILE_DISABLE": "1"},
    )
    with tarfile.open(output) as archive:
        for member in archive.getmembers():
            if any(
                part == ".env" or part.startswith((".env.", "._"))
                for part in Path(member.name).parts
            ):
                raise ValueError(f"disallowed archive member {member.name}")
        archive.extractall(root / "source", filter="data")
    tree = subprocess.check_output(
        ["git", "-C", repository, "rev-parse", f"{REVISION}^{{tree}}"], text=True
    ).strip()
    identity, count = source_identity(root / "source")
    source = {
        "revision": REVISION,
        "tree": tree,
        "archive_exclusions": list(EXCLUSIONS),
        "source_archive_sha256": digest(output.read_bytes()),
        "native_source_identity": identity,
        "native_identity_inputs": count,
        "paper_runner_sha256": digest(
            (root / "source/benchmarks/readme_defaults.py").read_bytes()
        ),
        "cpu_oracle_driver_sha256": digest(
            (root / "source/benchmarks/readme_defaults_accuracy.py").read_bytes()
        ),
        "geometry_sha256": digest(frozen_bytes),
        "inputs_sha256": {
            str(path.relative_to(root)): digest(path.read_bytes())
            for path in sorted((root / "inputs").glob("*.json"))
        },
        "historical_inputs_revision": COHORT_REVISION,
    }
    (root / "source.json").write_bytes(plain(source))
    (root / "paper-config.json").write_bytes(
        plain(old["reacquisition_launchers"]["campaign-config"])
    )
    for name in (
        "acquire.py",
        "build.py",
        "diagnose.py",
        "endpoint.py",
        "geometry.py",
        "run_cpu.sh",
    ):
        shutil.copyfile(HERE / name, root / name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", type=Path, required=True)
    parser.add_argument("--ergo-archive", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    stage(args.repository.resolve(), args.ergo_archive.resolve(), args.root.resolve())
