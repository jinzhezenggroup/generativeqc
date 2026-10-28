"""Opt-in reconstruction of frozen producer and CPU/CUDA generated sources.

The published base must already exist locally. This audit neither fetches Git
history nor modifies the checkout; normal numerical replay uses only reduce.py.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import lzma
import os
import subprocess
import sys
import tarfile
from pathlib import Path


def generate() -> None:
    """Generate both registries using the default policy from the frozen tree."""
    from tools import generate_libxc_semilocal_cpu_registry as cpu
    from tools import generate_libxc_semilocal_cuda_registry as cuda
    from tools.generate_build_identity import _inventory, _source_identity

    root = Path.cwd()
    expected = json.loads(Path("expected.json").read_text())
    identity = _source_identity(
        root, _inventory(root, root / "cmake/GenerativeQCSourceIdentity.json")
    )
    assert identity in expected["build_identity"]
    entries = cuda.select_entries(expected["names"])
    actual = {}
    for arm in ("literal", "smart", "cpu"):
        backend = cpu if arm == "cpu" else cuda
        prefix = "generated_libxc_semilocal" + ("" if arm == "cpu" else "_cuda")
        sources = {
            prefix + "_registry.hpp": backend.emit_header(),
            prefix + "_registry.cpp": backend.emit_registry(),
        }
        for index in range(cpu.SHARD_COUNT if arm == "cpu" else 32):
            source = (
                cpu.emit_shard(index, cpu.registry_entries())
                if arm == "cpu"
                else cuda.emit_shard(index, entries, share_parameters=arm == "smart")
            )
            sources[f"{prefix}_{index}.{'cpp' if arm == 'cpu' else 'cu'}"] = source
        directory = root / "generated" / arm
        directory.mkdir(parents=True)
        actual[arm] = {}
        for name, source in sources.items():
            (directory / name).write_text(source)
            actual[arm][name] = hashlib.sha256(source.encode()).hexdigest()
        assert actual[arm] == expected[arm], arm
        print(f"{arm}: {len(sources)} source identities match", flush=True)
    (root / "verified.json").write_text(json.dumps(actual, indent=2) + "\n")


def main() -> None:
    """Restore published inputs plus exact measured overlays in a new directory."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    with lzma.open(Path(__file__).with_name("receipts.json.xz"), "rt") as stream:
        data = json.load(stream)
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(output)
    source = data["source"]
    environment = dict(os.environ, GIT_NO_LAZY_FETCH="1")
    archive = subprocess.check_output(
        ["git", "archive", source["published_base"], *source["roots"]],
        cwd=Path(__file__).resolve().parents[3],
        env=environment,
        timeout=60,
    )
    output.mkdir(parents=True)
    with tarfile.open(fileobj=io.BytesIO(archive)) as stream:
        stream.extractall(output, filter="data")
    for name, contents in source["overlays"].items():
        target = output / name
        if contents is None:
            target.unlink()
        else:
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(contents)
    expected = {arm: package["generated"] for arm, package in data["packages"].items()}
    expected.update(
        cpu=data["cpu_generated"],
        names=data["names"],
        build_identity=data["build_identity"],
    )
    (output / "expected.json").write_text(json.dumps(expected))
    (output / "reproduce_generated.py").write_text(Path(__file__).read_text())
    environment["PYTHONPATH"] = os.pathsep.join((str(output / "python"), str(output)))
    subprocess.run(
        [sys.executable, "-c", "from reproduce_generated import generate; generate()"],
        cwd=output,
        env=environment,
        check=True,
        timeout=600,
    )


if __name__ == "__main__":
    main()
