"""Rebuild frozen CPU/CUDA source identities without compiling or using a GPU.

This opt-in historical audit requires the published ancestor locally. It never
fetches history or modifies the current checkout. Numerical reduction is separate
and remains usable without Git history or source generation dependencies.
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
    """Run inside the reconstructed tree, with fresh producer import caches."""
    from tools import generate_libxc_semilocal_cpu_registry as cpu
    from tools import generate_libxc_semilocal_cuda_registry as cuda

    expected = json.loads(Path("expected.json").read_text())
    entries = cuda.select_entries(expected["names"])
    actual = {}
    for arm in ("literal", "smart", "cpu"):
        directory = Path("generated") / arm
        directory.mkdir(parents=True)
        backend = cpu if arm == "cpu" else cuda
        prefix = "generated_libxc_semilocal" + ("" if arm == "cpu" else "_cuda")
        sources = {
            prefix + "_registry.hpp": backend.emit_header(),
            prefix + "_registry.cpp": backend.emit_registry(),
        }
        count = cpu.SHARD_COUNT if arm == "cpu" else 32
        for index in range(count):
            if arm == "cpu":
                source = cpu.emit_shard(index, cpu.registry_entries())
            else:
                source = cuda.emit_shard(
                    index,
                    entries,
                    share_parameters=arm == "smart",
                    parameter_policy="discover",
                )
            sources[f"{prefix}_{index}.{'cpp' if arm == 'cpu' else 'cu'}"] = source
        actual[arm] = {}
        for name, source in sources.items():
            (directory / name).write_text(source)
            actual[arm][name] = hashlib.sha256(source.encode()).hexdigest()
        assert actual[arm] == expected[arm], arm
        print(f"{arm}: {len(sources)} generated source identities match", flush=True)
    Path("verified.json").write_text(json.dumps(actual, indent=2) + "\n")


def main() -> None:
    """Restore exact producer overlays onto the published base, then generate."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--snapshot",
        choices=("accepted", "structural", "publication", "rejected"),
        default="accepted",
    )
    args = parser.parse_args()
    with lzma.open(Path(__file__).with_name("receipts.json.xz"), "rt") as stream:
        records = json.load(stream)
    old, new = records["structural_discovery"], records["paired_batch"]
    reconstruction = records["source_reconstruction"]
    overlay = dict(old["publication_sources"])
    if args.snapshot in ("structural", "publication"):
        overlay.update(reconstruction["structural_base_overrides"])
        if args.snapshot == "structural":
            overlay.update(old["measured_sources"])
        prefix = "publication_" if args.snapshot == "publication" else ""
        expected = dict(old[prefix + "generated"], cpu=old[prefix + "cpu_generated"])
    else:
        accepted = args.snapshot == "accepted"
        overlay.update(new["production" if accepted else "rejected_production"])
        prefix = "accepted/" if accepted else "final/"
        expected = {
            arm: new["packages"][prefix + arm]["generated"]
            for arm in ("literal", "smart")
        }
        expected["cpu"] = new["cpu_generated" if accepted else "rejected_cpu_generated"]
    expected["names"] = sorted(
        name for group in new["groups"] for name in group["names"]
    )
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(output)
    repository = Path(__file__).resolve().parents[3]
    # Archive only reconstruction inputs; unrelated historical evidence is not
    # required, and Git must not implicitly fetch missing promisor objects.
    environment = dict(os.environ, GIT_NO_LAZY_FETCH="1")
    archive = subprocess.check_output(
        [
            "git",
            "archive",
            reconstruction["published_base_commit"],
            "python",
            "tools",
            "src",
            "include",
            "upstream",
            "manifests",
        ],
        cwd=repository,
        env=environment,
        timeout=60,
    )
    output.mkdir(parents=True)
    with tarfile.open(fileobj=io.BytesIO(archive)) as stream:
        stream.extractall(output, filter="data")
    for name, source in overlay.items():
        target = output / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(source)
    if args.snapshot == "accepted":
        for name, digest in new["compiler_hashes"].items():
            assert hashlib.sha256((output / name).read_bytes()).hexdigest() == digest, (
                name
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
