"""Inspect the actual DF class objects offline, without initializing CUDA.

This opt-in experiment records machine instruction hashes as migration evidence;
they are not permanent source/implementation fingerprints required by CI.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--build", type=Path, required=True)
    parser.add_argument("--cuobjdump", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    sys.path[:0] = [str(args.source / "python"), str(args.source)]
    from tools.validate_direct_jk_aot_resources import cuobjdump_resources

    objects = sorted(
        (args.build / "CMakeFiles/generativeqc.dir/generated/df_shells").glob("*.cu.o")
    )
    assert len(objects) == 80, len(objects)
    rows = []
    for obj in objects:
        resource_output = subprocess.run(
            [str(args.cuobjdump), "--dump-resource-usage", str(obj)],
            capture_output=True,
            text=True,
            check=True,
            timeout=60,
        ).stdout
        resources = cuobjdump_resources(resource_output)
        sass_output = subprocess.run(
            [str(args.cuobjdump), "--dump-sass", str(obj)],
            capture_output=True,
            text=True,
            check=True,
            timeout=120,
        ).stdout
        instructions = {}
        for block in re.split(r"\bFunction : ", sass_output)[1:]:
            name = block.splitlines()[0].strip()
            words = re.findall(r"/\*\s*(0x[0-9a-f]+)\s*\*/", block)
            if words:
                instructions[name] = {
                    "instruction_bytes": len(words) * 8,
                    "sha256": hashlib.sha256("\n".join(words).encode()).hexdigest(),
                }
        kernels = []
        for name, values in resources.items():
            key = re.search(r"shell_(?:panel|packet)I(?:Lj\d+E){4}(?:Lb[01]E){3}", name)
            if key:
                assert name in instructions, name
                kernels.append(
                    {
                        "key": key[0],
                        "symbol": name,
                        "resources": values,
                        "instructions": instructions[name],
                    }
                )
        assert kernels, obj
        rows.append(
            {
                "object": obj.name,
                "bytes": obj.stat().st_size,
                "sha256": sha256(obj),
                "kernels": kernels,
            }
        )
    payload = {
        "schema": "generativeqc.df-schedule-object-census.v1",
        "objects": rows,
        "object_bytes": sum(row["bytes"] for row in rows),
        "kernel_count": sum(len(row["kernels"]) for row in rows),
        "source_inputs": {
            name: sha256(args.source / name)
            for name in (
                "src/scf/cuda/df_shell_kernel.cuh",
                "src/scf/cuda/df_shell_launch.cuh",
                "python/generativeqc_compiler/integral/df_shell_derivatives.py",
            )
        },
    }
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps({k: payload[k] for k in ("object_bytes", "kernel_count")}))


if __name__ == "__main__":
    main()
