"""Freeze an already installed owned scientific Python environment, without GPU work."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import sys
from pathlib import Path


def sha256(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            value.update(block)
    return value.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    if not __import__("os").environ.get("SLURM_JOB_ID"):
        raise RuntimeError("requires real owned CPU Slurm allocation")
    sys.dont_write_bytecode = True
    root = Path(sys.prefix).resolve()
    expected = {
        "numpy": "2.4.6",
        "pyscf": "2.14.0",
        "cupy-cuda12x": "13.6.0",
        "gpu4pyscf-cuda12x": "1.8.1",
        "sccache": "0.16.0",
    }
    packages = {
        distribution.metadata["Name"]: distribution.version
        for distribution in importlib.metadata.distributions()
    }
    for name, version in expected.items():
        if importlib.metadata.version(name) != version:
            raise ValueError(f"unexpected scientific dependency version: {name}")
    from pyscf.dft import libxc

    files = []
    for path in sorted(root.rglob("*")):
        if (
            path.is_symlink()
            or not path.is_file()
            or "__pycache__" in path.parts
            or path.suffix == ".pyc"
        ):
            continue
        if ".env" in path.parts or path.name.startswith(".env."):
            raise ValueError("refuse dotenv input")
        files.append(
            {
                "path": path.relative_to(root).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
                "executable": bool(path.stat().st_mode & 0o111),
            }
        )
    receipt = {
        "schema": "generativeqc.runtime-cache-state.environment.v1",
        "python_version": sys.version,
        "python_executable": sys.executable,
        "python_executable_sha256": sha256(Path(sys.executable).resolve()),
        "root": str(root),
        "packages": packages,
        "libxc_version": libxc.__version__,
        "files": files,
        "scope": "scientific Python only; native GenerativeQC SDK and GPU qualification INCOMPLETE",
    }
    args.output.write_text(
        json.dumps(receipt, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(
        json.dumps(
            {
                "files": len(files),
                "sha256": sha256(args.output),
                "libxc_version": libxc.__version__,
            }
        )
    )


if __name__ == "__main__":
    main()
