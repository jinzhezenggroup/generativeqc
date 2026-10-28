"""Canonical hashes and atomic publication shared with local profiles (#136).

These are the original profile helpers: moving ownership does not change the
JSON encoding, durability guarantees, or toolchain/cache compatibility policy.
"""

import hashlib
import json
import os
import shutil
import subprocess
import tempfile
import typing
from pathlib import Path


def canonical_hash(value: typing.Any) -> str:
    """Hash portable JSON with no non-finite numbers or path-dependent encoding."""
    return hashlib.sha256(
        json.dumps(
            value, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
    ).hexdigest()


def file_hash(path: Path) -> str:
    """Stream binary hashes without retaining a complete CUDA library in RAM."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def find_nvcc() -> Path | None:
    """Find a requested toolkit without guessing a different compiler version."""
    # An explicit but unavailable compiler/root must not silently select a
    # different toolkit from PATH. Root aliases support non-system installs.
    candidates = [os.environ.get("GENERATIVEQC_NVCC"), os.environ.get("CUDACXX")]
    candidates.extend(
        str(Path(root).expanduser() / "bin/nvcc") if root else None
        for root in (
            os.environ.get("CUDAToolkit_ROOT"),
            os.environ.get("CUDA_PATH"),
            os.environ.get("CUDA_HOME"),
        )
    )
    candidates.append("nvcc")
    for candidate in candidates:
        if candidate:
            resolved = shutil.which(str(Path(candidate).expanduser()))
            return Path(resolved).resolve() if resolved else None
    return None


def toolchain_identity(nvcc: Path) -> dict:
    """Record the exact compiler/assembler versions that generated the binary."""
    result = {}
    for name in ("nvcc", "ptxas"):
        output = subprocess.run(
            [str(nvcc.with_name(name)), "--version"],
            check=True,
            capture_output=True,
            text=True,
            timeout=30,
        )
        result[name] = (output.stdout + output.stderr).strip()
    return result


def atomic_json(path: Path, payload: dict) -> None:
    """Publish a complete file with same-filesystem replacement and fsync."""
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=".pending-", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w") as stream:
            json.dump(payload, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)
