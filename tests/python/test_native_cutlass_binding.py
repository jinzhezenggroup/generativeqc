"""Optional AOT provider qualification with real build and semantic provenance."""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import pytest
from generativeqc_compiler.common.paths import source_hashes
from generativeqc_compiler.common.provenance import (
    atomic_json,
    canonical_hash,
    file_hash,
    toolchain_identity,
)


def test_native_cutlass_execution(tmp_path: Path) -> None:
    """Compile the fixed SIMT family and compare every layout to affine sums."""
    if os.environ.get("GENERATIVEQC_CUTLASS_CUDA_TEST") != "1":
        pytest.skip("requires optional CUTLASS and finite Slurm real-GPU allocation")
    assert os.environ.get("SLURM_JOB_ID")
    compiler, cache = shutil.which("nvcc"), shutil.which("ccache")
    assert compiler and cache
    subprocess.run([cache, "--version"], check=True, capture_output=True)
    root = Path(__file__).resolve().parents[2]
    sdk = Path(os.environ["GENERATIVEQC_CUTLASS_ROOT"]).resolve() / "include"
    assert (sdk / "cutlass/gemm/device/gemm_batched.h").is_file()
    flags = ["-std=c++20", "-O2", "-arch=sm_120"]
    # Actual external header content is part of this artifact. Neither an
    # installation path nor a version string alone proves the compiled inputs.
    identity = {
        "toolchain": toolchain_identity(Path(compiler)),
        "flags": flags,
        "sources": source_hashes(
            "common",
            "tensor",
            assets=(
                "tests/native/test_native_cutlass.cu",
                *sorted(
                    path.relative_to(root).as_posix()
                    for directory in (root / "src/tensor", root / "src/runtime")
                    for path in directory.rglob("*")
                    if path.is_file() and path.suffix in (".hpp", ".cuh")
                ),
            ),
        ),
        "cutlass_headers": {
            path.relative_to(sdk).as_posix(): file_hash(path)
            for path in sorted(sdk.rglob("*"))
            if path.is_file()
        },
    }
    artifact = canonical_hash(identity)
    atomic_json(tmp_path / "artifact.json", {"identity": identity, "key": artifact})
    obj, executable = tmp_path / "cutlass.o", tmp_path / "cutlass"
    for command in (
        [
            cache,
            compiler,
            *flags,
            "-I" + str(root / "src"),
            "-I" + str(sdk),
            "-c",
            str(root / "tests/native/test_native_cutlass.cu"),
            "-o",
            str(obj),
        ],
        [compiler, str(obj), "-arch=sm_120", "-o", str(executable)],
        [str(executable), artifact],
    ):
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=240,
            check=False,
            env={**os.environ, "CCACHE_BASEDIR": str(root)},
        )
        assert result.returncode == 0, result.stdout + result.stderr
        print(result.stdout, end="")
    atomic_json(
        tmp_path / "qualified.json",
        {"artifact": artifact, "executable_sha256": file_hash(executable)},
    )
