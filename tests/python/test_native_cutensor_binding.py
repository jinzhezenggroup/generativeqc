"""Optional real-device cuTENSOR qualification, explicitly scheduled by the caller."""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import pytest
from _cc_owner_test_support import compile_owner
from generativeqc_compiler.tensor import (
    Index,
    IndexSpace,
    Program,
    TensorSpec,
    einsum,
    input_tensor,
)
from generativeqc_compiler.tensor.lowering import TensorLoweringAdapter
from generativeqc_compiler.tensor.native_lowering import affine_contraction_initializer


def test_general_affine_projection_retains_original_semantics(tmp_path: Path) -> None:
    """A real TensorIR projection validates independently of any GPU provider."""
    compiler = shutil.which("c++")
    if compiler is None:
        pytest.skip("host C++ compiler required")
    i, j, k, l = (
        Index(name, IndexSpace(name, "batch", size))
        for name, size in zip("ijkl", (2, 5, 3, 4), strict=True)
    )
    a = input_tensor("a", TensorSpec((i, k, l), role="input"))
    b = input_tensor("b", TensorSpec((l, k, j), role="input"))
    node = einsum("ikl,lkj->ji", a, b)
    adapter = TensorLoweringAdapter(Program({"result": node}))
    request = adapter.request(node, backend="cuda")
    initializer = affine_contraction_initializer(
        adapter, node, lambda axis: str(axis.extent), coefficient="1.0"
    )
    assert request.semantic_identity in initializer
    source = tmp_path / "affine.cpp"
    source.write_text(
        '#include "tensor/native_contraction.hpp"\nint main(){ auto request='
        + initializer
        + ";request.validate_affine();if(request.affine_summands()!=120) return 1;"
        + "try{request.validate();}catch(const std::exception&){return 0;}return 2;}\n"
    )
    executable = tmp_path / "affine"
    compile_owner(compiler, tmp_path, [source], executable)
    subprocess.run([str(executable)], check=True, capture_output=True, text=True)


def test_prepared_affine_cutensor_execution(tmp_path: Path) -> None:
    """Exercise physical modes, independent arithmetic and bounded replay state."""
    if os.environ.get("GENERATIVEQC_CUTENSOR_CUDA_TEST") != "1":
        pytest.skip("requires explicit finite Slurm real-device qualification")
    compiler, cache = shutil.which("nvcc"), shutil.which("ccache")
    provider_root = os.environ.get("GENERATIVEQC_CUTENSOR_ROOT")
    if compiler is None or cache is None or provider_root is None:
        pytest.fail(
            "cuTENSOR qualification requires nvcc, ccache and GENERATIVEQC_CUTENSOR_ROOT"
        )
    provider = Path(provider_root)
    library = next(iter(sorted((provider / "lib").glob("libcutensor.so*"))), None)
    assert library is not None, "cuTENSOR shared library is missing"
    root = Path(__file__).resolve().parents[2]
    subprocess.run([cache, "--version"], check=True, capture_output=True)
    obj, executable = tmp_path / "cutensor.o", tmp_path / "cutensor"
    build = subprocess.run(
        [
            cache,
            compiler,
            "-std=c++20",
            "-O2",
            "-arch=sm_120",
            "-DGENERATIVEQC_HAS_CUTENSOR=1",
            "-I" + str(root / "src"),
            "-I" + str(provider / "include"),
            "-c",
            str(root / "tests/native/test_native_cutensor.cu"),
            "-o",
            str(obj),
        ],
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
        env={**os.environ, "CCACHE_BASEDIR": str(root)},
    )
    assert build.returncode == 0, build.stdout + build.stderr
    link = subprocess.run(
        [
            compiler,
            str(obj),
            "-arch=sm_120",
            "-lcublas",
            "-Xlinker",
            str(library),
            "-Xlinker",
            "-rpath",
            "-Xlinker",
            str(provider / "lib"),
            "-o",
            str(executable),
        ],
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert link.returncode == 0, link.stdout + link.stderr
    run = subprocess.run(
        [str(executable)], capture_output=True, text=True, timeout=90, check=False
    )
    assert run.returncode == 0, run.stdout + run.stderr
