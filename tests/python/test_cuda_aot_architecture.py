"""AOT source identities and CUDA device-image requests have distinct lifetimes."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path
from typing import Any

import pytest
from generativeqc_compiler.common.cuda_adapter import CudaCompilerAdapter
from generativeqc_compiler.common.cuda_target import cuda_target_info
from generativeqc_compiler.common.native_runtime import (
    compile_cuda_object,
    link_cuda_objects,
)

ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize(
    ("requested", "expected"),
    [
        ("120-real;120-virtual", "120|120-real;120-virtual"),
        ("120-virtual;120-real;120-real", "120|120-real;120-virtual"),
        ("120-virtual;120;120-real", "120|120"),
        ("120-real", "120|120-real"),
        ("120-virtual", "120|120-virtual"),
        ("120-virtual;80-real;120-real", "80;120|80-real|120-real;120-virtual"),
    ],
)
def test_source_groups_preserve_every_requested_device_image(
    tmp_path: Path, requested: str, expected: str
) -> None:
    """Evaluate the actual CMake owner without a compiler or GPU dependency."""
    output = tmp_path / "result.txt"
    script = tmp_path / "group.cmake"
    script.write_text(
        "cmake_minimum_required(VERSION 3.24)\n"
        f'include("{ROOT / "cmake/GenerativeQCCuda.cmake"}")\n'
        # Repeated use must not inherit the preceding invocation's image list.
        "generativeqc_group_cuda_architectures(groups 120 80)\n"
        f"generativeqc_group_cuda_architectures(groups {requested})\n"
        'set(result "${groups}")\n'
        "foreach(architecture IN LISTS groups)\n"
        '  string(APPEND result "|${groups_${architecture}}")\n'
        "endforeach()\n"
        f'file(WRITE "{output}" "${{result}}")\n'
    )
    subprocess.run(["cmake", "-P", str(script)], check=True, timeout=30)
    assert output.read_text() == expected


@pytest.mark.parametrize("requested", ["native", "all", "sm_120", "120-invalid"])
def test_unresolved_source_architectures_are_rejected(
    tmp_path: Path, requested: str
) -> None:
    script = tmp_path / "reject.cmake"
    script.write_text(
        "cmake_minimum_required(VERSION 3.24)\n"
        f'include("{ROOT / "cmake/GenerativeQCCuda.cmake"}")\n'
        f"generativeqc_group_cuda_architectures(groups {requested})\n"
    )
    result = subprocess.run(
        ["cmake", "-P", str(script)],
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode != 0
    assert "requires numeric CUDA architectures" in result.stderr


def test_same_source_keeps_distinct_real_cuda_objects_and_links(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Opt-in NVCC qualification; compile/link only, with no device execution.

    The real compiler adapter selects and verifies its compiler-cache launcher.
    Both object and linked-runtime caches must retain the target witness even
    when the generated source path and bytes are shared.
    """
    requested = os.environ.get("GENERATIVEQC_NVCC")
    if not requested:
        pytest.skip("set GENERATIVEQC_NVCC to qualify actual multiarch binaries")
    source = tmp_path / "shared.cu"
    source.write_text(
        'extern "C" __global__ void architecture(int* out) {\n'
        "#ifdef __CUDA_ARCH__\n"
        "  *out = __CUDA_ARCH__;\n"
        "#endif\n}\n"
    )
    compilers = [
        CudaCompilerAdapter(Path(requested), cuda_target_info(architecture))
        for architecture in ("sm_80", "sm_120")
    ]
    objects = [compile_cuda_object(c, tmp_path, source) for c in compilers]
    linked = [
        link_cuda_objects(c, tmp_path, (obj,))
        for c, obj in zip(compilers, objects, strict=True)
    ]
    for products in (objects, linked):
        assert products[0].metadata["key"] != products[1].metadata["key"]
        assert (
            products[0].metadata["binary_sha256"]
            != products[1].metadata["binary_sha256"]
        )
        assert [p.metadata["identity"]["target"]["architecture"] for p in products] == [
            "sm_80",
            "sm_120",
        ]

    def forbidden(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("binary cache hit attempted compilation")

    monkeypatch.setattr(CudaCompilerAdapter, "_run_compiler", forbidden)
    for compiler, obj, library in zip(compilers, objects, linked, strict=True):
        assert compile_cuda_object(compiler, tmp_path, source) == obj
        assert link_cuda_objects(compiler, tmp_path, (obj,)) == library
