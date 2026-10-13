"""Compile the whole borrowed CUDA view boundary in runtime-supported C++17.

Host recording providers expose GEAM or omit it. These tests cover capability,
ABI forwarding and error provenance, not device numerics or CUDA execution.
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import TYPE_CHECKING

import pytest
from _cpp_source_support import cpp_function_definition, cpp_record_definition

if TYPE_CHECKING:
    from conftest import NativeCxx

ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize("standard", ("c++17", "c++20"))
@pytest.mark.parametrize("has_geam", (False, True))
def test_borrowed_cuda_view_language_contract(
    tmp_path: Path, required_native_cxx: NativeCxx, standard: str, has_geam: bool
) -> None:
    source = (ROOT / "src/tensor/cuda_runtime.cuh").read_text()
    first = cpp_record_definition(source, "ColumnGemmView")
    last = cpp_function_definition(source, "submit_column_geam", include_template=True)
    # Keep the entire contiguous provider section, including capability declarations
    # that selective function extraction would otherwise leave uncompiled.
    start = source.index(first)
    end = source.index(last) + len(last)
    (tmp_path / "borrowed_tensor_views.inc").write_text(source[start:end])
    executable = required_native_cxx.build_executable(
        [ROOT / "tests/native/test_tensor_borrowed_cuda_views.cpp"],
        tmp_path / "probe",
        compile_args=(
            f"-std={standard}",
            "-O0",
            "-Wall",
            "-Wextra",
            "-Werror",
            "-pedantic-errors",
            "-I" + str(tmp_path),
            *(("-DGENERATIVEQC_TEST_HAS_GEAM",) if has_geam else ()),
        ),
    )
    subprocess.run([str(executable)], check=True, timeout=15)
