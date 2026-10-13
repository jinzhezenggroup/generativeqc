"""Execute DF's borrowed FP64 submission with a numerical host cuBLAS double.

This covers host routing, storage, admission and error mapping, not real-device
numerics, CUDA Graph capture or timing. Only the CUDA-dependent adapter body and
its existing error mapper are extracted; neither is reimplemented in the probe.
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


@pytest.fixture(scope="module")
def product_probe(
    tmp_path_factory: pytest.TempPathFactory, required_native_cxx: NativeCxx
) -> Path:
    folder = tmp_path_factory.mktemp("df-matrix-product")
    bodies = []
    for path, name in (
        ("src/scf/cuda/df_runtime.cpp", "blas_failure"),
        ("src/scf/cuda/df_scf_library.cpp", "scf_matrix_product_view"),
    ):
        bodies.append(cpp_function_definition((ROOT / path).read_text(), name))
    header = (ROOT / "src/scf/cuda/df_scf_library.hpp").read_text()
    bodies.insert(0, cpp_record_definition(header, "ScfMatrixBatchView") + ";")
    for name in (
        "scf_matrix_product_batched_view",
        "scf_occupied_gram",
        "prepare_df_blas",
        "release_df_blas",
    ):
        bodies.append(
            cpp_function_definition(
                (ROOT / "src/scf/cuda/df_scf_library.cpp").read_text(), name
            )
        )
    (folder / "df_matrix_product.inc").write_text("\n".join(bodies))
    tensor = (ROOT / "src/tensor/cuda_runtime.cuh").read_text()
    response_bodies = []
    for name in ("ColumnGemmView", "ColumnBatchedGemmView", "ColumnGemvView"):
        response_bodies.append(cpp_record_definition(tensor, name) + ";")
    response_bodies.append(
        "template <class Handle>\n"
        + cpp_record_definition(tensor, "ColumnGeamView")
        + ";"
    )
    for name in (
        "submit_column_gemm",
        "submit_column_batched_gemm",
        "submit_column_gemv",
        "submit_column_geam",
    ):
        response_bodies.append(
            cpp_function_definition(
                tensor, name, include_template=(name == "submit_column_geam")
            )
        )
    (folder / "borrowed_response_ops.inc").write_text(
        "namespace generativeqc_tensor {\n" + "\n".join(response_bodies) + "\n}\n"
    )
    (folder / "cuda_runtime_api.h").write_text(
        "#pragma once\nstruct cudaStream; using cudaStream_t = cudaStream*;\n"
    )
    return required_native_cxx.build_executable(
        [ROOT / "tests/native/test_df_matrix_product_view.cpp"],
        folder / "probe",
        compile_args=(
            "-std=c++20",
            "-O0",
            "-Wall",
            "-Wextra",
            "-Werror",
            "-I" + str(ROOT / "tests/native/fixtures/shared_eigen_provider"),
            "-I" + str(ROOT / "include"),
            "-I" + str(folder),
        ),
    )


@pytest.mark.parametrize(
    "mode",
    (
        "products",
        "invalid",
        "failures",
        "batched",
        "batched_overlap",
        "batched_overflow",
        "batched_invalid",
        "gram",
        "owner",
        "borrowed_response",
    ),
)
def test_df_matrix_product_contract(product_probe: Path, mode: str) -> None:
    subprocess.run([str(product_probe), mode], check=True, timeout=15)
