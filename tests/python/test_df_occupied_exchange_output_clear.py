"""Run the occupied caller and provider with host CUDA/cuBLAS doubles.

Validate output overwrite, padding, layout and clear traffic without pinning
vendor-call spelling. This is not real-device or integral-source qualification.
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import TYPE_CHECKING

import pytest
from _cpp_source_support import cpp_function_definition, cpp_record_definition
from test_df_matrix_product_view import write_borrowed_response_ops

if TYPE_CHECKING:
    from conftest import NativeCxx

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def occupied_probe(
    tmp_path_factory: pytest.TempPathFactory, required_native_cxx: NativeCxx
) -> Path:
    folder = tmp_path_factory.mktemp("occupied-exchange-host")
    write_borrowed_response_ops(folder)
    header = (ROOT / "src/scf/cuda/df_scf_library.hpp").read_text()
    provider = (ROOT / "src/scf/cuda/df_scf_library.cpp").read_text()
    bodies = [cpp_record_definition(header, "ScfMatrixBatchView") + ";"]
    bodies.append(
        cpp_function_definition(
            (ROOT / "src/scf/cuda/df_runtime.cpp").read_text(), "blas_failure"
        )
    )
    for name in (
        "scf_matrix_product_view",
        "scf_matrix_product_batched_view",
        "scf_occupied_gram",
        "prepare_df_blas",
        "release_df_blas",
    ):
        bodies.append(cpp_function_definition(provider, name))
    (folder / "df_matrix_product.inc").write_text("\n".join(bodies))
    (folder / "df_occupied_exchange.inc").write_text(
        cpp_function_definition(
            (ROOT / "src/scf/cuda/df_occupied_exchange.cpp").read_text(),
            "build_occupied_exchange",
        )
    )
    (folder / "cuda_runtime_api.h").write_text(
        "#pragma once\nstruct cudaStream; using cudaStream_t = cudaStream*;\n"
    )
    return required_native_cxx.build_executable(
        [ROOT / "tests/native/test_df_occupied_exchange_host.cpp"],
        folder / "probe",
        compile_args=(
            "-std=c++20",
            "-Wall",
            "-Wextra",
            "-Werror",
            "-I" + str(ROOT / "tests/native/fixtures/shared_eigen_provider"),
            "-I" + str(ROOT / "include"),
            "-I" + str(ROOT / "src"),
            "-I" + str(folder),
        ),
    )


@pytest.mark.parametrize(
    "mode",
    ("resident", "tiled", "streamed", "zero", "projected", "projected_failure", "raw"),
)
def test_occupied_exchange_output_contract(occupied_probe: Path, mode: str) -> None:
    subprocess.run([str(occupied_probe), mode], check=True, timeout=15)
