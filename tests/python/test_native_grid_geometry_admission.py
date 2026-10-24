"""Execute combined topology/work admission without CUDA initialization."""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from conftest import NativeCxx

ROOT = Path(__file__).resolve().parents[2]


def _translation_unit(source: str) -> str:
    host = source[: source.index("unsigned blocks(")]
    host = host.replace('#include "response/grid_geometry_response.cuh"', "")
    host = host.replace("namespace ad = generativeqc_grid_geometry_ad;", "")
    start = source.index("cudaError_t cuda_grid_geometry_workspace(")
    workspace = source[
        start : source.index(
            "cudaError_t CudaGridGeometryResponse::authenticate(", start
        )
    ]
    return (
        "#include <cassert>\n"
        + host
        + "}\n"
        + workspace
        + "}\n"
        + r"""
int main() {
  using namespace generativeqc::dft;
  CudaGridGeometryOptions options;
  CudaGridGeometryShape shape{2, 4, 2, 8, 0, 4};
  options.max_ao_primitive_terms = 20 * shape.points * shape.ao_primitive_terms;
  std::size_t nonuniform{}, uniform{};
  assert(cuda_grid_geometry_workspace(shape, options, nonuniform) == cudaSuccess);
  shape.points_per_atom = 4;
  assert(cuda_grid_geometry_workspace(shape, options, uniform) == cudaSuccess);
  assert(uniform == nonuniform);
  shape.points_per_atom = 3;
  assert(cuda_grid_geometry_workspace(shape, options, uniform) == cudaErrorInvalidValue);
  assert(uniform == 0);
  shape.points_per_atom = 0;
  --options.max_ao_primitive_terms;
  assert(cuda_grid_geometry_workspace(shape, options, uniform) == cudaErrorInvalidValue);
  ++options.max_ao_primitive_terms;
  options.max_workspace_bytes = nonuniform - 1;
  assert(cuda_grid_geometry_workspace(shape, options, uniform) == cudaErrorInvalidValue);
  options.max_workspace_bytes = nonuniform;
  assert(cuda_grid_geometry_workspace(shape, options, uniform) == cudaSuccess);
  shape.ao_primitive_terms = 0;
  assert(cuda_grid_geometry_workspace(shape, options, uniform) == cudaErrorInvalidValue);
  options.max_ao_primitive_terms = 20 * shape.points * shape.nao * shape.primitives * 3;
  assert(cuda_grid_geometry_workspace(shape, options, uniform) == cudaSuccess);
  assert(uniform == nonuniform);
}
"""
    )


def test_ao_incidence_bound_and_nonuniform_grid_compose(
    tmp_path: Path, native_cxx: NativeCxx
) -> None:
    source = ROOT / "src/dft/cuda_grid_geometry.cu"
    fixture = tmp_path / "geometry_admission.cpp"
    fixture.write_text(_translation_unit(source.read_text()))
    executable = native_cxx.build_executable(
        [fixture],
        tmp_path / "probe",
        compile_args=(
            "-std=c++20",
            "-O0",
            f"-I{ROOT / 'tests/native/fixtures/hessian_budget'}",
            f"-I{ROOT / 'src'}",
        ),
    )
    result = subprocess.run(
        [str(executable)], capture_output=True, text=True, timeout=30, check=False
    )
    assert result.returncode == 0, result.stdout + result.stderr
