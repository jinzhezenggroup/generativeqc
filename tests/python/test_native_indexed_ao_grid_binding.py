"""Device-free parity and rejection gates for the emitted native CSR binding."""

from __future__ import annotations

import os
import shutil
import subprocess
from typing import TYPE_CHECKING

import pytest
from generativeqc_compiler.dft.ao_cuda import emit_grid_source
from generativeqc_compiler.dft.indexed_layout import AoGridBlockLayout
from generativeqc_compiler.dft.indexed_layout_native import emit_native_ao_grid_binding

if TYPE_CHECKING:
    from conftest import NativeCxx

if TYPE_CHECKING:
    from pathlib import Path


@pytest.fixture(scope="module")
def native_binding_probe(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Compile only generated host metadata, never CUDA or a numerical oracle."""
    compiler, cache = shutil.which("c++"), shutil.which("ccache")
    if compiler is None or cache is None:
        pytest.skip("requires c++ and ccache")
    subprocess.run([cache, "--version"], check=True, capture_output=True)
    directory = tmp_path_factory.mktemp("native-indexed-binding")
    source, binary = directory / "binding.cpp", directory / "binding"
    source.write_text(
        emit_native_ao_grid_binding()
        + r"""
#include <cstdio>
#include <string>
struct Owner {
  std::size_t nao{5}, npoint{7}, tile_points{3}, jets{4}, ao_map_entries{7};
  bool local_ao{true};
  int map_derivative_order{1};
};
int main(int argc, char** argv) {
  using generativeqc::dft::bind_native_ao_grid_block;
  Owner owner;
  std::vector<std::size_t> offsets{0, 2, 2, 7}, indices{1, 4, 0, 1, 2, 3, 4};
  const auto* ids = indices.data();
  std::size_t begin = 0;
  if (argc > 1) {
    const std::string fault = argv[1];
    if (fault == "unknown") owner.map_derivative_order = -1;
    else if (fault == "insufficient") owner.map_derivative_order = 0;
    else if (fault == "hessian") owner.jets = 10;
    else if (fault == "excessive") owner.map_derivative_order = 4;
    else if (fault == "jets") owner.jets = 2;
    else if (fault == "unaligned") begin = 1;
    else if (fault == "outside") begin = owner.npoint;
    else if (fault == "short_offsets") offsets.pop_back();
    else if (fault == "descending") { offsets[0] = 3; offsets[1] = 2; }
    else if (fault == "past_map") owner.ao_map_entries = 1;
    else if (fault == "ao_extent") { offsets[1] = 6; }
    else if (fault == "null_ids") ids = nullptr;
    else if (fault == "no_points") owner.npoint = 0;
    else if (fault == "no_basis") owner.nao = 0;
    else if (fault == "no_tile") owner.tile_points = 0;
    else return 2;
    try { (void)bind_native_ao_grid_block(owner, offsets, ids, begin); }
    catch (const std::invalid_argument&) { return 0; }
    return 1;
  }
  for (begin = 0; begin < owner.npoint; begin += owner.tile_points) {
    const auto block = bind_native_ao_grid_block(owner, offsets, ids, begin);
    if (block.owner_identity != &owner) return 3;
    if (block.indexed && block.nactive && block.ao_ids != ids + offsets[begin / 3])
      return 4;
    if ((!block.indexed || !block.nactive) && block.ao_ids) return 5;
    std::printf("%zu %zu %zu %zu %u %d %d\n", block.nao, block.nactive,
                block.npoint, block.point_start, block.derivative_order,
                block.map_derivative_order, static_cast<int>(block.indexed));
  }
  const auto block = bind_native_ao_grid_block(owner, offsets, ids, 0);
  Owner replacement = owner;
  const auto other = bind_native_ao_grid_block(replacement, offsets, ids, 0);
  if (block.owner_identity == other.owner_identity) return 6;
  owner.local_ao = false;
  owner.map_derivative_order = -1;
  const auto dense = bind_native_ao_grid_block(owner, {}, nullptr, 0);
  if (dense.indexed || dense.ao_ids || dense.nactive != owner.nao) return 7;
}
"""
    )
    subprocess.run(
        [
            cache,
            compiler,
            "-std=c++17",
            "-Wall",
            "-Wextra",
            "-Werror",
            str(source),
            "-o",
            str(binary),
        ],
        check=True,
        env={**os.environ, "CCACHE_BASEDIR": str(directory)},
    )
    return binary


def test_native_csr_spans_match_compiler_layout(native_binding_probe: Path) -> None:
    """Subset, empty and identity spans share the already admitted CSR labels."""
    result = subprocess.run(
        [str(native_binding_probe)], check=True, capture_output=True, text=True
    )
    actual = [tuple(map(int, row.split())) for row in result.stdout.splitlines()]
    expected = []
    for active, points, begin in ((2, 3, 0), (0, 3, 3), (5, 1, 6)):
        block = AoGridBlockLayout(
            5, active, points, 1, "immutable-owner", active < 5, 1, begin
        )
        block.require_derivative_order(1)
        expected.append(
            (
                block.nao,
                block.nactive,
                block.npoint,
                block.point_start,
                block.derivative_order,
                block.map_derivative_order,
                int(block.indexed),
            )
        )
    assert actual == expected


@pytest.mark.parametrize(
    "fault",
    [
        "unknown",
        "insufficient",
        "hessian",
        "excessive",
        "jets",
        "unaligned",
        "outside",
        "short_offsets",
        "descending",
        "past_map",
        "ao_extent",
        "null_ids",
        "no_points",
        "no_basis",
        "no_tile",
    ],
)
def test_native_csr_binding_rejects_unproven_domains(
    native_binding_probe: Path, fault: str
) -> None:
    subprocess.run([str(native_binding_probe), fault], check=True)


def test_native_grid_source_composes_the_binding_before_consumers() -> None:
    """Native CSR execution uses the emitted contract, not an extra map table."""
    source, _, _ = emit_grid_source(native_ks=True)
    assert source.index("struct NativeAoGridBlockLayout") < source.index(
        '#include "cuda_xc_kernels.cuh"'
    )
    ordinary, _, _ = emit_grid_source()
    assert ordinary.index("struct NativeAoGridBlockLayout") < ordinary.index(
        '#include "cuda_grid.cu"'
    )


def test_grid_work_header_participates_in_artifact_identity() -> None:
    """The emitted grid's transitive native work ABI must invalidate its cache."""
    assert "ao_grid_work.hpp" in {path.name for path in emit_grid_source()[2]}


def test_native_density_launcher_tracks_bound_point_span(
    tmp_path: Path, native_cxx: NativeCxx
) -> None:
    """Compile the real launcher selection for indexed/empty/full and tail tiles."""
    from pathlib import Path

    root = Path(__file__).resolve().parents[2]
    header = (root / "src/dft/cuda_xc_kernels.cuh").read_text()
    start = header.index("const auto density_launcher =")
    declaration = header[start : header.index(";", start) + 1]
    source, executable = tmp_path / "launcher.cpp", tmp_path / "launcher"
    source.write_text(
        emit_native_ao_grid_binding()
        + r"""
struct Owner {
  std::size_t nao{5}, npoint{7}, tile_points{3}, jets{4}, ao_map_entries{7};
  bool local_ao{true};
  int map_derivative_order{1};
};
int main() {
  Owner l;
  std::vector<std::size_t> offsets{0, 2, 2, 7};
  const std::size_t ids[]{1, 4, 0, 1, 2, 3, 4};
  const std::vector<int> local_density_launchers{101, 202, 303};
  struct Binding { int launch; };
  const Binding density_bindings[]{{404}, {505}};
  for (const bool indexed : {true, false}) {
    l.local_ao = indexed;
    for (const std::size_t begin : {0, 3, 6}) {
      const auto block = generativeqc::dft::bind_native_ao_grid_block(l, offsets, ids, begin);
      const auto count = block.npoint;
"""
        + declaration
        + r"""
      const int expected = indexed ? 101 * (1 + begin / l.tile_points)
                                   : (begin == 6 ? 505 : 404);
      if (density_launcher != expected) return 1;
    }
  }
}
"""
    )
    native_cxx.build_executable(
        (source,),
        executable,
        compile_args=("-std=c++17", "-Wall", "-Wextra", "-Werror"),
    )
    subprocess.run([str(executable)], check=True, timeout=10)


def test_native_density_factor_accepts_identity_and_sparse_spans(
    tmp_path: Path, native_cxx: NativeCxx
) -> None:
    """Execute the emitted provider gather with the real CSR binding contract."""
    from generativeqc_compiler.dft.xc_contraction_cuda import _emit_density_factor

    source, executable = tmp_path / "factor.cpp", tmp_path / "factor"
    source.write_text(
        emit_native_ao_grid_binding()
        + r"""
#include <cmath>
#define __global__
using I = long long;
struct { I x; } blockIdx{0}, blockDim{1}, threadIdx{0}, gridDim{1};
double finite(double value, int* error, int code) {
  if (!std::isfinite(value)) *error = code;
  return value;
}
struct Owner {
  std::size_t nao{5}, npoint{7}, tile_points{3}, jets{4}, ao_map_entries{7};
  bool local_ao{true};
  int map_derivative_order{1};
};
"""
        + _emit_density_factor(indexed=True)
        + r"""
int main() {
  Owner owner;
  const std::vector<std::size_t> offsets{0, 2, 2, 7};
  const std::size_t ids[]{1, 4, 0, 1, 2, 3, 4};
  double density[50];
  for (I spin = 0; spin < 2; ++spin)
    for (I row = 0; row < 5; ++row)
      for (I col = 0; col < 5; ++col)
        density[(spin * 5 + row) * 5 + col] = 100 * spin + 10 * row + col;
  for (const I spins : {1, 2}) {
    for (const std::size_t begin : {0, 3, 6}) {
      const auto block = generativeqc::dft::bind_native_ao_grid_block(
          owner, offsets, ids, begin);
      if (begin == 6 && (block.indexed || block.ao_ids)) return 1;
      double output[52];
      std::fill(output, output + 52, -1234.0);
      int error = 0;
      gather_density_factor(density, owner.nao, block.nactive, spins,
                            block.ao_ids, output + 1, &error);
      if (error) return 2;
      for (I spin = 0; spin < spins; ++spin)
        for (std::size_t row = 0; row < block.nactive; ++row)
          for (std::size_t col = 0; col < block.nactive; ++col) {
            const auto global_row = ids[offsets[begin / 3] + row];
            const auto global_col = ids[offsets[begin / 3] + col];
            const auto expected = 100 * spin + 5.5 * (global_row + global_col);
            if (output[1 + (spin * block.nactive + row) * block.nactive + col]
                != expected) return 3;
          }
      if (output[0] != -1234.0) return 4;
      for (std::size_t i = 1 + spins * block.nactive * block.nactive; i < 52; ++i)
        if (output[i] != -1234.0) return 5;
    }
  }
}
"""
    )
    native_cxx.build_executable(
        (source,),
        executable,
        compile_args=("-std=c++17", "-Wall", "-Wextra", "-Werror"),
    )
    subprocess.run([str(executable)], check=True, timeout=10)
