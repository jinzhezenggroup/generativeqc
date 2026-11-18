"""Execute the native, method-neutral force-map capacity and dense fallback."""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from conftest import NativeCxx

ROOT = Path(__file__).resolve().parents[2]


def test_native_force_maps_preserve_complete_dense_capacity(
    tmp_path: Path, native_cxx: NativeCxx
) -> None:
    """Spin, XC composition and Fock choice must not define map capability.

    The real shape authority must retain the entire mandatory dense arena,
    charge optional peak/host storage, decline unsupported jets and oversize
    maps, and reject arithmetic overflow without allocating or querying CUDA.
    """
    source = tmp_path / "capacity.cpp"
    source.write_text(
        r"""
#include <cassert>
#include <limits>
#include "dft/cuda_stationary_capacity.hpp"

std::size_t numeric(std::size_t, std::size_t, std::size_t, std::size_t,
                    std::size_t, std::size_t, std::size_t, bool cached) {
  return cached ? 8192 : 4096;
}

int main() {
  using namespace generativeqc::dft;
  CudaStationaryBinding binding;
  binding.numeric_bytes = numeric;
  binding.geometry_max_lanes = binding.geometry_max_threads = 32;
  binding.geometry_max_scratch_bytes = 1 << 20;
  constexpr std::size_t aos = 256, points = (std::size_t{1} << 35) / (aos * aos);
  for (const unsigned jets : {4U, 10U})
    for (const unsigned spins : {1U, 2U})
      for (const bool nonlocal : {false, true}) {
        binding.ao_jets = jets;
        binding.spin_blocks = spins;
        binding.model.nonlocal_b = nonlocal ? 6.0 : 0.0;
        const auto dense = cuda_stationary_capacity(binding, 48, aos, 512, points - 1, 256,
                                                     false, nonlocal ? 1024 : 0);
        const auto selected = cuda_stationary_capacity(binding, 48, aos, 512, points, 256,
                                                        false, nonlocal ? 1024 : 0);
        assert(dense.ao_map_bytes == 0 && dense.ao_map_host_bytes == 0);
        const auto tiles = points / selected.tile_points;
        const auto offsets = (tiles + 1) * sizeof(std::size_t);
        const auto masks = tiles * ((aos + 31) / 32) * sizeof(unsigned);
        assert(selected.ao_map_bytes == 2 * offsets + masks + aos * sizeof(std::size_t));
        assert(selected.ao_map_host_bytes == offsets);
        assert(selected.minimum_device_bytes == dense.minimum_device_bytes);
        assert(selected.grid.allocation_bytes == dense.grid.allocation_bytes);
        assert(selected.stationary_minimum_bytes == dense.stationary_minimum_bytes);
        assert(selected.cached_device_bytes == dense.cached_device_bytes + selected.ao_map_bytes);
        assert(selected.cached_grid_extra_bytes == selected.ao_map_bytes);
        const auto bounded = cuda_stationary_capacity(binding, 48, aos, 512, points, 1);
        assert(bounded.ao_map_bytes == 0 && bounded.ao_map_host_bytes == 0);
        assert(bounded.minimum_device_bytes > 0);
      }
  binding.model.nonlocal_b = 0;
  for (const unsigned jets : {1U, 20U}) {
    binding.ao_jets = jets;
    const auto unsupported = cuda_stationary_capacity(binding, 48, aos, 512, points, 256);
    assert(unsupported.ao_map_bytes == 0 && unsupported.minimum_device_bytes > 0);
  }
  bool rejected = false;
  try {
    (void)cuda_stationary_capacity(binding, 48, aos, 512,
                                    std::numeric_limits<std::size_t>::max(), 256);
  } catch (const std::overflow_error&) {
    rejected = true;
  }
  assert(rejected);
}
"""
    )
    executable = native_cxx.build_executable(
        [source],
        tmp_path / "capacity",
        compile_args=["-std=c++20", f"-I{ROOT / 'src'}", f"-I{ROOT / 'include'}"],
    )
    subprocess.run([str(executable)], check=True, timeout=10)
