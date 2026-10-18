"""Host gates for the exact tile domain and conservative residual-J rejection."""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import TYPE_CHECKING

from _cpp_source_support import cpp_record_definition

if TYPE_CHECKING:
    from conftest import NativeCxx

ROOT = Path(__file__).resolve().parents[2]


def test_residual_tile_domain_and_outward_bounds(
    tmp_path: Path, required_native_cxx: NativeCxx
) -> None:
    """Execute production helpers against scalar, directed-rounding host oracles."""
    source = tmp_path / "tiles.cpp"
    binary = tmp_path / "tiles"
    source.write_text(
        r"""
#include <algorithm>
#include <array>
#include <cassert>
#include <cfenv>
#include <cmath>
#include <limits>
#include <random>
#include <set>
#include <utility>
#define __host__
#define __device__
#include "scf/cuda/md_source_tiles.cuh"
using namespace generativeqc::scf::cuda_execution;

double product(double first, double second, int mode) {
  assert(std::fesetround(mode) == 0);
  volatile double result = first * second;
  assert(std::fesetround(FE_TONEAREST) == 0);
  return result;
}

void domain(unsigned bra_count, unsigned ket_count, bool triangular) {
  std::set<std::pair<unsigned, unsigned>> visited;
  const auto tiles = md_source_tile_count(bra_count, ket_count, triangular);
  for(std::uint64_t index = 0; index < tiles; ++index) {
    const auto tile = md_source_tile(index, ket_count, triangular);
    for(unsigned bra = tile.bra; bra < std::min(bra_count, tile.bra + kMdSourceTileWidth); ++bra)
      for(unsigned ket = tile.ket; ket < std::min(ket_count, tile.ket + kMdSourceTileWidth); ++ket)
        if(!triangular || bra >= ket) assert(visited.emplace(bra, ket).second);
  }
  assert(visited.size() == (triangular ? bra_count * (bra_count + 1) / 2 : bra_count * ket_count));
  for(unsigned bra = 0; bra < bra_count; ++bra)
    for(unsigned ket = 0; ket < ket_count; ++ket)
      assert(visited.count({bra, ket}) == (!triangular || bra >= ket));
}

struct PairBounds { double minimum, maximum, density; };

bool scalar_admitted(PairBounds bra, PairBounds ket, double screening, double allowance) {
  if(!(bra.maximum * ket.maximum >= screening && bra.minimum * ket.minimum < screening))
    return false;
  const double bound = product(bra.maximum, ket.maximum, FE_UPWARD);
  return !(bra.density == 0.0 && ket.density == 0.0) &&
         !(product(bound, bra.density, FE_UPWARD) < allowance &&
           product(bound, ket.density, FE_UPWARD) < allowance);
}

int main() {
  for(unsigned bra_count = 0; bra_count <= 33; ++bra_count) {
    domain(bra_count, bra_count, true);
    for(unsigned ket_count = 0; ket_count <= 33; ++ket_count)
      domain(bra_count, ket_count, false);
  }
  const auto maximum = std::numeric_limits<std::uint32_t>::max();
  const auto axis_tiles = (std::uint64_t{maximum} + 7) / 8;
  assert(md_source_tile_count(maximum, maximum, false) == axis_tiles * axis_tiles);
  assert(md_source_tile_count(maximum, maximum, true) == axis_tiles * (axis_tiles + 1) / 2);
  for(std::uint64_t row : {1ULL, 1048576ULL, 67108864ULL, 536870911ULL}) {
    const auto row_start = row * (row + 1) / 2;
    for(auto index : {row_start - 1, row_start, row_start + row}) {
      const auto tile = md_source_tile(index, maximum, true);
      const auto bra = std::uint64_t{tile.bra} / 8;
      const auto ket = std::uint64_t{tile.ket} / 8;
      assert(bra * (bra + 1) / 2 + ket == index);
      assert(bra >= ket && tile.bra < maximum && tile.ket < maximum);
    }
  }
  const std::array<double, 8> thresholds = {
      0.0, std::numeric_limits<double>::denorm_min(), 1e-300, 1e-12,
      std::nextafter(0.01, 0.0), 0.01, std::nextafter(0.01, INFINITY), 1e200};
  std::mt19937_64 generator(541849);
  unsigned rejected = 0, admitted = 0;
  for(unsigned sample = 0; sample < 4096; ++sample) {
    std::array<PairBounds, 16> pairs;
    double bra_minimum = INFINITY, ket_minimum = INFINITY;
    double bra_maximum = 0.0, ket_maximum = 0.0, bra_density = 0.0, ket_density = 0.0;
    for(unsigned index = 0; index < pairs.size(); ++index) {
      const double maximum_bound = sample < 8 ? 0.1 :
          std::ldexp(1.0 + (generator() % 1024) / 1024.0, int(generator() % 1201) - 600);
      pairs[index] = {sample % 3 ? maximum_bound * 0.5 : 0.0, maximum_bound,
                     sample % 5 ? std::ldexp(1.0, int(generator() % 1201) - 600) : 0.0};
      auto& minimum_bound = index < 8 ? bra_minimum : ket_minimum;
      auto& maximum_value = index < 8 ? bra_maximum : ket_maximum;
      auto& density_value = index < 8 ? bra_density : ket_density;
      minimum_bound = std::min(minimum_bound, pairs[index].minimum);
      maximum_value = std::max(maximum_value, pairs[index].maximum);
      density_value = std::max(density_value, pairs[index].density);
    }
    const double screening = thresholds[sample % thresholds.size()];
    const double allowance = std::min(screening, 1e-12) / 8192;
    const double upper = product(bra_maximum, ket_maximum, FE_UPWARD);
    const bool discard = md_source_tile_rejected(
        upper, product(bra_minimum, ket_minimum, FE_DOWNWARD),
        product(upper, bra_density, FE_UPWARD), product(upper, ket_density, FE_UPWARD),
        bra_density == 0.0 && ket_density == 0.0, screening, allowance);
    if(discard) ++rejected;
    for(unsigned bra = 0; bra < 8; ++bra) for(unsigned ket = 8; ket < 16; ++ket) {
      const bool live = scalar_admitted(pairs[bra], pairs[ket], screening, allowance);
      assert(!discard || !live);
      if(live) ++admitted;
    }
  }
  assert(rejected > 0 && admitted > 0);
  assert(!md_source_tile_rejected(1.0, 0.0, 1e-12, 1e-12, false, 1.0, 1e-12));
  assert(md_source_tile_rejected(INFINITY, 0.0, NAN, NAN, true, 1e-12, 1e-15));
}
"""
    )
    required_native_cxx.build_executable(
        [source],
        binary,
        compile_args=[
            "-std=c++20",
            "-frounding-math",
            "-ffp-contract=off",
            f"-I{ROOT / 'src'}",
        ],
    )
    subprocess.run([str(binary)], check=True, timeout=30)


def test_profile_event_lifetime_and_opt_out(
    tmp_path: Path, required_native_cxx: NativeCxx
) -> None:
    """CPU doubles verify event cleanup, disabled tracing and error propagation."""
    production = (ROOT / "src/scf/cuda/direct_jk.cpp").read_text()
    source = tmp_path / "profile.cpp"
    binary = tmp_path / "profile"
    source.write_text(
        r"""
#include <array>
#include <cassert>
#include <cstdio>
#include <cstdint>
#include <stdexcept>
using cudaEvent_t = void*;
using cudaStream_t = void*;
unsigned creates = 0, destroys = 0, records = 0, fences = 0;
unsigned fail_at = 99;
int cudaEventCreate(cudaEvent_t* event) {
  if(creates++ == fail_at) return 1;
  *event = reinterpret_cast<void*>(std::uintptr_t{creates}); return 0;
}
int cudaEventDestroy(cudaEvent_t) { ++destroys; return 0; }
int cudaEventRecord(cudaEvent_t, cudaStream_t) { ++records; return 0; }
int cudaEventSynchronize(cudaEvent_t) { ++fences; return 0; }
int cudaEventElapsedTime(float* value, cudaEvent_t, cudaEvent_t) { *value = 1.0F; return 0; }
void direct_jk_check(int error) { if(error) throw std::runtime_error("injected"); }
"""
        + cpp_record_definition(production, "MdJProfile")
        + r""";
int main() {
  { MdJProfile profile(false); profile.record(0, nullptr); profile.report(1); }
  assert(creates == 0 && destroys == 0 && records == 0 && fences == 0);
  { MdJProfile profile(true); profile.record(0, nullptr); profile.report(2); }
  assert(creates == 7 && destroys == 7 && records == 1 && fences == 1);
  for(unsigned failure = 0; failure < 7; ++failure) {
    creates = destroys = 0; fail_at = failure;
    bool threw = false;
    try { MdJProfile profile(true); } catch(const std::runtime_error&) { threw = true; }
    assert(threw && destroys == failure);
  }
}
"""
    )
    required_native_cxx.build_executable([source], binary, compile_args=["-std=c++20"])
    subprocess.run([str(binary)], check=True, timeout=10)
