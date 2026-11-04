"""Execute the response planner at capacity cliffs and mandatory-storage limits."""

from __future__ import annotations

import re
import subprocess
from pathlib import Path
from typing import TYPE_CHECKING

from _cpp_source_support import cpp_function_definition, cpp_if_block

if TYPE_CHECKING:
    from conftest import NativeCxx


def test_screening_headroom_preserves_bounded_response(
    tmp_path: Path, native_cxx: NativeCxx
) -> None:
    source = tmp_path / "response_budget.cpp"
    source.write_text(r"""
#include "scf/df_response_tile_budget.hpp"
#include <algorithm>
#include <cassert>
#include <limits>

int main() {
  using generativeqc::scf::df_response_tile_with_headroom;
  // Independently enumerate valid panels for linear and occupied scratch.
  // Optional storage cannot displace the mandatory minimum, including at
  // exact byte boundaries and when all panel widths share fixed scratch.
  for (bool occupied : {false, true}) {
    auto bytes = [=](std::size_t q) { return 113 + std::max(occupied ? 97UL : 0UL, 17*q); };
    for (std::size_t cap = 1; cap <= 20; ++cap) {
      for (std::size_t available = bytes(1); available <= bytes(20)+41; ++available) {
        for (std::size_t extra : {0U, 1U, 16U, 17U, 41U, 512U}) {
          std::size_t expected = cap;
          if (extra && extra <= available && bytes(1) <= available-extra) {
            expected = 1;
            for (std::size_t q = 1; q <= cap; ++q)
              if (bytes(q) <= available-extra) expected = q;
          }
          const auto actual = df_response_tile_with_headroom(cap, available, extra, bytes);
          assert(actual == expected && actual <= cap && actual > 0);
        }
      }
    }
  }
  // Exercise the observed PBE0-96 scale without overflowing at size_t limits.
  auto workspace = [](std::size_t q) { return 4*3616UL*3616*8 + (3+2*q)*768UL*768*8 + 2*3616UL*8; };
  const auto room = 4325728307UL - 768UL*768*8;
  const auto q = df_response_tile_with_headroom(412, room, 1189120, workspace);
  assert(q == 411 && workspace(q)+1189120 <= room);
  auto identity = [](std::size_t q) { return q; };
  const auto maximum = std::numeric_limits<std::size_t>::max();
  assert(df_response_tile_with_headroom(maximum, maximum, 1, identity) == maximum-1);
  assert(df_response_tile_with_headroom(1, 0, maximum, identity) == 1);
  assert(df_response_tile_with_headroom(0, 0, 0, identity) == 0);
  using generativeqc::scf::df_recycled_projection_preferred;
  // Recycled storage cannot displace retained factors or authorize a scratch
  // interval too small for the algorithm. It is optional when owned work fits.
  assert(!df_recycled_projection_preferred(100, 101, 1, 20, 80));
  assert(!df_recycled_projection_preferred(100, 60, 50, 60, 49));
  assert(!df_recycled_projection_preferred(100, 60, 20, 40, 80));
  assert(df_recycled_projection_preferred(100, 60, 20, 41, 80));
  assert(df_recycled_projection_preferred(100, 60, 50, 60, 50));
  // Falling short of the preferred 512-column panel does not justify
  // replacing a viable 499-column owned panel with 25 recycled columns.
  assert(!df_recycled_projection_preferred(600, 101, 1, 512, 25));
  assert(!df_recycled_projection_preferred(600, 101, 1, 512, 498));
  assert(df_recycled_projection_preferred(600, 101, 1, 512, 499));
  assert(df_recycled_projection_preferred(600, 101, 1, 512, 512));
  assert(!df_recycled_projection_preferred(maximum, maximum-20, 20, 20, 80));
  assert(df_recycled_projection_preferred(maximum, maximum-20, 20, 21, 80));
  using generativeqc::scf::df_occupied_response_workspace;
  assert(!df_occupied_response_workspace(0,1,1).elements);
  assert(!df_occupied_response_workspace(1,maximum,1).elements);
  assert(!df_occupied_response_workspace(maximum,1,1).elements);
  assert(!df_occupied_response_workspace(1,1,maximum).elements);
  // Distinct live intervals must fit the declared fixed allocation. In the
  // observed nonlocal replay, omitting dead metric scratch must admit the
  // retained factors without changing their representation or dropping them.
  for (auto n : {1U, 7U, 768U}) for (auto a : {1U, 11U, 3616U})
    for (auto terms : {1U, 2U, 3U}) for (bool compact : {false, true})
      for (bool spectral : {false, true}) {
        const auto plan = df_occupied_response_workspace(n,a,terms,compact,spectral);
        assert(plan.metric_adjoint+a*a <= plan.ao_temporary);
        assert(plan.ao_temporary+3UL*n*n <= plan.charges);
        assert(plan.charges+terms*a <= plan.potentials);
        assert(plan.potentials+terms*a == plan.elements);
        if (compact && spectral) assert(a*a <= plan.metric_adjoint);
      }
  using generativeqc::scf::df_spectral_occupied_workspace;
  for (const auto bad : {std::size_t{0}, maximum}) {
    assert(!df_spectral_occupied_workspace(bad,11,2,3));
    assert(!df_spectral_occupied_workspace(4,bad,2,3));
    assert(!df_spectral_occupied_workspace(4,11,bad,3));
  }
  assert(!df_spectral_occupied_workspace(4,11,5,3));
  assert(!df_spectral_occupied_workspace(4,11,2,12));
  using generativeqc::scf::df_source_response_fixed_workspace;
  using generativeqc::scf::df_response_optional_storage_fits;
  assert(!df_source_response_fixed_workspace(0,384,1,true));
  assert(!df_source_response_fixed_workspace(96,0,1,true));
  assert(!df_source_response_fixed_workspace(96,384,0,true));
  assert(!df_source_response_fixed_workspace(maximum,384,1,true));
  assert(!df_source_response_fixed_workspace(96,maximum,1,true));
  assert(!df_source_response_fixed_workspace(96,384,maximum,true));
  assert(!df_source_response_fixed_workspace(96,384,1,false,20));
  assert(!df_source_response_fixed_workspace(96,384,2,true,20));
  assert(!df_source_response_fixed_workspace(96,384,1,true,97));
  assert(!df_source_response_fixed_workspace(1,1,maximum/3,false));
  const auto large_n = (std::size_t{1} << ((std::numeric_limits<std::size_t>::digits-2)/2))-1;
  assert(!df_source_response_fixed_workspace(large_n,1,1,true));
  // Optional shell metadata must coexist with the selected response layout,
  // including truncated occupied factors or the complete dense spectral root.
  // A cap that admits only generic response must retain that route. Test every
  // byte of the generic-only interval, then the exact shell-admission boundary.
  constexpr std::size_t n = 96, a = 384, rank = 20;
  constexpr std::size_t shell_bytes = (96+384)*12+16;
  constexpr std::size_t coordinates_output = 6*32*sizeof(double);
  for (bool truncated : {false,true}) for (bool occupied : {false,true}) {
    if (occupied && !truncated) continue;
    const auto fixed = df_source_response_fixed_workspace(n,a,1,truncated,occupied ? rank : 0);
    const auto expected = occupied ? 4*a*a+2*a*rank*rank+2*n*n+n*rank
                                  : 4*a*a+4*n*n+2*a+(truncated ? a*n*n : 0);
    assert(fixed == expected);
    const auto generic_bytes = coordinates_output+(fixed+2*n*n)*sizeof(double);
    const auto shell_minimum = generic_bytes+shell_bytes;
    for (auto cap = generic_bytes; cap < shell_minimum; ++cap) {
      assert(generic_bytes <= cap);
      assert(!df_response_optional_storage_fits(generic_bytes,shell_bytes,cap,maximum,maximum));
      assert(!df_response_optional_storage_fits(generic_bytes,shell_bytes,maximum,cap,maximum));
      assert(!df_response_optional_storage_fits(generic_bytes,shell_bytes,maximum,maximum,cap));
    }
    assert(df_response_optional_storage_fits(generic_bytes,shell_bytes,
                                            shell_minimum,shell_minimum,shell_minimum));
    if (occupied) {
      assert(generic_bytes == 7488000 && shell_minimum == 7493776);
      assert(df_spectral_occupied_workspace(n,a,rank,2)+n*n == fixed+2*n*n);
      // The old full-rank estimate incorrectly admitted shell metadata here.
      const auto old = df_source_response_fixed_workspace(n,a,1,false);
      assert(coordinates_output+shell_bytes+(old+2*n*n)*sizeof(double) < generic_bytes);
    }
  }
  assert(!df_response_optional_storage_fits(-1,1,maximum,maximum,maximum));
  assert(!df_response_optional_storage_fits(1,-1,maximum,maximum,maximum));
  assert(!df_response_optional_storage_fits(std::numeric_limits<long double>::infinity(),
                                           1,maximum,maximum,maximum));
  // The actual A100 LR response ceiling must cover the declared compact
  // tensors, density upload and a 64-Q AO output panel. Dense spectral Z
  // alone exceeds that ceiling, independent of how its panels are tiled.
  const auto projected = df_spectral_occupied_workspace(768,3616,160,64);
  assert(projected && 8*(projected+768UL*768) < 2549049300UL);
  assert(8*3616UL*768*768 > 2549049300UL);
  assert(df_spectral_occupied_workspace(768,3616,160,65)-projected == 768UL*768);
  const auto general = df_occupied_response_workspace(768,3616,1);
  const auto fitted = df_occupied_response_workspace(768,3616,1,true,false);
  const auto spectral = df_occupied_response_workspace(768,3616,1,true,true);
  const auto factors = 3616UL*160*160*8;
  const auto density = 768UL*768*8;
  assert(factors+general.elements*8+density > 954531840);
  assert(factors+fitted.elements*8+density < 954531840);
  assert(fitted.elements < spectral.elements && spectral.elements < general.elements);
}
""")
    binary = tmp_path / "response_budget"
    root = Path(__file__).resolve().parents[2]
    native_cxx.build_executable(
        [source],
        binary,
        compile_args=(
            "-std=c++20",
            "-Wall",
            "-Wextra",
            "-Werror",
            "-I",
            str(root / "src"),
        ),
    )
    subprocess.run([str(binary)], check=True, capture_output=True, timeout=30)


def test_spectral_packed_source_uses_the_consumer_workspace_boundary(
    tmp_path: Path, native_cxx: NativeCxx
) -> None:
    """Execute the CUDA bridge's actual packed reader against guarded host storage.

    The source-admission count includes an independently uploaded density. The
    packed panel must start at the consumer's weight interval, not at that larger
    count. Extract both sides so a stale identifier fails compilation and an
    in-bounds but shifted pointer cannot pass a copied arithmetic-only probe.
    """
    root = Path(__file__).resolve().parents[2]
    bridge = (root / "src/scf/cuda/df_gradient_bridge.cu").read_text()
    fixed = re.search(
        r"\bconst\s+auto\s+source_fixed_elements\s*=[^;]+;"
        r"\s*if\s*\(\s*!source_fixed_elements\s*\)\s*throw\s+std::bad_alloc\(\);",
        bridge,
    )
    assert fixed is not None, "missing source admission declaration and guard"
    reader = cpp_if_block(bridge, "spectral_occupied && source")
    consumer = cpp_function_definition(
        (root / "src/scf/cuda/df_response_weights.cu").read_text(),
        "contract_spectral_occupied_exchange",
    )
    layout = re.search(
        r"\bconst\s+auto\s+rank\s*=.*?(?=\bconst\s+auto\s+ni\s*=)",
        consumer,
        re.DOTALL,
    )
    assert layout is not None, "missing spectral consumer layout"
    source = tmp_path / "spectral_packed_layout.cpp"
    source.write_text(
        SPECTRAL_PACKED_PRELUDE
        + fixed.group()
        + "\nreturn source_fixed_elements;\n}\n"
        + SPECTRAL_PACKED_PROBE
        + layout.group()
        + SPECTRAL_PACKED_READER_SETUP
        + reader
        + SPECTRAL_PACKED_MAIN
    )
    binary = tmp_path / "spectral_packed_layout"
    native_cxx.build_executable(
        [source],
        binary,
        compile_args=(
            "-std=c++20",
            "-O2",
            "-Wall",
            "-Wextra",
            "-Werror",
            "-I",
            str(root / "src"),
        ),
    )
    subprocess.run([str(binary)], check=True, capture_output=True, timeout=30)


SPECTRAL_PACKED_PRELUDE = r"""
#include "scf/df_response_tile_budget.hpp"
#include <algorithm>
#include <array>
#include <cassert>
#include <cstddef>
#include <new>
#include <stdexcept>
#include <string>
#include <vector>
using namespace generativeqc::scf;
constexpr int cudaSuccess=0, cudaErrorInvalidValue=1;
constexpr int GENERATIVEQC_STATUS_SUCCESS=0, GENERATIVEQC_STATUS_OUT_OF_MEMORY=2;
int cudaGetLastError() { return cudaSuccess; }
void check(int status) { assert(status==cudaSuccess); }
namespace runtime::cuda_trace {
void trace_counter(const char*, std::size_t) {}
}
struct Factor { std::size_t rank; const double* coefficients; };
struct Occupied { std::array<Factor,1> factors; };
struct Metric { bool full_rank; };
struct Arena { struct { std::size_t recomputed_value_bytes=0; } stats; int stream=0; };
struct Source {
  std::size_t n, a, epoch, calls=0;
  double* expected_packed;
  double value(std::size_t pair, std::size_t q) const {
    return static_cast<double>(epoch*1000000+q*10000+pair+1);
  }
};
int generate_cuda_density_fitting_packed_raw_tile(
    Source* source, std::size_t source_index, std::size_t first_pair, std::size_t pairs,
    std::size_t begin, std::size_t count, int stream, double* packed, std::string&) {
  assert(source_index==0 && first_pair==0 && stream==0);
  assert(pairs==source->n*(source->n+1)/2 && begin+count<=source->a);
  // Check before writing: even the one-column mutant stays a defined failure.
  assert(packed==source->expected_packed);
  ++source->calls;
  for (std::size_t pair=0; pair<pairs; ++pair)
    for (std::size_t q=0; q<count; ++q)
      packed[pair*count+q]=source->value(pair,begin+q);
  return GENERATIVEQC_STATUS_SUCCESS;
}
namespace cuda_df {
void launch_unpack_df_values(int stream, std::size_t n, std::size_t columns,
                             std::size_t first_row, std::size_t rows, std::size_t column,
                             std::size_t count, bool auxiliary_major,
                             const double* packed, double* values) {
  assert(stream==0 && first_row==0 && rows==n && count==1 && auxiliary_major);
  assert(column<columns);
  for (std::size_t i=0; i<n; ++i) for (std::size_t j=0; j<n; ++j) {
    const auto high=std::max(i,j), low=std::min(i,j);
    values[i*n+j]=packed[(high*(high+1)/2+low)*columns+column];
  }
}
}  // namespace cuda_df
std::size_t selected_fixed(std::size_t n, std::size_t a, std::size_t term_count,
                           bool truncated, std::size_t rank) {
  std::vector<int> terms(term_count);
  Metric metric{!truncated};
  const auto* device_metric=&metric;
  Occupied owner{{Factor{rank,nullptr}}};
  const auto* occupied=&owner;
  const bool spectral_occupied=rank!=0;
"""

SPECTRAL_PACKED_PROBE = r"""
int exercise_panel(std::size_t n, std::size_t a, std::size_t rank_value,
                    std::size_t tile, std::size_t epoch) {
  const auto expected_fixed=4*a*a+2*a*rank_value*rank_value+n*n+n*rank_value;
  const auto source_fixed_elements=selected_fixed(n,a,1,true,rank_value);
  assert(source_fixed_elements==expected_fixed+n*n);
  const auto elements=df_spectral_occupied_workspace(n,a,rank_value,tile);
  assert(elements==expected_fixed+tile*n*n);
  constexpr double untouched=-991, guard=-882;
  std::vector<double> storage(elements+2,untouched), density(n*n+2,guard);
  storage.front()=storage.back()=guard;
  auto* workspace=storage.data()+1;
  // These are the actual CUDA consumer's declaration, validation and pointers.
  const Factor factor{rank_value,density.data()+1};
  constexpr bool read_values=true;
"""

SPECTRAL_PACKED_READER_SETUP = r"""
  assert(bar_root==workspace && metric_temporary==workspace+a*a);
  assert(transformed==workspace+2*a*a && raw_projected==workspace+4*a*a);
  assert(fitted==raw_projected+a*rank*rank && raw==fitted+a*rank*rank);
  assert(half==raw+n*n && weights==half+n*rank);
  assert(weights==workspace+expected_fixed && weights+tile*n*n==workspace+elements);
  // The packed panel aliases only the still-dead derivative weight interval.
  // No metric/occupied matrix, raw AO slice or half projection may alias it.
  Source fixture{n,a,epoch,0,workspace+expected_fixed};
  auto* source=&fixture;
  constexpr bool spectral_occupied=true;
  constexpr std::size_t source_index=0;
  constexpr int stream_handle=0;
  std::string detail;
  Arena arena;
  std::size_t spectral_raw_begin=a, spectral_raw_count=0;
  auto read=[&](std::size_t p, double* values) {
"""

SPECTRAL_PACKED_MAIN = r"""
  };
  auto check_guards=[&] {
    assert(storage.front()==guard && storage.back()==guard);
    assert(std::all_of(workspace,raw,[&](double v) { return v==untouched; }));
    assert(std::all_of(half,weights,[&](double v) { return v==untouched; }));
    assert(std::all_of(density.begin(),density.end(),[&](double v) { return v==guard; }));
  };
  const auto pairs=n*(n+1)/2;
  for (std::size_t p=0; p<a; ++p) {
    const auto calls=fixture.calls;
    const bool new_panel=p%tile==0;
    // A new panel may overwrite only its live packed prefix. Cache hits must
    // preserve the actual earlier generated data, not reconstruct it in the test.
    if (new_panel) std::fill(weights,workspace+elements,untouched);
    read(p,raw);
    assert(fixture.calls==calls+new_panel);
    for (std::size_t i=0; i<n; ++i) for (std::size_t j=0; j<n; ++j) {
      const auto high=std::max(i,j), low=std::min(i,j);
      assert(raw[i*n+j]==fixture.value(high*(high+1)/2+low,p));
    }
    // Including the shorter last panel, storage beyond packed pairs is untouched.
    assert(std::all_of(weights+pairs*spectral_raw_count,workspace+elements,
                       [&](double v) { return v==untouched; }));
    const auto before_repeat=storage;
    read(p,raw);
    assert(storage==before_repeat && fixture.calls==calls+new_panel);
    check_guards();
  }
  assert(fixture.calls==(a+tile-1)/tile);
  assert(arena.stats.recomputed_value_bytes==pairs*a*sizeof(double));
  // A backwards read invalidates a later cached panel rather than serving it stale.
  const auto old_calls=fixture.calls;
  read(0,raw);
  assert(fixture.calls==old_calls+(a>tile));
  assert(raw[0]==fixture.value(0,0));
  check_guards();
  // Once projection is finished, the complete dense derivative panel reuses
  // the same interval without crossing the allocation's trailing sentinel.
  std::fill(weights,weights+tile*n*n,37);
  check_guards();
  return cudaSuccess;
}
int main() {
  // Full-rank and dense truncated source paths include their uploaded densities
  // and retain their own existing layouts; only occupied truncation uses rank.
  for (std::size_t n : {1U,2U,5U,17U}) for (std::size_t a : {1U,3U,8U}) {
    for (std::size_t terms : {1U,2U,3U}) for (bool truncated : {false,true}) {
      const auto expected=4*a*a+(3+terms)*n*n+2*terms*a+(truncated ? a*n*n : 0);
      assert(selected_fixed(n,a,terms,truncated,0)==expected);
    }
    for (std::size_t rank : {std::size_t{1},n})
      for (std::size_t tile : {std::size_t{1},std::min(std::size_t{3},a),a})
        for (std::size_t epoch : {1U,2U})
          assert(exercise_panel(n,a,rank,tile,epoch)==cudaSuccess);
  }
  // Declaration/guard extraction must retain admission failure as well as size.
  for (const auto shape : {std::array<std::size_t,5>{0,3,1,1,1},
                            std::array<std::size_t,5>{2,3,2,1,1},
                            std::array<std::size_t,5>{2,3,1,0,1},
                            std::array<std::size_t,5>{2,3,1,1,3}}) {
    bool rejected=false;
    try { (void)selected_fixed(shape[0],shape[1],shape[2],shape[3],shape[4]); }
    catch (const std::bad_alloc&) { rejected=true; }
    assert(rejected);
  }
}
"""
