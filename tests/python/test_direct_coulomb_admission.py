"""Qualify exact-zero Coulomb admission independently of the GPU schedule."""

import subprocess
from pathlib import Path
from typing import TYPE_CHECKING

from _cpp_source_support import cpp_function_definition
from generativeqc_compiler.integral.direct_pair_materialized_cuda import (
    emit_direct_pair_materialized_support,
)
from generativeqc_compiler.integral.lowering.fock_accumulation import (
    emit_direct_fock_accumulation_header,
)

if TYPE_CHECKING:
    from conftest import NativeCxx

ROOT = Path(__file__).resolve().parents[2]


def test_coulomb_admission_matches_complete_orbit(
    tmp_path: Path, native_cxx: "NativeCxx"
) -> None:
    """Enumerate signed, nonsymmetric, offset and spin-cancelling densities.

    A dense eightfold orbit, not the admission helper, decides whether any
    Coulomb contribution exists. Repeated indices exercise orbit degeneracy.
    """
    (tmp_path / "cuda_runtime.h").write_text(
        "#pragma once\n#define __device__\n#define __host__\n"
        "#define __forceinline__ inline\n"
        "template<class T> T atomicAdd(T* output,T value) { "
        "T previous=*output; *output+=value; return previous; }\n"
    )
    (tmp_path / "scatter.hpp").write_text(emit_direct_fock_accumulation_header())
    source, binary = tmp_path / "admission.cpp", tmp_path / "admission"
    source.write_text(
        r"""
#include <array>
#include "scatter.hpp"
using namespace generativeqc::scf::cuda_execution;
template<bool Unrestricted> int check_admission() {
  constexpr std::size_t dimension = 3, matrix = dimension * dimension;
  constexpr std::size_t physical_offset = matrix, spin_offset = 2 * matrix;
  for (unsigned mode = 0; mode < 4; ++mode)
    for (std::size_t active = 0; active < matrix; ++active) {
      std::array<double, 4 * matrix> density{};
      if constexpr (Unrestricted) {
        density[spin_offset + active] = mode == 0 ? 0.0 : -0.7;
        density[spin_offset + matrix + active] = mode == 2 ? 0.7 : (mode == 3 ? 0.2 : 0.0);
      } else {
        density[physical_offset + active] = mode == 0 ? 0.0 : -0.7;
      }
      for (std::size_t first = 0; first < dimension; ++first)
        for (std::size_t second = 0; second < dimension; ++second)
          for (std::size_t third = 0; third < dimension; ++third)
            for (std::size_t fourth = 0; fourth < dimension; ++fourth) {
              bool expected = false;
              const std::array<std::array<std::size_t, 4>, 8> orbit{{
                  {first, second, third, fourth}, {second, first, third, fourth},
                  {first, second, fourth, third}, {second, first, fourth, third},
                  {third, fourth, first, second}, {fourth, third, first, second},
                  {third, fourth, second, first}, {fourth, third, second, first}}};
              for (const auto& indices : orbit) {
                const auto entry = indices[2] * dimension + indices[3];
                const double total = Unrestricted
                    ? density[spin_offset + entry] + density[spin_offset + matrix + entry]
                    : density[physical_offset + entry];
                expected = expected || total != 0.0;
              }
              if (expected != direct_coulomb_density_active<Unrestricted>(
                      dimension, physical_offset, spin_offset, density.data(),
                      first, second, third, fourth)) return 1;
            }
    }
  return 0;
}
int main() { return check_admission<false>() || check_admission<true>(); }
"""
    )
    native_cxx.build_executable(
        [source],
        binary,
        compile_args=["-std=c++20", f"-I{tmp_path}", f"-I{ROOT / 'src'}"],
    )
    subprocess.run([binary], check=True, timeout=15)


def test_materialized_slots_cover_all_mixed_shell_classes(
    tmp_path: Path, native_cxx: "NativeCxx"
) -> None:
    """Independently enumerate component maxima for every s/p/d/f quartet."""
    emitted = emit_direct_pair_materialized_support()
    helpers = "\n".join(
        cpp_function_definition(emitted, name)
        for name in (
            "canonical_materialized_component_lanes",
            "canonical_materialized_component_slots",
        )
    )
    source, binary = tmp_path / "slots.cpp", tmp_path / "slots"
    source.write_text(
        "#include <algorithm>\n#include <array>\n"
        "#define __host__\n#define __device__\n"
        + helpers
        + r"""
int main() {
  std::array<unsigned, 13> maxima{};
  for (unsigned first = 0; first <= 3; ++first)
    for (unsigned second = 0; second <= 3; ++second)
      for (unsigned third = 0; third <= 3; ++third)
        for (unsigned fourth = 0; fourth <= 3; ++fourth) {
          unsigned components = 1;
          for (unsigned angular : {first, second, third, fourth})
            components *= (angular + 1) * (angular + 2) / 2;
          auto& maximum = maxima[first + second + third + fourth];
          maximum = std::max(maximum, components);
        }
  for (unsigned order = 0; order <= 12; ++order) {
    const auto lanes = canonical_materialized_component_lanes(order);
    const auto capacity = lanes * canonical_materialized_component_slots(order);
    if (capacity < maxima[order] || capacity - maxima[order] >= lanes) return 1;
  }
  return 0;
}
"""
    )
    native_cxx.build_executable([source], binary, compile_args=["-std=c++20"])
    subprocess.run([binary], check=True, timeout=15)
