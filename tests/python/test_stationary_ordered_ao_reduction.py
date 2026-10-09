"""Execute the emitted reducer against independent ordered atom/grid sums."""

import subprocess
from pathlib import Path
from typing import TYPE_CHECKING

from generativeqc_compiler.method.stationary_cuda import _STATIONARY_SCIENTIFIC_KERNELS
from test_stationary_task_work_budget import _block

if TYPE_CHECKING:
    from conftest import NativeCxx


def test_ordered_ao_reduction_preserves_bits_and_visits_each_label_once(
    tmp_path: Path, native_cxx: "NativeCxx"
) -> None:
    """Check bits, one label read/AO and bounded gradient accesses per atom run."""
    reducer = (
        _block(
            _STATIONARY_SCIENTIFIC_KERNELS,
            "__device__ void geometry_reduce_ao_panel(",
        )
        .replace("const int64_t* ao_atoms", "const AtomMap& ao_atoms")
        .replace("double* grad", "const GradientMap& grad")
    )
    source = tmp_path / "ordered_ao.cpp"
    source.write_text(PREFIX + reducer + DRIVER)
    executable = native_cxx.build_executable(
        [source],
        tmp_path / "ordered_ao",
        compile_args=["-std=c++20", "-O2", "-ffp-contract=off"],
    )
    subprocess.run([str(executable)], check=True, timeout=30)


PREFIX = r"""
#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <vector>
#define __device__
namespace generativeqc::dft {
struct GridTaskView {
  std::size_t nactive;
  const std::size_t* ao_ids;
};
}
struct AtomMap {
  const std::vector<int64_t>& values;
  mutable std::size_t reads = 0;
  int64_t operator[](std::size_t index) const {
    ++reads;
    return values.at(index);
  }
};
struct GradientMap {
  std::vector<double>& values;
  mutable std::size_t accesses = 0;
  double& operator[](std::size_t index) const {
    ++accesses;
    return values.at(index + 1);
  }
};
"""

DRIVER = r"""
int main() {
  for (std::size_t atoms : {1U, 2U, 12U, 48U, 96U, 128U}) {
    for (std::size_t active : {0U, 1U, 31U, 32U, 33U, 96U, 553U, 1024U}) {
      const std::size_t total_aos = 2 * active + 3;
      for (bool grouped : {false, true}) {
      std::vector<int64_t> labels(total_aos);
      for (std::size_t global_ao = 0; global_ao < total_aos; ++global_ao)
        labels[global_ao] = grouped ? global_ao / (total_aos / atoms + 1)
                                   : (7 * global_ao + global_ao / 3) % atoms;
      for (unsigned map_kind = 0; map_kind < 3; ++map_kind) {
        std::vector<std::size_t> indices(active);
        for (std::size_t ao_index = 0; ao_index < active; ++ao_index)
          indices[ao_index] = map_kind == 1 ? 2 * (active - 1 - ao_index)
                                           : (11 * (ao_index / 2)) % total_aos;
        const generativeqc::dft::GridTaskView view{
            active, map_kind == 0 ? nullptr : indices.data()};
        std::vector<double> panel(3 * active);
        for (std::size_t coordinate = 0; coordinate < panel.size(); ++coordinate)
          panel[coordinate] = std::ldexp(std::sin(0.37 * coordinate),
                                        int(coordinate % 37) - 18);
        for (std::size_t owner : {std::size_t(0), atoms - 1}) {
          std::vector<double> actual(9 * atoms + 2);
          for (std::size_t coordinate = 0; coordinate < actual.size(); ++coordinate)
            actual[coordinate] = 0.13 * std::cos(0.29 * coordinate);
          auto expected = actual;
          for (std::size_t atom = 0; atom < atoms; ++atom)
            for (std::size_t ao_index = 0; ao_index < active; ++ao_index) {
              const std::size_t global_ao = view.ao_ids ? view.ao_ids[ao_index] : ao_index;
              if (labels[global_ao] == int64_t(atom))
                for (std::size_t axis = 0; axis < 3; ++axis)
                  expected[1 + 3 * atom + axis] -= panel[3 * ao_index + axis];
            }
          for (std::size_t ao_index = 0; ao_index < active; ++ao_index)
            for (std::size_t axis = 0; axis < 3; ++axis)
              expected[1 + 3 * atoms + 3 * owner + axis] += panel[3 * ao_index + axis];
          AtomMap measured{labels};
          GradientMap destination{actual};
          geometry_reduce_ao_panel(view, measured, owner, atoms, panel.data(), destination);
          assert(measured.reads == active);
          std::size_t runs = 0, previous_atom = atoms;
          for (std::size_t ao_index = 0; ao_index < active; ++ao_index) {
            const auto atom = std::size_t(labels[view.ao_ids ? view.ao_ids[ao_index] : ao_index]);
            if (atom != previous_atom) ++runs;
            previous_atom = atom;
          }
          assert(destination.accesses == 6 * runs + 6);
          assert(std::memcmp(actual.data(), expected.data(), actual.size() * sizeof(double)) == 0);
        }
      }
      }
    }
  }
}
"""
