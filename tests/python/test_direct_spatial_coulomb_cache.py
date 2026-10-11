"""Execute compiler-emitted spatial reuse against independent state inventories."""

import subprocess
from pathlib import Path
from typing import TYPE_CHECKING

from generativeqc_compiler.integral.direct_pair_gradient_cuda import (
    emit_direct_high_order_pair_gradient_header,
)
from test_direct_force_schedule_composition import _definition

if TYPE_CHECKING:
    from conftest import NativeCxx


def test_spatial_cache_covers_only_box_and_single_response_faces(
    tmp_path: Path, native_cxx: "NativeCxx"
) -> None:
    """Check all axis partitions at orders 4/5/6, capacities and evaluation counts."""
    emitted = emit_direct_high_order_pair_gradient_header()
    definition = _definition(
        emitted,
        "template <unsigned AngularOrder, bool SpatialReuse>\n"
        "struct GradientSpatialCoulombValues",
    )
    source = tmp_path / "cache.cpp"
    source.write_text(PREFIX + definition + ";\n" + DRIVER)
    executable = native_cxx.build_executable(
        [source], tmp_path / "cache", compile_args=["-std=c++20", "-O2"]
    )
    result = subprocess.run(
        [executable], capture_output=True, text=True, check=False, timeout=60
    )
    assert result.returncode == 0, result.stderr


def test_cache_is_forwarded_only_into_explicit_order_six_gradient() -> None:
    """Preserve the old default and independently reachable long-range wrapper."""
    emitted = emit_direct_high_order_pair_gradient_header()
    begin = emitted.index("contracted_eri_cartesian_source_order6_gradient(")
    source = emitted[begin:]
    for first, second in ((6, 0), (5, 1), (4, 2), (3, 3)):
        assert (
            f"primitive_eri_order456_gradient<{first}, {second}, LongRange, SpatialReuse>("
            in source
        )
    assert "bool SpatialReuse = false>" in emitted


PREFIX = r"""
#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <limits>
#define __device__
struct Angular { unsigned x{}, y{}, z{}; };
template <unsigned Order> struct HighOrderCoulombWorkspace {};
unsigned evaluations{};
template <unsigned Order>
double high_order_coulomb(unsigned state, double rho,
                         const HighOrderCoulombWorkspace<Order>&, const double*) {
  ++evaluations;
  return (state + 0.25) * (rho + Order);
}
"""

DRIVER = r"""
template <unsigned Order>
void exercise() {
  const HighOrderCoulombWorkspace<Order + 1U> workspace{};
  const double rho = 0.375;
  for (unsigned maximum_x = 0; maximum_x <= Order; ++maximum_x) {
    for (unsigned maximum_y = 0; maximum_y <= Order - maximum_x; ++maximum_y) {
      const unsigned maximum_z = Order - maximum_x - maximum_y;
      const Angular total{maximum_x, maximum_y, maximum_z}, empty{};
      GradientSpatialCoulombValues<Order, true> cached;
      for (double& value : cached.values) value = std::numeric_limits<double>::quiet_NaN();
      evaluations = 0;
      cached.prepare(total, empty, empty, empty, rho, workspace, nullptr);
      const unsigned expected = (maximum_x + 1U) * (maximum_y + 1U) * (maximum_z + 1U) +
          (maximum_y + 1U) * (maximum_z + 1U) + (maximum_x + 1U) * (maximum_z + 1U) +
          (maximum_x + 1U) * (maximum_y + 1U);
      assert(evaluations == expected);
      assert((maximum_x + 2U) * cached.extent_y * cached.extent_z <= cached.capacity);
      if constexpr (Order == 6U) assert(expected >= 22U && expected <= 54U);
      for (unsigned axis_x = 0; axis_x <= maximum_x + 1U; ++axis_x)
        for (unsigned axis_y = 0; axis_y <= maximum_y + 1U; ++axis_y)
          for (unsigned axis_z = 0; axis_z <= maximum_z + 1U; ++axis_z) {
            const bool unused = unsigned(axis_x > maximum_x) + unsigned(axis_y > maximum_y) +
                                unsigned(axis_z > maximum_z) > 1U;
            const auto index = (axis_x * cached.extent_y + axis_y) * cached.extent_z + axis_z;
            assert(index < cached.capacity);
            if (unused) {
              assert(std::isnan(cached.values[index]));
            } else {
              const unsigned state = axis_x + 8U * axis_y + 64U * axis_z;
              const double independent = (state + 0.25) * (rho + Order + 1U);
              assert(cached.value(state, rho, workspace, nullptr) == independent);
            }
          }
      assert(evaluations == expected);
      GradientSpatialCoulombValues<Order, false> scalar;
      evaluations = 0;
      scalar.prepare(total, empty, empty, empty, rho, workspace, nullptr);
      assert(evaluations == 0);
      assert(scalar.value(0U, rho, workspace, nullptr) == 0.25 * (rho + Order + 1U));
      assert(evaluations == 1U);
    }
  }
}
int main() {
  static_assert(GradientSpatialCoulombValues<4U, true>::capacity == 36U);
  static_assert(GradientSpatialCoulombValues<5U, true>::capacity == 48U);
  static_assert(GradientSpatialCoulombValues<6U, true>::capacity == 64U);
  exercise<4U>();
  exercise<5U>();
  exercise<6U>();
}
"""
