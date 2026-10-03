"""Frozen operation-order gates for the shared native XC feature traversal.

This is a compiler/dispatch regression comparator, not a scientific oracle.
Independent point/integration physics fixtures remain in the native tests.
"""

import shutil
import subprocess
from pathlib import Path

import pytest
from generativeqc_compiler.dft.feature_policy import emit_feature_policy

ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize("optimization", ["-O0", "-O2"])
def test_mask_specialization_preserves_frozen_features(
    tmp_path: Path, optimization: str
) -> None:
    compiler = shutil.which("c++")
    cache = shutil.which("ccache")
    if compiler is None or cache is None:
        pytest.skip("requires ccache and a C++20 compiler")
    source = (ROOT / "src/dft/xc.cpp").read_text()
    begin = source.index("template <int IngredientMask, typename Function>\n")
    end = source.index("\nvoid sample_xc_capacity", begin)
    frozen = (ROOT / "tests/native/fixtures/xc_features_90c8b573.hpp").read_text()
    harness = tmp_path / "feature_masks.cpp"
    harness.write_text(
        r"""
#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <limits>
#include <type_traits>
#include <utility>
#include <vector>

namespace scf {
// Only the immutable factor accessors consumed by the actual feature body
// are substituted. Native density-source tests cover the real factor owner,
// exact witness validation, state/response roles and generated point ABI.
struct OccupiedDensityFactor {
  std::size_t occupied{};
  std::vector<double> weighted;
  std::size_t rank() const { return occupied; }
  const std::vector<double>& values() const { return weighted; }
};
}
namespace generated {
"""
        + emit_feature_policy()
        + "\n}\n"
        + frozen
        + source[begin:end]
        + r"""
bool same(double actual, double expected) {
  // Finite results, signed zeros and infinities retain their exact bits.
  // NaN payloads are not a public contract; preserve their classification.
  if (std::isnan(expected)) return std::isnan(actual);
  return std::bit_cast<std::uint64_t>(actual) == std::bit_cast<std::uint64_t>(expected);
}

bool check(const std::vector<double>& phi, const std::array<std::vector<double>, 3>& jets,
           const std::vector<double>& density, const scf::OccupiedDensityFactor* factor,
           unsigned mask, bool null_jets = false) {
  const std::array<const double*, 3> derivatives =
      null_jets ? std::array<const double*, 3>{}
                : std::array<const double*, 3>{jets[0].data(), jets[1].data(), jets[2].data()};
  volatile unsigned dynamic_mask = mask;
  const auto actual = rks_features(phi.data(), derivatives, phi.size(), density, factor,
                                   dynamic_mask);
  const auto expected = frozen_rks_features(phi.data(), derivatives, phi.size(), density, factor,
                                            dynamic_mask);
  for (unsigned feature = 0; feature < 5; ++feature) {
    if (!same(actual[feature], expected[feature])) {
      std::cerr << "feature mismatch: mask=" << mask << " n=" << phi.size()
                << " factor=" << (factor != nullptr) << " feature=" << feature << '\n';
      return false;
    }
  }
  return true;
}

int main() {
  // The unsupported/internal masks deliberately retain legacy behavior. For
  // example D always accumulates rho, whereas C's generated bilinear gates it.
  const unsigned masks[]{1U, 7U, 15U, 0U, 2U, 3U, 4U, 6U, 8U, 9U, 14U, 16U,
                         std::numeric_limits<unsigned>::max()};
  const double denorm = std::numeric_limits<double>::denorm_min();
  const double normal = std::numeric_limits<double>::min();
  const double scales[]{0.0, -0.0, denorm, normal, 1.0e-160, 1.0e-80, 1.0, 1.0e80, 1.0e160};
  for (std::size_t n : {1U, 2U, 5U, 13U}) {
    for (double scale : scales) {
      std::vector<double> phi(n), density(n * n);
      std::array<std::vector<double>, 3> jets;
      for (auto& jet : jets) jet.resize(n);
      for (std::size_t mu = 0; mu < n; ++mu) {
        phi[mu] = scale * (mu % 3 == 0 ? -0.75 : 0.5);
        for (unsigned axis = 0; axis < 3; ++axis)
          jets[axis][mu] = scale * ((mu + axis) % 3 == 0 ? 0.0 : (axis == 1 ? -0.25 : 0.5));
        // Signed/indefinite response-like D with accepted near-symmetry.
        for (std::size_t nu = 0; nu < n; ++nu) {
          const double value = (mu + nu) % 2 ? -0.375 : 0.25;
          density[mu * n + nu] = value + (mu < nu ? 5.0e-13 : (mu > nu ? -5.0e-13 : 0.0));
        }
      }
      for (unsigned mask : masks) {
        if (!check(phi, jets, density, nullptr, mask)) return 1;
        if (mask == 1 && !check(phi, jets, density, nullptr, mask, true)) return 2;
        for (std::size_t rank : {std::size_t{0}, std::size_t{1}, n}) {
          scf::OccupiedDensityFactor factor{rank, std::vector<double>(n * rank)};
          for (std::size_t mu = 0; mu < n; ++mu)
            for (std::size_t o = 0; o < rank; ++o)
              factor.weighted[mu * rank + o] = std::sqrt(2.0) *
                  ((mu + o) % 3 == 0 ? -0.25 : ((mu + o) % 3 == 1 ? 0.5 : 0.0));
          if (!check(phi, jets, density, &factor, mask)) return 3;
          if (mask == 1 && !check(phi, jets, density, &factor, mask, true)) return 4;
        }
      }
      // Linear response features must not reject negative density, and exact
      // zero D must not be replaced by an occupied-state approximation.
      std::fill(density.begin(), density.end(), 0.0);
      for (unsigned mask : masks)
        if (!check(phi, jets, density, nullptr, mask)) return 5;
    }
  }

  // Tiny and extreme density entries with ordinary jets exercise a different
  // underflow/overflow order than merely rescaling the jets.
  const std::vector<double> phi{1.0, -0.5, 0.25};
  const std::array<std::vector<double>, 3> jets{
      std::vector<double>{0.0, 0.5, -0.25}, std::vector<double>{-0.5, 0.25, 0.0},
      std::vector<double>{0.25, -0.5, 1.0}};
  for (double scale : {denorm, normal, 1.0, std::numeric_limits<double>::max()}) {
    std::vector<double> density(9);
    for (std::size_t i = 0; i < density.size(); ++i)
      density[i] = (i % 2 ? -0.5 : 0.5) * scale;
    for (unsigned mask : masks)
      if (!check(phi, jets, density, nullptr, mask)) return 6;
  }
  std::cout << "frozen XC D/C feature operation order retained\n";
}
"""
    )
    executable = tmp_path / "feature_masks"
    compiled = subprocess.run(
        [
            cache,
            compiler,
            "-std=c++20",
            optimization,
            str(harness),
            "-o",
            str(executable),
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert compiled.returncode == 0, compiled.stderr
    checked = subprocess.run(
        [str(executable)], check=False, capture_output=True, text=True, timeout=10
    )
    assert checked.returncode == 0, checked.stdout + checked.stderr
