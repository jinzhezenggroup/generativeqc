"""Execute the shared cold-density proof gates, including refusal boundaries."""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from conftest import NativeCxx


def test_density_seed_proof_and_cold_admission(
    tmp_path: Path, required_native_cxx: NativeCxx
) -> None:
    """Invalid spectra/errors cannot authorize a factor or weaken dense fallback."""
    root = Path(__file__).resolve().parents[2]
    source = tmp_path / "proof.cpp"
    source.write_text(
        r"""
#include <cassert>
#include <cmath>
#include <limits>
#include <vector>
#include "scf/df_density_seed_policy.hpp"
#include "dft/cuda_ks_cold_factor_policy.hpp"
int main() {
  using namespace generativeqc::scf;
  const auto inspect = [](std::initializer_list<double> v) {
    return density_seed_spectrum(std::span<const double>(v.begin(), v.size()));
  };
  const auto valid = inspect({-1e-14, 0., 1e-14, 0.3, 2.7});
  assert(valid.valid && valid.rank == 2 && valid.within_discarded_budget());
  // Occupations in a nonorthogonal AO basis are not Euclidean eigenvalues.
  const auto ensemble = inspect({0.1, 0.2, 0.3, 0.4});
  assert(ensemble.valid && ensemble.rank == 4);
  for (const auto v : {inspect({-1e-12, 2.}), inspect({2., 1.}),
                       inspect({0., std::numeric_limits<double>::quiet_NaN()}),
                       inspect({0., std::numeric_limits<double>::infinity()})})
    assert(!v.valid && !v.within_discarded_budget());
  std::vector<double> discarded(101, 1e-13);
  const auto too_many = density_seed_spectrum(discarded);
  assert(too_many.valid && too_many.rank == 0 && !too_many.within_discarded_budget());
  assert(density_seed_reconstruction_accepted(1e-12, 1e-13));
  assert(!density_seed_reconstruction_accepted(std::nextafter(1e-12, 1.), 1e-13));
  assert(!density_seed_reconstruction_accepted(1e-12, std::nextafter(1e-13, 1.)));
  for (double poison : {-1., std::numeric_limits<double>::quiet_NaN(),
                         std::numeric_limits<double>::infinity()}) {
    assert(!density_seed_reconstruction_accepted(poison, 0.));
    assert(!density_seed_reconstruction_accepted(0., poison));
  }
  using generativeqc::dft::cold_fitted_factor_requested;
  assert(cold_fitted_factor_requested(nullptr, 768, 160));
  assert(!cold_fitted_factor_requested("0", 768, 160));
  assert(!cold_fitted_factor_requested(nullptr, 383, 80));
  assert(cold_fitted_factor_requested("1", 17, 5));
  assert(!cold_fitted_factor_requested("1", 17, 9));
  assert(!cold_fitted_factor_requested("1", 768, 0));
  for (const char* bad : {"", "auto", "2", "01", "yes"}) {
    bool rejected = false;
    try { cold_fitted_factor_requested(bad, 768, 160); }
    catch (const std::invalid_argument&) { rejected = true; }
    assert(rejected);
  }
}
"""
    )
    executable = required_native_cxx.build_executable(
        [source], tmp_path / "proof", compile_args=["-std=c++20", f"-I{root / 'src'}"]
    )
    subprocess.run([executable], check=True, timeout=10)
