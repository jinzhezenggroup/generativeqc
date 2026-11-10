"""Execute stopping admission and HF work publication without a CUDA device."""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from conftest import NativeCxx


ROOT = Path(__file__).resolve().parents[2]


def test_orthogonal_uks_pulay_metric_preserves_qualified_domain_and_stopping(
    tmp_path: Path, required_native_cxx: NativeCxx
) -> None:
    """A proposal metric cannot promote other providers or weaken final proof."""
    source = tmp_path / "uks_diis_metric.cpp"
    source.write_text(
        r"""
#include <cassert>
#include "dft/cuda_ks_diis_policy.hpp"
#include "dft/ks_stopping_policy.hpp"
int main() {
  using namespace generativeqc::dft;
  const CudaKsDiisMetricDomain qualified{
      2, semilocal_family_code(SemilocalFamily::Pbe), true, true, true, true};
  assert(cuda_ks_orthogonal_diis_requested(qualified));
  for (unsigned spins : {0U, 1U, 3U}) {
    auto rejected = qualified;
    rejected.spins = spins;
    assert(!cuda_ks_orthogonal_diis_requested(rejected));
  }
  for (auto family : {SemilocalFamily::Lda, SemilocalFamily::B3lyp,
                     SemilocalFamily::R2scan, SemilocalFamily::Wb97mv}) {
    auto rejected = qualified;
    rejected.functional = semilocal_family_code(family);
    assert(!cuda_ks_orthogonal_diis_requested(rejected));
  }
  for (auto member : {&CudaKsDiisMetricDomain::exact_primary,
                      &CudaKsDiisMetricDomain::semilocal_only,
                      &CudaKsDiisMetricDomain::strict_fp64,
                      &CudaKsDiisMetricDomain::all_electron}) {
    auto rejected = qualified;
    rejected.*member = false;
    assert(!cuda_ks_orthogonal_diis_requested(rejected));
  }
  KsStoppingState state{2, false, true, true, 1e-12, 1e-10, 1e-10,
                        5e-9, 1e-9, 1e-5, 1e-9};
  assert(!ordinary_ks_converged(state));
  assert(!early_ks_final_probe_stationary(state, true));
  state.maximum_residual = 1e-10;
  assert(ordinary_ks_converged(state));
  assert(early_ks_final_probe_stationary(state, true));
}
"""
    )
    executable = required_native_cxx.build_executable(
        [source],
        tmp_path / "uks_diis_metric",
        compile_args=[
            "-std=c++20",
            f"-I{ROOT / 'src'}",
            f"-I{ROOT / 'include'}",
        ],
    )
    subprocess.run([executable], check=True, capture_output=True, text=True, timeout=10)


def test_default_ks_force_and_energy_share_orbital_stopping_but_reference_is_strict(
    tmp_path: Path, required_native_cxx: NativeCxx
) -> None:
    """Forces do not add an AO gate; strict controls/reference remain explicit."""
    source = tmp_path / "force_stationarity.cpp"
    source.write_text(
        r"""
#include <cassert>
#include <utility>
#include "dft/ks_stopping_policy.hpp"
#include "scf/convergence_limits.hpp"
int main() {
  using namespace generativeqc;
  assert(!dft::strict_ks_convergence_requested(nullptr));
  assert(!dft::strict_ks_convergence_requested(""));
  assert(dft::strict_ks_convergence_requested("strict"));
  bool rejected = false;
  try { dft::strict_ks_convergence_requested("loose"); }
  catch (const std::invalid_argument&) { rejected = true; }
  assert(rejected);
  for (bool standard : {false, true})
    for (bool nonlocal : {false, true})
      for (bool reference : {false, true}) {
          const bool admitted = dft::standard_ks_stationarity_allowance(
              standard, nonlocal, reference);
          assert(admitted == (standard && (nonlocal || !reference)));
          const double tolerance = admitted
              ? scf::scf_stationarity_tolerance(1e-9, 1e-5, 1e-9) : 1e-9;
          if (standard && !nonlocal && reference) assert(tolerance == 1e-9);
        }
  assert(dft::standard_ks_stationarity_allowance(true, false, false));
  assert(!dft::standard_ks_stationarity_allowance(true, false, true));
  for (auto controls : {std::pair{1e-10,1e-5}, std::pair{1e-9,1e-8}}) {
    const bool standard = scf::standard_scf_convergence_requested(controls.first, controls.second);
    assert(!standard);
    assert(!dft::standard_ks_stationarity_allowance(standard, false, false));
  }
  dft::KsStoppingState state{2,false,true,true,1e-10,1e-3,1e-6,1e-6,1e-9,1e-5,1e-9};
  assert(!dft::ordinary_ks_converged(state));
  assert(dft::orbital_ks_converged(state, 2e-5));
  assert(!dft::orbital_ks_converged(state, std::sqrt(state.energy_tolerance)));
  assert(!dft::orbital_ks_converged(state, std::numeric_limits<double>::quiet_NaN()));
  state.full_energy_history = false;
  assert(!dft::orbital_ks_converged(state, 1e-10));
}
"""
    )
    executable = required_native_cxx.build_executable(
        [source],
        tmp_path / "force_stationarity",
        compile_args=["-std=c++20", f"-I{ROOT / 'src'}"],
    )
    subprocess.run([executable], check=True, capture_output=True, text=True, timeout=10)


def test_early_final_probe_cannot_reopen_force_stationarity_allowance(
    tmp_path: Path, required_native_cxx: NativeCxx
) -> None:
    """A small RMS cannot hide a force-relevant AO maximum in an early proof."""
    source = tmp_path / "early_force_stationarity.cpp"
    source.write_text(
        r"""
#include <cassert>
#include <limits>
#include "dft/ks_stopping_policy.hpp"
int main() {
  using namespace generativeqc::dft;
  KsStoppingState state{2,false,true,true,1e-12,1e-10,1e-10,5e-9,1e-9,1e-5,1e-9};
  assert(!ordinary_ks_converged(state));
  assert(early_ks_final_probe_stationary(state, false));
  assert(!early_ks_final_probe_stationary(state, true));
  state.maximum_residual = 5e-10;
  assert(early_ks_final_probe_stationary(state, true));
  state.maximum_residual = state.stationarity_tolerance;
  assert(!early_ks_final_probe_stationary(state, true));
  state.density_tolerance = 1e-10;
  state.maximum_residual = 2e-10;
  for (bool force_or_reference : {false, true})
    assert(!early_ks_final_probe_stationary(state, force_or_reference));
  state.density_tolerance = 1e-5;
  state.maximum_residual = 0;
  state.residual = state.stationarity_tolerance;
  assert(!early_ks_final_probe_stationary(state, true));
  state.residual = 0;
  state.maximum_residual = std::numeric_limits<double>::quiet_NaN();
  for (bool force_or_reference : {false, true})
    assert(!early_ks_final_probe_stationary(state, force_or_reference));
}
"""
    )
    executable = required_native_cxx.build_executable(
        [source],
        tmp_path / "early_force_stationarity",
        compile_args=["-std=c++20", f"-I{ROOT / 'src'}"],
    )
    subprocess.run([executable], check=True, capture_output=True, text=True, timeout=10)


def test_early_final_default_requires_the_complete_qualified_model(
    tmp_path: Path, required_native_cxx: NativeCxx
) -> None:
    """A family label, altered mixture or nondefault grid cannot grant promotion."""
    source = tmp_path / "stopping.cpp"
    source.write_text(
        r"""
#include <cassert>
#include <limits>
#include "dft/ks_stopping_policy.hpp"
int main() {
  using namespace generativeqc::dft;
  const auto pbe = semilocal_family_code(SemilocalFamily::Pbe);
  const auto r2scan = semilocal_family_code(SemilocalFamily::R2scan);
  const auto b3lyp = semilocal_family_code(SemilocalFamily::B3lyp);
  assert(early_final_probe_default_model(pbe,1,1,0,true));
  assert(early_final_probe_default_model(pbe,.75,1,-.125,true));
  assert(early_final_probe_default_model(r2scan,1,1,0,true));
  assert(early_final_probe_default_model(b3lyp,1,1,-.1,true));
  for (auto family : {pbe, r2scan, b3lyp}) {
    assert(!early_final_probe_default_model(family,1,1,0,false));
    assert(!early_final_probe_default_model(family,1,.9,0,true));
    assert(!early_final_probe_default_model(family,.5,1,-.25,true));
    assert(!early_final_probe_default_model(family,1,1,
        std::numeric_limits<double>::quiet_NaN(),true));
  }
  for (auto family : {0U, 4U, 5U, 100U, 1024U, 0xffffffffU})
    assert(!early_final_probe_default_model(family,1,1,-.27,true));
  assert(!early_final_probe_default_model(r2scan,.75,1,-.125,true));
  assert(!early_final_probe_default_model(b3lyp,1,1,0,true));
  KsStoppingState state{2,false,true,true,1e-12,1e-10,1e-11,1e-11,1e-10,1e-8,1e-9};
  assert(ordinary_ks_converged(state));
  state.target_ready = false;
  assert(!ordinary_ks_converged(state));
  state.target_ready = true;
  state.full_energy_history = false;
  assert(!ordinary_ks_converged(state));
  state.full_energy_history = true;
  state.iterations = 1;
  assert(!ordinary_ks_converged(state));
  state.warm_energy_baseline = true;
  assert(ordinary_ks_converged(state));
  state.maximum_residual = state.stationarity_tolerance;
  assert(!ordinary_ks_converged(state));
  state.maximum_residual = std::numeric_limits<double>::quiet_NaN();
  assert(!ordinary_ks_converged(state));
}
"""
    )
    executable = required_native_cxx.build_executable(
        [source],
        tmp_path / "stopping",
        compile_args=["-std=c++20", f"-I{ROOT / 'src'}"],
    )
    subprocess.run([executable], check=True, capture_output=True, text=True, timeout=10)


def test_hf_work_never_infers_from_iterations_or_hides_an_attempt(
    tmp_path: Path, required_native_cxx: NativeCxx
) -> None:
    """Missing/discarded work is unavailable even after a successful solve."""
    source = tmp_path / "work.cpp"
    source.write_text(
        r"""
#include <cassert>
#include <limits>
#include "scf/hf_fock_work.hpp"
int main() {
  using namespace generativeqc::scf;
  ScfResult result;
  result.converged = true;
  result.iterations = 123;
  assert(!completed_hf_fock_work(result,false,false));
  assert(!completed_hf_fock_work(result,false,true));
  result.fock_builds = 17;
  assert(completed_hf_fock_work(result,false,true) == 17);
  auto& work = result.precision;
  work.operator_work_counters_valid = true;
  work.mixed_stage_fock_builds = 9;
  work.strict_stage_fock_builds = 3;
  work.post_scf_fock_builds = 2;
  assert(!completed_hf_fock_work(result,false,false));
  result.executed_fock_builds = 19;
  assert(completed_hf_fock_work(result,false,false) == 19);
  assert(!completed_hf_fock_work(result,true,false));
  assert(!completed_hf_fock_work(result,true,true));
  auto& guess = result.preliminary_guess;
  guess.target_attempts = 2;
  assert(!completed_hf_fock_work(result,false,false));
  assert(!completed_hf_fock_work(result,false,true));
  guess.target_attempts = 1;
  guess.work_counters_complete = false;
  assert(!completed_hf_fock_work(result,false,false));
  assert(!completed_hf_fock_work(result,false,true));
  guess.work_counters_complete = true;
  guess.discarded_target_fock_builds = 1;
  assert(!completed_hf_fock_work(result,false,false));
  guess.discarded_target_fock_builds = 0;
  result.executed_fock_builds.reset();
  assert(!completed_hf_fock_work(result,false,false));
  result.converged = false;
  assert(!completed_hf_fock_work(result,false,true));
}
"""
    )
    executable = required_native_cxx.build_executable(
        [source],
        tmp_path / "work",
        compile_args=["-std=c++20", f"-I{ROOT / 'src'}", f"-I{ROOT / 'include'}"],
    )
    subprocess.run([executable], check=True, capture_output=True, text=True, timeout=10)
