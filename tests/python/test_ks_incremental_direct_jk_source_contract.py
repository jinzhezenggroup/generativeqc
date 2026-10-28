from __future__ import annotations

import subprocess
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from conftest import NativeCxx

ROOT = Path(__file__).resolve().parents[2]


def _source(path: str) -> str:
    return (ROOT / path).read_text()


def test_cuda_ks_incremental_direct_jk_is_capability_based() -> None:
    method = _source("src/methods/dft_method_preparation.cpp")
    ks = _source("src/dft/cuda_ks.cpp")

    assert "GENERATIVEQC_KS_INCREMENTAL_DIRECT_JK" in method
    assert (
        "direct_jk_incremental_exact_eligible(*options.resolved_fock_build)" in method
    )
    assert (
        "CUDA KS incremental Direct-J/K requires one strict-FP64 exact Direct provider"
        in ks
    )

    # The legacy experiment remains reproducible, but the generic selector is
    # not restricted to one semilocal family or one spin mode.
    assert "GENERATIVEQC_PBE0_INCREMENTAL_DIRECT_JK" in method
    generic = method.index("const bool generic_incremental_direct_jk")
    legacy = method.index("if (legacy_pbe0_incremental_direct_jk)", generic)
    generic_block = method[generic:legacy]
    assert "SemilocalFamily::Pbe" not in generic_block
    assert "FockSpin::Restricted" not in generic_block


def test_cuda_ks_incremental_direct_jk_only_changes_linear_fock_inputs() -> None:
    ks = _source("src/dft/cuda_ks.cpp")

    assert "const double* jk_density = prepare_incremental_jk_density();" in ks
    assert "provider, jk_density, spins == 2 ? jk_density + matrix : nullptr" in ks
    assert "provider, *range_correction, jk_density" in ks
    assert "finalize_incremental_jk_components();" in ks

    # Full-density XC/NLC inputs are executed and checked for both quadrature
    # paths by test_ks_mixed_retry_handoff, including a distinct delta sentinel.


def test_cuda_ks_incremental_direct_jk_preserves_full_finalization() -> None:
    ks = _source("src/dft/cuda_ks.cpp")

    assert (
        "!precision_schedule.any_lower_precision() && !incremental_direct_jk && spins == 1"
        in ks
    )
    assert "++output.incremental_direct_jk.post_scf_full_builds;" in ks
    assert "incremental_anchor_range_exchange" in ks
    assert "output.incremental_direct_jk.anchor_updates" in ks
    assert (
        "incremental_policy_options.screening_tolerance = strategy.screening_tolerance;"
        in ks
    )
    assert "sum(options.max_iterations, kMaximumFinalCorrections)" in ks


def test_cuda_ks_final_closure_domain(tmp_path: Path, native_cxx: NativeCxx) -> None:
    """Execute the production gate for every existing and budget-Fock domain."""
    ks = _source("src/dft/cuda_ks.cpp")
    begin = ks.index("const bool budget_final_validation =")
    end = ks.index(";", ks.index("const bool strict_final_closure =", begin))
    gate = ks[begin : end + 1]
    source = tmp_path / "final_closure.cpp"
    source.write_text(
        """
#include <cassert>
#include <cmath>
#include <vector>
#include <initializer_list>
struct Provider {
  struct System { std::vector<int> ecp_terms; } value;
  const System& system() const { return value; }
};
struct Options { double density_tolerance; };
int main() {
  for (bool automatic_program : {false, true})
   for (bool budget_fock : {false, true})
    for (bool budget_stationarity : {false, true})
      for (double tolerance : {1e-6, std::nextafter(1e-4, 0.0), 1e-4,
                               std::nextafter(1e-4, 1.0), 1e-3})
        for (bool incremental_direct_jk : {false, true})
          for (unsigned spins : {1U, 2U})
            for (bool ecp : {false, true}) {
              Provider provider;
              if (ecp) provider.value.ecp_terms.push_back(1);
              const Options options{tolerance};
"""
        + gate
        + """
              bool expected = automatic_program || incremental_direct_jk || spins == 2 || ecp;
              if (budget_fock && (!budget_stationarity || tolerance < 1e-4))
                expected = true;
              assert(strict_final_closure == expected);
            }
}
""",
        encoding="utf-8",
    )
    executable = native_cxx.build_executable(
        [source], tmp_path / "final_closure", compile_args=["-std=c++17"]
    )
    completed = subprocess.run(
        [executable], capture_output=True, text=True, check=False, timeout=10
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr


def test_incremental_final_operator_requires_completed_physical_closure(
    tmp_path: Path, native_cxx: NativeCxx
) -> None:
    """Execute the admission predicate instead of pinning its internal spelling.

    The real CUDA E/F regression exercises final-state reading end to end;
    this host truth table additionally covers rejected incomplete histories.
    """
    ks = _source("src/dft/cuda_ks.cpp")
    begin = ks.index("const bool ordinary_final_operator =")
    gate = ks[begin : ks.index(";", begin) + 1]
    source = tmp_path / "incremental_final_operator.cpp"
    source.write_text(
        """
#include <cassert>
#include <initializer_list>
struct Consumer {
  bool incremental_direct_jk, standard, final_closure;
  unsigned incremental_energy_full_builds;
  bool standard_convergence_active() const { return standard; }
  bool admitted() const {
"""
        + gate
        + """
    return ordinary_final_operator;
  }
};
int main() {
  for (bool incremental : {false, true})
    for (bool standard : {false, true})
      for (bool closed : {false, true})
        for (unsigned full_builds : {0U, 1U, 2U, 3U}) {
          Consumer consumer{incremental, standard, closed, full_builds};
          bool expected = !incremental;
          if (incremental && standard && closed && full_builds > 1U)
            expected = true;
          assert(consumer.admitted() == expected);
        }
}
""",
        encoding="utf-8",
    )
    executable = native_cxx.build_executable(
        [source], tmp_path / "incremental_final_operator", compile_args=["-std=c++17"]
    )
    completed = subprocess.run(
        [executable], capture_output=True, text=True, check=False, timeout=10
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
