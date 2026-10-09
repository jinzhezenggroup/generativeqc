"""Execute the native policy with SCF and resource-planner boundaries injected."""

import shutil
import subprocess
from pathlib import Path

import pytest
from _cc_owner_test_support import compile_owner

ROOT = Path(__file__).resolve().parents[2]


def test_native_df_guess_admission_and_refusal(tmp_path: Path) -> None:
    """Protect topology, budget, density-only ownership and unchanged target controls."""
    compiler = shutil.which("c++")
    if not compiler:
        pytest.skip("requires host C++")
    source = tmp_path / "policy.cpp"
    source.write_text(
        r"""
#include <cassert>
#include <cstdlib>
#include <stdexcept>
#include "methods/df_hf_guess.hpp"
#include "molecule/basis.hpp"
#include "scf/mean_field.hpp"
#include "scf/density_fitting.hpp"
using namespace generativeqc;
int calls{}, behavior{};
bool resident = true;
namespace generativeqc::molecule {
std::size_t ao_count(const core::System& system) noexcept {
  std::size_t functions{};
  for (const auto& shell : system.shells) functions += 2 * shell.angular_momentum + 1;
  return functions;
}
std::size_t cartesian_ao_count(const core::System& system) noexcept {
  std::size_t functions{};
  for (const auto& shell : system.shells) functions += cartesian_count(shell.angular_momentum);
  return functions;
}
generativeqc_status validate_and_normalize(core::System& system, std::string&) {
  assert(system.atoms.size() == 8);
  assert(system.atoms[1].position[0] == 2.0);
  assert(system.basis_representation == GENERATIVEQC_BASIS_SPHERICAL);
  assert(ao_count(system) == 484);
  return GENERATIVEQC_STATUS_SUCCESS;
}
}
namespace generativeqc::scf {
std::size_t density_fitting_source_metadata_bytes(std::size_t, std::size_t, std::size_t,
    std::size_t, std::size_t, std::size_t) {return 0;}
std::size_t density_fitting_scf_diis_device_bytes(std::size_t, std::size_t, unsigned) noexcept {return 0;}
DensityFittingTilePlan plan_requested_density_fitting_tiles(DfPairStorageRequest, std::size_t,
    std::size_t, std::size_t, std::size_t, std::size_t, std::size_t,
    std::size_t, bool generated, std::size_t rank) {
  assert(generated && rank > 0);
  DensityFittingTilePlan plan;
  plan.stores_full_three_center = resident;
  plan.peak_workspace_bytes = 128ULL << 20;
  return plan;
}
ScfResult run_rhf_density_fitting_cuda(
    const core::System&, const core::System&, const ScfOptions& options, int device,
    const std::vector<double>* density, initial_guess::OverlapOrthogonalizer*) {
  ++calls;
  assert(device == 7 && density == nullptr);
  assert(options.max_iterations == 32 && options.diis_history == 9);
  assert(options.energy_tolerance == 1e-4 && options.density_tolerance == 1e-4);
  assert(options.precision_mode == GENERATIVEQC_PRECISION_FP64);
  assert(options.density_fitting_mode == GENERATIVEQC_DENSITY_FITTING_CUDA);
  assert(!options.density_fitting_host_retry);
  assert(!options.compute_forces && !options.export_physical_reference);
  assert(options.reference_memory_budget_bytes == (512ULL << 20));
  assert(options.density_fitting_memory_budget_bytes == options.reference_memory_budget_bytes);
  if (behavior == 2) throw std::bad_alloc();
  ScfResult result;
  result.converged = behavior != 1;
  result.iterations = 13;
  result.density = {2.0, 0.0, 0.0, 0.0};
  return result;
}
void validate_hf_warm_density(const core::System&, generativeqc_method method,
                              const std::vector<double>& density,
                              const initial_guess::EigenOperation&) {
  assert(method == GENERATIVEQC_METHOD_RHF && density.size() == 4);
  if (behavior == 3) throw std::invalid_argument("invalid provisional density");
}
}
int main() {
  unsetenv("GENERATIVEQC_DF_CCSDT_REFERENCE_GUESS");
  using namespace methods::detail;
  core::System source;
  source.basis_representation = GENERATIVEQC_BASIS_SPHERICAL;
  source.atoms.resize(8);
  for (auto& atom : source.atoms) atom.atomic_number = 1;
  source.atoms[0].atomic_number = source.atoms[1].atomic_number = 6;
  source.atoms[1].position[0] = 2.0;
  source.shells.resize(230);
  core::System correlation;
  generativeqc_method_descriptor descriptor{};
  descriptor.correlation_memory_budget_bytes = 1ULL << 30;
  descriptor.diis_history = 9;
  descriptor.energy_tolerance = 1e-12;
  descriptor.density_tolerance = 1e-11;
  const auto prepare = [&](bool enabled = true, std::size_t retained = 0) {
    return prepare_df_hf_guess(source, correlation, descriptor, retained, 7, enabled);
  };
  auto guess = prepare();
  assert(calls == 1 && guess.outcome == DFHFGuessOutcome::Used);
  assert(guess.iterations == 13 && guess.auxiliary_functions == 484);
  assert(guess.density.size() == 4 && guess.work_counters_complete);
  assert(descriptor.energy_tolerance == 1e-12 && descriptor.density_tolerance == 1e-11);
  assert(prepare(false).outcome == DFHFGuessOutcome::Disabled);
  setenv("GENERATIVEQC_DF_CCSDT_REFERENCE_GUESS", "direct", 1);
  assert(prepare().outcome == DFHFGuessOutcome::Disabled && calls == 1);
  setenv("GENERATIVEQC_DF_CCSDT_REFERENCE_GUESS", "bogus", 1);
  try { prepare(); return 1; } catch (const std::invalid_argument&) {}
  setenv("GENERATIVEQC_DF_CCSDT_REFERENCE_GUESS", "auto", 1);
  for (std::size_t count : {199, 401}) {
    source.shells.resize(count);
    assert(prepare().outcome == DFHFGuessOutcome::Used);
  }
  assert(calls == 3);
  resident = false;
  assert(prepare().outcome == DFHFGuessOutcome::BudgetSkipped && calls == 3);
  resident = true;
  source.shells.resize(34);
  assert(prepare().outcome == DFHFGuessOutcome::CacheSkipped);
  source.shells[0].angular_momentum = 2;
  auto small = prepare();
  assert(small.outcome == DFHFGuessOutcome::WorkSkipped);
  assert(small.auxiliary_functions == 484 && small.cartesian_functions == 39);
  assert(small.work_amortization_ratio < 1.0 && calls == 3);
  source.shells[0].angular_momentum = 0;
  source.shells.resize(4096);
  auto large = prepare();
  assert(large.outcome == DFHFGuessOutcome::BudgetSkipped);
  assert(large.preparation_peak_bytes > large.value_budget_bytes && calls == 3);
  source.shells.resize(230);
  source.atoms.resize(80);
  for (std::size_t index = 8; index < source.atoms.size(); ++index)
    source.atoms[index].atomic_number = 1;
  auto auxiliary_heavy = prepare();
  assert(auxiliary_heavy.outcome == DFHFGuessOutcome::WorkSkipped);
  assert(auxiliary_heavy.cartesian_functions == 230);
  assert(auxiliary_heavy.auxiliary_functions > 484 && calls == 3);
  source.atoms.resize(8);
  source.atoms[0].atomic_number = 8;
  assert(prepare().outcome == DFHFGuessOutcome::Ineligible);
  source.atoms[0].atomic_number = 6;
  source.basis_representation = GENERATIVEQC_BASIS_CARTESIAN;
  assert(prepare().outcome == DFHFGuessOutcome::Ineligible);
  source.basis_representation = GENERATIVEQC_BASIS_SPHERICAL;
  source.shells[0].angular_momentum = 4;
  assert(prepare().outcome == DFHFGuessOutcome::Ineligible);
  source.shells[0].angular_momentum = 0;
  assert(prepare(true, 900ULL << 20).outcome == DFHFGuessOutcome::BudgetSkipped);
  descriptor.correlation_memory_budget_bytes = 128ULL << 20;
  assert(prepare().outcome == DFHFGuessOutcome::BudgetSkipped && calls == 3);
  descriptor.correlation_memory_budget_bytes = 1ULL << 30;
  behavior = 1;
  guess = prepare();
  assert(guess.outcome == DFHFGuessOutcome::Failed && guess.density.empty());
  assert(guess.iterations == 13 && guess.work_counters_complete);
  for (int refusal : {2, 3}) {
    behavior = refusal;
    guess = prepare();
    assert(guess.outcome == DFHFGuessOutcome::Failed && guess.density.empty());
    assert(!guess.work_counters_complete);
  }
  assert(calls == 6);
}
"""
    )
    binary = tmp_path / "policy"
    compile_owner(
        compiler, tmp_path, [source, ROOT / "src/methods/df_hf_guess.cpp"], binary
    )
    subprocess.run([str(binary)], check=True, capture_output=True, timeout=10)
