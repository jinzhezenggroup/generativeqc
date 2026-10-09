#include "methods/df_hf_guess.hpp"

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <limits>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>

#include "methods/df_hf_guess_basis.hpp"
#include "molecule/basis.hpp"
#include "posthf/capacity.hpp"
#include "scf/mean_field.hpp"

namespace generativeqc::methods::detail {
DFHFGuess prepare_df_hf_guess(const core::System& system, const core::System& correlation_auxiliary,
                              const generativeqc_method_descriptor& descriptor,
                              std::size_t retained_bytes, int device_id, bool enabled) {
  DFHFGuess guess;
  if (const char* selector = std::getenv("GENERATIVEQC_DF_CCSDT_REFERENCE_GUESS")) {
    const std::string_view value(selector);
    if (value != "direct" && value != "auto")
      throw std::invalid_argument("GENERATIVEQC_DF_CCSDT_REFERENCE_GUESS must be auto or direct");
    enabled = enabled && value == "auto";
  }
  if (!enabled || device_id < 0) return guess;
  guess.outcome = DFHFGuessOutcome::Ineligible;
  const auto functions = molecule::ao_count(system);
  if (functions < 200 || functions > 400 ||
      system.basis_representation != GENERATIVEQC_BASIS_SPHERICAL || system.multiplicity != 1 ||
      system.charge != 0 || !system.ecp_terms.empty() ||
      std::any_of(system.shells.begin(), system.shells.end(),
                  [](const auto& shell) { return shell.angular_momentum > 3; }) ||
      std::any_of(system.atoms.begin(), system.atoms.end(), [](const auto& atom) {
        return atom.atomic_number != 1 && atom.atomic_number != 6;
      }))
    return guess;
  const auto requested = descriptor.correlation_memory_budget_bytes;
  if (requested > static_cast<std::uint64_t>(INT64_MAX) ||
      requested > std::numeric_limits<std::size_t>::max())
    throw std::invalid_argument("RCCSD budget exceeds numeric capacity");
  const auto budget = requested ? static_cast<std::size_t>(requested) : 256ULL << 20;
  const auto outside =
      posthf::checked_add(retained_bytes, posthf::source_capacity(correlation_auxiliary));
  guess.outcome = DFHFGuessOutcome::BudgetSkipped;
  if (outside >= budget || budget - outside < (256ULL << 20)) return guess;

  const auto started = std::chrono::steady_clock::now();
  guess.outcome = DFHFGuessOutcome::Failed;
  try {
    core::System auxiliary;
    auxiliary.atoms = system.atoms;
    auxiliary.charge = system.charge;
    auxiliary.multiplicity = system.multiplicity;
    auxiliary.basis_representation = GENERATIVEQC_BASIS_SPHERICAL;
    for (std::size_t atom_index = 0; atom_index < system.atoms.size(); ++atom_index) {
      for (const auto& entry : df_hf_guess_basis) {
        if (entry.atomic_number != system.atoms[atom_index].atomic_number) continue;
        core::Shell shell;
        shell.atom_index = static_cast<std::uint32_t>(atom_index);
        shell.angular_momentum = entry.angular_momentum;
        shell.primitives.push_back({entry.exponent, 1.0});
        auxiliary.shells.push_back(std::move(shell));
      }
    }
    std::string detail;
    if (molecule::validate_and_normalize(auxiliary, detail) != GENERATIVEQC_STATUS_SUCCESS)
      throw std::invalid_argument(detail);
    guess.auxiliary_functions = molecule::ao_count(auxiliary);
    scf::ScfOptions options;
    options.max_iterations = 32;
    options.diis_history = descriptor.diis_history ? descriptor.diis_history : 8;
    options.energy_tolerance = 1e-4;
    options.density_tolerance = 1e-4;
    options.screening_tolerance = 0;
    options.precision_mode = GENERATIVEQC_PRECISION_FP64;
    options.compute_forces = false;
    options.export_physical_reference = false;
    options.density_fitting_mode = GENERATIVEQC_DENSITY_FITTING_CUDA;
    options.reference_memory_budget_bytes = std::min<std::size_t>(budget - outside, 512ULL << 20);
    options.density_fitting_memory_budget_bytes = options.reference_memory_budget_bytes;
    auto preliminary = scf::run_rhf_density_fitting_cuda(system, auxiliary, options, device_id);
    guess.iterations = preliminary.iterations;
    if (preliminary.converged) {
      scf::validate_hf_warm_density(system, GENERATIVEQC_METHOD_RHF, preliminary.density);
      guess.density = std::move(preliminary.density);
      guess.outcome = DFHFGuessOutcome::Used;
    }
  } catch (const std::exception&) {
    guess.work_counters_complete = false;
  }
  guess.seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count();
  return guess;
}
}  // namespace generativeqc::methods::detail
