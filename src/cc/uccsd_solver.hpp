#pragma once

#include <cstddef>

#include "cc/solver.hpp"
#include "hf/reference.hpp"
#include "integrals/electron_interaction_source.hpp"
#include "posthf/native_provider.hpp"

namespace generativeqc::cc {

/** Native unrestricted endpoint record; no partial result is public on failure.
 * The five excitation blocks use independent alpha/beta orbital populations.
 * Numeric capacity counts simultaneous reference, provider, feed, arena, DIIS
 * and triples storage rather than only a generated residual workspace. */
struct UnrestrictedSolve {
  bool converged{};
  double correlation_energy{}, triples_energy{}, minimum_denominator{};
  double energy_change{}, singles_residual{}, doubles_residual{};
  double replay_singles_residual{}, replay_doubles_residual{};
  unsigned iterations{};
  std::size_t numeric_capacity_bytes{}, triples_tiles{}, triples_workspace_bytes{};
  std::size_t graph_calls{}, replay_calls{};
  std::size_t update_calls{};
  posthf::ProviderWork provider_work;
  double problem_seconds{}, solver_seconds{}, triples_seconds{};
  double iteration_seconds{}, replay_seconds{}, update_seconds{}, diis_seconds{};
};

/** Admission precedes reference execution and uses maximum configured work.
 * This deliberately bounded conventional owner supports at most twelve AOs.
 * Exceeding its complete numeric/work domain is an error, never a fallback. */
std::size_t unrestricted_capacity(std::size_t nbf, std::size_t alpha, std::size_t beta,
                                  const SolverOptions& options, bool triples,
                                  std::size_t reference_capacity);

/** Solve/replay CCSD and optionally replay all four standard (T) channels.
 * All scientific contractions are build-time-generated shared TensorIR; this
 * owner supplies unshifted physical UHF Fock/integrals and packed host DIIS. */
UnrestrictedSolve solve_unrestricted(const hf::UnrestrictedPhysicalReference& reference,
                                     const integrals::ElectronInteractionSource& source,
                                     const SolverOptions& options, bool triples,
                                     std::size_t reference_capacity);

const char* unrestricted_equation_hash() noexcept;
const char* unrestricted_triples_hash() noexcept;

}  // namespace generativeqc::cc
