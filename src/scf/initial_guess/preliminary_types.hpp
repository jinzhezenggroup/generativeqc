#ifndef GENERATIVEQC_SCF_INITIAL_GUESS_PRELIMINARY_TYPES_HPP
#define GENERATIVEQC_SCF_INITIAL_GUESS_PRELIMINARY_TYPES_HPP

#include <cstddef>
#include <cstdint>

namespace generativeqc::scf::initial_guess {

enum class PreliminaryKind : std::uint32_t { HartreeFock = 1, Lda = 2 };
enum class PreliminaryOutcome : std::uint32_t {
  Disabled = 0,
  ExplicitDensity = 1,
  Used = 2,
  PreparationFailed = 3,
  BudgetSkipped = 4,
  TargetRetried = 5,
};

/** Execution-only cold-start policy. Never changes the target method or controls. */
struct PreliminaryOptions {
  PreliminaryKind kind{PreliminaryKind::HartreeFock};
  unsigned max_iterations{32};
  unsigned diis_history{8};
  double energy_tolerance{1.0e-6};
  double density_tolerance{1.0e-4};
  std::size_t maximum_numeric_bytes{256U << 20};
  unsigned radial_points{8}, angular_polar{6}, angular_azimuth{12};
};

/** Per-call work, independent of warm-state publication and target convergence. */
struct PreliminaryDiagnostic {
  std::uint32_t requested_kind{};
  PreliminaryOutcome outcome{PreliminaryOutcome::Disabled};
  unsigned preliminary_iterations{};
  std::uint64_t preliminary_fock_builds{};
  unsigned target_attempts{};
  unsigned discarded_target_iterations{};
  std::uint64_t discarded_target_fock_builds{};
  std::size_t preparation_numeric_capacity{};
  double preparation_seconds{};
  /** False if an exception prevented a preparation/discarded-attempt work census. */
  bool work_counters_complete{true};
};

}  // namespace generativeqc::scf::initial_guess
#endif
