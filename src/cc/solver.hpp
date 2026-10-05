#pragma once

#include <cstddef>
#include <cstdint>
#include <span>
#include <string>
#include <vector>

namespace generativeqc::cc {

struct SolverOptions {
  unsigned max_iterations{100};
  unsigned diis_size{6};
  double energy_tolerance{1e-11};
  double residual_tolerance{1e-9};
  double denominator_threshold{1e-10};
  double damping{};
  double level_shift{};
  std::size_t max_bytes{256ULL << 20};
  // Internal DF scheduling control; dense/conventional paths are unaffected.
  // Admission retains the bounded original schedule when work or storage wins.
  bool df_auxiliary_reduction{true};
  // Optional compiler-packed FP64 matrix contractions, with scalar fallback.
  bool df_matrix_gemm{true};
  // Shape/budget admission may choose a smaller Q tile, including one slice.
  std::size_t df_auxiliary_batch_limit{8};
  // Native canonical CUDA construction may retain the small spectrum instead
  // of a full doubles denominator. Supplied Problems stay explicit by default.
  bool derived_denominators{true};
};

enum class DenominatorRepresentation { Explicit, CanonicalSpectrum };

struct Problem {
  std::size_t nocc{}, nvir{};
  std::vector<double> foo, fov, fvv;
  std::vector<double> ovov, ovvo, oovv, ovvv, ovoo, oooo, vvvv;
  std::vector<double> d1, d2;
  // CanonicalSpectrum requires empty d2 and an occupied-then-virtual spectrum.
  // d1 remains explicit (O(ov)); shift and physical safety threshold are part
  // of this representation's provenance, never inferred from a Fock diagonal.
  DenominatorRepresentation denominator_representation{DenominatorRepresentation::Explicit};
  std::vector<double> canonical_eps;
  double canonical_level_shift{}, canonical_denominator_threshold{};
  std::vector<double> initial_t1, initial_t2;
  double reference_energy{};
  double minimum_absolute_denominator{};
  std::size_t reference_retained_bytes{};
  std::size_t provider_peak_bytes{};
  std::size_t provider_host_bytes{};
  // Correlation-only DF virtual representation. For naux > 0 these row-major
  // Q-major factors replace ovvv/vvvv, which must be empty. The retained small
  // blocks must come from the same fitted Hamiltonian; Fock/reference energy
  // retain the explicitly selected reference contract (conventional RHF here).
  std::size_t naux{};
  std::vector<double> df_bov, df_bvv;
  // Optional for supplied energy/Lambda inputs; required by the physical
  // retained-block pullback. Native molecular sources always publish Boo.
  std::vector<double> df_boo;
  // Binds native physical factor derivatives to their immutable source/frame.
  // Supplied algebraic problems may leave this zero.
  std::uint64_t df_source_identity{};
};

enum class SolveStatus { Converged, NotConverged, NumericalFailure };

struct SolverDiagnostic {
  unsigned iterations{};
  unsigned diis_restarts{};
  double energy_change{};
  double r1_max{}, r2_max{};
  double replay_r1_max{}, replay_r2_max{};
  std::size_t numeric_capacity_bytes{};
  std::size_t owned_device_bytes{};
  std::size_t setup_h2d_bytes{};
  std::size_t scalar_d2h_bytes{};
  std::size_t amplitude_d2h_bytes{};
  std::uint64_t denominator_identity{};
  // Only iteration/Jacobi reconstructions; host initialization and Lambda's
  // diagonal construction are separate consumers, not included in this count.
  std::size_t derived_d2_iteration_evaluations{};
  std::size_t synchronizations{};
  std::size_t iteration_graph_calls{};
  std::size_t replay_graph_calls{};
  std::size_t update_calls{};
  std::size_t generated_error_checks{};
  std::size_t diis_gram_calls{};
  std::size_t diis_coefficient_calls{};
  std::size_t diis_combine_calls{};
  // History insertion counts destination bytes; chronological retirement moves
  // no tensor bytes. Dot/combine terms are scalar summands, not hardware FLOPs.
  std::size_t diis_history_insert_bytes{}, diis_history_shift_bytes{};
  std::size_t diis_residual_dot_terms{}, diis_gram_updates{}, diis_combine_terms{};
  // Complete auxiliary work, including trial evaluations and independent replay.
  std::size_t df_auxiliary_slices{};
  std::size_t df_virtual_operations{};
  std::size_t df_accumulation_calls{};
  std::size_t df_hoisted_evaluations{};
  std::size_t df_preparation_calls{};
  std::size_t df_contraction_terms{};
  bool df_matrix_gemm{};
  std::size_t df_gemm_calls{}, df_gemm_summands{}, df_packing_bytes{};
  std::size_t df_provider_capacity_bytes{};
  std::size_t df_auxiliary_batch_size{1}, df_auxiliary_tiles{}, df_accumulation_bytes{};
  double tensor_seconds{};
  double iteration_seconds{};
  double replay_seconds{};
  double update_seconds{};
  double diis_seconds{};
};

struct SolverResult {
  SolveStatus status{SolveStatus::NotConverged};
  std::string reason;
  double correlation_energy{};
  double total_energy{};
  std::vector<double> t1, t2;
  SolverDiagnostic diagnostic;
  [[nodiscard]] bool converged() const noexcept { return status == SolveStatus::Converged; }
};

// Shared admission defaults to the conventional representation. Only an owner
// that implements the full DF Q sum may opt in; Lambda/triples/force consumers
// must reject DF until their own factorized paths are implemented.
void validate_problem(const Problem& problem, bool allow_df_virtual = false);
void validate_options(const SolverOptions& options);
// Initialize canonical denominators using the compiler-owned ordered scalar
// expression. Derived admission validates O(ov) extrema, without an O(o²v²) pass.
void initialize_canonical_denominators(Problem& problem, std::span<const double> energies,
                                       const SolverOptions& options, bool derived);
// Callers must first validate the Problem and supply an in-range flattened index.
double doubles_denominator_at(const Problem& problem, std::size_t flat);
// Fingerprint the representation and its complete numeric provenance. No cache
// is introduced; future consumers must not equate explicit and derived inputs.
std::uint64_t denominator_identity(const Problem& problem);
std::size_t problem_host_bytes(const Problem& problem);
SolverResult solve_cpu(const Problem& problem, const SolverOptions& options);
SolverResult solve_cuda(const Problem& problem, const SolverOptions& options, int device);

}  // namespace generativeqc::cc
