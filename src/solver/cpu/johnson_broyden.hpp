#ifndef GENERATIVEQC_SOLVER_CPU_JOHNSON_BROYDEN_HPP
// xtbloom's CUDA/MKL additional permission is in CUDA_MKL_LINKING_EXCEPTION.

#define GENERATIVEQC_SOLVER_CPU_JOHNSON_BROYDEN_HPP

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "solver/johnson_broyden.hpp"

namespace generativeqc::solver::cpu {

inline constexpr std::size_t kBroydenWorkspaceAlignment = 64u;
inline constexpr std::size_t kBroydenMaximumFields = 4u;

/*
 * One field in a caller-owned ragged vector. The solver copies the
 * offsets into its immutable plan, so these borrowed construction views need
 * to remain valid only for make_broyden_plan.
 */
struct BroydenFieldLayoutView {
  std::size_t offset_bytes = 0u;
  std::size_t size_bytes = 0u;
  std::int64_t element_count = 0;
  const std::int64_t* system_offsets = nullptr;
  std::size_t system_offset_count = 0u;
};

/*
 * Ordered fields concatenated into one ragged FP64 mixer vector. Field order
 * and each system slice are retained exactly. Scientific field meanings and
 * terminal convergence belong to the consumer.
 */
struct BroydenVectorLayoutView {
  std::int64_t batch_size = 0;
  std::size_t workspace_size_bytes = 0u;
  // Power of two, at least alignof(double); every field offset is double-aligned.
  std::size_t workspace_alignment = 0u;
  std::array<BroydenFieldLayoutView, kBroydenMaximumFields> fields{};
  std::size_t field_count = 0u;
};

/* Exact mutable binding corresponding to BroydenVectorLayoutView. */
struct BroydenVectorView {
  void* workspace_base = nullptr;
  std::size_t workspace_size_bytes = 0u;
  std::array<double*, kBroydenMaximumFields> fields{};
  std::size_t field_count = 0u;
};

struct BroydenPlanData;

/*
 * Immutable topology, caller-selected numerical policy and status encoding.
 *
 * Construction may allocate metadata. All numerical transitions use only
 * caller-owned state and scratch, and successful steady-state calls allocate
 * nothing. The plan knows field packing but no scientific field semantics.
 * Retain a sealed plan (or a copy sharing its identity) and every borrowed
 * storage allocation for the entire binding lifetime. A raw plan_identity
 * records identity only; state and scratch views do not extend ownership.
 */
class BroydenPlan {
 public:
  BroydenPlan() noexcept = default;
  BroydenPlan(const BroydenPlan&) noexcept = default;
  BroydenPlan(BroydenPlan&&) noexcept = default;
  BroydenPlan& operator=(const BroydenPlan&) noexcept = default;
  BroydenPlan& operator=(BroydenPlan&&) noexcept = default;
  ~BroydenPlan() = default;

  [[nodiscard]] bool sealed() const noexcept;
  [[nodiscard]] std::int64_t batch_size() const noexcept;
  [[nodiscard]] std::int64_t history_size() const noexcept;
  [[nodiscard]] std::int64_t total_vector_elements() const noexcept;
  [[nodiscard]] std::int64_t maximum_vector_elements() const noexcept;
  [[nodiscard]] double damping() const noexcept;
  [[nodiscard]] double rms_tolerance() const noexcept;
  [[nodiscard]] double maximum_tolerance() const noexcept;
  [[nodiscard]] std::size_t state_size_bytes() const noexcept;
  [[nodiscard]] std::size_t workspace_size_bytes() const noexcept;
  [[nodiscard]] std::size_t resident_bytes() const noexcept;
  [[nodiscard]] const std::vector<std::int64_t>& vector_offsets() const noexcept;
  [[nodiscard]] bool matches_vector_layout(const BroydenVectorLayoutView& layout) const noexcept;
  [[nodiscard]] bool overlaps_storage(const void* data, std::size_t size_bytes) const noexcept;
  [[nodiscard]] const BroydenPlanData* identity() const noexcept;

 protected:
  explicit BroydenPlan(std::shared_ptr<const BroydenPlanData> data) noexcept;

 private:
  std::shared_ptr<const BroydenPlanData> data_;

  friend BroydenResult make_broyden_plan(const BroydenVectorLayoutView& layout,
                                         const BroydenPolicy& policy,
                                         BroydenStatusEncoding status_encoding, BroydenPlan& plan,
                                         std::string& error);
};

/* Persistent system-major history and residual diagnostics. The consumer may
 * overwrite converged after a successful transition with its terminal policy.
 * The mixer never uses that byte to suppress a requested transition. */
struct BroydenState {
  void* workspace_base = nullptr;
  std::size_t workspace_size_bytes = 0u;

  double* current_inputs = nullptr;
  double* previous_inputs = nullptr;
  double* previous_residuals = nullptr;
  double* df_history = nullptr;
  double* u_history = nullptr;
  double* omega = nullptr;

  double* residual_rms = nullptr;
  double* residual_maximum = nullptr;
  std::uint64_t* iterations = nullptr;
  std::uint64_t* restart_counts = nullptr;
  std::int32_t* system_statuses = nullptr;
  std::uint8_t* initialized = nullptr;
  std::uint8_t* converged = nullptr;

  BroydenStatusEncoding status_encoding{};
  const BroydenPlanData* plan_identity = nullptr;
};

/* Compact scratch reusable by one worker or the serial batch wrapper. */
struct BroydenWorkspace {
  void* workspace_base = nullptr;
  std::size_t workspace_size_bytes = 0u;

  double* residual = nullptr;
  double* mixed = nullptr;
  double* delta_f = nullptr;
  double* new_u = nullptr;
  double* beta = nullptr;
  double* coefficients = nullptr;
  std::int64_t* history_slots = nullptr;

  const BroydenPlanData* plan_identity = nullptr;
};

BroydenResult make_broyden_plan(const BroydenVectorLayoutView& layout, const BroydenPolicy& policy,
                                BroydenStatusEncoding status_encoding, BroydenPlan& plan,
                                std::string& error);

BroydenResult bind_broyden_state(const BroydenPlan& plan, void* workspace,
                                 std::size_t workspace_size, BroydenState& state,
                                 std::string& error);

BroydenResult bind_broyden_workspace(const BroydenPlan& plan, void* workspace,
                                     std::size_t workspace_size, BroydenWorkspace& view,
                                     std::string& error);

/* Read-only canonical-binding checks for higher-level allocation-free drivers. */
BroydenResult validate_broyden_state_binding(const BroydenPlan& plan, const BroydenState& state,
                                             std::string& error);
BroydenResult validate_broyden_workspace_binding(const BroydenPlan& plan,
                                                 const BroydenWorkspace& workspace,
                                                 std::string& error);

/* Initialization is all-or-nothing across the complete ragged batch. */
BroydenResult initialize_broyden_state(const BroydenPlan& plan, const BroydenVectorView& vector,
                                       const BroydenState& state, std::string& error);

/* Restart clears only the selected system after its new vector is validated. */
BroydenResult restart_broyden_system(const BroydenPlan& plan, std::int64_t system,
                                     const BroydenVectorView& vector, const BroydenState& state,
                                     std::string& error);

/*
 * Mix one raw vector in place. Numerical failure changes only the selected
 * system status; raw values and all persistent numerical history stay intact.
 */
BroydenResult mix_broyden_system(const BroydenPlan& plan, std::int64_t system,
                                 const BroydenVectorView& vector, const BroydenState& state,
                                 const BroydenWorkspace& workspace, std::string& error);

/* Serial wrapper retaining peer-local numerical failure isolation. */
BroydenResult mix_broyden_batch(const BroydenPlan& plan, const BroydenVectorView& vector,
                                const BroydenState& state, const BroydenWorkspace& workspace,
                                std::string& error);

/* Copy exactly one system into or out of a disjoint full-layout binding. */
BroydenResult prepare_broyden_system_transaction(const BroydenPlan& plan, std::int64_t system,
                                                 const BroydenState& source,
                                                 const BroydenState& staged, std::string& error);

BroydenResult commit_broyden_system_transaction(const BroydenPlan& plan, std::int64_t system,
                                                const BroydenState& staged,
                                                const BroydenState& destination,
                                                std::string& error);

}  // namespace generativeqc::solver::cpu

#endif  // GENERATIVEQC_SOLVER_CPU_JOHNSON_BROYDEN_HPP
