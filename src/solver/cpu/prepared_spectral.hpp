#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "solver/cpu/symmetric_eigen.hpp"

namespace generativeqc::solver::cpu {

inline constexpr std::size_t kSpectralCacheAlignment = 64;

enum class SpectralResult {
  success,
  invalid_argument,
  allocation_failure,
  backend_unavailable,
  backend_failure,
  numerical_failure
};

/** Borrowed status storage is int32, with caller-selected meanings. */
struct SpectralStatusEncoding {
  std::int32_t success = 0;
  std::int32_t numerical_failure = 1;
};

struct SpectralPlanData;

/** Immutable ragged matrix dimensions and numerical resources. No electronic,
 * spin, occupation, convergence or result-publication policy belongs here.
 * Preparation copies offsets, allocates metadata only, and never initializes or
 * calls a numerical provider. Copies share identity; retain a copy and all
 * borrowed allocations throughout the lifetime of cache/work bindings. */
class PreparedSpectralPlan {
 public:
  bool sealed() const noexcept;
  std::int64_t batch_size() const noexcept;
  std::int64_t total_matrix_elements() const noexcept;
  std::int64_t maximum_order() const noexcept;
  double minimum_overlap_rcond() const noexcept;
  std::size_t cache_size_bytes() const noexcept;
  std::size_t factor_offset_bytes() const noexcept;
  std::size_t generation_offset_bytes() const noexcept;
  std::size_t status_offset_bytes() const noexcept;
  std::size_t resident_bytes() const noexcept;
  const std::vector<std::int64_t>& orbital_offsets() const noexcept;
  const std::vector<std::int64_t>& matrix_offsets() const noexcept;
  const PreparedSymmetricEigen& symmetric_eigen() const noexcept;
  SpectralStatusEncoding status_encoding() const noexcept;
  const SpectralPlanData* identity() const noexcept;
  bool overlaps_storage(const void* data, std::size_t bytes) const noexcept;

 private:
  std::shared_ptr<const SpectralPlanData> data_;
  friend SpectralResult prepare_spectral_plan(const std::int64_t*, std::size_t, double,
                                              SpectralStatusEncoding, PreparedSpectralPlan&,
                                              std::string&);
};

struct SpectralOverlapCache {
  void* workspace_base = nullptr;
  std::size_t workspace_size_bytes = 0;
  double* factors = nullptr;
  std::uint64_t* generations = nullptr;
  std::int32_t* statuses = nullptr;
  const SpectralPlanData* plan_identity = nullptr;
};

/** Independent borrowed spans let a caller retain its existing arena packing.
 * Capacities are in elements. Execution uses the prepared maximum-system LAPACK
 * counts, even when solving a smaller system. No call allocates numerical work,
 * queries a provider, changes thread settings, or replaces these buffers. */
struct SpectralWorkspace {
  double* coefficients = nullptr;
  std::size_t coefficient_capacity = 0;
  double* eigenvalues = nullptr;
  std::size_t eigenvalue_capacity = 0;
  double* lapack_work = nullptr;
  std::size_t lapack_work_capacity = 0;
  LapackInt* lapack_integer_work = nullptr;
  std::size_t lapack_integer_work_capacity = 0;
};

struct SpectralFactorWorkspace {
  double* factors = nullptr;
  std::size_t factor_capacity = 0;
  std::uint64_t* generations = nullptr;
  std::size_t generation_capacity = 0;
  std::int32_t* statuses = nullptr;
  std::size_t status_capacity = 0;
  double* lapack_work = nullptr;
  std::size_t lapack_work_capacity = 0;
  LapackInt* lapack_integer_work = nullptr;
  std::size_t lapack_integer_work_capacity = 0;
};

/** Successful finite/symmetry preflight over a borrowed ragged matrix batch.
 * Retain a copy of the plan. Inputs and optional layout arrays must remain alive
 * and unchanged until its
 * last execution. Copying this token does not extend their lifetime. Private
 * state prevents bypassing preflight by fabricating an admitted descriptor. */
class AdmittedSpectralMatrices {
 public:
  const double* matrix(const PreparedSpectralPlan& plan, std::int64_t system,
                       std::int32_t spectrum) const noexcept;
  bool is_overlap_batch(const PreparedSpectralPlan& plan) const noexcept;
  bool overlaps_borrowed_storage(const void* pointer, std::size_t bytes) const noexcept;

 private:
  const SpectralPlanData* plan_identity_ = nullptr;
  const double* values_ = nullptr;
  const std::int64_t* offsets_ = nullptr;
  const std::int32_t* counts_ = nullptr;
  std::int64_t first_system_ = 0;
  std::size_t system_count_ = 0;
  std::size_t value_count_ = 0;
  friend SpectralResult admit_spectral_matrices(const PreparedSpectralPlan&, std::int64_t,
                                                std::size_t, const double*, const std::int64_t*,
                                                const std::int32_t*, AdmittedSpectralMatrices&);
};

/** Null offsets/counts select one matrix per system using the plan's partition.
 * Otherwise supply system_count+1 offsets and system_count positive counts.
 * Offsets are normalized by their first entry; each slice is exactly count*n*n.
 * Every matrix is checked before publishing a token, with no provider calls. */
SpectralResult admit_spectral_matrices(const PreparedSpectralPlan& plan, std::int64_t first_system,
                                       std::size_t system_count, const double* values,
                                       const std::int64_t* offsets, const std::int32_t* counts,
                                       AdmittedSpectralMatrices& admitted);

SpectralResult prepare_spectral_plan(const std::int64_t* orbital_offsets, std::size_t offset_count,
                                     double minimum_overlap_rcond,
                                     SpectralStatusEncoding status_encoding,
                                     PreparedSpectralPlan& plan, std::string& error);
SpectralResult bind_spectral_overlap_cache(const PreparedSpectralPlan& plan, void* storage,
                                           std::size_t bytes, SpectralOverlapCache& cache,
                                           std::string& error);
SpectralResult validate_spectral_overlap_cache(const PreparedSpectralPlan& plan,
                                               const SpectralOverlapCache& cache,
                                               std::string& error);

/** All structural and backend failures preserve the complete committed cache.
 * Numerical failures commit only generation/status for that member, retaining
 * its old factor bytes. Any nonzero caller generation is valid; this service
 * does not decide scientific identity or require monotonic stamps. The caller
 * owns the surrounding BLAS thread scope. */
SpectralResult factor_spectral_overlaps(const PreparedSpectralPlan& plan, const double* overlap,
                                        std::uint64_t generation,
                                        const tensor::cpu::CpuLinearAlgebraBackend& backend,
                                        const SpectralFactorWorkspace& workspace,
                                        const SpectralOverlapCache& cache, std::string& error);

/** Produce one unpublished column-major spectrum using a current cached factor.
 * A stale/failed factor returns numerical_failure without touching output work.
 * Provider/data failure may alter scratch but never cache or published results.
 * A caller can solve any number of independent spectra against one system. */
SpectralResult solve_prepared_spectrum(const PreparedSpectralPlan& plan, std::int64_t system,
                                       const SpectralOverlapCache& cache, std::uint64_t generation,
                                       const double* hamiltonian,
                                       const tensor::cpu::CpuLinearAlgebraBackend& backend,
                                       const SpectralWorkspace& workspace);

/** Execution phases for a caller that must preflight the complete batch before
 * entering its thread scope or performing any provider work. Resource bindings
 * are still validated; admitted matrix values are not rescanned. */
SpectralResult factor_admitted_spectral_overlaps(
    const PreparedSpectralPlan& plan, const AdmittedSpectralMatrices& admitted,
    std::uint64_t generation, const tensor::cpu::CpuLinearAlgebraBackend& backend,
    const SpectralFactorWorkspace& workspace, const SpectralOverlapCache& cache,
    std::string& error);
SpectralResult solve_admitted_spectrum(const PreparedSpectralPlan& plan, std::int64_t system,
                                       std::int32_t spectrum,
                                       const AdmittedSpectralMatrices& admitted,
                                       const SpectralOverlapCache& cache, std::uint64_t generation,
                                       const tensor::cpu::CpuLinearAlgebraBackend& backend,
                                       const SpectralWorkspace& workspace);

}  // namespace generativeqc::solver::cpu
