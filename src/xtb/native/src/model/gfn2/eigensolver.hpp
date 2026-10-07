#ifndef GENERATIVEQC_XTB_MODEL_GFN2_EIGENSOLVER_HPP
// xtbloom's CUDA/MKL additional permission is in CUDA_MKL_LINKING_EXCEPTION.

#define GENERATIVEQC_XTB_MODEL_GFN2_EIGENSOLVER_HPP

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "model/gfn2/wavefunction.hpp"
#include "runtime/types.hpp"
#include "tensor/cpu/lp64_provider.hpp"

namespace generativeqc::xtb::detail::gfn2 {

inline constexpr std::size_t kEigensolverWorkspaceAlignment = 64u;
using LapackInt = ::generativeqc::tensor::cpu::LapackInt;
using LapackDpotrfWork = ::generativeqc::tensor::cpu::LapackDpotrfWork;
using LapackDpoconWork = ::generativeqc::tensor::cpu::LapackDpoconWork;
using LapackDsyevdWork = ::generativeqc::tensor::cpu::LapackDsyevdWork;
using CblasDtrsm = ::generativeqc::tensor::cpu::CblasDtrsm;
using CblasDgemm = ::generativeqc::tensor::cpu::CblasDgemm;
using BlasSetNumThreadsLocal = ::generativeqc::tensor::cpu::BlasSetNumThreadsLocal;
using BlasThreadCleanup = ::generativeqc::tensor::cpu::BlasThreadCleanup;
using CpuLinearAlgebraBackend = ::generativeqc::tensor::cpu::CpuLinearAlgebraBackend;

generativeqc_xtb_status_t make_mkl_rt_lp64_backend(CpuLinearAlgebraBackend& backend, std::string& error);

/* Compatibility adapter: test admission and preflight remain shared. */
generativeqc_xtb_status_t make_internal_test_lp64_backend(
    LapackDpotrfWork dpotrf_work, LapackDpoconWork dpocon_work, LapackDsyevdWork dsyevd_work,
    CblasDtrsm dtrsm, CblasDgemm dgemm, BlasSetNumThreadsLocal set_num_threads_local,
    CpuLinearAlgebraBackend& backend, std::string& error,
    BlasThreadCleanup thread_cleanup = nullptr);

struct EigensolverPlanData;

/*
 * Model-neutral projection of the five electronic arrays consumed by the
 * generalized eigensolver. Model-specific SCC state may place other fields
 * between these ranges; the solver records and validates only this projection.
 */
struct EigensolverWavefunctionFieldLayout {
  std::size_t offset_bytes = 0u;
  std::int64_t element_count = 0;
  const std::int64_t* system_offsets = nullptr;
  std::size_t system_offset_count = 0u;
};

struct EigensolverWavefunctionLayout {
  std::int64_t batch_size = 0;
  std::size_t workspace_size_bytes = 0u;
  const std::int64_t* orbital_offsets = nullptr;
  std::size_t orbital_offset_count = 0u;
  const std::int32_t* spin_channels = nullptr;
  std::size_t spin_channel_count = 0u;
  const double* alpha_electron_counts = nullptr;
  const double* beta_electron_counts = nullptr;
  std::size_t electron_count_count = 0u;
  std::array<EigensolverWavefunctionFieldLayout, 5> fields{};
};

struct EigensolverWavefunctionView {
  void* workspace_base = nullptr;
  std::size_t workspace_size_bytes = 0u;
  double* coefficients = nullptr;
  double* eigenvalues = nullptr;
  double* occupations = nullptr;
  double* density = nullptr;
  double* energy_weighted_density = nullptr;
};

/*
 * Compact immutable handle for all topology, electronic, layout, and scratch
 * metadata required by the CPU eigensolver. Copies are O(1), remain cache-
 * compatible, and make hot-path plan validation O(1).
 */
class EigensolverPlan {
 public:
  EigensolverPlan() noexcept = default;
  EigensolverPlan(const EigensolverPlan&) noexcept = default;
  EigensolverPlan(EigensolverPlan&&) noexcept = default;
  EigensolverPlan& operator=(const EigensolverPlan&) noexcept = default;
  EigensolverPlan& operator=(EigensolverPlan&&) noexcept = default;
  ~EigensolverPlan() = default;

  [[nodiscard]] bool sealed() const noexcept;
  [[nodiscard]] std::int64_t batch_size() const noexcept;
  [[nodiscard]] std::int64_t total_matrix_elements() const noexcept;
  [[nodiscard]] std::int64_t maximum_orbitals() const noexcept;
  [[nodiscard]] double minimum_overlap_rcond() const noexcept;
  [[nodiscard]] std::size_t overlap_cache_size_bytes() const noexcept;
  [[nodiscard]] std::size_t worker_workspace_size_bytes() const noexcept;
  [[nodiscard]] std::size_t workspace_size_bytes() const noexcept;
  [[nodiscard]] std::size_t resident_bytes() const noexcept;
  [[nodiscard]] const std::vector<std::int64_t>& matrix_offsets() const noexcept;
  [[nodiscard]] const std::vector<std::int64_t>& orbital_offsets() const noexcept;
  [[nodiscard]] const std::vector<std::int32_t>& spin_channels() const noexcept;
  [[nodiscard]] const std::vector<double>& alpha_electron_counts() const noexcept;
  [[nodiscard]] const std::vector<double>& beta_electron_counts() const noexcept;
  /* True when a byte range aliases this plan's immutable object or backing storage. */
  [[nodiscard]] bool overlaps_storage(const void* data, std::size_t size_bytes) const noexcept;
  [[nodiscard]] const EigensolverPlanData* identity() const noexcept;

 private:
  explicit EigensolverPlan(std::shared_ptr<const EigensolverPlanData> data) noexcept;
  std::shared_ptr<const EigensolverPlanData> data_;

  friend generativeqc_xtb_status_t make_eigensolver_plan(const WavefunctionLayout& layout,
                                                EigensolverPlan& plan, std::string& error,
                                                double minimum_overlap_rcond);
  friend generativeqc_xtb_status_t make_eigensolver_plan(const EigensolverWavefunctionLayout& layout,
                                                EigensolverPlan& plan, std::string& error,
                                                double minimum_overlap_rcond);
};

struct EigensolverOverlapCache {
  void* workspace_base = nullptr;
  std::size_t workspace_size_bytes = 0u;
  double* cholesky_factors = nullptr;
  std::uint64_t* geometry_generations = nullptr;
  generativeqc_xtb_status_t* system_statuses = nullptr;
  const EigensolverPlanData* plan_identity = nullptr;
};

/*
 * One caller-owned numerical workspace descriptor with two binding modes.
 *
 * bind_eigensolver_worker_workspace binds only the leading maximum-system
 * scratch and leaves all staging pointers null. Its size depends only on the
 * largest system, so B parallel workers require O(B*max_system_size) memory.
 * bind_eigensolver_workspace additionally binds factor_staging and batch_*
 * arrays, whose unpublished full-batch results provide call-level atomicity.
 */
struct EigensolverWorkspace {
  void* workspace_base = nullptr;
  std::size_t workspace_size_bytes = 0u;
  double* coefficients = nullptr;
  double* densities = nullptr;
  double* energy_weighted_densities = nullptr;
  double* eigenvalues = nullptr;
  double* occupations = nullptr;
  double* lapack_work = nullptr;
  LapackInt* lapack_integer_work = nullptr;

  double* factor_staging = nullptr;
  std::uint64_t* factor_generation_staging = nullptr;
  generativeqc_xtb_status_t* factor_status_staging = nullptr;

  double* batch_coefficients = nullptr;
  double* batch_densities = nullptr;
  double* batch_energy_weighted_densities = nullptr;
  double* batch_eigenvalues = nullptr;
  double* batch_occupations = nullptr;
  generativeqc_xtb_status_t* batch_system_statuses = nullptr;
  double* batch_chemical_potentials = nullptr;
  double* batch_entropies = nullptr;
  double* batch_band_energies = nullptr;
  double* batch_free_energies = nullptr;

  const EigensolverPlanData* plan_identity = nullptr;
};

/*
 * Entropy is dimensionless (units of k_B). Temperature arguments are k_B*T in
 * Hartree, so free_energy = band_energy - temperature*entropy is in Hartree.
 */
struct EigensolverThermodynamicsView {
  generativeqc_xtb_status_t* system_statuses = nullptr;
  std::size_t system_status_capacity = 0u;
  double* chemical_potentials = nullptr;
  std::size_t chemical_potential_capacity = 0u;
  double* entropies = nullptr;
  std::size_t entropy_capacity = 0u;
  double* band_energies = nullptr;
  std::size_t band_energy_capacity = 0u;
  double* free_energies = nullptr;
  std::size_t free_energy_capacity = 0u;
};

generativeqc_xtb_status_t make_eigensolver_plan(const WavefunctionLayout& layout, EigensolverPlan& plan,
                                       std::string& error, double minimum_overlap_rcond = 1.0e-12);
generativeqc_xtb_status_t make_eigensolver_plan(const EigensolverWavefunctionLayout& layout,
                                       EigensolverPlan& plan, std::string& error,
                                       double minimum_overlap_rcond = 1.0e-12);

generativeqc_xtb_status_t bind_eigensolver_overlap_cache(const EigensolverPlan& plan, void* workspace,
                                                std::size_t workspace_size,
                                                EigensolverOverlapCache& cache, std::string& error);
generativeqc_xtb_status_t bind_eigensolver_workspace(const EigensolverPlan& plan, void* workspace,
                                            std::size_t workspace_size, EigensolverWorkspace& view,
                                            std::string& error);
generativeqc_xtb_status_t bind_eigensolver_worker_workspace(const EigensolverPlan& plan, void* workspace,
                                                   std::size_t workspace_size,
                                                   EigensolverWorkspace& view, std::string& error);

/* Read-only canonical-binding checks used by model-owned SCC orchestrators. */
generativeqc_xtb_status_t validate_eigensolver_overlap_cache_binding(const EigensolverPlan& plan,
                                                            const EigensolverOverlapCache& cache,
                                                            std::string& error);
generativeqc_xtb_status_t validate_eigensolver_worker_workspace_binding(
    const EigensolverPlan& plan, const EigensolverWorkspace& workspace, std::string& error);

/*
 * Factor packed symmetric overlaps into persistent column-major Cholesky
 * factors. Structural/backend errors are whole-call failures and publish
 * nothing; positive-definiteness and conditioning failures are recorded per
 * system when the complete staged batch is committed.
 */
generativeqc_xtb_status_t factor_overlap_cpu(const EigensolverPlan& plan, const double* overlap,
                                    std::uint64_t geometry_generation,
                                    const CpuLinearAlgebraBackend& backend,
                                    const EigensolverWorkspace& workspace,
                                    const EigensolverOverlapCache& cache, std::string& error);

/*
 * Solve all systems serially into unpublished full-batch staging after one
 * whole-call validation pass. Results are committed only after all systems
 * have ruled out call-level backend failures. The MKL backend temporarily
 * requests one BLAS thread, avoiding nested oversubscription.
 */
generativeqc_xtb_status_t solve_eigensystems_cpu(
    const EigensolverPlan& plan, const EigensolverOverlapCache& overlap_cache,
    std::uint64_t geometry_generation, const double* hamiltonians, double temperature,
    const CpuLinearAlgebraBackend& backend, const EigensolverWorkspace& workspace,
    const WavefunctionView& wavefunction, const EigensolverThermodynamicsView& thermodynamics,
    std::string& error);
generativeqc_xtb_status_t solve_eigensystems_cpu(
    const EigensolverPlan& plan, const EigensolverOverlapCache& overlap_cache,
    std::uint64_t geometry_generation, const double* hamiltonians, double temperature,
    const CpuLinearAlgebraBackend& backend, const EigensolverWorkspace& workspace,
    const EigensolverWavefunctionView& wavefunction,
    const EigensolverThermodynamicsView& thermodynamics, std::string& error);

/*
 * Allocation-free one-system worker primitive. Runtime schedulers may invoke
 * this concurrently for different systems when each worker owns a distinct
 * EigensolverWorkspace and std::string error object. Concurrent calls must
 * also target different systems so their wavefunction and thermodynamic output
 * slices are disjoint. Validation is O(1) plus the fixed number of output
 * fields and never scans other batch members.
 */
generativeqc_xtb_status_t solve_eigensystem_cpu(
    const EigensolverPlan& plan, std::int64_t system, const EigensolverOverlapCache& overlap_cache,
    std::uint64_t geometry_generation, const double* system_hamiltonians, double temperature,
    const CpuLinearAlgebraBackend& backend, const EigensolverWorkspace& workspace,
    const WavefunctionView& wavefunction, const EigensolverThermodynamicsView& thermodynamics,
    std::string& error);
generativeqc_xtb_status_t solve_eigensystem_cpu(
    const EigensolverPlan& plan, std::int64_t system, const EigensolverOverlapCache& overlap_cache,
    std::uint64_t geometry_generation, const double* system_hamiltonians, double temperature,
    const CpuLinearAlgebraBackend& backend, const EigensolverWorkspace& workspace,
    const EigensolverWavefunctionView& wavefunction,
    const EigensolverThermodynamicsView& thermodynamics, std::string& error);

/* Standalone tblite-compatible per-spin Aufbau/Fermi filling helper. */
generativeqc_xtb_status_t fill_occupations_cpu(std::int64_t orbital_count, const double* eigenvalues,
                                      double electron_count, double temperature,
                                      double* occupations, double& chemical_potential,
                                      double& entropy, std::string& error);

}  // namespace generativeqc::xtb::detail::gfn2

#endif  // GENERATIVEQC_XTB_MODEL_GFN2_EIGENSOLVER_HPP
