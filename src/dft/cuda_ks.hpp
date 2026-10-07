#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "dft/ao_grid.hpp"
#include "dft/cuda_ks_final_state.hpp"
#include "dft/grid.hpp"
#include "dft/nonlocal_correlation/vv10_integration.hpp"
#include "dft/nonlocal_correlation/vv10_runtime.hpp"
#include "dft/semilocal_family.hpp"
#include "generativeqc/generativeqc.h"
#include "scf/fock_prepared.hpp"
#include "scf/types.hpp"

namespace generativeqc::tensor {
struct PanelProductDiagnostic;
}

namespace generativeqc::dft {

/** Optional lowering resources already reserved by the enclosing resource
 * plan, in addition to the ordinary XC arena. Provider-internal device bytes
 * are withheld from the numeric allocation ledger; explicit cache allocations
 * remain charged to it. Zero budgets retain the ordinary generated binding.
 * This declaration stays usable without CUDA headers for host inventory. */
struct CudaXcPreparationBudget {
  std::size_t device_bytes{}, host_bytes{};
};

/** Explicit component ownership for composition into #203. Provider/context
 * overhead and host quadrature preparation remain distinct from the native
 * iteration arena. No independent user-level memory allowance is introduced. */
struct CudaKsResources {
  std::size_t state_device_bytes{}, xc_device_bytes{}, grid_device_bytes{}, provider_device_bytes{};
  std::size_t retained_host_numeric_bytes{};
};

/** Counts owned transport, not an estimate from the iteration count. One-
 * electron provider setup includes its existing explicit host export; its
 * preparation diagnostics are reported separately by PreparedFockPlan.
 * Explicit seed spectral inputs count as setup H2D; their orbital frames
 * count as matrix D2H and eigenvalues/status as scalar D2H. */
struct CudaKsTransfers {
  std::uint64_t setup_h2d_bytes{}, density_h2d_bytes{}, scalar_d2h_bytes{}, matrix_d2h_bytes{};
  std::uint64_t final_state_d2h_bytes{}, final_state_reads{};
  std::uint64_t synchronizations{}, iterations{};
  /** Intermediate orthonormal KS orbital frames retained entirely on device for
   * a future #991 warm-subspace admission attempt. These are implementation
   * diagnostics and are not part of the public C transport ABI. */
  std::uint64_t warm_orbital_frames_retained{}, warm_orbital_frame_invalidations{};
  /** Number of subsequent proposals using the CPU-compatible stationary-cycle
   * shift; cumulative across replays, independent of transfer counts. */
  std::uint64_t occupation_stabilized_proposals{};
  /** Internal execution evidence. A selected two-slot RKS chunk can submit one
   * bounded unused slot when its first physical iteration terminates. */
  std::uint64_t submitted_iterations{}, iteration_chunks{}, iteration_synchronizations{};
  /** Shared compiled-execution lifecycle evidence for the device-control region.
   * Ordinary host-controlled KS leaves these counters zero. */
  std::uint64_t execution_region_bindings{}, execution_region_invalidations{};
  std::uint64_t execution_region_executions{}, execution_region_failures{};
  std::uint64_t execution_region_recoveries{};
  /** Shared CUDA-Graph lifecycle evidence for the bounded solver region. */
  std::uint64_t execution_region_captures{}, execution_region_replays{};
  std::uint64_t execution_region_fallbacks{};
  /** Explicit host-unfused XC staging, separate from ordinary setup/seed movement. */
  std::uint64_t xc_host_d2h_bytes{}, xc_host_h2d_bytes{}, xc_host_synchronizations{};
  /** Full-range density-fitted exchange provenance. Dense includes the first
   * cold/warm-seed build where no canonical factor is available; occupied
   * counts only builds whose Cocc generated the exact current device density. */
  std::uint64_t fitted_dense_exchange_builds{}, fitted_occupied_exchange_builds{};
  /** Successful final RKS states that retained the exact density-generating Cocc
   * together with the already-computed complete resident U=B*Cocc projection. */
  std::uint64_t fitted_final_projection_leases{};
};

/** State arena plus bounded ordinary-eigensolver workspace admission. The
 * provider's actual host/device queries are checked before allocation. This
 * shape query performs no CUDA call and allocates no numeric buffers. */
std::size_t cuda_ks_state_bytes(std::size_t nao, unsigned spins, unsigned diis_history,
                                bool exact_exchange = false, bool range_correction = false,
                                bool incremental_direct_jk = false);

/** Borrowed exact occupied factor and fitted projection for a successful
 * restricted density-fitted hybrid final state. Both allocations stay owned by
 * the KS/provider pair and are ordered by the returned stream. A subsequent
 * provider scratch writer invalidates the binding even if the KS final-state
 * token and projection shape are unchanged; reacquire before each use. */
struct CudaKsResidentFittedProjectionBinding {
  int device_id{-1};
  const double* occupied_coefficients{};
  const double* projection{};
  std::size_t nbf{}, naux{}, rank{};
  void* stream{};
  std::uint64_t owner{}, solve_epoch{}, generation{};

  explicit operator bool() const noexcept {
    return device_id >= 0 && occupied_coefficients != nullptr && projection != nullptr &&
           nbf != 0 && naux != 0 && rank != 0 && stream != nullptr && owner != 0 &&
           solve_epoch != 0 && generation != 0;
  }
};

/** Borrowed device density for a successful immutable final-state token.
 * The allocation remains owned by CudaKsPlan and is valid only while that
 * exact token remains current. No transfer or synchronization is performed. */
struct CudaKsResidentDensityBinding {
  int device_id{-1};
  const double* alpha{};
  const double* beta{};
  std::size_t matrix_elements{};
  unsigned spins{};
  /** Stream owning the accepted final density generation. Cross-stream
   * borrowers must order device reads against this stream before source reuse. */
  void* stream{};
  std::uint64_t owner{}, solve_epoch{}, generation{};

  explicit operator bool() const noexcept {
    return device_id >= 0 && alpha != nullptr && matrix_elements != 0 &&
           (spins == 1 || (spins == 2 && beta != nullptr)) && stream != nullptr && owner != 0 &&
           solve_epoch != 0 && generation != 0;
  }
};

/** Borrowed total stationary D/W for one-electron force consumers.
 * W is formed from the exact accepted final C/epsilon/occupation frame on the
 * KS stream and published only after the final-state validation drain. For UKS,
 * density is the alpha+beta total. The storage is phase-local scratch owned by
 * CudaKsPlan and is valid only while the exact token remains current. */
struct CudaKsResidentStationaryWeightsBinding {
  int device_id{-1};
  const double* density{};
  const double* weighted_density{};
  std::size_t matrix_elements{};
  unsigned spins{};
  std::uint64_t owner{}, solve_epoch{}, generation{};

  explicit operator bool() const noexcept {
    return device_id >= 0 && density != nullptr && weighted_density != nullptr &&
           matrix_elements != 0 && (spins == 1 || spins == 2) && owner != 0 && solve_epoch != 0 &&
           generation != 0;
  }
};

/** Borrowed full-grid total rho/grad-rho retained by the device-fused
 * nonlocal KS owner for its successful final generation. These arrays are
 * read-only inputs for downstream resident nonlocal force composition; the
 * owner keeps allocation/lifetime responsibility and invalidates the lease
 * with the same exact final-state token as resident D. */
struct CudaKsResidentNonlocalFeaturesBinding {
  int device_id{-1};
  const double* density{};
  const double* gradient{};
  std::size_t point_count{};
  /** Producer stream that owns the final feature generation. Downstream
   * cross-stream copies must order against this stream before source reuse. */
  void* stream{};
  std::uint64_t owner{}, solve_epoch{}, generation{};

  explicit operator bool() const noexcept {
    return device_id >= 0 && density != nullptr && gradient != nullptr && point_count != 0 &&
           stream != nullptr && owner != 0 && solve_epoch != 0 && generation != 0;
  }
};

/** Intrusive fixed-density component profile against one exact final-state
 * token. Times are CUDA-event milliseconds for provider/XC device work only;
 * event creation, validation and status reads are deliberately outside them.
 * Bit 0=J, 1=primary full-range K, 2=range-correction K, 3=semilocal XC. */
struct CudaKsFixedDensityProfile {
  std::array<double, 4> milliseconds{};
  std::uint32_t present_mask{};
};

/** Native ordinary-stream LDA/PBE RKS/UKS trajectory. The borrowed common
 * Fock plan must outlive it. Model/grid/functional identity is immutable;
 * changing it requires a new owner. Symmetric overlap and core initial density
 * are constructed on the device; explicit host warm inputs are normalized at
 * admission. No CPU XC or matrix export occurs in an iteration. Final output
 * is a separate, measured operation.
 *
 * Split enqueue/finish operations let a native ragged batch enqueue all
 * active item streams before reading their small scalar records. Each owner
 * isolates pending/active/failed/converged and last-good warm states. */
class CudaKsPlan {
 public:
  /** Curated and generated codes share the same owner; keep range-exchange and
   * nonlocal bindings when adapting an existing curated functional to its code. */
  CudaKsPlan(const scf::PreparedFockPlan& fock, const AoBasis& basis, const MolecularGrid& grid,
             const scf::ScfOptions& options, std::uint32_t functional_code,
             std::size_t tile_points = 256,
             const scf::ResolvedFockBuild* range_correction = nullptr,
             nlc::Vv10Plan* nonlocal_correlation = nullptr,
             nlc::Vv10DensityDomain nonlocal_domain = nlc::Vv10DensityDomain::StrictPositive,
             CudaXcPreparationBudget xc_budget = {});
  CudaKsPlan(const scf::PreparedFockPlan& fock, const AoBasis& basis, const MolecularGrid& grid,
             const scf::ScfOptions& options, SemilocalFamily functional,
             std::size_t tile_points = 256,
             const scf::ResolvedFockBuild* range_correction = nullptr,
             nlc::Vv10Plan* nonlocal_correlation = nullptr,
             nlc::Vv10DensityDomain nonlocal_domain = nlc::Vv10DensityDomain::StrictPositive,
             CudaXcPreparationBudget xc_budget = {});
  ~CudaKsPlan();
  CudaKsPlan(const CudaKsPlan&) = delete;
  CudaKsPlan& operator=(const CudaKsPlan&) = delete;

  /** Start fresh DIIS/history. A null seed reuses a compatible last-good
   * device density when requested; explicit seeds are normalized per spin. */
  void begin(const std::vector<double>* initial_density = nullptr, bool reuse_warm = true);
  bool active() const noexcept;
  bool pending() const noexcept;
  bool failed() const noexcept;
  void enqueue_iteration();
  /** Resolve a submitted iteration or bounded chunk; returns true while another is needed. */
  bool finish_iteration();
  /** Terminal result; density export is optional and never used in an iteration. */
  scf::ScfResult result(bool export_density = true);
  /** Energy-only adapters leave the final density resident by disabling export. */
  scf::ScfResult run(const std::vector<double>* initial_density = nullptr, bool reuse_warm = true,
                     bool export_density = true);
  /** Export only the last converged state for a changed-geometry rebuild. */
  std::vector<double> warm_density();
  /** Borrow the ordinary GPU eigen provider for synchronous input admission.
   * Only symmetric (null S/X) solves are supported while this owner is idle.
   * Iteration-history scratch is reused without changing final/warm state.
   * The callback must not outlive this plan; shared seed validation retains
   * its existing reference fallback for legacy near-symmetric inputs. */
  scf::initial_guess::EigenOperation seed_eigen_operation();
  /** Freeze replacement without disabling reuse of the last-good density. */
  void set_warm_start_updates(bool enabled) noexcept;
  /** Forget the seed without downloading or changing the current result. */
  void clear_warm_start() noexcept;
  /** Revoke final-state eligibility without changing warm-start ownership. */
  void invalidate_final_state() noexcept;
  /** Read-only host eligibility query. It performs no CUDA call or transfer. */
  generativeqc_status final_state_token(CudaKsFinalStateToken& token, std::string& detail) const;
  /** Borrow the current converged density in device memory under the same
   * exact-token contract. This is a zero-transfer execution lease for native
   * downstream consumers; callers must not retain pointers across invalidation. */
  generativeqc_status resident_final_density(const CudaKsFinalStateToken& expected,
                                             CudaKsResidentDensityBinding& binding,
                                             std::string& detail) const;
  /** Borrow the exact Cocc and complete U=B*Cocc produced by the converged
   * restricted density-fitted hybrid's final physical K build. No K rebuild,
   * transfer or synchronization is performed; the returned stream owns both
   * the factor reconstruction and the provider projection lifetime. */
  generativeqc_status resident_final_fitted_projection(
      const CudaKsFinalStateToken& expected, CudaKsResidentFittedProjectionBinding& binding,
      std::string& detail) const;
  /** Borrow total stationary D/W already staged by a successful weighted
   * final-state read. This performs no CUDA launch, transfer, or synchronization. */
  generativeqc_status resident_final_stationary_weights(
      const CudaKsFinalStateToken& expected, CudaKsResidentStationaryWeightsBinding& binding,
      std::string& detail) const;
  /** Borrow final total rho/grad-rho already produced by device-fused
   * nonlocal XC. The binding is available only for the exact current token
   * and performs no transfer, synchronization or numerical launch. */
  generativeqc_status resident_final_nonlocal_features(
      const CudaKsFinalStateToken& expected, CudaKsResidentNonlocalFeaturesBinding& binding,
      std::string& detail) const;
  /** Intrusively replay J/K/XC at the exact resident final density without
   * executing SCF or publishing a new density/XC generation. This is a
   * diagnostic boundary only; nonlocal XC is intentionally excluded. */
  generativeqc_status profile_fixed_density_components(const CudaKsFinalStateToken& expected,
                                                       CudaKsFixedDensityProfile& profile,
                                                       std::string& detail);
  /** Export a detached, strictly validated current physical state. Exact-token
   * comparison
   * precedes transfer; eligibility is rechecked before publication.
   * W is built only when
   * explicitly requested. */
  generativeqc_status read_final_state(const CudaKsFinalStateToken& expected,
                                       bool compute_weighted_density, VerifiedKsFinalState& state,
                                       std::string& detail);
  const CudaKsResources& resources() const noexcept;
  /** Optional prepared-operation provenance; null means that the owner did
   * not admit a separate preparation budget. No selection happens here. */
  const tensor::PanelProductDiagnostic* density_provider_diagnostic() const noexcept;
  CudaKsTransfers transfers() const noexcept;

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};
}  // namespace generativeqc::dft
