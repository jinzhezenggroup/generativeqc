#pragma once

#include <cuda_runtime_api.h>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <span>
#include <string>

#include "integrals/cuda_one_electron_response.hpp"

namespace generativeqc::integrals {
struct CoulombDerivativeOmissions;
}

namespace generativeqc::dft {

/** Full+LR is the exact WB97M-V operator. Fitted primary J/K is NOT supplied
 * here: a DF owner may consume only the LR correction beside its fitted source.
 * Separate radial parts are exposed for independent scientific qualification. */
enum class CudaWb97mvJkPart {
  exact,
  primary,
  correction,
  coulomb,
  short_range,
  long_range,
  full_hybrid
};

struct CudaWb97mvJkShape {
  std::size_t atoms{}, primitives{}, nao{}, spins{};
};

struct CudaWb97mvJkOptions {
  /** Immutable physical exchange weights, independent of spin density convention. */
  struct RangeExchange {
    double short_fraction{}, long_minus_short{}, omega{};
    bool operator==(const RangeExchange&) const = default;
  };
  CudaWb97mvJkPart part{CudaWb97mvJkPart::exact};
  std::size_t max_workspace_bytes{512ULL << 20};
  /** Optional source/projection batching, charged in the same workspace arena.
   * Zero or insufficient capacity retains the ordered per-group schedule. */
  std::size_t max_batch_workspace_bytes{128ULL << 20};
  // Independent bound on read-only cross-window residency; zero streams it.
  std::size_t max_retained_window_source_bytes{SIZE_MAX};
  /** Reusable projected shell-group tile, within the same arena/cap. Small
   * tiles retain group/incidence sum order; zero disables optional batching. */
  std::size_t max_batch_projection_bytes{16ULL << 20};
  std::size_t max_groups{100'000};
  /** Bounded canonical-source replay when the ordered schedule or its requested
   * resident jet cache does not fit. Source/work bounds still apply across all
   * tiles, which reuse transient storage in the same arena.
   * Zero disables this fallback; it never enlarges the caller's arena. */
  std::size_t source_tile{1024};
  std::size_t max_source_groups{100'000};
  /** Certified dense-only replay bounds individually visited sources per pass.
   * Skipped sources are visited only as bitmap words, separately bounded by
   * ceil(limit/32) even when very few bits are set. Zero disables replay.
   * The primitive/projection/reduction and live arena limits still apply. */
  std::size_t max_frozen_replay_candidates{};
  /** Bound a certified window's right-atom union by grouping source quartets
   * with matching blocks of non-anchor atoms. Unscreened replay is unchanged. */
  std::size_t frozen_atom_block_size{};
  std::size_t radial_tile{64};  ///< Primitive streaming bound, in [1,64].
  std::uint64_t max_primitive_component_evaluations{67'108'864};
  std::uint64_t max_projection_terms{268'435'456};
  std::uint64_t max_reduction_terms{1'073'741'824};
  double symmetry_absolute_tolerance{1e-12};
  double symmetry_relative_tolerance{1e-10};
  /// Cap on persistent last-direction jets, not transient streamed tiles.
  std::size_t max_resident_integral_bytes{64ULL << 20};
  /** Prefer batched windows over unbatched resident jets for fresh directions.
   * Leave off for HVP pairs that can reuse the full last-direction cache. */
  bool prefer_fresh_direction_batches{};
  /** Permit the contracted s/p sweep only for a certified dense-only source.
   * The caller first checks the device's opt-in shared memory capacity. */
  bool contract_frozen_coulomb{};
  /** Physical fraction of same-spin exchange for full_hybrid; zero otherwise. */
  double full_exchange_fraction{};
  /** Bind both radial sources from the accepted KS state. Unbound leaf probes
   * retain the canonical WB97M-V component selectors. */
  std::optional<RangeExchange> range_exchange;
};

/** Borrow the actual normalized AoBasis and an immutable live owner. Geometry,
 * radial normalization and public Cartesian/spherical projection coefficients
 * remain on device. The owner authenticates storage/topology/generation as well
 * as its original device and stream. It and the arena outlive synchronous calls. */
struct CudaWb97mvJkReference {
  const double* basis{};
  int device_id{-1};
  cudaStream_t stream{};
  cudaError_t (*authenticate)(void*) noexcept {};
  void* owner{};
};

/** Five jointly published outputs. Fock panels are [spin,AO,AO], vectors are
 * [atom,xyz]. Fock direction is the frozen-density nuclear partial, whereas HVP
 * retains the supplied right density response in the stationary weight JVP.
 * This component alone is not a stationary Hessian or total energy gradient. */
struct CudaWb97mvJkOutput {
  double *energy{}, *fock{}, *fock_direction{}, *gradient{}, *hvp{};
};

struct CudaWb97mvJkWork {
  std::size_t workspace_bytes{}, groups{}, primitive_records_per_action{};
  std::uint64_t component_evaluations_per_action{}, projection_terms_bound_per_action{},
      reduction_terms_per_action{};
  std::uint64_t attempted_actions{}, completed_actions{}, component_evaluations{},
      projection_terms_bound{}, reduction_terms{};
  std::size_t resident_integral_bytes{};
  std::size_t batched_workspace_bytes{};
  /** Read-only window sources retained once in addition to the reused scratch. */
  std::size_t retained_window_source_bytes{};
  std::uint64_t integral_launches{}, projection_launches{}, reduction_launches{};
  std::uint64_t integral_source_builds{}, resident_integral_hits{};
  std::size_t streaming_tiles{}, source_groups{};
  std::size_t omitted_frozen_source_groups{};
  double omitted_frozen_derivative_bound{};
  /** Cache fill work and primitive components consuming prepared geometry. */
  std::uint64_t prepared_geometry_records_bound{}, prepared_geometry_components{};
  std::uint64_t indexed_component_blocks{};
};

/** The source census is the only admission failure a certified omission can retry. */
enum class CudaWb97mvJkAdmissionFailure { other, canonical_source_limit };

/** Host-only checked closure/work/arena admission, without CUDA initialization.
 * Every ordered shell class and raw component must be linked exactly once for
 * each requested radial kernel/omega. No name parsing, discovery or JIT exists. */
cudaError_t cuda_wb97mv_jk_workspace(
    const CudaWb97mvJkShape& shape, std::span<const integrals::CudaOneElectronShell> shells,
    const CudaWb97mvJkOptions& options, std::size_t& bytes, std::string* detail = nullptr,
    const integrals::CoulombDerivativeOmissions* omissions = nullptr,
    CudaWb97mvJkAdmissionFailure* failure = nullptr) noexcept;

/** Native compiler-AD J/K nuclear source, with signed symmetric D and dD.
 * Raw value/gradient/gradient-JVP are evaluated once per primitive component and
 * streamed through the current shell quartet (<=25*10^4 FP64 values through f).
 * When its explicit memory cap and the supplied arena permit, an optional
 * resident cache retains all jets for the last right direction. Its key is the
 * complete device direction, compared exactly, under the authenticated immutable
 * basis lease. D and dD are not cache keys: contractions always use their current
 * values. Reprepare and failed cache construction invalidate resident contents.
 * Resident groups share ERI eightfold symmetry through explicit Cartesian and
 * derivative-slot permutations; the disabled/budget fallback stays ordered.
 * Public transforms gather that mandatory bounded cache, never repeat integral
 * work per AO or left coordinate. All numerical contraction and deterministic
 * reductions run on CUDA; actions allocate nothing and invoke no Python/CPU
 * numerical work. Earlier failures preserve all public outputs. A CUDA failure
 * during publication invalidates all five outputs; callers must discard them.
 */
class CudaWb97mvNuclearJk {
 public:
  /** Single-use, host-only admission plan. Owns the topology and options
   * that sized the arena; the live basis is still authenticated on binding. */
  class Plan {
   public:
    Plan();
    ~Plan();
    Plan(const Plan&) = delete;
    Plan& operator=(const Plan&) = delete;

    cudaError_t build(const CudaWb97mvJkShape& shape,
                      std::span<const integrals::CudaOneElectronShell> shells,
                      const CudaWb97mvJkOptions& options, std::string* detail = nullptr,
                      const integrals::CoulombDerivativeOmissions* omissions = nullptr,
                      CudaWb97mvJkAdmissionFailure* failure = nullptr) noexcept;
    std::size_t bytes() const noexcept;

   private:
    friend class CudaWb97mvNuclearJk;
    struct Impl;
    std::unique_ptr<Impl> impl_;
  };

  CudaWb97mvNuclearJk();
  ~CudaWb97mvNuclearJk();
  CudaWb97mvNuclearJk(const CudaWb97mvNuclearJk&) = delete;
  CudaWb97mvNuclearJk& operator=(const CudaWb97mvNuclearJk&) = delete;
  CudaWb97mvNuclearJk(CudaWb97mvNuclearJk&&) = delete;
  CudaWb97mvNuclearJk& operator=(CudaWb97mvNuclearJk&&) = delete;

  cudaError_t prepare(const CudaWb97mvJkShape& shape,
                      std::span<const integrals::CudaOneElectronShell> shells,
                      const CudaWb97mvJkOptions& options, const CudaWb97mvJkReference& reference,
                      void* workspace, std::size_t workspace_bytes,
                      const integrals::CoulombDerivativeOmissions* omissions = nullptr) noexcept;
  cudaError_t prepare(Plan& plan, const CudaWb97mvJkReference& reference, void* workspace,
                      std::size_t workspace_bytes) noexcept;
  cudaError_t apply(const double* direction, const double* density, const double* delta_density,
                    const CudaWb97mvJkOutput& output, cudaStream_t stream) noexcept;
  /** Enqueue a fresh action into private buffers. The caller owns the sticky
   * transaction audit and must drain and accept it before using the outputs.
   * This path neither consumes nor authorizes the optional direction cache.
   * With a Cartesian column, the direction must be that unit vector and
   * delta_density must be zero. Only the directional Fock and HVP are written. */
  cudaError_t enqueue_private_fresh(const double* direction, const double* density,
                                    const double* delta_density, const CudaWb97mvJkOutput& output,
                                    int* transaction_status, cudaStream_t stream,
                                    std::size_t frozen_column = SIZE_MAX) noexcept;
  /** Visit streamed source windows once for the complete Cartesian sweep.
   * The caller owns direction scratch, a zero delta-density panel and a private
   * column-major [Fock columns][HVP columns] table. The table is readable only
   * after the enclosing transaction accepts it. NotSupported enqueues nothing. */
  cudaError_t enqueue_private_frozen_dense(double* direction, const double* density,
                                           const double* zero_delta_density, double* table,
                                           int* transaction_status, cudaStream_t stream) noexcept;
  const CudaWb97mvJkWork& work() const noexcept;

 private:
  cudaError_t apply_impl(const double* direction, const double* density,
                         const double* delta_density, const CudaWb97mvJkOutput& output,
                         int* transaction_status, cudaStream_t stream,
                         std::size_t frozen_column) noexcept;
  struct Impl;
  std::unique_ptr<Impl> impl_;
};
}  // namespace generativeqc::dft
