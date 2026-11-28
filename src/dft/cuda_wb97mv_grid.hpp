#pragma once

#include <cuda_runtime_api.h>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>

#include "dft/nonlocal_correlation/vv10_parameters.hpp"
#include "dft/semilocal_family.hpp"

namespace generativeqc::tensor {
class CudaContractionContext;
}

namespace generativeqc::dft {

inline constexpr int cuda_grid_nuclear_xc_invalid = 8;

struct CudaGridGeometryCartesianView;

using CudaWb97mvGridAuthenticate = cudaError_t (*)(void*) noexcept;

struct CudaWb97mvGridShape {
  std::size_t atoms{}, primitives{}, nao{}, points{};
  unsigned spins{1};
};

/** The same WB97M-V model can integrate its terms on one or two quadratures.
 * The nonlocal-only option integrates just the configured pair functional. */
enum class CudaWb97mvGridComponent { combined, semilocal_only, vv10_only };

struct CudaWb97mvGridOptions {
  std::size_t max_workspace_bytes{512ULL << 20};
  std::uint64_t max_pair_evaluations{16'777'216};
  /** Tile cap, not extra VV10 work. Smaller explicit tiles retain the bounded
   * low-memory schedule; 256 supplies enough independent pairs for modern GPUs. */
  std::size_t pair_tile_points{256};
  double symmetry_absolute_tolerance{1e-12};
  double symmetry_relative_tolerance{1e-10};
  bool nuclear_response{false};
  /** Replay the immutable affine mixed-contraction schedule with staged inputs.
   * Extra storage remains subordinate to max_workspace_bytes; unavailable graph
   * resources retain ordinary precompiled CUDA launches, never CPU execution. */
  bool replay_nuclear_maps{true};
  /** Budget for dense RKS AO cotangents shared by every left contraction. */
  std::size_t max_nuclear_adjoint_bytes{};
  /** Optional fixed-reference FP64 point Jacobians for repeated semilocal RKS actions. */
  std::size_t max_point_linearization_bytes{};
  std::uint64_t max_nuclear_affine_terms{1ULL << 34};
  /** Require bounded tiled FP64 contractions for supported wide dense RKS Hessians.
   * The enclosing owner must reserve the provider separately from the arena. */
  bool wide_cartesian_grid{false};
  /** Physical point code from the converged Fock identity. Curated and
   * split-hybrid families use disjoint codes. Response owners must verify
   * the corresponding stationary domain before constructing these options. */
  std::uint32_t functional_code{semilocal_family_code(SemilocalFamily::Wb97mv)};
  double exchange_scale{1.0}, correlation_scale{1.0};
  CudaWb97mvGridComponent component{CudaWb97mvGridComponent::combined};
  /** An explicit rVV10 provider replaces, rather than adds to, canonical
   * VV10. Molecular callers must source these options from the accepted KS
   * model; retaining only the semilocal part does not prepare a pair domain. */
  std::optional<nlc::Vv10Parameters> revised_nonlocal{};

  /** Point families independently admitted to the wide FP64 product layout. */
  [[nodiscard]] bool supports_wide_cartesian_grid() const noexcept {
    return functional_code == semilocal_family_code(SemilocalFamily::R2scan) ||
           functional_code == semilocal_family_code(SemilocalFamily::Pbe);
  }

  [[nodiscard]] bool includes_nonlocal() const noexcept {
    return (functional_code == semilocal_family_code(SemilocalFamily::Wb97mv) ||
            revised_nonlocal) &&
           component != CudaWb97mvGridComponent::semilocal_only;
  }
};

/** Immutable native quadrature/basis/density lease, not a detached state label.
 * Basis is the validated AoBasis normalized 16-double record format, points
 * are [point,xyz], weights [point], D [spin,AO,AO] row-major. RKS D is total;
 * UKS panels are physical alpha/beta D. All buffers are device-resident on the
 * supplied ordinary stream/device. The authenticator must check the actual
 * live owner, exact density token AND immutable basis/quadrature identity.
 * Owner/buffers/arena must survive all synchronous calls, without concurrent
 * writes. Copying this descriptor does not prove lifetime or scientific identity.
 */
struct CudaWb97mvGridReference {
  const double *basis{}, *points{}, *weights{}, *density{};
  int device_id{-1};
  cudaStream_t stream{};
  CudaWb97mvGridAuthenticate authenticate{};
  void* owner{};
};

struct CudaWb97mvGridWork {
  std::size_t workspace_bytes{}, active_nonlocal_points{};
  std::uint64_t reference_pairs{}, response_pairs{}, response_actions{};
  std::uint64_t nuclear_reference_pairs{}, nuclear_response_pairs{}, nuclear_affine_terms{};
  std::uint64_t nuclear_directions{}, nuclear_contractions{};
  std::uint64_t nuclear_graph_replays{};
  std::uint64_t nuclear_adjoint_builds{}, nuclear_adjoint_contractions{};
  std::uint64_t transition_projection_products{}, transition_point_corrections{};
  bool pair_budget_exceeded{};
};

/** Pure checked sizing. Pair storage and total ordered-pair WORK are separate
 * gates. Actual active-count squared is checked at preparation, before pairs.
 * The arena includes the complete dense first-jet panel and map scratch;
 * density-dependent screening is fixed by the reference on both VV10 legs.
 */
cudaError_t cuda_wb97mv_grid_workspace(const CudaWb97mvGridShape& shape,
                                       const CudaWb97mvGridOptions& options,
                                       std::size_t& bytes) noexcept;

/** Native fixed- and moving-grid XC response for the admitted physical models.
 * LDA/PBE/B3LYP reuse their SCF point models, including their vacuum domains; they
 * do not discover, allocate or traverse a VV10 pair domain.
 * Scalar gradients/JVPs and feature/potential maps are compiler AD, generated
 * at build time. Includes physical spin coupling, tau halves, both VV10 legs,
 * ordered-pair traversal and the local beta term. Optional CUDA graph resources
 * are created during prepare; actions do not allocate. No interpreter, runtime
 * compilation, CPU contraction, device switch or hidden fallback.
 * Each call fences at numerical/authentication gates; only status/active-count
 * metadata reaches the host. A failed action leaves public output unchanged
 * before publication; a CUDA error during publication invalidates that output.
 * This is the grid component of induced Fock, NOT J/K or a complete Hessian.
 */
class CudaWb97mvGridResponse {
 public:
  CudaWb97mvGridResponse();
  ~CudaWb97mvGridResponse();
  static std::size_t wide_grid_provider_allowance() noexcept;
  CudaWb97mvGridResponse(const CudaWb97mvGridResponse&) = delete;
  CudaWb97mvGridResponse& operator=(const CudaWb97mvGridResponse&) = delete;
  CudaWb97mvGridResponse(CudaWb97mvGridResponse&&) = delete;
  CudaWb97mvGridResponse& operator=(CudaWb97mvGridResponse&&) = delete;

  /** Precompute immutable AOs/fields/reference Vxc within the caller arena.
   * Failure revokes this instance. A retry requires an intact live reference.
   */
  cudaError_t prepare(const CudaWb97mvGridShape& shape, const CudaWb97mvGridOptions& options,
                      const CudaWb97mvGridReference& reference, void* workspace,
                      std::size_t workspace_bytes) noexcept;
  /** Attempt optional fixed-grid residency after the enclosing owner has
   * admitted all mandatory resources. Capacity refusal keeps the scalar JVP. */
  cudaError_t retain_fixed_point_linearization(std::size_t max_bytes) noexcept;
  /** Optional immutable P·AO planes for repeated wide nuclear directions. */
  cudaError_t retain_wide_nuclear_products(std::size_t max_bytes) noexcept;
  /** RKS semilocal tiles for the cached occupied/virtual XC operator. Fields
   * and their weighted tangents are [rotation][point][axis]; the row
   * dimension of both column-major matrices is axes * count. A sticky
   * refused flag requests the original full-operator build. The caller owns
   * all disjoint scratch and the stream fence before checking that flag. */
  unsigned transition_axes() const noexcept;
  cudaError_t enqueue_transition_census(int* count, cudaStream_t stream) noexcept;
  cudaError_t enqueue_transition_tile(const double* mo, std::size_t occupied, std::size_t begin,
                                      std::size_t count, double* mo_jets, double* fields,
                                      double* tangents, bool correct, int* refused,
                                      cudaStream_t stream) noexcept;
  /** Flatten the exact metric density JVP's grid features for all Cartesian
   * RHSs. Planes are [jet][point][AO] or [jet][point][axis,AO]. */
  cudaError_t enqueue_metric_feature_tile(std::size_t count, const int* atom_offsets,
                                          const double* all_jets, const double* occupied_jets,
                                          const double* all_rows, const double* occupied_rows,
                                          double* fields, int* refused,
                                          cudaStream_t stream) noexcept;
  void record_transition_projection(std::uint64_t products, std::uint64_t corrections) noexcept {
    work_.transition_projection_products = products;
    work_.transition_point_corrections = corrections;
  }
  /** Apply to a finite symmetric SIGNED/indefinite D direction, with both UKS
   * blocks together. Output is [spin,AO,AO], disjoint from all retained inputs.
   */
  cudaError_t apply(const double* direction, double* output, cudaStream_t stream) noexcept;
  /** Enqueue the same response into a private downstream Fock panel. A
   * numerical failure writes NaNs instead of publishing unchecked data; the
   * consumer must audit its candidate before exposing a result. The consumer
   * owns stream completion, including errors after partial enqueue, and
   * accounts actual replay rather than host record-time calls. */
  cudaError_t enqueue_private_response(const double* direction, double* output,
                                       cudaStream_t stream) noexcept;
  /** Borrow the prepared native value-side potential under the same live lease.
   * It remains internal, and must not be written or retained after revocation.
   */
  cudaError_t reference_potential(const double*& potential) noexcept;
  /** Opt-in moving-grid direction, including signed density motion. AO jets are
   * [4,point,AO], point motion [point,xyz]. Inputs are copied into the arena;
   * they need not survive this synchronous call. A successful call arms mixed
   * contractions and publishes [spin,AO,AO] dVxc. Any prepare/apply/direction
   * attempt expires that cache, including failed attempts. No pair traversal
   * is repeated for subsequent left-coordinate contractions. */
  cudaError_t prepare_nuclear_direction(const double* ao, const double* weights,
                                        const double* points, const double* density,
                                        double* potential, cudaStream_t stream) noexcept;
  /** Enqueue the right direction against a private device transaction. Only
   * mixed contractions using that same status may consume the right cache. */
  cudaError_t enqueue_private_nuclear_direction(const double* ao, const double* weights,
                                                const double* points, const double* density,
                                                double* potential, int* transaction_status,
                                                cudaStream_t stream) noexcept;
  /** Frozen-density counterpart, with no density-direction input to replay. */
  cudaError_t enqueue_private_frozen_nuclear_direction(const double* ao, const double* weights,
                                                       const double* points, double* potential,
                                                       int* transaction_status,
                                                       cudaStream_t stream) noexcept;
  /** Arm the private mixed contractions without assembling an unused dVxc. */
  cudaError_t enqueue_private_nuclear_covector(const double* ao, const double* weights,
                                               const double* points, const double* density,
                                               int* transaction_status,
                                               cudaStream_t stream) noexcept;
  /** Add the CPKS density to the retained frozen-motion features from the
   * same private transaction. No second AO/weight/point sweep is needed. */
  cudaError_t enqueue_private_induced_density_covector(const double* density,
                                                       int* transaction_status,
                                                       cudaStream_t stream) noexcept;
  /** Stationary left nuclear partial with zero left/mixed density motion.
   * The retained right density response participates in the mixed AO features.
   * Mixed point motion is zero for affine owner-atom coordinates. The scalar
   * output is device resident and transactional up to CUDA publication errors. */
  cudaError_t nuclear_mixed(const double* left_ao, const double* mixed_ao,
                            const double* left_weights, const double* mixed_weights,
                            const double* left_points, double* output,
                            cudaStream_t stream) noexcept;
  /** Enqueue a contraction into a private scalar, deferring its numerical
   * audit into the enclosing device transaction. The transaction owns a
   * separate, initialized status flag and must drain and check it before
   * any external publication. Neither argument may alias grid storage. */
  cudaError_t enqueue_private_nuclear_mixed(const double* left_ao, const double* mixed_ao,
                                            const double* left_weights, const double* mixed_weights,
                                            const double* left_points, double* output,
                                            int* transaction_status, cudaStream_t stream) noexcept;
  /** Admit the number of rows that fit the existing adjoint scratch. */
  bool admits_cartesian_mixed(std::size_t rows = 1) const noexcept;
  /** Borrow the frozen semilocal weight-derivative activity for this grid lease.
   * Null means every point is required. A zero is not a quadrature deletion. */
  const double* private_becke_activity() const noexcept;
  /** Borrow unweighted local XC energy density, fixed for this density lease. */
  const double* private_becke_energy() const noexcept;
  /** Contract consecutive rows of a live geometry lease into a private panel. */
  cudaError_t enqueue_private_cartesian_mixed(const CudaGridGeometryCartesianView& motion,
                                              double* output, int* transaction_status,
                                              cudaStream_t stream) noexcept;
  /** Borrow the opt-in prepared grid energy scalar under the live lease. */
  cudaError_t reference_energy(const double*& energy) noexcept;
  const CudaWb97mvGridWork& work() const noexcept { return work_; }

 private:
  void release_point_workspace() noexcept;
  void release_wide_nuclear_products() noexcept;
  cudaError_t authenticate(cudaStream_t stream, bool composable = false) noexcept;
  cudaError_t apply_impl(const double* direction, double* output, cudaStream_t stream,
                         bool private_panel) noexcept;
  cudaError_t evaluate(bool response, bool defer_status = false) noexcept;
  cudaError_t evaluate_nuclear(bool response, bool defer_status = false,
                               bool assemble_potential = true) noexcept;
  cudaError_t prepare_nuclear_direction_impl(const double* ao, const double* weights,
                                             const double* points, const double* density,
                                             double* potential, int* transaction_status,
                                             cudaStream_t stream,
                                             bool frozen_density = false) noexcept;
  cudaError_t build_nuclear_graph() noexcept;
  cudaError_t enqueue_nuclear_mixed(const double* left_ao, const double* mixed_ao,
                                    const double* left_weights, const double* mixed_weights,
                                    const double* left_points, cudaStream_t stream) noexcept;
  cudaError_t nuclear_mixed_impl(const double* left_ao, const double* mixed_ao,
                                 const double* left_weights, const double* mixed_weights,
                                 const double* left_points, double* output, int* transaction_status,
                                 cudaStream_t stream,
                                 const CudaGridGeometryCartesianView* cartesian = nullptr) noexcept;
  CudaWb97mvGridShape shape_{};
  CudaWb97mvGridOptions options_{};
  CudaWb97mvGridReference reference_{};
  CudaWb97mvGridWork work_{};
  std::unique_ptr<tensor::CudaContractionContext> wide_grid_blas_;
  void* workspace_{};
  void* fixed_point_workspace_{};
  double* wide_nuclear_products_{};
  std::size_t fixed_point_bytes_{}, fixed_point_mask_offset_{};
  unsigned fixed_point_axes_{};
  double beta_{};
  bool ready_{};
  bool nuclear_ready_{};
  bool nuclear_adjoint_ready_{};
  bool nuclear_frozen_density_{};
  int* private_nuclear_status_{};
  cudaGraphExec_t nuclear_graph_{};
};

}  // namespace generativeqc::dft
