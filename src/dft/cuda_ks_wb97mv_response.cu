#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <utility>

#include "dft/cuda_hessian_semilocal_model.hpp"
#include "dft/cuda_ks_grid_reference.hpp"
#include "dft/cuda_ks_wb97mv_model.hpp"
#include "dft/cuda_ks_wb97mv_response.hpp"
#include "dft/r2scan_rvv10_model.hpp"
#include "dft/semilocal_family.hpp"
#include "dft/split_hybrid_cuda_selection.hpp"
#include "generated_split_hybrid_registry.cuh"
#include "molecule/basis.hpp"
#include "r2scan_cuda_domain.hpp"
#include "response/wb97mv_fock_response.cuh"
#include "runtime/resource_cuda.cuh"
#include "scf/cuda_density_fitting_basis.hpp"
#include "scf/cuda_density_fitting_device.hpp"
#include "scf/cuda_direct_jk_device.hpp"
#include "scf/cuda_fock_execution.hpp"
#include "tensor/cuda_dense_operations.hpp"

namespace generativeqc::dft {
namespace {
/** An exact LR term shares no fitted lease. Fitted LR must retain its own
 * accepted operator/metric on the same orbital system as the fitted primary. */
bool range_provider_matches(const CudaKsResidentFockBinding& source) {
  const auto& correction = *source.correction;
  const auto& primary = source.primary->strategy();
  if (correction.spec.exchange.approximation == scf::FockApproximation::Exact)
    return !source.fitted_range && correction.metric_relative_threshold == 0;
  return correction.spec.exchange.approximation == scf::FockApproximation::DensityFitted &&
         primary.spec.coulomb.approximation == scf::FockApproximation::DensityFitted &&
         correction.metric_relative_threshold == primary.metric_relative_threshold &&
         source.fitted_range && source.fitted_range != source.primary &&
         source.fitted_range->strategy() == correction &&
         source.fitted_range->matches_system(source.primary->system());
}
}  // namespace

bool cuda_ks_wb97mv_model_matches(const CudaKsFinalStateToken& expected,
                                  const CudaKsResidentFockBinding& source) {
  if (!source) return false;
  namespace maps = generativeqc_wb97mv_fock_maps;
  const auto& model = expected.identity.model;
  const auto& strategy = source.primary->strategy();
  const auto& spec = strategy.spec;
  const auto occupation = source.spins == 1 ? 2.0 : 1.0;
  if (!(model.version == 1 &&
        model.scf_domain_version == semilocal_family_domain_version(SemilocalFamily::Wb97mv) &&
        model.functional == semilocal_family_code(SemilocalFamily::Wb97mv) &&
        model.spins == source.spins && model.device == source.device_id &&
        model.semilocal_exchange_scale == 1 && model.semilocal_correlation_scale == 1 &&
        model.nonlocal_correlation &&
        model.nonlocal_correlation->variant == nlc::Vv10Variant::vv10 &&
        model.nonlocal_correlation->b == maps::nonlocal_b &&
        model.nonlocal_correlation->c == maps::nonlocal_c &&
        model.nonlocal_correlation->coefficient == maps::nonlocal_coefficient &&
        model.nonlocal_density_domain == nlc::Vv10DensityDomain::MolecularV1))
    return false;
  if (!(strategy == expected.identity.determinant.model &&
        strategy.backend == scf::FockBackend::Cuda && spec.version == 1 &&
        spec.derivative_order == 0 &&
        spec.spin ==
            (source.spins == 1 ? scf::FockSpin::Restricted : scf::FockSpin::Unrestricted) &&
        spec.coulomb.present && spec.coulomb.coefficient == 1 &&
        spec.coulomb.op == scf::FockOperator::FullRange && spec.coulomb.omega == 0 &&
        spec.exchange.present && spec.exchange.op == scf::FockOperator::FullRange &&
        spec.exchange.omega == 0 &&
        spec.exchange.coefficient == -maps::short_exchange / occupation &&
        spec.coulomb.approximation == spec.exchange.approximation &&
        (spec.coulomb.approximation == scf::FockApproximation::Exact ||
         spec.coulomb.approximation == scf::FockApproximation::DensityFitted)))
    return false;
  if (!source.correction || !model.range_correction ||
      *source.correction != *model.range_correction)
    return false;
  const auto& correction = source.correction->spec;
  if (!(source.correction->backend == scf::FockBackend::Cuda && correction.version == 1 &&
        correction.derivative_order == 0 && correction.spin == spec.spin &&
        !correction.coulomb.present && correction.exchange.present &&
        correction.exchange.op == scf::FockOperator::LongRange &&
        correction.exchange.omega == maps::omega &&
        correction.exchange.coefficient == -maps::range_correction / occupation &&
        source.correction->screening_tolerance == strategy.screening_tolerance))
    return false;
  return range_provider_matches(source);
}

bool cuda_ks_hessian_model_matches(const CudaKsFinalStateToken& expected,
                                   const CudaKsResidentFockBinding& source,
                                   CudaWb97mvGridOptions* grid) {
  if (!source) return false;
  const auto& model = expected.identity.model;
  const auto* family = semilocal_family_metadata_from_code(model.functional);
  const auto split_domain = cuda_split_hybrid_variational_domain(model.functional);
  if (!family && !split_domain) return false;
  bool revised = false;
  if (family && family->family == SemilocalFamily::Wb97mv) {
    if (!cuda_ks_wb97mv_model_matches(expected, source)) return false;
  } else {
    const auto& strategy = source.primary->strategy();
    const auto& spec = strategy.spec;
    const auto approximation = spec.coulomb.approximation;
    const double occupation = source.spins == 1 ? 2 : 1;
    const double fraction = spec.exchange.present ? -spec.exchange.coefficient * occupation : 0;
    const bool range_pbe =
        family && family->family == SemilocalFamily::Pbe && model.range_correction.has_value();
    if (range_pbe) {
      // PBE carries no intrinsic range separation. Recover both HF fractions
      // from the accepted Fock terms (the correction can have either sign).
      if (!expected.strict_physical_fock || expected.relaxed_stationarity ||
          model.grid.partition_iterations != 3 || !source.correction ||
          *source.correction != *model.range_correction)
        return false;
      const auto& correction = *source.correction;
      const auto& range = correction.spec.exchange;
      const double long_fraction = fraction - occupation * range.coefficient;
      if (!(strategy.precision == scf::FockPrecision::Float64 &&
            correction.backend == scf::FockBackend::Cuda &&
            correction.precision == scf::FockPrecision::Float64 &&
            correction.schedule == scf::FockSchedule::CudaIndependent &&
            correction.spec.version == 1 && correction.spec.derivative_order == 0 &&
            correction.spec.spin == spec.spin && !correction.spec.coulomb.present &&
            range.present && range.op == scf::FockOperator::LongRange && range.coefficient != 0 &&
            std::isfinite(range.omega) && range.omega > 0 && fraction >= 0 && fraction <= 1 &&
            long_fraction >= 0 && long_fraction <= 1 &&
            correction.screening_tolerance == strategy.screening_tolerance &&
            range_provider_matches(source)))
        return false;
    }
    const bool r2scan = family && family->family == SemilocalFamily::R2scan;
    revised = r2scan && model.nonlocal_correlation &&
              r2scan_rvv10_model_matches(family->family, *model.nonlocal_correlation,
                                         model.semilocal_exchange_scale,
                                         model.semilocal_correlation_scale, spec.exchange.present);
    const bool matched_semilocal =
        split_domain ? cuda_split_hybrid_variational_model(
                           model.functional, strategy, source.spins, model.grid,
                           model.semilocal_exchange_scale, model.semilocal_correlation_scale)
                     : cuda_hessian_canonical_semilocal(family->family, range_pbe ? 0.0 : fraction,
                                                        model.semilocal_exchange_scale,
                                                        model.semilocal_correlation_scale);
    // The legacy split-hybrid domain holds work-coordinate derivatives, so
    // only a matched physical-coordinate identity may use the split-hybrid grid action.
    const auto domain = split_domain ? split_domain
                        : r2scan     ? kCudaR2scanVariationalDomain
                                     : family->domain_version;
    if (!matched_semilocal || model.version != 1 || model.scf_domain_version != domain ||
        model.spins != source.spins || model.device != source.device_id ||
        (!revised && (model.nonlocal_correlation || model.nonlocal_grid)) ||
        (revised && (model.nonlocal_density_domain != nlc::Vv10DensityDomain::MolecularV1 ||
                     model.grid.partition_iterations != 3 ||
                     (model.nonlocal_grid && (model.nonlocal_grid->partition_iterations != 3 ||
                                              *model.nonlocal_grid == model.grid)))) ||
        (!range_pbe && (model.range_correction || source.correction || source.fitted_range)) ||
        strategy != expected.identity.determinant.model ||
        strategy.backend != scf::FockBackend::Cuda || spec.version != 1 ||
        spec.derivative_order != 0 ||
        spec.spin !=
            (source.spins == 1 ? scf::FockSpin::Restricted : scf::FockSpin::Unrestricted) ||
        !spec.coulomb.present || spec.coulomb.coefficient != 1 ||
        spec.coulomb.op != scf::FockOperator::FullRange || spec.coulomb.omega != 0 ||
        (approximation != scf::FockApproximation::Exact &&
         approximation != scf::FockApproximation::DensityFitted))
      return false;
    if (spec.exchange.present &&
        (fraction == 0 || spec.exchange.op != scf::FockOperator::FullRange ||
         spec.exchange.omega != 0 || spec.exchange.approximation != approximation))
      return false;
  }
  if (grid) {
    grid->functional_code = model.functional;
    grid->exchange_scale = model.semilocal_exchange_scale;
    grid->correlation_scale = model.semilocal_correlation_scale;
    grid->revised_nonlocal = revised ? model.nonlocal_correlation : std::nullopt;
    grid->component = model.nonlocal_grid ? CudaWb97mvGridComponent::semilocal_only
                                          : CudaWb97mvGridComponent::combined;
  }
  return true;
}
namespace {
namespace maps = generativeqc_wb97mv_fock_maps;

__global__ void sum_grid_potentials(const double* primary, const double* vv10, double* output,
                                    std::size_t count) {
  for (auto index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x; index < count;
       index += static_cast<std::size_t>(blockDim.x) * gridDim.x)
    output[index] = primary[index] + vv10[index];
}

void require(bool condition, const char* detail) {
  if (!condition) throw std::invalid_argument(detail);
}
void check(cudaError_t error) {
  if (error == cudaErrorMemoryAllocation) throw std::bad_alloc();
  if (error != cudaSuccess) throw std::runtime_error(cudaGetErrorString(error));
}
std::size_t product(std::size_t first, std::size_t second) {
  require(!second || first <= std::numeric_limits<std::size_t>::max() / second,
          "physical Fock extent overflow");
  return first * second;
}
struct Layout {
  std::size_t bytes{};
  std::size_t add(std::size_t count) {
    const auto increment = product(count, sizeof(double));
    require(increment <= std::numeric_limits<std::size_t>::max() - 255 &&
                bytes <= std::numeric_limits<std::size_t>::max() - increment - 255,
            "physical Fock arena overflow");
    const auto offset = bytes;
    bytes = (bytes + increment + 255) / 256 * 256;
    return offset;
  }
};
bool overlap(const void* first, std::size_t first_bytes, const void* second,
             std::size_t second_bytes) {
  const auto left = reinterpret_cast<std::uintptr_t>(first);
  const auto right = reinterpret_cast<std::uintptr_t>(second);
  if (!first_bytes || !second_bytes) return false;
  return left <= right ? right - left < first_bytes : left - right < second_bytes;
}
__global__ void audit(const double* value, const double* reference, std::size_t nao, unsigned spins,
                      double absolute, double relative, int* status) {
  const auto matrix = nao * nao;
  for (std::size_t index = blockIdx.x * blockDim.x + threadIdx.x; index < spins * matrix;
       index += static_cast<std::size_t>(blockDim.x) * gridDim.x) {
    const auto transposed = index / matrix * matrix + index % nao * nao + index / nao % nao;
    const auto current = value[index];
    const auto other = reference ? reference[index] : value[transposed];
    if (!isfinite(current) || !isfinite(other))
      atomicOr(status, 2);
    else if (fabs(current - other) > absolute + relative * fmax(fabs(current), fabs(other)))
      atomicOr(status, 4);
  }
}

struct ActionCounts {
  std::uint64_t quartets, eris, pair_product_bound, actions, grid_actions;
};

// These slots have independent owners: each Direct enqueue resets its own
// flag. The generated maps must never clear a previous Direct failure.
constexpr unsigned status_count = 3;
constexpr std::size_t transition_tile_points = 128;
constexpr std::uint64_t transition_product_limit = 64ULL << 30;
// A wide FP64 semilocal operator can reuse this projection across every
// Cartesian solve; keep its one-time dense work independently bounded.
constexpr std::uint64_t wide_r2scan_transition_product_limit = 3ULL << 40;
constexpr std::uint64_t transition_correction_limit = 2ULL << 20;

__device__ bool failed_action(const int* status) { return status[0] || status[1] || status[2]; }

__global__ void count_action(const std::uint64_t* counts, std::uint64_t bound,
                             std::uint64_t grid_bound, std::uint64_t pair_products,
                             ActionCounts* total, int* status, bool grid) {
  if (grid && total->grid_actions == grid_bound) {
    status[0] |= 8;
    return;
  }
  // Grid quadrature was submitted even if the Fock candidate fails its audit.
  total->grid_actions += grid;
  if (failed_action(status)) return;
  if (counts[0] > bound || counts[2] > bound - counts[0] || counts[1] > bound ||
      counts[3] > bound - counts[1]) {
    status[0] |= 8;
    return;
  }
  const auto quartets = counts[0] + counts[2], eris = counts[1] + counts[3];
  if (quartets > UINT64_MAX - total->quartets || eris > UINT64_MAX - total->eris ||
      pair_products > UINT64_MAX - total->pair_product_bound || total->actions == UINT64_MAX) {
    status[0] |= 8;
    return;
  }
  total->quartets += quartets;
  total->eris += eris;
  total->pair_product_bound += pair_products;
  ++total->actions;
}

__global__ void publish_private_fock(const double* candidate, double* output, std::size_t count,
                                     const int* status) {
  for (std::size_t index = blockIdx.x * blockDim.x + threadIdx.x; index < count;
       index += static_cast<std::size_t>(gridDim.x) * blockDim.x)
    output[index] = failed_action(status) ? nan("") : candidate[index];
}
}  // namespace

struct CudaKsWb97mvResponse::Impl {
  methods::PreparedBatch& batch;
  std::size_t item;
  CudaKsFinalStateToken expected;
  CudaKsGridReference reference, vv10_reference;
  CudaKsResidentFockBinding source{};
  CudaKsResidentOrbitalFrameBinding frame{};
  scf::CudaDfReadonlySource fitted{};
  scf::CudaDfReadonlySource fitted_range{};
  scf::CudaDfReadonlyMetric range_metric{};
  scf::CudaDfReadonlyBasis range_basis{};
  scf::CudaDirectJkPlan* direct{};
  scf::CudaDirectJkPlan* range_direct{};
  CudaWb97mvGridResponse grid, vv10_grid;
  CudaKsWb97mvResponseOptions options;
  CudaKsWb97mvResponseWork receipt{};
  maps::Extents shape{}, range_shape{};
  cudaStream_t stream{};
  void* arena{};
  std::size_t arena_bytes{}, matrix{}, elements{}, map_bytes{}, compensation_elements{};
  void *map_workspace{}, *grid_workspace{};
  std::size_t grid_bytes{};
  tensor::CudaDenseOperations transition_blas;
  void* transition_workspace{};
  std::size_t transition_occupied{};
  std::uint64_t transition_products{};
  double *coulomb{}, *exchange{}, *long_range{}, *primary{}, *potential{}, *candidate{}, *zero{},
      *fitted_panel{}, *range_whitened{}, *seed{}, *exchange_coefficient{}, *range_coefficient{},
      *compensation{};
  int* status{};
  std::uint64_t *census{}, *range_census{};
  ActionCounts* action_counts{};
  std::uint64_t grid_actions{};
  std::size_t pair_product_bound{};
  bool ready{}, used{};
  const char* failure_check{};

  Impl(methods::PreparedBatch& owner, std::size_t index, CudaKsFinalStateToken token,
       CudaKsWb97mvResponseOptions controls)
      : batch(owner),
        item(index),
        expected(std::move(token)),
        reference(batch, item, expected),
        vv10_reference(batch, item, expected, CudaKsGridRole::separate_nonlocal),
        options(controls) {}
  ~Impl() {
    if (stream) (void)cudaStreamSynchronize(stream);
    scf::destroy_cuda_direct_jk_plan(range_direct);
    scf::destroy_cuda_direct_jk_plan(direct);
    (void)transition_blas.close();
    if (transition_workspace) (void)runtime::resource_cuda_free(transition_workspace);
    if (arena) (void)runtime::resource_cuda_free(arena);
  }

  cudaError_t authenticate(cudaStream_t requested, bool composable = false) {
    int device = -1;
    cudaStreamCaptureStatus capture{};
    if (requested != stream || cudaGetDevice(&device) != cudaSuccess || device != source.device_id)
      return cudaErrorInvalidResourceHandle;
    auto error = cudaStreamIsCapturing(stream, &capture);
    if (error != cudaSuccess) return error;
    if (capture != cudaStreamCaptureStatusNone &&
        !(composable && capture == cudaStreamCaptureStatusActive))
      return cudaErrorStreamCaptureUnsupported;
    CudaKsResidentFockBinding current;
    std::string detail;
    if (methods::detail::dft_cuda_resident_fock_sources(batch, item, expected, current, detail) !=
            GENERATIVEQC_STATUS_SUCCESS ||
        current != source)
      return cudaErrorInvalidResourceHandle;
    if (fitted &&
        scf::cuda_density_fitting_readonly_source(source.primary->cuda_fitted_source()) != fitted)
      return cudaErrorInvalidResourceHandle;
    if (fitted_range &&
        (scf::cuda_density_fitting_readonly_source(source.fitted_range->cuda_fitted_source()) !=
             fitted_range ||
         scf::cuda_density_fitting_readonly_metric(source.fitted_range->cuda_fitted_source()) !=
             range_metric ||
         scf::cuda_density_fitting_readonly_basis(fitted_range.integrals) != range_basis))
      return cudaErrorInvalidResourceHandle;
    const auto lease = reference.reference();
    if (!lease.authenticate) return cudaErrorInvalidResourceHandle;
    error = lease.authenticate(lease.owner);
    if (error != cudaSuccess || !expected.identity.model.nonlocal_grid) return error;
    const auto vv10_lease = vv10_reference.reference();
    return vv10_lease.authenticate ? vv10_lease.authenticate(vv10_lease.owner)
                                   : cudaErrorInvalidResourceHandle;
  }

  void validate_model() {
    require(cuda_ks_hessian_model_matches(expected, source, &options.grid),
            "physical response requires a matched native Hessian grid and Fock model");
  }

  cudaError_t gate(int* failure_flags = nullptr) {
    int failed[status_count]{};
    auto error = cudaMemcpyAsync(failed, status, sizeof(failed), cudaMemcpyDeviceToHost, stream);
    if (error != cudaSuccess) return error;
    if ((error = cudaStreamSynchronize(stream)) != cudaSuccess) return error;
    if (failure_flags) *failure_flags = failed[0];
    return failed[0] || failed[1] || failed[2] ? cudaErrorInvalidValue : cudaSuccess;
  }

  generativeqc_status prepare_resident(std::string& detail, bool repeated_rhs = false) {
    if (!direct) return GENERATIVEQC_STATUS_SUCCESS;
    const auto base_bytes = scf::cuda_direct_jk_plan_diagnostic(direct).device_bytes;
    auto remaining = std::min(options.max_resident_integral_bytes,
                              options.max_device_bytes - receipt.device_bytes);
    const auto required = scf::cuda_direct_jk_resident_value_bytes(direct);
    if (!required || required > remaining) return GENERATIVEQC_STATUS_SUCCESS;
    const auto optional = [&](generativeqc_status result) -> generativeqc_status {
      if (result != GENERATIVEQC_STATUS_OUT_OF_MEMORY) return result;
      // Optional allocation refusal is not a numerical fallback. Retire only
      // OOM; a real stream/device failure must still fail the response owner.
      const auto pending = cudaGetLastError();
      check(pending == cudaErrorMemoryAllocation ? cudaSuccess : pending);
      check(cudaStreamSynchronize(stream));
      detail.clear();
      return GENERATIVEQC_STATUS_SUCCESS;
    };
    auto result = fitted ? scf::prepare_cuda_direct_jk_range_resident_values(
                               direct, remaining, (1U << 28) - 1, scf::FockOperator::LongRange,
                               source.correction->spec.exchange.omega, detail)
                         : scf::prepare_cuda_direct_jk_resident_values(direct, remaining, detail);
    if (result != GENERATIVEQC_STATUS_SUCCESS) return optional(result);
    auto diagnostic = scf::cuda_direct_jk_plan_diagnostic(direct);
    receipt.resident_integral_bytes = diagnostic.resident_value_bytes;
    receipt.resident_source_evaluations = diagnostic.resident_values_completed;
    receipt.device_bytes = arena_bytes + diagnostic.device_bytes;
    const auto& spec = source.primary->strategy().spec;
    if (repeated_rhs && !fitted && !source.correction && source.spins == 1 &&
        spec.coulomb.present &&
        (!spec.exchange.present || spec.exchange.op == scf::FockOperator::FullRange)) {
      const auto allowance =
          std::min(options.max_resident_integral_bytes - receipt.resident_integral_bytes,
                   options.max_device_bytes - receipt.device_bytes);
      const auto max_products = static_cast<std::size_t>(std::min<std::uint64_t>(
          options.max_direct_eri_evaluations, std::numeric_limits<std::size_t>::max()));
      result = scf::prepare_cuda_direct_jk_response_pairs(direct, spec.exchange.present, allowance,
                                                          max_products, receipt.resident_pair_bytes,
                                                          pair_product_bound, detail);
      if (result != GENERATIVEQC_STATUS_SUCCESS) return optional(result);
      receipt.device_bytes += receipt.resident_pair_bytes;
      receipt.resident_integral_bytes += receipt.resident_pair_bytes;
    }
    if (fitted || !source.correction || !receipt.resident_integral_bytes)
      return GENERATIVEQC_STATUS_SUCCESS;
    remaining = options.max_device_bytes - receipt.device_bytes;
    if (required > options.max_resident_integral_bytes - receipt.resident_integral_bytes ||
        base_bytes > remaining || required > remaining - base_bytes)
      return GENERATIVEQC_STATUS_SUCCESS;
    result = scf::create_cuda_direct_jk_resident_plan_on_stream(
        source.device_id, {source.primary->system()}, 0, remaining, stream, &range_direct,
        diagnostic, detail);
    if (result != GENERATIVEQC_STATUS_SUCCESS) return optional(result);
    // Canonical metadata is optional inside Direct preparation for through-d
    // bases. A recovered allocation refusal may therefore publish a valid
    // replay-only plan that cannot retain values. Keep the original owner.
    if (!scf::cuda_direct_jk_linear_available(range_direct)) {
      scf::destroy_cuda_direct_jk_plan(range_direct);
      range_direct = nullptr;
      detail.clear();
      return GENERATIVEQC_STATUS_SUCCESS;
    }
    const auto allowance =
        std::min(options.max_resident_integral_bytes - receipt.resident_integral_bytes,
                 remaining - diagnostic.device_bytes);
    result = scf::prepare_cuda_direct_jk_range_resident_values(
        range_direct, allowance, (1U << 28) - 1, scf::FockOperator::LongRange,
        source.correction->spec.exchange.omega, detail);
    if (result != GENERATIVEQC_STATUS_SUCCESS) {
      scf::destroy_cuda_direct_jk_plan(range_direct);
      range_direct = nullptr;
      return optional(result);
    }
    diagnostic = scf::cuda_direct_jk_plan_diagnostic(range_direct);
    receipt.resident_integral_bytes += diagnostic.resident_value_bytes;
    receipt.resident_source_evaluations += diagnostic.resident_values_completed;
    receipt.device_bytes += diagnostic.device_bytes;
    return GENERATIVEQC_STATUS_SUCCESS;
  }

  cudaError_t evaluate(const double* density, const double* grid_potential, const double* hcore,
                       bool response, bool grid_action = true) {
    check(cudaMemsetAsync(status, 0, status_count * sizeof(int), stream));
    const auto* beta = source.spins == 2 ? density + matrix : nullptr;
    const auto& spec = source.primary->strategy().spec;
    auto* beta_exchange = source.spins == 2 && spec.exchange.present ? exchange + matrix : nullptr;
    std::string detail;
    if (fitted) {
      if (spec.exchange.present) {
        const auto launch = source.spins == 1 ? maps::launch_rks_fitted : maps::launch_uks_fitted;
        check(launch(shape, seed, density, exchange_coefficient, fitted_panel, primary,
                     map_workspace, map_bytes, status, stream));
      } else {
        const auto launch =
            source.spins == 1 ? maps::launch_rks_fitted_coulomb : maps::launch_uks_fitted_coulomb;
        check(launch(shape, density, fitted_panel, fitted.whitening, primary, map_workspace,
                     map_bytes, status, stream));
      }
    } else {
      if (response && pair_product_bound) {
        // The incumbent reference reconstruction was audited before this
        // optional layout was admitted. No canonical buckets run here.
        check(cudaMemsetAsync(census, 0, 2 * sizeof(*census), stream));
        check(scf::enqueue_cuda_direct_jk_response_pairs(
            direct, density, coulomb, spec.exchange.present ? exchange : nullptr));
      } else {
        const auto result = scf::enqueue_cuda_direct_jk_compensated_device(
            direct, spec, density, matrix, coulomb, spec.exchange.present ? exchange : nullptr,
            compensation, compensation_elements, status + 1, 0, census, detail, true, beta,
            beta_exchange, false);
        if (result != GENERATIVEQC_STATUS_SUCCESS) return cudaErrorNotSupported;
      }
      check(source.spins == 1
                ? maps::launch_rks_exact(shape, coulomb, exchange, exchange_coefficient, primary,
                                         map_workspace, map_bytes, status, stream)
                : maps::launch_uks_exact(shape, coulomb, exchange, exchange_coefficient, primary,
                                         map_workspace, map_bytes, status, stream));
    }
    // Absent exchange panels stay zero for the linear assembly maps. In a
    // pure model no exchange or LR integral action is enqueued.
    if (fitted_range) {
      const auto launch =
          source.spins == 1 ? maps::launch_rks_fitted_range : maps::launch_uks_fitted_range;
      check(launch(range_shape, seed, density, range_whitened, long_range, map_workspace, map_bytes,
                   status, stream));
    } else if (source.correction) {
      const auto result = scf::enqueue_cuda_direct_jk_compensated_device(
          range_direct ? range_direct : direct, source.correction->spec, density, matrix, nullptr,
          long_range, compensation, compensation_elements, status + 2, 0, range_census, detail,
          true, beta, source.spins == 2 ? long_range + matrix : nullptr, false);
      if (result != GENERATIVEQC_STATUS_SUCCESS) return cudaErrorNotSupported;
    }
    check(source.spins == 1 ? maps::launch_rks_assemble(shape, grid_potential, hcore, long_range,
                                                        primary, range_coefficient, candidate,
                                                        map_workspace, map_bytes, status, stream)
                            : maps::launch_uks_assemble(shape, grid_potential, hcore, long_range,
                                                        primary, range_coefficient, candidate,
                                                        map_workspace, map_bytes, status, stream));
    audit<<<maps::blocks(elements), 128, 0, stream>>>(
        candidate, response ? nullptr : source.fock, source.nao, source.spins,
        response ? options.grid.symmetry_absolute_tolerance : options.reference_absolute_tolerance,
        response ? options.grid.symmetry_relative_tolerance : options.reference_relative_tolerance,
        status);
    check(cudaGetLastError());
    if (response) {
      // Preparation bounded squared active-point counts per action. Bound the
      // aggregate for either grid before incrementing a real replay count.
      const auto max_points =
          std::max(grid.work().active_nonlocal_points, vv10_grid.work().active_nonlocal_points);
      const auto pairs = static_cast<std::uint64_t>(max_points) * max_points;
      const auto grid_bound = pairs ? UINT64_MAX / pairs : UINT64_MAX;
      count_action<<<1, 1, 0, stream>>>(census, options.max_direct_eri_evaluations, grid_bound,
                                        pair_product_bound, action_counts, status, grid_action);
      return cudaGetLastError();
    }
    int failure_flags{};
    const auto audited = gate(&failure_flags);
    if (audited == cudaErrorInvalidValue && (failure_flags & 2))
      throw std::runtime_error("physical response Fock audit encountered a nonfinite value");
    if (audited == cudaErrorInvalidValue && (failure_flags & 4))
      throw std::runtime_error("reconstructed Fock disagrees with the accepted SCF Fock");
    check(audited);
    std::uint64_t counts[4]{};
    check(cudaMemcpyAsync(counts, census, sizeof(counts), cudaMemcpyDeviceToHost, stream));
    check(cudaStreamSynchronize(stream));
    require(counts[0] <= options.max_direct_eri_evaluations &&
                counts[2] <= options.max_direct_eri_evaluations - counts[0] &&
                counts[1] <= options.max_direct_eri_evaluations &&
                counts[3] <= options.max_direct_eri_evaluations - counts[1],
            "physical response Direct census exceeded its admitted work bound");
    receipt.reference_quartets = counts[0] + counts[2];
    receipt.reference_eris = counts[1] + counts[3];
    return cudaSuccess;
  }
};

CudaKsWb97mvResponse::CudaKsWb97mvResponse() = default;
CudaKsWb97mvResponse::~CudaKsWb97mvResponse() = default;

generativeqc_status CudaKsWb97mvResponse::prepare(methods::PreparedBatch& batch, std::size_t item,
                                                  const CudaKsFinalStateToken& expected,
                                                  const CudaKsWb97mvResponseOptions& options,
                                                  std::string& detail) {
  impl_.reset();
  detail.clear();
  try {
    require(options.max_device_bytes && options.max_affine_term_visits &&
                options.max_direct_eri_evaluations &&
                std::isfinite(options.reference_absolute_tolerance) &&
                options.reference_absolute_tolerance >= 0 &&
                std::isfinite(options.reference_relative_tolerance) &&
                options.reference_relative_tolerance >= 0,
            "invalid physical response resource or reference gates");
    auto owner = std::make_unique<Impl>(batch, item, expected, options);
    auto result = owner->reference.prepare(detail);
    if (result != GENERATIVEQC_STATUS_SUCCESS) return result;
    result = methods::detail::dft_cuda_resident_fock_sources(batch, item, expected, owner->source,
                                                             detail);
    if (result != GENERATIVEQC_STATUS_SUCCESS) return result;
    owner->validate_model();
    const bool separate = expected.identity.model.nonlocal_grid.has_value();
    if (separate) {
      result = owner->vv10_reference.prepare(detail);
      if (result != GENERATIVEQC_STATUS_SUCCESS) return result;
    }
    result = methods::detail::dft_cuda_resident_orbital_frame(batch, item, expected, owner->frame,
                                                              detail);
    if (result != GENERATIVEQC_STATUS_SUCCESS) return result;
    const auto lease = owner->reference.reference();
    const auto grid_shape = owner->reference.shape();
    owner->stream = lease.stream;
    require(owner->source.device_id == lease.device_id && owner->source.nao == grid_shape.nao &&
                owner->source.spins == grid_shape.spins &&
                owner->source.stream == reinterpret_cast<void*>(lease.stream),
            "physical response grid and Fock sources differ");
    const auto binding = scf::prepared_cuda_fock_binding(*owner->source.primary);
    require(binding && binding.device_id == lease.device_id && binding.stream == lease.stream &&
                binding.nbf == grid_shape.nao,
            "physical response requires a resident primary Fock source");
    if (owner->source.primary->strategy().spec.coulomb.approximation ==
        scf::FockApproximation::DensityFitted) {
      owner->fitted =
          scf::cuda_density_fitting_readonly_source(owner->source.primary->cuda_fitted_source());
      require(owner->fitted && owner->fitted.nao == grid_shape.nao &&
                  owner->fitted.device_id == lease.device_id &&
                  owner->fitted.stream == lease.stream,
              "physical response requires an immutable fitted raw/whitening source");
    }
    owner->shape = {grid_shape.nao, owner->fitted ? owner->fitted.naux : 0, grid_shape.spins};
    if (owner->source.fitted_range) {
      const auto* plan = owner->source.fitted_range->cuda_fitted_source();
      owner->fitted_range = scf::cuda_density_fitting_readonly_source(plan);
      owner->range_metric = scf::cuda_density_fitting_readonly_metric(plan);
      owner->range_basis = scf::cuda_density_fitting_readonly_basis(owner->fitted_range.integrals);
      // The accepted KS stream already joined the LR producer. Replay only
      // immutable values/whitening on this stream, with independent storage.
      // With fixed geometry, both whitened legs use the same accepted W, even
      // when nuclear motion requires a fixed-rank spectral differential.
      require(owner->fitted_range && owner->range_metric && owner->range_basis &&
                  owner->fitted_range.device_id == lease.device_id &&
                  owner->fitted_range.nao == grid_shape.nao &&
                  owner->range_metric.naux == owner->fitted_range.naux &&
                  owner->range_metric.identity == owner->fitted_range.identity &&
                  owner->range_metric.relative_threshold ==
                      owner->source.correction->metric_relative_threshold &&
                  owner->range_basis.range_omega == owner->source.correction->spec.exchange.omega &&
                  owner->fitted_range.integrals != owner->fitted.integrals &&
                  owner->fitted_range.whitening != owner->fitted.whitening,
              "physical response requires a separate, cutoff-stable erf-fitted source");
      owner->range_shape = {grid_shape.nao, owner->fitted_range.naux, grid_shape.spins};
    }
    owner->matrix = product(grid_shape.nao, grid_shape.nao);
    owner->elements = product(grid_shape.spins, owner->matrix);
    const auto cartesian = molecule::cartesian_ao_count(owner->source.primary->system());
    const auto quartic = product(product(cartesian, cartesian), product(cartesian, cartesian));
    const auto direct_actions =
        unsigned(!owner->fitted) + unsigned(owner->source.correction && !owner->fitted_range);
    require(product(direct_actions, quartic) <= options.max_direct_eri_evaluations,
            "physical response Direct ERI work exceeds its explicit bound");
    std::size_t terms{}, matrix_terms{}, bytes{}, total_terms{}, total_matrix_terms{};
    const auto admit = [&](const maps::Extents& extent, auto workspace, auto work) {
      check(workspace(extent, bytes));
      require(work(extent, terms, matrix_terms) &&
                  terms <= options.max_affine_term_visits - total_terms,
              "physical response affine schedule exceeds its explicit term-visit bound");
      require(matrix_terms <= options.max_matrix_terms - total_matrix_terms,
              "physical response tiled matrix schedule exceeds its explicit product bound");
      total_terms += terms;
      total_matrix_terms += matrix_terms;
      owner->map_bytes = std::max(owner->map_bytes, bytes);
    };
    const auto admit_map = [&](auto workspace, auto work) { admit(owner->shape, workspace, work); };
    const bool whiten_primary =
        owner->fitted && owner->source.primary->strategy().spec.exchange.present;
    if (owner->fitted) {
      if (whiten_primary) {
        if (grid_shape.spins == 1)
          admit_map(maps::workspace_rks_fitted, maps::work_rks_fitted);
        else
          admit_map(maps::workspace_uks_fitted, maps::work_uks_fitted);
      } else if (grid_shape.spins == 1) {
        admit_map(maps::workspace_rks_fitted_coulomb, maps::work_rks_fitted_coulomb);
      } else {
        admit_map(maps::workspace_uks_fitted_coulomb, maps::work_uks_fitted_coulomb);
      }
    } else {
      if (grid_shape.spins == 1)
        admit_map(maps::workspace_rks_exact, maps::work_rks_exact);
      else
        admit_map(maps::workspace_uks_exact, maps::work_uks_exact);
    }
    if (grid_shape.spins == 1)
      admit_map(maps::workspace_rks_assemble, maps::work_rks_assemble);
    else
      admit_map(maps::workspace_uks_assemble, maps::work_uks_assemble);
    if (owner->fitted_range) {
      if (grid_shape.spins == 1)
        admit(owner->range_shape, maps::workspace_rks_fitted_range, maps::work_rks_fitted_range);
      else
        admit(owner->range_shape, maps::workspace_uks_fitted_range, maps::work_uks_fitted_range);
    }
    owner->receipt.affine_action_terms = total_terms;
    owner->receipt.matrix_action_terms = total_matrix_terms;
    if (whiten_primary) admit_map(maps::workspace_whiten, maps::work_whiten);
    if (owner->fitted_range) admit(owner->range_shape, maps::workspace_whiten, maps::work_whiten);
    owner->receipt.affine_reference_terms = total_terms;
    owner->receipt.matrix_reference_terms = total_matrix_terms;
    check(cuda_wb97mv_grid_workspace(grid_shape, owner->options.grid, owner->grid_bytes));
    auto vv10_options = owner->options.grid;
    std::size_t vv10_bytes{};
    if (separate) {
      // Both fixed-grid owners share the caller's response-grid cap.
      vv10_options.component = CudaWb97mvGridComponent::vv10_only;
      vv10_options.max_workspace_bytes -= owner->grid_bytes;
      check(cuda_wb97mv_grid_workspace(owner->vv10_reference.shape(), vv10_options, vv10_bytes));
    }
    Layout layout;
    const auto grid_offset = layout.add(owner->grid_bytes / sizeof(double));
    const auto vv10_offset = layout.add(vv10_bytes / sizeof(double));
    const auto map_offset = layout.add(owner->map_bytes / sizeof(double));
    const auto coulomb_offset = layout.add(owner->matrix);
    const auto exchange_offset = layout.add(owner->elements);
    const auto range_offset = layout.add(owner->elements);
    const auto primary_offset = layout.add(owner->elements);
    const auto potential_offset = layout.add(owner->elements);
    const auto candidate_offset = layout.add(owner->elements);
    const auto zero_offset = layout.add(owner->matrix);
    const auto fitted_offset =
        layout.add(owner->fitted ? product(owner->matrix, owner->fitted.naux) : 0);
    const auto range_whitened_offset =
        layout.add(owner->fitted_range ? product(owner->matrix, owner->fitted_range.naux) : 0);
    // Only exchange needs staging for prewhitening. Pure J retains raw values.
    const auto raw_offset =
        layout.add(product(owner->matrix, std::max(whiten_primary ? owner->fitted.naux : 0,
                                                   owner->fitted_range.naux)));
    const auto seed_offset = layout.add(1);
    const auto coefficient_offset = layout.add(1);
    const auto range_coefficient_offset = layout.add(1);
    owner->compensation_elements =
        direct_actions ? product(product(2, grid_shape.spins), product(cartesian, cartesian)) : 0;
    const auto compensation_offset = layout.add(owner->compensation_elements);
    const auto status_offset = layout.add(2);
    const auto census_offset = layout.add(4 + sizeof(ActionCounts) / sizeof(double));
    owner->arena_bytes = layout.bytes;
    require(owner->arena_bytes < options.max_device_bytes,
            "physical response workspace exceeds its explicit device bound");
    scf::CudaDirectJkDiagnostic diagnostic;
    if (direct_actions) {
      result = scf::create_cuda_direct_jk_resident_plan_on_stream(
          lease.device_id, {owner->source.primary->system()}, 0,
          options.max_device_bytes - owner->arena_bytes, lease.stream, &owner->direct, diagnostic,
          detail);
      if (result != GENERATIVEQC_STATUS_SUCCESS) return result;
      require(scf::cuda_direct_jk_linear_available(owner->direct),
              "physical response requires admitted canonical Direct storage");
    }
    owner->receipt.device_bytes = owner->arena_bytes + diagnostic.device_bytes;
    require(owner->receipt.device_bytes <= options.max_device_bytes,
            "physical response total device storage exceeds its explicit bound");
    check(runtime::resource_cuda_malloc(&owner->arena, owner->arena_bytes));
    const auto panel = [&](std::size_t offset) {
      return reinterpret_cast<double*>(static_cast<char*>(owner->arena) + offset);
    };
    owner->grid_workspace = panel(grid_offset);
    owner->map_workspace = panel(map_offset);
    owner->coulomb = panel(coulomb_offset);
    owner->exchange = panel(exchange_offset);
    owner->long_range = panel(range_offset);
    owner->primary = panel(primary_offset);
    owner->potential = panel(potential_offset);
    owner->candidate = panel(candidate_offset);
    owner->zero = panel(zero_offset);
    owner->fitted_panel = panel(fitted_offset);
    owner->range_whitened = panel(range_whitened_offset);
    owner->seed = panel(seed_offset);
    owner->exchange_coefficient = panel(coefficient_offset);
    owner->range_coefficient = panel(range_coefficient_offset);
    owner->compensation = panel(compensation_offset);
    owner->status = reinterpret_cast<int*>(panel(status_offset));
    owner->census = reinterpret_cast<std::uint64_t*>(panel(census_offset));
    owner->range_census = owner->census + 2;
    owner->action_counts = reinterpret_cast<ActionCounts*>(owner->census + 4);
    check(cudaMemsetAsync(owner->arena, 0, owner->arena_bytes, lease.stream));
    const double seed = 1;
    check(cudaMemcpyAsync(owner->seed, &seed, sizeof(seed), cudaMemcpyHostToDevice, lease.stream));
    const auto& exchange = owner->source.primary->strategy().spec.exchange;
    // The fitted energy uses the physical fraction; the exact map adds signed K.
    const double coefficient = !exchange.present ? 0
                               : owner->fitted
                                   ? -exchange.coefficient * (grid_shape.spins == 1 ? 2 : 1)
                                   : exchange.coefficient;
    check(cudaMemcpyAsync(owner->exchange_coefficient, &coefficient, sizeof(double),
                          cudaMemcpyHostToDevice, lease.stream));
    const double signed_range =
        owner->source.correction ? owner->source.correction->spec.exchange.coefficient : 0.0;
    check(cudaMemcpyAsync(owner->range_coefficient, &signed_range, sizeof(double),
                          cudaMemcpyHostToDevice, lease.stream));
    const auto whiten = [&](const scf::CudaDfReadonlySource& source, const maps::Extents& extent,
                            double* output) -> generativeqc_status {
      if (!source) return GENERATIVEQC_STATUS_SUCCESS;
      auto* raw = panel(raw_offset);
      const auto status = scf::enqueue_cuda_density_fitting_raw_values(
          source.integrals, 0, owner->matrix, reinterpret_cast<void*>(lease.stream), raw, detail);
      if (status != GENERATIVEQC_STATUS_SUCCESS) return status;
      check(maps::launch_whiten(extent, raw, source.whitening, output, owner->map_workspace,
                                owner->map_bytes, owner->status, lease.stream));
      check(owner->gate());
      return GENERATIVEQC_STATUS_SUCCESS;
    };
    if (owner->fitted) {
      result = whiten_primary
                   ? whiten(owner->fitted, owner->shape, owner->fitted_panel)
                   : scf::enqueue_cuda_density_fitting_raw_values(
                         owner->fitted.integrals, 0, owner->matrix,
                         reinterpret_cast<void*>(lease.stream), owner->fitted_panel, detail);
      if (result != GENERATIVEQC_STATUS_SUCCESS) return result;
    }
    result = whiten(owner->fitted_range, owner->range_shape, owner->range_whitened);
    if (result != GENERATIVEQC_STATUS_SUCCESS) return result;
    const auto prepare_grid = [](auto& grid, const auto& shape, const auto& policy,
                                 const auto& reference, void* workspace, std::size_t bytes) {
      const auto error = grid.prepare(shape, policy, reference, workspace, bytes);
      if (grid.work().pair_budget_exceeded)
        throw std::invalid_argument(
            "max_nonlocal_pairs exceeded: " + std::to_string(grid.work().active_nonlocal_points) +
            " active VV10 points require their square in ordered pairs "
            "per action (cap " +
            std::to_string(policy.max_pair_evaluations) + ")");
      check(error);
    };
    prepare_grid(owner->grid, grid_shape, owner->options.grid, lease, owner->grid_workspace,
                 owner->grid_bytes);
    if (!options.defer_resident_integrals) {
      result = owner->prepare_resident(detail);
      if (result != GENERATIVEQC_STATUS_SUCCESS) return result;
    }
    const double* reference_potential{};
    check(owner->grid.reference_potential(reference_potential));
    if (separate) {
      prepare_grid(owner->vv10_grid, owner->vv10_reference.shape(), vv10_options,
                   owner->vv10_reference.reference(), panel(vv10_offset), vv10_bytes);
      const double* vv10_potential{};
      check(owner->vv10_grid.reference_potential(vv10_potential));
      sum_grid_potentials<<<maps::blocks(owner->elements), 128, 0, lease.stream>>>(
          reference_potential, vv10_potential, owner->potential, owner->elements);
      check(cudaGetLastError());
      reference_potential = owner->potential;
    }
    check(owner->evaluate(lease.density, reference_potential, owner->source.hcore, false));
    check(owner->authenticate(lease.stream));
    owner->ready = true;
    impl_ = std::move(owner);
    return GENERATIVEQC_STATUS_SUCCESS;
  } catch (const std::bad_alloc&) {
    detail = "native induced Fock device workspace allocation failed";
    return GENERATIVEQC_STATUS_OUT_OF_MEMORY;
  } catch (const std::invalid_argument& error) {
    detail = error.what();
    return GENERATIVEQC_STATUS_INVALID_ARGUMENT;
  } catch (const std::exception& error) {
    detail = error.what();
    return GENERATIVEQC_STATUS_NUMERICAL_FAILURE;
  }
}

cudaError_t CudaKsWb97mvResponse::apply(const double* direction, double* output,
                                        cudaStream_t stream) noexcept {
  return apply_impl(direction, output, stream, false);
}

cudaError_t CudaKsWb97mvResponse::apply_impl(const double* direction, double* output,
                                             cudaStream_t stream, bool private_panel,
                                             bool local_xc) noexcept {
  if (!impl_ || !impl_->ready) return cudaErrorInvalidResourceHandle;
  try {
    auto& owner = *impl_;
    owner.failure_check = "induced Fock action buffer admission";
    auto error = owner.authenticate(stream, private_panel);
    if (error != cudaSuccess) {
      owner.ready = false;
      return error;
    }
    const auto bytes = owner.elements * sizeof(double);
    for (const auto* pointer : {direction, static_cast<const double*>(output)}) {
      cudaPointerAttributes attributes{};
      if (!pointer || reinterpret_cast<std::uintptr_t>(pointer) % alignof(double) ||
          cudaPointerGetAttributes(&attributes, pointer) != cudaSuccess ||
          attributes.type != cudaMemoryTypeDevice || attributes.device != owner.source.device_id ||
          overlap(pointer, bytes, owner.arena, owner.arena_bytes))
        return cudaErrorInvalidValue;
    }
    if (overlap(direction, bytes, output, bytes) ||
        overlap(output, bytes, owner.source.hcore, owner.matrix * sizeof(double)) ||
        overlap(output, bytes, owner.source.fock, bytes) ||
        overlap(output, bytes, owner.frame.coefficients, bytes) ||
        overlap(output, bytes, owner.frame.orbital_energies,
                owner.source.spins * owner.source.nao * sizeof(double)) ||
        overlap(output, bytes, owner.frame.density, bytes) ||
        overlap(output, bytes, owner.frame.overlap, owner.matrix * sizeof(double)))
      return cudaErrorInvalidValue;
    const auto grid_lease = owner.reference.reference();
    const auto grid_shape = owner.reference.shape();
    const auto basis_elements =
        3 * grid_shape.atoms + 2 * grid_shape.primitives + 16 * grid_shape.nao;
    if (overlap(output, bytes, grid_lease.basis, basis_elements * sizeof(double)) ||
        overlap(output, bytes, grid_lease.points, 3 * grid_shape.points * sizeof(double)) ||
        overlap(output, bytes, grid_lease.weights, grid_shape.points * sizeof(double)))
      return cudaErrorInvalidValue;
    for (const auto& fitted : {owner.fitted, owner.fitted_range})
      if (fitted &&
          overlap(output, bytes, fitted.whitening, fitted.naux * fitted.naux * sizeof(double)))
        return cudaErrorInvalidValue;
    if (owner.expected.identity.model.nonlocal_grid) {
      const auto vv10_lease = owner.vv10_reference.reference();
      const auto vv10_shape = owner.vv10_reference.shape();
      const auto vv10_basis =
          3 * vv10_shape.atoms + 2 * vv10_shape.primitives + 16 * vv10_shape.nao;
      if (overlap(output, bytes, vv10_lease.basis, vv10_basis * sizeof(double)) ||
          overlap(output, bytes, vv10_lease.points, 3 * vv10_shape.points * sizeof(double)) ||
          overlap(output, bytes, vv10_lease.weights, vv10_shape.points * sizeof(double)))
        return cudaErrorInvalidValue;
    }
    struct DrainOnFailure {
      cudaStream_t stream;
      bool active{true};
      ~DrainOnFailure() {
        // Later launch/API failure must not return while queued work still
        // borrows the caller's direction or output storage.
        if (active) (void)cudaStreamSynchronize(stream);
      }
    } drain{stream, !private_panel};
    owner.used = true;
    owner.failure_check = "induced Fock XC response";
    if (local_xc) check(owner.grid.enqueue_private_response(direction, owner.potential, stream));
    if (owner.expected.identity.model.nonlocal_grid) {
      check(owner.vv10_grid.enqueue_private_response(direction, owner.candidate, stream));
      sum_grid_potentials<<<maps::blocks(owner.elements), 128, 0, stream>>>(
          owner.potential, owner.candidate, owner.potential, owner.elements);
      check(cudaGetLastError());
    }
    owner.failure_check = "induced Fock assembly and finite symmetric audit";
    check(owner.evaluate(direction, local_xc ? owner.potential : owner.zero, owner.zero, true,
                         local_xc));
    if (!private_panel) check(owner.gate());
    error = owner.authenticate(stream, private_panel);
    if (error != cudaSuccess) {
      owner.ready = false;
      return error;
    }
    if (private_panel) {
      publish_private_fock<<<maps::blocks(owner.elements), 128, 0, stream>>>(
          owner.candidate, output, owner.elements, owner.status);
      return cudaGetLastError();
    }
    check(collect_work());
    check(cudaMemcpyAsync(output, owner.candidate, bytes, cudaMemcpyDeviceToDevice, stream));
    check(cudaStreamSynchronize(stream));
    drain.active = false;
    owner.failure_check = nullptr;
    return cudaSuccess;
  } catch (...) {
    return cudaErrorInvalidValue;
  }
}

const char* CudaKsWb97mvResponse::failure_check() const noexcept {
  return impl_ ? impl_->failure_check : nullptr;
}

generativeqc_status CudaKsWb97mvResponse::retain_integrals(std::size_t maximum_additional_bytes,
                                                           bool repeated_rhs, std::string& detail) {
  if (!impl_ || !impl_->ready || impl_->used || impl_->receipt.resident_integral_bytes) {
    detail = "optional integral admission requires an unused prepared response owner";
    return GENERATIVEQC_STATUS_INVALID_ARGUMENT;
  }
  auto& owner = *impl_;
  try {
    check(owner.authenticate(owner.stream));
    const auto remaining = owner.options.max_device_bytes - owner.receipt.device_bytes;
    owner.options.max_device_bytes =
        owner.receipt.device_bytes + std::min(remaining, maximum_additional_bytes);
    const auto result = owner.prepare_resident(detail, repeated_rhs);
    if (result != GENERATIVEQC_STATUS_SUCCESS) {
      owner.ready = false;
      return result;
    }
    check(owner.authenticate(owner.stream));
    detail.clear();
    return GENERATIVEQC_STATUS_SUCCESS;
  } catch (const std::exception& error) {
    owner.ready = false;
    detail = error.what();
    return GENERATIVEQC_STATUS_NUMERICAL_FAILURE;
  }
}

cudaError_t CudaKsWb97mvResponse::retain_fixed_grid_points(std::size_t max_bytes) noexcept {
  if (!impl_ || !impl_->ready || impl_->used) return cudaErrorInvalidResourceHandle;
  auto& owner = *impl_;
  auto error = owner.authenticate(owner.stream);
  if (error != cudaSuccess) return error;
  const auto before = owner.grid.work().workspace_bytes;
  error = owner.grid.retain_fixed_point_linearization(
      std::min(max_bytes, owner.options.max_device_bytes - owner.receipt.device_bytes));
  if (error != cudaSuccess) return error;
  owner.receipt.device_bytes += owner.grid.work().workspace_bytes - before;
  return cudaSuccess;
}

cudaError_t CudaKsWb97mvResponse::retain_transition_builder(const response::CudaCpksShape& shape,
                                                            std::size_t max_bytes) noexcept {
  if (!impl_ || !impl_->ready || impl_->used) return cudaErrorInvalidResourceHandle;
  auto& owner = *impl_;
  const auto axes = owner.grid.transition_axes();
  const auto points = owner.reference.shape().points;
  const auto occupied = shape.occupied[0];
  if (owner.transition_workspace || shape.spins != 1 || owner.source.spins != 1 ||
      shape.nao != owner.source.nao || shape.nao < 32 || shape.nao > 96 || !occupied ||
      occupied >= shape.nao || points < 4096 || (axes != 4 && axes != 5) ||
      (owner.fitted && axes != 5) ||
      owner.source.correction || owner.expected.identity.model.nonlocal_grid)
    return cudaSuccess;
  const auto rotations = occupied * (shape.nao - occupied);
  const auto products_per_point = axes * rotations * rotations;
  const auto limit = axes == 5 && shape.nao > 64 ? wide_r2scan_transition_product_limit
                                                : transition_product_limit;
  if (points > limit / products_per_point) return cudaSuccess;
  const auto bytes =
      (2 * axes * rotations + 4 * shape.nao) * transition_tile_points * sizeof(double) +
      sizeof(double);
  if (bytes > std::min(max_bytes, owner.options.max_device_bytes - owner.receipt.device_bytes))
    return cudaSuccess;
  auto error = owner.authenticate(owner.stream);
  if (error != cudaSuccess) return error;
  error = runtime::resource_cuda_malloc(&owner.transition_workspace, bytes);
  if (error == cudaErrorMemoryAllocation) {
    (void)cudaGetLastError();
    return cudaSuccess;
  }
  if (error != cudaSuccess) return error;
  auto library = owner.transition_blas.create();
  if (library == CUBLAS_STATUS_SUCCESS) library = owner.transition_blas.bind(owner.stream);
  if (library != CUBLAS_STATUS_SUCCESS) {
    (void)owner.transition_blas.close();
    return runtime::resource_cuda_free(std::exchange(owner.transition_workspace, nullptr));
  }
  owner.transition_occupied = occupied;
  owner.transition_products = products_per_point * points;
  owner.receipt.device_bytes += bytes;
  return cudaSuccess;
}

cudaError_t CudaKsWb97mvResponse::seed_local_xc(void* context, const response::CudaCpksShape& shape,
                                                const response::CudaCpksReference& reference,
                                                double* matrix, cudaStream_t stream) noexcept {
  if (!context || !matrix) return cudaErrorInvalidValue;
  auto& self = *static_cast<CudaKsWb97mvResponse*>(context);
  if (!self.impl_ || !self.impl_->ready) return cudaErrorInvalidResourceHandle;
  auto& owner = *self.impl_;
  if (!owner.transition_workspace || shape.nao != owner.source.nao || shape.spins != 1 ||
      shape.occupied[0] != owner.transition_occupied ||
      reference.coefficient_layout != response::CudaCpksLayout::column_major ||
      reference.coefficients != owner.frame.coefficients || reference.stream != owner.stream)
    return cudaErrorNotSupported;
  auto error = owner.authenticate(stream);
  if (error != cudaSuccess) return error;
  const auto axes = owner.grid.transition_axes();
  const auto points = owner.reference.shape().points;
  const auto rotations = owner.transition_occupied * (shape.nao - owner.transition_occupied);
  const auto panel = axes * rotations * transition_tile_points;
  auto* fields = static_cast<double*>(owner.transition_workspace);
  auto* tangents = fields + panel;
  auto* jets = tangents + panel;
  auto* refused = reinterpret_cast<int*>(jets + 4 * shape.nao * transition_tile_points);
  static_assert(2 * sizeof(int) <= sizeof(double));
  auto* correction_count = refused + 1;
  error = cudaMemsetAsync(refused, 0, 2 * sizeof(int), stream);
  if (error != cudaSuccess) return error;
  error = owner.grid.enqueue_transition_census(correction_count, stream);
  int correction_points{};
  if (error == cudaSuccess)
    error = cudaMemcpyAsync(&correction_points, correction_count, sizeof(int),
                            cudaMemcpyDeviceToHost, stream);
  const auto counted = cudaStreamSynchronize(stream);
  if (counted != cudaSuccess) return counted;
  if (error != cudaSuccess) return error;
  if (correction_points < 0 ||
      static_cast<std::uint64_t>(correction_points) > transition_correction_limit / rotations)
    return cudaErrorNotSupported;
  std::uint64_t products{};
  // A unit RKS rotation yields 4 times the bilinear feature (occupation
  // two and two mirrored density legs). The OV projection reads one leg.
  constexpr double occupation_and_lift = 4.0, zero = 0.0, one = 1.0;
  for (std::size_t begin = 0; begin < points && error == cudaSuccess;) {
    const auto count = std::min(points - begin, transition_tile_points);
    error = owner.grid.enqueue_transition_tile(reference.coefficients, owner.transition_occupied,
                                               begin, count, jets, fields, tangents,
                                               correction_points != 0, refused, stream);
    if (error != cudaSuccess) break;
    const auto rows = static_cast<int>(axes * count);
    const auto columns = static_cast<int>(rotations);
    const auto result = owner.transition_blas.product(true, false, columns, columns, rows,
                                                      &occupation_and_lift, fields, rows, tangents,
                                                      rows, begin ? &one : &zero, matrix, columns);
    if (result != CUBLAS_STATUS_SUCCESS) {
      error = cudaErrorNotSupported;
      break;
    }
    products += rotations * rotations * rows;
    begin += count;
  }
  int refused_host{};
  if (error == cudaSuccess)
    error = cudaMemcpyAsync(&refused_host, refused, sizeof(int), cudaMemcpyDeviceToHost, stream);
  const auto settled = cudaStreamSynchronize(stream);
  owner.grid.record_transition_projection(products, correction_points * rotations);
  if (settled != cudaSuccess) return settled;
  if (error != cudaSuccess) return error;
  error = owner.authenticate(stream);
  if (error != cudaSuccess) return error;
  if (refused_host || products != owner.transition_products) return cudaErrorNotSupported;
  return cudaSuccess;
}

cudaError_t CudaKsWb97mvResponse::collect_work() noexcept {
  if (!impl_ || !impl_->ready) return cudaErrorInvalidResourceHandle;
  auto& owner = *impl_;
  ActionCounts counts{};
  const auto copied = cudaMemcpyAsync(&counts, owner.action_counts, sizeof(counts),
                                      cudaMemcpyDeviceToHost, owner.stream);
  const auto completed = cudaStreamSynchronize(owner.stream);
  if (copied != cudaSuccess) return copied;
  if (completed != cudaSuccess) return completed;
  owner.receipt.response_quartets = counts.quartets;
  owner.receipt.response_eris = counts.eris;
  owner.receipt.response_pair_product_bound = counts.pair_product_bound;
  owner.receipt.actions = counts.actions;
  owner.grid_actions = counts.grid_actions;
  owner.failure_check = nullptr;
  return cudaSuccess;
}

CudaKsWb97mvResponseWork CudaKsWb97mvResponse::work() const noexcept {
  if (!impl_) return {};
  auto receipt = impl_->receipt;
  receipt.grid = impl_->grid.work();
  receipt.separate_vv10 = impl_->vv10_grid.work();
  auto grid_receipt = [&](CudaWb97mvGridWork& grid) {
    grid.response_actions = impl_->grid_actions;
    const auto points = static_cast<std::uint64_t>(grid.active_nonlocal_points);
    grid.response_pairs = (points * points) * grid.response_actions;
  };
  grid_receipt(receipt.grid);
  if (impl_->expected.identity.model.nonlocal_grid) grid_receipt(receipt.separate_vv10);
  return receipt;
}

cudaError_t CudaKsWb97mvResponse::induced_fock(void* context, const double* direction,
                                               double* output, const response::CudaCpksShape& shape,
                                               cudaStream_t stream) noexcept {
  if (!context) return cudaErrorInvalidValue;
  auto& self = *static_cast<CudaKsWb97mvResponse*>(context);
  if (!self.impl_ || shape.nao != self.impl_->source.nao || shape.spins != self.impl_->source.spins)
    return cudaErrorInvalidValue;
  return self.apply_impl(direction, output, stream, true);
}

cudaError_t CudaKsWb97mvResponse::induced_nonlocal_fock(void* context, const double* direction,
                                                        double* output,
                                                        const response::CudaCpksShape& shape,
                                                        cudaStream_t stream) noexcept {
  if (!context) return cudaErrorInvalidValue;
  auto& self = *static_cast<CudaKsWb97mvResponse*>(context);
  if (!self.impl_ || !self.impl_->transition_workspace || shape.nao != self.impl_->source.nao ||
      shape.spins != 1 || shape.occupied[0] != self.impl_->transition_occupied)
    return cudaErrorInvalidValue;
  return self.apply_impl(direction, output, stream, true, false);
}

}  // namespace generativeqc::dft
