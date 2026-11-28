#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <utility>

#include "dft/cuda_hessian_semilocal_model.hpp"
#include "dft/cuda_ks_grid_reference.hpp"
#include "dft/cuda_ks_wb97mv_model.hpp"
#include "dft/cuda_ks_wb97mv_response.hpp"
#include "dft/cuda_xc_transition_integer.cuh"
#include "dft/r2scan_rvv10_model.hpp"
#include "dft/semilocal_family.hpp"
#include "dft/split_hybrid_cuda_selection.hpp"
#include "generated_split_hybrid_registry.cuh"
#include "integrals/coulomb_derivative_envelope.hpp"
#include "molecule/basis.hpp"
#include "r2scan_cuda_domain.hpp"
#include "response/wb97mv_fock_response.cuh"
#include "runtime/cuda_component_trace.hpp"
#include "runtime/resource_cuda.cuh"
#include "scf/cuda_density_fitting_basis.hpp"
#include "scf/cuda_density_fitting_device.hpp"
#include "scf/cuda_direct_jk_device.hpp"
#include "scf/cuda_fock_execution.hpp"
#include "tensor/cuda_contraction.cuh"
#include "tensor/cuda_dense_operations.hpp"

namespace generativeqc::dft {
namespace {
/** Each s/p shell component has the same Cartesian normalization. Divide
 * the shell-pair sum bound by its ordered component count, rounding upward.
 * Fill both triangles in the public Cartesian AO order (shell-major). */
std::vector<double> certified_coulomb_norms(const core::System& system, std::size_t nao) {
  using Envelope = integrals::CoulombDerivativeEnvelope;
  if (system.basis_representation != core::kCartesianBasis || nao > 256) return {};
  std::vector<std::size_t> offsets{0};
  for (const auto& shell : system.shells) {
    if (shell.angular_momentum > 1 || shell.primitives.size() > 16 ||
        shell.atom_index >= system.atoms.size())
      return {};
    offsets.push_back(offsets.back() + 2 * shell.angular_momentum + 1);
  }
  if (offsets.back() != nao) return {};
  std::vector<double> norms(nao * nao);
  for (std::size_t left = 0; left < system.shells.size(); ++left)
    for (std::size_t right = 0; right <= left; ++right) {
      const auto& bra = system.shells[left];
      const auto& ket = system.shells[right];
      const auto bound = Envelope::pair(bra, ket, system.atoms[bra.atom_index].position,
                                        system.atoms[ket.atom_index].position);
      if (!bound || !std::isfinite((*bound)[0])) return {};
      const auto components =
          (offsets[left + 1] - offsets[left]) * (offsets[right + 1] - offsets[right]);
      const auto norm = Envelope::upper((*bound)[0] / components);
      for (auto mu = offsets[left]; mu < offsets[left + 1]; ++mu)
        for (auto nu = offsets[right]; nu < offsets[right + 1]; ++nu)
          norms[mu * nao + nu] = norms[nu * nao + mu] = norm;
    }
  return norms;
}

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
constexpr std::uint64_t complete_fitted_product_limit = 40ULL << 40;
constexpr std::uint64_t transition_correction_limit = 2ULL << 20;
constexpr std::size_t direct_transform_tile = 64;
constexpr std::size_t direct_occupied_tile = 16;
constexpr std::size_t metric_projection_max_ao = 256;
constexpr std::uint64_t direct_transform_product_limit = 1ULL << 40;
constexpr std::uint64_t direct_transform_gather_limit = 1ULL << 35;

__device__ bool failed_action(const int* status) { return status[0] || status[1] || status[2]; }

struct MetricProjection {
  const double* core{};
  double *rhs{}, *rows{}, *jets{}, *projected{}, *fields{};
  int *offsets{}, *refused{};
  std::size_t coordinates{}, rotations{};
  bool ready{};
};

__global__ void pack_metric_rows(std::size_t nao, std::size_t coordinates, const double* core,
                                 const int* offsets, double* rows, int* refused) {
  const auto matrix = nao * nao;
  for (auto index = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x;
       index < coordinates * matrix; index += std::size_t(gridDim.x) * blockDim.x) {
    const auto column = index / matrix, atom = column / 3;
    const auto row = index / nao % nao, ao = index % nao;
    const auto* overlap = core + (2 * column + 1) * matrix;
    const auto value = overlap[row * nao + ao];
    const auto begin = static_cast<std::size_t>(offsets[atom]);
    const auto end = static_cast<std::size_t>(offsets[atom + 1]);
    const auto owned_row = row >= begin && row < end;
    const auto owned_ao = ao >= begin && ao < end;
    if (!(isfinite(value) && ((owned_row || owned_ao) || value == 0))) atomicExch(refused, 1);
    // The diagonal owner block occurs in both E U and U^T E^T.
    if (owned_row)
      rows[ao * (3 * nao) + (column % 3) * nao + row] =
          owned_ao ? 0.25 * (value + overlap[ao * nao + row])
                   : 0.5 * (value + overlap[ao * nao + row]);
  }
}

__global__ void audit_metric_projection(std::size_t size, const double* rhs, int* refused) {
  for (auto index = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x; index < size;
       index += std::size_t(gridDim.x) * blockDim.x)
    if (!isfinite(rhs[index])) atomicExch(refused, 1);
}

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

__global__ void symmetric_coulomb_panel(const double* input, double* output, std::size_t nao) {
  for (std::size_t index = blockIdx.x * blockDim.x + threadIdx.x; index < nao * nao;
       index += static_cast<std::size_t>(gridDim.x) * blockDim.x) {
    const auto transpose = (index % nao) * nao + index / nao;
    output[index] = 0.5 * (input[index] + input[transpose]);
  }
}

__global__ void gather_fitted_slice(const double* raw, double* slice, std::size_t pairs,
                                    std::size_t auxiliary, std::size_t naux) {
  for (std::size_t index = blockIdx.x * blockDim.x + threadIdx.x; index < pairs;
       index += static_cast<std::size_t>(gridDim.x) * blockDim.x)
    slice[index] = raw[index * naux + auxiliary];
}

/** Expand one symmetric AO matrix per tile entry. The input has one
 * contiguous packed AO triangle per slice (an AO pair or an OV rotation). */
__global__ void unpack_coulomb_slices(const double* input, double* output, std::size_t stride,
                                      std::size_t nao, std::size_t elements) {
  const auto plane = nao * nao;
  for (auto index = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x; index < elements;
       index += std::size_t(gridDim.x) * blockDim.x) {
    const auto slice = index / plane;
    const auto row = index % nao, column = (index % plane) / nao;
    const auto high = row > column ? row : column, low = row > column ? column : row;
    output[index] = input[slice * stride + high * (high + 1) / 2 + low];
  }
}

__global__ void repeat_orbital_frame(const double* frame, double* copies, std::size_t plane,
                                     std::size_t elements) {
  for (auto index = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x; index < elements;
       index += std::size_t(gridDim.x) * blockDim.x)
    copies[index] = frame[index % plane];
}

/** The first half-transform produces [pair][rotation]. Transpose in padded
 * tiles so the second half can read contiguous triangles for each rotation. */
__global__ void transpose_coulomb_window(const double* input, double* output, std::size_t pairs,
                                         std::size_t rotations, std::size_t count) {
  __shared__ double tile[32][33];
  const auto rotation_base = std::size_t(blockIdx.x) * 32;
  const auto pair_base = std::size_t(blockIdx.y) * 32;
  const auto lane = threadIdx.x, strip = threadIdx.y;
  for (unsigned offset = strip; offset < 32; offset += blockDim.y) {
    const auto rotation = rotation_base + lane, pair = pair_base + offset;
    if (rotation < rotations && pair < count)
      tile[offset][lane] = input[pair * rotations + rotation];
  }
  __syncthreads();
  for (unsigned offset = strip; offset < 32; offset += blockDim.y) {
    const auto rotation = rotation_base + offset, pair = pair_base + lane;
    if (rotation < rotations && pair < count) output[rotation * pairs + pair] = tile[lane][offset];
  }
}

/** The projection is B^T (w E_xc'' B) on the cached linearization. Publish
 * the omitted triangle once; the physical residual still uses the unprojected
 * action. Never use this for rows that needed directional correction. */
__global__ void mirror_transition_triangle(double* matrix, std::size_t order) {
  for (auto index = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x; index < order * order;
       index += std::size_t(gridDim.x) * blockDim.x) {
    const auto row = index % order, column = index / order;
    if (row > column) matrix[row * order + column] = matrix[index];
  }
}

/** Submit disjoint rectangular products below the diagonal and full diagonal
 * tiles. Recursive bisection uses large rectangles instead of launching one
 * GEMM for every pair of small tiles. Count actual scalar products submitted. */
cublasStatus_t transition_triangle(const tensor::CudaDenseOperations& blas, const double* fields,
                                   const double* tangents, double* matrix, int leading, int rank,
                                   int first, int count, const double* alpha, const double* beta,
                                   std::uint64_t& products) {
  const auto submit = [&](int row, int column, int height, int width) {
    const auto error =
        blas.product(true, false, height, width, rank, alpha, fields + std::size_t(row) * rank,
                     rank, tangents + std::size_t(column) * rank, rank, beta,
                     matrix + std::size_t(column) * leading + row, leading);
    if (error == CUBLAS_STATUS_SUCCESS) products += std::uint64_t(height) * width * rank;
    return error;
  };
  constexpr int diagonal_tile = 512;
  if (count <= diagonal_tile) return submit(first, first, count, count);
  const auto left = ((count / 2 + diagonal_tile / 2) / diagonal_tile) * diagonal_tile;
  const auto right = count - left;
  auto error = submit(first + left, first, right, left);
  if (error == CUBLAS_STATUS_SUCCESS)
    error = transition_triangle(blas, fields, tangents, matrix, leading, rank, first, left, alpha,
                                beta, products);
  if (error == CUBLAS_STATUS_SUCCESS)
    error = transition_triangle(blas, fields, tangents, matrix, leading, rank, first + left, right,
                                alpha, beta, products);
  return error;
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
  std::size_t arena_bytes{}, grid_provider_bytes{}, matrix{}, elements{}, map_bytes{},
      compensation_elements{};
  void *map_workspace{}, *grid_workspace{};
  std::size_t grid_bytes{};
  tensor::CudaDenseOperations transition_blas;
  cublasHandle_t integer_blas{};
  tensor::CudaContractionContext fitted_coulomb_blas;
  double* fitted_coulomb_aux{};
  void* transition_workspace{};
  MetricProjection metric_projection;
  transition_integer::Workspace integer_projection;
  std::size_t transition_occupied{}, transition_tile{};
  std::uint64_t transition_products{};
  double *coulomb{}, *exchange{}, *long_range{}, *primary{}, *potential{}, *candidate{}, *zero{},
      *fitted_panel{}, *range_whitened{}, *seed{}, *exchange_coefficient{}, *range_coefficient{},
      *compensation{};
  int* status{};
  std::uint64_t *census{}, *range_census{};
  ActionCounts* action_counts{};
  std::uint64_t grid_actions{};
  std::size_t pair_product_bound{};
  bool ready{}, used{}, reference_audit_pending{};
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
    if (integer_blas) (void)cublasDestroy(integer_blas);
    if (integer_projection.product) (void)runtime::resource_cuda_free(integer_projection.product);
    if (metric_projection.rhs) (void)runtime::resource_cuda_free(metric_projection.rhs);
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

  generativeqc_status prepare_resident(std::string& detail, bool repeated_rhs = false,
                                       bool dense_only = false) {
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
    const auto& spec = source.primary->strategy().spec;
    std::vector<double> norms;
    if (dense_only && repeated_rhs && options.grid.wide_cartesian_grid && source.nao > 96 &&
        !fitted && !source.correction && source.spins == 1 && spec.coulomb.present &&
        !spec.exchange.present) {
      // Even all nAO² omitted terms would contribute at most
      // 2^-54 ||D||_max for nAO <= 256. This induced-norm budget applies
      // to signed response panels as well as positive reference densities.
      try {
        norms = certified_coulomb_norms(source.primary->system(), source.nao);
      } catch (const std::bad_alloc&) {
        // This metadata is optional; the identical slot layout remains exact.
      }
    }
    auto result =
        fitted ? scf::prepare_cuda_direct_jk_range_resident_values(
                     direct, remaining, (1U << 28) - 1, scf::FockOperator::LongRange,
                     source.correction->spec.exchange.omega, detail)
               : scf::prepare_cuda_direct_jk_resident_values(direct, remaining, detail, norms);
    if (result != GENERATIVEQC_STATUS_SUCCESS) return optional(result);
    auto diagnostic = scf::cuda_direct_jk_plan_diagnostic(direct);
    receipt.resident_integral_bytes = diagnostic.resident_value_bytes;
    receipt.resident_source_evaluations = diagnostic.resident_values_completed;
    receipt.device_bytes = arena_bytes + grid_provider_bytes + diagnostic.device_bytes;
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
      if (pair_product_bound && !spec.exchange.present && source.nao > 96) {
        // Only this private response plan uses the rows. The enclosing
        // transaction audits the reconstruction before admitting an action.
        // Both projected CPKS and its independent physical fallback consume
        // them. Neither the SCF plan nor the displaced nuclear provider does.
        result = scf::retire_cuda_direct_jk_response_values(direct, detail);
        if (result != GENERATIVEQC_STATUS_SUCCESS) return result;
      }
      diagnostic = scf::cuda_direct_jk_plan_diagnostic(direct);
      receipt.device_bytes = arena_bytes + grid_provider_bytes + diagnostic.device_bytes;
      receipt.resident_integral_bytes =
          diagnostic.resident_value_bytes + receipt.resident_pair_bytes;
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

  cudaError_t project_fitted_coulomb(const double* density) {
    const auto pairs = static_cast<int>(matrix);
    const auto aux = static_cast<int>(fitted.naux);
    auto* first = fitted_coulomb_aux;
    auto* second = first + aux;
    // Raw B is row-major [pair,aux], while the accepted W is [out_aux,source_aux].
    // BLAS sees their transposes. Apply B W^T W B^T without replacing the SCF
    // metric or duplicating the three-center tensor. Preserve both symmetric
    // projections of the generated RKS map, including for an arbitrary input.
    symmetric_coulomb_panel<<<maps::blocks(matrix), 128, 0, stream>>>(density, coulomb, shape.nao);
    auto error = cudaGetLastError();
    if (error != cudaSuccess) return error;
    constexpr double one = 1.0, zero = 0.0;
    const auto multiply = [&](bool transpose, int rows, int reduction, const double* left,
                              const double* right, double* result) {
      return generativeqc_tensor::submit_column_gemm({fitted_coulomb_blas.handle(), transpose,
                                                      false, rows, 1, reduction, &one, left, aux,
                                                      right, reduction, &zero, result, rows});
    };
    if (multiply(false, aux, pairs, fitted_panel, coulomb, first) != CUBLAS_STATUS_SUCCESS ||
        multiply(true, aux, aux, fitted.whitening, first, second) != CUBLAS_STATUS_SUCCESS ||
        multiply(false, aux, aux, fitted.whitening, second, first) != CUBLAS_STATUS_SUCCESS ||
        multiply(true, pairs, aux, fitted_panel, first, coulomb) != CUBLAS_STATUS_SUCCESS)
      return cudaErrorNotSupported;
    symmetric_coulomb_panel<<<maps::blocks(matrix), 128, 0, stream>>>(coulomb, primary, shape.nao);
    return cudaGetLastError();
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
      } else if (fitted_coulomb_blas.handle()) {
        failure_check = "resident fitted Coulomb response contraction";
        check(project_fitted_coulomb(density));
        failure_check = "induced Fock assembly and finite symmetric audit";
      } else {
        const auto launch =
            source.spins == 1 ? maps::launch_rks_fitted_coulomb : maps::launch_uks_fitted_coulomb;
        check(launch(shape, density, fitted_panel, fitted.whitening, primary, map_workspace,
                     map_bytes, status, stream));
      }
    } else {
      if (pair_product_bound && (response || reference_audit_pending)) {
        // The authenticated SCF Fock is still the reference oracle; the
        // resident rows avoid another traversal of the canonical quartets.
        check(cudaMemsetAsync(census, 0, 2 * sizeof(*census), stream));
        check(scf::enqueue_cuda_direct_jk_response_pairs(
            direct, density, coulomb, spec.exchange.present ? exchange : nullptr));
        if (!response) receipt.reference_pair_products = pair_product_bound;
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
    const bool try_fitted_coulomb_blas = owner->options.grid.wide_cartesian_grid && owner->fitted &&
                                         !whiten_primary && grid_shape.spins == 1 &&
                                         !owner->source.correction &&
                                         owner->matrix <= std::numeric_limits<int>::max() &&
                                         owner->fitted.naux <= std::numeric_limits<int>::max();
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
    const auto fitted_coulomb_aux_offset =
        layout.add(try_fitted_coulomb_blas ? product(2, owner->fitted.naux) : 0);
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
    if (owner->options.grid.wide_cartesian_grid)
      owner->grid_provider_bytes = CudaWb97mvGridResponse::wide_grid_provider_allowance();
    require(owner->arena_bytes < options.max_device_bytes &&
                owner->grid_provider_bytes < options.max_device_bytes - owner->arena_bytes,
            "physical response workspace exceeds its explicit device bound");
    const auto base_bytes = owner->arena_bytes + owner->grid_provider_bytes;
    scf::CudaDirectJkDiagnostic diagnostic;
    if (direct_actions) {
      result = scf::create_cuda_direct_jk_resident_plan_on_stream(
          lease.device_id, {owner->source.primary->system()}, 0,
          options.max_device_bytes - base_bytes, lease.stream, &owner->direct, diagnostic, detail);
      if (result != GENERATIVEQC_STATUS_SUCCESS) return result;
      require(scf::cuda_direct_jk_linear_available(owner->direct),
              "physical response requires admitted canonical Direct storage");
    }
    owner->receipt.device_bytes = base_bytes + diagnostic.device_bytes;
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
    owner->fitted_coulomb_aux = panel(fitted_coulomb_aux_offset);
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
    constexpr auto fitted_provider_bytes = tensor::CudaContractionContext::kProviderAllowance;
    // A conditional graph cannot own implicit BLAS allocation nodes. Prepare
    // the provider's bounded workspace before any CPKS action is captured.
    if (try_fitted_coulomb_blas &&
        fitted_provider_bytes <= options.max_device_bytes - owner->receipt.device_bytes &&
        owner->fitted_coulomb_blas.prepare(owner->stream))
      owner->receipt.device_bytes += fitted_provider_bytes;
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
    const auto& spec = owner->source.primary->strategy().spec;
    owner->reference_audit_pending =
        options.audit_after_resident_admission && options.defer_resident_integrals &&
        options.grid.wide_cartesian_grid && !separate && !owner->fitted &&
        !owner->source.correction && grid_shape.spins == 1 && spec.coulomb.present &&
        !spec.exchange.present;
    if (!owner->reference_audit_pending)
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
  if (!impl_ || !impl_->ready || impl_->reference_audit_pending)
    return cudaErrorInvalidResourceHandle;
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
    runtime::cuda_trace::TraceRegion xc_trace("hessian_cpks_physical_grid", stream);
    if (local_xc) check(owner.grid.enqueue_private_response(direction, owner.potential, stream));
    if (owner.expected.identity.model.nonlocal_grid) {
      check(owner.vv10_grid.enqueue_private_response(direction, owner.candidate, stream));
      sum_grid_potentials<<<maps::blocks(owner.elements), 128, 0, stream>>>(
          owner.potential, owner.candidate, owner.potential, owner.elements);
      check(cudaGetLastError());
    }
    xc_trace.finish();
    owner.failure_check = "induced Fock assembly and finite symmetric audit";
    check(runtime::cuda_trace::trace_call("hessian_cpks_physical_jk", stream, [&] {
      return owner.evaluate(direction, local_xc ? owner.potential : owner.zero, owner.zero, true,
                            local_xc);
    }));
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
                                                           bool repeated_rhs, std::string& detail,
                                                           bool dense_only) {
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
    const auto result = owner.prepare_resident(detail, repeated_rhs, dense_only);
    if (result != GENERATIVEQC_STATUS_SUCCESS) {
      owner.ready = false;
      return result;
    }
    if (owner.reference_audit_pending) {
      const double* potential{};
      check(owner.grid.reference_potential(potential));
      check(owner.evaluate(owner.reference.reference().density, potential, owner.source.hcore,
                           false));
      owner.reference_audit_pending = false;
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
  const bool direct_projection = can_seed_direct_operator(shape);
  const bool complete = can_seed_fitted_operator(shape) || direct_projection;
  if (direct_projection && !scf::cuda_direct_coulomb_rows(owner.direct, owner.stream).values)
    return cudaSuccess;
  if (owner.transition_workspace || shape.spins != 1 || owner.source.spins != 1 ||
      shape.nao != owner.source.nao || shape.nao < 32 || (shape.nao > 96 && !complete) ||
      !occupied || occupied >= shape.nao || points < 4096 || (axes != 4 && axes != 5) ||
      (owner.fitted && axes != 5 && !complete) || owner.source.correction ||
      owner.expected.identity.model.nonlocal_grid)
    return cudaSuccess;
  const auto rotations = occupied * (shape.nao - occupied);
  const auto products_per_point = axes * rotations * rotations;
  const auto limit = complete                      ? complete_fitted_product_limit
                     : axes == 5 && shape.nao > 64 ? wide_r2scan_transition_product_limit
                                                   : transition_product_limit;
  if (points > limit / products_per_point) return cudaSuccess;
  const auto tile =
      complete && shape.nao <= 128 ? transition_tile_points / 2 : transition_tile_points;
  const auto pairs = shape.nao * (shape.nao + 1) / 2;
  const auto occupied_tile = std::min(occupied, direct_occupied_tile);
  const auto tile_rotations = occupied_tile * (shape.nao - occupied);
  const auto coulomb_elements =
      direct_projection
          ? pairs * tile_rotations +
                direct_transform_tile * (2 * owner.matrix + shape.nao * occupied + tile_rotations)
      : complete ? owner.matrix + shape.nao * occupied + 2 * owner.fitted.naux * rotations
                 : 0;
  const auto bytes = std::max((2 * axes * rotations + 4 * shape.nao) * tile + 1, coulomb_elements) *
                     sizeof(double);
  const auto allowance =
      std::min(max_bytes, owner.options.max_device_bytes - owner.receipt.device_bytes);
  if (bytes > allowance) return cudaSuccess;
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
  owner.transition_tile = tile;
  owner.transition_products = products_per_point * points;
  owner.receipt.transition_workspace_bytes = bytes;
  owner.receipt.device_bytes += bytes;
  return cudaSuccess;
}

bool CudaKsWb97mvResponse::can_seed_fitted_operator(
    const response::CudaCpksShape& shape) const noexcept {
  if (!impl_ || !impl_->ready || !impl_->options.grid.wide_cartesian_grid || !impl_->fitted ||
      shape.spins != 1 || shape.nao != impl_->source.nao || shape.nao <= 96 || shape.nao > 256 ||
      !shape.occupied[0] || shape.occupied[0] >= shape.nao || impl_->fitted.naux > 512 ||
      impl_->source.primary->strategy().spec.exchange.present || impl_->source.correction ||
      impl_->expected.identity.model.nonlocal_correlation ||
      !impl_->options.grid.supports_wide_cartesian_grid())
    return false;
  const auto points = impl_->reference.shape().points;
  const auto rotations = shape.occupied[0] * (shape.nao - shape.occupied[0]);
  // Estimate eligibility before fixed points are retained. The builder later
  // validates the actual cached axes and can refuse without publishing a seed.
  const auto axes =
      impl_->options.grid.functional_code == semilocal_family_code(SemilocalFamily::R2scan) ? 5 : 4;
  return points >= 4096 && points <= complete_fitted_product_limit / (axes * rotations * rotations);
}

bool CudaKsWb97mvResponse::can_seed_direct_operator(
    const response::CudaCpksShape& shape) const noexcept {
  if (!impl_ || !impl_->ready || !impl_->options.grid.wide_cartesian_grid || impl_->fitted ||
      !impl_->direct || shape.spins != 1 || shape.nao != impl_->source.nao || shape.nao <= 96 ||
      shape.nao > 256 || !shape.occupied[0] || shape.occupied[0] >= shape.nao ||
      impl_->source.primary->strategy().spec.exchange.present || impl_->source.correction ||
      impl_->expected.identity.model.nonlocal_correlation ||
      !impl_->options.grid.supports_wide_cartesian_grid())
    return false;
  // NAO is bounded above before forming either product. The two AO-to-OV
  // transforms run once per lower-triangle AO pair and once per OV rotation.
  const auto occupied = shape.occupied[0], rotations = occupied * (shape.nao - occupied);
  const auto pairs = shape.nao * (shape.nao + 1) / 2;
  const auto tiles = 1 + (occupied - 1) / direct_occupied_tile;
  const auto points = impl_->reference.shape().points;
  const auto axes =
      impl_->options.grid.functional_code == semilocal_family_code(SemilocalFamily::R2scan) ? 5 : 4;
  return points >= 4096 &&
         points <= complete_fitted_product_limit / (axes * rotations * rotations) &&
         (pairs + rotations) * shape.nao * (shape.nao * occupied + rotations) <=
             direct_transform_product_limit &&
         (pairs * tiles + rotations) * shape.nao * shape.nao <= direct_transform_gather_limit;
}

bool CudaKsWb97mvResponse::has_transition_builder() const noexcept {
  return impl_ && impl_->transition_workspace;
}

cudaError_t CudaKsWb97mvResponse::retain_metric_projection(const double* frozen_core,
                                                           std::size_t max_bytes) noexcept {
  if (!impl_ || !impl_->ready || impl_->used) return cudaErrorInvalidResourceHandle;
  auto& owner = *impl_;
  auto& metric = owner.metric_projection;
  const auto shape = owner.reference.shape();
  if (!frozen_core || metric.rhs || !owner.transition_workspace ||
      !owner.options.grid.wide_cartesian_grid || owner.grid.transition_axes() != 5 ||
      shape.nao > metric_projection_max_ao || shape.atoms > metric_projection_max_ao)
    return cudaSuccess;
  const auto coordinates = 3 * shape.atoms;
  const auto rotations = owner.transition_occupied * (shape.nao - owner.transition_occupied);
  constexpr std::uint64_t max_products = 2ULL << 40;
  if (!coordinates || !rotations || shape.points > max_products / (5 * rotations * coordinates))
    return cudaSuccess;
  const auto& system = owner.source.primary->system();
  if (system.basis_representation != core::kCartesianBasis || system.atoms.size() != shape.atoms)
    return cudaSuccess;
  std::array<int, metric_projection_max_ao + 1> offsets{};
  std::size_t previous{};
  for (const auto& shell : system.shells) {
    const auto atom = shell.atom_index, angular = shell.angular_momentum;
    // Grid AO columns and the stationary shell order must agree. An
    // interleaved or unsupported shell topology retains the ordinary action.
    if (atom >= shape.atoms || atom < previous || angular > 1) return cudaSuccess;
    offsets[atom + 1] += (angular + 1) * (angular + 2) / 2;
    previous = atom;
  }
  for (std::size_t atom = 0; atom < shape.atoms; ++atom) offsets[atom + 1] += offsets[atom];
  if (static_cast<std::size_t>(offsets[shape.atoms]) != shape.nao) return cudaSuccess;

  const auto projected_size = rotations * coordinates;
  const auto row_size = 3 * owner.matrix;
  const auto jet_size = 8 * shape.nao * owner.transition_tile;
  const auto intermediate_size = 3 * jet_size;
  const auto field_size = 5 * coordinates * owner.transition_tile;
  const auto bytes =
      sizeof(double) * (projected_size + row_size + jet_size + intermediate_size + field_size) +
      sizeof(int) * (shape.atoms + 2);
  if (bytes > max_bytes || bytes > owner.options.max_device_bytes - owner.receipt.device_bytes)
    return cudaSuccess;
  auto error = owner.authenticate(owner.stream);
  if (error != cudaSuccess) return error;
  void* allocation{};
  error = runtime::resource_cuda_malloc(&allocation, bytes);
  if (error == cudaErrorMemoryAllocation) {
    (void)cudaGetLastError();
    return cudaSuccess;
  }
  if (error != cudaSuccess) return error;
  auto* rhs = static_cast<double*>(allocation);
  auto* rows = rhs + projected_size;
  auto* jets = rows + row_size;
  auto* projected = jets + jet_size;
  auto* fields = projected + intermediate_size;
  auto* device_offsets = reinterpret_cast<int*>(fields + field_size);
  error = cudaMemcpyAsync(device_offsets, offsets.data(), sizeof(int) * (shape.atoms + 1),
                          cudaMemcpyHostToDevice, owner.stream);
  const auto settled = cudaStreamSynchronize(owner.stream);
  if (error != cudaSuccess || settled != cudaSuccess) {
    (void)runtime::resource_cuda_free(allocation);
    return settled != cudaSuccess ? settled : error;
  }
  metric = {frozen_core, rhs,       rows,           jets,
            projected,   fields,    device_offsets, device_offsets + shape.atoms + 1,
            coordinates, rotations, false};
  owner.receipt.device_bytes += bytes;
  // This optional lease is restricted to a shape and architecture with
  // aligned int8 tensor products. It never changes the fixed-grid provider.
  const auto rank = 5 * owner.transition_tile;
  const auto block = std::size_t(transition_integer::block_width);
  if (rotations < 8 * block || rotations % block || rank > 640 || rank % 4) return cudaSuccess;
  int device{}, major{};
  if ((error = cudaGetDevice(&device)) != cudaSuccess ||
      (error = cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, device)) !=
          cudaSuccess)
    return error;
  if (major != 12) return cudaSuccess;
  const auto extent = ((rotations / 2 + block / 2) / block) * block;
  const auto integer_count = extent * extent;
  const auto digit_count = transition_integer::digits * rank * rotations;
  const auto integer_bytes = sizeof(int) * (integer_count + 2 * rotations) + 4 * digit_count;
  constexpr auto provider_allowance = tensor::CudaContractionContext::kProviderAllowance;
  if (integer_bytes + provider_allowance > max_bytes - bytes ||
      integer_bytes + provider_allowance >
          owner.options.max_device_bytes - owner.receipt.device_bytes)
    return cudaSuccess;
  if (cublasCreate(&owner.integer_blas) != CUBLAS_STATUS_SUCCESS ||
      cublasSetStream(owner.integer_blas, owner.stream) != CUBLAS_STATUS_SUCCESS) {
    if (owner.integer_blas) (void)cublasDestroy(std::exchange(owner.integer_blas, nullptr));
    return cudaSuccess;
  }
  void* integer_storage{};
  error = runtime::resource_cuda_malloc(&integer_storage, integer_bytes);
  if (error == cudaErrorMemoryAllocation) {
    (void)cudaGetLastError();
    (void)cublasDestroy(std::exchange(owner.integer_blas, nullptr));
    return cudaSuccess;
  }
  if (error != cudaSuccess) return error;
  auto* products = static_cast<int*>(integer_storage);
  auto* digit_storage = reinterpret_cast<signed char*>(products + integer_count + 2 * rotations);
  owner.integer_projection = {products,
                              products + integer_count,
                              digit_storage,
                              digit_storage + digit_count,
                              digit_storage + 2 * digit_count,
                              digit_storage + 3 * digit_count,
                              false};
  owner.receipt.device_bytes += integer_bytes + provider_allowance;
  return cudaSuccess;
}

const double* CudaKsWb97mvResponse::metric_projection(std::size_t column) const noexcept {
  if (!impl_ || !impl_->ready) return nullptr;
  const auto& metric = impl_->metric_projection;
  return metric.ready && column < metric.coordinates ? metric.rhs + column * metric.rotations
                                                     : nullptr;
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
  const auto panel = axes * rotations * owner.transition_tile;
  auto* fields = static_cast<double*>(owner.transition_workspace);
  auto* tangents = fields + panel;
  auto* jets = tangents + panel;
  auto* refused = reinterpret_cast<int*>(jets + 4 * shape.nao * owner.transition_tile);
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
  runtime::cuda_trace::trace_counter("cpks_transition_correction_points",
                                     std::max(correction_points, 0));
  runtime::cuda_trace::trace_counter("cpks_transition_point_tile", owner.transition_tile);
  if (correction_points < 0 ||
      static_cast<std::uint64_t>(correction_points) > transition_correction_limit / rotations)
    return cudaErrorNotSupported;
  std::uint64_t products{};
  const bool triangle =
      correction_points == 0 && rotations >= 2048 &&
      (self.can_seed_fitted_operator(shape) || self.can_seed_direct_operator(shape));
  auto& integer = owner.integer_projection;
  const bool integer_path = triangle && integer.product && !integer.disabled;
  std::uint64_t integer_products{};
  if (integer_path) {
    error = cudaMemsetAsync(correction_count, 0, sizeof(int), stream);
    if (error != cudaSuccess) return error;
  }
  auto& metric = owner.metric_projection;
  metric.ready = false;
  bool project_metric = metric.rhs && correction_points == 0;
  const auto order = static_cast<int>(shape.nao);
  const auto occupied = static_cast<int>(owner.transition_occupied);
  if (project_metric) {
    error = cudaMemsetAsync(metric.refused, 0, sizeof(int), stream);
    if (error != cudaSuccess) return error;
    pack_metric_rows<<<maps::blocks(metric.coordinates * owner.matrix), 128, 0, stream>>>(
        shape.nao, metric.coordinates, metric.core, metric.offsets, metric.rows, metric.refused);
    if ((error = cudaGetLastError()) != cudaSuccess) return error;
  }
  // A unit RKS rotation yields 4 times the bilinear feature (occupation
  // two and two mirrored density legs). The OV projection reads one leg.
  constexpr double occupation_and_lift = 4.0, zero = 0.0, one = 1.0;
  std::size_t begin{};
  for (; begin < points && error == cudaSuccess;) {
    const auto count = std::min(points - begin, owner.transition_tile);
    error = owner.grid.enqueue_transition_tile(reference.coefficients, owner.transition_occupied,
                                               begin, count, jets, fields, tangents,
                                               correction_points != 0, refused, stream);
    if (error != cudaSuccess) break;
    const auto rows = static_cast<int>(axes * count);
    const auto columns = static_cast<int>(rotations);
    const auto* beta = begin ? &one : &zero;
    if (integer_path && rows % 4 == 0) {
      error = transition_integer::project(owner.integer_blas, integer, fields, tangents, matrix,
                                          columns, rows, begin == 0, correction_count, stream);
      if (error != cudaSuccess) break;
      const auto logical = rotations * (rotations + transition_integer::block_width) / 2 * rows;
      products += logical;
      integer_products +=
          logical * (transition_integer::digits * (transition_integer::digits + 1) / 2);
    } else {
      const auto result =
          triangle ? transition_triangle(owner.transition_blas, fields, tangents, matrix, columns,
                                         rows, 0, columns, &occupation_and_lift, beta, products)
                   : owner.transition_blas.product(true, false, columns, columns, rows,
                                                   &occupation_and_lift, fields, rows, tangents,
                                                   rows, beta, matrix, columns);
      if (result != CUBLAS_STATUS_SUCCESS) {
        error = cudaErrorNotSupported;
        break;
      }
      if (!triangle) products += rotations * rotations * rows;
    }
    if (project_metric) {
      const auto dot_count = static_cast<int>(4 * count);
      const auto* mo = reference.coefficients;
      auto& blas = owner.transition_blas;
      const auto panel_size = order * dot_count;
      auto* occupied_jets = metric.jets + panel_size;
      auto* occupied_rows = metric.projected + 3 * panel_size;
      constexpr double two = 2;
      if (blas.product(false, false, order, dot_count, order, &one, mo, order, jets, order, &zero,
                       metric.jets, order) != CUBLAS_STATUS_SUCCESS ||
          blas.product(false, false, order, dot_count, occupied, &two, mo, order, jets, order,
                       &zero, occupied_jets, order) != CUBLAS_STATUS_SUCCESS ||
          blas.product(false, false, 3 * order, 2 * dot_count, order, &one, metric.rows, 3 * order,
                       metric.jets, order, &zero, metric.projected,
                       3 * order) != CUBLAS_STATUS_SUCCESS) {
        project_metric = false;
      } else {
        error = owner.grid.enqueue_metric_feature_tile(
            count, metric.offsets, metric.jets, occupied_jets, metric.projected, occupied_rows,
            metric.fields, metric.refused, stream);
        if (error != cudaSuccess) break;
        // The weighted transition tangent already owns the point XC Hessian.
        // Its transpose against the metric JVP yields the physical OV Fock
        // with no rotation lift (the metric JVP is a full density direction).
        project_metric = blas.product(true, false, columns, static_cast<int>(metric.coordinates),
                                      rows, &one, tangents, rows, metric.fields, rows, beta,
                                      metric.rhs, columns) == CUBLAS_STATUS_SUCCESS;
      }
    }
    begin += count;
  }
  if (error == cudaSuccess && triangle) {
    mirror_transition_triangle<<<maps::blocks(rotations * rotations), 128, 0, stream>>>(matrix,
                                                                                        rotations);
    error = cudaGetLastError();
  }
  int refused_host{};
  int metric_refused{};
  int integer_refused{};
  if (error == cudaSuccess && integer_path)
    error = cudaMemcpyAsync(&integer_refused, correction_count, sizeof(int), cudaMemcpyDeviceToHost,
                            stream);
  if (error == cudaSuccess && project_metric) {
    const auto size = rotations * metric.coordinates;
    audit_metric_projection<<<maps::blocks(size), 128, 0, stream>>>(size, metric.rhs,
                                                                    metric.refused);
    error = cudaGetLastError();
    if (error == cudaSuccess)
      error = cudaMemcpyAsync(&metric_refused, metric.refused, sizeof(int), cudaMemcpyDeviceToHost,
                              stream);
  }
  if (error == cudaSuccess)
    error = cudaMemcpyAsync(&refused_host, refused, sizeof(int), cudaMemcpyDeviceToHost, stream);
  const auto settled = cudaStreamSynchronize(stream);
  owner.grid.record_transition_projection(products, correction_points * rotations);
  if (settled != cudaSuccess) return settled;
  if (integer_path) {
    runtime::cuda_trace::trace_counter("cpks_transition_int8_products", integer_products);
    if (integer_refused || error == cudaErrorNotSupported) {
      // No consumer sees the partial matrix. Rebuild once from the original
      // features, with no further integer attempts for this retained owner.
      integer.disabled = true;
      return seed_local_xc(context, shape, reference, matrix, stream);
    }
  }
  if (error != cudaSuccess) return error;
  error = owner.authenticate(stream);
  if (error != cudaSuccess) return error;
  if (refused_host || begin != points || products > owner.transition_products)
    return cudaErrorNotSupported;
  metric.ready = project_metric && !metric_refused;
  if (metric.ready)
    runtime::cuda_trace::trace_counter("cpks_metric_projection_products",
                                       axes * points * rotations * metric.coordinates);
  return cudaSuccess;
}

cudaError_t CudaKsWb97mvResponse::seed_fitted_operator(void* context,
                                                       const response::CudaCpksShape& shape,
                                                       const response::CudaCpksReference& reference,
                                                       double* matrix,
                                                       cudaStream_t stream) noexcept {
  if (!context || !matrix) return cudaErrorInvalidValue;
  auto& self = *static_cast<CudaKsWb97mvResponse*>(context);
  if (!self.can_seed_fitted_operator(shape)) return cudaErrorNotSupported;
  auto error = seed_local_xc(context, shape, reference, matrix, stream);
  if (error != cudaSuccess) return error;
  auto& owner = *self.impl_;
  const auto nao = static_cast<int>(shape.nao), occupied = static_cast<int>(shape.occupied[0]);
  const auto virtuals = nao - occupied, rotations = occupied * virtuals;
  const auto naux = static_cast<int>(owner.fitted.naux);
  auto* slice = static_cast<double*>(owner.transition_workspace);
  auto* half = slice + owner.matrix;
  auto* projected = half + nao * occupied;
  auto* whitened = projected + naux * rotations;
  const auto* mo = reference.coefficients;
  const auto* virtual_coefficients = mo + nao * occupied;
  constexpr double one = 1, zero = 0, four = 4;
  auto& blas = owner.transition_blas;
  // The AO slice is symmetric. Project to [virtual,occupied] in column-major
  // order, exactly the CPKS rotation flattening. RKS P=2CC^T contributes
  // two mirrored legs, so the kernel in this basis is 4*(WT)^T*(WT).
  for (int aux = 0; aux < naux; ++aux) {
    gather_fitted_slice<<<maps::blocks(owner.matrix), 128, 0, stream>>>(owner.fitted_panel, slice,
                                                                        owner.matrix, aux, naux);
    if ((error = cudaGetLastError()) != cudaSuccess) break;
    if (blas.product(false, false, nao, occupied, nao, &one, slice, nao, mo, nao, &zero, half,
                     nao) != CUBLAS_STATUS_SUCCESS ||
        blas.product(true, false, virtuals, occupied, nao, &one, virtual_coefficients, nao, half,
                     nao, &zero, projected + aux * rotations, virtuals) != CUBLAS_STATUS_SUCCESS) {
      error = cudaErrorNotSupported;
      break;
    }
  }
  if (error == cudaSuccess &&
      (blas.product(false, false, rotations, naux, naux, &one, projected, rotations,
                    owner.fitted.whitening, naux, &zero, whitened,
                    rotations) != CUBLAS_STATUS_SUCCESS ||
       blas.product(false, true, rotations, rotations, naux, &four, whitened, rotations, whitened,
                    rotations, &one, matrix, rotations) != CUBLAS_STATUS_SUCCESS))
    error = cudaErrorNotSupported;
  const auto settled = cudaStreamSynchronize(stream);
  return settled != cudaSuccess ? settled : error;
}

cudaError_t CudaKsWb97mvResponse::seed_direct_operator(void* context,
                                                       const response::CudaCpksShape& shape,
                                                       const response::CudaCpksReference& reference,
                                                       double* matrix,
                                                       cudaStream_t stream) noexcept {
  if (!context || !matrix) return cudaErrorInvalidValue;
  auto& self = *static_cast<CudaKsWb97mvResponse*>(context);
  if (!self.can_seed_direct_operator(shape) || !self.has_transition_builder())
    return cudaErrorNotSupported;
  auto& owner = *self.impl_;
  const auto rows = scf::cuda_direct_coulomb_rows(owner.direct, stream);
  if (!rows.values) return cudaErrorNotSupported;
  auto error = runtime::cuda_trace::trace_call("hessian_cpks_seed_grid", stream, [&] {
    return seed_local_xc(context, shape, reference, matrix, stream);
  });
  if (error != cudaSuccess) return error;

  runtime::cuda_trace::TraceRegion coulomb_trace("hessian_cpks_seed_direct_j", stream);
  const auto nao = static_cast<int>(shape.nao), occupied = static_cast<int>(shape.occupied[0]);
  const auto virtuals = nao - occupied, rotations = occupied * virtuals;
  const std::size_t pairs = shape.nao * (shape.nao + 1) / 2;
  const auto half_size = nao * occupied;
  const auto occupied_tile = std::min(occupied, static_cast<int>(direct_occupied_tile));
  const auto tile_rotations = occupied_tile * virtuals;
  auto* intermediate = static_cast<double*>(owner.transition_workspace);
  auto* slices = intermediate + pairs * tile_rotations;
  auto* half = slices + direct_transform_tile * owner.matrix;
  auto* projected = half + direct_transform_tile * half_size;
  auto* mo = projected + direct_transform_tile * tile_rotations;
  const auto* virtual_coefficients = mo + half_size;
  const auto frames = direct_transform_tile * owner.matrix;
  repeat_orbital_frame<<<maps::blocks(frames), 128, 0, stream>>>(reference.coefficients, mo,
                                                                 owner.matrix, frames);
  if ((error = cudaGetLastError()) != cudaSuccess) return error;
  constexpr double one = 1, zero = 0, four = 4;
  const auto transform = [&](const double* input, std::size_t stride, int count, int first_occupied,
                             int occupied_count, double* output, const double* scale,
                             const double* beta) -> cudaError_t {
    const auto elements = count * owner.matrix;
    unpack_coulomb_slices<<<maps::blocks(elements), 128, 0, stream>>>(input, slices, stride, nao,
                                                                      elements);
    if (const auto launch = cudaGetLastError(); launch != cudaSuccess) return launch;
    auto& blas = owner.transition_blas;
    const auto half_stride = nao * occupied_count;
    const auto rotation_stride = virtuals * occupied_count;
    if (blas.product_strided(false, false, nao, occupied_count, nao, &one, slices, nao,
                             owner.matrix, mo + first_occupied * nao, nao, owner.matrix, &zero,
                             half, nao, half_stride, count) != CUBLAS_STATUS_SUCCESS)
      return cudaErrorNotSupported;
    owner.receipt.direct_projection_products += std::uint64_t(count) * nao * half_stride;
    const auto status = blas.product_strided(
        true, false, virtuals, occupied_count, nao, scale, virtual_coefficients, nao, owner.matrix,
        half, nao, half_stride, beta, output, virtuals, rotation_stride, count);
    if (status != CUBLAS_STATUS_SUCCESS) return cudaErrorNotSupported;
    owner.receipt.direct_projection_products += std::uint64_t(count) * nao * rotation_stride;
    return cudaSuccess;
  };
  // The column-major flattening is OV with virtual fastest, just like CPKS.
  // Expanding each packed triangle supplies both AO orientations once. The
  // second transform multiplies by four for RKS occupation and rotation lift.
  // Blocking occupied columns rereads pair rows but never repeats a GEMM
  // term. This keeps the live pair-by-rotation intermediate bounded.
  for (int first = 0; first < occupied && error == cudaSuccess; first += occupied_tile) {
    const auto occupied_count = std::min(occupied - first, occupied_tile);
    const auto columns = static_cast<std::size_t>(occupied_count * virtuals);
    for (std::size_t start = 0; start < pairs && error == cudaSuccess;
         start += direct_transform_tile) {
      const auto count = std::min(pairs - start, direct_transform_tile);
      error = transform(rows.values + start * rows.stride, rows.stride, static_cast<int>(count),
                        first, occupied_count, projected, &one, &zero);
      if (error != cudaSuccess) break;
      const dim3 blocks((columns + 31) / 32, (count + 31) / 32);
      transpose_coulomb_window<<<blocks, dim3(32, 8), 0, stream>>>(projected, intermediate + start,
                                                                   pairs, columns, count);
      error = cudaGetLastError();
    }
    for (std::size_t start = 0; start < columns && error == cudaSuccess;
         start += direct_transform_tile) {
      const auto count = std::min(columns - start, direct_transform_tile);
      error = transform(intermediate + start * pairs, pairs, static_cast<int>(count), 0, occupied,
                        matrix + (first * virtuals + start) * rotations, &four, &one);
    }
  }
  const auto settled = cudaStreamSynchronize(stream);
  return settled != cudaSuccess ? settled : error;
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
