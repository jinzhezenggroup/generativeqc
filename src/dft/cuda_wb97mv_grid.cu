#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <tuple>
#include <type_traits>
#include <utility>

#include "dft/cuda_grid_geometry.hpp"
#include "dft/cuda_wb97mv_grid.hpp"
#include "dft/hessian_aot.hpp"
#include "dft/split_hybrid_cuda_selection.hpp"
#include "dft/xc_point_response.hpp"
#include "generated_split_hybrid_registry.cuh"
#include "r2scan_cuda_domain.hpp"
#include "response/b3lyp_direction.cuh"
#include "response/grid_geometry_response.cuh"
#if GENERATIVEQC_ENABLE_M062X_STATIONARY_POINT
#include "response/m06_2x_stationary.cuh"
#endif
#if GENERATIVEQC_ENABLE_MN15_STATIONARY_POINT
#include "response/mn15_stationary.cuh"
#endif
#include "response/r2scan_stationary.cuh"
#include "response/wb97mv_grid_response.cuh"
#include "response/wb97mv_nuclear_grid.cuh"
#include "runtime/resource_cuda.cuh"

namespace generativeqc::dft {
namespace {
namespace maps = generativeqc_wb97mv_grid_maps;
namespace ad = generativeqc_wb97mv_grid_ad;
namespace nuclear_maps = generativeqc_wb97mv_nuclear_maps;
namespace nuclear_ad = generativeqc_wb97mv_nuclear_ad;
namespace adjoints = generativeqc_wb97mv_nuclear_ao_adjoints;
namespace geometry_ad = generativeqc_grid_geometry_ad;
constexpr unsigned adjoint_blocks = 192;

bool product(std::size_t left, std::size_t right, std::size_t& value) noexcept {
  if (right && left > std::numeric_limits<std::size_t>::max() / right) return false;
  value = left * right;
  return true;
}

double revised_beta(double damping) noexcept {
  return std::pow(3.0 / (damping * damping), 0.75) / 32.0;
}

struct Layout {
  std::size_t bytes{}, matrix{}, matrices{}, fields{}, basis_elements{}, ao_elements{};
  std::size_t status{}, count{}, active{}, ao{}, base{}, direction{}, coefficients{};
  std::size_t reference{}, candidate{}, pairs{}, tensor{}, tensor_bytes{};
  maps::Extents extents{};
  std::size_t full_fields{}, base_partial{}, right_partial{}, left_fields{}, mixed_fields{};
  std::size_t right_ao{}, right_density{}, right_weights{}, right_points{}, zero_density{};
  std::size_t energy{}, energy_scalar{}, scalar{};
  std::size_t staged_left_ao{}, staged_mixed_ao{}, staged_left_weights{}, staged_mixed_weights{},
      staged_left_points{};
  std::size_t adjoint_first{}, adjoint_second{}, adjoint_output_tile{}, adjoint_scratch{},
      adjoint_bytes{}, adjoint_partials{}, adjoint_build_work{}, adjoint_dot_work{}, adjoint_tile{};
  std::size_t cartesian_ao_owners{};
  bool replay{}, fused_adjoint{}, reuse_right_adjoint{}, factored_only{};
  std::size_t reference_work{}, direction_feature_work{}, direction_work{}, mixed_work{};
  std::size_t point_linearization{}, point_replay_mask{}, point_probe_status{};
  unsigned point_linearization_axes{};
  nuclear_maps::Extents nuclear_extents{};
};

constexpr std::size_t potential_tile_limit = 1024;

template <class Product>
bool factored_potential_work(const CudaWb97mvGridShape& shape, std::size_t& work) noexcept {
  const auto pairs = shape.nao * (shape.nao + 1) / 2;
  return nuclear_maps::add_work(work, {shape.points, pairs, Product::ao_count + 1}) &&
         nuclear_maps::add_work(work, {shape.points, shape.nao, Product::transform_terms}) &&
         nuclear_maps::add_work(work, {pairs, 1 + (shape.points - 1) / potential_tile_limit, 128});
}

bool layout_impl(const CudaWb97mvGridShape& shape, const CudaWb97mvGridOptions& options,
                 Layout& plan, bool factored) noexcept {
  const auto code = options.functional_code;
  const bool wb97mv = code == semilocal_family_code(SemilocalFamily::Wb97mv);
  const bool split_hybrid = cuda_split_hybrid_variational_domain(code) != 0;
  const bool nonlocal = options.includes_nonlocal();
  const bool unscaled = split_hybrid || code == semilocal_family_code(SemilocalFamily::B3lyp) ||
                        code == semilocal_family_code(SemilocalFamily::R2scan);
  if ((!wb97mv && !options.revised_nonlocal &&
       options.component != CudaWb97mvGridComponent::combined) ||
      (split_hybrid && nonlocal) ||
      (!wb97mv && !unscaled && code != semilocal_family_code(SemilocalFamily::Lda) &&
       code != semilocal_family_code(SemilocalFamily::Pbe)) ||
      ((wb97mv || unscaled) && (options.exchange_scale != 1 || options.correlation_scale != 1)) ||
      !std::isfinite(options.exchange_scale) || options.exchange_scale < 0 ||
      !std::isfinite(options.correlation_scale) || options.correlation_scale < 0 ||
      (nonlocal && (!options.pair_tile_points || options.pair_tile_points > 256 ||
                    !options.max_pair_evaluations)))
    return false;
  if (options.revised_nonlocal) {
    const auto& parameters = *options.revised_nonlocal;
    if (parameters.variant != nlc::Vv10Variant::rvv10 || !std::isfinite(parameters.b) ||
        parameters.b <= 0 || !std::isfinite(parameters.c) || parameters.c <= 0 ||
        !std::isfinite(parameters.coefficient) || parameters.coefficient <= 0)
      return false;
    const auto beta = revised_beta(parameters.b) * parameters.coefficient;
    if (!std::isfinite(beta) || beta <= 0) return false;
  }
  if (!shape.atoms || !shape.primitives || !shape.nao || !shape.points ||
      (shape.spins != 1 && shape.spins != 2) ||
      !std::isfinite(options.symmetry_absolute_tolerance) ||
      options.symmetry_absolute_tolerance < 0 ||
      !std::isfinite(options.symmetry_relative_tolerance) ||
      options.symmetry_relative_tolerance < 0 || options.symmetry_relative_tolerance >= 1 ||
      !product(shape.nao, shape.nao, plan.matrix) ||
      !product(plan.matrix, shape.spins, plan.matrices) ||
      !product(shape.points, 5 * shape.spins, plan.fields) ||
      !product(shape.points, shape.nao, plan.ao_elements) ||
      !product(plan.ao_elements, 4, plan.ao_elements))
    return false;
  if (factored && (!options.nuclear_response || nonlocal || shape.spins != 1 ||
                   !options.max_nuclear_adjoint_bytes || shape.nao <= 64 || shape.nao > 96 ||
                   shape.points < 4096))
    return false;
  plan.factored_only = factored;
  std::size_t centers{}, primitives{}, records{}, elements{};
  if (!product(shape.atoms, 3, centers) || !product(shape.primitives, 2, primitives) ||
      !product(shape.nao, 16, records) || centers > SIZE_MAX - primitives ||
      centers + primitives > SIZE_MAX - records)
    return false;
  plan.basis_elements = centers + primitives + records;
  const auto reserve = [&](std::size_t count, std::size_t& offset) {
    if (count > SIZE_MAX / sizeof(double) - elements) return false;
    offset = elements;
    elements += count;
    return true;
  };
  static_assert(sizeof(std::size_t) <= sizeof(double));
  plan.extents = {shape.nao, shape.points};
  std::size_t features{}, potential{}, pair_elements{};
  if ((!factored && (maps::workspace_features(plan.extents, features) != cudaSuccess ||
                     maps::workspace_potential(plan.extents, potential) != cudaSuccess)) ||
      (nonlocal && (!product(std::min(shape.points, options.pair_tile_points),
                             std::min(shape.points, options.pair_tile_points), pair_elements) ||
                    !product(pair_elements, options.nuclear_response ? 10 : 4, pair_elements))))
    return false;
  plan.tensor_bytes = std::max(features, potential);
  if (options.nuclear_response) {
    if (!options.max_nuclear_affine_terms ||
        !product(shape.points, 5 * shape.spins + 4, plan.full_fields))
      return false;
    plan.nuclear_extents = {shape.nao, shape.points, plan.full_fields};
    if (factored) {
      // Every consumer is bound to the three-strip feature kernel, tiled
      // potential and read-only fused adjoint. Do not reserve affine scratch
      // or admit an affine fallback under these lower semantic work counts.
      std::size_t energy_bytes{}, sum_work{};
      if (nuclear_maps::workspace_energy_sum(plan.nuclear_extents, energy_bytes) != cudaSuccess ||
          !nuclear_maps::work_energy_sum(plan.nuclear_extents, sum_work))
        return false;
      plan.tensor_bytes = std::max(
          {energy_bytes, 2 * plan.matrix * sizeof(double),
           2 * ad::MovingPotential::ao_count * shape.nao * potential_tile_limit * sizeof(double)});
      using nuclear_maps::add_work;
      const auto feature_bound = [&](std::size_t& total, unsigned products, unsigned epilogue) {
        return add_work(total, {shape.points, 96, shape.nao, products}) &&
               add_work(total, {shape.points, 96, epilogue}) && add_work(total, {2, plan.matrix});
      };
      if (!feature_bound(plan.reference_work, ad::large_fixed_features_products,
                         ad::large_fixed_features_epilogue) ||
          !factored_potential_work<ad::FixedPotential>(shape, plan.reference_work) ||
          !add_work(plan.reference_work, {sum_work}) ||
          !feature_bound(plan.direction_feature_work, ad::large_moving_features_products,
                         ad::large_moving_features_epilogue))
        return false;
      plan.direction_work = plan.direction_feature_work;
      if (!factored_potential_work<ad::MovingPotential>(shape, plan.direction_work) ||
          std::max(plan.reference_work, plan.direction_work) > options.max_nuclear_affine_terms)
        return false;
    } else {
      using Size = cudaError_t (*)(nuclear_maps::Extents, std::size_t&);
      for (Size size :
           {nuclear_maps::workspace_feature_direction, nuclear_maps::workspace_feature_mixed,
            nuclear_maps::workspace_potential_direction, nuclear_maps::workspace_energy_sum,
            nuclear_maps::workspace_mixed_contraction}) {
        std::size_t bytes{};
        if (size(plan.nuclear_extents, bytes) != cudaSuccess) return false;
        plan.tensor_bytes = std::max(plan.tensor_bytes, bytes);
      }
      std::size_t first_work{}, second_work{}, potential_work{}, sum_work{}, contraction_work{};
      std::size_t base_work{}, assembly_work{};
      if (!nuclear_maps::work_feature_direction(plan.nuclear_extents, first_work) ||
          !nuclear_maps::work_feature_mixed(plan.nuclear_extents, second_work) ||
          !nuclear_maps::work_potential_direction(plan.nuclear_extents, potential_work) ||
          !nuclear_maps::work_energy_sum(plan.nuclear_extents, sum_work) ||
          !nuclear_maps::work_mixed_contraction(plan.nuclear_extents, contraction_work) ||
          !maps::work_features(plan.extents, base_work) ||
          !maps::work_potential(plan.extents, assembly_work) ||
          !nuclear_maps::add_work(plan.reference_work, {base_work, shape.spins}) ||
          !nuclear_maps::add_work(plan.reference_work, {assembly_work, shape.spins}) ||
          !nuclear_maps::add_work(plan.reference_work, {sum_work}) ||
          !nuclear_maps::add_work(plan.direction_feature_work, {first_work, shape.spins}) ||
          !nuclear_maps::add_work(plan.direction_work, {plan.direction_feature_work}) ||
          !nuclear_maps::add_work(plan.direction_work, {potential_work, shape.spins}) ||
          !nuclear_maps::add_work(plan.mixed_work, {first_work, shape.spins}) ||
          !nuclear_maps::add_work(plan.mixed_work, {second_work, shape.spins}) ||
          !nuclear_maps::add_work(plan.mixed_work, {contraction_work}) ||
          std::max({plan.reference_work, plan.direction_work, plan.mixed_work}) >
              options.max_nuclear_affine_terms)
        return false;
    }
  }
  if (!reserve(1, plan.status) ||
      (nonlocal && (!reserve(1, plan.count) || !reserve(shape.points, plan.active))) ||
      !reserve(plan.ao_elements, plan.ao) || !reserve(plan.fields, plan.base) ||
      !reserve(plan.fields, plan.direction) || !reserve(plan.fields, plan.coefficients) ||
      !reserve(plan.matrices, plan.reference) || !reserve(plan.matrices, plan.candidate) ||
      !reserve(pair_elements, plan.pairs) ||
      !reserve(plan.tensor_bytes / sizeof(double), plan.tensor))
    return false;
  if (options.nuclear_response &&
      (!reserve(plan.full_fields, plan.base_partial) ||
       !reserve(plan.full_fields, plan.right_partial) ||
       !reserve(plan.full_fields, plan.left_fields) ||
       !reserve(plan.full_fields, plan.mixed_fields) || !reserve(plan.ao_elements, plan.right_ao) ||
       !reserve(plan.matrices, plan.right_density) || !reserve(shape.points, plan.right_weights) ||
       !reserve(3 * shape.points, plan.right_points) ||
       !reserve(plan.matrices, plan.zero_density) || !reserve(shape.points, plan.energy) ||
       !reserve(1, plan.energy_scalar) || !reserve(1, plan.scalar)))
    return false;
  const auto mandatory_elements = elements;
  if (options.nuclear_response && shape.spins == 1 && !nonlocal &&
      options.max_nuclear_adjoint_bytes) {
    // A private direction can retire right AO tiles after their last consumer.
    // The tile output may borrow tensor scratch, separate from adjoint scratch.
    const auto tile = std::min<std::size_t>(4096, shape.points);
    const auto tiles = 1 + (shape.points - 1) / tile;
    const adjoints::Extents extents{shape.nao, tile};
    const bool fused = factored || (shape.nao >= 8 && shape.nao <= 64 && shape.points >= 4096);
    std::size_t scratch_bytes{}, first{}, second{}, first_work{}, second_work{};
    const auto tile_values = 4 * shape.nao * tile;
    auto admitted_tile = tile;
    bool planned;
    if (fused) {
      // The fused build keeps its products in registers. Reserve only the
      // later column reduction scratch for the geometry producer's full warp.
      scratch_bytes = 3 * std::min(shape.atoms, cuda_grid_tiled_cartesian_atoms) * adjoint_blocks *
                      sizeof(double);
      planned = adjoints::work_fused_ao({shape.nao, shape.points}, plan.adjoint_build_work);
    } else {
      planned = adjoints::workspace_feature_direction(extents, first) == cudaSuccess &&
                adjoints::workspace_feature_mixed(extents, second) == cudaSuccess &&
                adjoints::work_feature_direction(extents, first_work) &&
                adjoints::work_feature_mixed(extents, second_work) &&
                adjoints::add_work(plan.adjoint_build_work, {first_work, tiles}) &&
                adjoints::add_work(plan.adjoint_build_work, {second_work, tiles});
      scratch_bytes = std::max(first, second);
    }
    const auto fits = [&] {
      return elements <= options.max_workspace_bytes / sizeof(double) &&
             elements - mandatory_elements <= options.max_nuclear_adjoint_bytes / sizeof(double);
    };
    const auto reserve_adjoint = [&](bool reuse_right) {
      elements = mandatory_elements;
      plan.adjoint_output_tile = plan.tensor;
      plan.adjoint_second = plan.right_ao;
      if (reuse_right) {
        if (!reserve(plan.ao_elements, plan.adjoint_first) ||
            !reserve(adjoint_blocks, plan.adjoint_partials) ||
            !reserve(scratch_bytes / sizeof(double), plan.adjoint_scratch) || !fits())
          return false;
        const auto available = std::min(
            options.max_workspace_bytes / sizeof(double) - elements,
            options.max_nuclear_adjoint_bytes / sizeof(double) - (elements - mandatory_elements));
        const auto tensor_values = plan.tensor_bytes / sizeof(double);
        const auto stride = 4 * shape.nao;
        admitted_tile = std::min(tile, std::max(available, tensor_values) / stride);
        // Do not trade the memory bound for an unbounded number of tiny launches.
        return admitted_tile >= std::min<std::size_t>(64, tile) &&
               (admitted_tile * stride <= tensor_values ||
                reserve(admitted_tile * stride, plan.adjoint_output_tile));
      }
      return reserve(plan.ao_elements, plan.adjoint_first) &&
             reserve(plan.ao_elements, plan.adjoint_second) &&
             (fused || tile_values <= plan.tensor_bytes / sizeof(double) ||
              reserve(tile_values, plan.adjoint_output_tile)) &&
             reserve(adjoint_blocks, plan.adjoint_partials) &&
             reserve(scratch_bytes / sizeof(double), plan.adjoint_scratch) && fits();
    };
    const auto admit_adjoint_storage = [&] {
      if (reserve_adjoint(false)) return true;
      // The factored plan has checked callers as well: their right AO panel
      // must stay read-only for the lifetime of the prepared direction.
      if (!fused || factored ||
          !adjoints::add_work(plan.adjoint_build_work, {plan.ao_elements, 2}) ||
          plan.adjoint_build_work > options.max_nuclear_affine_terms)
        return false;
      plan.reuse_right_adjoint = reserve_adjoint(true);
      return plan.reuse_right_adjoint;
    };
    if (planned &&
        adjoints::add_work(plan.adjoint_dot_work, {plan.ao_elements, fused ? 2UL : 3UL}) &&
        adjoints::add_work(plan.adjoint_dot_work, {shape.points, 5}) &&
        plan.adjoint_build_work <= options.max_nuclear_affine_terms &&
        plan.adjoint_dot_work <= options.max_nuclear_affine_terms &&
        (!factored ||
         plan.adjoint_dot_work <= options.max_nuclear_affine_terms - plan.adjoint_build_work) &&
        admit_adjoint_storage()) {
      plan.adjoint_bytes = scratch_bytes;
      plan.adjoint_tile = admitted_tile;
      plan.fused_adjoint = fused;
      const auto before = elements;
      // Packed rows are cheap in the small, fused Cartesian domain. Keep the
      // original masked scan when the adjoint cannot spare the extra storage.
      if (fused && shape.atoms >= 12 && shape.atoms <= cuda_grid_tiled_cartesian_atoms &&
          (!reserve((shape.atoms * (shape.nao + 1) + 1) / 2, plan.cartesian_ao_owners) ||
           !fits())) {
        elements = before;
        plan.cartesian_ao_owners = 0;
      }
    } else {
      elements = mandatory_elements;
    }
  }
  if (factored && !plan.adjoint_bytes) return false;
  const auto retained_elements = elements;
  if (!plan.adjoint_bytes && options.nuclear_response && options.replay_nuclear_maps &&
      reserve(plan.ao_elements, plan.staged_left_ao) &&
      reserve(plan.ao_elements, plan.staged_mixed_ao) &&
      reserve(shape.points, plan.staged_left_weights) &&
      reserve(shape.points, plan.staged_mixed_weights) &&
      reserve(3 * shape.points, plan.staged_left_points) &&
      elements <= options.max_workspace_bytes / sizeof(double))
    plan.replay = true;
  else
    elements = retained_elements;
  if (options.max_point_linearization_bytes && shape.spins == 1 && shape.points >= 4096 &&
      !nonlocal && options.component == CudaWb97mvGridComponent::combined &&
      (code == semilocal_family_code(SemilocalFamily::Lda) ||
       code == semilocal_family_code(SemilocalFamily::Pbe) ||
       code == semilocal_family_code(SemilocalFamily::B3lyp) ||
       code == semilocal_family_code(SemilocalFamily::R2scan))) {
    const unsigned axes = code == semilocal_family_code(SemilocalFamily::R2scan) ? 5 : 4;
    const auto before = elements;
    std::size_t coefficients{};
    if (product(axes + 1, plan.fields, coefficients) &&
        reserve(coefficients, plan.point_linearization) &&
        reserve(shape.points, plan.point_replay_mask) && reserve(1, plan.point_probe_status) &&
        elements <= options.max_workspace_bytes / sizeof(double) &&
        elements - before <= options.max_point_linearization_bytes / sizeof(double))
      plan.point_linearization_axes = axes;
    else
      elements = before;
  }
  plan.bytes = elements * sizeof(double);
  return plan.bytes <= options.max_workspace_bytes;
}

bool layout(const CudaWb97mvGridShape& shape, const CudaWb97mvGridOptions& options,
            Layout& plan) noexcept {
  if (layout_impl(shape, options, plan, false)) return true;
  plan = {};
  return layout_impl(shape, options, plan, true);
}

std::size_t cartesian_first_batch(const Layout& plan, const CudaWb97mvGridOptions& options,
                                  std::size_t rows) {
  const auto workspace_rows = plan.adjoint_bytes / (adjoint_blocks * sizeof(double));
  const auto first =
      plan.factored_only
          ? std::min(workspace_rows, (options.max_nuclear_affine_terms - plan.adjoint_build_work) /
                                         plan.adjoint_dot_work)
          : workspace_rows;
  if (rows <= first) return rows;
  // The second batch reuses the live adjoint, so only its contractions are
  // charged against the work limit. Require paired weight planes in both
  // batches; a singleton uses the standalone three-plane layout instead.
  const auto second =
      plan.factored_only
          ? std::min(workspace_rows, options.max_nuclear_affine_terms / plan.adjoint_dot_work)
          : workspace_rows;
  const auto split = std::min(first, rows - 2);
  return split >= 2 && rows - split <= second ? split : 0;
}

bool interval(const void* pointer, std::size_t elements, std::uintptr_t& end) noexcept {
  const auto begin = reinterpret_cast<std::uintptr_t>(pointer);
  std::size_t bytes{};
  if (!pointer || begin % alignof(double) || !product(elements, sizeof(double), bytes) ||
      bytes > UINTPTR_MAX - begin)
    return false;
  end = begin + bytes;
  return true;
}

bool disjoint(const void* left, std::size_t first, const void* right, std::size_t second) noexcept {
  std::uintptr_t left_end{}, right_end{};
  return interval(left, first, left_end) && interval(right, second, right_end) &&
         (left_end <= reinterpret_cast<std::uintptr_t>(right) ||
          right_end <= reinterpret_cast<std::uintptr_t>(left));
}

cudaError_t device_buffer(const void* pointer, int device) noexcept {
  cudaPointerAttributes attributes{};
  auto error = cudaPointerGetAttributes(&attributes, pointer);
  if (error != cudaSuccess) return error;
  return attributes.type == cudaMemoryTypeDevice && attributes.device == device
             ? cudaSuccess
             : cudaErrorInvalidValue;
}

cudaError_t execution(cudaStream_t stream, int device, bool composable = false) noexcept {
  if (!stream || device < 0) return cudaErrorInvalidValue;
  int current = -1;
  auto error = cudaGetDevice(&current);
  if (error != cudaSuccess) return error;
  if (current != device) return cudaErrorInvalidDevice;
  cudaStreamCaptureStatus capture{};
  error = cudaStreamIsCapturing(stream, &capture);
  if (error != cudaSuccess) return error;
  return capture == cudaStreamCaptureStatusNone ||
                 (composable && capture == cudaStreamCaptureStatusActive)
             ? cudaSuccess
             : cudaErrorStreamCaptureUnsupported;
}

cudaError_t drain(cudaError_t error, cudaStream_t stream) noexcept {
  const auto fence = cudaStreamSynchronize(stream);
  return error == cudaSuccess ? fence : error;
}

cudaError_t numerical_status(int* status, cudaStream_t stream) noexcept {
  int host{};
  auto error = cudaMemcpyAsync(&host, status, sizeof(host), cudaMemcpyDeviceToHost, stream);
  error = drain(error, stream);
  return error != cudaSuccess ? error : host ? cudaErrorInvalidValue : cudaSuccess;
}

__global__ void check_matrix(const double* density, std::size_t nao, unsigned spins,
                             double absolute, double relative, int* status) {
  const auto matrix = nao * nao;
  for (std::size_t index = blockIdx.x * blockDim.x + threadIdx.x; index < matrix * spins;
       index += static_cast<std::size_t>(gridDim.x) * blockDim.x) {
    const auto element = index % matrix;
    const auto other = index / matrix * matrix + element % nao * nao + element / nao;
    const auto value = density[index], transposed = density[other];
    if (!isfinite(value) || !isfinite(transposed) ||
        fabs(value - transposed) > absolute + relative * fmax(fabs(value), fabs(transposed)))
      atomicExch(status, 1);
  }
}

__global__ void publish_private_response(const double* candidate, double* output, std::size_t count,
                                         const int* status) {
  for (std::size_t index = blockIdx.x * blockDim.x + threadIdx.x; index < count;
       index += static_cast<std::size_t>(gridDim.x) * blockDim.x)
    output[index] = *status ? nan("") : candidate[index];
}

__global__ void finish_private_mixed(const double* candidate, double* output, const int* status,
                                     int* transaction_status) {
  if (*status || !isfinite(*candidate)) {
    atomicOr(transaction_status, 1);
    *output = nan("");
  } else {
    *output = *transaction_status ? nan("") : *candidate;
  }
}

__global__ void record_private_nuclear_status(const int* status, int* transaction_status) {
  if (*status) atomicOr(transaction_status, cuda_grid_nuclear_xc_invalid);
}

__global__ void discover(const double* fields, const double* weights, const double* points,
                         std::size_t count, unsigned spins, std::size_t* active,
                         std::size_t* selected, int* status) {
  if (blockIdx.x || threadIdx.x) return;
  std::size_t size{};
  for (std::size_t point = 0; point < count; ++point) {
    double total{};
    for (unsigned spin = 0; spin < spins; ++spin) {
      const auto rho = fields[5 * spin * count + point];
      const auto tau = fields[(5 * spin + 4) * count + point];
      if (!isfinite(rho) || !isfinite(tau) || rho < 0 || tau < 0) atomicExch(status, 1);
      total += rho;
      for (unsigned axis = 1; axis < 4; ++axis)
        if (!isfinite(fields[(5 * spin + axis) * count + point])) atomicExch(status, 1);
    }
    if (!isfinite(total) || !isfinite(weights[point])) atomicExch(status, 1);
    for (unsigned axis = 0; axis < 3; ++axis)
      if (!isfinite(points[3 * point + axis])) atomicExch(status, 1);
    if (total >= ad::vv10_density_threshold && fabs(weights[point]) > ad::vv10_weight_threshold)
      active[size++] = point;
  }
  *selected = size;
}

/** Differentiate the B3LYP SCF gradient Gram map as well as its scalar model.
 * At an empty spin the generated point action is restricted to its tangent
 * subspace. A finite gradient direction out of that subspace is not admitted. */
__device__ void b3lyp_point(unsigned spins, const double physical_rho[2],
                            const double physical_gradient[2][3],
                            const double physical_delta_rho[2],
                            const double physical_delta_gradient[2][3], point::Value& base,
                            point::Value& tangent) {
  // Polarize only the point evaluator's inputs. The moving weight contraction
  // still pairs its restricted derivative with the caller's total direction.
  const double spin_scale = spins == 1 ? 0.5 : 1.0;
  double gradient[2][3]{}, delta_gradient[2][3]{};
  double features[5]{}, direction[5]{};
  for (unsigned spin = 0; spin < 2; ++spin) {
    const unsigned source = spins == 1 ? 0 : spin;
    features[spin] = spin_scale * physical_rho[source];
    direction[spin] = spin_scale * physical_delta_rho[source];
    for (unsigned axis = 0; axis < 3; ++axis) {
      gradient[spin][axis] = spin_scale * physical_gradient[source][axis];
      delta_gradient[spin][axis] = spin_scale * physical_delta_gradient[source][axis];
    }
  }
  for (unsigned spin = 0; spin < 2; ++spin) {
    if (!isfinite(features[spin]) || features[spin] < 0 || !isfinite(direction[spin]) ||
        (features[spin] == 0 && direction[spin] != 0))
      base.valid = false;
    for (unsigned axis = 0; axis < 3; ++axis)
      if (!isfinite(gradient[spin][axis]) || !isfinite(delta_gradient[spin][axis]) ||
          (features[spin] == 0 && (gradient[spin][axis] != 0 || delta_gradient[spin][axis] != 0)))
        base.valid = false;
  }
  for (unsigned axis = 0; axis < 3; ++axis) {
    const auto alpha = gradient[0][axis], beta = gradient[1][axis];
    const auto delta_alpha = delta_gradient[0][axis], delta_beta = delta_gradient[1][axis];
    features[2] += alpha * alpha;
    features[3] += alpha * beta;
    features[4] += beta * beta;
    direction[2] += 2 * alpha * delta_alpha;
    direction[3] += delta_alpha * beta + alpha * delta_beta;
    direction[4] += 2 * beta * delta_beta;
  }
  if (!base.valid || !isfinite(features[0] + features[1])) {
    base.valid = false;
    return;
  }
  for (unsigned slot = 2; slot < 5; ++slot) {
    if (!isfinite(features[slot]) || !isfinite(direction[slot])) {
      base.valid = false;
      return;
    }
  }
  const auto point = generated::b3lyp_device_direction(features, direction);
  const auto& partials = point.feature_derivative;
  const auto& action = point.directional_derivative;
  base.energy = point.energy_density;
  base.valid = isfinite(base.energy);
  for (unsigned spin = 0; spin < spins; ++spin) {
    base.rho[spin] = partials[spin];
    tangent.rho[spin] = action[spin];
    const auto same = 2 + 2 * spin, other = 1 - spin;
    for (unsigned axis = 0; axis < 3; ++axis) {
      base.gradient[spin][axis] =
          2 * partials[same] * gradient[spin][axis] + partials[3] * gradient[other][axis];
      tangent.gradient[spin][axis] =
          2 * (action[same] * gradient[spin][axis] + partials[same] * delta_gradient[spin][axis]) +
          action[3] * gradient[other][axis] + partials[3] * delta_gradient[other][axis];
    }
  }
}

/** Apply the SCF point potential and its physical directional action.
 * Shared grid maps expect five fields per spin; LDA/GGA have no tau partial.
 * In RKS the primal sees two equal spin densities but its derivative is with
 * respect to their sum; restricted_response takes the *total* direction.
 * Nuclear response also differentiates the quadrature measure, adding w' V
 * and emitting d(E_xc density) for the weight cotangent. */
template <bool restricted_pbe = false>
__global__ void semilocal_points(const double* fields, const double* direction,
                                 const double* weights, const double* delta_weights,
                                 std::size_t points, unsigned input_spins, SemilocalFamily family,
                                 double exchange_scale, double correlation_scale, bool response,
                                 double* partials, double* energy, int* status,
                                 const double* replay_mask = nullptr, int seed = -1) {
  const auto spins = restricted_pbe ? 1U : input_spins;
  const auto functional = restricted_pbe ? SemilocalFamily::Pbe : family;
  for (std::size_t lane = blockIdx.x * blockDim.x + threadIdx.x; lane < points;
       lane += static_cast<std::size_t>(gridDim.x) * blockDim.x) {
    if (replay_mask && !replay_mask[lane]) continue;
    double rho[2]{}, gradient[2][3]{}, delta_rho[2]{}, delta_gradient[2][3]{};
    for (unsigned spin = 0; spin < spins; ++spin) {
      rho[spin] = fields[5 * spin * points + lane];
      if (response)
        delta_rho[spin] =
            seed >= 0 ? static_cast<double>(seed == 0) : direction[5 * spin * points + lane];
      for (unsigned axis = 0; axis < 3; ++axis) {
        gradient[spin][axis] = fields[(5 * spin + 1 + axis) * points + lane];
        if (response)
          delta_gradient[spin][axis] = seed >= 0 ? static_cast<double>(seed == int(1 + axis))
                                                 : direction[(5 * spin + 1 + axis) * points + lane];
      }
    }
    point::Value base, tangent;
    if (functional == SemilocalFamily::B3lyp) {
      b3lyp_point(spins, rho, gradient, delta_rho, delta_gradient, base, tangent);
    } else {
      const bool pbe = functional == SemilocalFamily::Pbe;
      if (response)
        tangent =
            spins == 1
                ? point::restricted_response(pbe, rho[0], gradient[0], delta_rho[0],
                                             delta_gradient[0], exchange_scale, correlation_scale)
                : point::unrestricted_response(pbe, rho, gradient, delta_rho, delta_gradient,
                                               exchange_scale, correlation_scale);
      if (spins == 1) {
        rho[1] = (rho[0] *= 0.5);
        for (unsigned axis = 0; axis < 3; ++axis) gradient[1][axis] = (gradient[0][axis] *= 0.5);
      }
      if (!response || energy)
        base = point::evaluate(pbe, rho, gradient, exchange_scale, correlation_scale);
    }
    const double weight = weights ? weights[lane] : 1.0;
    const double delta_weight = response && delta_weights ? delta_weights[lane] : 0.0;
    if (!base.valid || !tangent.valid || !isfinite(weight) || !isfinite(delta_weight)) {
      atomicExch(status, 1);
      continue;
    }
    double energy_direction = 0;
    for (unsigned spin = 0; spin < spins; ++spin) {
      const double rho_partial = response
                                     ? weight * tangent.rho[spin] + delta_weight * base.rho[spin]
                                     : weight * base.rho[spin];
      partials[5 * spin * points + lane] = rho_partial;
      if (!isfinite(rho_partial)) atomicExch(status, 1);
      if (response && energy) energy_direction += base.rho[spin] * delta_rho[spin];
      for (unsigned axis = 0; axis < 3; ++axis) {
        const double gradient_partial = response ? weight * tangent.gradient[spin][axis] +
                                                       delta_weight * base.gradient[spin][axis]
                                                 : weight * base.gradient[spin][axis];
        partials[(5 * spin + 1 + axis) * points + lane] = gradient_partial;
        if (!isfinite(gradient_partial)) atomicExch(status, 1);
        if (response && energy)
          energy_direction += base.gradient[spin][axis] * delta_gradient[spin][axis];
      }
      partials[(5 * spin + 4) * points + lane] = 0;
    }
    if (energy) {
      const double weight_partial = response ? energy_direction : base.energy;
      partials[5 * spins * points + lane] = weight_partial;
      for (unsigned axis = 1; axis <= 3; ++axis) partials[(5 * spins + axis) * points + lane] = 0;
      if (!response) energy[lane] = weight * base.energy;
      if (!isfinite(weight_partial) || (!response && !isfinite(energy[lane])))
        atomicExch(status, 1);
    }
  }
}

// The grid response still calls the same checked PBE evaluator for low
// density and out-of-cache directions. Specialize only its dispatch so those
// frequent replays don't carry the unrelated UKS and B3LYP device states.
void launch_semilocal_points(const double* fields, const double* direction, const double* weights,
                             const double* delta_weights, std::size_t points, unsigned spins,
                             SemilocalFamily functional, double exchange_scale,
                             double correlation_scale, bool response, double* partials,
                             double* energy, int* status, cudaStream_t stream,
                             const double* replay_mask = nullptr, int seed = -1) {
  const auto blocks = maps::blocks(points);
  const auto launch = [&]<bool restricted_pbe>() {
    semilocal_points<restricted_pbe><<<blocks, 128, 0, stream>>>(
        fields, direction, weights, delta_weights, points, spins, functional, exchange_scale,
        correlation_scale, response, partials, energy, status, replay_mask, seed);
  };
  if (spins == 1 && functional == SemilocalFamily::Pbe)
    launch.template operator()<true>();
  else
    launch.template operator()<false>();
}

/** Contract the variational point in physical rho/gradient/tau coordinates.
 * TensorIR already puts the kinetic half in the feature map and its adjoint. */
template <bool response, std::uint32_t domain>
__global__ void stationary_points(const double* fields, const double* direction,
                                  const double* weights, const double* delta_weights,
                                  std::size_t points, unsigned spins, double* partials,
                                  double* energy, int* status, const double* replay_mask,
                                  int seed) {
  for (std::size_t lane = blockIdx.x * blockDim.x + threadIdx.x; lane < points;
       lane += static_cast<std::size_t>(gridDim.x) * blockDim.x) {
    if (replay_mask && !replay_mask[lane]) continue;
    double point[10], delta[10]{};
    const double spin_factor = spins == 1 ? 0.5 : 1.0;
    bool valid = true;
    for (unsigned spin = 0; spin < 2; ++spin) {
      const unsigned panel_spin = spins == 1 ? 0 : spin;
      const unsigned indices[] = {spin, 2 + 3 * spin, 3 + 3 * spin, 4 + 3 * spin, 8 + spin};
      for (unsigned feature = 0; feature < 5; ++feature) {
        const auto offset = (5 * panel_spin + feature) * points + lane;
        const auto index = indices[feature];
        point[index] = spin_factor * fields[offset];
        if constexpr (response)
          delta[index] = spin_factor * (seed >= 0 ? static_cast<double>(seed == int(feature))
                                                  : direction[offset]);
        valid = valid && isfinite(point[index]) && isfinite(delta[index]) &&
                ((feature != 0 && feature != 4) || point[index] >= 0);
      }
    }
    const double weight = weights ? weights[lane] : 1.0;
    const double dw = response && delta_weights ? delta_weights[lane] : 0.0;
    if (!valid || !isfinite(weight) || !isfinite(dw)) {
      atomicExch(status, 1);
      continue;
    }
    const auto value = [&] {
#if GENERATIVEQC_ENABLE_M062X_STATIONARY_POINT
      if constexpr (domain == kCudaM062XVariationalDomain) {
        if constexpr (response)
          return generated::m06_2x_stationary_action(point, delta);
        else
          return generated::m06_2x_stationary_value(point);
      } else
#endif
#if GENERATIVEQC_ENABLE_MN15_STATIONARY_POINT
          if constexpr (domain == kCudaMN15VariationalDomain) {
        if constexpr (response)
          return generated::mn15_stationary_action(point, delta);
        else
          return generated::mn15_stationary_value(point);
      } else
#endif
      {
        if constexpr (response)
          return generated::r2scan_stationary_action(point, delta);
        else
          return generated::r2scan_stationary_value(point);
      }
    }();
    for (unsigned spin = 0; spin < spins; ++spin) {
      const unsigned indices[] = {spin, 2 + 3 * spin, 3 + 3 * spin, 4 + 3 * spin, 8 + spin};
      const unsigned other[] = {1, 5, 6, 7, 9};
      for (unsigned feature = 0; feature < 5; ++feature) {
        const auto index = indices[feature];
        double base = value.potential[index];
        double tangent = 0;
        if constexpr (response) tangent = value.action[index];
        if (spins == 1) {
          base = 0.5 * (base + value.potential[other[feature]]);
          if constexpr (response) tangent = 0.5 * (tangent + value.action[other[feature]]);
        }
        const double coefficient = response ? weight * tangent + dw * base : weight * base;
        partials[(5 * spin + feature) * points + lane] = coefficient;
        if (!isfinite(coefficient)) atomicExch(status, 1);
      }
    }
    if (energy) {
      double measure = value.energy;
      if constexpr (response) {
        measure = 0;
        for (unsigned feature = 0; feature < 10; ++feature)
          measure += value.potential[feature] * delta[feature];
      } else {
        energy[lane] = weight * value.energy;
        if (!isfinite(energy[lane])) atomicExch(status, 1);
      }
      partials[5 * spins * points + lane] = measure;
      for (unsigned axis = 1; axis <= 3; ++axis) partials[(5 * spins + axis) * points + lane] = 0;
      if (!isfinite(measure)) atomicExch(status, 1);
    }
  }
}

void launch_stationary_points(const double* fields, const double* direction, const double* weights,
                              const double* delta_weights, CudaWb97mvGridShape shape,
                              std::uint32_t code, bool response, double* partials, double* energy,
                              int* status, cudaStream_t stream, const double* replay_mask = nullptr,
                              int seed = -1) {
  const auto blocks = maps::blocks(shape.points);
  const auto launch = [&]<std::uint32_t domain>() {
    if (response)
      stationary_points<true, domain>
          <<<blocks, 128, 0, stream>>>(fields, direction, weights, delta_weights, shape.points,
                                       shape.spins, partials, energy, status, replay_mask, seed);
    else
      stationary_points<false, domain>
          <<<blocks, 128, 0, stream>>>(fields, nullptr, weights, nullptr, shape.points, shape.spins,
                                       partials, energy, status, replay_mask, seed);
  };
#if GENERATIVEQC_ENABLE_M062X_STATIONARY_POINT
  if (code == generated::kM062XFunctionalCode)
    return launch.template operator()<kCudaM062XVariationalDomain>();
#endif
#if GENERATIVEQC_ENABLE_MN15_STATIONARY_POINT
  if (code == generated::kMN15FunctionalCode)
    return launch.template operator()<kCudaMN15VariationalDomain>();
#endif
  (void)code;
  launch.template operator()<kCudaR2scanVariationalDomain>();
}

bool uses_stationary_point(std::uint32_t code) noexcept {
  return code == semilocal_family_code(SemilocalFamily::R2scan) ||
         cuda_split_hybrid_variational_domain(code) != 0;
}

/** Cache only regular rows. The original directional evaluator still owns
 * vacuum, extreme scaling and out-of-domain directions. A failed unit probe
 * is not a failed physical action; an unwritten probe stays NaN. */
__global__ void finish_point_linearization(const double* fields, std::size_t points, unsigned axes,
                                           double* coefficients) {
  for (std::size_t lane = blockIdx.x * blockDim.x + threadIdx.x; lane < points;
       lane += static_cast<std::size_t>(gridDim.x) * blockDim.x) {
    bool regular = fields[lane] >= 0x1p-60;
    for (unsigned field = 0; field < axes; ++field)
      regular = regular && fabs(fields[field * points + lane]) <= 0x1p32;
    for (unsigned entry = 0; entry < (axes + 1) * 5; ++entry)
      regular = regular && fabs(coefficients[entry * points + lane]) <= 0x1p250;
    if (!regular) coefficients[lane] = nan("");
  }
}

cudaError_t prepare_point_linearization(const CudaWb97mvGridShape& shape,
                                        const CudaWb97mvGridOptions& options, const double* base,
                                        unsigned axes, double* coefficients, int* probe_status,
                                        cudaStream_t stream) {
  const auto fields = 5 * shape.points;
  auto error = cudaMemsetAsync(probe_status, 0, sizeof(int), stream);
  if (error == cudaSuccess)
    error = cudaMemsetAsync(coefficients, 0xff, (axes + 1) * fields * sizeof(double), stream);
  if (error != cudaSuccess) return error;
  const auto code = options.functional_code;
  // Seed the existing compiled point JVP, not a second functional formula.
  for (int seed = -1; seed < int(axes); ++seed) {
    auto* output = coefficients + (seed + 1) * fields;
    if (uses_stationary_point(code))
      launch_stationary_points(base, nullptr, nullptr, nullptr, shape, code, seed >= 0, output,
                               nullptr, probe_status, stream, nullptr, seed);
    else
      launch_semilocal_points(base, nullptr, nullptr, nullptr, shape.points, 1,
                              semilocal_family_from_code(code), options.exchange_scale,
                              options.correlation_scale, seed >= 0, output, nullptr, probe_status,
                              stream, nullptr, seed);
    if ((error = cudaGetLastError()) != cudaSuccess) return error;
  }
  finish_point_linearization<<<maps::blocks(shape.points), 128, 0, stream>>>(base, shape.points,
                                                                             axes, coefficients);
  return cudaGetLastError();
}

/** Apply the retained physical point Jacobian and the independent weight
 * motion. Leave each unresolved row untouched for the checked scalar path. */
__global__ void apply_point_linearization(const double* coefficients, const double* direction,
                                          const double* weights, const double* delta_weights,
                                          std::size_t points, unsigned axes, double* partials,
                                          double* replay_mask) {
  for (std::size_t lane = blockIdx.x * blockDim.x + threadIdx.x; lane < points;
       lane += static_cast<std::size_t>(gridDim.x) * blockDim.x) {
    replay_mask[lane] = 1;
    if (!isfinite(coefficients[lane])) continue;
    const auto weight = weights[lane], delta_weight = delta_weights ? delta_weights[lane] : 0.0;
    bool regular = isfinite(weight) && fabs(delta_weight) <= 0x1p32;
    double delta[5], potential[5], measure{};
    for (unsigned axis = 0; axis < axes; ++axis) {
      delta[axis] = direction[axis * points + lane];
      regular = regular && fabs(delta[axis]) <= 0x1p32;
    }
    if (!regular) continue;
    for (unsigned field = 0; field < 5; ++field) {
      double tangent{};
      for (unsigned axis = 0; axis < axes; ++axis)
        tangent += coefficients[((axis + 1) * 5 + field) * points + lane] * delta[axis];
      const auto base = coefficients[field * points + lane];
      potential[field] = weight * tangent + delta_weight * base;
      if (delta_weights && field < axes) measure += base * delta[field];
      regular = regular && isfinite(potential[field]);
    }
    if (!regular || !isfinite(measure)) continue;
    for (unsigned field = 0; field < 5; ++field) partials[field * points + lane] = potential[field];
    if (delta_weights) {
      partials[5 * points + lane] = measure;
      for (unsigned field = 6; field < 9; ++field) partials[field * points + lane] = 0;
    }
    replay_mask[lane] = 0;
  }
}

cudaError_t launch_point_linearization(const CudaWb97mvGridShape& shape, unsigned axes,
                                       const double* coefficients, const double* direction,
                                       const double* weights, const double* delta_weights,
                                       double* partials, double* mask, cudaStream_t stream) {
  apply_point_linearization<<<maps::blocks(shape.points), 128, 0, stream>>>(
      coefficients, direction, weights, delta_weights, shape.points, axes, partials, mask);
  return cudaGetLastError();
}

/** Transform AO planes once per point tile, not for every trial rotation.
 * The native final-state coefficients are column-major [AO,MO]. */
__global__ void transition_mo_jets(std::size_t nao, std::size_t points, const double* ao,
                                   const double* mo, std::size_t begin, std::size_t count,
                                   double* jets) {
  const auto panel = nao * count;
  for (auto index = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x; index < 4 * panel;
       index += std::size_t(gridDim.x) * blockDim.x) {
    const auto orbital = index % nao;
    const auto point = begin + index / nao % count;
    const auto plane = index / panel;
    const auto* row = ao + (plane * points + point) * nao;
    double value{};
    for (std::size_t basis = 0; basis < nao; ++basis)
      value += row[basis] * mo[orbital * nao + basis];
    jets[index] = value;
  }
}

__global__ void pack_transition_tile(std::size_t nao, std::size_t points, unsigned axes,
                                     std::size_t occupied, std::size_t begin, std::size_t count,
                                     const double* jets, const double* coefficients,
                                     const double* weights, double* fields, double* tangents,
                                     int* refused) {
  const auto virtuals = nao - occupied;
  for (auto index = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x;
       index < count * occupied * virtuals; index += std::size_t(gridDim.x) * blockDim.x) {
    const auto local_point = index % count, rotation = index / count;
    const auto point = begin + local_point;
    const auto left = rotation / virtuals, right = occupied + rotation % virtuals;
    double row[4], column[4], features[5], dual[5]{};
    for (unsigned plane = 0; plane < 4; ++plane) {
      const auto offset = (plane * count + local_point) * nao;
      row[plane] = jets[offset + left];
      column[plane] = jets[offset + right];
    }
    ad::FixedPotential::transition(row, column, features);
    const auto weight = weights[point];
    bool valid = isfinite(weight);
    for (unsigned field = 0; field < axes; ++field)
      valid = valid && fabs(features[field]) <= 0x1p30;
    if (valid && weight != 0 && isfinite(coefficients[point])) {
      for (unsigned field = 0; field < axes; ++field) {
        double value{};
        for (unsigned axis = 0; axis < axes; ++axis)
          value += coefficients[((axis + 1) * 5 + field) * points + point] * features[axis];
        dual[field] = weight * value;
        valid = valid && fabs(dual[field]) <= std::numeric_limits<double>::max() / 4;
      }
    }
    if (!valid) atomicExch(refused, 1);
    for (unsigned field = 0; field < axes; ++field) {
      fields[index * axes + field] = valid ? features[field] : 0;
      tangents[index * axes + field] = valid ? dual[field] : 0;
    }
  }
}

__global__ void count_transition_corrections(std::size_t points, const double* coefficients,
                                             const double* weights, int* count) {
  for (auto point = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x; point < points;
       point += std::size_t(gridDim.x) * blockDim.x)
    if (weights[point] != 0 && !isfinite(coefficients[point])) atomicAdd(count, 1);
}

/** Replay only uncached, nonzero-weight points with the original directional
 * XC evaluator. Fields carry the single-leg feature; replay sees the actual
 * RKS perturbation. GEMM owns the common factor of four, so remove it here. */
template <SemilocalFamily family>
__global__ void correct_transition_tile(std::size_t points, unsigned axes, std::size_t rotations,
                                        std::size_t begin, std::size_t count, const double* base,
                                        const double* coefficients, const double* weights,
                                        const double* fields, double exchange, double correlation,
                                        double* tangents, int* refused) {
  for (auto index = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x; index < count * rotations;
       index += std::size_t(gridDim.x) * blockDim.x) {
    const auto point = begin + index % count;
    const auto weight = weights[point];
    if (weight == 0 || isfinite(coefficients[point])) continue;
    double action[5]{};
    bool valid = isfinite(weight);
    if constexpr (family == SemilocalFamily::R2scan) {
      constexpr unsigned alpha[]{0, 2, 3, 4, 8}, beta[]{1, 5, 6, 7, 9};
      double input[10], direction[10];
      for (unsigned field = 0; field < 5; ++field) {
        const auto half = 0.5 * base[field * points + point];
        input[alpha[field]] = input[beta[field]] = half;
        direction[alpha[field]] = direction[beta[field]] = 2 * fields[index * axes + field];
        valid = valid && isfinite(half) && ((field != 0 && field != 4) || half >= 0);
      }
      if (valid) {
        const auto value = generated::r2scan_stationary_action(input, direction);
        for (unsigned field = 0; field < 5; ++field) {
          action[field] = 0.5 * (value.action[alpha[field]] + value.action[beta[field]]);
          valid = valid && isfinite(value.potential[alpha[field]] + value.potential[beta[field]]);
        }
      }
    } else {
      double rho[2]{base[point]}, gradient[2][3]{}, delta_rho[2]{4 * fields[index * axes]},
          delta_gradient[2][3]{};
      for (unsigned axis = 0; axis < 3; ++axis) {
        gradient[0][axis] = base[(axis + 1) * points + point];
        delta_gradient[0][axis] = 4 * fields[index * axes + axis + 1];
      }
      point::Value primal, tangent;
      if constexpr (family == SemilocalFamily::B3lyp)
        b3lyp_point(1, rho, gradient, delta_rho, delta_gradient, primal, tangent);
      else
        tangent = point::restricted_response(true, rho[0], gradient[0], delta_rho[0],
                                             delta_gradient[0], exchange, correlation);
      valid = valid && primal.valid && tangent.valid;
      action[0] = tangent.rho[0];
      for (unsigned axis = 0; axis < 3; ++axis) action[axis + 1] = tangent.gradient[0][axis];
    }
    for (unsigned field = 0; field < axes; ++field) {
      const auto result = weight * action[field];
      valid = valid && isfinite(result);
      tangents[index * axes + field] = valid ? 0.25 * result : 0;
    }
    if (!valid) atomicExch(refused, 1);
  }
}

__global__ void pair_reduce(const double* values, const std::size_t* active, std::size_t row_begin,
                            std::size_t rows, std::size_t columns, std::size_t points,
                            unsigned spins, double* coefficients, int* status) {
  for (std::size_t index = blockIdx.x * blockDim.x + threadIdx.x; index < 4 * rows;
       index += static_cast<std::size_t>(gridDim.x) * blockDim.x) {
    const auto field = index / rows, row = index % rows;
    double value{};
    for (std::size_t column = 0; column < columns; ++column)
      value += values[(field * rows + row) * columns + column];
    for (unsigned spin = 0; spin < spins; ++spin) {
      const auto offset = (5 * spin + field) * points + active[row_begin + row];
      coefficients[offset] += value;
      if (!isfinite(coefficients[offset])) atomicExch(status, 1);
    }
  }
}

__global__ void local_beta(const double* weights, const std::size_t* active, std::size_t count,
                           std::size_t points, unsigned spins, double beta, double* coefficients,
                           int* status) {
  for (std::size_t index = blockIdx.x * blockDim.x + threadIdx.x; index < count;
       index += static_cast<std::size_t>(gridDim.x) * blockDim.x)
    for (unsigned spin = 0; spin < spins; ++spin) {
      const auto offset = 5 * spin * points + active[index];
      coefficients[offset] += beta * weights[active[index]];
      if (!isfinite(coefficients[offset])) atomicExch(status, 1);
    }
}

/** The symmetric second leg cancels the energy's half in first-leg partials.
 * Reduction is row-local and ordered; no floating-point atomic accumulation. */
__global__ void nuclear_pair_reduce(const double* values, const std::size_t* active,
                                    std::size_t row_begin, std::size_t rows, std::size_t columns,
                                    std::size_t points, unsigned spins, double* partials,
                                    double* energy, bool response, int* status) {
  for (std::size_t index = blockIdx.x * blockDim.x + threadIdx.x; index < 10 * rows;
       index += static_cast<std::size_t>(gridDim.x) * blockDim.x) {
    const auto field = index / rows, row = index % rows, point = active[row_begin + row];
    double value{};
    for (std::size_t column = 0; column < columns; ++column)
      value += values[(field * rows + row) * columns + column];
    if (!field) {
      if (!response) {
        energy[point] += value;
        if (!isfinite(energy[point])) atomicExch(status, 1);
      }
    } else if (field <= 5) {
      for (unsigned spin = 0; spin < spins; ++spin) {
        auto& target = partials[(5 * spin + field - 1) * points + point];
        target += value;
        if (!isfinite(target)) atomicExch(status, 1);
      }
    } else {
      auto& target = partials[(5 * spins + field - 6) * points + point];
      target += value;
      if (!isfinite(target)) atomicExch(status, 1);
    }
  }
}

/** Geometry vectors arrive point-major, whereas field cotangents are planar. */
__global__ void append_geometry(const double* weights, const double* points, std::size_t count,
                                unsigned spins, double* fields) {
  for (std::size_t point = blockIdx.x * blockDim.x + threadIdx.x; point < count;
       point += static_cast<std::size_t>(gridDim.x) * blockDim.x) {
    fields[5 * spins * count + point] = weights[point];
    for (unsigned axis = 0; axis < 3; ++axis)
      fields[(5 * spins + 1 + axis) * count + point] = points ? points[3 * point + axis] : 0.0;
  }
}

__device__ double adjoint_sum(double value) {
  __shared__ double terms[128];
  terms[threadIdx.x] = value;
  __syncthreads();
  for (unsigned stride = 64; stride; stride /= 2) {
    if (threadIdx.x < stride) terms[threadIdx.x] += terms[threadIdx.x + stride];
    __syncthreads();
  }
  return terms[0];
}
__global__ void contract_ao_adjoint(std::size_t ao_elements, std::size_t points,
                                    const double* first, const double* second,
                                    const double* left_coeff, const double* base_partial,
                                    const double* right_partial, const double* left_ao,
                                    const double* mixed_ao, const double* left_weights,
                                    const double* mixed_weights, const double* left_points,
                                    double* partials, int* status) {
  if (atomicAdd(status, 0)) return;
  const std::size_t start = blockIdx.x * blockDim.x + threadIdx.x;
  const auto stride = std::size_t(gridDim.x) * blockDim.x;
  double value = 0;
  for (std::size_t index = start; index < ao_elements; index += stride) {
    double left_value = first[index] * left_ao[index];
    if (left_coeff) left_value += left_coeff[index] * left_ao[index];
    value += left_value + second[index] * mixed_ao[index];
  }
  for (std::size_t point = start; point < points; point += stride) {
    value += right_partial[5 * points + point] * left_weights[point] +
             base_partial[5 * points + point] * mixed_weights[point];
    for (unsigned axis = 0; axis < 3; ++axis)
      value += right_partial[(6 + axis) * points + point] * left_points[3 * point + axis];
  }
  const auto sum = adjoint_sum(value);
  if (!threadIdx.x) partials[blockIdx.x] = sum;
}
__global__ void finish_ao_adjoint(const double* partials, double* output, int* status) {
  if (atomicAdd(status, 0)) return;
  double value = 0;
  for (unsigned index = threadIdx.x; index < adjoint_blocks; index += blockDim.x)
    value += partials[index];
  const auto sum = adjoint_sum(value);
  if (!threadIdx.x) {
    if (!isfinite(sum))
      atomicExch(status, 1);
    else
      *output = sum;
  }
}

/** Store each center's exact AO incidence, in AO order, for all later columns. */
__global__ void pack_cartesian_ao_owners(CudaWb97mvGridShape shape, const double* basis,
                                         unsigned* owners, const int* status) {
  if (*status) return;
  const auto* aos = basis + 3 * shape.atoms + 2 * shape.primitives;
  for (std::size_t atom = threadIdx.x; atom < shape.atoms; atom += blockDim.x) {
    auto* row = owners + atom * (shape.nao + 1);
    unsigned count{};
    for (unsigned ao = 0; ao < shape.nao; ++ao)
      if (aos[16 * ao] == atom) row[1 + count++] = ao;
    row[0] = count;
  }
}

template <bool sparse_rows, bool radial>
__global__ void contract_cartesian_ao_adjoint(CudaWb97mvGridShape shape,
                                              CudaGridGeometryCartesianView motion,
                                              const double* first, const double* second,
                                              const double* left_coeff, const double* base_partial,
                                              const double* right_partial, double* partials,
                                              const unsigned* owners, const int* status,
                                              const int* transaction) {
  if (*status || (transaction && *transaction)) return;
  const auto weight_planes = motion.rows > 1 ? 2 : 3;
  motion.left += blockIdx.y;
  motion.weights += blockIdx.y * (weight_planes * shape.points);
  partials += blockIdx.y * adjoint_blocks;
  const auto left = motion.left / 3, right = motion.right / 3;
  const auto left_axis = motion.left % 3, right_axis = motion.right % 3;
  const auto* offsets = motion.atom_offsets;
  const auto left_begin = offsets ? offsets[left] : left * motion.points_per_atom;
  const auto left_end = offsets ? offsets[left + 1] : left_begin + motion.points_per_atom;
  const auto right_begin = offsets ? offsets[right] : right * motion.points_per_atom;
  const auto right_end = offsets ? offsets[right + 1] : right_begin + motion.points_per_atom;
  const auto panel = shape.points * shape.nao;
  const auto* aos = motion.basis + 3 * shape.atoms + 2 * shape.primitives;
  const std::size_t start = blockIdx.x * blockDim.x + threadIdx.x;
  const auto stride = std::size_t(gridDim.x) * blockDim.x;
  double value{};
  // A unit column changes the point or the AO center by one, with opposite
  // signs. Derivative indices come from the compiler's spatial jet ordering.
  const auto accumulate = [&](std::size_t index, std::size_t point, std::size_t center, int sign) {
    const int mixed_sign =
        sign * (int(point >= right_begin && point < right_end) - int(center == right));
    double distance[3];
    if constexpr (radial)
      for (unsigned axis = 0; axis < 3; ++axis)
        distance[axis] = motion.radial_points[3 * point + axis] - motion.basis[3 * center + axis];
    const auto spatial = [&](unsigned jet) {
      return radial ? geometry_ad::s_radial_spatial(jet, distance, motion.spatial + index, panel)
                    : motion.spatial[jet * panel + index];
    };
    for (unsigned jet = 0; jet < 4; ++jet) {
      const auto entry = jet * panel + index;
      const auto first_jet = geometry_ad::cartesian_first_index(jet, left_axis);
      const auto left_value = sign * spatial(first_jet);
      const double left_terms = first[entry] * left_value;
      value += left_coeff ? left_terms + left_coeff[entry] * left_value : left_terms;
      if (mixed_sign) {
        const auto mixed_jet = geometry_ad::cartesian_mixed_index(jet, left_axis, right_axis);
        value += second[entry] * (mixed_sign * spatial(mixed_jet));
      }
    }
  };
  if constexpr (sparse_rows) {
    // A fixed AO center and its own moving points cancel exactly. Traverse
    // only the other-center AOs on this atom's points and this atom's AOs on
    // the other points. No distance or numerical screening is involved.
    for (std::size_t local = start; local < (left_end - left_begin) * shape.nao; local += stride) {
      const auto center = static_cast<std::size_t>(aos[16 * (local % shape.nao)]);
      if (center != left)
        accumulate(left_begin * shape.nao + local, left_begin + local / shape.nao, center, 1);
    }
    const auto* row = owners + left * (shape.nao + 1);
    const auto count = row[0];
    for (std::size_t local = start; local < (shape.points - (left_end - left_begin)) * count;
         local += stride) {
      auto point = local / count;
      if (point >= left_begin) point += left_end - left_begin;
      accumulate(point * shape.nao + row[1 + local % count], point, left, -1);
    }
  } else {
    for (std::size_t index = start; index < panel; index += stride) {
      const auto point = index / shape.nao;
      const auto center = static_cast<std::size_t>(aos[16 * (index % shape.nao)]);
      const int sign = int(point >= left_begin && point < left_end) - int(center == left);
      if (sign) accumulate(index, point, center, sign);
    }
  }
  for (std::size_t point = start; point < shape.points; point += stride) {
    value += right_partial[5 * shape.points + point] * motion.weights[point] +
             base_partial[5 * shape.points + point] *
                 motion.weights[(weight_planes - 1) * shape.points + point];
    if (point >= left_begin && point < left_end)
      value += right_partial[(6 + left_axis) * shape.points + point];
  }
  const auto sum = adjoint_sum(value);
  if (!threadIdx.x) partials[blockIdx.x] = sum;
}

__global__ void finish_cartesian_column(const double* partials, double* output, const int* status,
                                        int* transaction) {
  // Each row is private until the enclosing Hessian transaction commits.
  if (*status) {
    if (!threadIdx.x && !blockIdx.x) atomicOr(transaction, 1);
    return;
  }
  // A block-wide decision is needed before the reduction's barrier: another
  // row can set the transaction flag while this block is entering.
  __shared__ int failed;
  if (!threadIdx.x) failed = atomicAdd(transaction, 0);
  __syncthreads();
  if (failed) return;
  partials += blockIdx.x * adjoint_blocks;
  double value{};
  for (unsigned index = threadIdx.x; index < adjoint_blocks; index += blockDim.x)
    value += partials[index];
  const auto sum = adjoint_sum(value);
  if (!threadIdx.x) {
    if (!isfinite(sum))
      atomicOr(transaction, 1);
    else
      output[blockIdx.x] = sum;
  }
}

cudaError_t build_ao_adjoint(CudaWb97mvGridShape shape, const Layout& plan, double* arena,
                             const double* density, cudaStream_t stream) {
  const auto points = shape.points, panel = points * shape.nao;
  auto* scratch = arena + plan.adjoint_scratch;
  auto* status = reinterpret_cast<int*>(arena + plan.status);
  if (plan.fused_adjoint) {
    const auto launch = [&](auto reuse_right, std::size_t begin, std::size_t count) {
      adjoints::fused_ao<decltype(reuse_right)::value>
          <<<maps::blocks(count * shape.nao), 128, 0, stream>>>(
              {shape.nao, points}, begin, count, arena + plan.ao, arena + plan.right_ao, density,
              arena + plan.right_density, arena + plan.right_partial, arena + plan.base_partial,
              arena + plan.adjoint_first,
              arena + (reuse_right ? plan.adjoint_output_tile : plan.adjoint_second), status);
      return cudaGetLastError();
    };
    if (!plan.reuse_right_adjoint) return launch(std::false_type{}, 0, points);
    for (std::size_t begin = 0; begin < points;) {
      const auto count = std::min(plan.adjoint_tile, points - begin);
      auto error = launch(std::true_type{}, begin, count);
      if (error != cudaSuccess) return error;
      // Each output AO can read any right AO of its point. Commit only after
      // the whole point tile is done; the next tile reads disjoint points.
      const auto width = count * shape.nao * sizeof(double);
      error = cudaMemcpy2DAsync(arena + plan.right_ao + begin * shape.nao, panel * sizeof(double),
                                arena + plan.adjoint_output_tile, width, width, 4,
                                cudaMemcpyDeviceToDevice, stream);
      if (error != cudaSuccess) return error;
      begin += count;
    }
    return cudaSuccess;
  }
  for (std::size_t begin = 0; begin < points;) {
    const auto count = std::min(plan.adjoint_tile, points - begin);
    const adjoints::Extents extents{shape.nao, count};
    const auto offset = begin * shape.nao;
    const auto* ao = arena + plan.ao + offset;
    auto* right = arena + plan.right_ao + offset;
    const auto* first_partial = arena + plan.right_partial + begin;
    const auto* second_partial = arena + plan.base_partial + begin;
    auto* first = arena + plan.adjoint_first + offset;
    auto* second = arena + plan.adjoint_second + offset;
    auto* left = arena + plan.adjoint_output_tile;
    const auto tile_panel = count * shape.nao;
    auto error = adjoints::launch_feature_direction(
        extents, ao, ao + panel, ao + 2 * panel, ao + 3 * panel, first_partial + points,
        first_partial + 2 * points, first_partial + 3 * points, first_partial,
        first_partial + 4 * points, density, first, first + panel, first + 2 * panel,
        first + 3 * panel, scratch, plan.adjoint_bytes, status, stream);
    if (error != cudaSuccess) return error;
    error = adjoints::launch_feature_mixed(
        extents, ao, ao + panel, ao + 2 * panel, ao + 3 * panel, second_partial + points,
        second_partial + 2 * points, second_partial + 3 * points, second_partial,
        second_partial + 4 * points, right, right + panel, right + 2 * panel, right + 3 * panel,
        arena + plan.right_density, density, second, second + panel, second + 2 * panel,
        second + 3 * panel, left, left + tile_panel, left + 2 * tile_panel, left + 3 * tile_panel,
        scratch, plan.adjoint_bytes, status, stream);
    // Generated kernels must finish reading the original right tangent before
    // a left cotangent overwrites it. Both operations are on the same stream.
    if (error == cudaSuccess)
      error = cudaMemcpy2DAsync(right, panel * sizeof(double), left, tile_panel * sizeof(double),
                                tile_panel * sizeof(double), 4, cudaMemcpyDeviceToDevice, stream);
    if (error != cudaSuccess) return error;
    begin += count;
  }
  return cudaSuccess;
}

bool use_warp_features(const CudaWb97mvGridShape& shape) {
  return shape.nao >= 8 && shape.nao <= 32 && shape.points >= 4096;
}

unsigned feature_blocks(std::size_t points) {
  return static_cast<unsigned>(std::min<std::size_t>((points - 1) / 8 + 1, 65535));
}

bool use_wide_features(const CudaWb97mvGridShape& shape, const Layout& plan) {
  return shape.nao > 32 && shape.nao <= 96 && shape.points >= 4096 &&
         plan.matrix <= plan.tensor_bytes / (2 * sizeof(double));
}

// Wide variants keep two or three AO rows per lane. Transpose once per
// spin instead of duplicating large shared-memory density tiles per CTA.
__global__ void transpose_feature_densities(unsigned nao, const double* density,
                                            const double* direction, double* scratch) {
  const unsigned count = nao * nao;
  for (unsigned pair = blockIdx.x * blockDim.x + threadIdx.x; pair < count;
       pair += blockDim.x * gridDim.x) {
    const auto slot = (pair % nao) * nao + pair / nao;
    scratch[slot] = density[pair];
    if (direction) scratch[count + slot] = direction[pair];
  }
}

cudaError_t moving_features(const CudaWb97mvGridShape& shape, const Layout& plan, double* arena,
                            const double* ao, const double* density, const double* direction,
                            double* fields, cudaStream_t stream) noexcept {
  const auto panel = shape.points * shape.nao;
  auto* status = reinterpret_cast<int*>(arena + plan.status);
  for (unsigned spin = 0; spin < shape.spins; ++spin) {
    auto* field = fields + 5 * spin * shape.points;
    if (use_warp_features(shape)) {
      ad::moving_features<<<feature_blocks(shape.points), 256, 0, stream>>>(
          shape.points, static_cast<unsigned>(shape.nao), arena + plan.ao, ao,
          density + spin * plan.matrix, direction + spin * plan.matrix, field, status);
      const auto error = cudaGetLastError();
      if (error != cudaSuccess) return error;
      continue;
    }
    if (use_wide_features(shape, plan)) {
      const auto nao = static_cast<unsigned>(shape.nao);
      auto* scratch = arena + plan.tensor;
      transpose_feature_densities<<<maps::blocks(plan.matrix), 128, 0, stream>>>(
          nao, density + spin * plan.matrix, direction + spin * plan.matrix, scratch);
      auto error = cudaGetLastError();
      if (error != cudaSuccess) return error;
      if (nao <= 64)
        ad::wide_moving_features<<<feature_blocks(shape.points), 256, 0, stream>>>(
            shape.points, nao, arena + plan.ao, ao, scratch, scratch + plan.matrix, field, status);
      else
        ad::large_moving_features<<<feature_blocks(shape.points), 256, 0, stream>>>(
            shape.points, nao, arena + plan.ao, ao, scratch, scratch + plan.matrix, field, status);
      error = cudaGetLastError();
      if (error != cudaSuccess) return error;
      continue;
    }
    const auto error = nuclear_maps::launch_feature_direction(
        plan.nuclear_extents, arena + plan.ao, arena + plan.ao + panel, arena + plan.ao + 2 * panel,
        arena + plan.ao + 3 * panel, ao, ao + panel, ao + 2 * panel, ao + 3 * panel,
        direction + spin * plan.matrix, density + spin * plan.matrix, field + shape.points,
        field + 2 * shape.points, field + 3 * shape.points, field, field + 4 * shape.points,
        arena + plan.tensor, plan.tensor_bytes, status, stream);
    if (error != cudaSuccess) return error;
  }
  return cudaSuccess;
}

cudaError_t features(const CudaWb97mvGridShape& shape, const Layout& plan, double* arena,
                     const double* density, double* fields, cudaStream_t stream) noexcept {
  const auto panel = shape.points * shape.nao;
  auto* status = reinterpret_cast<int*>(arena + plan.status);
  for (unsigned spin = 0; spin < shape.spins; ++spin) {
    auto* field = fields + 5 * spin * shape.points;
    if (use_warp_features(shape)) {
      ad::fixed_features<<<feature_blocks(shape.points), 256, 0, stream>>>(
          shape.points, static_cast<unsigned>(shape.nao), arena + plan.ao, nullptr,
          density + spin * plan.matrix, nullptr, field, status);
      const auto error = cudaGetLastError();
      if (error != cudaSuccess) return error;
      continue;
    }
    if (use_wide_features(shape, plan)) {
      const auto nao = static_cast<unsigned>(shape.nao);
      auto* scratch = arena + plan.tensor;
      transpose_feature_densities<<<maps::blocks(plan.matrix), 128, 0, stream>>>(
          nao, density + spin * plan.matrix, nullptr, scratch);
      auto error = cudaGetLastError();
      if (error != cudaSuccess) return error;
      if (nao <= 64)
        ad::wide_fixed_features<<<feature_blocks(shape.points), 256, 0, stream>>>(
            shape.points, nao, arena + plan.ao, nullptr, scratch, nullptr, field, status);
      else
        ad::large_fixed_features<<<feature_blocks(shape.points), 256, 0, stream>>>(
            shape.points, nao, arena + plan.ao, nullptr, scratch, nullptr, field, status);
      error = cudaGetLastError();
      if (error != cudaSuccess) return error;
      continue;
    }
    const auto error = maps::launch_features(
        plan.extents, arena + plan.ao, arena + plan.ao + panel, arena + plan.ao + 2 * panel,
        arena + plan.ao + 3 * panel, density + spin * plan.matrix, field + shape.points,
        field + 2 * shape.points, field + 3 * shape.points, field, field + 4 * shape.points,
        arena + plan.tensor, plan.tensor_bytes, status, stream);
    if (error != cudaSuccess) return error;
  }
  return cudaSuccess;
}

bool use_fused_grid_potential(const CudaWb97mvGridShape& shape) {
  return shape.nao >= 8 && shape.nao <= 64 && shape.points >= 4096;
}

// Sums one canonical AO pair per block, writing both entries from the same
// FP64 reduction. The compiler owns the fused point product, not this schedule.
template <class Product>
__global__ void assemble_fused_grid_potential(CudaWb97mvGridShape shape, const double* ao,
                                              const double* motion, const double* cotangent,
                                              const double* tangent, double* output, int* status) {
  __shared__ int failed;
  if (!threadIdx.x) failed = atomicAdd(status, 0);
  __syncthreads();
  if (failed) return;
  const auto pair = blockIdx.x;
  const auto row = static_cast<unsigned>((sqrt(8.0 * pair + 1.0) - 1.0) * 0.5);
  const auto column = pair - row * (row + 1) / 2;
  const auto panel = shape.nao * shape.points;
  double total{};
  for (auto point = static_cast<std::size_t>(threadIdx.x); point < shape.points;
       point += blockDim.x) {
    double left[Product::ao_count], right[Product::ao_count], weights[Product::weight_count];
    for (unsigned slot = 0; slot < 4; ++slot) {
      const auto offset = slot * panel + point * shape.nao;
      left[slot] = ao[offset + row];
      right[slot] = ao[offset + column];
      if constexpr (Product::ao_count == 8) {
        left[4 + slot] = motion[offset + row];
        right[4 + slot] = motion[offset + column];
      }
    }
    for (unsigned field = 0; field < 5; ++field) {
      const auto offset = field * shape.points + point;
      weights[field] = cotangent[offset];
      if constexpr (Product::weight_count == 10) weights[5 + field] = tangent[offset];
    }
    total += Product::point(left, right, weights);
  }
  const auto value = adjoint_sum(total);
  if (!threadIdx.x) {
    if (!isfinite(value)) atomicExch(status, 1);
    output[row * shape.nao + column] = value;
    output[column * shape.nao + row] = value;
  }
}

template <class Product>
cudaError_t launch_fused_grid_potential(CudaWb97mvGridShape shape, const double* ao,
                                        const double* motion, const double* cotangent,
                                        const double* tangent, double* output, int* status,
                                        cudaStream_t stream) {
  assemble_fused_grid_potential<Product><<<shape.nao*(shape.nao + 1) / 2, 128, 0, stream>>>(
      shape, ao, motion, cotangent, tangent, output, status);
  return cudaGetLastError();
}

/** The point-contiguous tile stores each unweighted AO and transformed AO
 * once. The canonical pair reduction then streams both sets of coefficients. */
template <class Product>
__global__ void stage_grid_potential_tile(CudaWb97mvGridShape shape, const double* ao,
                                          const double* motion, const double* cotangent,
                                          const double* tangent, std::size_t begin,
                                          std::size_t points, double* tile, int* status) {
  const auto size = points * shape.nao;
  const auto panel = shape.points * shape.nao;
  for (auto index = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x; index < size;
       index += std::size_t(gridDim.x) * blockDim.x) {
    const auto point = begin + index % points;
    const auto orbital = index / points;
    double jets[Product::ao_count], weights[Product::weight_count], right[Product::ao_count];
    for (unsigned slot = 0; slot < 4; ++slot) {
      const auto offset = slot * panel + point * shape.nao + orbital;
      jets[slot] = ao[offset];
      if constexpr (Product::ao_count == 8) jets[slot + 4] = motion[offset];
    }
    for (unsigned field = 0; field < 5; ++field) {
      const auto offset = field * shape.points + point;
      weights[field] = cotangent[offset];
      if constexpr (Product::weight_count == 10) weights[field + 5] = tangent[offset];
    }
    Product::right_transform(jets, weights, right);
    for (unsigned slot = 0; slot < Product::ao_count; ++slot) {
      if (!isfinite(right[slot])) atomicExch(status, 1);
      tile[slot * size + index] = jets[slot];
      tile[(slot + Product::ao_count) * size + index] = right[slot];
    }
  }
}

template <class Product>
__global__ void reduce_grid_potential_tile(std::size_t nao, const double* tile, std::size_t points,
                                           bool first, double* output, int* status) {
  __shared__ int failed;
  if (!threadIdx.x) failed = atomicAdd(status, 0);
  __syncthreads();
  if (failed) return;
  const auto pair = blockIdx.x;
  const auto row = static_cast<unsigned>((sqrt(8.0 * pair + 1.0) - 1.0) * 0.5);
  const auto column = pair - row * (row + 1) / 2;
  const auto size = points * nao;
  double total{};
  for (auto point = std::size_t(threadIdx.x); point < points; point += blockDim.x) {
    double contribution{};
    for (unsigned slot = 0; slot < Product::ao_count; ++slot)
      contribution += tile[slot * size + row * points + point] *
                      tile[(slot + Product::ao_count) * size + column * points + point];
    total += contribution;
  }
  const auto sum = adjoint_sum(total);
  if (!threadIdx.x) {
    const auto value = first ? sum : output[row * nao + column] + sum;
    if (!isfinite(value)) atomicExch(status, 1);
    output[row * nao + column] = value;
    output[column * nao + row] = value;
  }
}

/** Reuse two AO rows and two transformed columns in registers. Keep the
 * incumbent per-point and 128-lane FP64 sums for each published pair. */
__global__ void reduce_fixed_potential_pair_tiles(std::size_t nao, const double* tile,
                                                  std::size_t points, bool first, double* output,
                                                  int* status) {
  if (__syncthreads_or(threadIdx.x == 0 && atomicAdd(status, 0))) return;
  const auto block = blockIdx.x;
  const auto row_tile = static_cast<unsigned>((sqrt(8.0 * block + 1.0) - 1.0) * 0.5);
  const auto row = 2 * row_tile;
  const auto column = 2 * (block - row_tile * (row_tile + 1) / 2);
  const auto size = points * nao;
  constexpr auto fields = ad::FixedPotential::ao_count;
  __shared__ double partial[4][128];
  double sums[4]{};
  for (auto point = std::size_t(threadIdx.x); point < points; point += blockDim.x) {
    double terms[4]{};
    for (unsigned field = 0; field < fields; ++field) {
      const auto* left = tile + field * size + row * points;
      const auto* right = tile + (fields + field) * size + column * points;
      const auto left0 = left[point], left1 = row + 1 < nao ? left[points + point] : 0.0;
      const auto right0 = right[point], right1 = column + 1 < nao ? right[points + point] : 0.0;
      terms[0] += left0 * right0;
      terms[1] += left0 * right1;
      terms[2] += left1 * right0;
      terms[3] += left1 * right1;
    }
    for (unsigned index = 0; index < 4; ++index) sums[index] += terms[index];
  }
  for (unsigned index = 0; index < 4; ++index) partial[index][threadIdx.x] = sums[index];
  __syncthreads();
  const auto pair = threadIdx.x / 32, lane = threadIdx.x % 32;
  const auto i = row + pair / 2, j = column + pair % 2;
  if (i >= nao || j >= nao || i < j) return;
  const auto* values = partial[pair];
  double sum = (values[lane] + values[lane + 64]) + (values[lane + 32] + values[lane + 96]);
  for (unsigned stride = 16; stride; stride /= 2) sum += __shfl_down_sync(0xffffffffU, sum, stride);
  if (!lane) {
    const auto value = first ? sum : output[i * nao + j] + sum;
    if (!isfinite(value)) atomicExch(status, 1);
    output[i * nao + j] = value;
    output[j * nao + i] = value;
  }
}

std::size_t potential_tile_points(CudaWb97mvGridShape shape, const Layout& plan,
                                  bool private_fixed_response = false) {
  // Above one AO strip, amortize the weighted transform across the partners
  // of each AO. Keep the direct contraction for narrow orbital panels.
  if (shape.nao <= 32 || shape.nao > 96 || shape.points < 4096) return 0;
  // Only private, fixed-grid CPKS consumes this wider tile. It needs half the
  // planes of a moving transform, so can cover twice the points in the same
  // lease. Keep public and nuclear-geometry action sum orders unchanged.
  const auto planes =
      private_fixed_response ? ad::FixedPotential::ao_count : ad::MovingPotential::ao_count;
  const auto span = 2 * planes * shape.nao * sizeof(double);
  const auto limit = potential_tile_limit * ad::MovingPotential::ao_count / planes;
  const auto points = std::min(limit, plan.tensor_bytes / span);
  return points < 128 ? 0 : points;
}

template <class Product>
cudaError_t launch_tiled_grid_potential(CudaWb97mvGridShape shape, std::size_t tile_points,
                                        const double* ao, const double* motion,
                                        const double* cotangent, const double* tangent,
                                        double* output, double* tile, int* status,
                                        cudaStream_t stream, bool private_fixed = false) {
  for (std::size_t begin = 0; begin < shape.points;) {
    const auto points = std::min(tile_points, shape.points - begin);
    stage_grid_potential_tile<Product><<<maps::blocks(points * shape.nao), 128, 0, stream>>>(
        shape, ao, motion, cotangent, tangent, begin, points, tile, status);
    auto error = cudaGetLastError();
    if (error != cudaSuccess) return error;
    if (private_fixed) {
      const auto ao_tiles = (shape.nao + 1) / 2;
      reduce_fixed_potential_pair_tiles<<<ao_tiles*(ao_tiles + 1) / 2, 128, 0, stream>>>(
          shape.nao, tile, points, begin == 0, output, status);
    } else {
      reduce_grid_potential_tile<Product><<<shape.nao*(shape.nao + 1) / 2, 128, 0, stream>>>(
          shape.nao, tile, points, begin == 0, output, status);
    }
    error = cudaGetLastError();
    if (error != cudaSuccess) return error;
    begin += points;
  }
  return cudaSuccess;
}

cudaError_t nuclear_buffers(const CudaWb97mvGridShape& shape, const Layout& plan,
                            const CudaWb97mvGridReference& reference, const void* workspace,
                            std::initializer_list<std::pair<const double*, std::size_t>> inputs,
                            double* output, std::size_t output_count) noexcept {
  const std::pair<const void*, std::size_t> retained[] = {{reference.basis, plan.basis_elements},
                                                          {reference.points, 3 * shape.points},
                                                          {reference.weights, shape.points},
                                                          {reference.density, plan.matrices},
                                                          {workspace, plan.bytes / sizeof(double)}};
  for (const auto& [pointer, count] : retained)
    if (output_count && !disjoint(output, output_count, pointer, count))
      return cudaErrorInvalidValue;
  for (const auto& [pointer, count] : inputs) {
    if ((output_count && !disjoint(output, output_count, pointer, count)) ||
        !disjoint(pointer, count, workspace, plan.bytes / sizeof(double)))
      return cudaErrorInvalidValue;
    const auto error = device_buffer(pointer, reference.device_id);
    if (error != cudaSuccess) return error;
  }
  return output_count ? device_buffer(output, reference.device_id) : cudaSuccess;
}
}  // namespace

cudaError_t cuda_wb97mv_grid_workspace(const CudaWb97mvGridShape& shape,
                                       const CudaWb97mvGridOptions& options,
                                       std::size_t& bytes) noexcept {
  Layout plan;
  bytes = 0;
  if (!layout(shape, options, plan)) return cudaErrorInvalidValue;
  bytes = plan.bytes;
  return cudaSuccess;
}

cudaError_t CudaWb97mvGridResponse::authenticate(cudaStream_t stream, bool composable) noexcept {
  if (!ready_ || stream != reference_.stream) return cudaErrorInvalidResourceHandle;
  auto error = execution(stream, reference_.device_id, composable);
  if (error != cudaSuccess) return error;
  error = reference_.authenticate(reference_.owner);
  if (error != cudaSuccess) {
    ready_ = false;
    nuclear_ready_ = false;
  }
  return error;
}

CudaWb97mvGridResponse::~CudaWb97mvGridResponse() {
  if (nuclear_graph_) (void)cudaGraphExecDestroy(nuclear_graph_);
  release_point_workspace();
}

void CudaWb97mvGridResponse::release_point_workspace() noexcept {
  if (!fixed_point_workspace_) return;
  (void)cudaStreamSynchronize(reference_.stream);
  (void)runtime::resource_cuda_free(std::exchange(fixed_point_workspace_, nullptr));
  fixed_point_bytes_ = 0;
  fixed_point_axes_ = 0;
}

cudaError_t CudaWb97mvGridResponse::prepare(const CudaWb97mvGridShape& shape,
                                            const CudaWb97mvGridOptions& options,
                                            const CudaWb97mvGridReference& reference,
                                            void* workspace, std::size_t workspace_bytes) noexcept {
  ready_ = false;
  nuclear_ready_ = false;
  if (nuclear_graph_) (void)cudaGraphExecDestroy(nuclear_graph_);
  nuclear_graph_ = nullptr;
  release_point_workspace();
  work_ = {};
  Layout plan;
  if (!layout(shape, options, plan) || workspace_bytes < plan.bytes || !reference.authenticate ||
      !reference.owner)
    return cudaErrorInvalidValue;
  auto error = execution(reference.stream, reference.device_id);
  if (error != cudaSuccess) return error;
  if ((error = reference.authenticate(reference.owner)) != cudaSuccess) return error;
  const void* inputs[] = {reference.basis, reference.points, reference.weights, reference.density};
  const std::size_t counts[] = {plan.basis_elements, 3 * shape.points, shape.points, plan.matrices};
  for (unsigned input = 0; input < 4; ++input) {
    if (!disjoint(workspace, plan.bytes / sizeof(double), inputs[input], counts[input]))
      return cudaErrorInvalidValue;
    if ((error = device_buffer(inputs[input], reference.device_id)) != cudaSuccess) return error;
  }
  if ((error = device_buffer(workspace, reference.device_id)) != cudaSuccess) return error;
  shape_ = shape;
  options_ = options;
  beta_ = options.revised_nonlocal ? revised_beta(options.revised_nonlocal->b) : ad::vv10_beta;
  reference_ = reference;
  workspace_ = workspace;
  work_.workspace_bytes = plan.bytes;
  auto* arena = static_cast<double*>(workspace);
  auto* status = reinterpret_cast<int*>(arena + plan.status);
  if ((error = cudaMemsetAsync(arena, 0, plan.bytes, reference.stream)) != cudaSuccess)
    return drain(error, reference.stream);
  check_matrix<<<maps::blocks(plan.matrices), 128, 0, reference.stream>>>(
      reference.density, shape.nao, shape.spins, options.symmetry_absolute_tolerance,
      options.symmetry_relative_tolerance, status);
  error = cudaGetLastError();
  if (error == cudaSuccess)
    error = hessian_aot::launch_ao(reference.basis, shape.atoms, shape.primitives, shape.nao,
                                   reference.points, shape.points, 4, arena + plan.ao, status,
                                   reference.stream);
  if (error == cudaSuccess)
    error = features(shape, plan, arena, reference.density, arena + plan.base, reference.stream);
  if (error != cudaSuccess) return drain(error, reference.stream);
  if (plan.cartesian_ao_owners) {
    pack_cartesian_ao_owners<<<1, 32, 0, reference.stream>>>(
        shape, reference.basis, reinterpret_cast<unsigned*>(arena + plan.cartesian_ao_owners),
        status);
    if ((error = cudaGetLastError()) != cudaSuccess) return drain(error, reference.stream);
  }
  if (options.includes_nonlocal()) {
    discover<<<1, 1, 0, reference.stream>>>(
        arena + plan.base, reference.weights, reference.points, shape.points, shape.spins,
        reinterpret_cast<std::size_t*>(arena + plan.active),
        reinterpret_cast<std::size_t*>(arena + plan.count), status);
    if ((error = cudaGetLastError()) != cudaSuccess) return drain(error, reference.stream);
    if ((error = cudaMemcpyAsync(&work_.active_nonlocal_points, arena + plan.count,
                                 sizeof(std::size_t), cudaMemcpyDeviceToHost, reference.stream)) !=
        cudaSuccess)
      return drain(error, reference.stream);
  }
  if ((error = numerical_status(status, reference.stream)) != cudaSuccess) return error;
  const auto active = work_.active_nonlocal_points;
  if (active && active > options.max_pair_evaluations / active) {
    work_.pair_budget_exceeded = true;
    return cudaErrorInvalidValue;
  }
  ready_ = true;
  if (options.nuclear_response) work_.nuclear_affine_terms = plan.reference_work;
  error = options.nuclear_response ? evaluate_nuclear(false) : evaluate(false);
  if (error == cudaSuccess && plan.point_linearization_axes) {
    error = prepare_point_linearization(
        shape, options, arena + plan.base, plan.point_linearization_axes,
        arena + plan.point_linearization, reinterpret_cast<int*>(arena + plan.point_probe_status),
        reference.stream);
    // Probes have their own status: a singular unit direction disables the
    // row rather than rejecting a physical action that could still be valid.
    error = drain(error, reference.stream);
  }
  if (error == cudaSuccess && plan.replay) error = build_nuclear_graph();
  if (error == cudaSuccess) error = authenticate(reference.stream);
  if (error != cudaSuccess) ready_ = false;
  return error;
}

cudaError_t CudaWb97mvGridResponse::retain_fixed_point_linearization(
    std::size_t max_bytes) noexcept {
  auto error = authenticate(reference_.stream);
  if (error != cudaSuccess) return error;
  if (options_.nuclear_response || fixed_point_workspace_ || options_.max_point_linearization_bytes)
    return cudaErrorInvalidValue;
  Layout baseline, candidate;
  if (!layout(shape_, options_, baseline)) return cudaErrorInvalidValue;
  auto policy = options_;
  policy.max_point_linearization_bytes = max_bytes;
  if (!layout(shape_, policy, candidate) || !candidate.point_linearization_axes) return cudaSuccess;
  const auto bytes = candidate.bytes - baseline.bytes;
  auto* arena = static_cast<double*>(workspace_);
  void* cache{};
  error = runtime::resource_cuda_malloc(&cache, bytes);
  if (error == cudaErrorMemoryAllocation) {
    (void)cudaGetLastError();
    return cudaSuccess;
  }
  if (error != cudaSuccess) return error;
  fixed_point_workspace_ = cache;
  const auto offset = baseline.bytes / sizeof(double);
  auto* coefficients = static_cast<double*>(cache);
  error = prepare_point_linearization(
      shape_, options_, arena + baseline.base, candidate.point_linearization_axes, coefficients,
      reinterpret_cast<int*>(coefficients + (candidate.point_probe_status - offset)),
      reference_.stream);
  error = drain(error, reference_.stream);
  if (error == cudaSuccess) error = authenticate(reference_.stream);
  if (error != cudaSuccess) {
    release_point_workspace();
    return error;
  }
  fixed_point_axes_ = candidate.point_linearization_axes;
  fixed_point_mask_offset_ = candidate.point_replay_mask - offset;
  fixed_point_bytes_ = bytes;
  work_.workspace_bytes += bytes;
  return cudaSuccess;
}

unsigned CudaWb97mvGridResponse::transition_axes() const noexcept {
  const auto code = options_.functional_code;
  if (code == semilocal_family_code(SemilocalFamily::R2scan)) return fixed_point_axes_ == 5 ? 5 : 0;
  const auto family = semilocal_family_from_code(code);
  return !uses_stationary_point(code) && fixed_point_axes_ == 4 &&
                 (family == SemilocalFamily::Pbe || family == SemilocalFamily::B3lyp)
             ? 4
             : 0;
}

cudaError_t CudaWb97mvGridResponse::enqueue_transition_census(int* count,
                                                              cudaStream_t stream) noexcept {
  if (auto error = authenticate(stream); error != cudaSuccess) return error;
  if (!count || !transition_axes() || shape_.points > std::numeric_limits<int>::max())
    return cudaErrorNotSupported;
  count_transition_corrections<<<maps::blocks(shape_.points), 128, 0, stream>>>(
      shape_.points, static_cast<const double*>(fixed_point_workspace_), reference_.weights, count);
  return cudaGetLastError();
}

cudaError_t CudaWb97mvGridResponse::enqueue_transition_tile(
    const double* mo, std::size_t occupied, std::size_t begin, std::size_t count, double* mo_jets,
    double* fields, double* tangents, bool correct, int* refused, cudaStream_t stream) noexcept {
  if (auto error = authenticate(stream); error != cudaSuccess) return error;
  Layout plan;
  if (!transition_axes() || shape_.spins != 1 || work_.active_nonlocal_points ||
      options_.nuclear_response || !mo || !mo_jets || !fields || !tangents || !refused ||
      !occupied || occupied >= shape_.nao || !count || count > shape_.points ||
      begin > shape_.points - count || !layout(shape_, options_, plan))
    return cudaErrorInvalidValue;
  const auto* arena = static_cast<const double*>(workspace_);
  transition_mo_jets<<<maps::blocks(4 * count * shape_.nao), 128, 0, stream>>>(
      shape_.nao, shape_.points, arena + plan.ao, mo, begin, count, mo_jets);
  auto error = cudaGetLastError();
  if (error != cudaSuccess) return error;
  const auto rotations = occupied * (shape_.nao - occupied);
  const auto blocks = maps::blocks(count * rotations);
  const auto* coefficients = static_cast<const double*>(fixed_point_workspace_);
  pack_transition_tile<<<blocks, 128, 0, stream>>>(shape_.nao, shape_.points, fixed_point_axes_,
                                                   occupied, begin, count, mo_jets, coefficients,
                                                   reference_.weights, fields, tangents, refused);
  error = cudaGetLastError();
  if (error != cudaSuccess || !correct) return error;
  const auto launch = [&]<SemilocalFamily family>() {
    correct_transition_tile<family><<<blocks, 128, 0, stream>>>(
        shape_.points, fixed_point_axes_, rotations, begin, count, arena + plan.base, coefficients,
        reference_.weights, fields, options_.exchange_scale, options_.correlation_scale, tangents,
        refused);
  };
  switch (semilocal_family_from_code(options_.functional_code)) {
    case SemilocalFamily::Pbe:
      launch.template operator()<SemilocalFamily::Pbe>();
      break;
    case SemilocalFamily::B3lyp:
      launch.template operator()<SemilocalFamily::B3lyp>();
      break;
    case SemilocalFamily::R2scan:
      launch.template operator()<SemilocalFamily::R2scan>();
      break;
    default:
      return cudaErrorNotSupported;
  }
  return cudaGetLastError();
}

cudaError_t CudaWb97mvGridResponse::evaluate(bool response, bool defer_status) noexcept {
  Layout plan;
  if (!layout(shape_, options_, plan)) return cudaErrorInvalidValue;
  auto* arena = static_cast<double*>(workspace_);
  auto* status = reinterpret_cast<int*>(arena + plan.status);
  auto* active = reinterpret_cast<std::size_t*>(arena + plan.active);
  const auto stream = reference_.stream;
  const auto failure = [=](cudaError_t error) {
    return defer_status ? error : drain(error, stream);
  };
  double* replay_mask = nullptr;
  const auto axes = fixed_point_workspace_ ? fixed_point_axes_ : plan.point_linearization_axes;
  if (response && axes) {
    const auto* coefficients = arena + plan.point_linearization;
    replay_mask = arena + plan.point_replay_mask;
    if (fixed_point_workspace_) {
      auto* cache = static_cast<double*>(fixed_point_workspace_);
      coefficients = cache;
      replay_mask = cache + fixed_point_mask_offset_;
    }
    const auto error = launch_point_linearization(
        shape_, axes, coefficients, arena + plan.direction, reference_.weights, nullptr,
        arena + plan.coefficients, replay_mask, stream);
    if (error != cudaSuccess) return failure(error);
  }
  if (options_.component == CudaWb97mvGridComponent::vv10_only) {
    // Pair and local-beta kernels accumulate into the point coefficients.
    const auto zero =
        cudaMemsetAsync(arena + plan.coefficients, 0, plan.fields * sizeof(double), stream);
    if (zero != cudaSuccess) return failure(zero);
  } else if (uses_stationary_point(options_.functional_code))
    launch_stationary_points(arena + plan.base, arena + plan.direction, reference_.weights, nullptr,
                             shape_, options_.functional_code, response, arena + plan.coefficients,
                             nullptr, status, stream, replay_mask);
  else if (options_.functional_code != semilocal_family_code(SemilocalFamily::Wb97mv))
    launch_semilocal_points(arena + plan.base, arena + plan.direction, reference_.weights, nullptr,
                            shape_.points, shape_.spins,
                            semilocal_family_from_code(options_.functional_code),
                            options_.exchange_scale, options_.correlation_scale, response,
                            arena + plan.coefficients, nullptr, status, stream, replay_mask);
  else if (shape_.spins == 1)
    ad::point_1<<<maps::blocks(shape_.points), 128, 0, stream>>>(
        arena + plan.base, arena + plan.direction, reference_.weights, shape_.points, response,
        arena + plan.coefficients, status);
  else
    ad::point_2<<<maps::blocks(shape_.points), 128, 0, stream>>>(
        arena + plan.base, arena + plan.direction, reference_.weights, shape_.points, response,
        arena + plan.coefficients, status);
  auto error = cudaGetLastError();
  if (error != cudaSuccess) return failure(error);
  const auto count = work_.active_nonlocal_points;
  auto& pairs = response ? work_.response_pairs : work_.reference_pairs;
  const auto work = static_cast<std::uint64_t>(count) * count;
  if (!defer_status && work > UINT64_MAX - pairs) return failure(cudaErrorInvalidValue);
  for (std::size_t row = 0; row < count; row += options_.pair_tile_points)
    for (std::size_t column = 0; column < count; column += options_.pair_tile_points) {
      const auto rows = std::min(count - row, options_.pair_tile_points);
      const auto columns = std::min(count - column, options_.pair_tile_points);
      if (const auto& revised = options_.revised_nonlocal)
        ad::rvv10_pairs<<<maps::blocks(rows * columns), 128, 0, stream>>>(
            arena + plan.base, arena + plan.direction, reference_.weights, reference_.points,
            shape_.points, shape_.spins, active, row, column, rows, columns, revised->b, revised->c,
            revised->coefficient, response, arena + plan.pairs, status);
      else
        ad::pairs<<<maps::blocks(rows * columns), 128, 0, stream>>>(
            arena + plan.base, arena + plan.direction, reference_.weights, reference_.points,
            shape_.points, shape_.spins, active, row, column, rows, columns, response,
            arena + plan.pairs, status);
      error = cudaGetLastError();
      if (error != cudaSuccess) return failure(error);
      if (!defer_status) pairs += rows * columns;
      pair_reduce<<<maps::blocks(4 * rows), 128, 0, stream>>>(arena + plan.pairs, active, row, rows,
                                                              columns, shape_.points, shape_.spins,
                                                              arena + plan.coefficients, status);
      if ((error = cudaGetLastError()) != cudaSuccess) return failure(error);
    }
  if (!response && count) {
    const auto beta =
        options_.revised_nonlocal ? beta_ * options_.revised_nonlocal->coefficient : beta_;
    local_beta<<<maps::blocks(count), 128, 0, stream>>>(reference_.weights, active, count,
                                                        shape_.points, shape_.spins, beta,
                                                        arena + plan.coefficients, status);
    if ((error = cudaGetLastError()) != cudaSuccess) return failure(error);
  }
  const auto panel = shape_.points * shape_.nao;
  for (unsigned spin = 0; spin < shape_.spins; ++spin) {
    auto* coefficients = arena + plan.coefficients + 5 * spin * shape_.points;
    auto* output = arena + (response ? plan.candidate : plan.reference) + spin * plan.matrix;
    if (const auto tile = potential_tile_points(shape_, plan, response && defer_status))
      error = launch_tiled_grid_potential<ad::FixedPotential>(
          shape_, tile, arena + plan.ao, nullptr, coefficients, nullptr, output,
          arena + plan.tensor, status, stream, response && defer_status);
    else if (use_fused_grid_potential(shape_))
      error = launch_fused_grid_potential<ad::FixedPotential>(
          shape_, arena + plan.ao, nullptr, coefficients, nullptr, output, status, stream);
    else
      error = maps::launch_potential(plan.extents, arena + plan.ao, arena + plan.ao + panel,
                                     arena + plan.ao + 2 * panel, arena + plan.ao + 3 * panel,
                                     coefficients + shape_.points, coefficients + 2 * shape_.points,
                                     coefficients + 3 * shape_.points, coefficients,
                                     coefficients + 4 * shape_.points, output, arena + plan.tensor,
                                     plan.tensor_bytes, status, stream);
    if (error != cudaSuccess) return failure(error);
  }
  check_matrix<<<maps::blocks(plan.matrices), 128, 0, stream>>>(
      arena + (response ? plan.candidate : plan.reference), shape_.nao, shape_.spins,
      options_.symmetry_absolute_tolerance, options_.symmetry_relative_tolerance, status);
  if ((error = cudaGetLastError()) != cudaSuccess) return failure(error);
  return defer_status ? cudaSuccess : numerical_status(status, stream);
}

cudaError_t CudaWb97mvGridResponse::evaluate_nuclear(bool response, bool defer_status,
                                                     bool assemble_potential) noexcept {
  Layout plan;
  if (!options_.nuclear_response || !layout(shape_, options_, plan)) return cudaErrorInvalidValue;
  const auto stream = reference_.stream;
  auto* arena = static_cast<double*>(workspace_);
  auto* status = reinterpret_cast<int*>(arena + plan.status);
  auto* active = reinterpret_cast<std::size_t*>(arena + plan.active);
  auto* partials = arena + (response ? plan.right_partial : plan.base_partial);
  const auto count = work_.active_nonlocal_points;
  const auto pair_count = static_cast<std::uint64_t>(count) * count;
  auto& pairs = response ? work_.nuclear_response_pairs : work_.nuclear_reference_pairs;
  if (pair_count > UINT64_MAX - pairs ||
      (!response && pair_count > UINT64_MAX - work_.reference_pairs))
    return cudaErrorInvalidValue;
  double* replay_mask = nullptr;
  if (response && plan.point_linearization_axes) {
    replay_mask = arena + plan.point_replay_mask;
    const auto error = launch_point_linearization(
        shape_, plan.point_linearization_axes, arena + plan.point_linearization,
        arena + plan.direction, reference_.weights, arena + plan.right_weights, partials,
        replay_mask, stream);
    if (error != cudaSuccess) return defer_status ? error : drain(error, stream);
  }
  if (options_.component == CudaWb97mvGridComponent::vv10_only) {
    // Clear all spin/weight/point cotangents, including screened-out rows.
    auto zero = cudaMemsetAsync(partials, 0, plan.full_fields * sizeof(double), stream);
    if (zero == cudaSuccess)
      zero = cudaMemsetAsync(arena + plan.energy, 0, shape_.points * sizeof(double), stream);
    if (zero != cudaSuccess) return drain(zero, stream);
  } else if (uses_stationary_point(options_.functional_code))
    launch_stationary_points(arena + plan.base, arena + plan.direction, reference_.weights,
                             arena + plan.right_weights, shape_, options_.functional_code, response,
                             partials, arena + plan.energy, status, stream, replay_mask);
  else if (options_.functional_code != semilocal_family_code(SemilocalFamily::Wb97mv))
    launch_semilocal_points(arena + plan.base, arena + plan.direction, reference_.weights,
                            arena + plan.right_weights, shape_.points, shape_.spins,
                            semilocal_family_from_code(options_.functional_code),
                            options_.exchange_scale, options_.correlation_scale, response, partials,
                            arena + plan.energy, status, stream, replay_mask);
  else if (shape_.spins == 1)
    nuclear_ad::point_1<<<maps::blocks(shape_.points), 128, 0, stream>>>(
        arena + plan.base, arena + plan.direction, reference_.weights, arena + plan.right_weights,
        shape_.points, response, partials, arena + plan.energy, status);
  else
    nuclear_ad::point_2<<<maps::blocks(shape_.points), 128, 0, stream>>>(
        arena + plan.base, arena + plan.direction, reference_.weights, arena + plan.right_weights,
        shape_.points, response, partials, arena + plan.energy, status);
  auto error = cudaGetLastError();
  if (error != cudaSuccess) return drain(error, stream);
  for (std::size_t row = 0; row < count; row += options_.pair_tile_points)
    for (std::size_t column = 0; column < count; column += options_.pair_tile_points) {
      const auto rows = std::min(count - row, options_.pair_tile_points);
      const auto columns = std::min(count - column, options_.pair_tile_points);
      if (const auto& revised = options_.revised_nonlocal)
        nuclear_ad::rvv10_pairs<<<maps::blocks(rows * columns), 128, 0, stream>>>(
            arena + plan.base, arena + plan.direction, reference_.weights, reference_.points,
            arena + plan.right_weights, arena + plan.right_points, shape_.points, shape_.spins,
            active, row, column, rows, columns, revised->b, revised->c, revised->coefficient,
            response, arena + plan.pairs, status);
      else
        nuclear_ad::pairs<<<maps::blocks(rows * columns), 128, 0, stream>>>(
            arena + plan.base, arena + plan.direction, reference_.weights, reference_.points,
            arena + plan.right_weights, arena + plan.right_points, shape_.points, shape_.spins,
            active, row, column, rows, columns, response, arena + plan.pairs, status);
      if ((error = cudaGetLastError()) != cudaSuccess) return drain(error, stream);
      pairs += rows * columns;
      if (!response) work_.reference_pairs += rows * columns;
      nuclear_pair_reduce<<<maps::blocks(10 * rows), 128, 0, stream>>>(
          arena + plan.pairs, active, row, rows, columns, shape_.points, shape_.spins, partials,
          arena + plan.energy, response, status);
      if ((error = cudaGetLastError()) != cudaSuccess) return drain(error, stream);
    }
  if (count) {
    if (const auto& revised = options_.revised_nonlocal)
      nuclear_ad::rvv10_local<<<maps::blocks(count), 128, 0, stream>>>(
          arena + plan.base, arena + plan.direction, reference_.weights, reference_.points,
          arena + plan.right_weights, arena + plan.right_points, shape_.points, shape_.spins,
          active, beta_, revised->coefficient, count, response, partials, arena + plan.energy,
          status);
    else
      nuclear_ad::local<<<maps::blocks(count), 128, 0, stream>>>(
          arena + plan.base, arena + plan.direction, reference_.weights, reference_.points,
          arena + plan.right_weights, arena + plan.right_points, shape_.points, shape_.spins,
          active, count, response, partials, arena + plan.energy, status);
    if ((error = cudaGetLastError()) != cudaSuccess) return drain(error, stream);
  }
  if (!assemble_potential) return cudaSuccess;
  const auto panel = shape_.points * shape_.nao;
  for (unsigned spin = 0; spin < shape_.spins; ++spin) {
    const auto* base = arena + plan.base_partial + 5 * spin * shape_.points;
    auto* output = arena + (response ? plan.candidate : plan.reference) + spin * plan.matrix;
    if (const auto tile = potential_tile_points(shape_, plan)) {
      if (response)
        error = launch_tiled_grid_potential<ad::MovingPotential>(
            shape_, tile, arena + plan.ao, arena + plan.right_ao, base,
            partials + 5 * spin * shape_.points, output, arena + plan.tensor, status, stream);
      else
        error = launch_tiled_grid_potential<ad::FixedPotential>(
            shape_, tile, arena + plan.ao, nullptr, base, nullptr, output, arena + plan.tensor,
            status, stream);
    } else if (use_fused_grid_potential(shape_)) {
      if (response)
        error = launch_fused_grid_potential<ad::MovingPotential>(
            shape_, arena + plan.ao, arena + plan.right_ao, base,
            partials + 5 * spin * shape_.points, output, status, stream);
      else
        error = launch_fused_grid_potential<ad::FixedPotential>(
            shape_, arena + plan.ao, nullptr, base, nullptr, output, status, stream);
    } else if (response) {
      const auto* right = partials + 5 * spin * shape_.points;
      const auto* motion = arena + plan.right_ao;
      error = nuclear_maps::launch_potential_direction(
          plan.nuclear_extents, arena + plan.ao, arena + plan.ao + panel,
          arena + plan.ao + 2 * panel, arena + plan.ao + 3 * panel, base + shape_.points,
          base + 2 * shape_.points, base + 3 * shape_.points, base, base + 4 * shape_.points,
          motion, motion + panel, motion + 2 * panel, motion + 3 * panel, right + shape_.points,
          right + 2 * shape_.points, right + 3 * shape_.points, right, right + 4 * shape_.points,
          output, arena + plan.tensor, plan.tensor_bytes, status, stream);
    } else {
      error = maps::launch_potential(
          plan.extents, arena + plan.ao, arena + plan.ao + panel, arena + plan.ao + 2 * panel,
          arena + plan.ao + 3 * panel, base + shape_.points, base + 2 * shape_.points,
          base + 3 * shape_.points, base, base + 4 * shape_.points, output, arena + plan.tensor,
          plan.tensor_bytes, status, stream);
    }
    if (error != cudaSuccess) return drain(error, stream);
  }
  if (!response) {
    error = nuclear_maps::launch_energy_sum(plan.nuclear_extents, arena + plan.energy,
                                            arena + plan.energy_scalar, arena + plan.tensor,
                                            plan.tensor_bytes, status, stream);
    if (error != cudaSuccess) return drain(error, stream);
  }
  check_matrix<<<maps::blocks(plan.matrices), 128, 0, stream>>>(
      arena + (response ? plan.candidate : plan.reference), shape_.nao, shape_.spins,
      options_.symmetry_absolute_tolerance, options_.symmetry_relative_tolerance, status);
  if ((error = cudaGetLastError()) != cudaSuccess) return drain(error, stream);
  return defer_status ? cudaSuccess : numerical_status(status, stream);
}

cudaError_t CudaWb97mvGridResponse::prepare_nuclear_direction(
    const double* ao, const double* weights, const double* points, const double* density,
    double* potential, cudaStream_t stream) noexcept {
  if (!potential) {
    nuclear_ready_ = false;
    return cudaErrorInvalidValue;
  }
  return prepare_nuclear_direction_impl(ao, weights, points, density, potential, nullptr, stream);
}

cudaError_t CudaWb97mvGridResponse::enqueue_private_nuclear_direction(
    const double* ao, const double* weights, const double* points, const double* density,
    double* potential, int* transaction_status, cudaStream_t stream) noexcept {
  if (!transaction_status || !potential) {
    nuclear_ready_ = false;
    return cudaErrorInvalidValue;
  }
  return prepare_nuclear_direction_impl(ao, weights, points, density, potential, transaction_status,
                                        stream);
}

cudaError_t CudaWb97mvGridResponse::enqueue_private_nuclear_covector(
    const double* ao, const double* weights, const double* points, const double* density,
    int* transaction_status, cudaStream_t stream) noexcept {
  if (!transaction_status) {
    nuclear_ready_ = false;
    return cudaErrorInvalidValue;
  }
  return prepare_nuclear_direction_impl(ao, weights, points, density, nullptr, transaction_status,
                                        stream);
}

cudaError_t CudaWb97mvGridResponse::prepare_nuclear_direction_impl(
    const double* ao, const double* weights, const double* points, const double* density,
    double* potential, int* transaction_status, cudaStream_t stream) noexcept {
  nuclear_ready_ = false;
  private_nuclear_status_ = nullptr;
  nuclear_adjoint_ready_ = false;
  auto error = authenticate(stream);
  if (error != cudaSuccess) return error;
  Layout plan;
  if (!options_.nuclear_response || !layout(shape_, options_, plan) ||
      work_.nuclear_directions == UINT64_MAX)
    return cudaErrorInvalidValue;
  const auto terms = potential ? plan.direction_work : plan.direction_feature_work;
  if (terms > UINT64_MAX - work_.nuclear_affine_terms) return cudaErrorInvalidValue;
  error = nuclear_buffers(shape_, plan, reference_, workspace_,
                          {{ao, plan.ao_elements},
                           {weights, shape_.points},
                           {points, 3 * shape_.points},
                           {density, plan.matrices}},
                          potential, potential ? plan.matrices : 0);
  if (error != cudaSuccess) return error;
  auto* arena = static_cast<double*>(workspace_);
  auto* status = reinterpret_cast<int*>(arena + plan.status);
  if ((error = cudaMemsetAsync(status, 0, sizeof(int), stream)) != cudaSuccess)
    return drain(error, stream);
  struct Copy {
    const double* input;
    std::size_t count, offset;
  };
  for (const auto& [input, count, offset] : {Copy{ao, plan.ao_elements, plan.right_ao},
                                             {weights, shape_.points, plan.right_weights},
                                             {points, 3 * shape_.points, plan.right_points},
                                             {density, plan.matrices, plan.right_density}}) {
    error = cudaMemcpyAsync(arena + offset, input, count * sizeof(double), cudaMemcpyDeviceToDevice,
                            stream);
    if (error != cudaSuccess) return drain(error, stream);
    maps::finite_input<<<maps::blocks(count), 128, 0, stream>>>(arena + offset, count, status);
    if ((error = cudaGetLastError()) != cudaSuccess) return drain(error, stream);
  }
  check_matrix<<<maps::blocks(plan.matrices), 128, 0, stream>>>(
      arena + plan.right_density, shape_.nao, shape_.spins, options_.symmetry_absolute_tolerance,
      options_.symmetry_relative_tolerance, status);
  if ((error = cudaGetLastError()) != cudaSuccess) return drain(error, stream);
  if (!transaction_status && (error = numerical_status(status, stream)) != cudaSuccess)
    return error;
  work_.nuclear_affine_terms += terms;
  error = moving_features(shape_, plan, arena, arena + plan.right_ao, reference_.density,
                          arena + plan.right_density, arena + plan.direction, stream);
  if (error != cudaSuccess) return drain(error, stream);
  if (!transaction_status && (error = numerical_status(status, stream)) != cudaSuccess)
    return error;
  error = evaluate_nuclear(true, transaction_status != nullptr, potential != nullptr);
  if (error != cudaSuccess) return error;
  if ((error = authenticate(stream)) != cudaSuccess) return error;
  if (transaction_status) {
    record_private_nuclear_status<<<1, 1, 0, stream>>>(status, transaction_status);
    if ((error = cudaGetLastError()) != cudaSuccess) return error;
    if (potential) {
      publish_private_response<<<maps::blocks(plan.matrices), 128, 0, stream>>>(
          arena + plan.candidate, potential, plan.matrices, transaction_status);
      error = cudaGetLastError();
    }
  } else {
    error = cudaMemcpyAsync(potential, arena + plan.candidate, plan.matrices * sizeof(double),
                            cudaMemcpyDeviceToDevice, stream);
    error = drain(error, stream);
  }
  if (error == cudaSuccess) {
    nuclear_ready_ = true;
    private_nuclear_status_ = transaction_status;
    ++work_.nuclear_directions;
  }
  return error;
}

cudaError_t CudaWb97mvGridResponse::enqueue_nuclear_mixed(
    const double* left_ao, const double* mixed_ao, const double* left_weights,
    const double* mixed_weights, const double* left_points, cudaStream_t stream) noexcept {
  Layout plan;
  if (!layout(shape_, options_, plan)) return cudaErrorInvalidValue;
  auto* arena = static_cast<double*>(workspace_);
  auto* status = reinterpret_cast<int*>(arena + plan.status);
  auto queued = moving_features(shape_, plan, arena, left_ao, reference_.density,
                                arena + plan.zero_density, arena + plan.left_fields, stream);
  if (queued != cudaSuccess) return queued;
  const auto panel = shape_.points * shape_.nao;
  const auto* right = arena + plan.right_ao;
  for (unsigned spin = 0; spin < shape_.spins; ++spin) {
    auto* fields = arena + plan.mixed_fields + 5 * spin * shape_.points;
    queued = nuclear_maps::launch_feature_mixed(
        plan.nuclear_extents, arena + plan.ao, arena + plan.ao + panel, arena + plan.ao + 2 * panel,
        arena + plan.ao + 3 * panel, right, right + panel, right + 2 * panel, right + 3 * panel,
        arena + plan.right_density + spin * plan.matrix, mixed_ao, mixed_ao + panel,
        mixed_ao + 2 * panel, mixed_ao + 3 * panel, reference_.density + spin * plan.matrix,
        left_ao, left_ao + panel, left_ao + 2 * panel, left_ao + 3 * panel, fields + shape_.points,
        fields + 2 * shape_.points, fields + 3 * shape_.points, fields, fields + 4 * shape_.points,
        arena + plan.tensor, plan.tensor_bytes, status, stream);
    if (queued != cudaSuccess) return queued;
  }
  append_geometry<<<maps::blocks(shape_.points), 128, 0, stream>>>(
      left_weights, left_points, shape_.points, shape_.spins, arena + plan.left_fields);
  if ((queued = cudaGetLastError()) != cudaSuccess) return queued;
  append_geometry<<<maps::blocks(shape_.points), 128, 0, stream>>>(
      mixed_weights, nullptr, shape_.points, shape_.spins, arena + plan.mixed_fields);
  if ((queued = cudaGetLastError()) != cudaSuccess) return queued;
  return nuclear_maps::launch_mixed_contraction(
      plan.nuclear_extents, arena + plan.base_partial, arena + plan.left_fields,
      arena + plan.mixed_fields, arena + plan.right_partial, arena + plan.scalar,
      arena + plan.tensor, plan.tensor_bytes, status, stream);
}

cudaError_t CudaWb97mvGridResponse::build_nuclear_graph() noexcept {
  Layout plan;
  if (!layout(shape_, options_, plan) || !plan.replay) return cudaErrorInvalidValue;
  auto* arena = static_cast<double*>(workspace_);
  cudaGraph_t graph{};
  const auto stream = reference_.stream;
  auto error = cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal);
  if (error == cudaSuccess) {
    const auto queued = enqueue_nuclear_mixed(
        arena + plan.staged_left_ao, arena + plan.staged_mixed_ao, arena + plan.staged_left_weights,
        arena + plan.staged_mixed_weights, arena + plan.staged_left_points, stream);
    const auto ended = cudaStreamEndCapture(stream, &graph);
    error = queued == cudaSuccess ? ended : queued;
    if (error == cudaSuccess) error = cudaGraphInstantiate(&nuclear_graph_, graph, 0);
  }
  if (graph) (void)cudaGraphDestroy(graph);
  if (error != cudaSuccess && error != cudaErrorMemoryAllocation &&
      error != cudaErrorNotSupported && error != cudaErrorStreamCaptureUnsupported)
    return drain(error, stream);
  if (error != cudaSuccess) {
    if (nuclear_graph_) (void)cudaGraphExecDestroy(nuclear_graph_);
    nuclear_graph_ = nullptr;
    (void)cudaGetLastError();
  }
  return cudaSuccess;
}

cudaError_t CudaWb97mvGridResponse::nuclear_mixed(const double* left_ao, const double* mixed_ao,
                                                  const double* left_weights,
                                                  const double* mixed_weights,
                                                  const double* left_points, double* output,
                                                  cudaStream_t stream) noexcept {
  return nuclear_mixed_impl(left_ao, mixed_ao, left_weights, mixed_weights, left_points, output,
                            nullptr, stream);
}

cudaError_t CudaWb97mvGridResponse::enqueue_private_nuclear_mixed(
    const double* left_ao, const double* mixed_ao, const double* left_weights,
    const double* mixed_weights, const double* left_points, double* output, int* transaction_status,
    cudaStream_t stream) noexcept {
  if (!transaction_status) return cudaErrorInvalidValue;
  return nuclear_mixed_impl(left_ao, mixed_ao, left_weights, mixed_weights, left_points, output,
                            transaction_status, stream);
}

bool CudaWb97mvGridResponse::admits_cartesian_mixed(std::size_t rows) const noexcept {
  Layout plan;
  if (!ready_ || !rows || rows > 3 * shape_.atoms || !layout(shape_, options_, plan) ||
      !plan.adjoint_bytes)
    return false;
  return rows == 1 || cartesian_first_batch(plan, options_, rows) != 0;
}

cudaError_t CudaWb97mvGridResponse::enqueue_private_cartesian_mixed(
    const CudaGridGeometryCartesianView& motion, double* output, int* transaction_status,
    cudaStream_t stream) noexcept {
  if (!transaction_status) return cudaErrorInvalidValue;
  return nuclear_mixed_impl(nullptr, nullptr, nullptr, nullptr, nullptr, output, transaction_status,
                            stream, &motion);
}

cudaError_t CudaWb97mvGridResponse::nuclear_mixed_impl(
    const double* left_ao, const double* mixed_ao, const double* left_weights,
    const double* mixed_weights, const double* left_points, double* output, int* transaction_status,
    cudaStream_t stream, const CudaGridGeometryCartesianView* cartesian) noexcept {
  auto error = authenticate(stream);
  if (error != cudaSuccess) return error;
  if (!nuclear_ready_ || (private_nuclear_status_ && private_nuclear_status_ != transaction_status))
    return cudaErrorInvalidResourceHandle;
  Layout plan;
  if (!layout(shape_, options_, plan) || work_.nuclear_contractions == UINT64_MAX ||
      work_.nuclear_graph_replays == UINT64_MAX)
    return cudaErrorInvalidValue;
  // The admitted wide plan uses a read-only adjoint even for checked callers.
  // Only the bound private transaction may repurpose its right AO panel.
  const bool use_adjoint = plan.factored_only || (private_nuclear_status_ && plan.adjoint_bytes);
  const auto rows = cartesian ? cartesian->rows : 1;
  if (!rows || rows > UINT64_MAX - work_.nuclear_contractions ||
      (use_adjoint && rows > UINT64_MAX - work_.nuclear_adjoint_contractions))
    return cudaErrorInvalidValue;
  const auto build_work = use_adjoint && !nuclear_adjoint_ready_ ? plan.adjoint_build_work : 0;
  const auto first_batch =
      cartesian && rows > 1 && use_adjoint ? cartesian_first_batch(plan, options_, rows) : 0;
  std::size_t contract_work{};
  if (!product(rows, use_adjoint ? plan.adjoint_dot_work : plan.mixed_work, contract_work) ||
      (plan.factored_only && !first_batch &&
       (build_work > options_.max_nuclear_affine_terms ||
        contract_work > options_.max_nuclear_affine_terms - build_work)) ||
      build_work > UINT64_MAX - work_.nuclear_affine_terms ||
      contract_work > UINT64_MAX - work_.nuclear_affine_terms - build_work)
    return cudaErrorInvalidValue;
  if (cartesian) {
    if (!use_adjoint || private_nuclear_status_ != transaction_status ||
        cartesian->basis != reference_.basis || cartesian->atoms != shape_.atoms ||
        (cartesian->radial_points && cartesian->radial_points != reference_.points) ||
        cartesian->left / 3 >= shape_.atoms || cartesian->right / 3 >= shape_.atoms ||
        rows > 3 * shape_.atoms - cartesian->left || (rows > 1 && !first_batch))
      return cudaErrorInvalidValue;
    const auto weight_planes = rows > 1 ? 2 : 3;
    error = nuclear_buffers(
        shape_, plan, reference_, workspace_,
        {{cartesian->spatial, (cartesian->radial_points ? 1 : 5) * plan.ao_elements},
         {cartesian->weights, rows * weight_planes * shape_.points}},
        output, rows);
  } else {
    error = nuclear_buffers(shape_, plan, reference_, workspace_,
                            {{left_ao, plan.ao_elements},
                             {mixed_ao, plan.ao_elements},
                             {left_weights, shape_.points},
                             {mixed_weights, shape_.points},
                             {left_points, 3 * shape_.points}},
                            output, 1);
  }
  if (error != cudaSuccess) return error;
  auto* arena = static_cast<double*>(workspace_);
  auto* status = reinterpret_cast<int*>(arena + plan.status);
  if ((error = cudaMemsetAsync(status, 0, sizeof(int), stream)) != cudaSuccess)
    return drain(error, stream);
  if (!cartesian)
    for (const auto& [input, count] : {std::pair{left_ao, plan.ao_elements},
                                       {mixed_ao, plan.ao_elements},
                                       {left_weights, shape_.points},
                                       {mixed_weights, shape_.points},
                                       {left_points, 3 * shape_.points}}) {
      maps::finite_input<<<maps::blocks(count), 128, 0, stream>>>(input, count, status);
      if ((error = cudaGetLastError()) != cudaSuccess) return drain(error, stream);
    }
  if (!transaction_status && (error = numerical_status(status, stream)) != cudaSuccess)
    return error;
  work_.nuclear_affine_terms += build_work + contract_work;
  if (use_adjoint) {
    if (!nuclear_adjoint_ready_) {
      error = build_ao_adjoint(shape_, plan, arena, reference_.density, stream);
      if (error != cudaSuccess) {
        nuclear_ready_ = false;  // The right panel may already hold some cotangent tiles.
        return drain(error, stream);
      }
      nuclear_adjoint_ready_ = true;
      ++work_.nuclear_adjoint_builds;
    }
    const auto* extra_left = plan.fused_adjoint ? nullptr : arena + plan.right_ao;
    if (cartesian) {
      // The adjoint generator has finished with its scratch on this stream.
      auto* partials = arena + (rows > 1 ? plan.adjoint_scratch : plan.adjoint_partials);
      const auto launch = [&](auto radial, const CudaGridGeometryCartesianView& batch) {
        constexpr bool compact = decltype(radial)::value;
        if (batch.rows > 1 && plan.cartesian_ao_owners)
          contract_cartesian_ao_adjoint<true, compact>
              <<<dim3(adjoint_blocks, batch.rows), 128, 0, stream>>>(
                  shape_, batch, arena + plan.adjoint_first, arena + plan.adjoint_second,
                  extra_left, arena + plan.base_partial, arena + plan.right_partial, partials,
                  reinterpret_cast<const unsigned*>(arena + plan.cartesian_ao_owners), status,
                  transaction_status);
        else
          contract_cartesian_ao_adjoint<false, compact>
              <<<dim3(adjoint_blocks, batch.rows), 128, 0, stream>>>(
                  shape_, batch, arena + plan.adjoint_first, arena + plan.adjoint_second,
                  extra_left, arena + plan.base_partial, arena + plan.right_partial, partials,
                  nullptr, status, batch.rows > 1 ? transaction_status : nullptr);
      };
      if (rows > 1) {
        for (std::size_t start = 0; start < rows;) {
          auto batch = *cartesian;
          batch.left += start;
          batch.rows = start ? rows - start : first_batch;
          batch.weights += 2 * start * shape_.points;
          if (batch.radial_points)
            launch(std::true_type{}, batch);
          else
            launch(std::false_type{}, batch);
          if ((error = cudaGetLastError()) != cudaSuccess) return drain(error, stream);
          if ((error = authenticate(stream)) != cudaSuccess) return error;
          finish_cartesian_column<<<batch.rows, 128, 0, stream>>>(partials, output + start, status,
                                                                  transaction_status);
          if ((error = cudaGetLastError()) != cudaSuccess) return drain(error, stream);
          work_.nuclear_contractions += batch.rows;
          work_.nuclear_adjoint_contractions += batch.rows;
          start += batch.rows;
        }
        return cudaSuccess;
      }
      if (cartesian->radial_points)
        launch(std::true_type{}, *cartesian);
      else
        launch(std::false_type{}, *cartesian);
    } else
      contract_ao_adjoint<<<adjoint_blocks, 128, 0, stream>>>(
          plan.ao_elements, shape_.points, arena + plan.adjoint_first, arena + plan.adjoint_second,
          extra_left, arena + plan.base_partial, arena + plan.right_partial, left_ao, mixed_ao,
          left_weights, mixed_weights, left_points, arena + plan.adjoint_partials, status);
    if ((error = cudaGetLastError()) != cudaSuccess) return drain(error, stream);
    finish_ao_adjoint<<<1, 128, 0, stream>>>(arena + plan.adjoint_partials, arena + plan.scalar,
                                             status);
    if ((error = cudaGetLastError()) != cudaSuccess) return drain(error, stream);
  }
  if (plan.replay && !use_adjoint) {
    for (const auto& [input, offset, count] :
         {std::tuple{left_ao, plan.staged_left_ao, plan.ao_elements},
          std::tuple{mixed_ao, plan.staged_mixed_ao, plan.ao_elements},
          std::tuple{left_weights, plan.staged_left_weights, shape_.points},
          std::tuple{mixed_weights, plan.staged_mixed_weights, shape_.points},
          std::tuple{left_points, plan.staged_left_points, 3 * shape_.points}}) {
      error = cudaMemcpyAsync(arena + offset, input, count * sizeof(double),
                              cudaMemcpyDeviceToDevice, stream);
      if (error != cudaSuccess) return drain(error, stream);
    }
    left_ao = arena + plan.staged_left_ao;
    mixed_ao = arena + plan.staged_mixed_ao;
    left_weights = arena + plan.staged_left_weights;
    mixed_weights = arena + plan.staged_mixed_weights;
    left_points = arena + plan.staged_left_points;
  }
  // Capture only immutable arithmetic/storage bindings. Authentication, input
  // validation and transactional publication remain outside the replay graph.
  if (!use_adjoint)
    error = nuclear_graph_ ? cudaGraphLaunch(nuclear_graph_, stream)
                           : enqueue_nuclear_mixed(left_ao, mixed_ao, left_weights, mixed_weights,
                                                   left_points, stream);
  if (error != cudaSuccess) return drain(error, stream);
  if (!transaction_status && (error = numerical_status(status, stream)) != cudaSuccess)
    return error;
  if ((error = authenticate(stream)) != cudaSuccess) return error;
  if (transaction_status) {
    finish_private_mixed<<<1, 1, 0, stream>>>(arena + plan.scalar, output, status,
                                              transaction_status);
    error = cudaGetLastError();
  } else {
    error = cudaMemcpyAsync(output, arena + plan.scalar, sizeof(double), cudaMemcpyDeviceToDevice,
                            stream);
    error = drain(error, stream);
  }
  if (error == cudaSuccess) {
    ++work_.nuclear_contractions;
    if (nuclear_graph_) ++work_.nuclear_graph_replays;
    if (use_adjoint) ++work_.nuclear_adjoint_contractions;
  }
  return error;
}

cudaError_t CudaWb97mvGridResponse::reference_energy(const double*& energy) noexcept {
  energy = nullptr;
  auto error = authenticate(reference_.stream);
  if (error != cudaSuccess) return error;
  Layout plan;
  if (!options_.nuclear_response || !layout(shape_, options_, plan)) return cudaErrorInvalidValue;
  energy = static_cast<double*>(workspace_) + plan.energy_scalar;
  return cudaSuccess;
}

cudaError_t CudaWb97mvGridResponse::apply(const double* direction, double* output,
                                          cudaStream_t stream) noexcept {
  return apply_impl(direction, output, stream, false);
}

cudaError_t CudaWb97mvGridResponse::enqueue_private_response(const double* direction,
                                                             double* output,
                                                             cudaStream_t stream) noexcept {
  return apply_impl(direction, output, stream, true);
}

cudaError_t CudaWb97mvGridResponse::apply_impl(const double* direction, double* output,
                                               cudaStream_t stream, bool private_panel) noexcept {
  nuclear_ready_ = false;
  auto error = authenticate(stream, private_panel);
  if (error != cudaSuccess) return error;
  Layout plan;
  if (!layout(shape_, options_, plan) || (!private_panel && work_.response_actions == UINT64_MAX))
    return cudaErrorInvalidValue;
  const void* inputs[] = {reference_.basis,   reference_.points, reference_.weights,
                          reference_.density, workspace_,        direction};
  const std::size_t counts[] = {plan.basis_elements, 3 * shape_.points,           shape_.points,
                                plan.matrices,       plan.bytes / sizeof(double), plan.matrices};
  for (unsigned input = 0; input < 6; ++input)
    if (!disjoint(output, plan.matrices, inputs[input], counts[input]))
      return cudaErrorInvalidValue;
  if (!disjoint(direction, plan.matrices, workspace_, plan.bytes / sizeof(double)))
    return cudaErrorInvalidValue;
  const auto fixed_point_elements = fixed_point_bytes_ / sizeof(double);
  if (fixed_point_elements &&
      (!disjoint(direction, plan.matrices, fixed_point_workspace_, fixed_point_elements) ||
       !disjoint(output, plan.matrices, fixed_point_workspace_, fixed_point_elements)))
    return cudaErrorInvalidValue;
  for (const auto* pointer : {direction, static_cast<const double*>(output)})
    if ((error = device_buffer(pointer, reference_.device_id)) != cudaSuccess) return error;
  auto* arena = static_cast<double*>(workspace_);
  auto* status = reinterpret_cast<int*>(arena + plan.status);
  const auto failure = [=](cudaError_t code) { return private_panel ? code : drain(code, stream); };
  if ((error = cudaMemsetAsync(status, 0, sizeof(int), stream)) != cudaSuccess)
    return failure(error);
  if (!private_panel) ++work_.response_actions;
  check_matrix<<<maps::blocks(plan.matrices), 128, 0, stream>>>(
      direction, shape_.nao, shape_.spins, options_.symmetry_absolute_tolerance,
      options_.symmetry_relative_tolerance, status);
  error = cudaGetLastError();
  if (error == cudaSuccess)
    error = features(shape_, plan, arena, direction, arena + plan.direction, stream);
  if (error != cudaSuccess) return failure(error);
  if (!private_panel && (error = numerical_status(status, stream)) != cudaSuccess) return error;
  if ((error = evaluate(true, private_panel)) != cudaSuccess) return error;
  if ((error = authenticate(stream, private_panel)) != cudaSuccess) return failure(error);
  if (private_panel) {
    // Keep the status on device. The enclosing Fock transaction audits the
    // assembled result, so stale or partially computed grid data cannot pass.
    publish_private_response<<<maps::blocks(plan.matrices), 128, 0, stream>>>(
        arena + plan.candidate, output, plan.matrices, status);
    return cudaGetLastError();
  }
  error = cudaMemcpyAsync(output, arena + plan.candidate, plan.matrices * sizeof(double),
                          cudaMemcpyDeviceToDevice, stream);
  return drain(error, stream);
}

cudaError_t CudaWb97mvGridResponse::reference_potential(const double*& potential) noexcept {
  potential = nullptr;
  auto error = authenticate(reference_.stream);
  if (error != cudaSuccess) return error;
  Layout plan;
  if (!layout(shape_, options_, plan)) return cudaErrorInvalidValue;
  potential = static_cast<double*>(workspace_) + plan.reference;
  return cudaSuccess;
}
}  // namespace generativeqc::dft
