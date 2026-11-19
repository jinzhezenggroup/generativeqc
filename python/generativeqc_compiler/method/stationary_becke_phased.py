"""Lower the shared Becke phase plan into the stationary owner's source panels.

The ordinary and composite owners use the same kernels and point ownership
contract. Allocation, capability checks, stream ordering and launches remain in
the native owner. Scalar science stays in the authoritative grid-response AD.
"""

from functools import lru_cache

from generativeqc_compiler.xc.becke_coefficients import emit_becke_pair_coefficients
from generativeqc_compiler.xc.becke_normalized_adjoint import (
    emit_becke_normalized_adjoint,
)
from generativeqc_compiler.xc.becke_partition import (
    recognize_becke_partition_domain_graph,
)
from generativeqc_compiler.xc.becke_scaled_products import emit_binary_scaled_becke
from generativeqc_compiler.xc.grid_partition_ir import grid_partition_domain_program

_KERNELS = r"""
#include <cuda/atomic>
namespace generativeqc_stationary_cuda {
struct PhasedBeckeInput {
  generativeqc_grid_phased::Workspace work;
  const double* points;
  const double* centers;
  // Immutable elemental radii, owned by the stationary topology. Null keeps
  // the legacy equal-radius partition; every pair phase uses the same graph.
  const double* atomic_radii;
  const int64_t* owners;
  size_t owner_offset, points_per_atom;
  double* seeds;
  const generativeqc_grid_adjoint::CenterPair* center_pairs;
  const uint2* indices;
  double* partial;
  int* error;
  bool normalized_adjoints{};
  bool binary_scaled_products{};
  bool zero_seed_elision{};
  unsigned long long* zero_seed_points{};
  __device__ size_t owner(size_t point) const {
    return owners ? size_t(owners[point])
        : (points_per_atom ? (owner_offset + point) / points_per_atom : size_t(-1));
  }
  __device__ bool failed() const {
    return cuda::atomic_ref<int, cuda::thread_scope_device>(*error)
        .load(cuda::memory_order_relaxed) != 0;
  }
  __device__ bool zero_seed(size_t point) const {
    return zero_seed_elision && seeds[point] == 0;
  }
};

// Layout descriptors, not scientific preparation: center partials continue to
// be refreshed by the existing topology/geometry owner at every geometry bind.
__global__ void phased_becke_indices(size_t atoms, uint2* indices) {
  const size_t first = blockIdx.x;
  for (size_t second = threadIdx.x; second < first; second += blockDim.x)
    indices[generativeqc_grid_adjoint::center_pair_index(first, second)] =
        make_uint2(first, second);
}

template <int Phase, bool Primitive = false, bool BinaryScaled = false>
__global__ void phased_becke_atom(PhasedBeckeInput input) {
  const size_t point = blockIdx.x * blockDim.x + threadIdx.x;
  const size_t atom = blockIdx.y;
  if (point >= input.work.points || input.failed()) return;
  using namespace generativeqc_grid_phased;
  bool valid = true;
  // Distance validation is never elided: zero cotangents must still reject a
  // nonfinite/coincident point. Gather initializes the skipped reverse output.
  if constexpr (Phase == 1)
    if (input.zero_seed(point)) return;
  if constexpr (Phase == 2) {
    if (input.zero_seed(point)) {
      for (size_t word = 0; word < 4; ++word) input.work.field(7 + word, point)[atom] = 0;
      return;
    }
  }
  if constexpr (Phase == 0)
    valid = distance_phase(input.work, point, atom, input.points, input.centers, local_norm);
  if constexpr (Phase == 1) {
    if constexpr (BinaryScaled)
      valid = generativeqc_grid_scaled::scaled_atom_products_phase<true>(input.work, point, atom);
    else atom_logs_phase(input.work, point, atom);
  }
  if constexpr (Phase == 2) {
    if constexpr (Primitive)
      generativeqc_grid_coefficients::atom_gather_coefficient_phase(
          input.work, point, atom, input.center_pairs);
    else atom_gather_phase(input.work, point, atom);
  }
  if constexpr (Phase == 3) {
    valid = point_motion_phase(input.work, point, atom, input.owner(point));
    // One point per admitted geometry lane. The existing all-source reduction
    // is the publication gate and runs only after this entire phase succeeds.
    for (size_t axis = 0; axis < 3; ++axis)
      input.partial[point * 9 * input.work.atoms + 6 * input.work.atoms + 3 * atom + axis] =
          input.work.field(8 + axis, point)[atom];
  }
  if (!valid) atomicExch(input.error, 1);
}

template <bool Reverse, bool Primitive = false, bool Normalized = false,
          bool BinaryScaled = false>
__global__ void phased_becke_pair(PhasedBeckeInput input) {
  static_assert(!BinaryScaled || (!Primitive && !Normalized));
  const size_t point = blockIdx.x * blockDim.x + threadIdx.x;
  if (point >= input.work.points || input.failed()) return;
  // These rows have no pair-panel consumer: logs/normalization/reverse skip
  // together, and gather writes exact zeros before point-motion publication.
  if (input.zero_seed(point)) return;
  using namespace generativeqc_grid_adjoint;
  using namespace generativeqc_grid_phased;
  const auto indices = input.indices[blockIdx.y];
  const PreparedCenterGeometry<decltype(&local_ratio_prepared)> geometry{
      input.center_pairs, local_ratio_prepared, input.atomic_radii};
  bool valid;
  if constexpr (Reverse) {
    if constexpr (Normalized)
      valid = generativeqc_grid_normalized::pair_reverse_phase(
          input.work, point, indices.x, indices.y, geometry, local_log);
    else if constexpr (Primitive)
      valid = generativeqc_grid_coefficients::pair_coefficient_reverse_phase(
          input.work, point, indices.x, indices.y, geometry, local_log);
    else if constexpr (BinaryScaled)
      valid = generativeqc_grid_scaled::scaled_pair_reverse_phase(
          input.work, point, indices.x, indices.y, geometry, local_log);
    else valid = pair_reverse_phase(input.work, point, indices.x, indices.y, geometry, local_log);
  } else {
    if constexpr (BinaryScaled)
      valid = generativeqc_grid_scaled::scaled_pair_primal_phase(
          input.work, point, indices.x, indices.y, geometry, local_log, local_becke);
    else valid = pair_primal_phase(input.work, point, indices.x, indices.y, geometry, local_log, local_becke);
  }
  if (!valid) atomicExch(input.error, 1);
}

// Analytic route (generativeqc_grid_phased::analytic_*_phase): forward-mode
// partition derivatives with the same point_pair geometry and no pair panel.
// Each (points, atom) block evaluates its atom's elemental adjustments against
// every neighbor once; larger owners keep the per-call adjustment.
template <bool Fallback, class Phase>
__device__ void analytic_atom_lane(PhasedBeckeInput input, size_t atom, Phase phase) {
  using namespace generativeqc_grid_adjoint;
  if constexpr (Fallback) {
    phase(PreparedCenterGeometry<decltype(&local_ratio_prepared)>{
        input.center_pairs, local_ratio_prepared, input.atomic_radii});
  } else {
    __shared__ double adjustments[stationary_becke_normalize_max_atoms];
    if (input.atomic_radii)
      for (size_t neighbor = threadIdx.x; neighbor < input.work.atoms; neighbor += blockDim.x)
        adjustments[neighbor] =
            neighbor == atom ? 0.0
                             : coordinate_adjustment(atom > neighbor ? atom : neighbor,
                                                     atom > neighbor ? neighbor : atom,
                                                     input.atomic_radii);
    __syncthreads();
    phase(generativeqc_grid_phased::LaneAdjustedGeometry<decltype(&local_ratio_prepared)>{
        input.center_pairs, local_ratio_prepared,
        input.atomic_radii ? adjustments : nullptr, atom});
  }
}

// Fallback keeps the per-call adjustment when one block cannot table every
// neighbor of its atom (radii with more than stationary_becke_normalize_max_atoms).
template <bool Fallback>
__global__ void analytic_becke_products(PhasedBeckeInput input) {
  const size_t point = blockIdx.x * blockDim.x + threadIdx.x;
  const size_t atom = blockIdx.y;
  const bool active =
      point < input.work.points && !input.failed() && !input.zero_seed(point);
  analytic_atom_lane<Fallback>(input, atom, [&](auto geometry) {
    if (active && !generativeqc_grid_phased::analytic_products_phase(
                      input.work, point, atom, geometry, local_log, local_becke))
      atomicExch(input.error, 1);
  });
}

// One warp per point: fixed lane-strided partial maxima/sums and xor folds.
__global__ void analytic_becke_normalize(PhasedBeckeInput input) {
  const size_t point = (blockIdx.x * size_t(blockDim.x) + threadIdx.x) / 32;
  const unsigned lane = threadIdx.x % 32;
  if (point >= input.work.points) return;  // Uniform across the warp.
  const bool stopped = __shfl_sync(0xffffffffU,
      lane == 0 && (input.failed() || input.zero_seed(point)), 0);
  if (stopped) return;
  const size_t atoms = input.work.atoms, owner = input.owner(point);
  const auto mantissa = input.work.field(4, point), exponent = input.work.field(6, point);
  const auto zeros = input.work.zero_counts(point);
  double largest = -INFINITY;
  for (size_t atom = lane; atom < atoms; atom += 32)
    if (!zeros[atom]) largest = fmax(largest, exponent[atom]);
  for (unsigned offset = 16; offset; offset /= 2)
    largest = fmax(largest, __shfl_xor_sync(0xffffffffU, largest, offset));
  if (owner >= atoms || !isfinite(input.seeds[point]) || !isfinite(largest)) {
    if (lane == 0) atomicExch(input.error, 1);
    return;
  }
  double sum = 0;
  for (size_t atom = lane; atom < atoms; atom += 32) {
    const double scaled = ldexp(mantissa[atom], int(exponent[atom] - largest));
    input.work.field(7, point)[atom] = scaled;
    const double product = zeros[atom] ? 0.0 : scaled;
    input.work.field(5, point)[atom] = product;
    sum += product;
  }
  for (unsigned offset = 16; offset; offset /= 2)
    sum += __shfl_xor_sync(0xffffffffU, sum, offset);
  if (lane == 0) {
    input.work.maximum[point] = 1 / sum;
    if (!(sum > 0) || !isfinite(sum)) atomicExch(input.error, 1);
  }
}

// Six 128-thread blocks per SM keep a 96-atom, 1024-point tile in one wave.
template <bool Fallback>
__global__ void __launch_bounds__(128, 6) analytic_becke_motion(PhasedBeckeInput input) {
  const size_t point = blockIdx.x * blockDim.x + threadIdx.x;
  const size_t atom = blockIdx.y;
  const bool inside = point < input.work.points && !input.failed();
  const size_t owner = inside ? input.owner(point) : 0;
  double* output = inside ? input.partial + point * 9 * input.work.atoms +
                                6 * input.work.atoms + 3 * atom
                          : nullptr;
  const bool quiet = inside && (input.zero_seed(point) || atom == owner);
  if (quiet)
    for (size_t axis = 0; axis < 3; ++axis) output[axis] = 0;
  analytic_atom_lane<Fallback>(input, atom, [&](auto geometry) {
    if (inside && !quiet &&
        !generativeqc_grid_phased::analytic_motion_phase(input.work, point, atom, owner,
                                                         input.seeds[point], geometry,
                                                         local_log, local_becke, output))
      atomicExch(input.error, 1);
  });
}

// One warp per point: the moving owner receives minus every other atom's motion.
__global__ void analytic_becke_owner(PhasedBeckeInput input) {
  const size_t point = (blockIdx.x * size_t(blockDim.x) + threadIdx.x) / 32;
  const unsigned lane = threadIdx.x % 32;
  if (point >= input.work.points) return;  // Uniform across the warp.
  const bool stopped = __shfl_sync(0xffffffffU, lane == 0 && input.failed(), 0);
  const size_t atoms = input.work.atoms, owner = input.owner(point);
  if (stopped || owner >= atoms) return;
  double* row = input.partial + point * 9 * atoms + 6 * atoms;
  for (size_t axis = 0; axis < 3; ++axis) {
    double sum = 0;
    for (size_t atom = lane; atom < atoms; atom += 32)
      if (atom != owner) sum += row[3 * atom + axis];
    for (unsigned offset = 16; offset; offset /= 2)
      sum += __shfl_xor_sync(0xffffffffU, sum, offset);
    if (lane == 0) row[3 * owner + axis] = -sum;
  }
}

template <bool BinaryScaled = false>
__global__ void phased_becke_normalize(PhasedBeckeInput input) {
  const size_t point = blockIdx.x * blockDim.x + threadIdx.x;
  if (point >= input.work.points || input.failed()) return;
  if (input.zero_seed(point)) {
    if (input.owner(point) >= input.work.atoms) atomicExch(input.error, 1);
    else atomicAdd(input.zero_seed_points, 1ULL);
    return;
  }
  if (BinaryScaled && input.normalized_adjoints) {
    atomicExch(input.error, 1);
    return;
  }
  const bool valid = BinaryScaled
      ? generativeqc_grid_scaled::scaled_point_normalize_phase(input.work, point,
          input.owner(point), input.seeds[point], local_ratio)
      : generativeqc_grid_phased::point_normalize_phase(input.work, point,
          input.owner(point), input.seeds[point], local_ratio);
  if (!valid) atomicExch(input.error, 1);
  if (input.normalized_adjoints)
    for (size_t atom = 0; atom < input.work.atoms; ++atom)
      generativeqc_grid_normalized::prepare_atom_weight(input.work, point, atom);
}

// Point lanes are contiguous in every global panel. Sixteen atom lanes share
// only the frozen maximum, products and ratio AD result; no scientific sum is
// reassociated and no point or atom scatter uses floating-point atomics.
template <bool BinaryScaled = false>
__device__ void normalize_cooperative(PhasedBeckeInput input) {
  constexpr size_t point_lanes = stationary_becke_normalize_point_lanes;
  constexpr size_t atom_lanes = 128 / point_lanes;
  __shared__ double products[stationary_becke_normalize_max_atoms * point_lanes];
  __shared__ double maxima[atom_lanes * point_lanes];
  __shared__ double objectives[3 * point_lanes];
  __shared__ bool active_points[point_lanes];
  __shared__ unsigned skipped_points[point_lanes];
  const size_t lane = threadIdx.x, atom_lane = threadIdx.y;
  const size_t point = blockIdx.x * point_lanes + lane;
  if (atom_lane == 0) {
    bool active = point < input.work.points && !input.failed();
    if (active && BinaryScaled &&
        (input.normalized_adjoints ||
         input.work.atoms > generativeqc_grid_scaled::maximum_scaled_atoms)) {
      atomicExch(input.error, 1);
      active = false;
    }
    skipped_points[lane] = 0;
    if (active && input.zero_seed(point)) {
      if (input.owner(point) >= input.work.atoms) atomicExch(input.error, 1);
      else skipped_points[lane] = 1;
      active = false;
    }
    active_points[lane] = active;
  }
  __syncthreads();
  // One atomic per cooperative block, outside the scientific ordered sums.
  // The cumulative count measures actual elision, not a launched-domain model.
  if (input.zero_seed_elision && lane == 0 && atom_lane == 0) {
    unsigned skipped = 0;
    for (size_t source = 0; source < point_lanes; ++source) skipped += skipped_points[source];
    if (skipped) atomicAdd(input.zero_seed_points, static_cast<unsigned long long>(skipped));
  }
  const bool active = active_points[lane];
  if (input.work.atoms > stationary_becke_normalize_max_atoms) {
    if (active && atom_lane == 0 &&
        !generativeqc_grid_phased::point_normalize_phase(input.work, point,
            input.owner(point), input.seeds[point], local_ratio)) atomicExch(input.error, 1);
    return;
  }
  using namespace generativeqc_grid_adjoint;
  double maximum = -std::numeric_limits<double>::infinity();
  if (active)
    for (size_t atom = atom_lane; atom < input.work.atoms; atom += atom_lanes)
      if (!input.work.zero_counts(point)[atom])
        maximum = std::max(maximum,
            input.work.field(BinaryScaled ? 7 : 4, point)[atom]);
  maxima[atom_lane * point_lanes + lane] = maximum;
  __syncthreads();
  if (atom_lane == 0) {
    for (size_t source = 1; source < atom_lanes; ++source)
      maximum = std::max(maximum, maxima[source * point_lanes + lane]);
    maxima[lane] = maximum;
    if (active && (!std::isfinite(maximum) || input.owner(point) >= input.work.atoms ||
                   !std::isfinite(input.seeds[point]))) {
      active_points[lane] = false;
      atomicExch(input.error, 1);
    }
    if (active_points[lane]) input.work.maximum[point] = maximum;
  }
  __syncthreads();
  if (active_points[lane]) {
    maximum = maxima[lane];
    for (size_t atom = atom_lane; atom < input.work.atoms; atom += atom_lanes) {
      const double product = BinaryScaled
          ? generativeqc_grid_scaled::atom_product(input.work, point, atom)
                .scaled_value(static_cast<int>(maximum))
          : normalized_product_value(input.work.field(4, point)[atom],
                input.work.zero_counts(point)[atom], maximum);
      if constexpr (BinaryScaled)
        generativeqc_grid_scaled::store_relative_exponent(input.work, point, atom, maximum);
      products[atom * point_lanes + lane] = product;
      input.work.field(5, point)[atom] = product;
    }
  }
  __syncthreads();
  if (active_points[lane] && atom_lane == 0) {
    const generativeqc_grid_phased::Strided<double> local_products{
        products + lane, point_lanes};
    const auto objective = normalized_product_objective(input.work.atoms,
        input.owner(point), local_products, local_ratio);
    for (size_t word = 0; word < 3; ++word)
      objectives[word * point_lanes + lane] = objective[word];
  }
  __syncthreads();
  if (active_points[lane]) {
    const std::array<double, 3> objective{objectives[lane],
        objectives[point_lanes + lane], objectives[2 * point_lanes + lane]};
    for (size_t atom = atom_lane; atom < input.work.atoms; atom += atom_lanes) {
      input.work.field(6, point)[atom] = normalized_product_bar(atom,
          input.owner(point), input.seeds[point], objective);
      if (input.normalized_adjoints)
        generativeqc_grid_normalized::prepare_atom_weight(input.work, point, atom);
    }
  }
}
template <bool BinaryScaled = false>
__global__ void phased_becke_normalize_cooperative(PhasedBeckeInput input) {
  normalize_cooperative<BinaryScaled>(input);
}
} // namespace generativeqc_stationary_cuda
"""


@lru_cache(maxsize=4, typed=True)
def emit_stationary_phased_becke_cuda(
    atom_limit: int = 128, *, iterations: int = 3, cooperative_normalize: bool = True
) -> str:
    """Bind dynamic native kernels to an authenticated bounded AD composition.

    Recognition is finite source-generation work. The native owner selects
    either ordinary phases or the qualification-only coefficient primitive;
    neither selection constructs graphs or recognizes them at force execution.
    Representation selection specializes kernels at compile time: the host
    launches the requested variant without adding scaled arithmetic to the
    incumbent kernel's register-allocation domain.
    The emitted bound is part of native admission, not just an annotation.
    """
    if type(cooperative_normalize) is not bool:
        raise ValueError("cooperative normalization selection must be boolean")
    operation = recognize_becke_partition_domain_graph(
        grid_partition_domain_program(atom_limit, iterations),
        atom_limit=atom_limit,
        iterations=iterations,
    )
    if operation is None:
        raise ValueError("stationary Becke domain does not match canonical AD")
    return (
        emit_becke_pair_coefficients(operation)
        + emit_becke_normalized_adjoint(operation)
        + emit_binary_scaled_becke()
        + "namespace generativeqc_stationary_cuda {\n"
        + f"constexpr size_t stationary_becke_primitive_max_atoms = {atom_limit};\n"
        + f"constexpr size_t stationary_becke_normalize_max_atoms = {min(atom_limit, 128)};\n"
        + "constexpr size_t stationary_becke_normalize_point_lanes = 8;\n"
        + "constexpr bool stationary_becke_cooperative_normalize = "
        + str(cooperative_normalize).lower()
        + ";\n"
        + "}\n"
        + _KERNELS
    )
