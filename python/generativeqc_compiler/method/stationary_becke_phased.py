"""Lower the shared Becke phase plan into the stationary owner's source panels.

The ordinary and composite owners use the same kernels and point ownership
contract. Allocation, capability checks, stream ordering and launches remain in
the native owner. Scalar science stays in the authoritative grid-response AD.
"""

from functools import lru_cache

from generativeqc_compiler.xc.becke_coefficients import emit_becke_pair_coefficients
from generativeqc_compiler.xc.becke_partition import (
    recognize_becke_partition_domain_graph,
)
from generativeqc_compiler.xc.grid_partition_ir import grid_partition_domain_program

_KERNELS = r"""
#include <cuda/atomic>
namespace generativeqc_stationary_cuda {
struct PhasedBeckeInput {
  generativeqc_grid_phased::Workspace work;
  const double* points;
  const double* centers;
  const int64_t* owners;
  size_t owner_offset, points_per_atom;
  double* seeds;
  const generativeqc_grid_adjoint::CenterPair* center_pairs;
  const uint2* indices;
  double* partial;
  int* error;
  __device__ size_t owner(size_t point) const {
    return owners ? size_t(owners[point])
        : (points_per_atom ? (owner_offset + point) / points_per_atom : size_t(-1));
  }
  __device__ bool failed() const {
    return cuda::atomic_ref<int, cuda::thread_scope_device>(*error)
        .load(cuda::memory_order_relaxed) != 0;
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

template <int Phase, bool Primitive = false>
__global__ void phased_becke_atom(PhasedBeckeInput input) {
  const size_t point = blockIdx.x * blockDim.x + threadIdx.x;
  const size_t atom = blockIdx.y;
  if (point >= input.work.points || input.failed()) return;
  using namespace generativeqc_grid_phased;
  bool valid = true;
  if constexpr (Phase == 0)
    valid = distance_phase(input.work, point, atom, input.points, input.centers, local_norm);
  if constexpr (Phase == 1) atom_logs_phase(input.work, point, atom);
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

template <bool Reverse, bool Primitive = false>
__global__ void phased_becke_pair(PhasedBeckeInput input) {
  const size_t point = blockIdx.x * blockDim.x + threadIdx.x;
  if (point >= input.work.points || input.failed()) return;
  using namespace generativeqc_grid_adjoint;
  using namespace generativeqc_grid_phased;
  const auto indices = input.indices[blockIdx.y];
  const PreparedCenterGeometry<decltype(&local_ratio_prepared)> geometry{
      input.center_pairs, local_ratio_prepared};
  bool valid;
  if constexpr (Reverse) {
    if constexpr (Primitive)
      valid = generativeqc_grid_coefficients::pair_coefficient_reverse_phase(
          input.work, point, indices.x, indices.y, geometry, local_log);
    else valid = pair_reverse_phase(input.work, point, indices.x, indices.y, geometry, local_log);
  } else
    valid = pair_primal_phase(input.work, point, indices.x, indices.y, geometry, local_log, local_becke);
  if (!valid) atomicExch(input.error, 1);
}

__global__ void phased_becke_normalize(PhasedBeckeInput input) {
  const size_t point = blockIdx.x * blockDim.x + threadIdx.x;
  if (point >= input.work.points || input.failed()) return;
  if (!generativeqc_grid_phased::point_normalize_phase(input.work, point,
          input.owner(point), input.seeds[point], local_ratio)) atomicExch(input.error, 1);
}

// Point lanes are contiguous in every global panel. Sixteen atom lanes share
// only the frozen maximum, products and ratio AD result; no scientific sum is
// reassociated and no point or atom scatter uses floating-point atomics.
__device__ void normalize_cooperative(PhasedBeckeInput input) {
  constexpr size_t point_lanes = stationary_becke_normalize_point_lanes;
  constexpr size_t atom_lanes = 128 / point_lanes;
  __shared__ double products[stationary_becke_normalize_max_atoms * point_lanes];
  __shared__ double maxima[atom_lanes * point_lanes];
  __shared__ double objectives[3 * point_lanes];
  __shared__ bool active_points[point_lanes];
  const size_t lane = threadIdx.x, atom_lane = threadIdx.y;
  const size_t point = blockIdx.x * point_lanes + lane;
  if (atom_lane == 0)
    active_points[lane] = point < input.work.points && !input.failed();
  __syncthreads();
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
        maximum = std::max(maximum, input.work.field(4, point)[atom]);
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
      const double product = normalized_product_value(input.work.field(4, point)[atom],
          input.work.zero_counts(point)[atom], maximum);
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
    for (size_t atom = atom_lane; atom < input.work.atoms; atom += atom_lanes)
      input.work.field(6, point)[atom] = normalized_product_bar(atom,
          input.owner(point), input.seeds[point], objective);
  }
}
__global__ void phased_becke_normalize_cooperative(PhasedBeckeInput input) {
  normalize_cooperative(input);
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
