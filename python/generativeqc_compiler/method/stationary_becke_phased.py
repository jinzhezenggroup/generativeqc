"""Lower the shared Becke phase plan into the stationary owner's source panels.

The ordinary and composite owners use the same kernels and point ownership
contract. Allocation, capability checks, stream ordering and launches remain in
the native owner. Scalar science stays in the authoritative grid-response AD.
"""

from generativeqc_compiler.xc.grid_phased import emit_phased_becke

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

template <int Phase>
__global__ void phased_becke_atom(PhasedBeckeInput input) {
  const size_t point = blockIdx.x * blockDim.x + threadIdx.x;
  const size_t atom = blockIdx.y;
  if (point >= input.work.points || input.failed()) return;
  using namespace generativeqc_grid_phased;
  bool valid = true;
  if constexpr (Phase == 0)
    valid = distance_phase(input.work, point, atom, input.points, input.centers, local_norm);
  if constexpr (Phase == 1) atom_logs_phase(input.work, point, atom, local_log);
  if constexpr (Phase == 2) atom_gather_phase(input.work, point, atom);
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

template <bool Reverse>
__global__ void phased_becke_pair(PhasedBeckeInput input) {
  const size_t point = blockIdx.x * blockDim.x + threadIdx.x;
  if (point >= input.work.points || input.failed()) return;
  using namespace generativeqc_grid_adjoint;
  using namespace generativeqc_grid_phased;
  const auto indices = input.indices[blockIdx.y];
  const PreparedCenterGeometry<decltype(&local_ratio_prepared)> geometry{
      input.center_pairs, local_ratio_prepared};
  bool valid;
  if constexpr (Reverse)
    valid = pair_reverse_phase(input.work, point, indices.x, indices.y, geometry, local_log);
  else
    valid = pair_primal_phase(input.work, point, indices.x, indices.y, geometry, local_log, local_becke);
  if (!valid) atomicExch(input.error, 1);
}

__global__ void phased_becke_normalize(PhasedBeckeInput input) {
  const size_t point = blockIdx.x * blockDim.x + threadIdx.x;
  if (point >= input.work.points || input.failed()) return;
  if (!generativeqc_grid_phased::point_normalize_phase(input.work, point,
          input.owner(point), input.seeds[point], local_ratio)) atomicExch(input.error, 1);
}
} // namespace generativeqc_stationary_cuda
"""


def emit_stationary_phased_becke_cuda() -> str:
    """Emit the plan's iteration domains and ordered source-panel consumption."""
    return emit_phased_becke() + _KERNELS
