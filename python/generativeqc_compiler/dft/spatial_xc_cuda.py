"""Shared compact matrix arithmetic and compensated spatial XC reduction.

Only the transport differs from native XC: the grid caches eleven row-major
point outputs. Scalar bilinears and the tiled matrix body have one owner.
"""

from .xc_contraction_cuda import emit_spatial_xc_panels, emit_xc_potential_core
from .xc_totals_cuda import emit_compensated_xc_totals


def emit_spatial_xc_assembly() -> str:
    """Adapt cached Cartesian coefficients to native half-weighted panels."""
    lines = [
        emit_compensated_xc_totals(),
        emit_xc_potential_core(),
        emit_spatial_xc_panels(),
        r"""
__device__ void spatial_potential_panels_body(unsigned ingredients, const double* values,
    const double* ao, const double* weights, I n, I count, double* work, int* error) {
  const I terms = ingredients & 8U ? 5 : ingredients & 2U ? 4 : 1;
  compact_potential_panels_body(ao, values, weights, n, count, 2, terms,
                                ingredients & 8U ? 4 : 1, work, error);
}

__global__ void spatial_potential_panels(unsigned ingredients, const double* values,
    const double* ao, const double* weights, I n, I count, double* work, int* error) {
  spatial_potential_panels_body(ingredients, values, ao, weights, n, count, work, error);
}

// Fixed warp order and two-word accumulation retain small signed contributions
// without one lane serializing the full tile. The six words are private scratch.
__global__ void xc_integrals_kernel(const double* values, const double* features,
    const double* weights, I count, double* totals, int* error) {
  const I lane = threadIdx.x;
  for (I channel = 0; channel < 3; ++channel) {
    double sum = 0.0, correction = 0.0;
    const double* input = channel == 0 ? values : features + (channel-1)*5*count;
    for (I point = lane; point < count; point += 32)
      add_xc_total(weights[point]*input[point], sum, correction);
    for (int offset = 16; offset > 0; offset /= 2) {
      const double other_sum = __shfl_down_sync(0xffffffffU, sum, offset);
      const double other_correction = __shfl_down_sync(0xffffffffU, correction, offset);
      if (lane + offset < 32) {
        add_xc_total(other_sum, sum, correction);
        add_xc_total(other_correction, sum, correction);
      }
    }
    if (lane == 0) {
      totals[channel] = finite(__dadd_rn(sum, correction), error, 3);
      totals[channel+3] = correction;
    }
  }
}
""",
    ]
    return "\n".join(lines)


def emit_spatial_xc_batches() -> str:
    """Batch the existing point ABI and compact bodies over immutable tiles."""
    from .xc_tile_batch_cuda import emit_xc_ordered_scatter

    return (
        emit_xc_ordered_scatter()
        + r"""
__global__ void spatial_batch_points(const CudaXcCompactTile* tiles, I capacity,
    int evaluator, bool restricted, const double* features, const double* weights,
    double* values, double* totals, const int* tile_errors, int* error) {
  const auto& tile = tiles[blockIdx.y];
  const I slot = tile.begin, count = tile.count;
  if (tile_errors[blockIdx.y]) atomicCAS(error, 0, tile_errors[blockIdx.y]);
  for (I point = I(blockIdx.x)*blockDim.x+threadIdx.x; point < capacity;
       point += I(blockDim.x)*gridDim.x) {
    double energy = 0.0, a = 0.0, b = 0.0;
    if (point < count) {
      const double* f = features+13*slot;
      auto* v = values+11*slot;
      const auto xc = evaluate_xc_point(evaluator, restricted, f, count, point);
      if (!xc.valid) atomicCAS(error, 0, 3);
      v[point] = finite(xc.energy, error, 3);
      for (int spin = 0; spin < 2; ++spin) {
        v[(1+spin)*count+point] = finite(xc.rho[spin], error, 3);
        for (int axis = 0; axis < 3; ++axis)
          v[(3+3*spin+axis)*count+point] = finite(xc.gradient[spin][axis], error, 3);
        v[(9+spin)*count+point] = finite(xc.kinetic[spin], error, 3);
      }
      const double weight = weights[slot+point];
      energy = weight*xc.energy; a = weight*f[point]; b = weight*f[5*count+point];
    }
    // Pad only reduction scratch, never mathematical point evaluations.
    totals[3*slot+point] = energy;
    totals[3*slot+capacity+point] = a;
    totals[3*slot+2*capacity+point] = b;
  }
}
__global__ void spatial_batch_panels(const CudaXcCompactTile* tiles, unsigned ingredients,
    const double* values, const double* ao, const double* weights, double* work, int* error) {
  const auto& tile = tiles[blockIdx.y];
  spatial_potential_panels_body(ingredients, values+11*tile.begin, ao+tile.ao_offset,
      weights+tile.begin, tile.active, tile.count, work+tile.work_offset, error);
}
template <class Arithmetic>
__global__ void spatial_batch_potential(const CudaXcCompactTile* tiles,
    const double* ao, const double* work, const double* weights, double* potential,
    I work_jets, int* error) {
  const auto& tile = tiles[blockIdx.z/2];
  const I columns = (tile.active+15)/16;
  if (blockIdx.x >= columns*(columns+1)/2) return;
  tiled_potential_body<true, Arithmetic, true>(ao+tile.ao_offset, work+tile.work_offset,
      tile.active, tile.count, work_jets, nullptr, potential+tile.potential_offset,
      nullptr, false, error, nullptr, tile.active, blockIdx.z%2, weights+tile.begin);
}
"""
    )
