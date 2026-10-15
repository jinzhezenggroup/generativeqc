"""Bounded compensated reductions for complete-grid FP64 XC totals.

The point program and potential matrix are unchanged. Three low words survive
between point batches; a full warp reduces each channel without serializing
every point through a single lane.
"""


def emit_compensated_xc_totals() -> str:
    """Emit a fixed-warp, allocation-free reduction over compact tile channels.

    The launch contract is one full 32-thread warp. Each lane retains its own
    high/low pair, then merges both words of every participating subtree.
    The owner retains three high and three low words across batches; only the
    final fold publishes the ordinary three-double result ABI.
    """
    return r"""
__device__ inline void add_xc_total(double value, double& sum, double& correction) {
  const double next = __dadd_rn(sum, value);
  const double residual = fabs(sum) >= fabs(value)
      ? __dadd_rn(__dadd_rn(sum, -next), value)
      : __dadd_rn(__dadd_rn(value, -next), sum);
  correction = __dadd_rn(correction, residual);
  sum = next;
}

__global__ void accumulate_compensated_xc_totals(
    const double* points, I count, I tile_points, double* totals, int* error) {
  const I lane = threadIdx.x;
  for (I channel = 0; channel < 3; ++channel) {
    double sum = 0.0, correction = 0.0;
    for (I begin = 0; begin < count; begin += tile_points) {
      const I remaining = count - begin;
      const I tile_count = remaining < tile_points ? remaining : tile_points;
      const double* values = points + 3 * begin + channel * tile_count;
      for (I point = lane; point < tile_count; point += 32)
        add_xc_total(values[point], sum, correction);
    }
    for (int offset = 16; offset > 0; offset /= 2) {
      const double other_sum = __shfl_down_sync(0xffffffffU, sum, offset);
      const double other_correction = __shfl_down_sync(0xffffffffU, correction, offset);
      if (lane + offset < 32) {
        add_xc_total(other_sum, sum, correction);
        add_xc_total(other_correction, sum, correction);
      }
    }
    if (lane == 0) {
      add_xc_total(sum, totals[channel], totals[channel + 3]);
      add_xc_total(correction, totals[channel], totals[channel + 3]);
      finite(__dadd_rn(totals[channel], totals[channel + 3]), error, 3);
    }
  }
}

__global__ void fold_compensated_xc_totals(double* totals, int* error) {
  const I channel = threadIdx.x;
  if (channel < 3)
    totals[channel] = finite(__dadd_rn(totals[channel], totals[channel + 3]), error, 3);
}
"""
