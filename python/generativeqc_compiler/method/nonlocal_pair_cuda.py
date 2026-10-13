"""Emit bounded parallel VV10 pair traversal over the canonical scalar roots.

Native owners admit the numerical domain, exclusive output leases and stream.
The CUDA templates only schedule/reduce those already generated pair outputs.
"""

VV10_PARALLEL_PAIR_SCHEDULE_VERSION = "vv10-parallel-bilateral-fp64-v1"


def native_parallel_pair_cuda() -> str:
    """Return CUDA-only templates with O(N) caller-owned output scratch."""
    return (
        "// Schedule identity: "
        + VV10_PARALLEL_PAIR_SCHEDULE_VERSION
        + "\n"
        + r"""

#if defined(__CUDACC__)
// The output rows are borrowed as three raw feature sums until finalization.
// Geometry outputs may reuse the expired input gradient on the same stream.
// No allocation or mathematical approximation is part of this schedule.
struct ParallelPairInputs {
  std::size_t point_count;
  const double *points, *density, *omega, *kappa;
  const double *domega_drho, *domega_dsigma, *dkappa_drho, *weighted_density;
  const std::uint64_t *indices, *active_count;
  double coefficient, beta;
  const int* rejected;
};
struct ParallelPairOutputs {
  double *energy_terms, *vrho, *vsigma, *point_derivative, *weight_derivative;
  int* failed;
};
__device__ inline double* parallel_pair_partial(ParallelPairOutputs output, std::size_t row,
                                                unsigned field) {
  if (field == 0) return output.energy_terms + row;
  if (field == 1) return output.vrho + row;
  if (field == 2) return output.vsigma + row;
  return output.point_derivative + 3 * row + field - 3;
}
template <bool Geometry>
__global__ void parallel_pair_initialize(ParallelPairInputs input, ParallelPairOutputs output) {
  if (*input.rejected != 0) return;
  const auto row = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (row >= input.point_count) return;
  output.energy_terms[row] = output.vrho[row] = output.vsigma[row] = 0.0;
  if constexpr (Geometry) {
    output.point_derivative[3 * row] = output.point_derivative[3 * row + 1] =
        output.point_derivative[3 * row + 2] = 0.0;
    output.weight_derivative[row] = 0.0;
  }
}
// Each persistent warp owns an unordered pair of 128-row tiles. Four rows
// per lane amortize the partner reductions; diagonal tiles retain all ordered
// pairs. Cross-tile contributions reduce in FP64 then atomically accumulate
// both legs. This changes summation order, not pair formulas or active domain.
__device__ inline double parallel_pair_warp_sum(double value) {
  for (int shift = 16; shift; shift /= 2) value += __shfl_down_sync(0xffffffffU, value, shift);
  return value;
}
template <bool Geometry>
__global__ void parallel_pair_kernel(ParallelPairInputs input, ParallelPairOutputs output) {
  if (*input.rejected != 0) return;
  const auto nactive = *input.active_count;
  constexpr unsigned rows_per_lane = 4, tile = 32 * rows_per_lane;
  const unsigned lane = threadIdx.x % 32;
  const size_t warp = (size_t(blockIdx.x) * blockDim.x + threadIdx.x) / 32;
  const size_t stride = size_t(gridDim.x) * blockDim.x / 32;
  const size_t tiles = (nactive + tile - 1) / tile, count = tiles * (tiles + 1) / 2;
  for (size_t ordinal = warp; ordinal < count; ordinal += stride) {
    size_t a = size_t((sqrt(8.0 * double(ordinal) + 1.0) - 1.0) * 0.5);
    while ((a + 1) * (a + 2) / 2 <= ordinal) ++a;
    while (a * (a + 1) / 2 > ordinal) --a;
    const size_t b = ordinal - a * (a + 1) / 2;
    double sum[rows_per_lane][6]{};
    for (unsigned k = 0; k < tile && tile * b + k < nactive; ++k) {
      const size_t j = input.indices[tile * b + k];
      const bool live_j = !(input.weighted_density[j] == 0.0 && signbit(input.weighted_density[j]));
      double pj[6]{};
#pragma unroll
      for (unsigned row = 0; row < rows_per_lane; ++row) {
        const size_t row_slot = tile * a + 32 * row + lane;
        const size_t i = row_slot < nactive ? input.indices[row_slot] : 0;
        const bool live_i = row_slot < nactive && !(input.weighted_density[i] == 0.0 &&
                                                    signbit(input.weighted_density[i]));
        if (live_i && live_j) {
          const double fi = input.weighted_density[i], fj = input.weighted_density[j];
          const double dx = input.points[3 * j] - input.points[3 * i],
                       dy = input.points[3 * j + 1] - input.points[3 * i + 1],
                       dz = input.points[3 * j + 2] - input.points[3 * i + 2];
          const double r2 = dx * dx + dy * dy + dz * dz;
          const auto pair = pair_values_vv10_bilateral<Geometry>(r2, input.omega[i], input.omega[j],
                                                                 input.kappa[i], input.kappa[j]);
          const double phi = pair.phi, wi = pair.dphi_domega, ki = pair.dphi_dkappa;
          const double wj = pair.dphi_domega_j, kj = pair.dphi_dkappa_j, radial = pair.dphi_dr2;
          sum[row][0] += fj * phi;
          sum[row][1] += fj * wi;
          sum[row][2] += fj * ki;
          pj[0] += fi * phi;
          pj[1] += fi * wj;
          pj[2] += fi * kj;
          if constexpr (Geometry) {
            sum[row][3] += -2.0 * fj * radial * dx;
            sum[row][4] += -2.0 * fj * radial * dy;
            sum[row][5] += -2.0 * fj * radial * dz;
            pj[3] += 2.0 * fi * radial * dx;
            pj[4] += 2.0 * fi * radial * dy;
            pj[5] += 2.0 * fi * radial * dz;
          }
        }
      }
      if (a != b) {
        for (unsigned field = 0; field < (Geometry ? 6U : 3U); ++field) {
          const double total = parallel_pair_warp_sum(pj[field]);
          if (lane == 0 && live_j) {
            atomicAdd(parallel_pair_partial(output, j, field), total);
          }
        }
      }
    }
#pragma unroll
    for (unsigned row = 0; row < rows_per_lane; ++row) {
      const size_t row_slot = tile * a + 32 * row + lane;
      const size_t i = row_slot < nactive ? input.indices[row_slot] : 0;
      const bool live_i = row_slot < nactive &&
                          !(input.weighted_density[i] == 0.0 && signbit(input.weighted_density[i]));
      if (live_i)
        for (unsigned field = 0; field < (Geometry ? 6U : 3U); ++field) {
          atomicAdd(parallel_pair_partial(output, i, field), sum[row][field]);
        }
    }
  }
}

template <bool Geometry>
__global__ void parallel_pair_finalize(ParallelPairInputs input, ParallelPairOutputs output) {
  if (*input.rejected != 0) return;
  const auto row = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (row >= input.point_count) return;
  const double factor = input.weighted_density[row];
  if (factor == 0.0 && signbit(factor)) return;
  double phi = output.energy_terms[row], w = output.vrho[row], k = output.vsigma[row];
  // Positive-zero integration weights have observable feature/weight response,
  // but exert no back reaction on nonzero partners. Evaluate these rows once
  // against the compact nonzero list; they are absent from bilateral atomics.
  // The density-screened negative-zero domain returned above remains excluded.
  if (factor == 0.0) {
    for (std::uint64_t slot = 0; slot < *input.active_count; ++slot) {
      const auto j = input.indices[slot];
      const double dx = input.points[3*j] - input.points[3*row];
      const double dy = input.points[3*j+1] - input.points[3*row+1];
      const double dz = input.points[3*j+2] - input.points[3*row+2];
      const auto pair = pair_values_vv10_admitted<false>(
          dx*dx+dy*dy+dz*dz, input.omega[row], input.omega[j], input.kappa[row], input.kappa[j]);
      const double partner = input.weighted_density[j];
      phi += partner * pair.phi;
      w += partner * pair.dphi_domega;
      k += partner * pair.dphi_dkappa;
    }
  }
  const double rho = input.density[row];
  output.energy_terms[row] = input.coefficient * factor * (input.beta + 0.5 * phi);
  row_feature_values(phi, w, k, rho, input.domega_drho[row], input.domega_dsigma[row],
                     input.dkappa_drho[row], input.beta, input.coefficient, output.vrho[row],
                     output.vsigma[row]);
  bool invalid = !isfinite(output.energy_terms[row]) || !isfinite(output.vrho[row]) ||
                 !isfinite(output.vsigma[row]);
  if constexpr (Geometry) {
    for (unsigned axis = 0; axis < 3; ++axis) {
      output.point_derivative[3 * row + axis] *= input.coefficient * factor;
      invalid = invalid || !isfinite(output.point_derivative[3 * row + axis]);
    }
    output.weight_derivative[row] = input.coefficient * rho * (input.beta + phi);
    invalid = invalid || !isfinite(output.weight_derivative[row]);
  }
  if (invalid) atomicExch(output.failed, 1);
}
#endif
"""
    )
