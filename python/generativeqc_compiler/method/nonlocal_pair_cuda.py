"""Emit bounded parallel VV10 pair traversal over the canonical scalar roots.

Native owners admit the numerical domain, exclusive output leases and stream.
The CUDA templates only schedule/reduce those already generated pair outputs.
"""

VV10_PARALLEL_PAIR_SCHEDULE_VERSION = "vv10-parallel-bilateral-warp-packet-mixed-v5"


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
  bool mixed_pair_arithmetic;
  std::uint64_t* arithmetic_work;
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
// Each persistent warp owns an unordered pair of 128- or 256-row tiles.
// Four or eight rows per lane amortize the partner reductions; diagonal tiles retain all ordered
// pairs. Cross-tile contributions reduce in FP64 then atomically accumulate
// both legs. This changes summation order, not pair formulas or active domain.
__device__ inline double parallel_pair_warp_sum(double value) {
  for (int shift = 16; shift; shift /= 2) value += __shfl_down_sync(0xffffffffU, value, shift);
  return value;
}
template <bool Geometry, unsigned RowsPerLane = 4, bool HoistRows = false, bool Adaptive = false>
__global__ void parallel_pair_kernel(ParallelPairInputs input, ParallelPairOutputs output) {
  if (*input.rejected != 0) return;
  const auto nactive = *input.active_count;
  static_assert(RowsPerLane == 4 || RowsPerLane == 8);
  // Each branch reads the device-owned count. The two adaptive launches are
  // mutually exclusive, avoiding a host synchronization or retained selector.
  if constexpr (Adaptive) {
    if ((nactive >= 65536) != (RowsPerLane == 8)) return;
  }
  constexpr unsigned rows_per_lane = RowsPerLane, tile = 32 * rows_per_lane;
  const unsigned lane = threadIdx.x % 32;
  const size_t warp = (size_t(blockIdx.x) * blockDim.x + threadIdx.x) / 32;
  const size_t stride = size_t(gridDim.x) * blockDim.x / 32;
  const size_t tiles = (nactive + tile - 1) / tile, count = tiles * (tiles + 1) / 2;
  unsigned long long mixed_pairs = 0, strict_pairs = 0;
  for (size_t ordinal = warp; ordinal < count; ordinal += stride) {
    size_t a = size_t((sqrt(8.0 * double(ordinal) + 1.0) - 1.0) * 0.5);
    while ((a + 1) * (a + 2) / 2 <= ordinal) ++a;
    while (a * (a + 1) / 2 > ordinal) --a;
    const size_t b = ordinal - a * (a + 1) / 2;
    double sum[rows_per_lane][6]{};
    // Large admitted domains amortize these immutable row inputs over every
    // partner. Small domains retain the lower-register four-row schedule.
    bool row_live[rows_per_lane];
    double row_f[rows_per_lane], row_w[rows_per_lane], row_k[rows_per_lane];
    double row_x[rows_per_lane], row_y[rows_per_lane], row_z[rows_per_lane];
    if constexpr (HoistRows) {
#pragma unroll
      for (unsigned row = 0; row < rows_per_lane; ++row) {
        const size_t slot = tile * a + 32 * row + lane;
        const size_t i = slot < nactive ? input.indices[slot] : 0;
        row_live[row] = slot < nactive && !(input.weighted_density[i] == 0.0 &&
                                            signbit(input.weighted_density[i]));
        row_f[row] = input.weighted_density[i];
        row_w[row] = input.omega[i]; row_k[row] = input.kappa[i];
        row_x[row] = input.points[3*i]; row_y[row] = input.points[3*i+1];
        row_z[row] = input.points[3*i+2];
      }
    }
    for (unsigned k = 0; k < tile && tile * b + k < nactive; ++k) {
      const size_t j = input.indices[tile * b + k];
      const bool live_j = !(input.weighted_density[j] == 0.0 && signbit(input.weighted_density[j]));
      double pj[6]{};
#pragma unroll
      for (unsigned row = 0; row < rows_per_lane; ++row) {
        const size_t row_slot = tile * a + 32 * row + lane;
        const size_t i = row_slot < nactive ? input.indices[row_slot] : 0;
        const bool live_i = HoistRows ? row_live[row] :
            row_slot < nactive && !(input.weighted_density[i] == 0.0 &&
                                    signbit(input.weighted_density[i]));
        if (live_i && live_j) {
          const double fi = HoistRows ? row_f[row] : input.weighted_density[i];
          const double fj = input.weighted_density[j];
          const double dx = input.points[3 * j] - (HoistRows ? row_x[row] : input.points[3 * i]),
                       dy = input.points[3 * j + 1] - (HoistRows ? row_y[row] : input.points[3 * i + 1]),
                       dz = input.points[3 * j + 2] - (HoistRows ? row_z[row] : input.points[3 * i + 2]);
          const double r2 = dx * dx + dy * dy + dz * dz;
          BilateralPairValues pair{};
          if (input.mixed_pair_arithmetic && try_pair_values_vv10_bilateral_fp32<Geometry>(
                  r2, HoistRows ? row_w[row] : input.omega[i], input.omega[j],
                  HoistRows ? row_k[row] : input.kappa[i], input.kappa[j], pair))
            ++mixed_pairs;
          else {
            pair = pair_values_vv10_bilateral<Geometry>(r2, HoistRows ? row_w[row] : input.omega[i], input.omega[j],
                                                         HoistRows ? row_k[row] : input.kappa[i], input.kappa[j]);
            ++strict_pairs;
          }
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
  if (input.arithmetic_work) {
    atomicAdd(reinterpret_cast<unsigned long long*>(input.arithmetic_work), mixed_pairs);
    atomicAdd(reinterpret_cast<unsigned long long*>(input.arithmetic_work + 1), strict_pairs);
  }
}

// A packet carries one partner and its bilateral partials around all 32 lanes.
// After a complete rotation the original owner publishes that partner's sums.
// This coalesces partner loads/stores and replaces a warp reduction per partner
// with register exchange. All lanes, including padded partners, must rotate;
// only scientific pair evaluation and publication may be predicated.
template <bool Geometry, bool RetainScaleProofs = false>
__global__ void parallel_pair_packet_kernel(ParallelPairInputs input, ParallelPairOutputs output) {
  if (*input.rejected != 0) return;
  constexpr unsigned rows_per_lane = 4, tile = 32 * rows_per_lane;
  constexpr unsigned fields = Geometry ? 6 : 3;
  const auto active_count = *input.active_count;
  const unsigned lane = threadIdx.x % 32;
  const size_t warp = (size_t(blockIdx.x) * blockDim.x + threadIdx.x) / 32;
  const size_t stride = size_t(gridDim.x) * blockDim.x / 32;
  const size_t tiles = (active_count + tile - 1) / tile;
  const size_t count = tiles * (tiles + 1) / 2;
  unsigned long long mixed_pairs = 0, strict_pairs = 0;
  for (size_t ordinal = warp; ordinal < count; ordinal += stride) {
    size_t outer = size_t((sqrt(8.0 * double(ordinal) + 1.0) - 1.0) * 0.5);
    while ((outer + 1) * (outer + 2) / 2 <= ordinal) ++outer;
    while (outer * (outer + 1) / 2 > ordinal) --outer;
    const size_t inner = ordinal - outer * (outer + 1) / 2;
    double sum[rows_per_lane][fields]{};
    int row_state[rows_per_lane];
    bool row_live[rows_per_lane];
    double row_factor[rows_per_lane], row_omega[rows_per_lane], row_kappa[rows_per_lane];
    double row_x[rows_per_lane], row_y[rows_per_lane], row_z[rows_per_lane];
#pragma unroll
    for (unsigned row = 0; row < rows_per_lane; ++row) {
      const size_t slot = tile * outer + 32 * row + lane;
      const size_t index = slot < active_count ? input.indices[slot] : 0;
      row_factor[row] = input.weighted_density[index];
      row_omega[row] = input.omega[index];
      row_kappa[row] = input.kappa[index];
      // Partner bit zero owns the live row; bit one retains its FP32 scale proof.
      // Rotation carries both flags in the existing partner-state shuffle,
      // avoiding eight immutable scale comparisons for every physical pair.
      row_live[row] = slot < active_count &&
                      !(row_factor[row] == 0.0 && signbit(row_factor[row]));
      if constexpr (RetainScaleProofs) {
        row_state[row] = input.mixed_pair_arithmetic &&
            fp32_pair_scale_domain(row_omega[row], row_kappa[row]) ? 2 : 0;
      }
      row_x[row] = input.points[3 * index];
      row_y[row] = input.points[3 * index + 1];
      row_z[row] = input.points[3 * index + 2];
    }
    for (unsigned batch = 0; batch < tile && tile * inner + batch < active_count; batch += 32) {
      const size_t slot = tile * inner + batch + lane;
      const size_t index = slot < active_count ? input.indices[slot] : 0;
      double factor = input.weighted_density[index];
      double omega = input.omega[index], kappa = input.kappa[index];
      double point_x = input.points[3 * index], point_y = input.points[3 * index + 1];
      double point_z = input.points[3 * index + 2];
      int live = slot < active_count && !(factor == 0.0 && signbit(factor));
      if constexpr (RetainScaleProofs)
        if (input.mixed_pair_arithmetic && fp32_pair_scale_domain(omega, kappa)) live |= 2;
      double partner[fields]{};
#pragma unroll 1
      for (unsigned step = 0; step < 32; ++step) {
#pragma unroll
        for (unsigned row = 0; row < rows_per_lane; ++row) {
          const bool partner_live = RetainScaleProofs ? (live & 1) != 0 : live != 0;
          if (partner_live && row_live[row]) {
            const double dx = point_x - row_x[row], dy = point_y - row_y[row];
            const double dz = point_z - row_z[row];
            const double r2 = dx * dx + dy * dy + dz * dz;
            BilateralPairValues pair{};
            // Scale proofs travel with their own rows; only distance varies.
            bool mixed_pair = false;
            if constexpr (RetainScaleProofs) {
              mixed_pair = ((live & row_state[row]) & 2) && fp32_pair_radius_domain(r2);
              if (mixed_pair)
                pair = pair_values_vv10_bilateral_fp32<Geometry>(
                    r2, row_omega[row], omega, row_kappa[row], kappa);
            } else {
              mixed_pair = input.mixed_pair_arithmetic && try_pair_values_vv10_bilateral_fp32<Geometry>(
                  r2, row_omega[row], omega, row_kappa[row], kappa, pair);
            }
            if (mixed_pair) {
              ++mixed_pairs;
            } else {
              pair = pair_values_vv10_bilateral<Geometry>(
                  r2, row_omega[row], omega, row_kappa[row], kappa);
              ++strict_pairs;
            }
            sum[row][0] += factor * pair.phi;
            sum[row][1] += factor * pair.dphi_domega;
            sum[row][2] += factor * pair.dphi_dkappa;
            if (outer != inner) {
              partner[0] += row_factor[row] * pair.phi;
              partner[1] += row_factor[row] * pair.dphi_domega_j;
              partner[2] += row_factor[row] * pair.dphi_dkappa_j;
            }
            if constexpr (Geometry) {
              sum[row][3] += -2.0 * factor * pair.dphi_dr2 * dx;
              sum[row][4] += -2.0 * factor * pair.dphi_dr2 * dy;
              sum[row][5] += -2.0 * factor * pair.dphi_dr2 * dz;
              if (outer != inner) {
                partner[3] += 2.0 * row_factor[row] * pair.dphi_dr2 * dx;
                partner[4] += 2.0 * row_factor[row] * pair.dphi_dr2 * dy;
                partner[5] += 2.0 * row_factor[row] * pair.dphi_dr2 * dz;
              }
            }
          }
        }
        const unsigned next = (lane + 1) % 32;
        live = __shfl_sync(0xffffffffU, live, next);
        factor = __shfl_sync(0xffffffffU, factor, next);
        omega = __shfl_sync(0xffffffffU, omega, next);
        kappa = __shfl_sync(0xffffffffU, kappa, next);
        point_x = __shfl_sync(0xffffffffU, point_x, next);
        point_y = __shfl_sync(0xffffffffU, point_y, next);
        point_z = __shfl_sync(0xffffffffU, point_z, next);
#pragma unroll
        for (unsigned field = 0; field < fields; ++field)
          partner[field] = __shfl_sync(0xffffffffU, partner[field], next);
      }
      // Diagonal tiles already evaluated both ordered legs in the row sums.
      const bool partner_live = RetainScaleProofs ? (live & 1) != 0 : live != 0;
      if (outer != inner && partner_live) {
#pragma unroll
        for (unsigned field = 0; field < fields; ++field)
          atomicAdd(parallel_pair_partial(output, index, field), partner[field]);
      }
    }
#pragma unroll
    for (unsigned row = 0; row < rows_per_lane; ++row) {
      if (row_live[row]) {
        const size_t index = input.indices[tile * outer + 32 * row + lane];
#pragma unroll
        for (unsigned field = 0; field < fields; ++field)
          atomicAdd(parallel_pair_partial(output, index, field), sum[row][field]);
      }
    }
  }
  if (input.arithmetic_work) {
    atomicAdd(reinterpret_cast<unsigned long long*>(input.arithmetic_work), mixed_pairs);
    atomicAdd(reinterpret_cast<unsigned long long*>(input.arithmetic_work + 1), strict_pairs);
  }
}

// Positive-zero rows still have observable feature/weight response. A warp
// owns each such row and reduces the compact nonzero partners in FP64. The
// decision is uniform within a warp; inactive lanes contribute zero. These
// rows never receive bilateral atomics, so the row owner writes each sum once.
// Work remains Z*N, but neighboring lanes now load neighboring partner slots.

__global__ void parallel_pair_zero_rows(ParallelPairInputs input, ParallelPairOutputs output) {
  if (*input.rejected != 0) return;
  const unsigned lane = threadIdx.x % 32;
  const size_t warp = (size_t(blockIdx.x) * blockDim.x + threadIdx.x) / 32;
  const size_t stride = size_t(gridDim.x) * blockDim.x / 32;
  unsigned long long mixed_pairs = 0, strict_pairs = 0;
  for (size_t row = warp; row < input.point_count; row += stride) {
    const double factor = input.weighted_density[row];
    if (factor != 0.0 || signbit(factor)) continue;
    double phi = 0.0, w = 0.0, k = 0.0;
    for (std::uint64_t slot = lane; slot < *input.active_count; slot += 32) {
      const auto j = input.indices[slot];
      const double dx = input.points[3 * j] - input.points[3 * row];
      const double dy = input.points[3 * j + 1] - input.points[3 * row + 1];
      const double dz = input.points[3 * j + 2] - input.points[3 * row + 2];
      const double r2 = dx * dx + dy * dy + dz * dz;
      PairValues p{};
      if (input.mixed_pair_arithmetic && try_pair_values_fp32<Vv10Variant::vv10, true, false>(
              r2, input.omega[row], input.omega[j], input.kappa[row], input.kappa[j], 0.0, p))
        ++mixed_pairs;
      else {
        p = pair_values_vv10_admitted<false>(r2, input.omega[row], input.omega[j], input.kappa[row], input.kappa[j]);
        ++strict_pairs;
      }
      const double partner = input.weighted_density[j];
      phi += partner * p.phi;
      w += partner * p.dphi_domega;
      k += partner * p.dphi_dkappa;
    }
    phi = parallel_pair_warp_sum(phi);
    w = parallel_pair_warp_sum(w);
    k = parallel_pair_warp_sum(k);
    if (lane == 0) {
      output.energy_terms[row] = phi;
      output.vrho[row] = w;
      output.vsigma[row] = k;
    }
  }
  if (input.arithmetic_work) {
    atomicAdd(reinterpret_cast<unsigned long long*>(input.arithmetic_work), mixed_pairs);
    atomicAdd(reinterpret_cast<unsigned long long*>(input.arithmetic_work + 1), strict_pairs);
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
