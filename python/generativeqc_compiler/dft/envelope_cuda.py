"""Device lowering of the existing conservative pre-collocation AO domain.

The independent Python oracle remains ``ao_region_envelopes``. This lowering
bounds complete through-order jets from normalized packed contractions before
any AO is sampled. Unsupported polynomial degrees and arithmetic overflow
retain an AO; a loose region changes cost, never the omission criterion.
"""


def emit_ao_region_screen_cuda() -> str:
    """Emit allocation-free classification using the caller's dead grid scratch.

    ``bounds`` needs six doubles and flags need one unsigned per global AO.
    They are temporary classification storage, not a second sparse layout.
    The caller binds the resulting domain with ``AoGridBlockLayout`` and must
    preserve its basis, geometry, point range and derivative capability.
    """
    return r"""
namespace {
__device__ double ao_bound_add(double left, double right) {
  if (left == 0) return right;
  if (right == 0) return left;
  return nextafter(left + right, HUGE_VAL);
}
__device__ double ao_bound_multiply(double left, double right) {
  if (left == 0 || right == 0) return 0;
  return nextafter(left * right, HUGE_VAL);
}
__device__ double ao_region_axis_bound(int power, int derivative, double alpha,
                                      double lower, double upper) {
  // Fixed storage is a resource guard, not angular-momentum screening.
  constexpr int maximum_degree = 11;
  if (power < 0 || derivative < 0 || derivative > 3 ||
      power + derivative > maximum_degree || !isfinite(lower) || !isfinite(upper))
    return INFINITY;
  double coefficients[maximum_degree + 1]{};
  coefficients[power] = 1;
  for (int order = 0; order < derivative; ++order) {
    double following[maximum_degree + 1]{};
    for (int degree = 0; degree <= power + order; ++degree) {
      if (degree)
        following[degree - 1] = ao_bound_add(
            following[degree - 1], ao_bound_multiply(degree, coefficients[degree]));
      following[degree + 1] = ao_bound_add(
          following[degree + 1], ao_bound_multiply(
              ao_bound_multiply(2.0, alpha), coefficients[degree]));
    }
    for (int degree = 0; degree <= maximum_degree; ++degree)
      coefficients[degree] = following[degree];
  }
  const double maximum = fmax(fabs(lower), fabs(upper));
  const double minimum = lower <= 0 && upper >= 0
                             ? 0 : fmin(fabs(lower), fabs(upper));
  double polynomial = 0, monomial = 1;
  for (int degree = 0; degree <= power + derivative; ++degree) {
    polynomial = ao_bound_add(
        polynomial, ao_bound_multiply(coefficients[degree], monomial));
    monomial = ao_bound_multiply(monomial, maximum);
  }
  if (!isfinite(polynomial)) return INFINITY;
  const double square = fmax(0.0, nextafter(minimum * minimum, -HUGE_VAL));
  const double exponent = fmax(0.0, nextafter(alpha * square, -HUGE_VAL));
  double gaussian = nextafter(exp(-exponent), HUGE_VAL);
  gaussian = ao_bound_multiply(gaussian, 1.0 + 32.0 * 0x1p-52);
  return ao_bound_multiply(polynomial, gaussian);
}

__global__ void ao_region_box_kernel(const double* points, size_t count,
                                    double* bounds, int* error) {
  // The bounded tile scan writes one shared domain, never one box per AO.
  if (blockIdx.x || threadIdx.x) return;
  for (size_t axis = 0; axis < 3; ++axis) {
    double lower = INFINITY, upper = -INFINITY;
    for (size_t point = 0; point < count; ++point) {
      const double value = points[3 * point + axis];
      if (!isfinite(value)) atomicExch(error, 1);
      lower = fmin(lower, value);
      upper = fmax(upper, value);
    }
    bounds[axis] = lower;
    bounds[3 + axis] = upper;
  }
}

__device__ bool ao_region_retained(const double* basis, size_t atoms,
                                  size_t primitives_count, size_t ao,
                                  size_t jets, const double* bounds,
                                  double cutoff, const int* error) {
  const double* primitives = basis + 3 * atoms;
  const double* records = primitives + 2 * primitives_count;
    const double* record = records + 16 * ao;
    const size_t atom = size_t(record[0]);
    double lower[3], upper[3];
    for (size_t axis = 0; axis < 3; ++axis) {
      lower[axis] = nextafter(bounds[axis] - basis[3 * atom + axis], -HUGE_VAL);
      upper[axis] = nextafter(bounds[3 + axis] - basis[3 * atom + axis], HUGE_VAL);
    }
    bool retain = *error != 0;
    for (size_t jet = 0; jet < jets; ++jet) {
      double total = 0;
      const size_t first = size_t(record[1]), end = first + size_t(record[2]);
      for (size_t primitive = first; primitive < end; ++primitive) {
        const double alpha = primitives[2 * primitive];
        for (int term = 0; term < int(record[3]); ++term) {
          double value = ao_bound_multiply(
              fabs(primitives[2 * primitive + 1]), fabs(record[7 + 4 * term]));
          for (size_t axis = 0; axis < 3; ++axis)
            value = ao_bound_multiply(value, ao_region_axis_bound(
                int(record[4 + 4 * term + axis]), derivatives[jet][axis], alpha,
                lower[axis], upper[axis]));
          total = ao_bound_add(total, value);
        }
      }
      retain = retain || total > cutoff;
    }
    return retain;
}

__global__ void ao_region_screen_kernel(const double* basis, size_t atoms,
                                       size_t primitives_count, size_t aos,
                                       size_t jets, const double* bounds,
                                       double cutoff, unsigned* selected, int* error) {
  for (size_t ao = size_t(blockIdx.x) * blockDim.x + threadIdx.x; ao < aos;
       ao += size_t(blockDim.x) * gridDim.x) {
    selected[ao] = ao_region_retained(basis, atoms, primitives_count, ao, jets,
                                     bounds, cutoff, error);
  }
}

__global__ void ao_region_boxes_kernel(const double* points, size_t count,
                                      size_t tile_points, size_t tiles,
                                      double* bounds, int* error) {
  if (threadIdx.x) return;
  for (size_t tile = blockIdx.x; tile < tiles; tile += gridDim.x) {
    const size_t begin = tile * tile_points;
    const size_t end = min(count, begin + tile_points);
    for (size_t axis = 0; axis < 3; ++axis) {
      double lower = INFINITY, upper = -INFINITY;
      for (size_t point = begin; point < end; ++point) {
        const double value = points[3 * point + axis];
        if (!isfinite(value)) atomicExch(error, 1);
        lower = fmin(lower, value);
        upper = fmax(upper, value);
      }
      bounds[6 * tile + axis] = lower;
      bounds[6 * tile + 3 + axis] = upper;
    }
  }
}

__global__ void ao_region_mask_kernel(const double* basis, size_t atoms,
                                     size_t primitives, size_t aos, size_t jets,
                                     const double* bounds, size_t tiles,
                                     double cutoff, unsigned* masks,
                                     unsigned long long* counts, int* error) {
  const size_t words = (aos + 31) / 32;
  const unsigned lane = threadIdx.x % 32;
  const size_t word = size_t(blockIdx.x) * (blockDim.x / 32) + threadIdx.x / 32;
  for (size_t tile = blockIdx.y; tile < tiles; tile += gridDim.y) {
    const size_t ao = 32 * word + lane;
    const bool retained = ao < aos && ao_region_retained(
        basis, atoms, primitives, ao, jets, bounds + 6 * tile, cutoff, error);
    const unsigned mask = __ballot_sync(0xffffffff, retained);
    if (!lane && word < words) {
      masks[tile * words + word] = mask;
      atomicAdd(counts + tile + 1, static_cast<unsigned long long>(__popc(mask)));
    }
  }
}

__global__ void ao_region_compact_kernel(const unsigned* masks, size_t words,
                                        size_t tiles, const size_t* offsets,
                                        size_t* indices) {
  // Ascending word/lane order is the sorted-unique AO invariant. Only integer
  // prefix sums are parallelized; this is not a scientific reduction.
  __shared__ size_t prefix[5];
  const unsigned lane = threadIdx.x % 32, warp = threadIdx.x / 32;
  for (size_t tile = blockIdx.x; tile < tiles; tile += gridDim.x) {
    if (!threadIdx.x) prefix[4] = offsets[tile];
    __syncthreads();
    for (size_t first = 0; first < words; first += 4) {
      const size_t word = first + warp;
      const unsigned mask = word < words ? masks[tile * words + word] : 0;
      if (!lane) prefix[warp] = __popc(mask);
      __syncthreads();
      if (!threadIdx.x) {
        size_t position = prefix[4];
        for (unsigned group = 0; group < 4; ++group) {
          const size_t count = prefix[group];
          prefix[group] = position;
          position += count;
        }
        prefix[4] = position;
      }
      __syncthreads();
      if (mask & (1U << lane))
        indices[prefix[warp] + __popc(mask & ((1U << lane) - 1U))] = word * 32 + lane;
      __syncthreads();
    }
  }
}
}  // namespace
"""
