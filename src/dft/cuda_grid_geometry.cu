#include <algorithm>
#include <climits>
#include <cmath>
#include <limits>
#include <type_traits>

#include "dft/cuda_grid_geometry.hpp"
#include "dft/hessian_aot.hpp"
#include "response/grid_geometry_response.cuh"

namespace generativeqc::dft {
namespace {
namespace ad = generativeqc_grid_geometry_ad;

struct Layout {
  std::size_t bytes{}, spatial{}, candidate_ao{}, candidate_weights{}, pairs{}, distances{},
      products{}, exponents{}, zero{}, status{}, panel{}, basis_elements{}, point_pairs{},
      ao_primitive_terms{}, replay_terms{}, point_directions{}, resident_points{}, replay_points{};
  std::size_t wide_scratch{}, wide_pairs{}, wide_failures{}, wide_tile{}, wide_repair_points{};
  bool wide_shared{};
};
constexpr std::size_t spatial_planes(const CudaGridGeometryShape& shape) {
  return shape.radial_s ? 4 : 20;
}
struct PartitionPair {
  ad::Jet distance, reciprocal;
  double adjustment;
};
static_assert(sizeof(PartitionPair) == 9 * sizeof(double));
// The full-width warp packs complementary switches to fit in 48 KiB. The
// smaller specialization keeps its direct lookup for the prefix/suffix loops.
constexpr std::size_t unpacked_cartesian_atoms = 30;
// Both product planes and all radial jets fit in 48 KiB for this extent.
constexpr std::size_t shared_wide_cartesian_atoms = 36;
struct CartesianPair {
  ad::Jet inverse[3];
  double adjustment;
};
static_assert(sizeof(CartesianPair) == 13 * sizeof(double));
bool batched_cartesian(const CudaGridGeometryShape& shape, const CudaGridGeometryOptions& options) {
  return options.cache_cartesian_weights && shape.atoms > 1 &&
         shape.atoms <= cuda_grid_cartesian_atoms &&
         shape.points <= 65535 &&
         9 * shape.atoms + 1 <= 12 * shape.nao &&
         13 * (shape.atoms - 1) <= 8 * std::min(shape.points, options.point_tile);
}
bool tiled_cartesian(const CudaGridGeometryShape& shape, const CudaGridGeometryOptions& options) {
  return options.cache_cartesian_weights && shape.atoms > cuda_grid_cartesian_atoms &&
         shape.atoms <= cuda_grid_tiled_cartesian_atoms && shape.points <= UINT_MAX &&
         6 * shape.atoms <= 8 * shape.nao;
}
bool product(std::size_t first, std::size_t second, std::size_t& result) {
  if (second && first > SIZE_MAX / second) return false;
  result = first * second;
  return true;
}
bool append(Layout& plan, std::size_t count, std::size_t& offset) {
  std::size_t bytes{};
  if (!product(count, sizeof(double), bytes) || bytes > SIZE_MAX - 255 ||
      plan.bytes > SIZE_MAX - bytes - 255)
    return false;
  offset = plan.bytes;
  plan.bytes = (plan.bytes + bytes + 255) / 256 * 256;
  return true;
}
bool layout(const CudaGridGeometryShape& shape, const CudaGridGeometryOptions& options,
            Layout& plan, std::size_t* minimum = nullptr) {
  if (!shape.atoms || !shape.primitives || !shape.nao || !shape.points || !options.point_tile ||
      !options.max_workspace_bytes || options.partition_iterations < 1 ||
      options.partition_iterations > 5 || !std::isfinite(options.coincident_tolerance) ||
      options.coincident_tolerance < 0 || !std::isfinite(options.reference_absolute_tolerance) ||
      options.reference_absolute_tolerance < 0 ||
      !std::isfinite(options.reference_relative_tolerance) ||
      options.reference_relative_tolerance < 0)
    return false;
  std::size_t points{}, matrix{}, tile{}, spatial{}, moving{}, atomic{}, primitive{}, records{};
  const auto planes = spatial_planes(shape);
  plan.ao_primitive_terms = shape.ao_primitive_terms;
  if (shape.radial_s && !plan.ao_primitive_terms) return false;
  if (!plan.ao_primitive_terms && (!product(shape.nao, shape.primitives, plan.ao_primitive_terms) ||
                                   !product(3, plan.ao_primitive_terms, plan.ao_primitive_terms)))
    return false;
  if ((shape.points_per_atom &&
       (!product(shape.atoms, shape.points_per_atom, points) || points != shape.points)) ||
      !product(shape.points, shape.nao, plan.panel) || !product(planes, plan.panel, spatial) ||
      !product(12, plan.panel, moving) || spatial > options.max_ao_jet_values ||
      !product(planes, plan.ao_primitive_terms, plan.ao_primitive_terms) ||
      !product(shape.points, plan.ao_primitive_terms, plan.ao_primitive_terms) ||
      plan.ao_primitive_terms > options.max_ao_primitive_terms ||
      !product(shape.atoms, shape.atoms, matrix) ||
      !product(shape.atoms, shape.atoms - 1, plan.point_pairs) ||
      !product(shape.points, plan.point_pairs / 2, plan.point_pairs) ||
      plan.point_pairs > options.max_partition_pairs ||
      !product(std::min(options.point_tile, shape.points), shape.atoms, tile) ||
      !product(3, shape.atoms, atomic) || !product(2, shape.primitives, primitive) ||
      !product(16, shape.nao, records) || atomic > SIZE_MAX - primitive ||
      atomic + primitive > SIZE_MAX - records)
    return false;
  plan.basis_elements = atomic + primitive + records;
  if (plan.basis_elements > SIZE_MAX / sizeof(double)) return false;
  std::size_t count{};
  if (!append(plan, moving, plan.candidate_ao) || !product(3, shape.points, count) ||
      !append(plan, count, plan.candidate_weights) ||
      !product(sizeof(PartitionPair) / sizeof(double), (matrix - shape.atoms) / 2, count) ||
      !append(plan, count, plan.pairs) || !product(4, tile, count) ||
      !append(plan, count, plan.distances) || !append(plan, count, plan.products) ||
      !append(plan, tile, plan.exponents) || !append(plan, atomic, plan.zero) ||
      !product(6, shape.points, count) || !append(plan, count, plan.point_directions) ||
      !append(plan, 1, plan.status) || plan.bytes > options.max_workspace_bytes)
    return false;
  // Keep the largest immutable prefix after reserving one replay tile. The
  // mixed output keeps its full layout regardless of how its jets are supplied.
  plan.resident_points = shape.points;
  const auto terms_per_point = plan.ao_primitive_terms / shape.points;
  const auto replayable =
      options.retain_partial_jets
          ? std::min<std::uint64_t>(shape.points, options.max_ao_replay_terms / terms_per_point)
          : (plan.ao_primitive_terms <= options.max_ao_replay_terms ? shape.points : 0);
  const auto required_resident = shape.points - replayable;
  if (minimum) {
    auto smallest = plan;
    const auto values = planes * (required_resident + (replayable != 0)) * shape.nao;
    if (!append(smallest, values, smallest.spatial)) return false;
    *minimum = smallest.bytes;
  }
  const auto available_values = (options.max_workspace_bytes - plan.bytes) / 256 * 32;
  if (spatial > available_values) {
    const auto capacity = available_values / (planes * shape.nao);
    if (capacity <= required_resident) return false;
    plan.replay_points = std::min<std::uint64_t>(options.point_tile, capacity - required_resident);
    plan.resident_points = options.retain_partial_jets ? capacity - plan.replay_points : 0;
    plan.replay_terms = terms_per_point * (shape.points - plan.resident_points);
    spatial = planes * (plan.resident_points + plan.replay_points) * shape.nao;
  }
  if (!append(plan, spatial, plan.spatial) || plan.bytes > options.max_workspace_bytes) return false;
  if (!plan.replay_points && tiled_cartesian(shape, options)) {
    // Keep prefix and switch duals in block-local storage when they fit;
    // larger shapes use a bounded point tile, never the full grid.
    constexpr std::size_t tile_byte_limit = 32ULL << 20;
    const auto words_per_point = 4 * matrix;
    const auto repair_cost = 3 * shape.atoms * (matrix - shape.atoms) / 2;
    auto candidate = plan;
    candidate.wide_pairs = plan.distances;
    const auto pair_words = (sizeof(CartesianPair) / sizeof(double)) * (matrix - shape.atoms) / 2;
    candidate.wide_repair_points =
        std::min<std::uint64_t>(shape.points, options.max_cartesian_repair_pairs / repair_cost);
    // The ordinary partition tile can house the center coefficients only
    // when it is wide enough; short tiles reserve the coefficients separately.
    if ((pair_words <= 4 * tile || append(candidate, pair_words, candidate.wide_pairs)) &&
        append(candidate, 1 + (candidate.wide_repair_points + 1) / 2,
               candidate.wide_failures) && candidate.bytes <= options.max_workspace_bytes) {
      if (shape.atoms <= shared_wide_cartesian_atoms) {
        candidate.wide_shared = true;
        candidate.wide_tile = std::min(shape.points, std::size_t{65535});
        plan = candidate;
        return true;
      }
      const auto available = options.max_workspace_bytes - candidate.bytes;
      const auto bytes_per_point = words_per_point * sizeof(double);
      candidate.wide_tile = std::min({std::size_t{256}, options.point_tile, shape.points,
                                      tile_byte_limit / bytes_per_point,
                                      available / bytes_per_point});
      if (candidate.wide_tile &&
          append(candidate, candidate.wide_tile * words_per_point, candidate.wide_scratch) &&
          candidate.bytes <= options.max_workspace_bytes)
        plan = candidate;
    }
  }
  return true;
}
unsigned blocks(std::size_t count) {
  return static_cast<unsigned>(std::min<std::size_t>((count + 127) / 128, 65535));
}
bool disjoint(const void* first, std::size_t first_elements, const void* second,
              std::size_t second_elements) {
  std::size_t left_bytes{}, right_bytes{};
  if (!first || !second || !product(first_elements, sizeof(double), left_bytes) ||
      !product(second_elements, sizeof(double), right_bytes))
    return false;
  const auto left = reinterpret_cast<std::uintptr_t>(first);
  const auto right = reinterpret_cast<std::uintptr_t>(second);
  return left <= right ? right - left >= left_bytes : left - right >= right_bytes;
}
cudaError_t device_buffer(const void* pointer, int device) {
  if (!pointer || reinterpret_cast<std::uintptr_t>(pointer) % alignof(double))
    return cudaErrorInvalidValue;
  cudaPointerAttributes attributes{};
  auto error = cudaPointerGetAttributes(&attributes, pointer);
  if (error != cudaSuccess) return error;
  return attributes.type == cudaMemoryTypeDevice && attributes.device == device
             ? cudaSuccess
             : cudaErrorInvalidValue;
}
cudaError_t drain(cudaError_t error, cudaStream_t stream) {
  const auto completion = cudaStreamSynchronize(stream);
  return error == cudaSuccess ? completion : error;
}
cudaError_t gate(int* status, cudaStream_t stream) {
  int failed{};
  auto error = cudaMemcpyAsync(&failed, status, sizeof(failed), cudaMemcpyDeviceToHost, stream);
  error = drain(error, stream);
  return error == cudaSuccess && failed ? cudaErrorInvalidValue : error;
}
__device__ bool finite(ad::Jet jet) {
  return isfinite(jet.value) && isfinite(jet.left) && isfinite(jet.right) && isfinite(jet.mixed);
}
__device__ double maximum(ad::Jet jet) {
  return fmax(fmax(fabs(jet.value), fabs(jet.left)), fmax(fabs(jet.right), fabs(jet.mixed)));
}
__device__ ad::Jet scale(ad::Jet jet, int exponent) {
  // The same representable binary factor applies to all four lanes. A
  // normal FP64 factor preserves ldexp rounding, including subnormal
  // results, without four independent exponent-decomposition paths.
  if (exponent == 0) return jet;
  if (exponent >= -1022 && exponent <= 1023) {
    const auto factor = __hiloint2double((exponent + 1023) << 20, 0);
    return {jet.value * factor, jet.left * factor, jet.right * factor, jet.mixed * factor};
  }
  return {ldexp(jet.value, exponent), ldexp(jet.left, exponent), ldexp(jet.right, exponent),
          ldexp(jet.mixed, exponent)};
}
__device__ bool normalize(ad::Jet& jet, long long& exponent) {
  if (!finite(jet)) return false;
  const auto magnitude = maximum(jet);
  if (!magnitude) {
    exponent = 0;
    return true;
  }
  // For a normal finite positive binary64, frexp's exponent is the stored
  // exponent minus 1022. Let libdevice handle subnormal inputs.
  const auto biased = static_cast<int>(__double_as_longlong(magnitude) >> 52);
  int shift = biased - 1022;
  if (!biased) (void)frexp(magnitude, &shift);
  jet = scale(jet, -shift);
  exponent += shift;
  return true;
}
// Scale the entire jet by one exact binary factor, including a zero primal's
// nonzero tangents. AD owns multiplication; exponent bookkeeping only changes
// representation and cancels between numerator and denominator at this point.
__device__ bool multiply(ad::Jet& running, long long& exponent, ad::Jet factor) {
  long long factor_exponent{};
  if (!normalize(factor, factor_exponent)) return false;
  running = ad::multiply(running, factor);
  exponent += factor_exponent;
  return normalize(running, exponent);
}
__device__ ad::Jet becke(ad::Jet coordinate, unsigned iterations) {
  if (coordinate.value <= -1)
    coordinate = {-1, 0, 0, 0};
  else if (coordinate.value >= 1)
    coordinate = {1, 0, 0, 0};
  ad::Jet value;
  switch (iterations) {
    case 1:
      value = ad::becke_1(coordinate);
      break;
    case 2:
      value = ad::becke_2(coordinate);
      break;
    case 3:
      value = ad::becke_3(coordinate);
      break;
    case 4:
      value = ad::becke_4(coordinate);
      break;
    default:
      value = ad::becke_5(coordinate);
      break;
  }
  if (value.value < 0)
    value = {};
  else if (value.value > 1)
    value = {1, 0, 0, 0};
  return value;
}
__device__ ad::Jet partition_switch(ad::Jet coordinate, bool adjusted, double adjustment,
                                    unsigned iterations) {
  if (adjusted) {
    if (coordinate.value <= -1)
      coordinate = {-1, 0, 0, 0};
    else if (coordinate.value >= 1)
      coordinate = {1, 0, 0, 0};
    coordinate = ad::treutler_adjust(coordinate, {adjustment, 0, 0, 0});
  }
  return becke(coordinate, iterations);
}
__device__ bool matches_partition(ad::Jet partition, double weight, double atomic,
                                   CudaGridGeometryOptions options) {
  // Compare in the dimensionless unit; a far-radial measure can be large.
  const auto expected = atomic != 0 ? weight / atomic : 0;
  return (atomic != 0 || weight == 0) && isfinite(expected) &&
         (atomic == 0 || fabs(partition.value - expected) <=
                             options.reference_absolute_tolerance +
                                 options.reference_relative_tolerance *
                                     fmax(fabs(partition.value), fabs(expected)));
}
__global__ void finite_directions(const double* left, const double* right, std::size_t count,
                                  int* status) {
  for (std::size_t index = blockIdx.x * blockDim.x + threadIdx.x; index < count;
       index += static_cast<std::size_t>(blockDim.x) * gridDim.x)
    if (!isfinite(left[index]) || !isfinite(right[index])) atomicExch(status, 1);
}
__global__ void finite_values(const double* values, std::size_t count, int* status) {
  if (atomicAdd(status, 0)) return;
  for (std::size_t index = blockIdx.x * blockDim.x + threadIdx.x; index < count;
       index += static_cast<std::size_t>(blockDim.x) * gridDim.x)
    if (!isfinite(values[index])) atomicExch(status, 1);
}
// The generic AO leaf trusts normalized AoBasis records. Validate indices on
// device before that leaf or nuclear-direction indexing can dereference them.
__global__ void validate_basis(CudaGridGeometryShape shape, const double* basis, int* status) {
  if (atomicAdd(status, 0)) return;
  const auto* primitives = basis + 3 * shape.atoms;
  const auto* records = primitives + 2 * shape.primitives;
  if (!blockIdx.x && !threadIdx.x && shape.ao_primitive_terms) {
    double incidences{};
    for (std::size_t index = 0; index < shape.nao; ++index) {
      const auto* record = records + 16 * index;
      incidences += record[2] * record[3];
    }
    if (!isfinite(incidences) || incidences > shape.ao_primitive_terms) atomicExch(status, 1);
  }
  for (std::size_t index = blockIdx.x * blockDim.x + threadIdx.x;
       index < max(shape.nao, shape.primitives);
       index += static_cast<std::size_t>(blockDim.x) * gridDim.x) {
    if (index < shape.primitives && !(primitives[2 * index] > 0)) atomicExch(status, 1);
    if (index >= shape.nao) continue;
    const auto* record = records + 16 * index;
    const auto atom = record[0], begin = record[1], count = record[2], terms = record[3];
    if (atom < 0 || atom >= shape.atoms || floor(atom) != atom || begin < 0 ||
        begin > shape.primitives || floor(begin) != begin || count < 1 ||
        count > shape.primitives - begin || floor(count) != count || terms < 1 || terms > 3 ||
        floor(terms) != terms) {
      atomicExch(status, 1);
      continue;
    }
    if (shape.radial_s && (terms != 1 || record[4] != 0 || record[5] != 0 || record[6] != 0)) {
      atomicExch(status, 1);
      continue;
    }
    for (unsigned term = 0; term < static_cast<unsigned>(terms); ++term)
      for (unsigned axis = 0; axis < 3; ++axis) {
        const auto power = record[4 + 4 * term + axis];
        if (power < 0 || power > 3 || floor(power) != power) atomicExch(status, 1);
      }
  }
}
template <bool cartesian_rows = false>
__global__ void pair_geometry(const double* centers, const double* left, const double* right,
                              const generativeqc_grid_pair_geometry::CenterPair* prepared,
                              const double* radii, std::size_t atoms, double tolerance,
                              PartitionPair* pairs, int* status) {
  if (atomicAdd(status, 0)) return;
  if constexpr (cartesian_rows) pairs += blockIdx.y * (atoms * (atoms - 1) / 2);
  for (std::size_t index = blockIdx.x * blockDim.x + threadIdx.x; index < atoms * atoms;
       index += static_cast<std::size_t>(blockDim.x) * gridDim.x) {
    const auto first = index / atoms, second = index % atoms;
    if (second >= first) continue;
    double left_delta[3], right_delta[3];
    for (unsigned axis = 0; axis < 3; ++axis) {
      if constexpr (cartesian_rows)
        left_delta[axis] = int(3 * first + axis == blockIdx.y) - int(3 * second + axis == blockIdx.y);
      else
        left_delta[axis] = left[3 * first + axis] - left[3 * second + axis];
      right_delta[axis] = right[3 * first + axis] - right[3 * second + axis];
    }
    ad::Jet distance;
    if (prepared) {
      const auto pair = generativeqc_grid_pair_geometry::center_pair_index(first, second);
      distance = ad::prepared_center_norm(prepared[pair], left_delta, right_delta);
    } else {
      ad::Jet coordinates[3];
      for (unsigned axis = 0; axis < 3; ++axis)
        coordinates[axis] = {centers[3 * first + axis] - centers[3 * second + axis],
                             left_delta[axis], right_delta[axis], 0};
      distance = ad::norm(coordinates[0], coordinates[1], coordinates[2]);
    }
    if (!finite(distance) || distance.value <= tolerance) {
      atomicExch(status, 1);
      continue;
    }
    // The divisor and its tangents are identical for all grid points. The
    // generated quotient is numerator times its reciprocal jet. Keep the
    // ordinary quotient in the extended exponent domain or when the
    // reciprocal's derivatives are not separately representable.
    ad::Jet reciprocal{};
    if (distance.value >= 4.9090934652977266e-91 && distance.value <= 2.0370359763344861e90) {
      const auto inverse = ad::ratio({1, 0, 0, 0}, distance);
      if (finite(inverse)) reciprocal = inverse;
    }
    // Treutler's radius adjustment depends only on the two elements, not the
    // grid point or its motion. Compute it once per pair for this dispatch.
    double adjustment = 0;
    if (radii) {
      const double ratio = sqrt(radii[second] / radii[first]);
      adjustment = fmax(-0.5, fmin(0.5, 0.25 * (ratio - 1 / ratio)));
    }
    pairs[generativeqc_grid_pair_geometry::center_pair_index(first, second)] =
        {distance, reciprocal, adjustment};
  }
}
/** Prefix metadata is fixed under nuclear motion; no density pruning or
 * quadrature regeneration enters first/mixed geometry derivatives. */
__device__ std::size_t grid_owner(CudaGridGeometryShape shape, const std::uint64_t* offsets,
                                  std::size_t point) {
  if (!offsets) return point / shape.points_per_atom;
  std::size_t low = 0, high = shape.atoms;
  while (low + 1 < high) {
    const auto middle = (low + high) / 2;
    if (offsets[middle] <= point)
      low = middle;
    else
      high = middle;
  }
  return low;
}

/** Authenticate the stationary quadrature without building any directional
 * jets. Pair lanes share complementary switches; each atom then consumes its
 * own factors in the same order and scale as the general partition kernel. */
__global__ void partition_reference_weights(CudaGridGeometryShape shape,
                                             CudaGridGeometryOptions options,
                                             const double* centers, const double* points,
                                             const double* weights, const double* atomic_weights,
                                             const std::uint64_t* offsets, bool adjusted,
                                             const PartitionPair* pairs, int* status) {
  if (__syncthreads_or(!threadIdx.x && atomicAdd(status, 0))) return;
  const auto point = static_cast<std::size_t>(blockIdx.x);
  const auto atoms = shape.atoms;
  const auto atom = static_cast<std::size_t>(threadIdx.x);
  extern __shared__ double reference_scratch[];
  auto* distances = reference_scratch;
  auto* products = distances + atoms;
  auto* exponents = reinterpret_cast<long long*>(products + atoms);
  auto* switches = reinterpret_cast<double*>(exponents + atoms);
  const auto measure = atomic_weights[point];
  if (atoms == 1) {
    if (!atom && !matches_partition({1, 0, 0, 0}, weights[point], measure, options))
      atomicExch(status, 1);
    return;
  }
  bool valid = true;
  if (atom < atoms) {
    const auto dx = points[3 * point] - centers[3 * atom];
    const auto dy = points[3 * point + 1] - centers[3 * atom + 1];
    const auto dz = points[3 * point + 2] - centers[3 * atom + 2];
    const auto square = (dx * dx + dy * dy) + dz * dz;
    const auto distance = sqrt(square);
    valid = isfinite(distance) && distance > 0;
    // Retain the general norm's domain check at exceptional exponents.
    if (valid && (square < 0x1p-500 || square > 0x1p500))
      valid = finite(ad::norm({dx, 0, 0, 0}, {dy, 0, 0, 0}, {dz, 0, 0, 0}));
    distances[atom] = distance;
  }
  if (!__syncthreads_and(valid)) {
    if (!atom) atomicExch(status, 1);
    return;
  }
  if (atom < atoms) {
    const auto pair_count = (atoms - 1) / 2 + (atoms % 2 == 0 && atom < atoms / 2);
    for (std::size_t step = 1; step <= pair_count; ++step) {
      const auto other = (atom + step < atoms) ? atom + step : atom + step - atoms;
      const auto first = max(atom, other), second = min(atom, other);
      const auto index = generativeqc_grid_pair_geometry::center_pair_index(first, second);
      const auto pair = pairs[index];
      const auto difference = distances[first] - distances[second];
      const auto coordinate = pair.reciprocal.value ? difference * pair.reciprocal.value
                                                   : difference / pair.distance.value;
      const auto factor = partition_switch({coordinate, 0, 0, 0}, adjusted, pair.adjustment,
                                           options.partition_iterations);
      valid = valid && isfinite(coordinate) && finite(factor);
      switches[index] = factor.value;
    }
  }
  if (!__syncthreads_and(valid)) {
    if (!atom) atomicExch(status, 1);
    return;
  }
  if (atom < atoms) {
    ad::Jet product{1, 0, 0, 0};
    long long exponent{};
    for (std::size_t other = 0; other < atoms; ++other) {
      if (other == atom) continue;
      const auto index = generativeqc_grid_pair_geometry::center_pair_index(max(atom, other),
                                                                            min(atom, other));
      const auto factor = atom > other ? switches[index] : 1 - switches[index];
      if (factor == 0) {
        product = {};
        break;
      }
      valid = multiply(product, exponent, {factor, 0, 0, 0}) && valid;
    }
    products[atom] = product.value;
    exponents[atom] = exponent;
  }
  if (!__syncthreads_and(valid)) {
    if (!atom) atomicExch(status, 1);
    return;
  }
  if (atom) return;
  long long common = LLONG_MIN;
  for (std::size_t center = 0; center < atoms; ++center)
    if (products[center]) common = max(common, exponents[center]);
  if (common == LLONG_MIN) {
    atomicExch(status, 1);
    return;
  }
  const auto owner = grid_owner(shape, offsets, point);
  double total{}, numerator{};
  for (std::size_t center = 0; center < atoms; ++center) {
    const auto shift = exponents[center] - common;
    const auto value = products[center] && shift >= -1074
                           ? scale({products[center], 0, 0, 0}, static_cast<int>(shift)).value
                           : 0.0;
    total += value;
    if (center == owner) numerator = value;
  }
  const auto value = numerator / total;
  if (!(total > 0) || !isfinite(value * measure) ||
      !matches_partition({value, 0, 0, 0}, weights[point], measure, options))
    atomicExch(status, 1);
}

__global__ void validate_grid_topology(CudaGridGeometryShape shape, const std::uint64_t* offsets,
                                       const double* radii, int* status) {
  for (std::size_t atom = blockIdx.x * blockDim.x + threadIdx.x; atom < shape.atoms;
       atom += std::size_t(blockDim.x) * gridDim.x) {
    if (offsets[atom + 1] <= offsets[atom] || !isfinite(radii[atom]) || radii[atom] <= 0 ||
        offsets[0] || offsets[shape.atoms] != shape.points)
      atomicExch(status, 1);
  }
}

__global__ void cartesian_pair_geometry(const double* centers, const double* radii,
                                         std::size_t atoms, std::size_t column,
                                         double tolerance, CartesianPair* pairs) {
  for (std::size_t index = blockIdx.x * blockDim.x + threadIdx.x; index < atoms * atoms;
       index += blockDim.x * gridDim.x) {
    const auto first = index / atoms, second = index % atoms;
    if (second >= first) continue;
    auto& pair = pairs[generativeqc_grid_pair_geometry::center_pair_index(first, second)];
    pair.adjustment = 0;
    if (radii) {
      const auto ratio = sqrt(radii[second] / radii[first]);
      pair.adjustment = fmax(-0.5, fmin(0.5, 0.25 * (ratio - 1 / ratio)));
    }
    for (unsigned left = 0; left < 3; ++left) {
      ad::Jet separation[3];
      for (unsigned axis = 0; axis < 3; ++axis)
        separation[axis] = {centers[3 * first + axis] - centers[3 * second + axis],
                            static_cast<double>(left == axis),
                            static_cast<double>(3 * first + axis == column) -
                                static_cast<double>(3 * second + axis == column),
                            0};
      const auto distance = ad::norm(separation[0], separation[1], separation[2]);
      pair.inverse[left] = finite(distance) && distance.value > tolerance &&
                                  distance.value >= 0x1p-32 && distance.value <= 0x1p32
                              ? ad::ratio({1, 0, 0, 0}, distance)
                              : ad::Jet{};
    }
  }
}

// A right dual packs a scalar and its right derivative. Arithmetic still
// uses the existing generated product and sum rules; zero left lanes vanish.
struct RightDual {
  double value, right;
};
__device__ RightDual primal_dual(ad::Jet jet) { return {jet.value, jet.right}; }
__device__ RightDual left_dual(ad::Jet jet) { return {jet.left, jet.mixed}; }
__device__ ad::Jet dual_jet(RightDual dual) { return {dual.value, 0, dual.right, 0}; }
__device__ RightDual dual_product(RightDual first, RightDual second) {
  return primal_dual(ad::multiply(dual_jet(first), dual_jet(second)));
}
__device__ RightDual dual_add(RightDual first, RightDual second) {
  return primal_dual(ad::add(dual_jet(first), dual_jet(second)));
}
__device__ RightDual dual_subtract(RightDual first, RightDual second) {
  return primal_dual(ad::subtract(dual_jet(first), dual_jet(second)));
}

/** One warp owns a point. An atom lane differentiates its entire product
 * using exclusive prefix/suffix products, then the warp folds atomic rows.
 * Each factor depends on just its two centers with the point held fixed.
 * Move the point with its owner by assigning the negative sum of fixed-point
 * center derivatives to that owner (translation of the entire system).
 */
template <bool compact, bool packed_pairs = false, bool wide = false, bool block_local = false>
__global__ void cartesian_partition_rows(CudaGridGeometryShape shape,
                                         CudaGridGeometryOptions options, const double* centers,
                                         const double* points, const double* weights,
                                         const double* atomic_weights, const std::uint64_t* offsets,
                                         bool adjusted, const CartesianPair* pairs,
                                         std::size_t column, double* rows, double* accepted,
                                         double* right_weights, std::size_t begin = 0,
                                         RightDual* scratch = nullptr,
                                         unsigned* failure_count = nullptr,
                                         unsigned repair_capacity = 0, int* status = nullptr) {
  const auto point = begin + static_cast<std::size_t>(blockIdx.x);
  const auto atom = static_cast<std::size_t>(threadIdx.x);
  const auto owner = grid_owner(shape, offsets, point);
  const auto atoms = shape.atoms;
  constexpr auto shared_atoms = block_local ? shared_wide_cartesian_atoms
                                : wide ? cuda_grid_tiled_cartesian_atoms
                                : packed_pairs ? cuda_grid_cartesian_atoms
                                               : unpacked_cartesian_atoms;
  constexpr auto shared_slots = wide && !block_local ? 1 : shared_atoms * shared_atoms;
  constexpr auto slope_slots = packed_pairs ? 1 : shared_slots;
  constexpr auto product_slots = wide ? shared_atoms : 1;
  static_assert(!wide || (compact && packed_pairs));
  static_assert(!block_local || wide);
  static_assert((2 * shared_slots + slope_slots + product_slots) * sizeof(RightDual) +
                    3 * shared_atoms * sizeof(ad::Jet) <= 48 * 1024);
  const auto slots = wide ? atoms : shared_atoms;
  __shared__ ad::Jet norm[shared_atoms][3];
  __shared__ RightDual shared_prefix[shared_slots], shared_switches[shared_slots];
  __shared__ RightDual slopes[slope_slots];
  __shared__ RightDual atom_products[product_slots];
  auto* prefix = shared_prefix;
  auto* switches = shared_switches;
  if constexpr (wide && !block_local) {
    prefix = scratch + 2 * blockIdx.x * slots * slots;
    switches = prefix + slots * slots;
  }
  const auto join = [] {
    if constexpr (wide) __syncthreads();
    else __syncwarp();
  };
  const auto all = [](bool valid) {
    if constexpr (wide) return __syncthreads_and(valid) != 0;
    else return __all_sync(0xffffffffU, valid) != 0;
  };
  const auto finish = [&](bool valid) {
    if (atom) return;
    if constexpr (wide) {
      if (!valid) {
        const auto slot = atomicAdd(failure_count, 1U);
        if (slot < repair_capacity) failure_count[1 + slot] = static_cast<unsigned>(point);
        else atomicExch(status, 1);
      }
    } else {
      accepted[point] = static_cast<double>(valid);
    }
  };
  const auto load_switch = [&](std::size_t other) {
    const auto index = generativeqc_grid_pair_geometry::center_pair_index(max(atom, other),
                                                                          min(atom, other));
    const auto primal = switches[2 * index];
    const auto slope = switches[2 * index + 1];
    const ad::Jet jet{primal.value, slope.value, primal.right, slope.right};
    return atom > other ? jet : ad::complement(jet);
  };
  const auto measure = atomic_weights[point];
  bool valid = fabs(measure) <= 0x1p32;
  if (atom < atoms) {
    for (unsigned left = 0; left < 3; ++left) {
      ad::Jet relative[3];
      for (unsigned axis = 0; axis < 3; ++axis)
        relative[axis] = {points[3 * point + axis] - centers[3 * atom + axis],
                          left == axis ? -1.0 : 0.0,
                          static_cast<double>(3 * owner + axis == column) -
                              static_cast<double>(3 * atom + axis == column),
                          0};
      const auto radial = ad::norm(relative[0], relative[1], relative[2]);
      valid = valid && maximum(radial) <= 0x1p32 && finite(radial) && radial.value >= 0x1p-32;
      norm[atom][left] = radial;
    }
  }
  if (!all(valid)) {
    finish(false);
    return;
  }
  join();
  if (atom < atoms) {
    // An unordered pair supplies complementary switches. Walk cyclic offsets
    // to share the expensive switch/jet evaluation without concentrating a
    // triangular workload in the last lanes. For even sizes, break the tie
    // between opposite lanes so every shared-memory entry has one writer.
    const auto pairs_per_lane = (atoms - 1) / 2 + (atoms % 2 == 0 && atom < atoms / 2);
    for (std::size_t offset = 1; offset <= pairs_per_lane; ++offset) {
      const auto other = atom + offset < atoms ? atom + offset : atom + offset - atoms;
      const auto first = max(atom, other), second = min(atom, other);
      const auto& pair = pairs[generativeqc_grid_pair_geometry::center_pair_index(first, second)];
      const auto inverse = pair.inverse[0];
      const auto numerator = ad::subtract(norm[first][0], norm[second][0]);
      const auto relative = dual_product(primal_dual(numerator), primal_dual(inverse));
      const auto factor = partition_switch({relative.value, 1, relative.right, 0}, adjusted,
                                           pair.adjustment, options.partition_iterations);
      const auto complement = ad::complement(factor);
      valid = valid && inverse.value > 0 && finite(factor) && finite(complement) &&
              maximum(factor) <= 0x1p64 && maximum(complement) <= 0x1p64;
      if constexpr (packed_pairs) {
        // Two duals per unordered pair fit in the same plane that will hold
        // one dual per ordered row after the suffix pass's warp barrier.
        const auto index = generativeqc_grid_pair_geometry::center_pair_index(first, second);
        switches[2 * index] = primal_dual(factor);
        switches[2 * index + 1] = left_dual(factor);
      } else {
        switches[second * slots + first] = primal_dual(factor);
        switches[first * slots + second] = primal_dual(complement);
        slopes[second * slots + first] = left_dual(factor);
        slopes[first * slots + second] = left_dual(complement);
      }
    }
  }
  // The paired stores cross lanes; finish them before preserving the
  // original ordered prefix/suffix products in each atom lane.
  join();
  RightDual product{};
  bool inactive = false;
  if (atom < atoms) {
    product = {1, 0};
    double nonzero_product = 1;
    unsigned zero_duals = 0;
    bool regular_factors = true;
    for (std::size_t other = 0; other < atoms; ++other) {
      if (other == atom) continue;
      RightDual switched;
      if constexpr (packed_pairs)
        switched = primal_dual(load_switch(other));
      else
        switched = switches[other * slots + atom];
      // Check the full nonzero product, not just the individual factors:
      // the number of factors varies with geometry size. Exact zero remains
      // valid because exclusive products do not divide by a factor.
      regular_factors = regular_factors && (switched.value == 0 || switched.value >= 0x1p-56);
      if (switched.value != 0) nonzero_product *= switched.value;
      else if constexpr (wide) zero_duals += switched.right == 0;
      prefix[other * slots + atom] = product;
      product = dual_product(product, switched);
    }
    // Removing one factor for a left derivative still leaves an exact zero
    // right dual when two are present. No quotient by a switch is necessary.
    if constexpr (wide) inactive = zero_duals >= 2;
    valid = valid && (inactive || (regular_factors && nonzero_product >= 0x1p-952));
  }
  if (!all(valid)) {
    finish(false);
    return;
  }
  if (atom < atoms && !inactive) {
    RightDual suffix{1, 0};
    for (std::size_t other = atoms; other-- > 0;) {
      if (other == atom) continue;
      const auto exclusive = dual_product(prefix[other * slots + atom], suffix);
      RightDual switched, slope;
      if constexpr (packed_pairs) {
        const auto jet = load_switch(other);
        switched = primal_dual(jet);
        slope = left_dual(jet);
      } else {
        switched = switches[other * slots + atom];
        slope = slopes[other * slots + atom];
      }
      suffix = dual_product(switched, suffix);
      // This prefix entry is dead; keep the full exclusive slope here.
      prefix[other * slots + atom] = dual_product(exclusive, slope);
    }
  } else if constexpr (wide) {
    if (atom < atoms)
      for (std::size_t other = 0; other < atoms; ++other) prefix[other * slots + atom] = {};
  }
  RightDual sum{}, numerator{};
  if constexpr (wide) {
    if (atom < atoms) atom_products[atom] = product;
    join();
    if (!atom) {
      for (std::size_t center = 0; center < atoms; ++center) sum = dual_add(sum, atom_products[center]);
      numerator = atom_products[owner];
    }
    join();
    if (!atom) {
      atom_products[0] = sum;
      atom_products[1] = numerator;
    }
    join();
    sum = atom_products[0];
    numerator = atom_products[1];
  } else {
    join();
    sum = product;
    for (int stride = 16; stride; stride /= 2)
      sum = dual_add(sum, {__shfl_xor_sync(0xffffffffU, sum.value, stride),
                           __shfl_xor_sync(0xffffffffU, sum.right, stride)});
    numerator = {__shfl_sync(0xffffffffU, product.value, owner),
                 __shfl_sync(0xffffffffU, product.right, owner)};
  }
  // A regular denominator and measure bound amplification of small products.
  // The scaled path below owns the extended exponent domain.
  if (!(sum.value >= 0x1p-48 && isfinite(sum.right))) {
    finish(false);
    return;
  }
  const auto inverse = primal_dual(ad::ratio({1, 0, 0, 0}, dual_jet(sum)));
  const auto partition = dual_product(numerator, inverse);
  const auto weighted_right = measure * partition.right;
  if (!atom) {
    valid = valid && isfinite(weighted_right) &&
            matches_partition(dual_jet(partition), weights[point], measure, options);
    if constexpr (compact) right_weights[point] = weighted_right;
  }
  if constexpr (wide) {
    // At a fixed point a center's tangent sees only its incident switches.
    // Each such switch contributes to the two complementary atomic products;
    // combine their slopes while the common motion is still in registers.
    if (atom < atoms && atom != owner) {
      RightDual denominator_tangent[3]{}, numerator_tangent[3]{};
      for (std::size_t other = 0; other < atoms; ++other) {
        if (other == atom) continue;
        const auto first = max(atom, other), second = min(atom, other);
        const auto& pair = pairs[generativeqc_grid_pair_geometry::center_pair_index(first, second)];
        const auto self = prefix[other * slots + atom];
        const auto neighbor = prefix[atom * slots + other];
        const auto total = dual_add(self, neighbor);
        const auto selected = other == owner ? neighbor : RightDual{};
#pragma unroll
        for (unsigned axis = 0; axis < 3; ++axis) {
          const auto reciprocal = pair.inverse[axis];
          const auto difference =
              dual_subtract(primal_dual(norm[first][axis]), primal_dual(norm[second][axis]));
          const auto motion = dual_add(
              dual_product(difference, left_dual(reciprocal)),
              dual_product(left_dual(norm[atom][axis]), primal_dual(reciprocal)));
          const auto signed_motion = atom == first ? motion : dual_subtract({}, motion);
          denominator_tangent[axis] =
              dual_add(denominator_tangent[axis], dual_product(total, signed_motion));
          numerator_tangent[axis] =
              dual_add(numerator_tangent[axis], dual_product(selected, signed_motion));
        }
      }
#pragma unroll
      for (unsigned axis = 0; axis < 3; ++axis) {
        const auto derivative =
            dual_product(dual_subtract(numerator_tangent[axis],
                                       dual_product(partition, denominator_tangent[axis])),
                         inverse);
        // The switch plane is dead after the suffix pass. Only the three
        // normalized row values are needed to translate the grid owner.
        switches[axis * slots + atom] = dual_product(derivative, {measure, 0});
      }
    }
    join();
    if (atom < atoms) {
#pragma unroll
      for (unsigned axis = 0; axis < 3; ++axis) {
        RightDual published{};
        if (atom == owner) {
          // The point moves with this center. Translate after projecting the
          // fixed-point rows, using the same right dual for the whole column.
          for (std::size_t other = 0; other < atoms; ++other)
            if (other != owner)
              published = dual_subtract(published, switches[axis * slots + other]);
        } else {
          published = switches[axis * slots + atom];
        }
        valid = valid && isfinite(published.value) && isfinite(published.right);
        auto* row = rows + 2 * (3 * atom + axis) * shape.points;
        row[point] = published.value;
        row[shape.points + point] = published.right;
      }
    }
  } else {
    // The switch planes are dead. Publish one axis at a time through the same
    // shared plane instead of retaining three atom-by-atom derivative tables.
    for (unsigned axis = 0; axis < 3; ++axis) {
      if (atom < atoms && inactive) {
        // Other lanes still reduce this atom product's column. Overwrite the
        // packed switch data, but no fixed-point center derivative is needed.
        for (std::size_t row = 0; row < atoms; ++row) switches[row * slots + atom] = {};
      } else if (atom < atoms) {
        RightDual diagonal{};
        for (std::size_t other = atoms; other-- > 0;) {
          if (other == atom) continue;
          const auto first = max(atom, other), second = min(atom, other);
          const auto& pair =
              pairs[generativeqc_grid_pair_geometry::center_pair_index(first, second)];
          const auto slope = prefix[other * slots + atom];
          // At a fixed point, the denominator's two center tangents have
          // opposite signs; the numerator has separate norm jets.
          const auto inverse = pair.inverse[axis];
          const auto numerator = ad::subtract(norm[first][axis], norm[second][axis]);
          const auto denominator_motion = dual_product(primal_dual(numerator), left_dual(inverse));
          const auto first_motion = dual_add(
              denominator_motion, dual_product(left_dual(norm[first][axis]), primal_dual(inverse)));
          const auto second_motion =
              dual_add(denominator_motion,
                       dual_product(left_dual(norm[second][axis]), primal_dual(inverse)));
          const auto first_derivative = dual_product(slope, first_motion);
          const auto second_derivative = dual_subtract({}, dual_product(slope, second_motion));
          diagonal = dual_add(diagonal, atom == first ? first_derivative : second_derivative);
          switches[other * slots + atom] = atom == first ? second_derivative : first_derivative;
        }
        switches[atom * slots + atom] = diagonal;
        // Opposite fixed-point motion of all other centers gives the moving
        // owner's derivative by translational invariance.
        RightDual translated{};
        for (std::size_t other = 0; other < atoms; ++other)
          if (other != owner)
            translated = dual_subtract(translated, switches[other * slots + atom]);
        switches[owner * slots + atom] = translated;
      }
      join();
      if (atom < atoms) {
        RightDual tangent{};
        for (std::size_t row = 0; row < atoms; ++row)
          tangent = dual_add(tangent, switches[atom * slots + row]);
        // In the right dual algebra, d_left(N/S) = (d_left(N) - (N/S)*d_left(S))/S.
        // The denominator and the right-only partition do not depend on which
        // left row is requested; reuse them for its first and mixed derivatives.
        const auto derivative = dual_product(
            dual_subtract(switches[atom * slots + owner], dual_product(partition, tangent)),
            inverse);
        const auto published = dual_product(derivative, {measure, 0});
        valid = valid && isfinite(published.value) && isfinite(published.right);
        const auto coordinate = 3 * atom + axis;
        auto* row = rows + (compact ? 2 : 3) * coordinate * shape.points;
        row[point] = published.value;
        if constexpr (!compact) row[shape.points + point] = weighted_right;
        row[(compact ? 1 : 2) * shape.points + point] = published.right;
      }
      join();
    }
  }
  finish(all(valid));
}

// A diagonal unit column has identical left and right motion. Carry its
// second derivative through each product directly; no exclusive products or
// all-center slope matrix are needed for the one output row.
__global__ void cartesian_partition_diagonal(
    CudaGridGeometryShape shape, CudaGridGeometryOptions options, const double* centers,
    const double* points, const double* weights, const double* atomic_weights,
    const std::uint64_t* offsets, bool adjusted, const CartesianPair* pairs, std::size_t column,
    double* rows, double* accepted) {
  __shared__ ad::Jet distances[cuda_grid_cartesian_atoms], products[cuda_grid_cartesian_atoms];
  const auto point = static_cast<std::size_t>(blockIdx.x);
  const auto atom = static_cast<std::size_t>(threadIdx.x);
  const auto owner = grid_owner(shape, offsets, point);
  const auto right_atom = column / 3;
  const auto right_axis = column % 3;
  const auto measure = atomic_weights[point];
  bool valid = fabs(measure) <= 0x1p32;
  if (atom < shape.atoms) {
    ad::Jet relative[3];
    for (unsigned axis = 0; axis < 3; ++axis) {
      const auto motion = axis == right_axis ? static_cast<double>(owner == right_atom) -
                                                   static_cast<double>(atom == right_atom)
                                             : 0.0;
      relative[axis] = {points[3 * point + axis] - centers[3 * atom + axis], motion, motion, 0};
    }
    const auto radial = ad::norm(relative[0], relative[1], relative[2]);
    valid = valid && finite(radial) && maximum(radial) <= 0x1p32 && radial.value >= 0x1p-32;
    distances[atom] = radial;
  }
  if (!__all_sync(0xffffffffU, valid)) {
    if (!atom) accepted[point] = 0;
    return;
  }
  __syncwarp();
  if (atom < shape.atoms) {
    ad::Jet product{1, 0, 0, 0};
    double nonzero_product = 1;
    for (std::size_t other = 0; other < shape.atoms; ++other) {
      if (atom == other) continue;
      const auto first = max(atom, other), second = min(atom, other);
      const auto& pair = pairs[generativeqc_grid_pair_geometry::center_pair_index(first, second)];
      const auto inverse = pair.inverse[right_axis];
      const auto motion = static_cast<double>(first == right_atom) -
                          static_cast<double>(second == right_atom);
      // The prepared inverse differentiates the first center on the left.
      // The actual column's signed displacement supplies that seed instead.
      const auto coordinate = ad::multiply(ad::subtract(distances[first], distances[second]),
                                           {inverse.value, inverse.right, inverse.right,
                                            motion * inverse.mixed});
      const auto factor = partition_switch(coordinate, adjusted, pair.adjustment,
                                           options.partition_iterations);
      const auto switched = atom == first ? factor : ad::complement(factor);
      valid = valid && inverse.value > 0 && finite(switched) && maximum(switched) <= 0x1p64 &&
              (switched.value == 0 || switched.value >= 0x1p-56);
      if (switched.value != 0) nonzero_product *= switched.value;
      product = ad::multiply(product, switched);
    }
    valid = valid && nonzero_product >= 0x1p-952 && finite(product);
    products[atom] = product;
  }
  if (!__all_sync(0xffffffffU, valid)) {
    if (!atom) accepted[point] = 0;
    return;
  }
  __syncwarp();
  if (!atom) {
    ad::Jet sum{};
    for (std::size_t center = 0; center < shape.atoms; ++center)
      sum = ad::add(sum, products[center]);
    if (!(sum.value >= 0x1p-48 && finite(sum))) {
      accepted[point] = 0;
      return;
    }
    const auto partition = ad::ratio(products[owner], sum);
    const auto published = ad::multiply(partition, {measure, 0, 0, 0});
    accepted[point] = finite(published) && matches_partition(partition, weights[point], measure,
                                                            options);
    rows[point] = published.left;
    rows[shape.points + point] = published.right;
    rows[2 * shape.points + point] = published.mixed;
  }
}

template <bool diagonal_only, bool compact = false>
cudaError_t launch_cartesian_rows(CudaGridGeometryShape shape, CudaGridGeometryOptions options,
                                  const CudaGridGeometryReference& reference,
                                  const CartesianPair* pairs, std::size_t column, double* rows,
                                  double* accepted, cudaStream_t stream,
                                  double* right_weights = nullptr) {
  if constexpr (diagonal_only)
    cartesian_partition_diagonal<<<static_cast<unsigned>(shape.points), 32, 0, stream>>>(
        shape, options, reference.basis, reference.points, reference.weights,
        reference.atomic_weights, reference.atom_offsets, reference.atomic_radii != nullptr, pairs,
        column, rows, accepted);
  else if (shape.atoms > unpacked_cartesian_atoms)
    cartesian_partition_rows<compact, true><<<static_cast<unsigned>(shape.points), 32, 0, stream>>>(
        shape, options, reference.basis, reference.points, reference.weights,
        reference.atomic_weights, reference.atom_offsets, reference.atomic_radii != nullptr, pairs,
        column, rows, accepted, right_weights);
  else
    cartesian_partition_rows<compact><<<static_cast<unsigned>(shape.points), 32, 0, stream>>>(
        shape, options, reference.basis, reference.points, reference.weights,
        reference.atomic_weights, reference.atom_offsets, reference.atomic_radii != nullptr, pairs,
        column, rows, accepted, right_weights);
  return cudaGetLastError();
}

template <bool block_local, bool cartesian_rows = false, bool repair_queue = false>
__global__ void partition_geometry(CudaGridGeometryShape shape, CudaGridGeometryOptions options,
                                   const double* centers, const double* points,
                                   const double* weights, const double* atomic_weights,
                                   const std::uint64_t* offsets, const double* radii,
                                   const double* left, const double* right,
                                   const PartitionPair* pairs, std::size_t begin, std::size_t count,
                                   ad::Jet* distances, ad::Jet* products, long long* exponents,
                                   double* output, const double* accepted, int* status,
                                   double* right_weights, std::size_t right_column,
                                   const unsigned* failures = nullptr,
                                   const CartesianPair* cartesian_pairs = nullptr) {
  if (atomicAdd(status, 0)) return;
  static_assert(!repair_queue || cartesian_rows);
  if constexpr (repair_queue) count = min(count, static_cast<std::size_t>(failures[0]));
  if constexpr (cartesian_rows) {
    if constexpr (!repair_queue) pairs += blockIdx.y * (shape.atoms * (shape.atoms - 1) / 2);
    output += blockIdx.y * (2 * shape.points);
  }
  if constexpr (block_local) {
    extern __shared__ double scratch_words[];
    const auto jets = shape.atoms * blockDim.x;
    distances = reinterpret_cast<ad::Jet*>(scratch_words);
    products = distances + jets;
    exponents = reinterpret_cast<long long*>(products + jets);
  }
  const auto stride = block_local ? blockDim.x : std::size_t{1};
  for (std::size_t local = blockIdx.x * blockDim.x + threadIdx.x; local < count;
       local += static_cast<std::size_t>(blockDim.x) * gridDim.x) {
    const auto point = repair_queue ? failures[1 + local] : begin + local;
    if (accepted && accepted[point]) continue;
    const auto owner = grid_owner(shape, offsets, point);
    const auto scratch = block_local ? threadIdx.x : local * shape.atoms;
    auto result = ad::Jet{1, 0, 0, 0};
    bool valid = isfinite(atomic_weights[point]) && isfinite(weights[point]);
    if (shape.atoms > 1) {
      for (std::size_t atom = 0; atom < shape.atoms; ++atom) {
        ad::Jet coordinates[3];
        for (unsigned axis = 0; axis < 3; ++axis) {
          double left_delta;
          if constexpr (cartesian_rows)
            left_delta = int(3 * owner + axis == blockIdx.y) - int(3 * atom + axis == blockIdx.y);
          else
            left_delta = left[3 * owner + axis] - left[3 * atom + axis];
          coordinates[axis] = {points[3 * point + axis] - centers[3 * atom + axis], left_delta,
                               right[3 * owner + axis] - right[3 * atom + axis], 0};
        }
        const auto slot = scratch + atom * stride;
        distances[slot] = ad::norm(coordinates[0], coordinates[1], coordinates[2]);
        valid = valid && finite(distances[slot]) && distances[slot].value > 0;
        products[slot] = {1, 0, 0, 0};
        exponents[slot] = 0;
      }
      if (!valid) {
        atomicExch(status, 1);
        continue;
      }
      for (std::size_t first = 0; first < shape.atoms; ++first)
        for (std::size_t second = 0; second < first; ++second) {
          const auto first_slot = scratch + first * stride;
          const auto second_slot = scratch + second * stride;
          const auto pair_index = generativeqc_grid_pair_geometry::center_pair_index(first, second);
          const auto numerator = ad::subtract(distances[first_slot], distances[second_slot]);
          ad::Jet coordinate;
          double adjustment;
          if constexpr (repair_queue) {
            const auto& prepared = cartesian_pairs[pair_index];
            const auto left_atom = blockIdx.y / 3, left_axis = blockIdx.y % 3;
            const auto sign = int(first == left_atom) - int(second == left_atom);
            const auto inverse = prepared.inverse[left_axis];
            if (inverse.value != 0) {
              coordinate = ad::multiply(numerator,
                                        {inverse.value, sign * inverse.left, inverse.right,
                                         sign * inverse.mixed});
            } else {
              ad::Jet delta[3];
              for (unsigned axis = 0; axis < 3; ++axis)
                delta[axis] = {centers[3 * first + axis] - centers[3 * second + axis],
                               axis == left_axis ? static_cast<double>(sign) : 0.0,
                               right[3 * first + axis] - right[3 * second + axis], 0};
              const auto distance = ad::norm(delta[0], delta[1], delta[2]);
              valid = valid && finite(distance) && distance.value > options.coincident_tolerance;
              coordinate = ad::ratio(numerator, distance);
            }
            adjustment = prepared.adjustment;
          } else {
            const auto& geometry = pairs[pair_index];
            coordinate = geometry.reciprocal.value != 0
                             ? ad::multiply(numerator, geometry.reciprocal)
                             : ad::ratio(numerator, geometry.distance);
            adjustment = geometry.adjustment;
          }
          const auto pair = partition_switch(coordinate, radii != nullptr, adjustment,
                                              options.partition_iterations);
          valid = valid && finite(coordinate) && finite(pair);
          valid = multiply(products[first_slot], exponents[first_slot], pair) && valid;
          valid = multiply(products[second_slot], exponents[second_slot], ad::complement(pair)) &&
                  valid;
        }
      long long common = LLONG_MIN;
      for (std::size_t atom = 0; atom < shape.atoms; ++atom) {
        const auto slot = scratch + atom * stride;
        if (maximum(products[slot])) common = max(common, exponents[slot]);
      }
      if (!valid || common == LLONG_MIN) {
        atomicExch(status, 1);
        continue;
      }
      ad::Jet total{}, numerator{};
      // A common exponent keeps every component of the normalized quotient in
      // one unit. Never divide by a pair factor: it may round to exact zero.
      for (std::size_t atom = 0; atom < shape.atoms; ++atom) {
        const auto slot = scratch + atom * stride;
        const auto difference = exponents[slot] - common;
        const auto value = maximum(products[slot]) && difference >= -1074
                               ? scale(products[slot], static_cast<int>(difference))
                               : ad::Jet{};
        total = ad::add(total, value);
        if (atom == owner) numerator = value;
      }
      if (!valid || !finite(total) || !(total.value > 0)) {
        atomicExch(status, 1);
        continue;
      }
      result = ad::ratio(numerator, total);
    }
    // Test the partition in its natural dimensionless unit. Large far-radial
    // measures can amplify one switching ulp beyond a fixed weight tolerance.
    // Division is by the actual atomic measure, never by a partition weight.
    valid = valid && matches_partition(result, weights[point], atomic_weights[point], options);
    result = ad::multiply(result, {atomic_weights[point], 0, 0, 0});
    if (!valid || !finite(result)) atomicExch(status, 1);
    output[point] = result.left;
    if constexpr (cartesian_rows) {
      // Only the diagonal row publishes the right tangent. Its own scaled
      // quotient supplies it even when the fast point sweep was rejected.
      if (blockIdx.y == right_column) right_weights[point] = result.right;
    } else {
      output[shape.points + point] = result.right;
    }
    output[(cartesian_rows ? 1 : 2) * shape.points + point] = result.mixed;
  }
}

__global__ void ao_radial_moments(CudaGridGeometryShape shape, const double* basis,
                                  const double* points, std::size_t count, double* output,
                                  int* status) {
  if (*status) return;
  const auto* primitives = basis + 3 * shape.atoms;
  const auto* records = primitives + 2 * shape.primitives;
  const auto panel = count * shape.nao;
  for (std::size_t index = blockIdx.x * blockDim.x + threadIdx.x; index < panel;
       index += static_cast<std::size_t>(blockDim.x) * gridDim.x) {
    const auto* record = records + 16 * (index % shape.nao);
    const auto atom = static_cast<std::size_t>(record[0]);
    const auto begin = static_cast<std::size_t>(record[1]);
    const auto end = begin + static_cast<std::size_t>(record[2]);
    const auto point = index / shape.nao;
    double distance[3], moments[4]{};
    for (unsigned axis = 0; axis < 3; ++axis)
      distance[axis] = points[3 * point + axis] - basis[3 * atom + axis];
    for (std::size_t primitive = begin; primitive < end; ++primitive)
      ad::accumulate_s_radial(distance, primitives[2 * primitive],
                              record[7] * primitives[2 * primitive + 1], moments);
    for (unsigned order = 0; order < 4; ++order) {
      if (!isfinite(moments[order])) atomicExch(status, 1);
      output[order * panel + index] = moments[order];
    }
  }
}

cudaError_t launch_spatial(CudaGridGeometryShape shape, const double* basis, const double* points,
                           std::size_t count, double* output, int* status, cudaStream_t stream) {
  if (!shape.radial_s)
    return hessian_aot::launch_ao(basis, shape.atoms, shape.primitives, shape.nao, points, count,
                                   20, output, status, stream);
  ao_radial_moments<<<blocks(count * shape.nao), 128, 0, stream>>>(shape, basis, points, count,
                                                                 output, status);
  return cudaGetLastError();
}

template <bool radial>
__global__ void ao_cartesian_direction(CudaGridGeometryShape shape, const double* basis,
                                       const double* spatial, const double* reference_points,
                                       const std::uint64_t* offsets,
                                       std::size_t column, double* output, double* points,
                                       int* status) {
  if (atomicAdd(status, 0)) return;
  const auto panel = shape.points * shape.nao;
  const auto moving_atom = column / 3, moving_axis = column % 3;
  const auto* aos = basis + 3 * shape.atoms + 2 * shape.primitives;
  for (std::size_t index = blockIdx.x * blockDim.x + threadIdx.x; index < panel;
       index += static_cast<std::size_t>(blockDim.x) * gridDim.x) {
    const auto point = index / shape.nao, orbital = index % shape.nao;
    const auto atom = static_cast<std::size_t>(aos[16 * orbital]);
    const bool moves = grid_owner(shape, offsets, point) == moving_atom;
    const int relative = int(moves) - int(atom == moving_atom);
    if (!orbital)
      for (unsigned axis = 0; axis < 3; ++axis)
        points[3 * point + axis] = static_cast<double>(moves && axis == moving_axis);
    double distance[3];
    if constexpr (radial)
      for (unsigned axis = 0; axis < 3; ++axis)
        distance[axis] = reference_points[3 * point + axis] - basis[3 * atom + axis];
    for (unsigned jet = 0; jet < 4; ++jet) {
      const auto first = ad::cartesian_first_index(jet, moving_axis);
      const double value = relative * (radial
          ? ad::s_radial_spatial(first, distance, spatial + index, panel)
          : spatial[first * panel + index]);
      if (!isfinite(value)) atomicExch(status, 1);
      output[jet * panel + index] = value;
    }
  }
}
template <bool radial>
__global__ void ao_geometry(CudaGridGeometryShape shape, const double* basis, const double* spatial,
                            const double* reference_points,
                            std::size_t begin, std::size_t count, const std::uint64_t* offsets,
                            const double* left, const double* right, double* output,
                            double* point_directions, int* status) {
  if (atomicAdd(status, 0)) return;
  const auto panel = shape.points * shape.nao;
  const auto tile_panel = count * shape.nao;
  for (std::size_t index = blockIdx.x * blockDim.x + threadIdx.x; index < tile_panel;
       index += static_cast<std::size_t>(blockDim.x) * gridDim.x) {
    const auto orbital = index % shape.nao, point = begin + index / shape.nao;
    const auto atom =
        static_cast<std::size_t>(basis[3 * shape.atoms + 2 * shape.primitives + 16 * orbital]);
    const auto owner = grid_owner(shape, offsets, point);
    double left_motion[3], right_motion[3], values[12];
    for (unsigned axis = 0; axis < 3; ++axis) {
      left_motion[axis] = left[3 * owner + axis] - left[3 * atom + axis];
      right_motion[axis] = right[3 * owner + axis] - right[3 * atom + axis];
      if (!orbital) {
        point_directions[3 * point + axis] = left[3 * owner + axis];
        point_directions[3 * (shape.points + point) + axis] = right[3 * owner + axis];
      }
    }
    if constexpr (radial) {
      double distance[3], jets[20];
      for (unsigned axis = 0; axis < 3; ++axis)
        distance[axis] = reference_points[3 * point + axis] - basis[3 * atom + axis];
#pragma unroll
      for (unsigned jet = 0; jet < 20; ++jet)
        jets[jet] = ad::s_radial_spatial(jet, distance, spatial + index, tile_panel);
      ad::ao_motion(jets, 1, left_motion, right_motion, values);
    } else {
      ad::ao_motion(spatial + index, tile_panel, left_motion, right_motion, values);
    }
    for (unsigned jet = 0; jet < 4; ++jet)
      for (unsigned direction = 0; direction < 3; ++direction) {
        const auto value = values[3 * jet + direction];
        if (!isfinite(value)) atomicExch(status, 1);
        output[(4 * direction + jet) * panel + begin * shape.nao + index] = value;
      }
  }
}
}  // namespace

cudaError_t cuda_grid_geometry_workspace(const CudaGridGeometryShape& shape,
                                         const CudaGridGeometryOptions& options, std::size_t& bytes,
                                         std::size_t* minimum) noexcept {
  bytes = 0;
  Layout plan;
  std::size_t least{};
  if (minimum) *minimum = 0;
  if (!layout(shape, options, plan, minimum ? &least : nullptr)) return cudaErrorInvalidValue;
  bytes = plan.bytes;
  if (minimum) *minimum = least;
  return cudaSuccess;
}

cudaError_t CudaGridGeometryResponse::authenticate(cudaStream_t stream) noexcept {
  if (!ready_ || stream != reference_.stream) return cudaErrorInvalidResourceHandle;
  int device = -1;
  auto error = cudaGetDevice(&device);
  if (error != cudaSuccess) return error;
  if (device != reference_.device_id) return cudaErrorInvalidResourceHandle;
  cudaStreamCaptureStatus capture{};
  if ((error = cudaStreamIsCapturing(stream, &capture)) != cudaSuccess) return error;
  if (capture != cudaStreamCaptureStatusNone) return cudaErrorStreamCaptureUnsupported;
  error = reference_.authenticate(reference_.owner);
  if (error != cudaSuccess) ready_ = false;
  return error;
}

cudaError_t CudaGridGeometryResponse::partition(const double* left, const double* right,
                                               double* output, const double* accepted,
                                               bool reference_only) noexcept {
  Layout plan;
  if (!layout(shape_, options_, plan)) return cudaErrorInvalidValue;
  auto* base = static_cast<char*>(workspace_);
  auto* status = reinterpret_cast<int*>(base + plan.status);
  auto* pairs = reinterpret_cast<PartitionPair*>(base + plan.pairs);
  const auto stream = reference_.stream;
  pair_geometry<><<<blocks(shape_.atoms * shape_.atoms), 128, 0, stream>>>(
      reference_.basis, left, right, reference_.center_pairs, reference_.atomic_radii, shape_.atoms,
      options_.coincident_tolerance, pairs, status);
  auto error = cudaGetLastError();
  if (error != cudaSuccess) return drain(error, stream);
  if (reference_only && shape_.atoms <= cuda_grid_tiled_cartesian_atoms && shape_.points <= INT_MAX) {
    constexpr auto max_atoms = cuda_grid_tiled_cartesian_atoms;
    static_assert(sizeof(long long) == sizeof(double) &&
                  (3 * max_atoms + max_atoms * (max_atoms - 1) / 2) * sizeof(double) < 40 * 1024);
    const auto atoms = shape_.atoms;
    const auto bytes = (3 * atoms + atoms * (atoms - 1) / 2) * sizeof(double);
    partition_reference_weights<<<shape_.points, (atoms + 31) / 32 * 32, bytes, stream>>>(
        shape_, options_, reference_.basis, reference_.points, reference_.weights,
        reference_.atomic_weights, reference_.atom_offsets, reference_.atomic_radii != nullptr,
        pairs, status);
    return cudaGetLastError();
  }
  // Each lane owns its scratch, staying within default shared-memory limits.
  // Let independent point blocks fill the device for mid-size grids rather
  // than repeatedly dispatching small, globally backed tiles.
  constexpr std::size_t shared_limit = 40 * 1024;
  constexpr std::size_t lane_scratch = 2 * sizeof(ad::Jet) + sizeof(long long);
  // The lane-local fallback has no warp collectives. Half a warp keeps the
  // larger masked Cartesian sweeps within the same per-block scratch cap.
  const auto local_threads = shape_.atoms <= shared_limit / (32 * lane_scratch) ? 32 : 16;
  const auto grid = [](std::size_t count, std::size_t threads) {
    return std::min<std::size_t>(1 + (count - 1) / threads, 65535);
  };
  const auto* basis = reference_.basis;
  auto* weights = output ? output : reinterpret_cast<double*>(base + plan.candidate_weights);
  const auto scratch_per_atom = local_threads * lane_scratch;
  if (shape_.atoms <= shared_limit / scratch_per_atom &&
      (shape_.points <= 16384 || (accepted && shape_.points <= 65535)) &&
      options_.point_tile == CudaGridGeometryOptions{}.point_tile) {
    const auto bytes = shape_.atoms * scratch_per_atom;
    partition_geometry<true><<<grid(shape_.points, local_threads), local_threads, bytes, stream>>>(
        shape_, options_, basis, reference_.points, reference_.weights, reference_.atomic_weights,
        reference_.atom_offsets, reference_.atomic_radii, left, right, pairs, 0, shape_.points,
        nullptr, nullptr, nullptr, weights, accepted, status, nullptr, SIZE_MAX);
    error = cudaGetLastError();
    return error == cudaSuccess ? error : drain(error, stream);
  }
  for (std::size_t begin = 0; begin < shape_.points;
       begin += std::min(options_.point_tile, shape_.points - begin)) {
    const auto count = std::min(options_.point_tile, shape_.points - begin);
    // Points are independent and each thread serially visits all center pairs.
    // One warp per block distributes the bounded tile over more SMs.
    partition_geometry<false><<<grid(count, 32), 32, 0, stream>>>(
        shape_, options_, basis, reference_.points, reference_.weights, reference_.atomic_weights,
        reference_.atom_offsets, reference_.atomic_radii, left, right, pairs, begin, count,
        reinterpret_cast<ad::Jet*>(base + plan.distances),
        reinterpret_cast<ad::Jet*>(base + plan.products),
        reinterpret_cast<long long*>(base + plan.exponents), weights, accepted, status, nullptr,
        SIZE_MAX);
    if ((error = cudaGetLastError()) != cudaSuccess) return drain(error, stream);
  }
  return cudaSuccess;
}

cudaError_t CudaGridGeometryResponse::prepare(const CudaGridGeometryShape& shape,
                                              const CudaGridGeometryOptions& options,
                                              const CudaGridGeometryReference& reference,
                                              void* workspace,
                                              std::size_t workspace_bytes) noexcept {
  ready_ = false;
  motion_ready_ = false;
  cartesian_right_ = SIZE_MAX;
  work_ = {};
  Layout plan;
  if (!layout(shape, options, plan) || workspace_bytes < plan.bytes || !reference.stream ||
      !reference.authenticate || !reference.owner ||
      (bool(reference.atom_offsets) != bool(reference.atomic_radii)) ||
      (!shape.points_per_atom && !reference.atom_offsets))
    return cudaErrorInvalidValue;
  const void* pointers[] = {reference.basis, reference.points, reference.weights,
                            reference.atomic_weights};
  const std::size_t counts[] = {plan.basis_elements, 3 * shape.points, shape.points, shape.points};
  for (unsigned index = 0; index < 4; ++index)
    if (!disjoint(workspace, plan.bytes / sizeof(double), pointers[index], counts[index]))
      return cudaErrorInvalidValue;
  for (const auto* pointer :
       {pointers[0], pointers[1], pointers[2], pointers[3], static_cast<const void*>(workspace)}) {
    const auto error = device_buffer(pointer, reference.device_id);
    if (error != cudaSuccess) return error;
  }
  if (reference.atom_offsets) {
    for (const auto* metadata : {static_cast<const void*>(reference.atom_offsets),
                                 static_cast<const void*>(reference.atomic_radii)}) {
      const auto error = device_buffer(metadata, reference.device_id);
      if (error != cudaSuccess) return error;
      const auto count = shape.atoms + (metadata == reference.atom_offsets ? 1 : 0);
      if (!disjoint(workspace, plan.bytes / sizeof(double), metadata, count))
        return cudaErrorInvalidValue;
    }
  }
  if (reference.center_pairs) {
    const auto error = device_buffer(reference.center_pairs, reference.device_id);
    if (error != cudaSuccess) return error;
    const auto pairs = shape.atoms * (shape.atoms - 1) / 2;
    constexpr auto words = sizeof(*reference.center_pairs) / sizeof(double);
    if (!pairs ||
        !disjoint(workspace, plan.bytes / sizeof(double), reference.center_pairs, words * pairs))
      return cudaErrorInvalidValue;
  }
  shape_ = shape;
  options_ = options;
  reference_ = reference;
  workspace_ = workspace;
  ready_ = true;
  auto error = authenticate(reference.stream);
  if (error != cudaSuccess) {
    ready_ = false;
    return error;
  }
  auto* base = static_cast<char*>(workspace);
  auto* status = reinterpret_cast<int*>(base + plan.status);
  auto* zero = reinterpret_cast<double*>(base + plan.zero);
  error = cudaMemsetAsync(status, 0, sizeof(int), reference.stream);
  if (error == cudaSuccess)
    error = cudaMemsetAsync(zero, 0, 3 * shape.atoms * sizeof(double), reference.stream);
  for (unsigned index = 0; index < 4 && error == cudaSuccess; ++index) {
    finite_values<<<blocks(counts[index]), 128, 0, reference.stream>>>(
        static_cast<const double*>(pointers[index]), counts[index], status);
    error = cudaGetLastError();
  }
  if (error == cudaSuccess && reference.atom_offsets) {
    validate_grid_topology<<<blocks(shape.atoms), 128, 0, reference.stream>>>(
        shape, reference.atom_offsets, reference.atomic_radii, status);
    error = cudaGetLastError();
  }
  if (error == cudaSuccess) error = gate(status, reference.stream);
  if (error == cudaSuccess) {
    validate_basis<<<blocks(std::max(shape.nao, shape.primitives)), 128, 0, reference.stream>>>(
        shape, reference.basis, status);
    error = cudaGetLastError();
  }
  if (error == cudaSuccess) error = gate(status, reference.stream);
  const auto resident_values = spatial_planes(shape) * plan.resident_points * shape.nao;
  if (error == cudaSuccess && resident_values)
    error = launch_spatial(shape, reference.basis, reference.points, plan.resident_points,
                           reinterpret_cast<double*>(base + plan.spatial), status,
                           reference.stream);
  if (error == cudaSuccess && resident_values) {
    finite_values<<<blocks(resident_values), 128, 0, reference.stream>>>(
        reinterpret_cast<double*>(base + plan.spatial), resident_values, status);
    error = cudaGetLastError();
  }
  if (error == cudaSuccess) error = partition(zero, zero, nullptr, nullptr, true);
  if (error == cudaSuccess)
    error = gate(status, reference.stream);
  else
    error = drain(error, reference.stream);
  if (error == cudaSuccess) error = authenticate(reference.stream);
  if (error != cudaSuccess) {
    ready_ = false;
    return error;
  }
  work_ = {plan.bytes, resident_values, plan.point_pairs};
  work_.prepared_ao_primitive_term_bound = plan.ao_primitive_terms - plan.replay_terms;
  return cudaSuccess;
}

cudaError_t CudaGridGeometryResponse::mixed(const double* left, const double* right,
                                            double* ao_output, double* weight_output,
                                            cudaStream_t stream) noexcept {
  motion_ready_ = false;
  auto error = authenticate(stream);
  if (error != cudaSuccess) return error;
  Layout plan;
  if (!layout(shape_, options_, plan)) return cudaErrorInvalidValue;
  const void* inputs[] = {reference_.basis,
                          reference_.points,
                          reference_.weights,
                          reference_.atomic_weights,
                          workspace_,
                          left,
                          right};
  const std::size_t counts[] = {plan.basis_elements, 3 * shape_.points,           shape_.points,
                                shape_.points,       plan.bytes / sizeof(double), 3 * shape_.atoms,
                                3 * shape_.atoms};
  for (unsigned input = 0; input < 7; ++input)
    if (!disjoint(ao_output, 12 * plan.panel, inputs[input], counts[input]) ||
        !disjoint(weight_output, 3 * shape_.points, inputs[input], counts[input]))
      return cudaErrorInvalidValue;
  if (reference_.atom_offsets) {
    for (const auto* metadata : {static_cast<const void*>(reference_.atom_offsets),
                                 static_cast<const void*>(reference_.atomic_radii)}) {
      const auto count = shape_.atoms + (metadata == reference_.atom_offsets ? 1 : 0);
      if (!disjoint(ao_output, 12 * plan.panel, metadata, count) ||
          !disjoint(weight_output, 3 * shape_.points, metadata, count))
        return cudaErrorInvalidValue;
    }
  }
  if (reference_.center_pairs) {
    const auto pairs = shape_.atoms * (shape_.atoms - 1) / 2;
    constexpr auto words = sizeof(*reference_.center_pairs) / sizeof(double);
    if (!disjoint(ao_output, 12 * plan.panel, reference_.center_pairs, words * pairs) ||
        !disjoint(weight_output, 3 * shape_.points, reference_.center_pairs, words * pairs))
      return cudaErrorInvalidValue;
  }
  if (!disjoint(ao_output, 12 * plan.panel, weight_output, 3 * shape_.points))
    return cudaErrorInvalidValue;
  for (const auto* pointer : {ao_output, weight_output})
    if ((error = device_buffer(pointer, reference_.device_id)) != cudaSuccess) return error;
  if ((error = evaluate_mixed(left, right, stream)) != cudaSuccess) return error;
  auto* base = static_cast<char*>(workspace_);
  error = cudaMemcpyAsync(ao_output, base + plan.candidate_ao, 12 * plan.panel * sizeof(double),
                          cudaMemcpyDeviceToDevice, stream);
  if (error == cudaSuccess)
    error = cudaMemcpyAsync(weight_output, base + plan.candidate_weights,
                            3 * shape_.points * sizeof(double), cudaMemcpyDeviceToDevice, stream);
  error = drain(error, stream);
  if (error == cudaSuccess) finish_mixed();
  return error;
}

cudaError_t CudaGridGeometryResponse::admit_actions(std::size_t actions) noexcept {
  auto error = authenticate(reference_.stream);
  if (error != cudaSuccess) return error;
  Layout plan;
  if (!layout(shape_, options_, plan)) return cudaErrorInvalidValue;
  if (plan.replay_terms &&
      (actions > options_.max_ao_replay_terms / plan.replay_terms ||
       actions > (UINT64_MAX - work_.replayed_ao_primitive_term_bound) / plan.replay_terms))
    return cudaErrorInvalidValue;
  return cudaSuccess;
}

namespace {
__global__ void record_geometry_status(const int* status, int* transaction_status,
                                        const unsigned* failures = nullptr,
                                        std::size_t repair_capacity = 0) {
  const auto flags = (*status ? 1 : 0) |
      (failures && *failures > repair_capacity ? cuda_grid_cartesian_repair_exhausted : 0);
  if (flags) atomicOr(transaction_status, flags);
}
__global__ void validate_cartesian_directions(const double* left, const double* right,
                                               std::size_t coordinates, std::size_t left_column,
                                               std::size_t right_column, int* status) {
  for (std::size_t index = blockIdx.x * blockDim.x + threadIdx.x; index < coordinates;
       index += static_cast<std::size_t>(blockDim.x) * gridDim.x)
    if ((left && left[index] != static_cast<double>(index == left_column)) ||
        right[index] != static_cast<double>(index == right_column))
      atomicExch(status, 1);
}
}  // namespace

cudaError_t CudaGridGeometryResponse::evaluate_mixed(const double* left, const double* right,
                                                     cudaStream_t stream, int* transaction_status,
                                                     std::size_t left_column,
                                                     std::size_t right_column,
                                                     bool weights_only) noexcept {
  auto error = admit_actions(1);
  if (error != cudaSuccess) return error;
  Layout plan;
  if (!layout(shape_, options_, plan) || work_.attempted_actions == UINT64_MAX ||
      work_.action_partition_pairs > UINT64_MAX - plan.point_pairs ||
      work_.action_point_values > UINT64_MAX - 6 * shape_.points ||
      work_.action_ao_jet_values > UINT64_MAX - 12 * plan.panel ||
      !disjoint(left, 3 * shape_.atoms, workspace_, plan.bytes / sizeof(double)) ||
      !disjoint(right, 3 * shape_.atoms, workspace_, plan.bytes / sizeof(double)))
    return cudaErrorInvalidValue;
  for (const auto* pointer : {left, right})
    if ((error = device_buffer(pointer, reference_.device_id)) != cudaSuccess) return error;
  auto* base = static_cast<char*>(workspace_);
  auto* status = reinterpret_cast<int*>(base + plan.status);
  if ((error = cudaMemsetAsync(status, 0, sizeof(int), stream)) != cudaSuccess)
    return drain(error, stream);
  ++work_.attempted_actions;
  const auto coordinates = 3 * shape_.atoms;
  const bool cartesian = left_column != SIZE_MAX;
  if (cartesian)
    validate_cartesian_directions<<<blocks(coordinates), 128, 0, stream>>>(
        left, right, coordinates, left_column, right_column, status);
  else
    finite_directions<<<blocks(coordinates), 128, 0, stream>>>(left, right, coordinates, status);
  error = cudaGetLastError();
  if (cartesian && batched_cartesian(shape_, options_)) {
    const bool diagonal = !weights_only;
    auto* ao_scratch = reinterpret_cast<double*>(base + plan.candidate_ao);
    // A diagonal view produces the AO panels after partitioning, so only its
    // acceptance mask borrows that scratch. A weight-row sweep keeps it live.
    auto* rows = diagonal ? reinterpret_cast<double*>(base + plan.candidate_weights) : ao_scratch;
    auto* accepted = diagonal ? ao_scratch : rows + 3 * coordinates * shape_.points;
    auto& sweeps = diagonal ? work_.cartesian_diagonal_sweeps
                            : work_.cartesian_partition_sweeps;
    if (error == cudaSuccess && (diagonal || cartesian_right_ != right_column)) {
      cartesian_right_ = SIZE_MAX;
      if (sweeps == UINT64_MAX) return cudaErrorInvalidValue;
      // Center coefficients live in point scratch only during this launch
      // pair, before legacy partitioning is allowed to reuse that scratch.
      auto* pairs = reinterpret_cast<CartesianPair*>(base + plan.distances);
      cartesian_pair_geometry<<<blocks(shape_.atoms * shape_.atoms), 128, 0, stream>>>(
          reference_.basis, reference_.atomic_radii, shape_.atoms, right_column,
          options_.coincident_tolerance, pairs);
      error = cudaGetLastError();
      if (error == cudaSuccess)
        error = diagonal
                    ? launch_cartesian_rows<true>(shape_, options_, reference_, pairs, right_column,
                                                   rows, accepted, stream)
                    : launch_cartesian_rows<false>(shape_, options_, reference_, pairs, right_column,
                                                    rows, accepted, stream);
      if (error == cudaSuccess) {
        if (!diagonal) cartesian_right_ = right_column;
        ++sweeps;
      }
    }
    if (error == cudaSuccess)
      error = partition(left, right,
                        rows + (diagonal ? 0 : 3 * left_column * shape_.points), accepted);
  } else {
    cartesian_right_ = SIZE_MAX;
    if (error == cudaSuccess) error = partition(left, right);
  }
  if (error != cudaSuccess) return drain(error, stream);
  auto* resident = reinterpret_cast<double*>(base + plan.spatial);
  auto* replay = resident + spatial_planes(shape_) * plan.resident_points * shape_.nao;
  for (std::size_t begin = 0; !weights_only && begin < shape_.points;) {
    const bool cached = begin < plan.resident_points;
    const auto count =
        cached ? plan.resident_points : std::min(plan.replay_points, shape_.points - begin);
    auto* spatial = cached ? resident : replay;
    if (!cached) {
      error = launch_spatial(shape_, reference_.basis, reference_.points + 3 * begin, count,
                             spatial, status, stream);
      if (error != cudaSuccess) return drain(error, stream);
    }
    const auto launch = [&](auto radial) {
      ao_geometry<decltype(radial)::value><<<blocks(count * shape_.nao), 128, 0, stream>>>(
          shape_, reference_.basis, spatial, reference_.points, begin, count, reference_.atom_offsets,
          left, right, reinterpret_cast<double*>(base + plan.candidate_ao),
          reinterpret_cast<double*>(base + plan.point_directions), status);
    };
    if (shape_.radial_s) launch(std::true_type{});
    else launch(std::false_type{});
    if ((error = cudaGetLastError()) != cudaSuccess) return drain(error, stream);
    begin += count;
  }
  if (transaction_status) {
    record_geometry_status<<<1, 1, 0, stream>>>(status, transaction_status);
    if ((error = cudaGetLastError()) != cudaSuccess) return drain(error, stream);
  } else if ((error = gate(status, stream)) != cudaSuccess) {
    return error;
  }
  return authenticate(stream);
}

cudaError_t CudaGridGeometryResponse::enqueue_private_cartesian_view(
    const double* left, const double* right, std::size_t left_column, std::size_t right_column,
    CudaGridGeometryCartesianView& view, int* transaction_status, cudaStream_t stream) noexcept {
  view = {};
  motion_ready_ = false;
  auto error = authenticate(stream);
  if (error != cudaSuccess) return error;
  if (!transaction_status || left_column / 3 >= shape_.atoms || right_column / 3 >= shape_.atoms)
    return cudaErrorInvalidValue;
  Layout plan;
  if (!layout(shape_, options_, plan)) return cudaErrorInvalidValue;
  if (plan.resident_points != shape_.points) return cudaErrorNotSupported;
  if ((error = evaluate_mixed(left, right, stream, transaction_status, left_column,
                              right_column, true)) != cudaSuccess)
    return error;
  const auto* base = static_cast<const char*>(workspace_);
  const auto* weights = batched_cartesian(shape_, options_)
                            ? reinterpret_cast<const double*>(base + plan.candidate_ao) +
                                  3 * left_column * shape_.points
                            : reinterpret_cast<const double*>(base + plan.candidate_weights);
  view = {reference_.basis,
          reinterpret_cast<const double*>(base + plan.spatial),
          weights,
          reference_.atom_offsets,
          shape_.atoms,
          shape_.points_per_atom,
          left_column,
          right_column, 1, shape_.radial_s ? reference_.points : nullptr};
  ++work_.completed_actions;
  work_.action_partition_pairs += plan.point_pairs;
  return cudaSuccess;
}

cudaError_t CudaGridGeometryResponse::enqueue_private_cartesian_column(
    const double* right, std::size_t right_column, CudaGridGeometryCartesianView& view,
    int* transaction_status, cudaStream_t stream, CudaGridGeometryDirectionView* rhs) noexcept {
  view = {};
  if (rhs) *rhs = {};
  motion_ready_ = false;
  const auto coordinates = 3 * shape_.atoms;
  auto error = admit_actions(coordinates);
  if (error != cudaSuccess) return error;
  if (!transaction_status || right_column >= coordinates) return cudaErrorInvalidValue;
  Layout plan;
  if (!layout(shape_, options_, plan)) return cudaErrorInvalidValue;
  const bool wide = plan.wide_tile != 0;
  if (plan.resident_points != shape_.points ||
      (!wide && (!batched_cartesian(shape_, options_) ||
                 options_.point_tile != CudaGridGeometryOptions{}.point_tile)))
    return cudaErrorNotSupported;

  const auto pair_words = shape_.atoms * (shape_.atoms - 1) / 2 * sizeof(PartitionPair) /
                          sizeof(double);
  const auto row_words = 2 * coordinates * shape_.points;
  const auto scratch_words = 12 * plan.panel - row_words;
  // The full table and acceptance mask borrow idle AO panels. Use the tail
  // first for fallback pair jets, then for the RHS AO view on the same stream.
  if ((!wide && coordinates * pair_words > scratch_words - shape_.points) ||
      (rhs && 4 * plan.panel > scratch_words))
    return cudaErrorNotSupported;
  const auto repair_bound = plan.wide_repair_points * coordinates * shape_.atoms *
                            (shape_.atoms - 1) / 2;
  const auto pair_visits = wide ? plan.point_pairs + repair_bound : coordinates * plan.point_pairs;
  if (!disjoint(right, coordinates, workspace_, plan.bytes / sizeof(double)) ||
      coordinates > UINT64_MAX - work_.attempted_actions ||
      pair_visits > UINT64_MAX - work_.action_partition_pairs ||
      (rhs && (work_.action_ao_jet_values > UINT64_MAX - 4 * plan.panel ||
               work_.action_point_values > UINT64_MAX - 3 * shape_.points)) ||
      work_.cartesian_partition_sweeps == UINT64_MAX)
    return cudaErrorInvalidValue;
  if ((error = device_buffer(right, reference_.device_id)) != cudaSuccess) return error;

  cartesian_right_ = SIZE_MAX;
  auto* base = static_cast<char*>(workspace_);
  auto* rows = reinterpret_cast<double*>(base + plan.candidate_ao);
  auto* accepted = rows + row_words;
  auto* right_weights = reinterpret_cast<double*>(base + plan.candidate_weights);
  auto* fast_pairs = reinterpret_cast<CartesianPair*>(base + (wide ? plan.wide_pairs : plan.distances));
  auto* fallback_pairs = reinterpret_cast<PartitionPair*>(accepted + shape_.points);
  const auto* spatial = reinterpret_cast<const double*>(base + plan.spatial);
  auto* point_directions = reinterpret_cast<double*>(base + plan.point_directions);
  auto* status = reinterpret_cast<int*>(base + plan.status);
  if ((error = cudaMemsetAsync(status, 0, sizeof(int), stream)) != cudaSuccess)
    return drain(error, stream);
  work_.attempted_actions += coordinates;
  validate_cartesian_directions<<<blocks(coordinates), 128, 0, stream>>>(
      nullptr, right, coordinates, SIZE_MAX, right_column, status);
  if ((error = cudaGetLastError()) != cudaSuccess) return drain(error, stream);
  const auto pair_blocks = blocks(shape_.atoms * shape_.atoms);
  cartesian_pair_geometry<<<pair_blocks, 128, 0, stream>>>(
      reference_.basis, reference_.atomic_radii, shape_.atoms, right_column,
      options_.coincident_tolerance, fast_pairs);
  if ((error = cudaGetLastError()) != cudaSuccess) return drain(error, stream);
  auto* failures = wide ? reinterpret_cast<unsigned*>(base + plan.wide_failures) : nullptr;
  if (wide) {
    auto* scratch = reinterpret_cast<RightDual*>(base + plan.wide_scratch);
    if ((error = cudaMemsetAsync(failures, 0, sizeof(*failures), stream)) != cudaSuccess)
      return drain(error, stream);
    const auto threads = (shape_.atoms + 31) / 32 * 32;
    for (std::size_t begin = 0; begin < shape_.points;) {
      const auto count = std::min(plan.wide_tile, shape_.points - begin);
      const auto launch = [&](auto block_local) {
        cartesian_partition_rows<true, true, true, decltype(block_local)::value>
            <<<count, threads, 0, stream>>>(
                shape_, options_, reference_.basis, reference_.points, reference_.weights,
                reference_.atomic_weights, reference_.atom_offsets, reference_.atomic_radii != nullptr,
                fast_pairs, right_column, rows, nullptr, right_weights, begin, scratch, failures,
                static_cast<unsigned>(plan.wide_repair_points), status);
      };
      if (plan.wide_shared) launch(std::true_type{});
      else launch(std::false_type{});
      if ((error = cudaGetLastError()) != cudaSuccess) return drain(error, stream);
      begin += count;
    }
    // Schedule at most the pre-admitted repairs. The queue is populated only
    // by rejected points; a capacity fault suppresses the entire transaction.
    if (plan.wide_repair_points) {
      constexpr unsigned repair_threads = 4;
      const auto shared = repair_threads * shape_.atoms * (2 * sizeof(ad::Jet) + sizeof(long long));
      const auto blocks = (plan.wide_repair_points + repair_threads - 1) / repair_threads;
      partition_geometry<true, true, true>
          <<<dim3(blocks, coordinates), repair_threads, shared, stream>>>(
              shape_, options_, reference_.basis, reference_.points, reference_.weights,
              reference_.atomic_weights, reference_.atom_offsets, reference_.atomic_radii,
              nullptr, right, nullptr, 0, plan.wide_repair_points, nullptr, nullptr, nullptr,
              rows, nullptr, status, right_weights, right_column, failures, fast_pairs);
    }
  } else {
    if ((error = launch_cartesian_rows<false, true>(shape_, options_, reference_, fast_pairs,
                                                   right_column, rows, accepted, stream,
                                                   right_weights)) != cudaSuccess)
      return drain(error, stream);
    pair_geometry<true><<<dim3(pair_blocks, coordinates), 128, 0, stream>>>(
        reference_.basis, nullptr, right, reference_.center_pairs, reference_.atomic_radii,
        shape_.atoms, options_.coincident_tolerance, fallback_pairs, status);
    if ((error = cudaGetLastError()) != cudaSuccess) return drain(error, stream);
    constexpr unsigned threads = 16;
    const auto shared = threads * shape_.atoms * (2 * sizeof(ad::Jet) + sizeof(long long));
    partition_geometry<true, true>
        <<<dim3((shape_.points + threads - 1) / threads, coordinates), threads, shared, stream>>>(
            shape_, options_, reference_.basis, reference_.points, reference_.weights,
            reference_.atomic_weights, reference_.atom_offsets, reference_.atomic_radii,
            nullptr, right, fallback_pairs, 0, shape_.points, nullptr, nullptr, nullptr, rows,
            accepted, status, right_weights, right_column);
  }
  if ((error = cudaGetLastError()) != cudaSuccess) return drain(error, stream);
  if (rhs) {
    const auto launch = [&](auto radial) {
      ao_cartesian_direction<decltype(radial)::value><<<blocks(plan.panel), 128, 0, stream>>>(
          shape_, reference_.basis, spatial, reference_.points, reference_.atom_offsets,
          right_column, accepted, point_directions, status);
    };
    if (shape_.radial_s) launch(std::true_type{});
    else launch(std::false_type{});
    if ((error = cudaGetLastError()) != cudaSuccess) return drain(error, stream);
  }
  record_geometry_status<<<1, 1, 0, stream>>>(status, transaction_status, failures,
                                             plan.wide_repair_points);
  if ((error = cudaGetLastError()) != cudaSuccess) return drain(error, stream);
  if ((error = authenticate(stream)) != cudaSuccess) return error;
  view = {reference_.basis, spatial, rows,
          reference_.atom_offsets, shape_.atoms, shape_.points_per_atom, 0, right_column,
          coordinates, shape_.radial_s ? reference_.points : nullptr};
  if (rhs) {
    *rhs = {accepted, right_weights, point_directions};
    work_.action_ao_jet_values += 4 * plan.panel;
    work_.action_point_values += 3 * shape_.points;
  }
  ++work_.cartesian_partition_sweeps;
  work_.completed_actions += coordinates;
  work_.action_partition_pairs += pair_visits;
  if (wide) {
    ++work_.tiled_cartesian_sweeps;
    work_.tiled_cartesian_repair_pair_bound += repair_bound;
  }
  return cudaSuccess;
}

void CudaGridGeometryResponse::finish_mixed() noexcept {
  Layout plan;
  if (!layout(shape_, options_, plan)) return;
  ++work_.completed_actions;
  work_.action_partition_pairs += plan.point_pairs;
  work_.action_ao_jet_values += 12 * plan.panel;
  work_.action_point_values += 6 * shape_.points;
  work_.replayed_ao_primitive_term_bound += plan.replay_terms;
  motion_ready_ = true;
}

cudaError_t CudaGridGeometryResponse::mixed_view(const double* left, const double* right,
                                                 CudaGridGeometryMixedView& view,
                                                 cudaStream_t stream) noexcept {
  return mixed_view_impl(left, right, view, nullptr, stream);
}

cudaError_t CudaGridGeometryResponse::enqueue_private_mixed_view(const double* left,
                                                                 const double* right,
                                                                 CudaGridGeometryMixedView& view,
                                                                 int* transaction_status,
                                                                 cudaStream_t stream) noexcept {
  if (!transaction_status) {
    view = {};
    motion_ready_ = false;
    return cudaErrorInvalidValue;
  }
  return mixed_view_impl(left, right, view, transaction_status, stream);
}

cudaError_t CudaGridGeometryResponse::mixed_view_impl(const double* left, const double* right,
                                                      CudaGridGeometryMixedView& view,
                                                      int* transaction_status,
                                                      cudaStream_t stream,
                                                      std::size_t diagonal_column) noexcept {
  view = {};
  motion_ready_ = false;
  auto error = authenticate(stream);
  if (error != cudaSuccess) return error;
  if ((error = evaluate_mixed(left, right, stream, transaction_status, diagonal_column,
                              diagonal_column)) != cudaSuccess)
    return error;
  Layout plan;
  if (!layout(shape_, options_, plan)) return cudaErrorInvalidValue;
  const auto* base = static_cast<const char*>(workspace_);
  view = {reinterpret_cast<const double*>(base + plan.candidate_ao),
          reinterpret_cast<const double*>(base + plan.candidate_weights),
          reinterpret_cast<const double*>(base + plan.point_directions)};
  finish_mixed();
  if (transaction_status) motion_ready_ = false;
  return cudaSuccess;
}

cudaError_t CudaGridGeometryResponse::enqueue_private_cartesian_diagonal_view(
    const double* direction, std::size_t column, CudaGridGeometryMixedView& view,
    int* transaction_status, cudaStream_t stream) noexcept {
  if (!transaction_status || column / 3 >= shape_.atoms) {
    view = {};
    motion_ready_ = false;
    return cudaErrorInvalidValue;
  }
  return mixed_view_impl(direction, direction, view, transaction_status, stream, column);
}

cudaError_t CudaGridGeometryResponse::spatial_jets(const double*& jets) noexcept {
  jets = nullptr;
  const auto error = authenticate(reference_.stream);
  if (error != cudaSuccess) return error;
  Layout plan;
  if (!layout(shape_, options_, plan)) return cudaErrorInvalidValue;
  if (shape_.radial_s || plan.resident_points != shape_.points) return cudaErrorNotSupported;
  jets = reinterpret_cast<double*>(static_cast<char*>(workspace_) + plan.spatial);
  return cudaSuccess;
}

cudaError_t CudaGridGeometryResponse::point_directions(const double*& directions) noexcept {
  directions = nullptr;
  auto error = authenticate(reference_.stream);
  if (error != cudaSuccess) return error;
  if (!motion_ready_) return cudaErrorInvalidResourceHandle;
  Layout plan;
  if (!layout(shape_, options_, plan)) return cudaErrorInvalidValue;
  directions = reinterpret_cast<double*>(static_cast<char*>(workspace_) + plan.point_directions);
  return cudaSuccess;
}

}  // namespace generativeqc::dft
