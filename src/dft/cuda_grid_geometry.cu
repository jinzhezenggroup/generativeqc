#include <algorithm>
#include <climits>
#include <cmath>
#include <limits>

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
};
struct PartitionPair {
  ad::Jet distance;
  double adjustment;
};
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
  plan.ao_primitive_terms = shape.ao_primitive_terms;
  if (!plan.ao_primitive_terms && (!product(shape.nao, shape.primitives, plan.ao_primitive_terms) ||
                                   !product(3, plan.ao_primitive_terms, plan.ao_primitive_terms)))
    return false;
  if ((shape.points_per_atom &&
       (!product(shape.atoms, shape.points_per_atom, points) || points != shape.points)) ||
      !product(shape.points, shape.nao, plan.panel) || !product(20, plan.panel, spatial) ||
      !product(12, plan.panel, moving) || spatial > options.max_ao_jet_values ||
      !product(20, plan.ao_primitive_terms, plan.ao_primitive_terms) ||
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
      !append(plan, count, plan.candidate_weights) || !product(5, matrix, count) ||
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
    const auto values = 20 * (required_resident + (replayable != 0)) * shape.nao;
    if (!append(smallest, values, smallest.spatial)) return false;
    *minimum = smallest.bytes;
  }
  const auto available_values = (options.max_workspace_bytes - plan.bytes) / 256 * 32;
  if (spatial > available_values) {
    const auto capacity = available_values / (20 * shape.nao);
    if (capacity <= required_resident) return false;
    plan.replay_points = std::min<std::uint64_t>(options.point_tile, capacity - required_resident);
    plan.resident_points = options.retain_partial_jets ? capacity - plan.replay_points : 0;
    plan.replay_terms = terms_per_point * (shape.points - plan.resident_points);
    spatial = 20 * (plan.resident_points + plan.replay_points) * shape.nao;
  }
  return append(plan, spatial, plan.spatial) && plan.bytes <= options.max_workspace_bytes;
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
    for (unsigned term = 0; term < static_cast<unsigned>(terms); ++term)
      for (unsigned axis = 0; axis < 3; ++axis) {
        const auto power = record[4 + 4 * term + axis];
        if (power < 0 || power > 3 || floor(power) != power) atomicExch(status, 1);
      }
  }
}
__global__ void pair_geometry(const double* centers, const double* left, const double* right,
                              const generativeqc_grid_pair_geometry::CenterPair* prepared,
                              const double* radii, std::size_t atoms, double tolerance,
                              PartitionPair* pairs, int* status) {
  if (atomicAdd(status, 0)) return;
  for (std::size_t index = blockIdx.x * blockDim.x + threadIdx.x; index < atoms * atoms;
       index += static_cast<std::size_t>(blockDim.x) * gridDim.x) {
    const auto first = index / atoms, second = index % atoms;
    if (second >= first) continue;
    double left_delta[3], right_delta[3];
    for (unsigned axis = 0; axis < 3; ++axis) {
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
    if (!finite(distance) || distance.value <= tolerance) atomicExch(status, 1);
    // Treutler's radius adjustment depends only on the two elements, not the
    // grid point or its motion. Compute it once per pair for this dispatch.
    double adjustment = 0;
    if (radii) {
      const double ratio = sqrt(radii[second] / radii[first]);
      adjustment = fmax(-0.5, fmin(0.5, 0.25 * (ratio - 1 / ratio)));
    }
    pairs[index] = {distance, adjustment};
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
__global__ void validate_grid_topology(CudaGridGeometryShape shape, const std::uint64_t* offsets,
                                       const double* radii, int* status) {
  for (std::size_t atom = blockIdx.x * blockDim.x + threadIdx.x; atom < shape.atoms;
       atom += std::size_t(blockDim.x) * gridDim.x) {
    if (offsets[atom + 1] <= offsets[atom] || !isfinite(radii[atom]) || radii[atom] <= 0 ||
        offsets[0] || offsets[shape.atoms] != shape.points)
      atomicExch(status, 1);
  }
}
template <bool block_local>
__global__ void partition_geometry(CudaGridGeometryShape shape, CudaGridGeometryOptions options,
                                   const double* centers, const double* points,
                                   const double* weights, const double* atomic_weights,
                                   const std::uint64_t* offsets, const double* radii,
                                   const double* left, const double* right,
                                   const PartitionPair* pairs, std::size_t begin, std::size_t count,
                                   ad::Jet* distances, ad::Jet* products, long long* exponents,
                                   double* output, int* status) {
  if (atomicAdd(status, 0)) return;
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
    const auto point = begin + local, owner = grid_owner(shape, offsets, point);
    const auto scratch = block_local ? threadIdx.x : local * shape.atoms;
    auto result = ad::Jet{1, 0, 0, 0};
    bool valid = isfinite(atomic_weights[point]) && isfinite(weights[point]);
    if (shape.atoms > 1) {
      for (std::size_t atom = 0; atom < shape.atoms; ++atom) {
        ad::Jet coordinates[3];
        for (unsigned axis = 0; axis < 3; ++axis)
          coordinates[axis] = {points[3 * point + axis] - centers[3 * atom + axis],
                               left[3 * owner + axis] - left[3 * atom + axis],
                               right[3 * owner + axis] - right[3 * atom + axis], 0};
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
          const auto& geometry = pairs[first * shape.atoms + second];
          auto coordinate = ad::ratio(ad::subtract(distances[first_slot], distances[second_slot]),
                                      geometry.distance);
          if (radii) {
            if (coordinate.value <= -1)
              coordinate = {-1, 0, 0, 0};
            else if (coordinate.value >= 1)
              coordinate = {1, 0, 0, 0};
            coordinate = ad::treutler_adjust(coordinate, {geometry.adjustment, 0, 0, 0});
          }
          const auto pair = becke(coordinate, options.partition_iterations);
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
    const auto expected = atomic_weights[point] != 0 ? weights[point] / atomic_weights[point] : 0;
    if ((!atomic_weights[point] && weights[point]) || !isfinite(expected) ||
        (atomic_weights[point] != 0 &&
         fabs(result.value - expected) >
             options.reference_absolute_tolerance +
                 options.reference_relative_tolerance * fmax(fabs(result.value), fabs(expected))))
      valid = false;
    result = ad::multiply(result, {atomic_weights[point], 0, 0, 0});
    if (!valid || !finite(result)) atomicExch(status, 1);
    output[point] = result.left;
    output[shape.points + point] = result.right;
    output[2 * shape.points + point] = result.mixed;
  }
}
__global__ void ao_geometry(CudaGridGeometryShape shape, const double* basis, const double* spatial,
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
    ad::ao_motion(spatial + index, tile_panel, left_motion, right_motion, values);
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

cudaError_t CudaGridGeometryResponse::partition(const double* left, const double* right) noexcept {
  Layout plan;
  if (!layout(shape_, options_, plan)) return cudaErrorInvalidValue;
  auto* base = static_cast<char*>(workspace_);
  auto* status = reinterpret_cast<int*>(base + plan.status);
  auto* pairs = reinterpret_cast<PartitionPair*>(base + plan.pairs);
  const auto stream = reference_.stream;
  pair_geometry<<<blocks(shape_.atoms * shape_.atoms), 128, 0, stream>>>(
      reference_.basis, left, right, reference_.center_pairs, reference_.atomic_radii, shape_.atoms,
      options_.coincident_tolerance, pairs, status);
  auto error = cudaGetLastError();
  if (error != cudaSuccess) return drain(error, stream);
  // Full-grid dispatch only for small, default-sized plans. Each lane owns its
  // scratch; up to 24 KiB per block stays within default shared-memory limits.
  // Retain the same bounded arena and tiled path for larger or explicit tiles.
  constexpr std::size_t threads = 32;
  const auto grid = [=](std::size_t count) {
    return std::min<std::size_t>(1 + (count - 1) / threads, 65535);
  };
  const auto* basis = reference_.basis;
  auto* weights = reinterpret_cast<double*>(base + plan.candidate_weights);
  constexpr std::size_t scratch_per_atom = threads * (2 * sizeof(ad::Jet) + sizeof(long long));
  if (shape_.atoms <= (24 * 1024) / scratch_per_atom && shape_.points <= 8192 &&
      options_.point_tile == CudaGridGeometryOptions{}.point_tile) {
    const auto bytes = shape_.atoms * scratch_per_atom;
    partition_geometry<true><<<grid(shape_.points), threads, bytes, stream>>>(
        shape_, options_, basis, reference_.points, reference_.weights, reference_.atomic_weights,
        reference_.atom_offsets, reference_.atomic_radii, left, right, pairs, 0, shape_.points,
        nullptr, nullptr, nullptr, weights, status);
    error = cudaGetLastError();
    return error == cudaSuccess ? error : drain(error, stream);
  }
  for (std::size_t begin = 0; begin < shape_.points;
       begin += std::min(options_.point_tile, shape_.points - begin)) {
    const auto count = std::min(options_.point_tile, shape_.points - begin);
    // Points are independent and each thread serially visits all center pairs.
    // One warp per block distributes the bounded tile over more SMs.
    partition_geometry<false><<<grid(count), threads, 0, stream>>>(
        shape_, options_, basis, reference_.points, reference_.weights, reference_.atomic_weights,
        reference_.atom_offsets, reference_.atomic_radii, left, right, pairs, begin, count,
        reinterpret_cast<ad::Jet*>(base + plan.distances),
        reinterpret_cast<ad::Jet*>(base + plan.products),
        reinterpret_cast<long long*>(base + plan.exponents), weights, status);
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
  const auto resident_values = 20 * plan.resident_points * shape.nao;
  if (error == cudaSuccess && resident_values)
    error = hessian_aot::launch_ao(reference.basis, shape.atoms, shape.primitives, shape.nao,
                                   reference.points, plan.resident_points, 20,
                                   reinterpret_cast<double*>(base + plan.spatial), status,
                                   reference.stream);
  if (error == cudaSuccess && resident_values) {
    finite_values<<<blocks(resident_values), 128, 0, reference.stream>>>(
        reinterpret_cast<double*>(base + plan.spatial), resident_values, status);
    error = cudaGetLastError();
  }
  if (error == cudaSuccess) error = partition(zero, zero);
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
__global__ void record_geometry_status(const int* status, int* transaction_status) {
  if (*status) atomicOr(transaction_status, 1);
}
}  // namespace

cudaError_t CudaGridGeometryResponse::evaluate_mixed(const double* left, const double* right,
                                                     cudaStream_t stream, int* transaction_status,
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
  finite_directions<<<blocks(3 * shape_.atoms), 128, 0, stream>>>(left, right, 3 * shape_.atoms,
                                                                  status);
  if ((error = cudaGetLastError()) == cudaSuccess) error = partition(left, right);
  if (error != cudaSuccess) return drain(error, stream);
  auto* resident = reinterpret_cast<double*>(base + plan.spatial);
  auto* replay = resident + 20 * plan.resident_points * shape_.nao;
  for (std::size_t begin = 0; !weights_only && begin < shape_.points;) {
    const bool cached = begin < plan.resident_points;
    const auto count =
        cached ? plan.resident_points : std::min(plan.replay_points, shape_.points - begin);
    auto* spatial = cached ? resident : replay;
    if (!cached) {
      error =
          hessian_aot::launch_ao(reference_.basis, shape_.atoms, shape_.primitives, shape_.nao,
                                 reference_.points + 3 * begin, count, 20, spatial, status, stream);
      if (error != cudaSuccess) return drain(error, stream);
    }
    ao_geometry<<<blocks(count * shape_.nao), 128, 0, stream>>>(
        shape_, reference_.basis, spatial, begin, count, reference_.atom_offsets, left, right,
        reinterpret_cast<double*>(base + plan.candidate_ao),
        reinterpret_cast<double*>(base + plan.point_directions), status);
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
  if ((error = evaluate_mixed(left, right, stream, transaction_status, true)) != cudaSuccess)
    return error;
  const auto* base = static_cast<const char*>(workspace_);
  view = {reference_.basis,
          reinterpret_cast<const double*>(base + plan.spatial),
          reinterpret_cast<const double*>(base + plan.candidate_weights),
          reference_.atom_offsets,
          shape_.atoms,
          shape_.points_per_atom,
          left_column,
          right_column};
  ++work_.completed_actions;
  work_.action_partition_pairs += plan.point_pairs;
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
                                                      cudaStream_t stream) noexcept {
  view = {};
  motion_ready_ = false;
  auto error = authenticate(stream);
  if (error != cudaSuccess) return error;
  if ((error = evaluate_mixed(left, right, stream, transaction_status)) != cudaSuccess)
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

cudaError_t CudaGridGeometryResponse::spatial_jets(const double*& jets) noexcept {
  jets = nullptr;
  const auto error = authenticate(reference_.stream);
  if (error != cudaSuccess) return error;
  Layout plan;
  if (!layout(shape_, options_, plan)) return cudaErrorInvalidValue;
  if (plan.resident_points != shape_.points) return cudaErrorNotSupported;
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
