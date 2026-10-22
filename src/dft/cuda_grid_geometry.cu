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
      ao_primitive_terms{}, point_directions{};
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
            Layout& plan) {
  if (!shape.atoms || !shape.primitives || !shape.nao || !shape.points || !options.point_tile ||
      !options.max_workspace_bytes || options.partition_iterations < 1 ||
      options.partition_iterations > 5 || !std::isfinite(options.coincident_tolerance) ||
      options.coincident_tolerance < 0 || !std::isfinite(options.reference_absolute_tolerance) ||
      options.reference_absolute_tolerance < 0 ||
      !std::isfinite(options.reference_relative_tolerance) ||
      options.reference_relative_tolerance < 0)
    return false;
  std::size_t points{}, matrix{}, tile{}, spatial{}, moving{}, atomic{}, primitive{}, records{};
  if ((shape.points_per_atom &&
       (!product(shape.atoms, shape.points_per_atom, points) || points != shape.points)) ||
      !product(shape.points, shape.nao, plan.panel) || !product(20, plan.panel, spatial) ||
      !product(12, plan.panel, moving) || spatial > options.max_ao_jet_values ||
      !product(spatial, shape.primitives, plan.ao_primitive_terms) ||
      !product(3, plan.ao_primitive_terms, plan.ao_primitive_terms) ||
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
  return append(plan, spatial, plan.spatial) && append(plan, moving, plan.candidate_ao) &&
         product(3, shape.points, count) && append(plan, count, plan.candidate_weights) &&
         product(4, matrix, count) && append(plan, count, plan.pairs) && product(4, tile, count) &&
         append(plan, count, plan.distances) && append(plan, count, plan.products) &&
         append(plan, tile, plan.exponents) && append(plan, atomic, plan.zero) &&
         product(6, shape.points, count) && append(plan, count, plan.point_directions) &&
         append(plan, 1, plan.status) && plan.bytes <= options.max_workspace_bytes;
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
  int shift{};
  (void)frexp(magnitude, &shift);
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
                              std::size_t atoms, double tolerance, ad::Jet* pairs, int* status) {
  if (atomicAdd(status, 0)) return;
  for (std::size_t index = blockIdx.x * blockDim.x + threadIdx.x; index < atoms * atoms;
       index += static_cast<std::size_t>(blockDim.x) * gridDim.x) {
    const auto first = index / atoms, second = index % atoms;
    if (second >= first) continue;
    ad::Jet coordinates[3];
    for (unsigned axis = 0; axis < 3; ++axis)
      coordinates[axis] = {centers[3 * first + axis] - centers[3 * second + axis],
                           left[3 * first + axis] - left[3 * second + axis],
                           right[3 * first + axis] - right[3 * second + axis], 0};
    const auto distance = ad::norm(coordinates[0], coordinates[1], coordinates[2]);
    if (!finite(distance) || distance.value <= tolerance) atomicExch(status, 1);
    pairs[index] = distance;
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
__global__ void partition_geometry(CudaGridGeometryShape shape, CudaGridGeometryOptions options,
                                   const double* centers, const double* points,
                                   const double* weights, const double* atomic_weights,
                                   const std::uint64_t* offsets, const double* radii,
                                   const double* left, const double* right, const ad::Jet* pairs,
                                   std::size_t begin, std::size_t count, ad::Jet* distances,
                                   ad::Jet* products, long long* exponents, double* output,
                                   int* status) {
  if (atomicAdd(status, 0)) return;
  for (std::size_t local = blockIdx.x * blockDim.x + threadIdx.x; local < count;
       local += static_cast<std::size_t>(blockDim.x) * gridDim.x) {
    const auto point = begin + local, owner = grid_owner(shape, offsets, point);
    const auto scratch = local * shape.atoms;
    auto result = ad::Jet{1, 0, 0, 0};
    bool valid = isfinite(atomic_weights[point]) && isfinite(weights[point]);
    if (shape.atoms > 1) {
      for (std::size_t atom = 0; atom < shape.atoms; ++atom) {
        ad::Jet coordinates[3];
        for (unsigned axis = 0; axis < 3; ++axis)
          coordinates[axis] = {points[3 * point + axis] - centers[3 * atom + axis],
                               left[3 * owner + axis] - left[3 * atom + axis],
                               right[3 * owner + axis] - right[3 * atom + axis], 0};
        distances[scratch + atom] = ad::norm(coordinates[0], coordinates[1], coordinates[2]);
        valid = valid && finite(distances[scratch + atom]) && distances[scratch + atom].value > 0;
        products[scratch + atom] = {1, 0, 0, 0};
        exponents[scratch + atom] = 0;
      }
      if (!valid) {
        atomicExch(status, 1);
        continue;
      }
      for (std::size_t first = 0; first < shape.atoms; ++first)
        for (std::size_t second = 0; second < first; ++second) {
          auto coordinate =
              ad::ratio(ad::subtract(distances[scratch + first], distances[scratch + second]),
                        pairs[first * shape.atoms + second]);
          if (radii) {
            if (coordinate.value <= -1)
              coordinate = {-1, 0, 0, 0};
            else if (coordinate.value >= 1)
              coordinate = {1, 0, 0, 0};
            const double ratio = sqrt(radii[second] / radii[first]);
            const double adjustment = fmax(-0.5, fmin(0.5, 0.25 * (ratio - 1 / ratio)));
            coordinate = ad::treutler_adjust(coordinate, {adjustment, 0, 0, 0});
          }
          const auto pair = becke(coordinate, options.partition_iterations);
          valid = valid && finite(coordinate) && finite(pair);
          valid = multiply(products[scratch + first], exponents[scratch + first], pair) && valid;
          valid = multiply(products[scratch + second], exponents[scratch + second],
                           ad::complement(pair)) &&
                  valid;
        }
      long long common = LLONG_MIN;
      for (std::size_t atom = 0; atom < shape.atoms; ++atom)
        if (maximum(products[scratch + atom])) common = max(common, exponents[scratch + atom]);
      if (!valid || common == LLONG_MIN) {
        atomicExch(status, 1);
        continue;
      }
      ad::Jet total{}, numerator{};
      // A common exponent keeps every component of the normalized quotient in
      // one unit. Never divide by a pair factor: it may round to exact zero.
      for (std::size_t atom = 0; atom < shape.atoms; ++atom) {
        const auto difference = exponents[scratch + atom] - common;
        const auto value = maximum(products[scratch + atom]) && difference >= -1074
                               ? scale(products[scratch + atom], static_cast<int>(difference))
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
                            const std::uint64_t* offsets, const double* left, const double* right,
                            double* output, double* point_directions, int* status) {
  if (atomicAdd(status, 0)) return;
  const auto panel = shape.points * shape.nao;
  for (std::size_t index = blockIdx.x * blockDim.x + threadIdx.x; index < panel;
       index += static_cast<std::size_t>(blockDim.x) * gridDim.x) {
    const auto orbital = index % shape.nao, point = index / shape.nao;
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
    ad::ao_motion(spatial + index, panel, left_motion, right_motion, values);
    for (unsigned jet = 0; jet < 4; ++jet)
      for (unsigned direction = 0; direction < 3; ++direction) {
        const auto value = values[3 * jet + direction];
        if (!isfinite(value)) atomicExch(status, 1);
        output[(4 * direction + jet) * panel + index] = value;
      }
  }
}
}  // namespace

cudaError_t cuda_grid_geometry_workspace(const CudaGridGeometryShape& shape,
                                         const CudaGridGeometryOptions& options,
                                         std::size_t& bytes) noexcept {
  bytes = 0;
  Layout plan;
  if (!layout(shape, options, plan)) return cudaErrorInvalidValue;
  bytes = plan.bytes;
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
  auto* pairs = reinterpret_cast<ad::Jet*>(base + plan.pairs);
  const auto stream = reference_.stream;
  pair_geometry<<<blocks(shape_.atoms * shape_.atoms), 128, 0, stream>>>(
      reference_.basis, left, right, shape_.atoms, options_.coincident_tolerance, pairs, status);
  auto error = cudaGetLastError();
  if (error != cudaSuccess) return drain(error, stream);
  for (std::size_t begin = 0; begin < shape_.points;
       begin += std::min(options_.point_tile, shape_.points - begin)) {
    const auto count = std::min(options_.point_tile, shape_.points - begin);
    partition_geometry<<<blocks(count), 128, 0, stream>>>(
        shape_, options_, reference_.basis, reference_.points, reference_.weights,
        reference_.atomic_weights, reference_.atom_offsets, reference_.atomic_radii, left, right,
        pairs, begin, count, reinterpret_cast<ad::Jet*>(base + plan.distances),
        reinterpret_cast<ad::Jet*>(base + plan.products),
        reinterpret_cast<long long*>(base + plan.exponents),
        reinterpret_cast<double*>(base + plan.candidate_weights), status);
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
  if (error == cudaSuccess)
    error = hessian_aot::launch_ao(
        reference.basis, shape.atoms, shape.primitives, shape.nao, reference.points, shape.points,
        20, reinterpret_cast<double*>(base + plan.spatial), status, reference.stream);
  if (error == cudaSuccess) {
    finite_values<<<blocks(20 * plan.panel), 128, 0, reference.stream>>>(
        reinterpret_cast<double*>(base + plan.spatial), 20 * plan.panel, status);
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
  work_ = {plan.bytes, 20 * plan.panel, plan.point_pairs};
  work_.prepared_ao_primitive_term_bound = plan.ao_primitive_terms;
  return cudaSuccess;
}

cudaError_t CudaGridGeometryResponse::mixed(const double* left, const double* right,
                                            double* ao_output, double* weight_output,
                                            cudaStream_t stream) noexcept {
  motion_ready_ = false;
  auto error = authenticate(stream);
  if (error != cudaSuccess) return error;
  Layout plan;
  if (!layout(shape_, options_, plan) || work_.attempted_actions == UINT64_MAX ||
      work_.action_partition_pairs > UINT64_MAX - plan.point_pairs ||
      work_.action_point_values > UINT64_MAX - 6 * shape_.points ||
      work_.action_ao_jet_values > UINT64_MAX - 12 * plan.panel)
    return cudaErrorInvalidValue;
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
  if (!disjoint(ao_output, 12 * plan.panel, weight_output, 3 * shape_.points) ||
      !disjoint(left, 3 * shape_.atoms, workspace_, plan.bytes / sizeof(double)) ||
      !disjoint(right, 3 * shape_.atoms, workspace_, plan.bytes / sizeof(double)))
    return cudaErrorInvalidValue;
  for (const auto* pointer : {left, right, static_cast<const double*>(ao_output),
                              static_cast<const double*>(weight_output)})
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
  ao_geometry<<<blocks(plan.panel), 128, 0, stream>>>(
      shape_, reference_.basis, reinterpret_cast<double*>(base + plan.spatial),
      reference_.atom_offsets, left, right, reinterpret_cast<double*>(base + plan.candidate_ao),
      reinterpret_cast<double*>(base + plan.point_directions), status);
  if ((error = cudaGetLastError()) != cudaSuccess) return drain(error, stream);
  if ((error = gate(status, stream)) != cudaSuccess) return error;
  if ((error = authenticate(stream)) != cudaSuccess) return error;
  error = cudaMemcpyAsync(ao_output, base + plan.candidate_ao, 12 * plan.panel * sizeof(double),
                          cudaMemcpyDeviceToDevice, stream);
  if (error == cudaSuccess)
    error = cudaMemcpyAsync(weight_output, base + plan.candidate_weights,
                            3 * shape_.points * sizeof(double), cudaMemcpyDeviceToDevice, stream);
  error = drain(error, stream);
  if (error == cudaSuccess) {
    ++work_.completed_actions;
    work_.action_partition_pairs += plan.point_pairs;
    work_.action_ao_jet_values += 12 * plan.panel;
    work_.action_point_values += 6 * shape_.points;
    motion_ready_ = true;
  }
  return error;
}

cudaError_t CudaGridGeometryResponse::spatial_jets(const double*& jets) noexcept {
  jets = nullptr;
  const auto error = authenticate(reference_.stream);
  if (error != cudaSuccess) return error;
  Layout plan;
  if (!layout(shape_, options_, plan)) return cudaErrorInvalidValue;
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
