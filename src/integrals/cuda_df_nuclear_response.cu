#include <algorithm>
#include <array>
#include <climits>
#include <cmath>
#include <new>
#include <vector>

#include "dft/hessian_aot.hpp"
#include "integrals/cuda_df_nuclear_response.hpp"
#include "response/nuclear_integral_response.cuh"

namespace generativeqc::integrals {
namespace {
namespace aot = dft::hessian_aot;
namespace ad = generativeqc_nuclear_integral_ad;
using Shell = CudaDfNuclearShell;
using Basis = scf::CudaDfReadonlyBasis;
using Expansion = scf::cuda_execution::DfPublicAoExpansion;
bool add(std::size_t& value, std::size_t increment) {
  if (increment > SIZE_MAX - value) return false;
  value += increment;
  return true;
}
bool multiply(std::size_t first, std::size_t second, std::size_t& result) {
  if (second && first > SIZE_MAX / second) return false;
  result = first * second;
  return true;
}
std::size_t cartesian(std::size_t angular) { return (angular + 1) * (angular + 2) / 2; }
unsigned blocks(std::size_t count) {
  return static_cast<unsigned>(std::min((count + 127) / 128, std::size_t{65535}));
}
struct DeviceGroup {
  Shell shells[3]{};
  long long atoms[3]{};
  std::size_t centers{}, components{}, public_components{};
  bool metric{};
};
struct DeviceTile {
  std::uint32_t components[16]{}, count{};
};
struct Tile {
  const aot::Program* program{};
  DeviceTile device{};
};
struct Group {
  DeviceGroup device{};
  std::size_t radials{}, closure{};
};
struct Layout {
  std::size_t bytes{}, shells{}, groups{}, tiles{}, records{}, raw{}, cache{}, projected{},
      candidates{}, status{}, raw_elements{}, metric_elements{}, candidate_count{},
      max_components{}, max_fields{}, max_public{};
  CudaDfNuclearWork work{};
  bool reserve(std::size_t count, std::size_t size, std::size_t& offset) {
    offset = bytes;
    std::size_t increment{};
    if (!multiply(count, size, increment) || !add(bytes, increment) || !add(bytes, 255))
      return false;
    bytes = bytes / 256 * 256;
    return true;
  }
};
struct Schedule {
  Layout layout;
  std::vector<Group> groups;
  std::vector<Tile> tiles;
  std::array<std::vector<std::size_t>, 150> closure;
};
cudaError_t schedule(const CudaDfNuclearShape& shape, std::span<const Shell> orbital,
                     std::span<const Shell> auxiliary, const CudaDfNuclearOptions& options,
                     Schedule& result) {
  if (!shape.atoms || !shape.primitives || !shape.nao || !shape.naux || !shape.cartesian_nao ||
      !shape.cartesian_naux || orbital.empty() || auxiliary.empty() || shape.atoms > INT32_MAX ||
      shape.primitives > INT64_MAX || shape.cartesian_nao > INT32_MAX ||
      shape.cartesian_naux > INT32_MAX - shape.cartesian_nao || orbital.size() > INT32_MAX ||
      auxiliary.size() > INT32_MAX - orbital.size() || !options.max_workspace_bytes ||
      !options.max_groups || options.radial_tile < 1 || options.radial_tile > 64 ||
      !options.max_component_evaluations || !options.max_projection_terms ||
      !options.max_reduction_terms_bound || !std::isfinite(options.symmetry_absolute_tolerance) ||
      options.symmetry_absolute_tolerance < 0 ||
      !std::isfinite(options.symmetry_relative_tolerance) ||
      options.symmetry_relative_tolerance < 0)
    return cudaErrorInvalidValue;
  auto& plan = result.layout;
  std::size_t primitives{};
  const auto topology = [&](std::span<const Shell> shells, std::size_t angular_bound,
                            std::size_t public_count, std::size_t cartesian_count) {
    std::size_t aos{}, cartesian_aos{};
    for (const auto& shell : shells) {
      if (shell.atom >= shape.atoms || shell.angular > angular_bound || !shell.primitive_count ||
          shell.primitive_begin != primitives || shell.ao_begin != aos ||
          shell.cartesian_begin != cartesian_aos ||
          (shell.ao_count != cartesian(shell.angular) && shell.ao_count != 2 * shell.angular + 1) ||
          !add(primitives, shell.primitive_count) || !add(aos, shell.ao_count) ||
          !add(cartesian_aos, cartesian(shell.angular)))
        return false;
    }
    return aos == public_count && cartesian_aos == cartesian_count;
  };
  if (!topology(orbital, 3, shape.nao, shape.cartesian_nao) ||
      !topology(auxiliary, 4, shape.naux, shape.cartesian_naux) || primitives != shape.primitives)
    return cudaErrorInvalidValue;
  std::size_t groups{}, extra{}, count{};
  if (!multiply(orbital.size(), orbital.size(), groups) ||
      !multiply(groups, auxiliary.size(), groups) ||
      !multiply(auxiliary.size(), auxiliary.size(), extra) || !add(groups, extra) ||
      groups > options.max_groups || !multiply(shape.nao, shape.nao, count) ||
      !multiply(count, shape.naux, plan.raw_elements) ||
      !multiply(shape.naux, shape.naux, plan.metric_elements))
    return cudaErrorInvalidValue;
  result.groups.reserve(groups);
  const auto append_group = [&](const Shell& first, const Shell& second, const Shell* third) {
    Group group;
    group.device.centers = third ? 3 : 2;
    group.device.metric = !third;
    group.device.shells[0] = first;
    group.device.shells[1] = second;
    if (third) group.device.shells[2] = *third;
    group.device.components = group.device.public_components = group.radials = 1;
    std::size_t key{};
    for (unsigned slot = 0; slot < group.device.centers; ++slot) {
      const auto& shell = group.device.shells[slot];
      group.device.atoms[slot] = shell.atom;
      key = key * 5 + shell.angular;
      if (!multiply(group.radials, shell.primitive_count, group.radials) ||
          !multiply(group.device.components, cartesian(shell.angular), group.device.components) ||
          !multiply(group.device.public_components, shell.ao_count, group.device.public_components))
        return cudaErrorInvalidValue;
    }
    if (!third) key += 125;
    group.closure = key;
    auto& closure = result.closure[key];
    if (closure.empty()) {
      std::array<bool, 1500> covered{};
      for (const auto& program : aot::programs()) {
        const auto& signature = program.signature;
        if (program.kind != aot::Kind::integral ||
            signature.family != (third ? "three_center_eri" : "coulomb_metric") ||
            signature.angular.size() != group.device.centers)
          continue;
        bool matches = true;
        for (unsigned slot = 0; slot < group.device.centers; ++slot)
          matches &= signature.angular[slot] == group.device.shells[slot].angular;
        if (!matches) continue;
        if (signature.radial != "full_range" || signature.omega != 0 || !program.integral ||
            program.centers != group.device.centers ||
            program.record_stride != 4 * group.device.centers + 1 ||
            program.outputs != 1 + 6 * group.device.centers || program.components.empty() ||
            program.components.size() > 16)
          return cudaErrorNotSupported;
        Tile tile{&program, {}};
        tile.device.count = program.components.size();
        for (unsigned index = 0; index < tile.device.count; ++index) {
          const auto component = program.components[index];
          if (component >= group.device.components || covered[component])
            return cudaErrorNotSupported;
          covered[component] = true;
          tile.device.components[index] = component;
        }
        closure.push_back(result.tiles.size());
        result.tiles.push_back(tile);
      }
      if (!std::all_of(covered.begin(), covered.begin() + group.device.components,
                       [](bool value) { return value; }))
        return cudaErrorNotSupported;
    }
    std::size_t components{}, projections{}, reductions{}, records{}, cache_terms{},
        projection_reductions{}, scatter_terms{};
    const auto coordinates = 3 * group.device.centers;
    const auto fields = 1 + 2 * coordinates;
    const auto batches =
        group.radials / options.radial_tile + (group.radials % options.radial_tile != 0);
    // Count submitted records for every component tile, primitive-field sums,
    // cache additions, sparse projection sums and deterministic scatter. The
    // reduction bound covers either pass, including coincident center writes.
    if (!multiply(group.radials, group.device.components, components) ||
        !multiply(group.radials, closure.size(), records) ||
        !multiply(group.device.public_components, third ? 54 : 36, projections) ||
        !multiply(projections, 1 + 3 * coordinates, projection_reductions) ||
        !multiply(projections, coordinates, projections) ||
        !multiply(components, fields, reductions) ||
        !multiply(group.device.components, fields, cache_terms) ||
        !multiply(cache_terms, batches, cache_terms) ||
        !multiply(group.device.public_components + 1, 2 * coordinates, scatter_terms) ||
        !add(reductions, cache_terms) || !add(reductions, projection_reductions) ||
        !add(reductions, scatter_terms) || !add(plan.work.radial_records_per_pass, records) ||
        !add(plan.work.component_evaluations_per_pass, components) ||
        !add(plan.work.projection_terms_bound_per_pass, projections) ||
        !add(plan.work.reduction_terms_bound_per_pass, reductions))
      return cudaErrorInvalidValue;
    plan.max_components = std::max(plan.max_components, group.device.components);
    plan.max_public = std::max(plan.max_public, group.device.public_components);
    plan.max_fields = std::max(plan.max_fields, 1 + 6 * group.device.centers);
    result.groups.push_back(group);
    return cudaSuccess;
  };
  for (const auto& first : orbital)
    for (const auto& second : orbital)
      for (const auto& third : auxiliary)
        if (const auto error = append_group(first, second, &third); error != cudaSuccess)
          return error;
  for (const auto& first : auxiliary)
    for (const auto& second : auxiliary)
      if (const auto error = append_group(first, second, nullptr); error != cudaSuccess)
        return error;
  if (plan.work.component_evaluations_per_pass > options.max_component_evaluations ||
      plan.work.projection_terms_bound_per_pass > options.max_projection_terms ||
      plan.work.reduction_terms_bound_per_pass > options.max_reduction_terms_bound)
    return cudaErrorInvalidValue;
  plan.candidate_count = plan.raw_elements;
  if (!add(plan.candidate_count, plan.metric_elements) ||
      !multiply(plan.candidate_count, 2, plan.candidate_count) || !multiply(shape.atoms, 6, extra))
    return cudaErrorInvalidValue;
  plan.candidate_count = std::max(plan.candidate_count, extra);
  if (!plan.reserve(orbital.size() + auxiliary.size(), sizeof(Shell), plan.shells) ||
      !plan.reserve(groups, sizeof(DeviceGroup), plan.groups) ||
      !plan.reserve(result.tiles.size(), sizeof(DeviceTile), plan.tiles) ||
      !plan.reserve(13 * options.radial_tile, sizeof(double), plan.records) ||
      !plan.reserve(16 * plan.max_fields, sizeof(double), plan.raw) ||
      !plan.reserve(plan.max_components * plan.max_fields, sizeof(double), plan.cache) ||
      !plan.reserve(plan.max_public * 18, sizeof(double), plan.projected) ||
      !plan.reserve(plan.candidate_count, sizeof(double), plan.candidates) ||
      !plan.reserve(1, sizeof(int), plan.status) || plan.bytes > options.max_workspace_bytes)
    return cudaErrorInvalidValue;
  plan.work.workspace_bytes = plan.bytes;
  plan.work.groups = groups;
  return cudaSuccess;
}
bool disjoint(const void* first, std::size_t first_bytes, const void* second,
              std::size_t second_bytes) {
  const auto left = reinterpret_cast<std::uintptr_t>(first),
             right = reinterpret_cast<std::uintptr_t>(second);
  return left <= right ? right - left >= first_bytes : left - right >= second_bytes;
}
cudaError_t device_buffer(const void* pointer, int device) {
  if (!pointer) return cudaErrorInvalidValue;
  cudaPointerAttributes attributes{};
  const auto error = cudaPointerGetAttributes(&attributes, pointer);
  if (error != cudaSuccess) return error;
  return attributes.type == cudaMemoryTypeDevice && attributes.device == device
             ? cudaSuccess
             : cudaErrorInvalidValue;
}
cudaError_t drain(cudaError_t error, cudaStream_t stream) {
  const auto completed = cudaStreamSynchronize(stream);
  return error == cudaSuccess ? completed : error;
}
cudaError_t gate(int* status, cudaStream_t stream) {
  int failed{};
  auto error = cudaMemcpyAsync(&failed, status, sizeof(failed), cudaMemcpyDeviceToHost, stream);
  error = drain(error, stream);
  return error == cudaSuccess && failed ? cudaErrorInvalidValue : error;
}
struct Range {
  const void* pointer{};
  std::size_t bytes{}, alignment{alignof(double)};
};
std::array<Range, 10> basis_ranges(const Basis& basis) {
  const auto shells = basis.orbital_shells + basis.auxiliary_shells;
  return {{{basis.positions, 3 * basis.atoms * sizeof(double)},
           {basis.exponents, (basis.primitives + 1) * sizeof(double)},
           {basis.coefficients, (basis.primitives + 1) * sizeof(double)},
           {basis.component_coefficients,
            (basis.cartesian_nao + basis.cartesian_naux + 1) * sizeof(double)},
           {basis.shell_atoms, (shells + 1) * sizeof(std::int32_t), alignof(std::int32_t)},
           {basis.shell_angular, (shells + 1) * sizeof(std::uint8_t), alignof(std::uint8_t)},
           {basis.shell_primitive_offsets, (shells + 2) * sizeof(std::int64_t)},
           {basis.shell_cartesian_offsets, (shells + 2) * sizeof(std::int64_t)},
           {basis.orbital_expansions, basis.nao * sizeof(Expansion)},
           {basis.auxiliary_expansions, basis.naux * sizeof(Expansion)}}};
}
__global__ void finite(const double* values, std::size_t count, int* status) {
  for (std::size_t index = blockIdx.x * blockDim.x + threadIdx.x; index < count;
       index += static_cast<std::size_t>(blockDim.x) * gridDim.x)
    if (!isfinite(values[index])) atomicExch(status, 1);
}
__global__ void symmetric(const double* values, std::size_t extent, std::size_t channels,
                          double absolute, double relative, int* status) {
  if (atomicAdd(status, 0)) return;
  for (std::size_t index = blockIdx.x * blockDim.x + threadIdx.x;
       index < extent * extent * channels;
       index += static_cast<std::size_t>(blockDim.x) * gridDim.x) {
    const auto pair = index / channels;
    const auto other = ((pair % extent) * extent + pair / extent) * channels + index % channels;
    const auto first_value = values[index], second_value = values[other];
    if (!isfinite(first_value) || !isfinite(second_value) ||
        fabs(first_value - second_value) >
            absolute + relative * fmax(fabs(first_value), fabs(second_value)))
      atomicExch(status, 1);
  }
}
__global__ void validate_basis(Basis basis, const Shell* shells, int* status) {
  for (std::size_t index = blockIdx.x * blockDim.x + threadIdx.x;
       index < max(basis.primitives, basis.orbital_shells + basis.auxiliary_shells);
       index += static_cast<std::size_t>(blockDim.x) * gridDim.x) {
    if (index < basis.primitives &&
        (!(basis.exponents[index] > 0) || !isfinite(basis.exponents[index]) ||
         !isfinite(basis.coefficients[index])))
      atomicExch(status, 1);
    if (index >= basis.orbital_shells + basis.auxiliary_shells) continue;
    const auto shell = shells[index];
    const auto auxiliary = index >= basis.orbital_shells;
    const auto cartesian_base = auxiliary ? basis.cartesian_nao : 0;
    const auto count = (shell.angular + 1) * (shell.angular + 2) / 2;
    if (basis.shell_atoms[index] != shell.atom || basis.shell_angular[index] != shell.angular ||
        basis.shell_primitive_offsets[index] != shell.primitive_begin ||
        basis.shell_primitive_offsets[index + 1] != shell.primitive_begin + shell.primitive_count ||
        basis.shell_cartesian_offsets[index] != cartesian_base + shell.cartesian_begin ||
        basis.shell_cartesian_offsets[index + 1] !=
            cartesian_base + shell.cartesian_begin + count) {
      atomicExch(status, 1);
      continue;
    }
    const auto* expansions = auxiliary ? basis.auxiliary_expansions : basis.orbital_expansions;
    for (std::size_t orbital = 0; orbital < shell.ao_count; ++orbital) {
      const auto expansion = expansions[shell.ao_begin + orbital];
      if (!expansion.count || expansion.count > (auxiliary ? 6U : 3U)) {
        atomicExch(status, 1);
        continue;
      }
      for (unsigned term = 0; term < expansion.count; ++term) {
        const auto component = expansion.cartesian[term];
        if (component < 0 || component < shell.cartesian_begin ||
            component >= shell.cartesian_begin + count || !isfinite(expansion.coefficients[term]) ||
            !isfinite(basis.component_coefficients[cartesian_base + component]))
          atomicExch(status, 1);
      }
    }
  }
}
__global__ void build_records(const DeviceGroup* group, Basis basis, std::size_t begin,
                              std::size_t count, double* records, int* status) {
  // Always initialize every submitted record, even after a prior component
  // failed: the linked leaf reports errors but does not consume an abort flag.
  const auto centers = group->centers, stride = 4 * centers + 1;
  for (std::size_t index = threadIdx.x; index < count; index += blockDim.x) {
    auto radial = begin + index;
    auto* record = records + stride * index;
    double scale = 1;
    for (unsigned slot = centers; slot-- > 0;) {
      const auto shell = group->shells[slot];
      const auto primitive = shell.primitive_begin + radial % shell.primitive_count;
      radial /= shell.primitive_count;
      record[slot] = basis.exponents[primitive];
      scale *= basis.coefficients[primitive];
      for (unsigned axis = 0; axis < 3; ++axis)
        record[centers + 3 * slot + axis] = basis.positions[3 * shell.atom + axis];
    }
    if (!isfinite(scale)) {
      atomicExch(status, 1);
      scale = 0;
    }
    record[stride - 1] = scale;
  }
}
__global__ void accumulate(const DeviceGroup* group, const DeviceTile* tile, const double* raw,
                           double* cache, int* status) {
  if (atomicAdd(status, 0)) return;
  const auto fields = 1 + 6 * group->centers;
  for (unsigned index = threadIdx.x; index < tile->count * fields; index += blockDim.x)
    cache[index % fields * group->components + tile->components[index / fields]] += raw[index];
}
template <bool Contract>
__global__ void project(CudaDfNuclearShape shape, const DeviceGroup* group, Basis basis,
                        const double* direction, CudaDfNuclearWeights weights, const double* cache,
                        double* candidate, double* projected, int* status) {
  if (atomicAdd(status, 0)) return;
  const auto centers = group->centers, coordinates = 3 * centers;
  for (std::size_t index = blockIdx.x * blockDim.x + threadIdx.x; index < group->public_components;
       index += static_cast<std::size_t>(blockDim.x) * gridDim.x) {
    Expansion expansions[3]{};
    std::size_t aos[3]{};
    auto remaining = index;
    for (unsigned slot = centers; slot-- > 0;) {
      const auto shell = group->shells[slot];
      aos[slot] = shell.ao_begin + remaining % shell.ao_count;
      remaining /= shell.ao_count;
      const auto auxiliary = group->metric || slot == 2;
      expansions[slot] =
          (auxiliary ? basis.auxiliary_expansions : basis.orbital_expansions)[aos[slot]];
    }
    const auto output_index = group->metric ? aos[0] * shape.naux + aos[1]
                                            : (aos[0] * shape.nao + aos[1]) * shape.naux + aos[2];
    double weight{}, delta_weight{};
    if constexpr (Contract) {
      weight = (group->metric ? weights.metric : weights.three_center)[output_index];
      delta_weight =
          (group->metric ? weights.direction_metric : weights.direction_three_center)[output_index];
    }
    double value{}, motion{}, gradient[9]{}, hvp[9]{};
    for (unsigned first = 0; first < expansions[0].count; ++first)
      for (unsigned second = 0; second < expansions[1].count; ++second)
        for (unsigned third = 0; third < (centers == 3 ? expansions[2].count : 1); ++third) {
          const unsigned terms[3]{first, second, third};
          std::size_t component{};
          double projection = 1;
          for (unsigned slot = 0; slot < centers; ++slot) {
            const auto absolute = expansions[slot].cartesian[terms[slot]];
            const auto auxiliary = group->metric || slot == 2;
            const auto cartesian_base = auxiliary ? shape.cartesian_nao : 0;
            component = component * ((group->shells[slot].angular + 1) *
                                     (group->shells[slot].angular + 2) / 2) +
                        absolute - group->shells[slot].cartesian_begin;
            projection *= expansions[slot].coefficients[terms[slot]] *
                          basis.component_coefficients[cartesian_base + absolute];
          }
          for (unsigned coordinate = 0; coordinate < coordinates; ++coordinate) {
            double result[4];
            if (!ad::contract(cache[component],
                              cache[(1 + coordinate) * group->components + component],
                              cache[(1 + coordinates + coordinate) * group->components + component],
                              weight, delta_weight, projection,
                              direction[3 * group->atoms[coordinate / 3] + coordinate % 3], result))
              atomicExch(status, 1);
            if (!coordinate) value += result[0];
            motion += result[1];
            if constexpr (Contract) {
              gradient[coordinate] += result[2];
              hvp[coordinate] += result[3];
            }
          }
        }
    if constexpr (Contract) {
      for (unsigned coordinate = 0; coordinate < coordinates; ++coordinate) {
        projected[index * coordinates + coordinate] = gradient[coordinate];
        projected[(group->public_components + index) * coordinates + coordinate] = hvp[coordinate];
      }
    } else {
      const auto raw_count = shape.nao * shape.nao * shape.naux,
                 metric_count = shape.naux * shape.naux;
      const auto offset = group->metric ? raw_count : 0;
      candidate[offset + output_index] = value;
      candidate[raw_count + metric_count + offset + output_index] = motion;
    }
  }
}
__global__ void scatter(const DeviceGroup* group, std::size_t atoms, const double* projected,
                        double* candidate, int* status) {
  if (atomicAdd(status, 0)) return;
  const auto axis = threadIdx.x;
  if (axis >= 3) return;
  const auto coordinates = 3 * group->centers;
  // Fixed-order physical-axis ownership retains every coincident mathematical
  // center without double atomics or a left-coordinate primitive traversal.
  for (unsigned center = 0; center < group->centers; ++center) {
    double gradient{}, hvp{};
    for (std::size_t index = 0; index < group->public_components; ++index) {
      gradient += projected[index * coordinates + 3 * center + axis];
      hvp += projected[(group->public_components + index) * coordinates + 3 * center + axis];
    }
    candidate[3 * group->atoms[center] + axis] += gradient;
    candidate[3 * (atoms + group->atoms[center]) + axis] += hvp;
  }
}
}  // namespace

struct CudaDfNuclearResponse::Impl {
  CudaDfNuclearShape shape{};
  CudaDfNuclearOptions options{};
  CudaDfNuclearReference reference{};
  Schedule schedule;
  CudaDfNuclearWork receipt{};
  void* arena{};
  bool ready{};
  template <class Value>
  Value* at(std::size_t offset) const {
    return reinterpret_cast<Value*>(static_cast<char*>(arena) + offset);
  }
  cudaError_t authenticate(cudaStream_t stream) const {
    if (!ready || stream != reference.stream) return cudaErrorInvalidResourceHandle;
    int device{-1};
    auto error = cudaGetDevice(&device);
    if (error != cudaSuccess) return error;
    if (device != reference.basis.device_id) return cudaErrorInvalidDevice;
    cudaStreamCaptureStatus capture{};
    if ((error = cudaStreamIsCapturing(stream, &capture)) != cudaSuccess) return error;
    if (capture != cudaStreamCaptureStatusNone) return cudaErrorStreamCaptureUnsupported;
    return reference.authenticate(reference.owner);
  }
  cudaError_t buffers(std::span<const Range> inputs, std::span<const Range> outputs) const {
    const auto& plan = schedule.layout;
    for (const auto& input : inputs) {
      if (!input.bytes) continue;
      if (!disjoint(arena, plan.bytes, input.pointer, input.bytes) ||
          reinterpret_cast<std::uintptr_t>(input.pointer) % input.alignment)
        return cudaErrorInvalidValue;
      if (const auto error = device_buffer(input.pointer, reference.basis.device_id);
          error != cudaSuccess)
        return error;
    }
    for (std::size_t index = 0; index < outputs.size(); ++index) {
      const auto& output = outputs[index];
      if (!disjoint(arena, plan.bytes, output.pointer, output.bytes) ||
          reinterpret_cast<std::uintptr_t>(output.pointer) % alignof(double))
        return cudaErrorInvalidValue;
      if (const auto error = device_buffer(output.pointer, reference.basis.device_id);
          error != cudaSuccess)
        return error;
      for (const auto& input : inputs)
        if (input.bytes && !disjoint(output.pointer, output.bytes, input.pointer, input.bytes))
          return cudaErrorInvalidValue;
      for (std::size_t previous = 0; previous < index; ++previous)
        if (!disjoint(output.pointer, output.bytes, outputs[previous].pointer,
                      outputs[previous].bytes))
          return cudaErrorInvalidValue;
    }
    return cudaSuccess;
  }
  cudaError_t symmetric_panel(const double* values, bool metric) const {
    const auto& plan = schedule.layout;
    symmetric<<<blocks(metric ? plan.metric_elements : plan.raw_elements), 128, 0,
                reference.stream>>>(values, metric ? shape.naux : shape.nao,
                                    metric ? 1 : shape.naux, options.symmetry_absolute_tolerance,
                                    options.symmetry_relative_tolerance, at<int>(plan.status));
    return cudaGetLastError();
  }
  template <bool Contract>
  cudaError_t execute(const double* direction, CudaDfNuclearWeights weights,
                      std::span<const Range> outputs, cudaStream_t stream) {
    if (receipt.attempted_passes == UINT64_MAX) return cudaErrorInvalidValue;
    ++receipt.attempted_passes;
    auto error = authenticate(stream);
    if (error != cudaSuccess) return error;
    const auto& plan = schedule.layout;
    auto basis = basis_ranges(reference.basis);
    std::array<Range, 15> inputs{};
    std::copy(basis.begin(), basis.end(), inputs.begin());
    inputs[10] = {direction, 3 * shape.atoms * sizeof(double)};
    if constexpr (Contract) {
      inputs[11] = {weights.three_center, plan.raw_elements * sizeof(double)};
      inputs[12] = {weights.metric, plan.metric_elements * sizeof(double)};
      inputs[13] = {weights.direction_three_center, plan.raw_elements * sizeof(double)};
      inputs[14] = {weights.direction_metric, plan.metric_elements * sizeof(double)};
    }
    if ((error = buffers(inputs, outputs)) != cudaSuccess) return error;
    if (reinterpret_cast<std::uintptr_t>(direction) % alignof(double)) return cudaErrorInvalidValue;
    if (receipt.radial_records > UINT64_MAX - plan.work.radial_records_per_pass ||
        receipt.component_evaluations > UINT64_MAX - plan.work.component_evaluations_per_pass ||
        receipt.projection_terms_bound > UINT64_MAX - plan.work.projection_terms_bound_per_pass ||
        receipt.reduction_terms_bound > UINT64_MAX - plan.work.reduction_terms_bound_per_pass)
      return cudaErrorInvalidValue;
    auto* status = at<int>(plan.status);
    error = cudaMemsetAsync(status, 0, sizeof(int), stream);
    if (error == cudaSuccess) {
      finite<<<blocks(3 * shape.atoms), 128, 0, stream>>>(direction, 3 * shape.atoms, status);
      error = cudaGetLastError();
    }
    if constexpr (Contract) {
      for (unsigned index = 11; index < 15 && error == cudaSuccess; ++index) {
        if (reinterpret_cast<std::uintptr_t>(inputs[index].pointer) % alignof(double))
          return drain(cudaErrorInvalidValue, stream);
        error = symmetric_panel(static_cast<const double*>(inputs[index].pointer), index % 2 == 0);
      }
    }
    if (error == cudaSuccess) error = gate(status, stream);
    if (error != cudaSuccess) return drain(error, stream);
    auto* candidate = at<double>(plan.candidates);
    auto* records = at<double>(plan.records);
    auto* raw = at<double>(plan.raw);
    auto* cache = at<double>(plan.cache);
    auto* projected = at<double>(plan.projected);
    const auto* groups = at<DeviceGroup>(plan.groups);
    const auto* tiles = at<DeviceTile>(plan.tiles);
    error = cudaMemsetAsync(candidate, 0, plan.candidate_count * sizeof(double), stream);
    for (std::size_t index = 0; index < schedule.groups.size() && error == cudaSuccess; ++index) {
      const auto& group = schedule.groups[index];
      error = cudaMemsetAsync(
          cache, 0, group.device.components * (1 + 6 * group.device.centers) * sizeof(double),
          stream);
      for (const auto tile_index : schedule.closure[group.closure]) {
        const auto& tile = schedule.tiles[tile_index];
        for (std::size_t begin = 0; begin < group.radials && error == cudaSuccess;) {
          const auto count = std::min(options.radial_tile, group.radials - begin);
          build_records<<<1, 128, 0, stream>>>(groups + index, reference.basis, begin, count,
                                               records, status);
          error = cudaGetLastError();
          if (error == cudaSuccess) {
            receipt.radial_records += count;
            error = static_cast<cudaError_t>(tile.program->integral(
                records, count, groups[index].atoms, direction, raw, status, stream));
          }
          if (error == cudaSuccess) {
            receipt.component_evaluations += count * tile.device.count;
            receipt.reduction_terms_bound +=
                count * tile.device.count * (1 + 6 * group.device.centers);
            accumulate<<<1, 128, 0, stream>>>(groups + index, tiles + tile_index, raw, cache,
                                              status);
            error = cudaGetLastError();
            if (error == cudaSuccess)
              receipt.reduction_terms_bound += tile.device.count * (1 + 6 * group.device.centers);
          }
          begin += count;
        }
      }
      if (error == cudaSuccess) {
        project<Contract><<<blocks(group.device.public_components), 128, 0, stream>>>(
            shape, groups + index, reference.basis, direction, weights, cache, candidate, projected,
            status);
        error = cudaGetLastError();
        receipt.projection_terms_bound += group.device.public_components *
                                          (group.device.metric ? 36 : 54) * 3 *
                                          group.device.centers;
        receipt.reduction_terms_bound += group.device.public_components *
                                         (group.device.metric ? 36 : 54) *
                                         (1 + 9 * group.device.centers);
      }
      if constexpr (Contract)
        if (error == cudaSuccess) {
          scatter<<<1, 32, 0, stream>>>(groups + index, shape.atoms, projected, candidate, status);
          error = cudaGetLastError();
          receipt.reduction_terms_bound +=
              6 * group.device.centers * (group.device.public_components + 1);
        }
    }
    if (error == cudaSuccess) {
      if constexpr (Contract) {
        finite<<<blocks(6 * shape.atoms), 128, 0, stream>>>(candidate, 6 * shape.atoms, status);
        error = cudaGetLastError();
      } else {
        error = symmetric_panel(candidate, false);
        if (error == cudaSuccess) error = symmetric_panel(candidate + plan.raw_elements, true);
        if (error == cudaSuccess)
          error = symmetric_panel(candidate + plan.raw_elements + plan.metric_elements, false);
        if (error == cudaSuccess)
          error = symmetric_panel(candidate + 2 * plan.raw_elements + plan.metric_elements, true);
      }
    }
    if (error == cudaSuccess) error = gate(status, stream);
    if (error != cudaSuccess) return drain(error, stream);
    if ((error = authenticate(stream)) != cudaSuccess) return error;
    std::size_t offset{};
    for (const auto& output : outputs) {
      error = cudaMemcpyAsync(const_cast<void*>(output.pointer), candidate + offset, output.bytes,
                              cudaMemcpyDeviceToDevice, stream);
      if (error != cudaSuccess) break;
      offset += output.bytes / sizeof(double);
    }
    error = drain(error, stream);
    if (error == cudaSuccess) {
      if constexpr (Contract)
        ++receipt.contraction_passes;
      else
        ++receipt.value_passes;
    }
    return error;
  }
};

CudaDfNuclearResponse::CudaDfNuclearResponse() = default;
CudaDfNuclearResponse::~CudaDfNuclearResponse() = default;
cudaError_t cuda_df_nuclear_workspace(const CudaDfNuclearShape& shape,
                                      std::span<const Shell> orbital,
                                      std::span<const Shell> auxiliary,
                                      const CudaDfNuclearOptions& options,
                                      std::size_t& bytes) noexcept {
  bytes = 0;
  try {
    Schedule state;
    const auto error = schedule(shape, orbital, auxiliary, options, state);
    if (error == cudaSuccess) bytes = state.layout.bytes;
    return error;
  } catch (const std::bad_alloc&) {
    return cudaErrorMemoryAllocation;
  } catch (...) {
    return cudaErrorUnknown;
  }
}
cudaError_t CudaDfNuclearResponse::prepare(const CudaDfNuclearShape& shape,
                                           std::span<const Shell> orbital,
                                           std::span<const Shell> auxiliary,
                                           const CudaDfNuclearOptions& options,
                                           const CudaDfNuclearReference& reference, void* workspace,
                                           std::size_t workspace_bytes) noexcept {
  impl_.reset();
  try {
    auto state = std::make_unique<Impl>();
    auto error = schedule(shape, orbital, auxiliary, options, state->schedule);
    if (error != cudaSuccess) return error;
    const auto& basis = reference.basis;
    const auto& plan = state->schedule.layout;
    if (!basis || !reference.stream || !reference.authenticate || !reference.owner || !workspace ||
        reinterpret_cast<std::uintptr_t>(workspace) % alignof(double) ||
        workspace_bytes < plan.bytes || basis.atoms != shape.atoms ||
        basis.primitives != shape.primitives || basis.nao != shape.nao ||
        basis.naux != shape.naux || basis.cartesian_nao != shape.cartesian_nao ||
        basis.cartesian_naux != shape.cartesian_naux || basis.orbital_shells != orbital.size() ||
        basis.auxiliary_shells != auxiliary.size())
      return cudaErrorInvalidValue;
    state->shape = shape;
    state->options = options;
    state->reference = reference;
    state->arena = workspace;
    state->ready = true;
    if ((error = state->authenticate(reference.stream)) != cudaSuccess) return error;
    if ((error = device_buffer(workspace, basis.device_id)) != cudaSuccess) return error;
    if ((error = state->buffers(basis_ranges(basis), {})) != cudaSuccess) return error;
    auto* status = state->at<int>(plan.status);
    auto* shells = state->at<Shell>(plan.shells);
    error = cudaMemsetAsync(status, 0, sizeof(int), reference.stream);
    if (error == cudaSuccess)
      error = cudaMemcpyAsync(shells, orbital.data(), orbital.size_bytes(), cudaMemcpyHostToDevice,
                              reference.stream);
    if (error == cudaSuccess)
      error = cudaMemcpyAsync(shells + orbital.size(), auxiliary.data(), auxiliary.size_bytes(),
                              cudaMemcpyHostToDevice, reference.stream);
    for (std::size_t index = 0; index < state->schedule.groups.size() && error == cudaSuccess;
         ++index)
      error = cudaMemcpyAsync(state->at<DeviceGroup>(plan.groups) + index,
                              &state->schedule.groups[index].device, sizeof(DeviceGroup),
                              cudaMemcpyHostToDevice, reference.stream);
    for (std::size_t index = 0; index < state->schedule.tiles.size() && error == cudaSuccess;
         ++index)
      error = cudaMemcpyAsync(state->at<DeviceTile>(plan.tiles) + index,
                              &state->schedule.tiles[index].device, sizeof(DeviceTile),
                              cudaMemcpyHostToDevice, reference.stream);
    if (error == cudaSuccess) {
      finite<<<blocks(3 * shape.atoms), 128, 0, reference.stream>>>(basis.positions,
                                                                    3 * shape.atoms, status);
      error = cudaGetLastError();
    }
    if (error == cudaSuccess) {
      validate_basis<<<blocks(std::max(shape.primitives, orbital.size() + auxiliary.size())), 128,
                       0, reference.stream>>>(basis, shells, status);
      error = cudaGetLastError();
    }
    if (error == cudaSuccess) error = gate(status, reference.stream);
    error = drain(error, reference.stream);
    if (error != cudaSuccess) return error;
    if ((error = state->authenticate(reference.stream)) != cudaSuccess) return error;
    state->receipt = plan.work;
    impl_ = std::move(state);
    return cudaSuccess;
  } catch (const std::bad_alloc&) {
    return cudaErrorMemoryAllocation;
  } catch (...) {
    return cudaErrorUnknown;
  }
}
cudaError_t CudaDfNuclearResponse::values(const double* direction,
                                          const CudaDfNuclearValues& output,
                                          cudaStream_t stream) noexcept {
  if (!impl_) return cudaErrorInvalidResourceHandle;
  const auto& plan = impl_->schedule.layout;
  const Range ranges[]{{output.three_center, plan.raw_elements * sizeof(double)},
                       {output.metric, plan.metric_elements * sizeof(double)},
                       {output.direction_three_center, plan.raw_elements * sizeof(double)},
                       {output.direction_metric, plan.metric_elements * sizeof(double)}};
  return impl_->execute<false>(direction, {}, ranges, stream);
}
cudaError_t CudaDfNuclearResponse::contract(const double* direction,
                                            const CudaDfNuclearWeights& weights,
                                            const CudaDfNuclearVectors& output,
                                            cudaStream_t stream) noexcept {
  if (!impl_) return cudaErrorInvalidResourceHandle;
  const Range ranges[]{{output.gradient, 3 * impl_->shape.atoms * sizeof(double)},
                       {output.hvp, 3 * impl_->shape.atoms * sizeof(double)}};
  return impl_->execute<true>(direction, weights, ranges, stream);
}
const CudaDfNuclearWork& CudaDfNuclearResponse::work() const noexcept {
  static const CudaDfNuclearWork empty;
  return impl_ ? impl_->receipt : empty;
}
}  // namespace generativeqc::integrals
