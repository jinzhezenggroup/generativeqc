#include <algorithm>
#include <array>
#include <climits>
#include <cmath>
#include <cuda/atomic>
#include <new>
#include <vector>

#include "dft/factorized_hessian_integral.hpp"
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
__host__ __device__ std::size_t cartesian(std::size_t angular) {
  return (angular + 1) * (angular + 2) / 2;
}
__host__ __device__ unsigned expansion_bound(std::size_t angular, bool auxiliary) {
  const unsigned support = cartesian(angular);
  const unsigned capacity = auxiliary ? 6 : 3;
  return support < capacity ? support : capacity;
}
unsigned blocks(std::size_t count) {
  return static_cast<unsigned>(std::min((count + 127) / 128, std::size_t{65535}));
}
struct DeviceGroup {
  Shell shells[3]{};
  long long atoms[3]{};
  std::size_t centers{}, components{}, public_components{};
  std::size_t tape_offset{};
  bool metric{};
};
__host__ __device__ bool includes_atom(const DeviceGroup& group, long long atom) {
  return atom < 0 || group.atoms[0] == atom || group.atoms[1] == atom ||
         (!group.metric && group.atoms[2] == atom);
}
struct DeviceTile {
  std::uint32_t components[16]{}, count{};
};
struct Tile {
  const aot::Program* program{};
  DeviceTile device{};
};
struct Group {
  std::size_t radials{}, closure{}, projection_bound{};
};
struct Layout {
  std::size_t bytes{}, shells{}, groups{}, tiles{}, records{}, raw{}, cache{}, projected{},
      candidates{}, status{}, raw_elements{}, metric_elements{}, candidate_count{},
      max_components{}, max_fields{}, max_public{};
  std::size_t batch_width{}, batch_radials{}, batch_records{}, batch_raw{}, batch_projected{},
      batch_inputs{}, atom_offsets{}, atom_incidence{}, retained_raw{}, retained_direction{},
      direction_mismatch{};
  std::size_t active_sp_groups{};
  bool batch_sp{};
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
bool reserve_directional_tape(Layout& layout, std::size_t groups, std::size_t atoms) {
  return layout.reserve(groups, 19 * sizeof(double), layout.retained_raw) &&
         layout.reserve(atoms, 3 * sizeof(double), layout.retained_direction) &&
         layout.reserve(2, sizeof(int), layout.direction_mismatch);
}
struct SpWindow {
  struct ActiveSpan {
    std::uint32_t offset{}, count{};
  };
  std::size_t first{}, stop{}, radials{}, projections{}, projection_reductions{};
  std::array<std::size_t, 96> atom_radials{};
  std::array<ActiveSpan, 96> active{};
};
struct Schedule {
  Layout layout;
  std::vector<Group> groups;
  // Keep upload records contiguous; the host-only launch selectors never
  // cross the device boundary and the group order is shared by the arrays.
  std::vector<DeviceGroup> descriptors;
  std::vector<Tile> tiles;
  std::array<std::vector<std::size_t>, 150> closure;
  std::vector<SpWindow> sp_windows;
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
      options.symmetry_relative_tolerance < 0 || !std::isfinite(options.range_omega) ||
      options.range_omega < 0)
    return cudaErrorInvalidValue;
  const auto radial = options.range_omega == 0 ? "full_range" : "long_range";
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
  const bool batch_sp = options.batch_cartesian_sp;
  const auto small_angular = [](const Shell& shell) { return shell.angular <= 1; };
  if (batch_sp &&
      (options.range_omega != 0 || shape.atoms > 96 || !options.retain_directional_source ||
       shape.nao != shape.cartesian_nao || shape.naux != shape.cartesian_naux ||
       !std::all_of(orbital.begin(), orbital.end(), small_angular) ||
       !std::all_of(auxiliary.begin(), auxiliary.end(), small_angular)))
    return cudaErrorNotSupported;
  const auto pairs = [batch_sp](std::size_t shells, std::size_t& result) {
    if (!multiply(shells, shells + batch_sp, result)) return false;
    if (batch_sp) result /= 2;
    return true;
  };
  std::size_t groups{}, extra{}, count{};
  if (!pairs(orbital.size(), groups) || !multiply(groups, auxiliary.size(), groups) ||
      !pairs(auxiliary.size(), extra) || !add(groups, extra) || groups > options.max_groups ||
      !multiply(shape.nao, shape.nao, count) || !multiply(count, shape.naux, plan.raw_elements) ||
      !multiply(shape.naux, shape.naux, plan.metric_elements))
    return cudaErrorInvalidValue;
  // Reject impossible descriptor storage before growing host metadata or
  // traversing the ordered product. Remaining workspace is checked below.
  if (groups > options.max_workspace_bytes / sizeof(DeviceGroup)) return cudaErrorInvalidValue;
  result.groups.reserve(groups);
  result.descriptors.reserve(groups);
  std::size_t tape_elements{};
  const auto append_group = [&](const Shell& first, const Shell& second, const Shell* third) {
    Group group;
    DeviceGroup device;
    device.centers = third ? 3 : 2;
    device.metric = !third;
    device.shells[0] = first;
    device.shells[1] = second;
    if (third) device.shells[2] = *third;
    device.components = device.public_components = group.radials = group.projection_bound = 1;
    std::size_t key{};
    for (unsigned slot = 0; slot < device.centers; ++slot) {
      const auto& shell = device.shells[slot];
      device.atoms[slot] = shell.atom;
      key = key * 5 + shell.angular;
      if (!multiply(group.radials, shell.primitive_count, group.radials) ||
          !multiply(device.components, cartesian(shell.angular), device.components) ||
          !multiply(device.public_components, shell.ao_count, device.public_components) ||
          !multiply(group.projection_bound, expansion_bound(shell.angular, !third || slot == 2),
                    group.projection_bound))
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
            signature.angular.size() != device.centers || signature.radial != radial ||
            !program.accepts_omega(options.range_omega))
          continue;
        bool matches = true;
        for (unsigned slot = 0; slot < device.centers; ++slot)
          matches &= signature.angular[slot] == device.shells[slot].angular;
        if (!matches) continue;
        if (!program.integral || program.centers != device.centers ||
            program.record_stride != 4 * device.centers + 1 ||
            program.outputs != 1 + 6 * device.centers || program.components.empty() ||
            program.components.size() > 16)
          return cudaErrorNotSupported;
        Tile tile{&program, {}};
        tile.device.count = program.components.size();
        for (unsigned index = 0; index < tile.device.count; ++index) {
          const auto component = program.components[index];
          if (component >= device.components || covered[component]) return cudaErrorNotSupported;
          covered[component] = true;
          tile.device.components[index] = component;
        }
        closure.push_back(result.tiles.size());
        result.tiles.push_back(tile);
      }
      if (!std::all_of(covered.begin(), covered.begin() + device.components,
                       [](bool value) { return value; }))
        return cudaErrorNotSupported;
    }
    std::size_t components{}, projections{}, reductions{}, records{}, cache_terms{},
        projection_reductions{}, scatter_terms{};
    const auto coordinates = 3 * device.centers;
    const auto fields = 1 + 2 * coordinates;
    if (batch_sp) {
      device.tape_offset = tape_elements;
      if (group.radials > options.radial_tile || !add(tape_elements, device.components * fields))
        return cudaErrorInvalidValue;
    }
    const auto batches =
        group.radials / options.radial_tile + (group.radials % options.radial_tile != 0);
    // Count submitted records for every component tile, primitive-field sums,
    // cache additions, sparse projection sums and deterministic scatter. The
    // reduction bound covers either pass, including coincident center writes.
    if (!multiply(group.radials, device.components, components) ||
        !multiply(group.radials, closure.size(), records) ||
        !multiply(device.public_components, group.projection_bound, projections) ||
        !multiply(projections, 1 + 3 * coordinates, projection_reductions) ||
        !multiply(projections, coordinates, projections) ||
        !multiply(components, fields, reductions) ||
        !multiply(device.components, fields, cache_terms) ||
        !multiply(cache_terms, batches, cache_terms) ||
        !multiply(device.public_components + 1, 2 * coordinates, scatter_terms) ||
        !add(reductions, cache_terms) || !add(reductions, projection_reductions) ||
        !add(reductions, scatter_terms) || !add(plan.work.radial_records_per_pass, records) ||
        !add(plan.work.component_evaluations_per_pass, components) ||
        !add(plan.work.projection_terms_bound_per_pass, projections) ||
        !add(plan.work.reduction_terms_bound_per_pass, reductions) ||
        plan.work.component_evaluations_per_pass > options.max_component_evaluations ||
        plan.work.projection_terms_bound_per_pass > options.max_projection_terms ||
        plan.work.reduction_terms_bound_per_pass > options.max_reduction_terms_bound)
      return cudaErrorInvalidValue;
    plan.max_components = std::max(plan.max_components, device.components);
    plan.max_public = std::max(plan.max_public, device.public_components);
    plan.max_fields = std::max(plan.max_fields, 1 + 6 * device.centers);
    result.groups.push_back(group);
    result.descriptors.push_back(device);
    return cudaSuccess;
  };
  if (batch_sp) {
    // Independent AO outputs may be reordered by linked closure. The later
    // atom reduction still owns a deterministic FP64 sum over all incidences.
    std::array<std::vector<const Shell*>, 2> orbitals, auxiliaries;
    for (const auto& shell : orbital) orbitals[shell.angular].push_back(&shell);
    for (const auto& shell : auxiliary) auxiliaries[shell.angular].push_back(&shell);
    for (const auto& left : orbitals)
      for (const auto& right : orbitals)
        for (const auto& fitting : auxiliaries)
          for (const auto* first : left)
            for (const auto* second : right) {
              if (first->ao_begin > second->ao_begin) continue;
              for (const auto* third : fitting)
                if (const auto error = append_group(*first, *second, third); error != cudaSuccess)
                  return error;
            }
    for (const auto& left : auxiliaries)
      for (const auto& right : auxiliaries)
        for (const auto* first : left)
          for (const auto* second : right) {
            if (first->ao_begin > second->ao_begin) continue;
            if (const auto error = append_group(*first, *second, nullptr); error != cudaSuccess)
              return error;
          }
  } else {
    for (const auto& first : orbital)
      for (const auto& second : orbital)
        for (const auto& third : auxiliary)
          if (const auto error = append_group(first, second, &third); error != cudaSuccess)
            return error;
    for (const auto& first : auxiliary)
      for (const auto& second : auxiliary)
        if (const auto error = append_group(first, second, nullptr); error != cudaSuccess)
          return error;
  }
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
  const bool single_component =
      plan.max_components == 1 && plan.max_public == 1 && groups <= UINT32_MAX / 8;
  if (batch_sp) {
    if (groups > UINT32_MAX / 8 ||
        !std::all_of(result.tiles.begin(), result.tiles.end(),
                     [](const Tile& tile) { return tile.program->integral_batch != nullptr; }))
      return cudaErrorNotSupported;
    std::size_t radials{};
    for (const auto& group : result.groups) radials = std::max(radials, group.radials);
    for (auto width = std::min<std::size_t>(4096, groups); width >= 2; width /= 2) {
      auto trial = plan;
      if (!trial.reserve(width * radials * 13, sizeof(double), trial.batch_records) ||
          !trial.reserve(width * 16 * 19, sizeof(double), trial.batch_raw) ||
          !trial.reserve(width * 18, sizeof(double), trial.batch_projected) ||
          !trial.reserve(width, sizeof(aot::FactorizedIntegralBatch), trial.batch_inputs) ||
          !trial.reserve(shape.atoms + 1, sizeof(std::uint32_t), trial.atom_offsets) ||
          !trial.reserve(3 * groups, sizeof(std::uint32_t), trial.atom_incidence) ||
          !trial.reserve(tape_elements, sizeof(double), trial.retained_raw) ||
          !trial.reserve(3 * shape.atoms, sizeof(double), trial.retained_direction) ||
          !trial.reserve(2, sizeof(int), trial.direction_mismatch) ||
          trial.bytes > options.max_workspace_bytes)
        continue;
      trial.batch_width = width;
      trial.batch_radials = radials;
      trial.batch_sp = true;
      plan = trial;
      break;
    }
    if (!plan.batch_sp) return cudaErrorMemoryAllocation;
    // Optional sparse dispatch. Its upper bound is one index per distinct
    // center incidence. No numerical screening or new primitive leaf is used.
    auto indexed = plan;
    if (indexed.reserve(3 * groups, sizeof(std::uint32_t), indexed.active_sp_groups) &&
        indexed.bytes <= options.max_workspace_bytes)
      plan = indexed;
    std::uint32_t active_offset{};
    for (std::size_t first = 0; first < groups;) {
      SpWindow window{};
      window.first = window.stop = first;
      const auto closure = result.groups[first].closure;
      while (window.stop < groups && window.stop - first < plan.batch_width &&
             result.groups[window.stop].closure == closure) {
        const auto& group = result.groups[window.stop];
        const auto& device = result.descriptors[window.stop++];
        window.radials += group.radials;
        const auto coordinates = 3 * device.centers;
        window.projections += coordinates * group.projection_bound * device.public_components;
        window.projection_reductions +=
            (1 + 3 * coordinates) * group.projection_bound * device.public_components +
            2 * coordinates * (device.public_components + 1);
        for (unsigned center = 0; center < device.centers; ++center) {
          const auto atom = device.atoms[center];
          if (std::find(device.atoms, device.atoms + center, atom) == device.atoms + center) {
            window.atom_radials[atom] += group.radials;
            ++window.active[atom].count;
          }
        }
      }
      first = window.stop;
      result.sp_windows.push_back(window);
    }
    // Adjacent windows for one atom share a contiguous active list, so a
    // JVP can batch across those windows without moving the retained tape.
    if (plan.active_sp_groups)
      for (std::size_t atom = 0; atom < shape.atoms; ++atom)
        for (auto& window : result.sp_windows) {
          auto& span = window.active[atom];
          span.offset = active_offset;
          active_offset += span.count;
        }
    plan.work.workspace_bytes = plan.bytes;
    plan.work.groups = groups;
    return cudaSuccess;
  }
  std::size_t batch_radials{};
  if (single_component) {
    for (const auto& group : result.groups) batch_radials = std::max(batch_radials, group.radials);
  }
  if (batch_radials && batch_radials <= options.radial_tile &&
      std::all_of(result.tiles.begin(), result.tiles.end(), [](const Tile& tile) {
        return tile.device.count == 1 && tile.program->integral_batch;
      })) {
    // A fully contracted s-shell group needs only one consumer of its local
    // output. Windows share scratch and keep the original group order.
    for (auto width = std::min<std::size_t>(4096, groups); width >= 2; width /= 2) {
      Layout trial = plan;
      if (!trial.reserve(width * batch_radials * 13, sizeof(double), trial.batch_records) ||
          !trial.reserve(width * 19, sizeof(double), trial.batch_raw) ||
          !trial.reserve(width * 18, sizeof(double), trial.batch_projected) ||
          !trial.reserve(width, sizeof(aot::FactorizedIntegralBatch), trial.batch_inputs) ||
          !trial.reserve(shape.atoms + 1, sizeof(std::uint32_t), trial.atom_offsets) ||
          !trial.reserve(3 * groups, sizeof(std::uint32_t), trial.atom_incidence) ||
          trial.bytes > options.max_workspace_bytes)
        continue;
      trial.batch_width = width;
      trial.batch_radials = batch_radials;
      plan = trial;
      break;
    }
  }
  if (plan.batch_width && options.retain_directional_source) {
    Layout trial = plan;
    if (reserve_directional_tape(trial, groups, shape.atoms) &&
        trial.bytes <= options.max_workspace_bytes)
      plan = trial;
  }
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
__global__ void compare_direction(const double* direction, const double* retained,
                                  std::size_t count, int* mismatch) {
  for (std::size_t index = blockIdx.x * blockDim.x + threadIdx.x; index < count;
       index += static_cast<std::size_t>(blockDim.x) * gridDim.x)
    if (__double_as_longlong(direction[index]) != __double_as_longlong(retained[index]))
      atomicExch(mismatch, 1);
}
__global__ void identify_unit_direction(const double* direction, std::size_t count, int* audit) {
  for (std::size_t index = blockIdx.x * blockDim.x + threadIdx.x; index < count;
       index += static_cast<std::size_t>(blockDim.x) * gridDim.x) {
    const auto value = direction[index];
    if (value == 0) continue;
    if (value != 1 || atomicCAS(audit, 0, static_cast<int>(index) + 1) != 0)
      atomicExch(audit + 1, 1);
  }
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
      if (!expansion.count || expansion.count > expansion_bound(shell.angular, auxiliary)) {
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
__device__ void fill_records(const DeviceGroup* group, Basis basis, std::size_t begin,
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
__global__ void build_records(const DeviceGroup* group, Basis basis, std::size_t begin,
                              std::size_t count, double* records, int* status) {
  fill_records(group, basis, begin, count, records, status);
}
__global__ void build_batch(const DeviceGroup* groups, Basis basis, std::size_t stride,
                            double* records, double* raw, double* discarded,
                            aot::FactorizedIntegralBatch* batches, long long atom, int* status) {
  const auto slot = blockIdx.x;
  const auto* group = groups + slot;
  std::size_t count = 1;
  for (unsigned center = 0; center < group->centers; ++center)
    count *= group->shells[center].primitive_count;
  records += slot * stride;
  const bool active = includes_atom(*group, atom);
  if (threadIdx.x == 0)
    batches[slot] = {records,
                     active ? count : 0,
                     {group->atoms[0], group->atoms[1], group->atoms[2], 0},
                     (active ? raw : discarded) + slot * 19,
                     nullptr};
  // An empty radial chunk emits zeros into window scratch, not into the
  // stationary tape. The batched DF leaf needs no four-center selector.
  if (active) fill_records(group, basis, 0, count, records, status);
}
__global__ void accumulate(const DeviceGroup* group, const DeviceTile* tile, const double* raw,
                           double* cache, int* status) {
  if (atomicAdd(status, 0)) return;
  const auto fields = 1 + 6 * group->centers;
  for (unsigned index = threadIdx.x; index < tile->count * fields; index += blockDim.x)
    cache[index % fields * group->components + tile->components[index / fields]] += raw[index];
}

constexpr std::size_t sp_raw_stride = 16 * 19;

__global__ void build_sp_batch(const DeviceGroup* groups, const std::uint32_t* active_groups,
                               Basis basis, std::size_t stride, double* records, double* raw,
                               aot::FactorizedIntegralBatch* batches, long long atom, int* status) {
  const auto slot = blockIdx.x;
  const auto* group = groups + (active_groups ? active_groups[slot] : slot);
  std::size_t count = 1;
  for (unsigned center = 0; center < group->centers; ++center)
    count *= group->shells[center].primitive_count;
  records += slot * stride;
  const bool active = includes_atom(*group, atom);
  if (threadIdx.x == 0)
    batches[slot] = {records,
                     active ? count : 0,
                     {group->atoms[0], group->atoms[1], group->atoms[2], 0},
                     raw + slot * sp_raw_stride,
                     nullptr};
  if (active) fill_records(group, basis, 0, count, records, status);
}

/** Each disjoint component tile carries a full contracted primitive sum.
 * Inactive groups keep their primal and first derivative from the stationary
 * pass; the projection never reads their stale directional field. */
__global__ void store_sp_batch(const DeviceGroup* groups, const std::uint32_t* active_groups,
                               const DeviceTile* tile, const double* raw, double* tape,
                               long long atom, int* status) {
  const auto& group = groups[active_groups ? active_groups[blockIdx.x] : blockIdx.x];
  if (*status || !includes_atom(group, atom)) return;
  const auto fields = 1 + 6 * group.centers;
  const auto* source = raw + blockIdx.x * sp_raw_stride;
  auto* target = tape + group.tape_offset;
  for (unsigned index = threadIdx.x; index < tile->count * fields; index += blockDim.x)
    target[(index % fields) * group.components + tile->components[index / fields]] = source[index];
}

template <bool Contract, bool SpBatch = false, bool FixedWeights = false>
__global__ void project(CudaDfNuclearShape shape, const DeviceGroup* group, Basis basis,
                        const double* direction, CudaDfNuclearWeights weights, const double* cache,
                        double* candidate, double* projected, int* status, long long atom = -1,
                        bool direction_only = false, const std::uint32_t* active_groups = nullptr,
                        std::uint32_t projected_origin = 0) {
  constexpr auto full_warp = 0xffffffffu;
  if constexpr (SpBatch) {
    // A warp owns the group in either mode. This sticky flag only permits
    // early exit; the final stream gate, not this read, controls publication.
    const int failed = threadIdx.x == 0
                           ? cuda::atomic_ref<int, cuda::thread_scope_device>(*status).load(
                                 cuda::memory_order_relaxed)
                           : 0;
    if (__shfl_sync(full_warp, failed, 0)) return;
  } else if (atomicAdd(status, 0)) {
    return;
  }
  const auto group_index = active_groups ? active_groups[blockIdx.y] : blockIdx.y;
  group += group_index;
  cache += SpBatch ? group->tape_offset : 19 * blockIdx.y;
  if constexpr (Contract) projected += 18 * (group_index - projected_origin);
  // At most 3×3×3 components have to cross lanes. Keep the per-component
  // VJPs in the producing warp and preserve the serial component fold order.
  __shared__ double component_weights[SpBatch && Contract ? 27 * 18 : 1];
  auto* components = SpBatch && Contract ? component_weights : projected;
  const auto centers = group->centers, coordinates = 3 * centers;
  const bool moving = includes_atom(*group, atom);
  if constexpr (FixedWeights) {
    // With fixed weights an inactive integral contributes no curvature.
    // Write explicit zeroes for its incidences in the ordered scatter.
    if (!moving) {
      if (threadIdx.x < 2 * coordinates) projected[threadIdx.x] = 0.0;
      return;
    }
  }
  // Distinct shells in an s/p batch own both AO (or auxiliary) incidences.
  // Equal-shell groups retain their full Cartesian component tile.
  const bool mirror = SpBatch && group->shells[0].ao_begin != group->shells[1].ao_begin;
  // Inactive groups have an exactly zero tangent. The caller has cleared
  // those slots; there is no stationary panel to reconstruct in JVP mode.
  if constexpr (!Contract)
    if (direction_only && !moving) return;
  for (std::size_t index = blockIdx.x * blockDim.x + threadIdx.x; index < group->public_components;
       index += static_cast<std::size_t>(blockDim.x) * gridDim.x) {
    const Expansion* expansions[3]{};
    std::size_t aos[3]{};
    auto remaining = index;
    for (unsigned slot = centers; slot-- > 0;) {
      const auto shell = group->shells[slot];
      aos[slot] = shell.ao_begin + remaining % shell.ao_count;
      remaining /= shell.ao_count;
      const auto auxiliary = group->metric || slot == 2;
      expansions[slot] =
          (auxiliary ? basis.auxiliary_expansions : basis.orbital_expansions) + aos[slot];
    }
    const auto output_index = group->metric ? aos[0] * shape.naux + aos[1]
                                            : (aos[0] * shape.nao + aos[1]) * shape.naux + aos[2];
    const auto transpose = group->metric ? aos[1] * shape.naux + aos[0]
                                         : (aos[1] * shape.nao + aos[0]) * shape.naux + aos[2];
    double weight{}, delta_weight{};
    if constexpr (Contract) {
      const auto* stationary = group->metric ? weights.metric : weights.three_center;
      const auto* tangent =
          group->metric ? weights.direction_metric : weights.direction_three_center;
      weight = stationary[output_index];
      if constexpr (!FixedWeights) delta_weight = tangent[output_index];
      if (mirror) {
        weight += stationary[transpose];
        if constexpr (!FixedWeights) delta_weight += tangent[transpose];
      }
    }
    double value{}, motion{}, gradient[9]{}, hvp[9]{};
    const auto accumulate = [&](unsigned first, unsigned second, unsigned third) {
      const unsigned terms[3]{first, second, third};
      std::size_t component{};
      double projection = 1;
      for (unsigned slot = 0; slot < centers; ++slot) {
        const auto absolute = expansions[slot]->cartesian[terms[slot]];
        const auto auxiliary = group->metric || slot == 2;
        const auto cartesian_base = auxiliary ? shape.cartesian_nao : 0;
        const auto shell = group->shells[slot];
        const auto extent = (shell.angular + 1) * (shell.angular + 2) / 2;
        if constexpr (SpBatch) {
          // The canonical s/p transforms are scaled permutations. Check the
          // borrowed map before indexing the compact component tape.
          if (expansions[slot]->count != 1 || absolute < 0 ||
              static_cast<std::size_t>(absolute) < shell.cartesian_begin ||
              static_cast<std::size_t>(absolute) - shell.cartesian_begin >= extent) {
            atomicExch(status, 1);
            return;
          }
        }
        component = component * extent + absolute - shell.cartesian_begin;
        projection *= expansions[slot]->coefficients[terms[slot]] *
                      basis.component_coefficients[cartesian_base + absolute];
      }
      for (unsigned coordinate = 0; coordinate < coordinates; ++coordinate) {
        double result[4];
        if (!ad::contract(
                cache[component], cache[(1 + coordinate) * group->components + component],
                moving ? cache[(1 + coordinates + coordinate) * group->components + component]
                       : 0.0,
                weight, delta_weight, projection,
                direction[3 * group->atoms[coordinate / 3] + coordinate % 3], result))
          atomicExch(status, 1);
        if (!coordinate) value += result[0];
        motion += result[1];
        if constexpr (Contract) {
          if constexpr (!FixedWeights) gradient[coordinate] += result[2];
          hvp[coordinate] += result[3];
        }
      }
    };
    if constexpr (SpBatch) {
      accumulate(0, 0, 0);
    } else {
      for (unsigned first = 0; first < expansions[0]->count; ++first)
        for (unsigned second = 0; second < expansions[1]->count; ++second)
          for (unsigned third = 0; third < (centers == 3 ? expansions[2]->count : 1); ++third)
            accumulate(first, second, third);
    }
    if constexpr (Contract) {
      for (unsigned coordinate = 0; coordinate < coordinates; ++coordinate) {
        components[index * coordinates + coordinate] = gradient[coordinate];
        components[(group->public_components + index) * coordinates + coordinate] = hvp[coordinate];
      }
    } else {
      const auto raw_count = shape.nao * shape.nao * shape.naux,
                 metric_count = shape.naux * shape.naux;
      const auto offset = group->metric ? raw_count : 0;
      if (!direction_only) candidate[offset + output_index] = value;
      candidate[raw_count + metric_count + offset + output_index] = motion;
      if (mirror) {
        if (!direction_only) candidate[offset + transpose] = value;
        candidate[raw_count + metric_count + offset + transpose] = motion;
      }
    }
  }
  if constexpr (SpBatch && Contract) {
    __syncwarp(full_warp);
    const auto field = threadIdx.x;
    if (field < 2 * coordinates) {
      const auto offset =
          (field / coordinates) * group->public_components * coordinates + field % coordinates;
      double sum{};
      for (std::size_t component = 0; component < group->public_components; ++component)
        sum += components[offset + component * coordinates];
      projected[field] = sum;
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
__global__ void scatter_batch(std::size_t atoms, std::uint32_t begin, std::uint32_t end,
                              const std::uint32_t* offsets, const std::uint32_t* incidence,
                              const double* projected, double* candidate, const int* status) {
  if (*status) return;
  const auto lane = threadIdx.x % 32;
  for (std::size_t coordinate = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
       coordinate < 3 * atoms;
       coordinate += static_cast<std::size_t>(blockDim.x) * gridDim.x / 32) {
    const auto atom = coordinate / 3, axis = coordinate % 3;
    std::uint32_t first{}, last{};
    const auto lower = [&](std::uint32_t left, std::uint32_t right, std::uint32_t group) {
      while (left < right) {
        const auto middle = left + (right - left) / 2;
        if (incidence[middle] / 8 < group)
          left = middle + 1;
        else
          right = middle;
      }
      return left;
    };
    if (!lane) {
      first = lower(offsets[atom], offsets[atom + 1], begin);
      last = lower(first, offsets[atom + 1], end);
    }
    first = __shfl_sync(0xffffffffU, first, 0);
    last = __shfl_sync(0xffffffffU, last, 0);
    if (first == last) continue;
    double gradient{}, hvp{};
    // Some groups contribute thousands of incidences to the same atom. A
    // warp folds each coordinate in a fixed FP64 tree instead of making one
    // lane carry the longest bin. The metric bit avoids a group-table gather.
    for (auto entry = first + lane; entry < last; entry += 32) {
      const auto packed = incidence[entry], index = packed / 8, center = packed % 4;
      const auto* terms = projected + (index - begin) * 18 + 3 * center + axis;
      gradient += terms[0];
      hvp += terms[packed & 4U ? 6 : 9];
    }
    for (unsigned stride = 16; stride; stride /= 2) {
      const auto other_gradient = __shfl_down_sync(0xffffffffU, gradient, stride);
      const auto other_hvp = __shfl_down_sync(0xffffffffU, hvp, stride);
      if (lane < stride && lane + stride < last - first) {
        gradient += other_gradient;
        hvp += other_hvp;
      }
    }
    if (!lane) {
      candidate[coordinate] += gradient;
      candidate[3 * atoms + coordinate] += hvp;
    }
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
  void* tape_arena{};
  Layout tape_layout{};
  bool ready{};
  bool retained_ready{};
  bool stationary_ready{};
  long long retained_atom{-1};
  template <class Value>
  Value* at(std::size_t offset) const {
    return reinterpret_cast<Value*>(static_cast<char*>(arena) + offset);
  }
  template <class Value>
  Value* tape_at(std::size_t offset) const {
    return reinterpret_cast<Value*>(static_cast<char*>(tape_arena ? tape_arena : arena) + offset);
  }
  bool disjoint_workspaces(const void* pointer, std::size_t bytes) const {
    return disjoint(arena, schedule.layout.bytes, pointer, bytes) &&
           (!tape_arena || disjoint(tape_arena, tape_layout.bytes, pointer, bytes));
  }
  bool plan_tape(Layout& layout) const {
    return schedule.layout.batch_width && !schedule.layout.retained_raw && !tape_arena &&
           !receipt.attempted_passes &&
           reserve_directional_tape(layout, schedule.groups.size(), shape.atoms);
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
    for (const auto& input : inputs) {
      if (!input.bytes) continue;
      if (!disjoint_workspaces(input.pointer, input.bytes) ||
          reinterpret_cast<std::uintptr_t>(input.pointer) % input.alignment)
        return cudaErrorInvalidValue;
      if (const auto error = device_buffer(input.pointer, reference.basis.device_id);
          error != cudaSuccess)
        return error;
    }
    for (std::size_t index = 0; index < outputs.size(); ++index) {
      const auto& output = outputs[index];
      if (!disjoint_workspaces(output.pointer, output.bytes) ||
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
  template <bool Contract, bool FixedWeights = false>
  cudaError_t execute(const double* direction, CudaDfNuclearWeights weights,
                      std::span<const Range> outputs, cudaStream_t stream,
                      bool direction_only = false) {
    // A tape is a single-use capability. Even a failed or mismatched action
    // retires it; the original stream orders the reader behind the writer.
    const bool check_retained = Contract && retained_ready;
    retained_ready = false;
    const auto& plan = schedule.layout;
    if constexpr (FixedWeights)
      if (!plan.batch_sp) return cudaErrorNotSupported;
    const auto& tape = tape_arena ? tape_layout : plan;
    const bool retain = !Contract && plan.batch_width && (tape_arena || plan.retained_raw);
    const bool check_unit = retain && stationary_ready && shape.atoms <= INT_MAX / 3;
    if constexpr (!Contract) stationary_ready = false;
    if (receipt.attempted_passes == UINT64_MAX) return cudaErrorInvalidValue;
    ++receipt.attempted_passes;
    auto error = authenticate(stream);
    if (error != cudaSuccess) return error;
    auto basis = basis_ranges(reference.basis);
    std::array<Range, 15> inputs{};
    std::copy(basis.begin(), basis.end(), inputs.begin());
    inputs[10] = {direction, 3 * shape.atoms * sizeof(double)};
    if constexpr (Contract) {
      inputs[11] = {weights.three_center, plan.raw_elements * sizeof(double)};
      inputs[12] = {weights.metric, plan.metric_elements * sizeof(double)};
      if constexpr (!FixedWeights) {
        inputs[13] = {weights.direction_three_center, plan.raw_elements * sizeof(double)};
        inputs[14] = {weights.direction_metric, plan.metric_elements * sizeof(double)};
      }
    }
    if ((error = buffers(inputs, outputs)) != cudaSuccess) return error;
    if (reinterpret_cast<std::uintptr_t>(direction) % alignof(double)) return cudaErrorInvalidValue;
    if (receipt.radial_records > UINT64_MAX - plan.work.radial_records_per_pass ||
        receipt.component_evaluations > UINT64_MAX - plan.work.component_evaluations_per_pass ||
        receipt.projection_terms_bound > UINT64_MAX - plan.work.projection_terms_bound_per_pass ||
        receipt.reduction_terms_bound > UINT64_MAX - plan.work.reduction_terms_bound_per_pass)
      return cudaErrorInvalidValue;
    auto* status = at<int>(plan.status);
    int direction_audit[2]{1, 0};
    error = cudaMemsetAsync(status, 0, sizeof(int), stream);
    if (error == cudaSuccess) {
      finite<<<blocks(3 * shape.atoms), 128, 0, stream>>>(direction, 3 * shape.atoms, status);
      error = cudaGetLastError();
    }
    if constexpr (Contract) {
      for (unsigned index = 11; index < (FixedWeights ? 13 : 15) && error == cudaSuccess; ++index) {
        if (reinterpret_cast<std::uintptr_t>(inputs[index].pointer) % alignof(double))
          return drain(cudaErrorInvalidValue, stream);
        error = symmetric_panel(static_cast<const double*>(inputs[index].pointer), index % 2 == 0);
      }
    }
    if (check_retained && error == cudaSuccess) {
      auto* flag = tape_at<int>(tape.direction_mismatch);
      error = cudaMemsetAsync(flag, 0, sizeof(int), stream);
      if (error == cudaSuccess) {
        compare_direction<<<blocks(3 * shape.atoms), 128, 0, stream>>>(
            direction, tape_at<double>(tape.retained_direction), 3 * shape.atoms, flag);
        error = cudaGetLastError();
      }
      if (error == cudaSuccess)
        error = cudaMemcpyAsync(direction_audit, flag, sizeof(int), cudaMemcpyDeviceToHost, stream);
    }
    if (check_unit && error == cudaSuccess) {
      auto* audit = tape_at<int>(tape.direction_mismatch);
      error = cudaMemsetAsync(audit, 0, sizeof(direction_audit), stream);
      if (error == cudaSuccess) {
        identify_unit_direction<<<blocks(3 * shape.atoms), 128, 0, stream>>>(
            direction, 3 * shape.atoms, audit);
        error = cudaGetLastError();
      }
      if (error == cudaSuccess)
        error = cudaMemcpyAsync(direction_audit, audit, sizeof(direction_audit),
                                cudaMemcpyDeviceToHost, stream);
    }
    if (error == cudaSuccess) error = gate(status, stream);
    if (error != cudaSuccess) return drain(error, stream);
    const bool reuse = check_retained && direction_audit[0] == 0;
    const auto atom = reuse ? retained_atom
                      : check_unit && direction_audit[0] && !direction_audit[1]
                          ? static_cast<long long>((direction_audit[0] - 1) / 3)
                          : -1;
    auto* candidate = at<double>(plan.candidates);
    auto* records = at<double>(plan.records);
    auto* raw = at<double>(plan.raw);
    auto* cache = at<double>(plan.cache);
    auto* projected = at<double>(plan.projected);
    const auto* groups = at<DeviceGroup>(plan.groups);
    const auto* tiles = at<DeviceTile>(plan.tiles);
    const auto first_output = FixedWeights     ? 3 * shape.atoms
                              : direction_only ? plan.raw_elements + plan.metric_elements
                                               : 0;
    const auto clear_begin = direction_only && plan.batch_sp ? first_output : 0;
    error = cudaMemsetAsync(candidate + clear_begin, 0,
                            (plan.candidate_count - clear_begin) * sizeof(double), stream);
    if (plan.batch_sp) {
      auto* batches = at<aot::FactorizedIntegralBatch>(plan.batch_inputs);
      auto* raw_batch = at<double>(plan.batch_raw);
      auto* tape = at<double>(plan.retained_raw);
      auto* compact = at<double>(plan.batch_projected);
      for (std::size_t position = 0; position < schedule.sp_windows.size();) {
        if (error != cudaSuccess) break;
        const auto& window = schedule.sp_windows[position++];
        const auto first = window.first, stop = window.stop;
        const auto count = stop - first;
        const auto* task = groups + first;
        const auto& descriptor = schedule.descriptors[first];
        const auto closure = schedule.groups[first].closure;
        auto radials = atom < 0 ? window.radials : window.atom_radials[atom];
        const auto indexed = atom >= 0 && plan.active_sp_groups;
        auto active_count = indexed ? window.active[atom].count : count;
        const auto* active_groups =
            indexed ? at<std::uint32_t>(plan.active_sp_groups) + window.active[atom].offset
                    : nullptr;
        auto projections = window.projections;
        auto projection_reductions = window.projection_reductions;
        if constexpr (!Contract) {
          // Forward slots have unique owners and no inter-group reduction.
          // Batch same-closure groups across windows while the existing raw
          // arena still holds the complete batch. The transpose uses its old
          // window boundaries and ordered scatter, independent of this choice.
          while (indexed && direction_only && position < schedule.sp_windows.size()) {
            const auto& next = schedule.sp_windows[position];
            const auto additional = next.active[atom].count;
            if (schedule.groups[next.first].closure != closure ||
                additional > plan.batch_width - active_count)
              break;
            active_count += additional;
            radials += next.atom_radials[atom];
            projections += next.projections;
            projection_reductions += next.projection_reductions;
            ++position;
          }
        }
        const auto* producer_groups = indexed ? groups : task;
        if (!reuse && radials) {
          build_sp_batch<<<active_count, 32, 0, stream>>>(
              producer_groups, active_groups, reference.basis, plan.batch_radials * 13,
              at<double>(plan.batch_records), raw_batch, batches, atom, status);
          error = cudaGetLastError();
          for (const auto index : schedule.closure[closure]) {
            if (error != cudaSuccess) break;
            const auto& tile = schedule.tiles[index];
            error = static_cast<cudaError_t>(tile.program->integral_batch(
                options.range_omega, batches, active_count, direction, -1, status, stream));
            if (error != cudaSuccess) break;
            receipt.radial_records += radials;
            receipt.component_evaluations += radials * tile.device.count;
            receipt.reduction_terms_bound +=
                (radials + active_count) * tile.device.count * (1 + 6 * descriptor.centers);
            store_sp_batch<<<active_count, 128, 0, stream>>>(
                producer_groups, active_groups, tiles + index, raw_batch, tape, atom, status);
            error = cudaGetLastError();
          }
        }
        if (error != cudaSuccess) break;
        // Only a moving-weight cotangent needs stationary first derivatives
        // from inactive groups. Keep fixed-weight slots in the original
        // scatter order and zero the inactive slots without launching a warp
        // for every group. Non-unit directions retain the all-group path.
        const auto* project_groups = direction_only || FixedWeights ? active_groups : nullptr;
        const auto project_count = project_groups ? active_count : count;
        if (!project_count) continue;
        if constexpr (FixedWeights) {
          if (project_groups && project_count != count) {
            error = cudaMemsetAsync(compact, 0, count * 18 * sizeof(double), stream);
            if (error != cudaSuccess) break;
          }
        }
        project<Contract, true, FixedWeights><<<dim3(1, project_count), 32, 0, stream>>>(
            shape, project_groups ? groups : task, reference.basis, direction, weights, tape,
            candidate, compact, status, atom, direction_only, project_groups,
            project_groups ? static_cast<std::uint32_t>(first) : 0);
        error = cudaGetLastError();
        if (error != cudaSuccess) break;
        receipt.projection_terms_bound += projections;
        receipt.reduction_terms_bound += projection_reductions;
        if constexpr (Contract) {
          scatter_batch<<<blocks(32 * 3 * shape.atoms), 128, 0, stream>>>(
              shape.atoms, first, stop, at<std::uint32_t>(plan.atom_offsets),
              at<std::uint32_t>(plan.atom_incidence), compact, candidate, status);
          error = cudaGetLastError();
        }
      }
    } else if (plan.batch_width) {
      auto* batches = at<aot::FactorizedIntegralBatch>(plan.batch_inputs);
      auto* projected_batch = at<double>(plan.batch_projected);
      for (std::size_t first = 0; first < schedule.groups.size() && error == cudaSuccess;) {
        const auto closure = schedule.groups[first].closure;
        auto stop = first + 1;
        const auto records_for = [&](std::size_t index) {
          return includes_atom(schedule.descriptors[index], atom) ? schedule.groups[index].radials
                                                                  : std::size_t{};
        };
        auto radials = records_for(first);
        while (stop < schedule.groups.size() && stop - first < plan.batch_width &&
               schedule.groups[stop].closure == closure) {
          radials += records_for(stop++);
        }
        const auto count = stop - first;
        const auto& program = *schedule.tiles[schedule.closure[closure].front()].program;
        // Batched groups share a single all-s output component. Charge only
        // submitted windows; an enqueue failure never counts the unseen tail.
        const auto coordinates = 3 * schedule.descriptors[first].centers;
        const auto fields = 1 + 2 * coordinates;
        auto* raw_batch = retain || reuse ? tape_at<double>(tape.retained_raw) + first * 19
                                          : at<double>(plan.batch_raw);
        if (!reuse && radials) {
          build_batch<<<count, 32, 0, stream>>>(groups + first, reference.basis,
                                                plan.batch_radials * 13,
                                                at<double>(plan.batch_records), raw_batch,
                                                at<double>(plan.batch_raw), batches, atom, status);
          error = cudaGetLastError();
          if (error == cudaSuccess) {
            receipt.radial_records += radials;
            error = static_cast<cudaError_t>(program.integral_batch(
                options.range_omega, batches, count, direction, -1, status, stream));
            if (error == cudaSuccess) {
              receipt.component_evaluations += radials;
              receipt.reduction_terms_bound += (radials + count) * fields;
            }
          }
        }
        if (error == cudaSuccess) {
          project<Contract><<<dim3(1, count), 32, 0, stream>>>(
              shape, groups + first, reference.basis, direction, weights, raw_batch, candidate,
              projected_batch, status, atom);
          error = cudaGetLastError();
          if (error == cudaSuccess) {
            receipt.projection_terms_bound += count * coordinates;
            receipt.reduction_terms_bound += count * (1 + 3 * coordinates);
          }
        }
        if constexpr (Contract) {
          if (error == cudaSuccess) {
            scatter_batch<<<blocks(32 * 3 * shape.atoms), 128, 0, stream>>>(
                shape.atoms, first, stop, at<std::uint32_t>(plan.atom_offsets),
                at<std::uint32_t>(plan.atom_incidence), projected_batch, candidate, status);
            error = cudaGetLastError();
            if (error == cudaSuccess) receipt.reduction_terms_bound += 4 * count * coordinates;
          }
        }
        first = stop;
      }
    } else
      for (std::size_t index = 0; index < schedule.groups.size() && error == cudaSuccess; ++index) {
        const auto& group = schedule.groups[index];
        const auto& device = schedule.descriptors[index];
        error = cudaMemsetAsync(
            cache, 0, device.components * (1 + 6 * device.centers) * sizeof(double), stream);
        for (const auto tile_index : schedule.closure[group.closure]) {
          const auto& tile = schedule.tiles[tile_index];
          for (std::size_t begin = 0; begin < group.radials && error == cudaSuccess;) {
            const auto count = std::min(options.radial_tile, group.radials - begin);
            build_records<<<1, 128, 0, stream>>>(groups + index, reference.basis, begin, count,
                                                 records, status);
            error = cudaGetLastError();
            if (error == cudaSuccess) {
              receipt.radial_records += count;
              error = aot::launch_integral_at_omega(*tile.program, options.range_omega, records,
                                                    count, groups[index].atoms, direction, raw,
                                                    status, stream);
            }
            if (error == cudaSuccess) {
              receipt.component_evaluations += count * tile.device.count;
              receipt.reduction_terms_bound += count * tile.device.count * (1 + 6 * device.centers);
              accumulate<<<1, 128, 0, stream>>>(groups + index, tiles + tile_index, raw, cache,
                                                status);
              error = cudaGetLastError();
              if (error == cudaSuccess)
                receipt.reduction_terms_bound += tile.device.count * (1 + 6 * device.centers);
            }
            begin += count;
          }
        }
        if (error == cudaSuccess) {
          project<Contract><<<blocks(device.public_components), 128, 0, stream>>>(
              shape, groups + index, reference.basis, direction, weights, cache, candidate,
              projected, status);
          error = cudaGetLastError();
          receipt.projection_terms_bound +=
              device.public_components * group.projection_bound * 3 * device.centers;
          receipt.reduction_terms_bound +=
              device.public_components * group.projection_bound * (1 + 9 * device.centers);
        }
        if constexpr (Contract)
          if (error == cudaSuccess) {
            scatter<<<1, 32, 0, stream>>>(groups + index, shape.atoms, projected, candidate,
                                          status);
            error = cudaGetLastError();
            receipt.reduction_terms_bound += 6 * device.centers * (device.public_components + 1);
          }
      }
    if (error == cudaSuccess) {
      if constexpr (Contract) {
        finite<<<blocks(6 * shape.atoms), 128, 0, stream>>>(candidate, 6 * shape.atoms, status);
        error = cudaGetLastError();
      } else {
        if (!direction_only) {
          error = symmetric_panel(candidate, false);
          if (error == cudaSuccess) error = symmetric_panel(candidate + plan.raw_elements, true);
        }
        if (error == cudaSuccess)
          error = symmetric_panel(candidate + plan.raw_elements + plan.metric_elements, false);
        if (error == cudaSuccess)
          error = symmetric_panel(candidate + 2 * plan.raw_elements + plan.metric_elements, true);
      }
    }
    if (error == cudaSuccess) error = gate(status, stream);
    if (error != cudaSuccess) return drain(error, stream);
    if ((error = authenticate(stream)) != cudaSuccess) return error;
    if (retain) {
      error = cudaMemcpyAsync(tape_at<double>(tape.retained_direction), direction, inputs[10].bytes,
                              cudaMemcpyDeviceToDevice, stream);
      if (error != cudaSuccess) return drain(error, stream);
    }
    auto offset = first_output;
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
      else {
        ++receipt.value_passes;
        retained_ready = retain;
        stationary_ready = retain;
        retained_atom = atom;
      }
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
    auto admitted = options;
    admitted.max_workspace_bytes = std::min(options.max_workspace_bytes, workspace_bytes);
    auto error = schedule(shape, orbital, auxiliary, admitted, state->schedule);
    if (error != cudaSuccess) return error;
    const auto& basis = reference.basis;
    auto& plan = state->schedule.layout;
    if (!basis || !reference.stream || !reference.authenticate || !reference.owner || !workspace ||
        reinterpret_cast<std::uintptr_t>(workspace) % alignof(double) ||
        workspace_bytes < plan.bytes || basis.range_omega != options.range_omega ||
        basis.atoms != shape.atoms || basis.primitives != shape.primitives ||
        basis.nao != shape.nao || basis.naux != shape.naux ||
        basis.cartesian_nao != shape.cartesian_nao ||
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
    std::vector<std::uint32_t> offsets, incidence;
    if (plan.batch_width) {
      try {
        offsets.resize(shape.atoms + 1);
        for (const auto& group : state->schedule.descriptors)
          for (unsigned center = 0; center < group.centers; ++center)
            ++offsets[group.atoms[center] + 1];
        for (std::size_t atom = 0; atom < shape.atoms; ++atom) offsets[atom + 1] += offsets[atom];
        incidence.resize(offsets.back());
        auto cursor = offsets;
        for (std::uint32_t index = 0; index < state->schedule.descriptors.size(); ++index) {
          const auto& group = state->schedule.descriptors[index];
          for (unsigned center = 0; center < group.centers; ++center)
            incidence[cursor[group.atoms[center]]++] = index * 8 + (group.metric ? 4 : 0) + center;
        }
      } catch (const std::bad_alloc&) {
        if (plan.batch_sp) return cudaErrorMemoryAllocation;
        plan.batch_width = 0;
      }
    }
    std::vector<std::uint32_t> active_groups;
    if (plan.active_sp_groups) {
      try {
        const auto& last = state->schedule.sp_windows.back().active[shape.atoms - 1];
        active_groups.resize(last.offset + last.count);
        for (const auto& window : state->schedule.sp_windows) {
          auto cursor = window.active;
          for (auto index = window.first; index < window.stop; ++index) {
            const auto& group = state->schedule.descriptors[index];
            for (unsigned center = 0; center < group.centers; ++center) {
              const auto atom = group.atoms[center];
              if (std::find(group.atoms, group.atoms + center, atom) == group.atoms + center)
                active_groups[cursor[atom].offset++] = index;
            }
          }
        }
      } catch (const std::bad_alloc&) {
        plan.active_sp_groups = 0;
      }
    }
    auto* status = state->at<int>(plan.status);
    auto* shells = state->at<Shell>(plan.shells);
    error = cudaMemsetAsync(status, 0, sizeof(int), reference.stream);
    if (error == cudaSuccess)
      error = cudaMemcpyAsync(shells, orbital.data(), orbital.size_bytes(), cudaMemcpyHostToDevice,
                              reference.stream);
    if (error == cudaSuccess)
      error = cudaMemcpyAsync(shells + orbital.size(), auxiliary.data(), auxiliary.size_bytes(),
                              cudaMemcpyHostToDevice, reference.stream);
    if (error == cudaSuccess)
      error =
          cudaMemcpyAsync(state->at<DeviceGroup>(plan.groups), state->schedule.descriptors.data(),
                          state->schedule.descriptors.size() * sizeof(DeviceGroup),
                          cudaMemcpyHostToDevice, reference.stream);
    if (plan.batch_width && error == cudaSuccess)
      error = cudaMemcpyAsync(state->at<std::uint32_t>(plan.atom_offsets), offsets.data(),
                              offsets.size() * sizeof(std::uint32_t), cudaMemcpyHostToDevice,
                              reference.stream);
    if (plan.batch_width && error == cudaSuccess)
      error = cudaMemcpyAsync(state->at<std::uint32_t>(plan.atom_incidence), incidence.data(),
                              incidence.size() * sizeof(std::uint32_t), cudaMemcpyHostToDevice,
                              reference.stream);
    if (plan.active_sp_groups && error == cudaSuccess)
      error = cudaMemcpyAsync(state->at<std::uint32_t>(plan.active_sp_groups), active_groups.data(),
                              active_groups.size() * sizeof(std::uint32_t), cudaMemcpyHostToDevice,
                              reference.stream);
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
std::size_t CudaDfNuclearResponse::directional_source_workspace() const noexcept {
  Layout layout;
  return impl_ && impl_->plan_tape(layout) ? layout.bytes : 0;
}
cudaError_t CudaDfNuclearResponse::bind_directional_source(void* workspace,
                                                           std::size_t bytes) noexcept {
  if (!impl_) return cudaErrorInvalidResourceHandle;
  auto& state = *impl_;
  Layout layout;
  if (!state.plan_tape(layout) || bytes < layout.bytes || !workspace ||
      reinterpret_cast<std::uintptr_t>(workspace) % alignof(double) ||
      !state.disjoint_workspaces(workspace, layout.bytes) ||
      state.receipt.workspace_bytes > SIZE_MAX - layout.bytes)
    return cudaErrorInvalidValue;
  const auto& reference = state.reference;
  if (auto error = state.authenticate(reference.stream); error != cudaSuccess) return error;
  if (auto error = device_buffer(workspace, reference.basis.device_id); error != cudaSuccess)
    return error;
  for (const auto& input : basis_ranges(reference.basis))
    if (input.bytes && !disjoint(workspace, layout.bytes, input.pointer, input.bytes))
      return cudaErrorInvalidValue;
  state.tape_layout = layout;
  state.tape_arena = workspace;
  state.receipt.workspace_bytes += layout.bytes;
  return cudaSuccess;
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
  if (!output.three_center && !output.metric)
    return impl_->execute<false>(direction, {}, std::span(ranges).subspan(2), stream, true);
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
cudaError_t CudaDfNuclearResponse::contract_fixed_weights(const double* direction,
                                                          const double* three_center,
                                                          const double* metric, double* hvp,
                                                          cudaStream_t stream) noexcept {
  if (!impl_) return cudaErrorInvalidResourceHandle;
  const Range range{hvp, 3 * impl_->shape.atoms * sizeof(double)};
  return impl_->execute<true, true>(direction, {three_center, metric}, {&range, 1}, stream);
}
const CudaDfNuclearWork& CudaDfNuclearResponse::work() const noexcept {
  static const CudaDfNuclearWork empty;
  return impl_ ? impl_->receipt : empty;
}
}  // namespace generativeqc::integrals
