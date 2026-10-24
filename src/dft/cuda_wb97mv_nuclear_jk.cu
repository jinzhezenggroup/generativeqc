#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <map>
#include <vector>

#include "dft/cuda_wb97mv_nuclear_jk.hpp"
#include "dft/factorized_hessian_integral.hpp"
#include "dft/hessian_aot.hpp"
#include "response/nuclear_integral_response.cuh"
#include "response/wb97mv_nuclear_jk.cuh"

namespace generativeqc::dft {
namespace {
namespace aot = hessian_aot;
namespace maps = generativeqc_wb97mv_nuclear_jk_ad;
using Shell = integrals::CudaOneElectronShell;
using Quartet = std::array<std::size_t, 4>;
constexpr std::array<std::array<unsigned, 4>, 8> permutations{{{{0, 1, 2, 3}},
                                                               {{1, 0, 2, 3}},
                                                               {{0, 1, 3, 2}},
                                                               {{1, 0, 3, 2}},
                                                               {{2, 3, 0, 1}},
                                                               {{3, 2, 0, 1}},
                                                               {{2, 3, 1, 0}},
                                                               {{3, 2, 1, 0}}}};

struct DeviceGroup {
  Shell shells[4];
  long long atoms[4];
  std::size_t components{}, quartets{};
  maps::Kernel kernel;
  std::size_t cache_strides[4]{};
  unsigned cache_slots[4]{0, 1, 2, 3};
  std::size_t resident_offset{}, records_offset{}, projected_offset{};
  bool source{};
};
struct DeviceTile {
  unsigned components[16]{};
  unsigned count{};
};
struct Tile {
  const aot::Program* program{};
  DeviceTile device;
};
struct Group {
  DeviceGroup device;
  std::size_t radials{}, closure{}, resident_offset{}, resident_source{};
};
std::size_t projection_bound(const DeviceGroup& group) {
  auto terms = group.quartets;
  for (const auto& shell : group.shells)
    if (shell.angular > 1 && shell.ao_count != (shell.angular + 1) * (shell.angular + 2) / 2)
      terms *= 3;
  return terms;
}
struct Layout {
  std::size_t bytes{}, matrix{}, panels{}, basis_elements{}, groups{}, tiles{}, shells{}, records{},
      raw{}, cache{}, projected{}, candidates{}, status{}, max_components{}, max_quartets{},
      radial_records{}, candidate_count{}, resident{}, resident_direction{}, resident_bytes{};
  std::uint64_t components{}, projections{}, reductions{};
  std::size_t batch_records{}, batch_raw{}, batch_inputs{}, batch_reductions{}, batch_projected{},
      reduction_offsets{}, reduction_entries{}, batch_bytes{};
};
struct RawBatch {
  std::size_t group{}, tile{}, begin{}, count{}, record{}, raw{};
};
struct BatchReduction {
  std::size_t group{}, tile{}, first{}, count{};
};
struct BatchRun {
  std::size_t tile{}, first{}, count{};
};
struct ProjectionRun {
  std::size_t first{}, count{}, offset{};
};
struct ReductionEntry {
  std::size_t group{};
  unsigned incidence{}, first{}, second{};
};
struct Schedule {
  Layout plan;
  std::vector<Group> groups;
  std::vector<Tile> tiles;
  std::array<std::vector<std::size_t>, 1280> closure;
  std::vector<RawBatch> batches;
  std::vector<BatchReduction> batch_reductions;
  std::vector<BatchRun> batch_runs;
  std::vector<ProjectionRun> projection_runs;
  std::vector<std::size_t> reduction_offsets;
  std::vector<ReductionEntry> reduction_entries;
  std::vector<Schedule> windows;
  std::vector<DeviceGroup> device_groups;
  std::vector<DeviceTile> device_tiles;
  std::vector<aot::FactorizedIntegralBatch> inputs;
  std::size_t ordered_groups{}, source_groups{};
  const char* refusal{"shape, topology, or options"};
};

bool multiply(std::size_t first, std::size_t second, std::size_t& result) {
  if (second && first > SIZE_MAX / second) return false;
  result = first * second;
  return true;
}
bool add(std::size_t& value, std::size_t term) {
  if (term > SIZE_MAX - value) return false;
  value += term;
  return true;
}
bool append(Layout& plan, std::size_t count, std::size_t width, std::size_t& offset) {
  std::size_t bytes{};
  if (!multiply(count, width, bytes) || bytes > SIZE_MAX - 255 ||
      plan.bytes > SIZE_MAX - bytes - 255)
    return false;
  offset = plan.bytes;
  plan.bytes = (plan.bytes + bytes + 255) / 256 * 256;
  return true;
}
__host__ __device__ unsigned cartesian_count(unsigned angular) {
  return (angular + 1) * (angular + 2) / 2;
}
unsigned blocks(std::size_t count) { return std::min<std::size_t>((count - 1) / 128 + 1, 65535); }

void batch_schedule(const CudaWb97mvJkShape& shape, const CudaWb97mvJkOptions& options,
                    Schedule& schedule) {
  auto& plan = schedule.plan;
  if (!plan.resident_bytes || !options.max_batch_workspace_bytes ||
      !options.max_batch_projection_bytes)
    return;
  for (const auto& tile : schedule.tiles)
    if (!tile.program->integral_batch) return;
  auto candidate = plan;
  std::size_t records{}, raw{}, inputs{}, reductions{}, projected{}, entries{}, count{},
      maximum_projected{};
  constexpr unsigned pairs[4][2] = {{0, 1}, {2, 3}, {0, 2}, {1, 3}};
  for (std::size_t index = 0; index < schedule.groups.size(); ++index) {
    auto& group = schedule.groups[index];
    group.device.resident_offset = group.resident_offset;
    group.device.projected_offset = projected;
    group.device.source = group.resident_source == index;
    if (!multiply(group.device.quartets, 41, count) || !add(projected, count) || !add(entries, 1))
      return;
    maximum_projected = std::max(maximum_projected, count);
    for (const auto& pair : pairs) {
      if (!multiply(group.device.shells[pair[0]].ao_count, group.device.shells[pair[1]].ao_count,
                    count) ||
          !multiply(count, shape.spins, count) || !add(entries, count))
        return;
    }
    for (unsigned center = 0; center < 4; ++center) {
      bool first = true;
      for (unsigned previous = 0; previous < center; ++previous)
        first = first && group.device.atoms[previous] != group.device.atoms[center];
      if (first && !add(entries, 3)) return;
    }
    if (!group.device.source) continue;
    group.device.records_offset = records;
    const auto chunks = (group.radials - 1) / options.radial_tile + 1;
    if (!multiply(group.radials, 5, count) || !add(records, count) ||
        !multiply(group.device.components, 25, count) || !multiply(count, chunks, count) ||
        !add(raw, count) || !add(reductions, schedule.closure[group.closure].size()) ||
        !multiply(chunks, schedule.closure[group.closure].size(), count) || !add(inputs, count))
      return;
  }
  // Admit the entire optional layout before constructing its host metadata.
  // No partial batch replaces the already admitted ordered streaming schedule.
  if (!append(candidate, records, sizeof(double), candidate.batch_records) ||
      !append(candidate, raw, sizeof(double), candidate.batch_raw) ||
      !append(candidate, inputs, sizeof(aot::FactorizedIntegralBatch), candidate.batch_inputs) ||
      !append(candidate, reductions, sizeof(BatchReduction), candidate.batch_reductions) ||
      !append(candidate, 2 + plan.panels + 3 * shape.atoms, sizeof(std::size_t),
              candidate.reduction_offsets) ||
      !append(candidate, entries, sizeof(ReductionEntry), candidate.reduction_entries) ||
      candidate.bytes > options.max_workspace_bytes ||
      candidate.bytes - plan.bytes > options.max_batch_workspace_bytes)
    return;
  const auto available =
      std::min(options.max_workspace_bytes - candidate.bytes,
               options.max_batch_workspace_bytes - (candidate.bytes - plan.bytes));
  const auto projection_values =
      std::min({projected, options.max_batch_projection_bytes / sizeof(double),
                available / 256 * 256 / sizeof(double)});
  if (projection_values < maximum_projected ||
      !append(candidate, projection_values, sizeof(double), candidate.batch_projected))
    return;
  // Projection and publication reduction consume one ordered group interval at
  // a time. Retaining a full public quartet tensor must not be a prerequisite
  // for batching the much smaller immutable integral source.
  for (std::size_t first = 0; first < schedule.groups.size();) {
    const auto offset = schedule.groups[first].device.projected_offset;
    auto end = first;
    std::size_t used{};
    while (end < schedule.groups.size()) {
      const auto values = schedule.groups[end].device.quartets * 41;
      if (values > projection_values - used) break;
      used += values;
      ++end;
    }
    schedule.projection_runs.push_back({first, end - first, offset});
    first = end;
  }
  std::vector<std::vector<RawBatch>> tiles(schedule.tiles.size());
  raw = 0;
  for (std::size_t index = 0; index < schedule.groups.size(); ++index) {
    const auto& group = schedule.groups[index];
    if (!group.device.source) continue;
    for (const auto tile : schedule.closure[group.closure])
      for (std::size_t begin = 0; begin < group.radials; begin += options.radial_tile) {
        tiles[tile].push_back({index, tile, begin,
                               std::min(options.radial_tile, group.radials - begin),
                               group.device.records_offset + 5 * begin, raw});
        raw += schedule.tiles[tile].device.count * 25;
      }
  }
  for (std::size_t tile = 0; tile < tiles.size(); ++tile) {
    const auto first = schedule.batches.size();
    for (const auto& batch : tiles[tile]) {
      if (schedule.batches.size() == first || schedule.batches.back().group != batch.group)
        schedule.batch_reductions.push_back({batch.group, tile, schedule.batches.size(), 0});
      schedule.batches.push_back(batch);
      ++schedule.batch_reductions.back().count;
    }
    for (std::size_t begin = first; begin < schedule.batches.size(); begin += 65535)
      schedule.batch_runs.push_back(
          {tile, begin, std::min<std::size_t>(65535, schedule.batches.size() - begin)});
  }
  // Primal and directional outputs have identical incidence lists. Share the
  // sparse metadata, not their numerical reductions or summation order.
  std::vector<std::vector<ReductionEntry>> owners(1 + plan.panels + 3 * shape.atoms);
  const auto vector_begin = 1 + plan.panels;
  for (std::size_t index = 0; index < schedule.groups.size(); ++index) {
    const auto& group = schedule.groups[index].device;
    owners[0].push_back({index});
    for (unsigned incidence = 0; incidence < 4; ++incidence) {
      const auto first = pairs[incidence][0], second = pairs[incidence][1];
      for (unsigned spin = 0; spin < shape.spins; ++spin)
        for (unsigned row = 0; row < group.shells[first].ao_count; ++row)
          for (unsigned column = 0; column < group.shells[second].ao_count; ++column) {
            const auto target = 1 + spin * plan.matrix +
                                (group.shells[first].ao_begin + row) * shape.nao +
                                group.shells[second].ao_begin + column;
            owners[target].push_back({index, incidence, row, column});
          }
    }
    for (unsigned center = 0; center < 4; ++center) {
      bool first = true;
      for (unsigned previous = 0; previous < center; ++previous)
        first = first && group.atoms[previous] != group.atoms[center];
      if (!first) continue;
      for (unsigned axis = 0; axis < 3; ++axis) {
        const auto target = vector_begin + 3 * group.atoms[center] + axis;
        owners[target].push_back({index, center, axis});
      }
    }
  }
  for (const auto& owner : owners) {
    schedule.reduction_offsets.push_back(schedule.reduction_entries.size());
    schedule.reduction_entries.insert(schedule.reduction_entries.end(), owner.begin(), owner.end());
  }
  schedule.reduction_offsets.push_back(schedule.reduction_entries.size());
  candidate.batch_bytes = candidate.bytes - plan.bytes;
  plan = candidate;
}

cudaError_t ordered_schedule(const CudaWb97mvJkShape& shape, std::span<const Shell> shells,
                             const CudaWb97mvJkOptions& options, Schedule& result,
                             std::span<const Quartet> selection = {}) {
  auto& plan = result.plan;
  if (!shape.atoms || !shape.primitives || !shape.nao || shells.empty() ||
      (shape.spins != 1 && shape.spins != 2) || !options.radial_tile || options.radial_tile > 64 ||
      shape.atoms > static_cast<std::size_t>(std::numeric_limits<long long>::max()) / 3 ||
      !std::isfinite(options.symmetry_absolute_tolerance) ||
      options.symmetry_absolute_tolerance < 0 ||
      !std::isfinite(options.symmetry_relative_tolerance) ||
      options.symmetry_relative_tolerance < 0 || !multiply(shape.nao, shape.nao, plan.matrix) ||
      !multiply(plan.matrix, shape.spins, plan.panels) ||
      !multiply(shape.atoms, 3, plan.basis_elements))
    return cudaErrorInvalidValue;
  std::size_t count{};
  if (!multiply(shape.primitives, 2, count) || !add(plan.basis_elements, count) ||
      !multiply(shape.nao, 16, count) || !add(plan.basis_elements, count) ||
      plan.basis_elements > SIZE_MAX / sizeof(double))
    return cudaErrorInvalidValue;
  std::size_t primitive_begin{}, ao_begin{};
  for (const auto& shell : shells) {
    if (shell.atom >= shape.atoms || shell.angular > 3 || !shell.primitive_count ||
        shell.primitive_begin != primitive_begin || shell.ao_begin != ao_begin ||
        (shell.ao_count != cartesian_count(shell.angular) &&
         shell.ao_count != 2 * shell.angular + 1) ||
        !add(primitive_begin, shell.primitive_count) || !add(ao_begin, shell.ao_count) ||
        primitive_begin > shape.primitives || ao_begin > shape.nao)
      return cudaErrorInvalidValue;
  }
  if (primitive_begin != shape.primitives || ao_begin != shape.nao) return cudaErrorInvalidValue;
  result.refusal = "ordered group bound";
  std::vector<maps::Kernel> kernels;
  switch (options.part) {
    case CudaWb97mvJkPart::exact:
      kernels = {maps::Kernel::k_primary, maps::Kernel::k_correction};
      break;
    case CudaWb97mvJkPart::primary:
      kernels = {maps::Kernel::k_primary};
      break;
    case CudaWb97mvJkPart::correction:
      kernels = {maps::Kernel::k_correction};
      break;
    case CudaWb97mvJkPart::coulomb:
      kernels = {maps::Kernel::k_coulomb};
      break;
    case CudaWb97mvJkPart::short_range:
      kernels = {maps::Kernel::k_short};
      break;
    case CudaWb97mvJkPart::long_range:
      kernels = {maps::Kernel::k_long};
      break;
    default:
      return cudaErrorInvalidValue;
  }
  count = kernels.size();
  if (selection.empty()) {
    for (unsigned slot = 0; slot < 4; ++slot)
      if (!multiply(count, shells.size(), count)) return cudaErrorInvalidValue;
  } else if (!multiply(count, selection.size(), count)) {
    return cudaErrorInvalidValue;
  }
  if (count > options.max_groups) return cudaErrorInvalidValue;
  result.groups.reserve(count);
  // Complete ordered closure, not symmetry reconstruction. Each package is
  // scanned once per selected angular tuple and method part, never per radial.
  const auto quartets = count / kernels.size();
  for (std::size_t index = 0; index < quartets; ++index)
    for (const auto kernel : kernels) {
      Quartet indices{};
      if (selection.empty()) {
        auto remaining = index;
        for (unsigned slot = 4; slot-- > 0;) {
          indices[slot] = remaining % shells.size();
          remaining /= shells.size();
        }
      } else {
        indices = selection[index];
      }
      Group group{{{shells[indices[0]], shells[indices[1]], shells[indices[2]], shells[indices[3]]},
                   {},
                   1,
                   1,
                   kernel},
                  1,
                  0};
      const auto selected = static_cast<std::size_t>(kernel);
      auto key = selected;
      for (unsigned slot = 0; slot < 4; ++slot) {
        const auto& shell = group.device.shells[slot];
        group.device.atoms[slot] = shell.atom;
        key = key * 4 + shell.angular;
        if (!multiply(group.radials, shell.primitive_count, group.radials) ||
            !multiply(group.device.components, cartesian_count(shell.angular),
                      group.device.components) ||
            !multiply(group.device.quartets, shell.ao_count, group.device.quartets))
          return cudaErrorInvalidValue;
      }
      group.closure = key;
      auto& closure = result.closure[key];
      if (closure.empty()) {
        std::array<bool, 10000> covered{};
        for (const auto& program : aot::programs()) {
          const auto& signature = program.signature;
          if (program.kind != aot::Kind::integral || signature.family != "eri" ||
              signature.angular.size() != 4 || signature.radial != maps::radial[selected])
            continue;
          bool matches = true;
          for (unsigned slot = 0; slot < 4; ++slot)
            matches &= signature.angular[slot] == group.device.shells[slot].angular;
          if (!matches) continue;
          if (signature.omega != maps::omega[selected] || !program.integral ||
              program.centers != 4 || program.record_stride != 17 || program.outputs != 25 ||
              program.components.empty() || program.components.size() > 16)
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
      const auto chunks = (group.radials - 1) / options.radial_tile + 1;
      result.refusal = "primitive, projection, or reduction work bound";
      if (!multiply(group.radials, group.device.components, count) ||
          !add(plan.components, count) || !add(plan.projections, projection_bound(group.device)) ||
          !multiply(group.device.quartets, 25 + 8 * shape.spins, count) ||
          !add(plan.reductions, count) || !multiply(group.device.components, 25, count) ||
          !multiply(count, chunks, count) || !add(plan.reductions, count) ||
          !multiply(group.radials, closure.size(), count) || !add(plan.radial_records, count) ||
          (selection.empty() && plan.components > options.max_primitive_component_evaluations) ||
          plan.projections > options.max_projection_terms ||
          plan.reductions > options.max_reduction_terms)
        return cudaErrorInvalidValue;
      plan.max_components = std::max(plan.max_components, group.device.components);
      plan.max_quartets = std::max(plan.max_quartets, group.device.quartets);
      std::size_t stride = 1;
      for (unsigned slot = 4; slot-- > 0;) {
        group.device.cache_strides[slot] = stride;
        stride *= cartesian_count(group.device.shells[slot].angular);
      }
      result.groups.push_back(group);
    }
  plan.candidate_count = 1;
  result.refusal = "minimum ordered arena bound";
  if (!multiply(plan.panels, 2, count) || !add(plan.candidate_count, count) ||
      !multiply(shape.atoms, 6, count) || !add(plan.candidate_count, count) ||
      !append(plan, result.groups.size(), sizeof(DeviceGroup), plan.groups) ||
      !append(plan, result.tiles.size(), sizeof(DeviceTile), plan.tiles) ||
      !append(plan, shells.size(), sizeof(Shell), plan.shells) ||
      !append(plan, options.radial_tile * 17, sizeof(double), plan.records) ||
      !append(plan, 16 * 25, sizeof(double), plan.raw) ||
      !append(plan, plan.max_components * 25, sizeof(double), plan.cache) ||
      !append(plan, plan.max_quartets * 41, sizeof(double), plan.projected) ||
      !append(plan, plan.candidate_count, sizeof(double), plan.candidates) ||
      !append(plan, 2, sizeof(int), plan.status) || plan.bytes > options.max_workspace_bytes)
    return cudaErrorInvalidValue;
  // Optional residency never changes minimum admission or the streaming bound.
  // Jets contain a right-direction HVP, so a basis identity alone is not a key.
  if (!options.max_resident_integral_bytes) return cudaSuccess;
  auto resident_plan = plan;
  std::size_t fields{};
  bool representable = true;
  std::map<std::array<std::size_t, 5>, std::size_t> sources;
  for (std::size_t index = 0; index < result.groups.size(); ++index) {
    auto& group = result.groups[index];
    std::array<std::size_t, 5> key;
    for (unsigned slot = 0; slot < 4; ++slot) key[slot] = group.device.shells[slot].primitive_begin;
    key[4] = static_cast<std::size_t>(group.device.kernel);
    auto selected = permutations.front();
    for (const auto& permutation : permutations) {
      auto candidate = key;
      for (unsigned slot = 0; slot < 4; ++slot)
        candidate[slot] = group.device.shells[permutation[slot]].primitive_begin;
      if (candidate < key) {
        key = candidate;
        selected = permutation;
      }
    }
    const auto [position, inserted] = sources.try_emplace(key, index);
    group.resident_source = position->second;
    if (inserted) {
      group.resident_offset = fields;
      representable =
          representable && multiply(group.device.components, 25, count) && add(fields, count);
    } else {
      group.resident_offset = result.groups[group.resident_source].resident_offset;
    }
    // Canonical shell order is lexicographically first, hence already built on
    // the same stream. Permute Cartesian strides AND both derivative field legs.
    std::size_t stride = 1;
    for (unsigned slot = 4; slot-- > 0;) {
      const auto original = selected[slot];
      group.device.cache_slots[original] = slot;
      group.device.cache_strides[original] = stride;
      stride *= cartesian_count(group.device.shells[original].angular);
    }
  }
  if (representable && options.max_resident_integral_bytes &&
      append(resident_plan, fields, sizeof(double), resident_plan.resident) &&
      append(resident_plan, 3 * shape.atoms, sizeof(double), resident_plan.resident_direction) &&
      resident_plan.bytes <= options.max_workspace_bytes &&
      resident_plan.bytes - plan.bytes <= options.max_resident_integral_bytes) {
    resident_plan.resident_bytes = resident_plan.bytes - plan.bytes;
    plan = resident_plan;
    plan.components = plan.radial_records = plan.reductions = 0;
    for (std::size_t index = 0; index < result.groups.size(); ++index) {
      const auto& group = result.groups[index];
      plan.reductions += group.device.quartets * (25 + 8 * shape.spins);
      if (group.resident_source != index) continue;
      plan.components += group.radials * group.device.components;
      plan.radial_records += group.radials * result.closure[group.closure].size();
      plan.reductions +=
          25 * group.device.components * ((group.radials - 1) / options.radial_tile + 1);
    }
  } else {
    for (auto& group : result.groups) {
      std::size_t stride = 1;
      for (unsigned slot = 4; slot-- > 0;) {
        group.device.cache_slots[slot] = slot;
        group.device.cache_strides[slot] = stride;
        stride *= cartesian_count(group.device.shells[slot].angular);
      }
    }
  }
  batch_schedule(shape, options, result);
  for (std::size_t index = 0; index < result.groups.size(); ++index)
    if (!plan.resident_bytes || result.groups[index].resident_source == index)
      ++result.source_groups;
  return cudaSuccess;
}

cudaError_t schedule(const CudaWb97mvJkShape& shape, std::span<const Shell> shells,
                     const CudaWb97mvJkOptions& options, Schedule& result) {
  const auto original = ordered_schedule(shape, shells, options, result);
  if (original != cudaErrorInvalidValue || !options.source_tile || !options.max_source_groups ||
      !options.max_batch_workspace_bytes || !options.max_batch_projection_bytes)
    return original;
  Layout shared;
  shared.candidate_count = 1;
  std::size_t panels{}, vectors{};
  if (!multiply(result.plan.panels, 2, panels) || !multiply(shape.atoms, 6, vectors) ||
      !add(shared.candidate_count, panels) || !add(shared.candidate_count, vectors) ||
      !append(shared, shared.candidate_count, sizeof(double), shared.candidates) ||
      !append(shared, 2, sizeof(int), shared.status) || shared.bytes >= options.max_workspace_bytes)
    return original;
  result = {};
  result.refusal = "canonical source group bound";
  std::size_t pairs{}, sources{};
  if (!multiply(shells.size(), shells.size() + 1, pairs)) return original;
  pairs /= 2;
  if (!multiply(pairs, pairs + 1, sources)) return original;
  sources /= 2;
  const auto kernels = options.part == CudaWb97mvJkPart::exact ? 2 : 1;
  if (!multiply(sources, kernels, sources) || sources > options.max_source_groups) return original;
  std::vector<Quartet> pending;
  auto tile_options = options;
  tile_options.max_workspace_bytes -= shared.bytes;
  tile_options.max_resident_integral_bytes = tile_options.max_workspace_bytes;
  const auto flush = [&](auto&& self, std::span<const Quartet> canonical) -> cudaError_t {
    if (canonical.empty()) return cudaSuccess;
    std::vector<Quartet> selected;
    for (const auto& source : canonical) {
      std::array<Quartet, 8> orbit;
      for (unsigned index = 0; index < permutations.size(); ++index)
        for (unsigned slot = 0; slot < 4; ++slot)
          orbit[index][slot] = source[permutations[index][slot]];
      std::sort(orbit.begin(), orbit.end());
      selected.insert(selected.end(), orbit.begin(), std::unique(orbit.begin(), orbit.end()));
    }
    std::sort(selected.begin(), selected.end());
    Schedule window;
    auto error = ordered_schedule(shape, shells, tile_options, window, selected);
    result.refusal = window.refusal;
    // A tile must reuse each canonical source. Never turn a capacity fallback
    // into eightfold primitive replay or hundreds of thousands of launches.
    if (error == cudaSuccess && (!window.plan.resident_bytes || !window.plan.batch_bytes)) {
      result.refusal = "canonical tile resident/batch arena admission";
      error = cudaErrorInvalidValue;
    }
    if (error == cudaErrorInvalidValue && canonical.size() > 1) {
      const auto half = canonical.size() / 2;
      error = self(self, canonical.first(half));
      return error == cudaSuccess ? self(self, canonical.subspan(half)) : error;
    }
    if (error != cudaSuccess) return error;
    auto& plan = result.plan;
    result.refusal = "aggregate canonical primitive, projection, or reduction work bound";
    if (!add(plan.components, window.plan.components) ||
        !add(plan.projections, window.plan.projections) ||
        !add(plan.reductions, window.plan.reductions) ||
        !add(plan.radial_records, window.plan.radial_records) ||
        plan.components > options.max_primitive_component_evaluations ||
        plan.projections > options.max_projection_terms ||
        plan.reductions > options.max_reduction_terms)
      return cudaErrorInvalidValue;
    plan.bytes = std::max(plan.bytes, window.plan.bytes);
    plan.batch_bytes = std::max(plan.batch_bytes, window.plan.batch_bytes);
    result.ordered_groups += window.groups.size();
    result.windows.push_back(std::move(window));
    return cudaSuccess;
  };
  for (std::size_t first = 0; first < shells.size(); ++first)
    for (std::size_t second = first; second < shells.size(); ++second)
      for (std::size_t third = first; third < shells.size(); ++third)
        for (std::size_t fourth = third == first ? second : third; fourth < shells.size();
             ++fourth) {
          pending.push_back({first, second, third, fourth});
          if (pending.size() == options.source_tile) {
            const auto error = flush(flush, pending);
            if (error != cudaSuccess) return error;
            pending.clear();
          }
        }
  const auto error = flush(flush, pending);
  if (error != cudaSuccess) return error;
  if (result.windows.empty()) return cudaErrorInvalidValue;
  auto& plan = result.plan;
  const auto first = result.windows.front().plan;
  plan.matrix = first.matrix;
  plan.panels = first.panels;
  plan.basis_elements = first.basis_elements;
  plan.candidate_count = first.candidate_count;
  result.refusal = "aggregate tiled arena bound";
  if (!append(plan, plan.candidate_count, sizeof(double), plan.candidates) ||
      !append(plan, 2, sizeof(int), plan.status) || plan.bytes > options.max_workspace_bytes)
    return cudaErrorInvalidValue;
  for (auto& window : result.windows) {
    window.plan.candidates = plan.candidates;
    window.plan.status = plan.status;
  }
  result.source_groups = sources;
  return cudaSuccess;
}

bool disjoint(const void* first, std::size_t first_bytes, const void* second,
              std::size_t second_bytes) {
  if (!first || !second) return false;
  const auto left = reinterpret_cast<std::uintptr_t>(first),
             right = reinterpret_cast<std::uintptr_t>(second);
  return left <= right ? right - left >= first_bytes : left - right >= second_bytes;
}
cudaError_t device_buffer(const void* pointer, int device) {
  if (!pointer || reinterpret_cast<std::uintptr_t>(pointer) % alignof(double))
    return cudaErrorInvalidValue;
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
cudaError_t gate(int* status, cudaStream_t stream, bool* same_direction = nullptr) {
  std::array<int, 2> flags{};
  auto error = cudaMemcpyAsync(flags.data(), status, same_direction ? sizeof(flags) : sizeof(int),
                               cudaMemcpyDeviceToHost, stream);
  error = drain(error, stream);
  if (same_direction) *same_direction = flags[1] == 0;
  return error == cudaSuccess && flags[0] ? cudaErrorInvalidValue : error;
}
__global__ void compare_direction(const double* current, const double* saved, std::size_t count,
                                  int* different) {
  for (std::size_t index = threadIdx.x; index < count; index += blockDim.x)
    if (current[index] != saved[index]) atomicExch(different, 1);
}
__global__ void finite_values(const double* values, std::size_t count, int* status) {
  for (std::size_t index = blockIdx.x * blockDim.x + threadIdx.x; index < count;
       index += static_cast<std::size_t>(blockDim.x) * gridDim.x)
    if (!isfinite(values[index])) atomicExch(status, 1);
}
__global__ void symmetric(const double* values, std::size_t nao, std::size_t spins, double absolute,
                          double relative, int* status) {
  for (std::size_t index = blockIdx.x * blockDim.x + threadIdx.x; index < spins * nao * nao;
       index += static_cast<std::size_t>(blockDim.x) * gridDim.x) {
    const auto within = index % (nao * nao), spin = index / (nao * nao);
    const auto value = values[index],
               other = values[spin * nao * nao + (within % nao) * nao + within / nao];
    if (!isfinite(value) || !isfinite(other) ||
        fabs(value - other) > absolute + relative * fmax(fabs(value), fabs(other)))
      atomicExch(status, 1);
  }
}
__global__ void validate_basis(CudaWb97mvJkShape shape, const Shell* shells, std::size_t count,
                               const double* basis, int* status) {
  const auto* primitives = basis + 3 * shape.atoms;
  const auto* aos = primitives + 2 * shape.primitives;
  for (std::size_t index = blockIdx.x * blockDim.x + threadIdx.x;
       index < max(count, shape.primitives);
       index += static_cast<std::size_t>(blockDim.x) * gridDim.x) {
    if (index < shape.primitives && !(primitives[2 * index] > 0)) atomicExch(status, 1);
    if (index >= count) continue;
    const auto shell = shells[index];
    for (std::size_t orbital = 0; orbital < shell.ao_count; ++orbital) {
      const auto* record = aos + 16 * (shell.ao_begin + orbital);
      const auto terms =
          shell.angular > 1 && shell.ao_count != cartesian_count(shell.angular) ? 3 : 1;
      if (record[0] != shell.atom || record[1] != shell.primitive_begin ||
          record[2] != shell.primitive_count || record[3] < 1 || record[3] > terms ||
          floor(record[3]) != record[3]) {
        atomicExch(status, 1);
        continue;
      }
      for (unsigned term = 0; term < static_cast<unsigned>(record[3]); ++term) {
        double degree{};
        for (unsigned axis = 0; axis < 3; ++axis) {
          const auto power = record[4 + 4 * term + axis];
          if (!isfinite(power) || power < 0 || power > 3 || floor(power) != power)
            atomicExch(status, 1);
          degree += power;
        }
        if (degree != shell.angular) atomicExch(status, 1);
      }
    }
  }
}
__device__ void fill_records(CudaWb97mvJkShape shape, const DeviceGroup* group, const double* basis,
                             std::size_t begin, std::size_t count, double* records, int* status,
                             bool compact = false) {
  if (atomicAdd(status, 0)) return;
  const auto* primitives = basis + 3 * shape.atoms;
  for (std::size_t index = threadIdx.x; index < count; index += blockDim.x) {
    auto radial = begin + index;
    auto* record = records + (compact ? 5 : 17) * index;
    double scale = 1;
    for (unsigned slot = 4; slot-- > 0;) {
      const auto shell = group->shells[slot];
      const auto primitive = shell.primitive_begin + radial % shell.primitive_count;
      radial /= shell.primitive_count;
      record[slot] = primitives[2 * primitive];
      scale *= primitives[2 * primitive + 1];
      if (!compact)
        for (unsigned axis = 0; axis < 3; ++axis)
          record[4 + 3 * slot + axis] = basis[3 * shell.atom + axis];
    }
    if (!isfinite(scale)) {
      atomicExch(status, 1);
      scale = 0;
    }
    record[compact ? 4 : 16] = scale;
  }
}
__global__ void build_records(CudaWb97mvJkShape shape, const DeviceGroup* group,
                              const double* basis, std::size_t begin, std::size_t count,
                              double* records, int* status) {
  fill_records(shape, group, basis, begin, count, records, status);
}
__global__ void build_batch_records(CudaWb97mvJkShape shape, const DeviceGroup* groups,
                                    std::size_t count, const double* basis, double* records,
                                    int* status) {
  for (std::size_t index = blockIdx.x; index < count; index += gridDim.x) {
    const auto* group = groups + index;
    if (!group->source) continue;
    std::size_t radials = 1;
    for (const auto& shell : group->shells) radials *= shell.primitive_count;
    fill_records(shape, group, basis, 0, radials, records + group->records_offset, status, true);
  }
}
__global__ void accumulate_batches(const DeviceGroup* groups, const DeviceTile* tiles,
                                   const aot::FactorizedIntegralBatch* batches,
                                   const BatchReduction* reductions, std::size_t count,
                                   double* resident, int* status) {
  if (atomicAdd(status, 0)) return;
  for (std::size_t index = blockIdx.x; index < count; index += gridDim.x) {
    const auto reduction = reductions[index];
    const auto group = groups[reduction.group];
    const auto tile = tiles[reduction.tile];
    for (unsigned field = threadIdx.x; field < tile.count * 25; field += blockDim.x) {
      double value{};
      for (std::size_t chunk = 0; chunk < reduction.count; ++chunk)
        value += batches[reduction.first + chunk].output[field];
      resident[group.resident_offset + (field % 25) * group.components +
               tile.components[field / 25]] = value;
    }
  }
}
__global__ void accumulate(const DeviceGroup* group, const DeviceTile* tile, const double* raw,
                           double* cache, int* status) {
  if (atomicAdd(status, 0)) return;
  for (unsigned index = threadIdx.x; index < tile->count * 25; index += blockDim.x) {
    const auto component = tile->components[index / 25], field = index % 25;
    cache[field * group->components + component] += raw[index];
  }
}
__device__ std::size_t cartesian_index(const double* term, unsigned angular) {
  const auto remainder = angular - static_cast<unsigned>(term[0]);
  return remainder * (remainder + 1) / 2 + static_cast<unsigned>(term[2]);
}
__device__ void quartet_coordinates(std::size_t linear, const DeviceGroup* group,
                                    std::size_t* local) {
  for (unsigned slot = 4; slot-- > 0;) {
    local[slot] = linear % group->shells[slot].ao_count;
    linear /= group->shells[slot].ao_count;
  }
}
__device__ std::size_t quartet_index(const std::size_t* local, const DeviceGroup* group) {
  std::size_t result{};
  for (unsigned slot = 0; slot < 4; ++slot)
    result = result * group->shells[slot].ao_count + local[slot];
  return result;
}
__device__ void project_values(CudaWb97mvJkShape shape, const DeviceGroup* group,
                               const double* basis, const double* direction, const double* density,
                               const double* delta_density, const double* cache, double* projected,
                               int* status, std::size_t first, std::size_t stride) {
  if (atomicAdd(status, 0)) return;
  for (std::size_t quartet = first; quartet < group->quartets; quartet += stride) {
    std::size_t local[4], ao[4];
    quartet_coordinates(quartet, group, local);
    const double* records[4];
    for (unsigned slot = 0; slot < 4; ++slot) {
      ao[slot] = group->shells[slot].ao_begin + local[slot];
      records[slot] = basis + 3 * shape.atoms + 2 * shape.primitives + 16 * ao[slot];
    }
    double integral[25]{};
    // Public AO polynomials gather the cached raw fields. Component factors
    // appear here exactly once, including real-spherical expansion coefficients.
    for (unsigned first = 0; first < static_cast<unsigned>(records[0][3]); ++first)
      for (unsigned second = 0; second < static_cast<unsigned>(records[1][3]); ++second)
        for (unsigned third = 0; third < static_cast<unsigned>(records[2][3]); ++third)
          for (unsigned fourth = 0; fourth < static_cast<unsigned>(records[3][3]); ++fourth) {
            const unsigned terms[] = {first, second, third, fourth};
            std::size_t component{};
            double coefficient = 1;
            for (unsigned slot = 0; slot < 4; ++slot) {
              const auto angular = static_cast<unsigned>(group->shells[slot].angular);
              const auto* term = records[slot] + 4 + 4 * terms[slot];
              component += group->cache_strides[slot] * cartesian_index(term, angular);
              coefficient *= term[3];
            }
            integral[0] += coefficient * cache[component];
            for (unsigned coordinate = 0; coordinate < 12; ++coordinate) {
              const auto source = 3 * group->cache_slots[coordinate / 3] + coordinate % 3;
              integral[1 + coordinate] +=
                  coefficient * cache[(1 + source) * group->components + component];
              integral[13 + coordinate] +=
                  coefficient * cache[(13 + source) * group->components + component];
            }
          }
    double motion{};
    for (unsigned coordinate = 0; coordinate < 12; ++coordinate)
      motion +=
          integral[1 + coordinate] * direction[3 * group->atoms[coordinate / 3] + coordinate % 3];
    maps::Result result;
    if (!maps::evaluate(group->kernel, shape.spins, density, delta_density, shape.nao, ao,
                        integral[0], motion, result))
      atomicExch(status, 1);
    auto* output = projected + 41 * quartet;
    output[0] = result.energy;
    for (unsigned incidence = 0; incidence < 8; ++incidence) {
      output[1 + incidence] = result.fock[incidence];
      output[9 + incidence] = result.motion[incidence];
    }
    for (unsigned coordinate = 0; coordinate < 12; ++coordinate) {
      double contraction[4];
      if (!generativeqc_nuclear_integral_ad::contract(integral[0], integral[1 + coordinate],
                                                      integral[13 + coordinate], result.weight,
                                                      result.delta_weight, 1, 0, contraction))
        atomicExch(status, 1);
      output[17 + coordinate] = contraction[2];
      output[29 + coordinate] = contraction[3];
    }
  }
}
__global__ void project(CudaWb97mvJkShape shape, const DeviceGroup* group, const double* basis,
                        const double* direction, const double* density, const double* delta_density,
                        const double* cache, double* projected, int* status) {
  project_values(shape, group, basis, direction, density, delta_density, cache, projected, status,
                 blockIdx.x * blockDim.x + threadIdx.x,
                 static_cast<std::size_t>(blockDim.x) * gridDim.x);
}
__global__ void project_batches(CudaWb97mvJkShape shape, const DeviceGroup* groups,
                                std::size_t first, std::size_t count, std::size_t offset,
                                const double* basis, const double* direction, const double* density,
                                const double* delta_density, const double* resident,
                                double* projected, int* status) {
  for (std::size_t index = first + blockIdx.x; index < first + count; index += gridDim.x) {
    const auto* group = groups + index;
    project_values(shape, group, basis, direction, density, delta_density,
                   resident + group->resident_offset,
                   projected + (group->projected_offset - offset), status, threadIdx.x, blockDim.x);
  }
}

__device__ std::size_t lower_group(const ReductionEntry* entries, std::size_t begin,
                                   std::size_t end, std::size_t group) {
  while (begin < end) {
    const auto middle = begin + (end - begin) / 2;
    if (entries[middle].group < group)
      begin = middle + 1;
    else
      end = middle;
  }
  return begin;
}

__global__ void reduce_batches(CudaWb97mvJkShape shape, const DeviceGroup* groups,
                               const std::size_t* offsets, const ReductionEntry* entries,
                               const double* projected, std::size_t first_group,
                               std::size_t group_count, std::size_t projected_offset,
                               std::size_t count, double* candidate, int* status) {
  if (atomicAdd(status, 0)) return;
  const auto panels = shape.spins * shape.nao * shape.nao;
  const auto vector_begin = 1 + 2 * panels;
  constexpr unsigned pairs[4][2] = {{0, 1}, {2, 3}, {0, 2}, {1, 3}};
  const auto lane = threadIdx.x % 32;
  for (std::size_t target = (blockIdx.x * blockDim.x + threadIdx.x) / 32; target < count;
       target += static_cast<std::size_t>(gridDim.x) * blockDim.x / 32) {
    double total = candidate[target];
    // One warp owns an output. Lanes evaluate independent group contributions,
    // then lane zero adds them in the original sparse-list order. This exposes
    // parallel reads without a tree reduction or changed floating-point sums.
    const auto owner = !target ? 0
                       : target < vector_begin
                           ? 1 + (target - 1) % panels
                           : 1 + panels + (target - vector_begin) % (3 * shape.atoms);
    const auto first = lower_group(entries, offsets[owner], offsets[owner + 1], first_group);
    const auto end = lower_group(entries, first, offsets[owner + 1], first_group + group_count);
    for (auto begin = first; begin < end; begin += 32) {
      const auto index = begin + lane;
      double value{};
      if (index < end) {
        const auto entry = entries[index];
        const auto* group = groups + entry.group;
        const auto* values = projected + (group->projected_offset - projected_offset);
        if (!target) {
          for (std::size_t quartet = 0; quartet < group->quartets; ++quartet)
            value += values[41 * quartet];
        } else if (target < vector_begin) {
          const auto motion = target > panels;
          const auto spin = ((target - 1) % panels) / (shape.nao * shape.nao);
          const auto first = pairs[entry.incidence][0], second = pairs[entry.incidence][1];
          unsigned rest[2], used{};
          std::size_t local[4]{};
          local[first] = entry.first;
          local[second] = entry.second;
          for (unsigned slot = 0; slot < 4; ++slot)
            if (slot != first && slot != second) rest[used++] = slot;
          for (local[rest[0]] = 0; local[rest[0]] < group->shells[rest[0]].ao_count;
               ++local[rest[0]])
            for (local[rest[1]] = 0; local[rest[1]] < group->shells[rest[1]].ao_count;
                 ++local[rest[1]])
              value += values[41 * quartet_index(local, group) + (motion ? 9 : 1) +
                              2 * entry.incidence + spin];
        } else {
          const auto motion = target >= vector_begin + 3 * shape.atoms;
          for (std::size_t quartet = 0; quartet < group->quartets; ++quartet)
            for (unsigned slot = entry.incidence; slot < 4; ++slot)
              if (group->atoms[slot] == group->atoms[entry.incidence])
                value += values[41 * quartet + (motion ? 29 : 17) + 3 * slot + entry.first];
        }
      }
      const auto active = static_cast<unsigned>(end - begin < 32 ? end - begin : 32);
      for (unsigned source = 0; source < active; ++source) {
        const auto contribution = __shfl_sync(0xffffffffU, value, source);
        if (!lane) total += contribution;
      }
    }
    if (!lane) candidate[target] = total;
  }
}
__device__ void reduce_fock(CudaWb97mvJkShape shape, const DeviceGroup* group, unsigned incidence,
                            const double* projected, double* candidate, int* status) {
  if (atomicAdd(status, 0)) return;
  const unsigned pairs[4][2] = {{0, 1}, {2, 3}, {0, 2}, {1, 3}};
  const auto first = pairs[incidence][0], second = pairs[incidence][1];
  unsigned rest[2], used{};
  for (unsigned slot = 0; slot < 4; ++slot)
    if (slot != first && slot != second) rest[used++] = slot;
  const auto width = group->shells[second].ao_count;
  const auto pairs_count = group->shells[first].ao_count * width;
  for (std::size_t index = blockIdx.x * blockDim.x + threadIdx.x; index < shape.spins * pairs_count;
       index += static_cast<std::size_t>(blockDim.x) * gridDim.x) {
    const auto spin = index / pairs_count, pair = index % pairs_count;
    std::size_t local[4]{};
    local[first] = pair / width;
    local[second] = pair % width;
    double value{}, motion{};
    for (local[rest[0]] = 0; local[rest[0]] < group->shells[rest[0]].ao_count; ++local[rest[0]])
      for (local[rest[1]] = 0; local[rest[1]] < group->shells[rest[1]].ao_count; ++local[rest[1]]) {
        const auto* output = projected + 41 * quartet_index(local, group);
        value += output[1 + 2 * incidence + spin];
        motion += output[9 + 2 * incidence + spin];
      }
    const auto target = spin * shape.nao * shape.nao +
                        (group->shells[first].ao_begin + local[first]) * shape.nao +
                        group->shells[second].ao_begin + local[second];
    candidate[1 + target] += value;
    candidate[1 + shape.spins * shape.nao * shape.nao + target] += motion;
  }
}
__device__ void reduce_vectors(CudaWb97mvJkShape shape, const DeviceGroup* group,
                               const double* projected, double* candidate, int* status) {
  if (atomicAdd(status, 0)) return;
  const auto coordinate = threadIdx.x;
  if (coordinate >= 12) return;
  const auto center = coordinate / 3, axis = coordinate % 3;
  // Only the first mathematical occurrence owns a physical atom-axis. Retain
  // all coincident centers inside its reduction, without floating atomics.
  for (unsigned previous = 0; previous < center; ++previous)
    if (group->atoms[previous] == group->atoms[center]) return;
  double gradient{}, hvp{};
  for (std::size_t quartet = 0; quartet < group->quartets; ++quartet)
    for (unsigned slot = center; slot < 4; ++slot)
      if (group->atoms[slot] == group->atoms[center]) {
        gradient += projected[41 * quartet + 17 + 3 * slot + axis];
        hvp += projected[41 * quartet + 29 + 3 * slot + axis];
      }
  const auto base = 1 + 2 * shape.spins * shape.nao * shape.nao;
  candidate[base + 3 * group->atoms[center] + axis] += gradient;
  candidate[base + 3 * shape.atoms + 3 * group->atoms[center] + axis] += hvp;
}
__device__ void reduce_energy(const DeviceGroup* group, const double* projected, double* candidate,
                              int* status) {
  if (atomicAdd(status, 0)) return;
  double value{};
  for (std::size_t quartet = 0; quartet < group->quartets; ++quartet)
    value += projected[41 * quartet];
  candidate[0] += value;
}
__global__ void reduce_group(CudaWb97mvJkShape shape, const DeviceGroup* group,
                             const double* projected, double* candidate, int* status) {
  // One CTA preserves the old incidence order and its inter-incidence barriers.
  // Energy and nuclear vectors own disjoint outputs; no floating atomics enter.
  for (unsigned incidence = 0; incidence < 4; ++incidence) {
    reduce_fock(shape, group, incidence, projected, candidate, status);
    __syncthreads();
  }
  reduce_vectors(shape, group, projected, candidate, status);
  if (threadIdx.x == 0) reduce_energy(group, projected, candidate, status);
}

void stage_schedule(Schedule& schedule, void* workspace, const double* basis) {
  auto* base = static_cast<char*>(workspace);
  const auto& plan = schedule.plan;
  for (const auto& group : schedule.groups) schedule.device_groups.push_back(group.device);
  for (const auto& tile : schedule.tiles) schedule.device_tiles.push_back(tile.device);
  for (const auto& batch : schedule.batches) {
    aot::FactorizedIntegralBatch input;
    input.records = reinterpret_cast<double*>(base + plan.batch_records) + batch.record;
    input.count = batch.count;
    input.output = reinterpret_cast<double*>(base + plan.batch_raw) + batch.raw;
    input.coordinates = basis;
    std::copy_n(schedule.groups[batch.group].device.atoms, 4, input.atoms);
    schedule.inputs.push_back(input);
  }
}

cudaError_t upload_schedule(const Schedule& schedule, CudaWb97mvJkShape shape, const double* basis,
                            void* workspace, int* status, cudaStream_t stream) {
  auto* base = static_cast<char*>(workspace);
  const auto& plan = schedule.plan;
  auto error = cudaSuccess;
  const auto upload = [&](const auto& values, std::size_t offset) {
    if (error == cudaSuccess && !values.empty())
      error = cudaMemcpyAsync(base + offset, values.data(), values.size() * sizeof(values[0]),
                              cudaMemcpyHostToDevice, stream);
  };
  upload(schedule.device_groups, plan.groups);
  upload(schedule.device_tiles, plan.tiles);
  if (plan.batch_bytes) {
    upload(schedule.inputs, plan.batch_inputs);
    upload(schedule.batch_reductions, plan.batch_reductions);
    upload(schedule.reduction_offsets, plan.reduction_offsets);
    upload(schedule.reduction_entries, plan.reduction_entries);
    if (error == cudaSuccess) {
      build_batch_records<<<std::min<std::size_t>(schedule.groups.size(), 65535), 128, 0, stream>>>(
          shape, reinterpret_cast<const DeviceGroup*>(base + plan.groups), schedule.groups.size(),
          basis, reinterpret_cast<double*>(base + plan.batch_records), status);
      error = cudaGetLastError();
    }
  }
  return error;
}
}  // namespace

struct CudaWb97mvNuclearJk::Impl {
  CudaWb97mvJkShape shape;
  CudaWb97mvJkOptions options;
  CudaWb97mvJkReference reference;
  CudaWb97mvJkWork work;
  Schedule schedule;
  void* workspace{};
  bool ready{}, resident_valid{};
  cudaError_t authenticate(cudaStream_t stream) noexcept {
    if (!ready || stream != reference.stream) return cudaErrorInvalidResourceHandle;
    int device{-1};
    auto error = cudaGetDevice(&device);
    if (error != cudaSuccess) return error;
    if (device != reference.device_id) return cudaErrorInvalidResourceHandle;
    cudaStreamCaptureStatus capture{};
    error = cudaStreamIsCapturing(stream, &capture);
    if (error != cudaSuccess) return error;
    if (capture != cudaStreamCaptureStatusNone) return cudaErrorStreamCaptureUnsupported;
    error = reference.authenticate(reference.owner);
    if (error != cudaSuccess) ready = false;
    return error;
  }
};

cudaError_t cuda_wb97mv_jk_workspace(const CudaWb97mvJkShape& shape, std::span<const Shell> shells,
                                     const CudaWb97mvJkOptions& options, std::size_t& bytes,
                                     std::string* detail) noexcept {
  bytes = 0;
  try {
    Schedule selected;
    const auto error = schedule(shape, shells, options, selected);
    if (detail) {
      if (error == cudaSuccess)
        detail->clear();
      else
        *detail = std::string(selected.refusal) +
                  " (components=" + std::to_string(selected.plan.components) +
                  ", projections=" + std::to_string(selected.plan.projections) +
                  ", reductions=" + std::to_string(selected.plan.reductions) +
                  ", bytes=" + std::to_string(selected.plan.bytes) + ")";
    }
    if (error == cudaSuccess) bytes = selected.plan.bytes;
    return error;
  } catch (...) {
    return cudaErrorMemoryAllocation;
  }
}
CudaWb97mvNuclearJk::CudaWb97mvNuclearJk() : impl_(std::make_unique<Impl>()) {}
CudaWb97mvNuclearJk::~CudaWb97mvNuclearJk() = default;
const CudaWb97mvJkWork& CudaWb97mvNuclearJk::work() const noexcept { return impl_->work; }

cudaError_t CudaWb97mvNuclearJk::prepare(const CudaWb97mvJkShape& shape,
                                         std::span<const Shell> shells,
                                         const CudaWb97mvJkOptions& options,
                                         const CudaWb97mvJkReference& reference, void* workspace,
                                         std::size_t workspace_bytes) noexcept {
  impl_->ready = false;
  impl_->resident_valid = false;
  int device{-1};
  auto error = cudaGetDevice(&device);
  if (error != cudaSuccess) return error;
  if (device != reference.device_id) return cudaErrorInvalidResourceHandle;
  cudaStreamCaptureStatus capture{};
  if ((error = cudaStreamIsCapturing(reference.stream, &capture)) != cudaSuccess) return error;
  if (capture != cudaStreamCaptureStatusNone) return cudaErrorStreamCaptureUnsupported;
  try {
    Schedule selected;
    if ((error = schedule(shape, shells, options, selected)) != cudaSuccess) return error;
    const auto& plan = selected.plan;
    if (workspace_bytes < plan.bytes || !reference.stream || !reference.authenticate ||
        !reference.owner ||
        !disjoint(workspace, plan.bytes, reference.basis, plan.basis_elements * sizeof(double)))
      return cudaErrorInvalidValue;
    for (const auto* pointer : {reference.basis, static_cast<const double*>(workspace)})
      if ((error = device_buffer(pointer, reference.device_id)) != cudaSuccess) return error;
    auto& state = *impl_;
    state.shape = shape;
    state.options = options;
    state.reference = reference;
    state.schedule = std::move(selected);
    state.workspace = workspace;
    state.ready = true;
    if ((error = state.authenticate(reference.stream)) != cudaSuccess) {
      state.ready = false;
      return error;
    }
    if (state.schedule.windows.empty())
      stage_schedule(state.schedule, workspace, reference.basis);
    else
      for (auto& window : state.schedule.windows)
        stage_schedule(window, workspace, reference.basis);
    const auto& layout = state.schedule.plan;
    const auto& initial =
        state.schedule.windows.empty() ? state.schedule : state.schedule.windows.front();
    const auto stream = reference.stream;
    auto* base = static_cast<char*>(workspace);
    auto* status = reinterpret_cast<int*>(base + layout.status);
    auto* device_shells = reinterpret_cast<Shell*>(base + initial.plan.shells);
    error = cudaMemsetAsync(status, 0, sizeof(int), stream);
    if (error == cudaSuccess)
      error = cudaMemcpyAsync(device_shells, shells.data(), shells.size_bytes(),
                              cudaMemcpyHostToDevice, stream);
    if (error == cudaSuccess) {
      finite_values<<<blocks(layout.basis_elements), 128, 0, stream>>>(
          reference.basis, layout.basis_elements, status);
      error = cudaGetLastError();
    }
    // Complete the finite gate before interpreting any packed floating metadata.
    error = error == cudaSuccess ? gate(status, stream) : drain(error, stream);
    if (error == cudaSuccess) {
      validate_basis<<<blocks(std::max(shells.size(), shape.primitives)), 128, 0, stream>>>(
          shape, device_shells, shells.size(), reference.basis, status);
      error = cudaGetLastError();
      error = error == cudaSuccess ? gate(status, stream) : drain(error, stream);
    }
    if (error == cudaSuccess && state.schedule.windows.empty()) {
      error = upload_schedule(state.schedule, shape, reference.basis, workspace, status, stream);
      error = error == cudaSuccess ? gate(status, stream) : drain(error, stream);
    }
    if (error == cudaSuccess) error = state.authenticate(stream);
    state.ready = error == cudaSuccess;
    if (state.ready)
      state.work = {layout.bytes,
                    state.schedule.windows.empty() ? state.schedule.groups.size()
                                                   : state.schedule.ordered_groups,
                    layout.radial_records,
                    layout.components,
                    layout.projections,
                    layout.reductions};
    if (state.ready) state.work.resident_integral_bytes = layout.resident_bytes;
    if (state.ready) state.work.batched_workspace_bytes = layout.batch_bytes;
    if (state.ready) state.work.streaming_tiles = state.schedule.windows.size();
    if (state.ready) state.work.source_groups = state.schedule.source_groups;
    return error;
  } catch (...) {
    impl_->ready = false;
    return cudaErrorMemoryAllocation;
  }
}

cudaError_t CudaWb97mvNuclearJk::apply(const double* direction, const double* density,
                                       const double* delta_density,
                                       const CudaWb97mvJkOutput& output,
                                       cudaStream_t stream) noexcept {
  auto& state = *impl_;
  if (state.work.attempted_actions == UINT64_MAX) return cudaErrorInvalidValue;
  ++state.work.attempted_actions;
  auto error = state.authenticate(stream);
  if (error != cudaSuccess) return error;
  const auto& plan = state.schedule.plan;
  const auto& shape = state.shape;
  const auto& reference = state.reference;
  const double* inputs[] = {reference.basis, direction, density, delta_density};
  const std::size_t input_counts[] = {plan.basis_elements, 3 * shape.atoms, plan.panels,
                                      plan.panels};
  double* outputs[] = {output.energy, output.fock, output.fock_direction, output.gradient,
                       output.hvp};
  const std::size_t output_counts[] = {1, plan.panels, plan.panels, 3 * shape.atoms,
                                       3 * shape.atoms};
  for (unsigned index = 0; index < 4; ++index) {
    if (!disjoint(state.workspace, plan.bytes, inputs[index], input_counts[index] * sizeof(double)))
      return cudaErrorInvalidValue;
    if ((error = device_buffer(inputs[index], reference.device_id)) != cudaSuccess) return error;
  }
  for (unsigned index = 0; index < 5; ++index) {
    if (!disjoint(state.workspace, plan.bytes, outputs[index],
                  output_counts[index] * sizeof(double)))
      return cudaErrorInvalidValue;
    for (unsigned input = 0; input < 4; ++input)
      if (!disjoint(outputs[index], output_counts[index] * sizeof(double), inputs[input],
                    input_counts[input] * sizeof(double)))
        return cudaErrorInvalidValue;
    for (unsigned previous = 0; previous < index; ++previous)
      if (!disjoint(outputs[index], output_counts[index] * sizeof(double), outputs[previous],
                    output_counts[previous] * sizeof(double)))
        return cudaErrorInvalidValue;
    if ((error = device_buffer(outputs[index], reference.device_id)) != cudaSuccess) return error;
  }
  if (state.work.component_evaluations > UINT64_MAX - plan.components ||
      state.work.projection_terms_bound > UINT64_MAX - plan.projections ||
      state.work.reduction_terms > UINT64_MAX - plan.reductions)
    return cudaErrorInvalidValue;
  auto* base = static_cast<char*>(state.workspace);
  auto* status = reinterpret_cast<int*>(base + plan.status);
  auto* candidate = reinterpret_cast<double*>(base + plan.candidates);
  error = cudaMemsetAsync(status, 0, 2 * sizeof(int), stream);
  if (error == cudaSuccess) {
    finite_values<<<blocks(3 * shape.atoms), 128, 0, stream>>>(direction, 3 * shape.atoms, status);
    error = cudaGetLastError();
  }
  for (const auto* panel : {density, delta_density})
    if (error == cudaSuccess) {
      symmetric<<<blocks(plan.panels), 128, 0, stream>>>(
          panel, shape.nao, shape.spins, state.options.symmetry_absolute_tolerance,
          state.options.symmetry_relative_tolerance, status);
      error = cudaGetLastError();
    }
  bool reuse{};
  if (error == cudaSuccess && state.resident_valid) {
    compare_direction<<<1, 128, 0, stream>>>(
        direction, reinterpret_cast<const double*>(base + plan.resident_direction), 3 * shape.atoms,
        status + 1);
    error = cudaGetLastError();
  }
  error = error == cudaSuccess ? gate(status, stream, &reuse) : drain(error, stream);
  if (error != cudaSuccess) return error;
  reuse = reuse && state.resident_valid;
  state.resident_valid = false;
  if (reuse)
    ++state.work.resident_integral_hits;
  else
    ++state.work.integral_source_builds;
  error = cudaMemsetAsync(candidate, 0, plan.candidate_count * sizeof(double), stream);
  const auto run_schedule = [&](const Schedule& selected) {
    const auto& plan = selected.plan;
    auto* records = reinterpret_cast<double*>(base + plan.records);
    auto* raw = reinterpret_cast<double*>(base + plan.raw);
    auto* streaming_cache = reinterpret_cast<double*>(base + plan.cache);
    auto* projected = reinterpret_cast<double*>(base + plan.projected);
    const auto* groups = reinterpret_cast<DeviceGroup*>(base + plan.groups);
    const auto* tiles = reinterpret_cast<DeviceTile*>(base + plan.tiles);
    if (!state.schedule.windows.empty() && error == cudaSuccess)
      error = upload_schedule(selected, shape, reference.basis, state.workspace, status, stream);
    if (plan.batch_bytes && error == cudaSuccess) {
      auto* batches = reinterpret_cast<aot::FactorizedIntegralBatch*>(base + plan.batch_inputs);
      auto* resident = reinterpret_cast<double*>(base + plan.resident);
      if (!reuse) {
        for (const auto& run : selected.batch_runs) {
          const auto& tile = selected.tiles[run.tile];
          error = static_cast<cudaError_t>(tile.program->integral_batch(
              batches + run.first, run.count, direction, status, stream));
          if (error != cudaSuccess) break;
          ++state.work.integral_launches;
          for (std::size_t index = run.first; index < run.first + run.count; ++index)
            state.work.component_evaluations += selected.batches[index].count * tile.device.count;
        }
        if (error == cudaSuccess) {
          accumulate_batches<<<std::min<std::size_t>(selected.batch_reductions.size(), 65535), 128,
                               0, stream>>>(
              groups, tiles, batches,
              reinterpret_cast<const BatchReduction*>(base + plan.batch_reductions),
              selected.batch_reductions.size(), resident, status);
          error = cudaGetLastError();
          for (const auto& batch : selected.batches)
            state.work.reduction_terms += 25 * selected.tiles[batch.tile].device.count;
        }
      }
      for (const auto& run : selected.projection_runs) {
        if (error != cudaSuccess) break;
        project_batches<<<std::min<std::size_t>(run.count, 65535), 128, 0, stream>>>(
            shape, groups, run.first, run.count, run.offset, reference.basis, direction, density,
            delta_density, resident, reinterpret_cast<double*>(base + plan.batch_projected),
            status);
        error = cudaGetLastError();
        ++state.work.projection_launches;
        if (error != cudaSuccess) break;
        reduce_batches<<<std::min<std::size_t>((plan.candidate_count - 1) / 4 + 1, 65535), 128, 0,
                         stream>>>(
            shape, groups, reinterpret_cast<const std::size_t*>(base + plan.reduction_offsets),
            reinterpret_cast<const ReductionEntry*>(base + plan.reduction_entries),
            reinterpret_cast<const double*>(base + plan.batch_projected), run.first, run.count,
            run.offset, plan.candidate_count, candidate, status);
        error = cudaGetLastError();
        ++state.work.reduction_launches;
        for (std::size_t index = run.first; index < run.first + run.count; ++index)
          state.work.reduction_terms +=
              selected.groups[index].device.quartets * (25 + 8 * shape.spins);
      }
      state.work.projection_terms_bound += plan.projections;
    }
    for (std::size_t index = 0;
         !plan.batch_bytes && index < selected.groups.size() && error == cudaSuccess; ++index) {
      const auto& group = selected.groups[index];
      auto* cache = plan.resident_bytes
                        ? reinterpret_cast<double*>(base + plan.resident) + group.resident_offset
                        : streaming_cache;
      const bool build = !reuse && (!plan.resident_bytes || group.resident_source == index);
      if (build)
        error = cudaMemsetAsync(cache, 0, group.device.components * 25 * sizeof(double), stream);
      for (const auto tile_index : selected.closure[group.closure]) {
        if (!build) break;
        const auto& tile = selected.tiles[tile_index];
        for (std::size_t begin = 0; begin < group.radials && error == cudaSuccess;) {
          const auto count = std::min(state.options.radial_tile, group.radials - begin);
          build_records<<<1, 128, 0, stream>>>(shape, groups + index, reference.basis, begin, count,
                                               records, status);
          error = cudaGetLastError();
          if (error == cudaSuccess)
            error = static_cast<cudaError_t>(tile.program->integral(
                records, count, groups[index].atoms, direction, raw, status, stream));
          if (error == cudaSuccess) {
            state.work.component_evaluations += count * tile.device.count;
            ++state.work.integral_launches;
            accumulate<<<1, 128, 0, stream>>>(groups + index, tiles + tile_index, raw, cache,
                                              status);
            error = cudaGetLastError();
            state.work.reduction_terms += 25 * tile.device.count;
          }
          begin += count;
        }
      }
      if (error == cudaSuccess) {
        project<<<blocks(group.device.quartets), 128, 0, stream>>>(
            shape, groups + index, reference.basis, direction, density, delta_density, cache,
            projected, status);
        error = cudaGetLastError();
        state.work.projection_terms_bound += projection_bound(group.device);
        ++state.work.projection_launches;
      }
      if (error == cudaSuccess) {
        reduce_group<<<1, 128, 0, stream>>>(shape, groups + index, projected, candidate, status);
        error = cudaGetLastError();
        state.work.reduction_terms += group.device.quartets * (25 + 8 * shape.spins);
        ++state.work.reduction_launches;
      }
    }
  };
  if (state.schedule.windows.empty())
    run_schedule(state.schedule);
  else
    for (const auto& window : state.schedule.windows) {
      if (error != cudaSuccess) break;
      run_schedule(window);
    }
  for (unsigned panel = 0; panel < 2 && error == cudaSuccess; ++panel) {
    symmetric<<<blocks(plan.panels), 128, 0, stream>>>(
        candidate + 1 + panel * plan.panels, shape.nao, shape.spins,
        state.options.symmetry_absolute_tolerance, state.options.symmetry_relative_tolerance,
        status);
    error = cudaGetLastError();
  }
  if (error == cudaSuccess) {
    finite_values<<<blocks(plan.candidate_count), 128, 0, stream>>>(candidate, plan.candidate_count,
                                                                    status);
    error = cudaGetLastError();
  }
  error = error == cudaSuccess ? gate(status, stream) : drain(error, stream);
  if (error == cudaSuccess && plan.resident_bytes)
    error = cudaMemcpyAsync(base + plan.resident_direction, direction,
                            3 * shape.atoms * sizeof(double), cudaMemcpyDeviceToDevice, stream);
  if (error == cudaSuccess) error = state.authenticate(stream);
  if (error != cudaSuccess) return drain(error, stream);
  std::size_t offset{};
  for (unsigned index = 0; index < 5 && error == cudaSuccess; ++index) {
    error =
        cudaMemcpyAsync(outputs[index], candidate + offset, output_counts[index] * sizeof(double),
                        cudaMemcpyDeviceToDevice, stream);
    offset += output_counts[index];
  }
  error = drain(error, stream);
  if (error == cudaSuccess) {
    ++state.work.completed_actions;
    state.resident_valid = plan.resident_bytes != 0;
  }
  return error;
}
}  // namespace generativeqc::dft
