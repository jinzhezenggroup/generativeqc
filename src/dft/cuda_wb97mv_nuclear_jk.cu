#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <iterator>
#include <limits>
#include <map>
#include <new>
#include <vector>

#include "dft/cuda_wb97mv_nuclear_jk.hpp"
#include "dft/factorized_hessian_integral.hpp"
#include "dft/factorized_hessian_radial_pair.hpp"
#include "dft/hessian_aot.hpp"
#include "integrals/coulomb_derivative_envelope.hpp"
#include "integrals/eri_geometry.hpp"
#include "response/nuclear_integral_response.cuh"
#include "response/wb97mv_nuclear_jk.cuh"

namespace generativeqc::dft {
namespace {
namespace aot = hessian_aot;
namespace maps = generativeqc_wb97mv_nuclear_jk_ad;
using Shell = integrals::CudaOneElectronShell;
using Quartet = std::array<std::size_t, 4>;
// A closure key encodes one generated selector followed by four shell momenta (s–f).
constexpr unsigned angular_count = 4;
constexpr unsigned frozen_fields = 20;
constexpr unsigned max_sp_quartets = 81;
constexpr std::size_t closure_count =
    std::size(maps::radial) * angular_count * angular_count * angular_count * angular_count;
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
  double exchange_fraction{};
};
struct DeviceTile {
  unsigned components[16]{};
  unsigned count{};
};
struct Tile {
  const aot::Program* program{};
  DeviceTile device;
  double omega{};
};
struct Group {
  DeviceGroup device;
  std::size_t radials{}, closure{}, resident_offset{}, resident_source{};
};
// The compiler's Coulomb map publishes (ab) and (cd). Its exchange incidences
// (ac) and (bd) are identically zero, not density- or geometry-screened.
__host__ __device__ unsigned fock_incidence_count(maps::Kernel kernel) {
  return kernel == maps::Kernel::k_coulomb ? 2 : 4;
}
std::size_t quartet_reduction_bound(const DeviceGroup& group, std::size_t spins) {
  return 25 + 2 * spins * fock_incidence_count(group.kernel);
}
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
  std::size_t frozen_entry_counts{}, frozen_value_offsets{};
  std::size_t retained_window_source_bytes{};
  std::size_t prepared_geometry{}, prepared_count{};
  std::size_t active_components{}, active_component_capacity{};
  std::size_t window_radials{};
  bool packed_frozen_projection{}, single_cartesian_component{}, paired_radial_cache{};
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
  std::uint32_t group{};
  std::uint8_t incidence{}, first{}, second{}, packed_offset{};
};
struct ActiveComponentSpan {
  std::size_t offset{}, count{}, max_radials{};
  std::uint64_t evaluations{}, reduction_terms{};
};
using ActiveComponentAtom = std::array<ActiveComponentSpan, 5>;
struct Schedule {
  Layout plan;
  std::vector<Group> groups;
  std::vector<Tile> tiles;
  std::array<std::vector<std::size_t>, closure_count> closure;
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
  std::vector<std::size_t> frozen_atoms;
  std::vector<ActiveComponentAtom> active_atoms;
  std::vector<aot::FactorizedActiveComponent> active_tasks;
  std::vector<std::uint64_t> replay_source_words;
  std::size_t replay_windows{};
  std::size_t translation_anchor{SIZE_MAX};
  std::size_t ordered_groups{}, source_groups{};
  const char* refusal{"shape, topology, or options"};
  CudaWb97mvJkAdmissionFailure admission_failure{CudaWb97mvJkAdmissionFailure::other};
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
      !options.max_batch_projection_bytes || schedule.groups.size() > UINT32_MAX)
    return;
  for (const auto& tile : schedule.tiles)
    if (!tile.program->integral_batch) return;
  auto candidate = plan;
  candidate.packed_frozen_projection = options.prefer_fresh_direction_batches;
  candidate.single_cartesian_component = true;
  // Fuse only a complete window for which every source can use the same
  // two-chunk publisher. Keep the general tiled scheduler for everything else.
  candidate.paired_radial_cache = options.prefer_fresh_direction_batches &&
      options.radial_tile == 64 &&
      std::all_of(schedule.tiles.begin(), schedule.tiles.end(), [](const Tile& tile) {
        return tile.program->signature.radial == "full_range" && tile.omega == 0 &&
               std::ranges::all_of(tile.program->signature.angular, [](auto order) {
                 return order <= 1;
               });
      }) && std::all_of(schedule.groups.begin(), schedule.groups.end(), [](const Group& group) {
        return group.radials <= 128;
      });
  std::size_t records{}, raw{}, inputs{}, reductions{}, projected{}, entries{}, count{},
      maximum_projected{};
  constexpr unsigned pairs[4][2] = {{0, 1}, {2, 3}, {0, 2}, {1, 3}};
  for (std::size_t index = 0; index < schedule.groups.size(); ++index) {
    auto& group = schedule.groups[index];
    group.device.resident_offset = group.resident_offset;
    group.device.projected_offset = projected;
    group.device.source = group.resident_source == index;
    for (const auto& shell : group.device.shells) {
      candidate.packed_frozen_projection &= shell.angular <= 1;
      candidate.single_cartesian_component &= shell.angular == 0;
    }
    if (!multiply(group.device.quartets, 41, count) || !add(projected, count) || !add(entries, 1))
      return;
    maximum_projected = std::max(maximum_projected, count);
    for (unsigned incidence = 0; incidence < fock_incidence_count(group.device.kernel);
         ++incidence) {
      const auto& pair = pairs[incidence];
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
    const auto tiles = schedule.closure[group.closure].size();
    if (!multiply(group.radials, 5, count) || !add(records, count) ||
        !multiply(candidate.paired_radial_cache ? 1 : chunks, tiles, count) ||
        !add(inputs, count))
      return;
    if (!candidate.paired_radial_cache &&
        (!multiply(group.device.components, 25, count) || !multiply(count, chunks, count) ||
         !add(raw, count) || !add(reductions, tiles)))
      return;
  }
  // Admit the entire optional layout before constructing its host metadata.
  // No partial batch replaces the already admitted ordered streaming schedule.
  if (!append(candidate, records, sizeof(double), candidate.batch_records) ||
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
  // Ordered radial folding finishes before any projection, on this same
  // stream. Projection may overwrite the raw outputs but not resident jets.
  const auto scratch_values = std::max(raw, projection_values);
  if (projection_values < maximum_projected || scratch_values > available / 256 * 32 ||
      !append(candidate, scratch_values, sizeof(double), candidate.batch_raw))
    return;
  candidate.batch_projected = candidate.batch_raw;
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
    for (const auto tile : schedule.closure[group.closure]) {
      if (candidate.paired_radial_cache) {
        tiles[tile].push_back({index, tile, 0, group.radials, group.device.records_offset, 0});
        continue;
      }
      for (std::size_t begin = 0; begin < group.radials; begin += options.radial_tile) {
        tiles[tile].push_back({index, tile, begin,
                               std::min(options.radial_tile, group.radials - begin),
                               group.device.records_offset + 5 * begin, raw});
        raw += schedule.tiles[tile].device.count * 25;
      }
    }
  }
  for (std::size_t tile = 0; tile < tiles.size(); ++tile) {
    const auto first = schedule.batches.size();
    for (const auto& batch : tiles[tile]) {
      if (!candidate.paired_radial_cache) {
        if (schedule.batches.size() == first || schedule.batches.back().group != batch.group)
          schedule.batch_reductions.push_back({batch.group, tile, schedule.batches.size(), 0});
        ++schedule.batch_reductions.back().count;
      }
      schedule.batches.push_back(batch);
    }
    for (std::size_t begin = first; begin < schedule.batches.size(); begin += 65535)
      schedule.batch_runs.push_back(
          {tile, begin, std::min<std::size_t>(65535, schedule.batches.size() - begin)});
  }
  // Primal and directional outputs have identical incidence lists. Share the
  // sparse metadata, not their numerical reductions or summation order.
  std::vector<std::vector<ReductionEntry>> owners(1 + plan.panels + 3 * shape.atoms);
  const auto vector_begin = 1 + plan.panels;
  const auto entry = [](std::size_t group, unsigned incidence = 0, unsigned first = 0,
                        unsigned second = 0, unsigned packed_offset = 0) {
    // Ordered admission bounds local orbitals by the Cartesian f shell (10).
    return ReductionEntry{static_cast<std::uint32_t>(group), static_cast<std::uint8_t>(incidence),
                          static_cast<std::uint8_t>(first), static_cast<std::uint8_t>(second),
                          static_cast<std::uint8_t>(packed_offset)};
  };
  for (std::size_t index = 0; index < schedule.groups.size(); ++index) {
    const auto& group = schedule.groups[index].device;
    owners[0].push_back(entry(index));
    unsigned packed_base = 12;
    for (unsigned incidence = 0; incidence < fock_incidence_count(group.kernel); ++incidence) {
      const auto first = pairs[incidence][0], second = pairs[incidence][1];
      for (unsigned spin = 0; spin < shape.spins; ++spin)
        for (unsigned row = 0; row < group.shells[first].ao_count; ++row)
          for (unsigned column = 0; column < group.shells[second].ao_count; ++column) {
            const auto target = 1 + spin * plan.matrix +
                                (group.shells[first].ao_begin + row) * shape.nao +
                                group.shells[second].ao_begin + column;
            const auto packed_offset = candidate.packed_frozen_projection
                                           ? packed_base + (row * group.shells[second].ao_count +
                                                            column) * shape.spins + spin
                                           : 0;
            owners[target].push_back(entry(index, incidence, row, column, packed_offset));
          }
      packed_base += shape.spins * group.shells[first].ao_count * group.shells[second].ao_count;
    }
    for (unsigned center = 0; center < 4; ++center) {
      bool first = true;
      for (unsigned previous = 0; previous < center; ++previous)
        first = first && group.atoms[previous] != group.atoms[center];
      if (!first) continue;
      for (unsigned axis = 0; axis < 3; ++axis) {
        const auto target = vector_begin + 3 * group.atoms[center] + axis;
        owners[target].push_back(entry(index, center, axis, 0, 3 * center + axis));
      }
    }
  }
  for (const auto& owner : owners) {
    schedule.reduction_offsets.push_back(schedule.reduction_entries.size());
    schedule.reduction_entries.insert(schedule.reduction_entries.end(), owner.begin(), owner.end());
  }
  schedule.reduction_offsets.push_back(schedule.reduction_entries.size());
  candidate.batch_bytes = candidate.bytes - plan.bytes;
  candidate.window_radials = records / 5;
  plan = candidate;
}

void retain_window_sources(Schedule& schedule, const CudaWb97mvJkOptions& options) {
  auto& plan = schedule.plan;
  const auto reserve = [](Layout& tail, const Schedule& window, Layout& placement) {
    const auto& scratch = window.plan;
    return append(tail, window.groups.size(), sizeof(DeviceGroup), placement.groups) &&
           append(tail, window.tiles.size(), sizeof(DeviceTile), placement.tiles) &&
           append(tail, scratch.batch_inputs - scratch.batch_records, 1, placement.batch_records) &&
           append(tail, window.batches.size(), sizeof(aot::FactorizedIntegralBatch),
                  placement.batch_inputs) &&
           append(tail, window.batch_reductions.size(), sizeof(BatchReduction),
                  placement.batch_reductions) &&
           append(tail, window.reduction_offsets.size(), sizeof(std::size_t),
                  placement.reduction_offsets) &&
           append(tail, window.reduction_entries.size(), sizeof(ReductionEntry),
                  placement.reduction_entries);
  };
  // Source metadata and compact primitive records are independent of the
  // direction. Admit the complete read-only tail before changing any binding;
  // raw jets, projected quartets and reductions still reuse a single window.
  auto candidate = plan;
  for (const auto& window : schedule.windows) {
    auto placement = window.plan;
    if (!reserve(candidate, window, placement) || candidate.bytes > options.max_workspace_bytes ||
        candidate.bytes - plan.bytes > options.max_batch_workspace_bytes - plan.batch_bytes)
      return;
  }
  const auto retained = candidate.bytes - plan.bytes;
  for (auto& window : schedule.windows) {
    auto placement = window.plan;
    (void)reserve(plan, window, placement);
    window.plan = placement;
  }
  plan.retained_window_source_bytes = retained;
}

cudaError_t ordered_schedule(const CudaWb97mvJkShape& shape, std::span<const Shell> shells,
                             const CudaWb97mvJkOptions& options, Schedule& result,
                             std::span<const Quartet> selection = {}) {
  auto& plan = result.plan;
  if (options.part == CudaWb97mvJkPart::full_hybrid
          ? !(options.full_exchange_fraction > 0 && options.full_exchange_fraction <= 1)
          : options.full_exchange_fraction != 0)
    return cudaErrorInvalidValue;
  const auto range = options.range_exchange;
  if (range &&
      ((options.part != CudaWb97mvJkPart::exact && options.part != CudaWb97mvJkPart::correction) ||
       !(range->short_fraction >= 0 && range->short_fraction <= 1) ||
       !(range->long_minus_short >= -1 && range->long_minus_short <= 1) ||
       !std::isfinite(range->omega) || range->omega <= 0))
    return cudaErrorInvalidValue;
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
    if (shell.atom >= shape.atoms || shell.angular >= angular_count || !shell.primitive_count ||
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
      kernels = range ? std::vector{range->short_fraction ? maps::Kernel::k_full_hybrid
                                                          : maps::Kernel::k_coulomb,
                                    maps::Kernel::k_range_hybrid}
                      : std::vector{maps::Kernel::k_primary, maps::Kernel::k_correction};
      break;
    case CudaWb97mvJkPart::primary:
      kernels = {maps::Kernel::k_primary};
      break;
    case CudaWb97mvJkPart::correction:
      kernels = {range ? maps::Kernel::k_range_hybrid : maps::Kernel::k_correction};
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
    case CudaWb97mvJkPart::full_hybrid:
      kernels = {maps::Kernel::k_full_hybrid};
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
      group.device.exchange_fraction = options.full_exchange_fraction;
      if (range)
        group.device.exchange_fraction = kernel == maps::Kernel::k_range_hybrid
                                             ? range->long_minus_short
                                             : range->short_fraction;
      const auto selected = static_cast<std::size_t>(kernel);
      const double omega =
          kernel == maps::Kernel::k_range_hybrid ? range->omega : maps::omega[selected];
      auto key = selected;
      for (unsigned slot = 0; slot < 4; ++slot) {
        const auto& shell = group.device.shells[slot];
        group.device.atoms[slot] = shell.atom;
        key = key * angular_count + shell.angular;
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
          if (!program.accepts_omega(omega) || program.centers != 4 ||
              program.record_stride != 17 || program.outputs != 25 || program.components.empty() ||
              program.components.size() > 16)
            return cudaErrorNotSupported;
          Tile tile{&program, {}, omega};
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
          !multiply(group.device.quartets, quartet_reduction_bound(group.device, shape.spins),
                    count) ||
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
      plan.reductions += group.device.quartets * quartet_reduction_bound(group.device, shape.spins);
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

bool canonical_window_options(const CudaWb97mvJkShape& shape, const CudaWb97mvJkOptions& options,
                              std::size_t panels, CudaWb97mvJkOptions& tile_options) {
  Layout shared;
  shared.candidate_count = 1;
  std::size_t matrix_values{}, vector_values{};
  if (!multiply(panels, 2, matrix_values) || !multiply(shape.atoms, 6, vector_values) ||
      !add(shared.candidate_count, matrix_values) || !add(shared.candidate_count, vector_values) ||
      !append(shared, shared.candidate_count, sizeof(double), shared.candidates) ||
      !append(shared, 2, sizeof(int), shared.status) || shared.bytes >= options.max_workspace_bytes)
    return false;
  tile_options = options;
  tile_options.max_workspace_bytes -= shared.bytes;
  tile_options.max_resident_integral_bytes = tile_options.max_workspace_bytes;
  return true;
}

template <class Visit>
cudaError_t visit_canonical_windows(const CudaWb97mvJkShape& shape, std::span<const Shell> shells,
                                    const CudaWb97mvJkOptions& tile_options,
                                    std::span<const std::uint64_t> source_words, const char*& refusal,
                                    Visit&& visit) {
  std::vector<Quartet> pending;
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
    refusal = window.refusal;
    // A tile must reuse each canonical source. Never turn a capacity fallback
    // into eightfold primitive replay or hundreds of thousands of launches.
    if (error == cudaSuccess && (!window.plan.resident_bytes || !window.plan.batch_bytes)) {
      refusal = "canonical tile resident/batch arena admission";
      error = cudaErrorInvalidValue;
    }
    if (error == cudaErrorInvalidValue && canonical.size() > 1) {
      const auto half = canonical.size() / 2;
      // Release a failed expanded window before planning its children.
      window = {};
      std::vector<Quartet>{}.swap(selected);
      error = self(self, canonical.first(half));
      return error == cudaSuccess ? self(self, canonical.subspan(half)) : error;
    }
    if (error != cudaSuccess) return error;
    if (tile_options.prefer_fresh_direction_batches) {
      // All ordered permutations preserve the source's atom incidences.
      // Use a common anchor only when every source in the window has it.
      for (const auto slot : canonical.front()) {
        const auto atom = shells[slot].atom;
        if (std::all_of(canonical.begin(), canonical.end(), [&](const Quartet& source) {
              return std::any_of(source.begin(), source.end(),
                                 [&](std::size_t shell) { return shells[shell].atom == atom; });
            })) {
          window.translation_anchor = atom;
          break;
        }
      }
      for (const auto& source : canonical) {
        const long long atoms[]{static_cast<long long>(shells[source[0]].atom),
                                static_cast<long long>(shells[source[1]].atom),
                                static_cast<long long>(shells[source[2]].atom),
                                static_cast<long long>(shells[source[3]].atom)};
        for (const auto atom : atoms)
          if (static_cast<std::size_t>(atom) != window.translation_anchor &&
              aot::active_eri_centers(atoms, atom))
            window.frozen_atoms.push_back(static_cast<std::size_t>(atom));
      }
      std::sort(window.frozen_atoms.begin(), window.frozen_atoms.end());
      window.frozen_atoms.erase(std::unique(window.frozen_atoms.begin(), window.frozen_atoms.end()),
                                window.frozen_atoms.end());
    }
    return visit(std::move(window));
  };
  const auto emit = [&](const Quartet& quartet) {
    // Keep a shared translation anchor within a window; don't combine
    // different first atoms just to fill the last tile.
    if (tile_options.prefer_fresh_direction_batches && !pending.empty() &&
        shells[pending.front()[0]].atom != shells[quartet[0]].atom) {
      const auto error = flush(flush, pending);
      if (error != cudaSuccess) return error;
      pending.clear();
    }
    pending.push_back(quartet);
    if (pending.size() == tile_options.source_tile) {
      const auto error = flush(flush, pending);
      if (error != cudaSuccess) return error;
      pending.clear();
    }
    return cudaSuccess;
  };
  if (source_words.empty()) {
    for (std::size_t first = 0; first < shells.size(); ++first)
      for (std::size_t second = first; second < shells.size(); ++second)
        for (std::size_t third = first; third < shells.size(); ++third)
          for (std::size_t fourth = third == first ? second : third; fourth < shells.size();
               ++fourth) {
            const auto error = emit({first, second, third, fourth});
            if (error != cudaSuccess) return error;
          }
  } else {
    // Visit only set bits in canonical order. Decode triangular rows once
    // as they advance, rather than deriving shell indices for skipped entries.
    std::vector<std::array<std::size_t, 2>> pairs;
    pairs.reserve(shells.size() * (shells.size() + 1) / 2);
    for (std::size_t first = 0; first < shells.size(); ++first)
      for (std::size_t second = first; second < shells.size(); ++second)
        pairs.push_back({first, second});
    std::size_t row{}, row_begin{}, row_end = pairs.size();
    for (std::size_t word = 0; word < source_words.size(); ++word)
      for (auto live = source_words[word]; live; live &= live - 1) {
        const auto index = 64 * word + std::countr_zero(live);
        while (index >= row_end) {
          row_begin = row_end;
          if (++row == pairs.size()) return cudaErrorInvalidValue;
          row_end += pairs.size() - row;
        }
        const auto& right = pairs[row + index - row_begin];
        const auto error = emit({pairs[row][0], pairs[row][1], right[0], right[1]});
        if (error != cudaSuccess) return error;
      }
  }
  return flush(flush, pending);
}

bool admits_compact_frozen(const Schedule& window) {
  if (!window.plan.packed_frozen_projection || window.groups.empty()) return false;
  const auto& last = window.groups.back().device;
  const auto bound = last.quartets * 41;
  return bound <= UINT32_MAX && last.projected_offset <= UINT32_MAX - bound &&
         window.reduction_entries.size() <= UINT32_MAX;
}

template<class Emit>
bool visit_active_batches(const Schedule& window, Emit&& emit) {
  if (window.batches.size() > UINT32_MAX) return false;
  for (std::size_t index = 0; index < window.batches.size(); ++index) {
    const auto& batch = window.batches[index];
    const auto& group = window.groups[batch.group].device;
    const auto& tile = window.tiles[batch.tile].device;
    unsigned mask = 0;
    for (unsigned slot = 0; slot < 4; ++slot) mask |= group.shells[slot].angular << slot;
    for (unsigned slot = 0; slot < 4; ++slot) {
      const auto atom = group.atoms[slot];
      if (std::find(group.atoms, group.atoms + slot, atom) != group.atoms + slot ||
          !aot::active_eri_centers(group.atoms, atom))
        continue;
      const auto position = std::lower_bound(window.frozen_atoms.begin(), window.frozen_atoms.end(),
                                             static_cast<std::size_t>(atom));
      if (position == window.frozen_atoms.end() || *position != static_cast<std::size_t>(atom))
        continue;
      if (!emit(position - window.frozen_atoms.begin(), index, batch, tile, mask)) return false;
    }
  }
  return true;
}

/** The radial cache fields have one writer per canonical component. Index
 * only the writers relevant to each frozen atom, sharing the index across
 * its Cartesian axes. Reordering the independent writes cannot reorder a
 * radial fold or the later density contraction. */
void build_active_components(Schedule& window, std::size_t capacity) {
  if (!capacity) return;
  std::vector<ActiveComponentAtom> atoms(window.frozen_atoms.size());
  std::size_t total{};
  if (!visit_active_batches(window, [&](auto atom, auto, const auto& batch, const auto& tile,
                                        auto mask) {
        if (tile.count > capacity - total) return false;
        auto& span = atoms[atom][std::popcount(mask)];
        total += tile.count;
        span.count += tile.count;
        span.max_radials = std::max(span.max_radials, batch.count);
        span.evaluations += batch.count * tile.count;
        span.reduction_terms += (batch.count > 64 ? 50 : 25) * tile.count;
        return true;
      }) || !total)
    return;
  std::size_t end{};
  for (auto& atom : atoms) {
    for (auto& span : atom) {
      span.offset = end;
      end += span.count;
    }
  }
  window.active_tasks.resize(total);
  visit_active_batches(window, [&](auto atom, auto index, const auto&, const auto& tile,
                                   auto mask) {
    auto& span = atoms[atom][std::popcount(mask)];
    for (unsigned component = 0; component < tile.count; ++component)
      window.active_tasks[span.offset++] = {static_cast<std::uint32_t>(index),
                                           (mask << 7) | tile.components[component]};
    return true;
  });
  for (auto& atom : atoms)
    for (auto& span : atom) span.offset -= span.count;
  window.active_atoms = std::move(atoms);
}

void bind_window_tail(Schedule& window, const Layout& plan) {
  window.plan.candidates = plan.candidates;
  window.plan.status = plan.status;
  if (plan.frozen_value_offsets && admits_compact_frozen(window)) {
    window.plan.frozen_entry_counts = plan.frozen_entry_counts;
    window.plan.frozen_value_offsets = plan.frozen_value_offsets;
  }
}

cudaError_t canonical_schedule(const CudaWb97mvJkShape& shape, std::span<const Shell> shells,
                               const CudaWb97mvJkOptions& options, std::size_t ordered_panels,
                               Schedule& result, std::span<const Quartet> omitted = {}) {
  CudaWb97mvJkOptions tile_options;
  if (!canonical_window_options(shape, options, ordered_panels, tile_options))
    return cudaErrorInvalidValue;
  result.refusal = "canonical source group bound";
  std::size_t pairs{}, sources{};
  if (!multiply(shells.size(), shells.size() + 1, pairs)) return cudaErrorInvalidValue;
  pairs /= 2;
  if (!multiply(pairs, pairs + 1, sources)) return cudaErrorInvalidValue;
  sources /= 2;
  const auto kernels = options.part == CudaWb97mvJkPart::exact ? 2 : 1;
  if (!multiply(sources, kernels, sources) || omitted.size() >= sources)
    return cudaErrorInvalidValue;
  const auto retained = sources - omitted.size();
  const bool replay = retained > options.max_source_groups;
  if (replay && (omitted.empty() || retained > options.max_frozen_replay_candidates)) {
    result.admission_failure = CudaWb97mvJkAdmissionFailure::canonical_source_limit;
    return cudaErrorInvalidValue;
  }
  std::vector<std::uint64_t> source_words;
  if (!omitted.empty()) {
    const auto count = sources / 64 + (sources % 64 != 0);
    const auto limit = replay ? options.max_frozen_replay_candidates : options.max_source_groups;
    if (kernels != 1 || count > limit / 32 + (limit % 32 != 0)) {
      result.admission_failure = CudaWb97mvJkAdmissionFailure::canonical_source_limit;
      result.refusal = "canonical source bitmap bound";
      return cudaErrorInvalidValue;
    }
    source_words.assign(count, UINT64_MAX);
    if (const auto tail = sources % 64) source_words.back() = (std::uint64_t{1} << tail) - 1;
    const auto prefix = [](std::size_t row, std::size_t width) {
      return row * (2 * width - row + 1) / 2;
    };
    for (const auto& quartet : omitted) {
      if (quartet[0] > quartet[1] || quartet[2] > quartet[3] ||
          quartet[1] >= shells.size() || quartet[3] >= shells.size())
        return cudaErrorInvalidValue;
      const auto first = prefix(quartet[0], shells.size()) + quartet[1] - quartet[0];
      const auto second = prefix(quartet[2], shells.size()) + quartet[3] - quartet[2];
      if (first > second) return cudaErrorInvalidValue;
      const auto index = prefix(first, pairs) + second - first;
      source_words[index / 64] &= ~(std::uint64_t{1} << (index % 64));
    }
  }
  auto& plan = result.plan;
  std::size_t compact_entries{};
  using Task = aot::FactorizedActiveComponent;
  constexpr std::size_t max_task_slot = 4ULL << 20;
  std::vector<std::pair<std::size_t, std::size_t>> geometry_windows;
  const auto error = visit_canonical_windows(
      shape, shells, tile_options, source_words, result.refusal, [&](Schedule&& window) {
        if (!result.ordered_groups) {
          plan.matrix = window.plan.matrix;
          plan.panels = window.plan.panels;
          plan.basis_elements = window.plan.basis_elements;
          plan.candidate_count = window.plan.candidate_count;
          plan.shells = window.plan.shells;
        }
        result.refusal = "aggregate canonical primitive, projection, or reduction work bound";
        if (!add(plan.components, window.plan.components) ||
            !add(plan.projections, window.plan.projections) ||
            !add(plan.reductions, window.plan.reductions) ||
            !add(plan.radial_records, window.plan.radial_records) ||
            !add(result.ordered_groups, window.groups.size()) ||
            plan.components > options.max_primitive_component_evaluations ||
            plan.projections > options.max_projection_terms ||
            plan.reductions > options.max_reduction_terms)
          return cudaErrorInvalidValue;
        plan.bytes = std::max(plan.bytes, window.plan.bytes);
        plan.batch_bytes = std::max(plan.batch_bytes, window.plan.batch_bytes);
        if (admits_compact_frozen(window))
          compact_entries = std::max(compact_entries, window.reduction_entries.size());
        if (window.plan.paired_radial_cache && window.frozen_atoms.size() >= 2) {
          std::size_t tasks{};
          if (replay && !visit_active_batches(window, [&](auto, auto, const auto&, const auto& tile,
                                                           auto) {
                tasks += tile.count;
                return tasks <= max_task_slot / sizeof(Task);
              }))
            tasks = 0;
          geometry_windows.emplace_back(window.plan.window_radials, tasks);
        }
        if (replay)
          ++result.replay_windows;
        else
          result.windows.push_back(std::move(window));
        return cudaSuccess;
      });
  if (error != cudaSuccess) return error;
  if (!result.ordered_groups) return cudaErrorInvalidValue;
  result.refusal = "aggregate tiled arena bound";
  if (!append(plan, plan.candidate_count, sizeof(double), plan.candidates) ||
      !append(plan, 2, sizeof(int), plan.status) || plan.bytes > options.max_workspace_bytes)
    return cudaErrorInvalidValue;
  for (auto& window : result.windows) bind_window_tail(window, plan);
  if (options.prefer_fresh_direction_batches) {
    if (omitted.empty()) retain_window_sources(result, options);
    // Only the source-major sweep reuses one atom's owners across its three
    // axes. Reserve one shared list after the ordinary streamed plan fits;
    // retaining all sources or shrinking the storage cap must not force it.
    if (!plan.retained_window_source_bytes) {
      auto compact = plan;
      if (compact_entries && append(compact, plan.panels + 3 * shape.atoms,
                                    sizeof(std::uint32_t), compact.frozen_entry_counts) &&
          append(compact, compact_entries, sizeof(std::uint32_t), compact.frozen_value_offsets) &&
          compact.bytes <= options.max_workspace_bytes &&
          compact.bytes - plan.bytes <= options.max_batch_workspace_bytes - plan.batch_bytes) {
        for (auto& window : result.windows) bind_window_tail(window, compact);
        compact.batch_bytes += compact.bytes - plan.bytes;
        plan = compact;
      }
      // One typed cache can serve all directions of the current window. Only
      // spend spare arena capacity; never split a window to fit this option.
      constexpr std::size_t cache_limit = 32ULL << 20;
      const auto spare = std::min({cache_limit, options.max_workspace_bytes - plan.bytes,
                                   options.max_batch_workspace_bytes - plan.batch_bytes});
      auto prepared = plan;
      const auto capacity = spare / 256 * 256 / sizeof(aot::FactorizedPairGeometry);
      for (const auto& [count, unused_tasks] : geometry_windows)
        if (count <= capacity) prepared.prepared_count = std::max(prepared.prepared_count, count);
      if (prepared.prepared_count &&
          append(prepared, prepared.prepared_count, sizeof(aot::FactorizedPairGeometry),
                 prepared.prepared_geometry)) {
        prepared.batch_bytes += prepared.bytes - plan.bytes;
        plan = prepared;
      }
      // Both source schedules share one device slot. Replay retains only
      // its extent and regenerates the index with the window metadata.
      if (plan.prepared_count) {
        const auto slack = std::min({max_task_slot, options.max_workspace_bytes - plan.bytes,
                                     options.max_batch_workspace_bytes - plan.batch_bytes});
        const auto capacity = slack / 256 * 256 / sizeof(Task);
        std::size_t host_bytes = 24ULL << 20, largest{};
        if (replay)
          for (const auto& [radials, tasks] : geometry_windows)
            if (radials <= plan.prepared_count && tasks <= capacity)
              largest = std::max(largest, tasks);
        for (auto& window : result.windows) {
          if (!window.plan.paired_radial_cache || window.frozen_atoms.size() < 2 ||
              window.plan.window_radials > plan.prepared_count)
            continue;
          if (window.frozen_atoms.size() > host_bytes / sizeof(ActiveComponentAtom)) continue;
          const auto spans = window.frozen_atoms.size() * sizeof(ActiveComponentAtom);
          try {
            build_active_components(window, std::min((host_bytes - spans) / sizeof(Task), capacity));
          } catch (const std::bad_alloc&) {
            break;
          }
          if (window.active_tasks.empty()) continue;
          largest = std::max(largest, window.active_tasks.size());
          host_bytes -= spans + window.active_tasks.size() * sizeof(Task);
        }
        const auto before = plan.bytes;
        if (largest && append(plan, largest, sizeof(Task), plan.active_components)) {
          plan.active_component_capacity = largest;
          plan.batch_bytes += plan.bytes - before;
        }
      }
    }
  }
  if (replay) result.replay_source_words = std::move(source_words);
  result.source_groups = retained;
  return cudaSuccess;
}

cudaError_t schedule(const CudaWb97mvJkShape& shape, std::span<const Shell> shells,
                     const CudaWb97mvJkOptions& options, Schedule& result,
                     const integrals::CoulombDerivativeOmissions* omissions = nullptr) {
  if (omissions) {
    result.refusal = "screened dense Cartesian source certificate";
    const std::span<const Quartet> omitted = omissions->sources;
    std::size_t panels{};
    if (!options.prefer_fresh_direction_batches || !options.source_tile ||
        !options.max_batch_workspace_bytes || !options.max_batch_projection_bytes ||
        shape.spins != 1 || options.range_exchange ||
        (options.part != CudaWb97mvJkPart::coulomb &&
         options.part != CudaWb97mvJkPart::full_hybrid) || omitted.empty() ||
        !(omissions->bound > 0 && omissions->bound <= omissions->max_bound) ||
        !std::is_sorted(omitted.begin(), omitted.end()) ||
        std::adjacent_find(omitted.begin(), omitted.end()) != omitted.end() ||
        !multiply(shape.nao, shape.nao, panels))
      return cudaErrorInvalidValue;
    return canonical_schedule(shape, shells, options, panels, result, omitted);
  }
  const auto ordered = ordered_schedule(shape, shells, options, result);
  if (!options.source_tile || !options.max_source_groups || !options.max_batch_workspace_bytes ||
      !options.max_batch_projection_bytes)
    return ordered;
  // Paired HVP actions can reuse a resident direction. A dense frozen-column
  // sweep builds each direction once, so retaining unbatched jets does not pay
  // for the per-group launches. Try the same bounded canonical windows that
  // preserve eightfold source sharing when residency does not fit.
  const bool unbatched_fresh = ordered == cudaSuccess && result.plan.resident_bytes &&
                               !result.plan.batch_bytes && options.prefer_fresh_direction_batches;
  if (ordered == cudaSuccess && ((result.plan.resident_bytes && !unbatched_fresh) ||
                                 !options.max_resident_integral_bytes || shells.size() < 2))
    return ordered;
  if (ordered != cudaSuccess && ordered != cudaErrorInvalidValue) return ordered;
  const auto ordered_panels = result.plan.panels;
  if (ordered != cudaSuccess) {
    const auto* ordered_refusal = result.refusal;
    result = {};
    result.refusal = ordered_refusal;
    return canonical_schedule(shape, shells, options, ordered_panels, result);
  }
  try {
    Schedule tiled;
    if (canonical_schedule(shape, shells, options, ordered_panels, tiled) == cudaSuccess &&
        (tiled.plan.components < result.plan.components ||
         (unbatched_fresh && tiled.plan.components == result.plan.components)))
      result = std::move(tiled);
  } catch (const std::bad_alloc&) {
    // Optional metadata must not invalidate the already admitted ordered plan.
  }
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
__global__ void accumulate_failure(const int* status, int* transaction_status) {
  if (*status) atomicOr(transaction_status, 1);
}
__global__ void publish_private_panel(const double* candidate, double* output, std::size_t count,
                                      const int* status) {
  if (*status) return;
  for (std::size_t index = blockIdx.x * blockDim.x + threadIdx.x; index < count;
       index += static_cast<std::size_t>(blockDim.x) * gridDim.x)
    output[index] = candidate[index];
}
__global__ void finite_values(const double* values, std::size_t count, int* status) {
  for (std::size_t index = blockIdx.x * blockDim.x + threadIdx.x; index < count;
       index += static_cast<std::size_t>(blockDim.x) * gridDim.x)
    if (!isfinite(values[index])) atomicExch(status, 1);
}

__global__ void screen_density_bound(const double* density, std::size_t count, int* status) {
  for (std::size_t index = blockIdx.x * blockDim.x + threadIdx.x; index < count;
       index += static_cast<std::size_t>(blockDim.x) * gridDim.x)
    if (!(fabs(density[index]) <= integrals::CoulombDerivativeOmissions::max_density_element))
      atomicExch(status, 1);
}
__global__ void frozen_unit(double* direction, std::size_t coordinates, std::size_t column) {
  for (std::size_t index = blockIdx.x * blockDim.x + threadIdx.x; index < coordinates;
       index += static_cast<std::size_t>(blockDim.x) * gridDim.x)
    direction[index] = static_cast<double>(index == column);
}
__global__ void validate_frozen_column(const double* direction, std::size_t coordinates,
                                       const double* delta_density, std::size_t panels,
                                       std::size_t column, int* status) {
  for (std::size_t index = blockIdx.x * blockDim.x + threadIdx.x; index < max(coordinates, panels);
       index += std::size_t(blockDim.x) * gridDim.x)
    if ((index < coordinates && direction[index] != static_cast<double>(index == column)) ||
        (index < panels && delta_density[index] != 0.0))
      atomicExch(status, 1);
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

/** Bind the same degree-specific geometry as the primitive producer. The
 * canonical record offset is shared by all of a shell quartet's components. */
__global__ void prepare_batch_geometry(const DeviceGroup* groups, std::size_t count,
                                       const double* basis, const double* records,
                                       aot::FactorizedPairGeometry* geometry, int* status) {
  for (std::size_t index = blockIdx.x; index < count; index += gridDim.x) {
    const auto& group = groups[index];
    if (!group.source || !aot::active_eri_centers(group.atoms, group.atoms[0])) continue;
    double coordinates[12];
    unsigned degree = 0;
    std::size_t radials = 1;
    for (unsigned slot = 0; slot < 4; ++slot) {
      degree += group.shells[slot].angular;
      radials *= group.shells[slot].primitive_count;
      for (unsigned axis = 0; axis < 3; ++axis)
        coordinates[3 * slot + axis] = basis[3 * group.atoms[slot] + axis];
    }
    for (std::size_t radial = threadIdx.x; radial < radials; radial += blockDim.x) {
      const auto offset = group.records_offset / 5 + radial;
      auto& target = geometry[offset];
      const auto* exponents = records + 5 * offset;
      constexpr auto range = integrals::CoulombRange::Full;
      bool valid = false;
      switch (degree) {
#define GQC_PREPARE_DEGREE(value) \
  case value: \
    valid = integrals::make_eri_geometry<aot::FactorizedPairGeometry, value + 2, 0, true>( \
        exponents, coordinates, value + 2, range, 0, target); \
    break
        GQC_PREPARE_DEGREE(0);
        GQC_PREPARE_DEGREE(1);
        GQC_PREPARE_DEGREE(2);
        GQC_PREPARE_DEGREE(3);
        GQC_PREPARE_DEGREE(4);
#undef GQC_PREPARE_DEGREE
      }
      if (!valid) {
        target = {};
        atomicExch(status, 1);
      }
    }
  }
}
__global__ void accumulate_batches(const DeviceGroup* groups, const DeviceTile* tiles,
                                   const aot::FactorizedIntegralBatch* batches,
                                   const BatchReduction* reductions, std::size_t count,
                                   double* resident, long long frozen_atom, int* status) {
  if (atomicAdd(status, 0)) return;
  for (std::size_t index = blockIdx.x; index < count; index += gridDim.x) {
    const auto reduction = reductions[index];
    const auto group = groups[reduction.group];
    if (!aot::active_eri_centers(group.atoms, frozen_atom)) continue;
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
// A nonzero field stride selects only the directional Fock and HVP fields.
template <unsigned FrozenStride = 0>
__device__ void project_values(CudaWb97mvJkShape shape, const DeviceGroup* group,
                               const double* basis, const double* direction, const double* density,
                               const double* delta_density, const double* cache, double* projected,
                               int* status, std::size_t first, std::size_t stride) {
  // The packed caller checks status uniformly before entering a shared-memory tile.
  if (!FrozenStride && atomicAdd(status, 0)) return;
  const auto quartets = FrozenStride == 1 ? 1 : group->quartets;
  for (std::size_t quartet = first; quartet < quartets; quartet += stride) {
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
                        integral[0], motion, result, group->exchange_fraction))
      atomicExch(status, 1);
    if constexpr (FrozenStride != 0) {
      for (unsigned incidence = 0; incidence < 8; ++incidence)
        projected[incidence * FrozenStride + quartet] = result.motion[incidence];
    } else {
      auto* output = projected + 41 * quartet;
      output[0] = result.energy;
      for (unsigned incidence = 0; incidence < 8; ++incidence) {
        output[1 + incidence] = result.fock[incidence];
        output[9 + incidence] = result.motion[incidence];
      }
    }
    for (unsigned coordinate = 0; coordinate < 12; ++coordinate) {
      double contraction[4];
      if (!generativeqc_nuclear_integral_ad::contract(integral[0], integral[1 + coordinate],
                                                      integral[13 + coordinate], result.weight,
                                                      result.delta_weight, 1, 0, contraction))
        atomicExch(status, 1);
      if constexpr (FrozenStride != 0) {
        projected[(8 + coordinate) * FrozenStride + quartet] = contraction[3];
      } else {
        projected[41 * quartet + 17 + coordinate] = contraction[2];
        projected[41 * quartet + 29 + coordinate] = contraction[3];
      }
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
                                double* projected, long long frozen_atom, int* status) {
  for (std::size_t index = first + blockIdx.x; index < first + count; index += gridDim.x) {
    const auto* group = groups + index;
    if (!aot::active_eri_centers(group->atoms, frozen_atom)) continue;
    project_values(shape, group, basis, direction, density, delta_density,
                   resident + group->resident_offset,
                   projected + (group->projected_offset - offset), status, threadIdx.x, blockDim.x);
  }
}

// Fold each small shell quartet in its projection CTA. The global scratch now
// holds one value per sparse owner entry, not a tensor with unconsumed primal
// fields. Both the local traversal and the final FP64 warp sum keep their order.
__global__ void project_frozen_batches(
    CudaWb97mvJkShape shape, const DeviceGroup* groups, std::size_t first, std::size_t count,
    std::size_t offset, const double* basis, const double* direction, const double* density,
    const double* delta_density, const double* resident, double* projected, long long frozen_atom,
    int* status) {
  __shared__ double fields[frozen_fields * max_sp_quartets];
  if (__syncthreads_or(threadIdx.x == 0 && atomicAdd(status, 0))) return;
  for (std::size_t index = first + blockIdx.x; index < first + count; index += gridDim.x) {
    const auto* group = groups + index;
    if (!aot::active_eri_centers(group->atoms, frozen_atom)) continue;
    project_values<max_sp_quartets>(shape, group, basis, direction, density, delta_density,
                                    resident + group->resident_offset, fields, status, threadIdx.x,
                                    blockDim.x);
    __syncthreads();
    auto* output = projected + (group->projected_offset - offset);
    if (threadIdx.x < 12) {
      const auto center = threadIdx.x / 3, axis = threadIdx.x % 3;
      double value{};
      for (unsigned quartet = 0; quartet < group->quartets; ++quartet)
        for (unsigned slot = center; slot < 4; ++slot)
          if (group->atoms[slot] == group->atoms[center])
            value += fields[(8 + 3 * slot + axis) * max_sp_quartets + quartet];
      output[threadIdx.x] = value;
    } else {
      constexpr unsigned pairs[4][2] = {{0, 1}, {2, 3}, {0, 2}, {1, 3}};
      auto pair = threadIdx.x - 12;
      const auto incidences = fock_incidence_count(group->kernel);
      for (unsigned incidence = 0; incidence < incidences; ++incidence) {
        const auto row = pairs[incidence][0], column = pairs[incidence][1];
        const auto width = group->shells[column].ao_count;
        const auto size = shape.spins * group->shells[row].ao_count * width;
        if (pair < size) {
          const auto spin = pair % shape.spins, local_pair = pair / shape.spins;
          unsigned rest[2], used{};
          std::size_t local[4]{};
          local[row] = local_pair / width;
          local[column] = local_pair % width;
          for (unsigned slot = 0; slot < 4; ++slot)
            if (slot != row && slot != column) rest[used++] = slot;
          const auto* motion = fields + (2 * incidence + spin) * max_sp_quartets;
          double value{};
          for (local[rest[0]] = 0; local[rest[0]] < group->shells[rest[0]].ao_count;
               ++local[rest[0]])
            for (local[rest[1]] = 0; local[rest[1]] < group->shells[rest[1]].ao_count;
                 ++local[rest[1]])
              value += motion[quartet_index(local, group)];
          output[threadIdx.x] = value;
          break;
        }
        pair -= size;
      }
    }
    if (index + gridDim.x < first + count) __syncthreads();
  }
}

// All-s windows have one public component per group, so no CTA communication
// is needed. Keep coincident-center folding identical to the s/p block path.
__global__ void project_single_frozen_batches(
    CudaWb97mvJkShape shape, const DeviceGroup* groups, std::size_t first, std::size_t count,
    std::size_t offset, const double* basis, const double* direction, const double* density,
    const double* delta_density, const double* resident, double* projected, long long frozen_atom,
    int* status) {
  if (atomicAdd(status, 0)) return;
  for (std::size_t item = blockIdx.x * blockDim.x + threadIdx.x; item < count;
       item += static_cast<std::size_t>(gridDim.x) * blockDim.x) {
    const auto* group = groups + first + item;
    if (!aot::active_eri_centers(group->atoms, frozen_atom)) continue;
    double fields[frozen_fields];
    project_values<1>(shape, group, basis, direction, density, delta_density,
                       resident + group->resident_offset, fields, status, 0, 1);
    auto* output = projected + (group->projected_offset - offset);
    for (unsigned center = 0; center < 4; ++center)
      for (unsigned axis = 0; axis < 3; ++axis) {
        double value{};
        for (unsigned slot = center; slot < 4; ++slot)
          if (group->atoms[slot] == group->atoms[center]) value += fields[8 + 3 * slot + axis];
        output[3 * center + axis] = value;
      }
    const auto incidences = fock_incidence_count(group->kernel);
    for (unsigned incidence = 0; incidence < incidences; ++incidence)
      for (unsigned spin = 0; spin < shape.spins; ++spin)
        output[12 + incidence * shape.spins + spin] = fields[2 * incidence + spin];
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

/** Keep the significant entries in owner and projection order. Each owner
 * uses its original range as a capacity, so warps need no global scan and
 * no atomic allocation. The energy owner is not part of a frozen column. */
__global__ void compact_frozen_owners(const DeviceGroup* groups, const std::size_t* offsets,
                                       const ReductionEntry* entries, std::size_t owners,
                                       long long atom, std::uint32_t* counts,
                                       std::uint32_t* values, const int* status) {
  if (*status) return;
  const unsigned lane = threadIdx.x % 32;
  for (std::size_t owner = (blockIdx.x * blockDim.x + threadIdx.x) / 32; owner < owners;
       owner += static_cast<std::size_t>(gridDim.x) * blockDim.x / 32) {
    const auto first = offsets[owner + 1], end = offsets[owner + 2];
    unsigned count = 0;
    for (auto begin = first; begin < end; begin += 32) {
      const auto index = begin + lane;
      ReductionEntry entry{};
      bool active = false;
      if (index < end) {
        entry = entries[index];
        active = aot::active_eri_centers(groups[entry.group].atoms, atom);
      }
      const auto mask = __ballot_sync(0xffffffffU, active);
      if (active) {
        const auto rank = __popc(mask & ((1U << lane) - 1));
        values[first + count + rank] = groups[entry.group].projected_offset + entry.packed_offset;
      }
      count += __popc(mask);
    }
    if (!lane) counts[owner] = count;
  }
}

__device__ std::size_t lower_projection_offset(const std::uint32_t* values, std::size_t begin,
                                                std::size_t end, std::size_t offset) {
  while (begin < end) {
    const auto middle = begin + (end - begin) / 2;
    if (values[middle] < offset)
      begin = middle + 1;
    else
      end = middle;
  }
  return begin;
}

/** Fold the active owner entries with the fixed FP64 warp tree. Offsets
 * select only the projected interval currently resident in scratch. */
__global__ void reduce_compact_frozen(const std::size_t* offsets, const std::uint32_t* counts,
                                      const std::uint32_t* values, const double* projected,
                                      std::size_t projected_begin, std::size_t projected_end,
                                      std::size_t panels, std::size_t owners,
                                      double* fock, double* hvp, double* anchor_fock,
                                      double* anchor_hvp, const int* status) {
  if (*status) return;
  const unsigned lane = threadIdx.x % 32;
  for (std::size_t owner = (blockIdx.x * blockDim.x + threadIdx.x) / 32; owner < owners;
       owner += static_cast<std::size_t>(gridDim.x) * blockDim.x / 32) {
    const auto first = offsets[owner + 1];
    const auto last = first + counts[owner];
    if (first == last) continue;
    const auto begin = projected_begin ? lower_projection_offset(values, first, last,
                                                                 projected_begin) : first;
    const auto end = projected_end == SIZE_MAX ? last
                     : lower_projection_offset(values, begin, last, projected_end);
    if (begin == end) continue;
    double value{};
    for (auto index = begin + lane; index < end; index += 32)
      value += projected[values[index] - projected_begin];
    for (unsigned stride = 16; stride; stride /= 2)
      value += __shfl_down_sync(0xffffffffU, value, stride);
    if (!lane) {
      if (owner < panels) {
        fock[owner] += value;
        if (anchor_fock) anchor_fock[owner] -= value;
      } else {
        hvp[owner - panels] += value;
        if (anchor_hvp) anchor_hvp[owner - panels] -= value;
      }
    }
  }
}

template <bool Packed = false>
__global__ void reduce_batches(CudaWb97mvJkShape shape, const DeviceGroup* groups,
                               const std::size_t* offsets, const ReductionEntry* entries,
                               const double* projected, std::size_t first_group,
                               std::size_t group_count, std::size_t projected_offset,
                               std::size_t count, double* candidate, long long frozen_atom,
                               int* status, double* frozen_fock, double* frozen_hvp,
                               double* anchor_fock, double* anchor_hvp) {
  if (atomicAdd(status, 0)) return;
  const auto panels = shape.spins * shape.nao * shape.nao;
  const auto vector_begin = 1 + 2 * panels;
  constexpr unsigned pairs[4][2] = {{0, 1}, {2, 3}, {0, 2}, {1, 3}};
  const auto lane = threadIdx.x % 32;
  const auto owners = Packed ? panels + 3 * shape.atoms : count;
  for (std::size_t owner_index = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
       owner_index < owners;
       owner_index += static_cast<std::size_t>(gridDim.x) * blockDim.x / 32) {
    const auto target = Packed ? owner_index + 1 + panels +
                                                   (owner_index >= panels ? 3 * shape.atoms : 0)
                               : owner_index;
    if (frozen_atom >= 0 &&
        (target <= panels || (target >= vector_begin && target < vector_begin + 3 * shape.atoms)))
      continue;
    auto* destination = frozen_fock ? (target < vector_begin
                                           ? frozen_fock + (target - 1 - panels)
                                           : frozen_hvp + (target - vector_begin - 3 * shape.atoms))
                                    : candidate + target;
    double total = *destination;
    // One warp owns an output. Public arbitrary-direction actions keep the
    // original sparse-list order. Dense unit columns use a fixed FP64 lane
    // reduction; no atomics or race-dependent sums enter the column.
    double lane_total{};
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
        if (!aot::active_eri_centers(group->atoms, frozen_atom)) {
          // Inactive groups have no projected data in this column.
        } else if constexpr (Packed) {
          value = values[entry.packed_offset];
        } else if (!target) {
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
      if (frozen_atom >= 0) {
        lane_total += value;
      } else {
        const auto active = static_cast<unsigned>(end - begin < 32 ? end - begin : 32);
        for (unsigned source = 0; source < active; ++source) {
          const auto contribution = __shfl_sync(0xffffffffU, value, source);
          if (!lane) total += contribution;
        }
      }
    }
    if (frozen_atom >= 0) {
      for (unsigned stride = 16; stride; stride /= 2)
        lane_total += __shfl_down_sync(0xffffffffU, lane_total, stride);
      total += lane_total;
    }
    if (!lane) {
      *destination = total;
      if (anchor_fock) {
        auto* anchor = target < vector_begin
                           ? anchor_fock + (target - 1 - panels)
                           : anchor_hvp + (target - vector_begin - 3 * shape.atoms);
        *anchor -= lane_total;
      }
    }
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
  for (unsigned incidence = 0; incidence < fock_incidence_count(group->kernel); ++incidence) {
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
    const auto& group = schedule.groups[batch.group].device;
    if (plan.paired_radial_cache) {
      input.output = reinterpret_cast<double*>(base + plan.resident) + group.resident_offset;
    } else {
      input.output = reinterpret_cast<double*>(base + plan.batch_raw) + batch.raw;
    }
    input.coordinates = basis;
    std::copy_n(group.atoms, 4, input.atoms);
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
// Both dense traversal orders use the same primitive, projection and FP64 fold.
cudaError_t run_batched(const Schedule& selected, CudaWb97mvJkShape shape, char* base,
                       const double* basis, const double* direction, const double* density,
                       const double* delta_density, double* candidate, long long atom,
                       bool reuse, int* status, cudaStream_t stream, CudaWb97mvJkWork& work,
                       double* frozen_fock = nullptr, double* frozen_hvp = nullptr,
                       double* anchor_fock = nullptr, double* anchor_hvp = nullptr,
                       bool first_atom_axis = true,
                       const aot::FactorizedPairGeometry* prepared = nullptr,
                       const ActiveComponentAtom* active_atom = nullptr,
                       const aot::FactorizedActiveComponent* active_tasks = nullptr) {
  auto error = cudaSuccess;
  const auto& plan = selected.plan;
  const auto* groups = reinterpret_cast<const DeviceGroup*>(base + plan.groups);
  const auto* tiles = reinterpret_cast<const DeviceTile*>(base + plan.tiles);
  auto* batches = reinterpret_cast<aot::FactorizedIntegralBatch*>(base + plan.batch_inputs);
  auto* resident = reinterpret_cast<double*>(base + plan.resident);
  const auto* offsets = reinterpret_cast<const std::size_t*>(base + plan.reduction_offsets);
  const auto* entries = reinterpret_cast<const ReductionEntry*>(base + plan.reduction_entries);
  const auto packed = atom >= 0 && plan.packed_frozen_projection;
  const auto compact = packed && frozen_fock && plan.frozen_value_offsets;
  auto* counts = reinterpret_cast<std::uint32_t*>(base + plan.frozen_entry_counts);
  auto* values = reinterpret_cast<std::uint32_t*>(base + plan.frozen_value_offsets);
  const auto owners = plan.panels + 3 * shape.atoms;
  const auto packed_blocks = std::min<std::size_t>((owners + 3) / 4, 65535);
  if (compact && first_atom_axis) {
    compact_frozen_owners<<<packed_blocks, 128, 0, stream>>>(groups, offsets, entries, owners,
                                                            atom, counts, values, status);
    if ((error = cudaGetLastError()) != cudaSuccess) return error;
  }
  if (active_atom && !reuse) {
    const auto* records = reinterpret_cast<const double*>(base + plan.batch_records);
    for (unsigned degree = 0; degree < active_atom->size(); ++degree) {
      const auto& span = (*active_atom)[degree];
      for (std::size_t offset = 0; offset < span.count;) {
        const auto count = std::min<std::size_t>(span.count - offset, 65535);
        error = static_cast<cudaError_t>(aot::launch_factorized_active_components(
            degree, active_tasks + span.offset + offset, count, span.max_radials, batches,
            direction, status,
            stream, prepared, records));
        if (error != cudaSuccess) return error;
        offset += count;
        ++work.integral_launches;
      }
      work.component_evaluations += span.evaluations;
      work.prepared_geometry_components += span.evaluations;
      work.indexed_component_blocks += span.count;
      work.reduction_terms += span.reduction_terms;
    }
  } else if (!reuse) {
    for (const auto& run : selected.batch_runs) {
      const auto& tile = selected.tiles[run.tile];
      auto first = run.first + run.count;
      auto last = run.first;
      std::uint64_t components{}, radial_terms{};
      for (std::size_t index = run.first; index < run.first + run.count; ++index) {
        const auto& batch = selected.batches[index];
        if (aot::active_eri_centers(selected.groups[batch.group].device.atoms, atom)) {
          first = std::min(first, index);
          last = index + 1;
          components += batch.count * tile.device.count;
          if (plan.paired_radial_cache)
            radial_terms += (batch.count > 64 ? 50 : 25) * tile.device.count;
        }
      }
      // The paired kernel writes independent cache slots. Inactive groups
      // never reach projection, so their leading/trailing blocks (and empty
      // runs) need not launch. Keep interior holes to avoid extra launches.
      if (plan.paired_radial_cache) {
        if (first >= last) continue;
        error = static_cast<cudaError_t>(aot::launch_factorized_radial_pair(
            tile.program->signature.angular.data(), tile.program->components.data(),
            tile.program->components.size(), batches + first, last - first, direction, atom,
            status, stream, prepared,
            prepared ? reinterpret_cast<const double*>(base + plan.batch_records) : nullptr));
      } else {
        error = static_cast<cudaError_t>(tile.program->integral_batch(
            tile.omega, batches + run.first, run.count, direction, atom, status, stream));
      }
      if (error != cudaSuccess) break;
      ++work.integral_launches;
      work.component_evaluations += components;
      if (prepared) work.prepared_geometry_components += components;
      work.reduction_terms += radial_terms;
    }
    if (error == cudaSuccess && !plan.paired_radial_cache) {
      accumulate_batches<<<std::min<std::size_t>(selected.batch_reductions.size(), 65535), 128,
                           0, stream>>>(
          groups, tiles, batches,
          reinterpret_cast<const BatchReduction*>(base + plan.batch_reductions),
          selected.batch_reductions.size(), resident, atom, status);
      error = cudaGetLastError();
      for (const auto& batch : selected.batches)
        if (aot::active_eri_centers(selected.groups[batch.group].device.atoms, atom))
          work.reduction_terms += 25 * selected.tiles[batch.tile].device.count;
    }
  }
  for (const auto& run : selected.projection_runs) {
    if (error != cudaSuccess) break;
    auto* projected = reinterpret_cast<double*>(base + plan.batch_projected);
    const auto projection_blocks = std::min<std::size_t>(run.count, 65535);
    if (packed && plan.single_cartesian_component)
      project_single_frozen_batches<<<blocks(run.count), 128, 0, stream>>>(
          shape, groups, run.first, run.count, run.offset, basis, direction, density,
          delta_density, resident, projected, atom, status);
    else if (packed)
      project_frozen_batches<<<projection_blocks, 128, 0, stream>>>(
          shape, groups, run.first, run.count, run.offset, basis, direction, density,
          delta_density, resident, projected, atom, status);
    else
      project_batches<<<projection_blocks, 128, 0, stream>>>(
          shape, groups, run.first, run.count, run.offset, basis, direction, density,
          delta_density, resident, projected, atom, status);
    error = cudaGetLastError();
    ++work.projection_launches;
    if (error != cudaSuccess) break;
    if (compact) {
      const auto next = run.first + run.count;
      const auto projected_end = next == selected.groups.size() ? SIZE_MAX
                                  : selected.groups[next].device.projected_offset;
      reduce_compact_frozen<<<packed_blocks, 128, 0, stream>>>(
          offsets, counts, values, projected, run.offset, projected_end, plan.panels, owners,
          frozen_fock, frozen_hvp, anchor_fock, anchor_hvp, status);
    } else if (packed)
      reduce_batches<true><<<packed_blocks, 128, 0, stream>>>(
          shape, groups, offsets, entries, projected, run.first, run.count, run.offset,
          plan.candidate_count, candidate, atom, status, frozen_fock, frozen_hvp, anchor_fock,
          anchor_hvp);
    else
      reduce_batches<><<<std::min<std::size_t>((plan.candidate_count + 3) / 4, 65535), 128, 0,
                          stream>>>(
          shape, groups, offsets, entries, projected, run.first, run.count, run.offset,
          plan.candidate_count, candidate, atom, status, frozen_fock, frozen_hvp, anchor_fock,
          anchor_hvp);
    error = cudaGetLastError();
    ++work.reduction_launches;
    for (std::size_t index = run.first; index < run.first + run.count; ++index) {
      const auto& group = selected.groups[index].device;
      if (aot::active_eri_centers(group.atoms, atom)) {
        work.projection_terms_bound += projection_bound(group);
        work.reduction_terms += group.quartets * quartet_reduction_bound(group, shape.spins);
      }
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
  std::vector<Shell> replay_shells;
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

struct CudaWb97mvNuclearJk::Plan::Impl {
  Schedule selected;
  CudaWb97mvJkShape shape;
  CudaWb97mvJkOptions options;
  std::vector<Shell> shells;
  std::size_t omitted_sources{};
  double omitted_bound{};
};

CudaWb97mvNuclearJk::Plan::Plan() = default;
CudaWb97mvNuclearJk::Plan::~Plan() = default;
std::size_t CudaWb97mvNuclearJk::Plan::bytes() const noexcept {
  return impl_ ? impl_->selected.plan.bytes : 0;
}
cudaError_t CudaWb97mvNuclearJk::Plan::build(
    const CudaWb97mvJkShape& shape, std::span<const Shell> shells,
    const CudaWb97mvJkOptions& options, std::string* detail,
    const integrals::CoulombDerivativeOmissions* omissions,
    CudaWb97mvJkAdmissionFailure* failure) noexcept {
  impl_.reset();
  if (failure) *failure = CudaWb97mvJkAdmissionFailure::other;
  try {
    Schedule selected;
    const auto error = schedule(shape, shells, options, selected, omissions);
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
    if (failure && error == cudaErrorInvalidValue) *failure = selected.admission_failure;
    if (error == cudaSuccess) {
      impl_ = std::make_unique<Impl>(Impl{std::move(selected), shape, options,
                                        {shells.begin(), shells.end()},
                                        omissions ? omissions->sources.size() : 0,
                                        omissions ? omissions->bound : 0});
    }
    return error;
  } catch (...) {
    return cudaErrorMemoryAllocation;
  }
}
cudaError_t cuda_wb97mv_jk_workspace(const CudaWb97mvJkShape& shape, std::span<const Shell> shells,
                                     const CudaWb97mvJkOptions& options, std::size_t& bytes,
                                     std::string* detail,
                                     const integrals::CoulombDerivativeOmissions* omissions,
                                     CudaWb97mvJkAdmissionFailure* failure) noexcept {
  CudaWb97mvNuclearJk::Plan plan;
  const auto error = plan.build(shape, shells, options, detail, omissions, failure);
  bytes = plan.bytes();
  return error;
}
CudaWb97mvNuclearJk::CudaWb97mvNuclearJk() : impl_(std::make_unique<Impl>()) {}
CudaWb97mvNuclearJk::~CudaWb97mvNuclearJk() = default;
const CudaWb97mvJkWork& CudaWb97mvNuclearJk::work() const noexcept { return impl_->work; }

cudaError_t CudaWb97mvNuclearJk::prepare(const CudaWb97mvJkShape& shape,
                                         std::span<const Shell> shells,
                                         const CudaWb97mvJkOptions& options,
                                         const CudaWb97mvJkReference& reference, void* workspace,
                                         std::size_t workspace_bytes,
                                         const integrals::CoulombDerivativeOmissions* omissions) noexcept {
  Plan plan;
  const auto error = plan.build(shape, shells, options, nullptr, omissions);
  if (error == cudaSuccess) return prepare(plan, reference, workspace, workspace_bytes);
  impl_->ready = false;
  impl_->resident_valid = false;
  return error;
}

cudaError_t CudaWb97mvNuclearJk::prepare(Plan& admitted, const CudaWb97mvJkReference& reference,
                                         void* workspace, std::size_t workspace_bytes) noexcept {
  impl_->ready = false;
  impl_->resident_valid = false;
  if (!admitted.impl_) return cudaErrorInvalidValue;
  int device{-1};
  auto error = cudaGetDevice(&device);
  if (error != cudaSuccess) return error;
  if (device != reference.device_id) return cudaErrorInvalidResourceHandle;
  cudaStreamCaptureStatus capture{};
  if ((error = cudaStreamIsCapturing(reference.stream, &capture)) != cudaSuccess) return error;
  if (capture != cudaStreamCaptureStatusNone) return cudaErrorStreamCaptureUnsupported;
  try {
    auto& selected = admitted.impl_->selected;
    const auto shape = admitted.impl_->shape;
    const auto& plan = selected.plan;
    if (workspace_bytes < plan.bytes || !reference.stream || !reference.authenticate ||
        !reference.owner ||
        !disjoint(workspace, plan.bytes, reference.basis, plan.basis_elements * sizeof(double)))
      return cudaErrorInvalidValue;
    for (const auto* pointer : {reference.basis, static_cast<const double*>(workspace)})
      if ((error = device_buffer(pointer, reference.device_id)) != cudaSuccess) return error;
    auto bound = std::move(admitted.impl_);
    const std::span<const Shell> shells = bound->shells;
    auto& state = *impl_;
    state.shape = shape;
    state.options = bound->options;
    state.reference = reference;
    state.schedule = std::move(selected);
    state.workspace = workspace;
    state.ready = true;
    if ((error = state.authenticate(reference.stream)) != cudaSuccess) {
      state.ready = false;
      return error;
    }
    const bool replay = state.schedule.replay_windows != 0;
    if (!replay && state.schedule.windows.empty())
      stage_schedule(state.schedule, workspace, reference.basis);
    else
      for (auto& window : state.schedule.windows)
        stage_schedule(window, workspace, reference.basis);
    const auto& layout = state.schedule.plan;
    const auto stream = reference.stream;
    auto* base = static_cast<char*>(workspace);
    auto* status = reinterpret_cast<int*>(base + layout.status);
    auto* device_shells = reinterpret_cast<Shell*>(base + layout.shells);
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
    if (error == cudaSuccess && !replay &&
        (state.schedule.windows.empty() || layout.retained_window_source_bytes)) {
      if (state.schedule.windows.empty())
        error = upload_schedule(state.schedule, shape, reference.basis, workspace, status, stream);
      else
        for (const auto& window : state.schedule.windows) {
          error = upload_schedule(window, shape, reference.basis, workspace, status, stream);
          if (error != cudaSuccess) break;
        }
      error = error == cudaSuccess ? gate(status, stream) : drain(error, stream);
    }
    if (error == cudaSuccess) error = state.authenticate(stream);
    state.ready = error == cudaSuccess;
    if (state.ready)
      state.work = {layout.bytes,
                    state.schedule.ordered_groups ? state.schedule.ordered_groups
                                                  : state.schedule.groups.size(),
                    layout.radial_records,
                    layout.components,
                    layout.projections,
                    layout.reductions};
    if (state.ready) state.work.resident_integral_bytes = layout.resident_bytes;
    if (state.ready) state.work.batched_workspace_bytes = layout.batch_bytes;
    if (state.ready) state.work.retained_window_source_bytes = layout.retained_window_source_bytes;
    if (state.ready)
      state.work.streaming_tiles = state.schedule.windows.size() + state.schedule.replay_windows;
    if (state.ready) state.work.source_groups = state.schedule.source_groups;
    if (state.ready) {
      state.work.omitted_frozen_source_groups = bound->omitted_sources;
      state.work.omitted_frozen_derivative_bound = bound->omitted_bound;
      state.replay_shells = replay ? std::move(bound->shells) : std::vector<Shell>{};
    }
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
  return apply_impl(direction, density, delta_density, output, nullptr, stream, SIZE_MAX);
}

cudaError_t CudaWb97mvNuclearJk::enqueue_private_fresh(const double* direction,
                                                       const double* density,
                                                       const double* delta_density,
                                                       const CudaWb97mvJkOutput& output,
                                                       int* transaction_status, cudaStream_t stream,
                                                       std::size_t frozen_column) noexcept {
  if (!transaction_status) return cudaErrorInvalidValue;
  return apply_impl(direction, density, delta_density, output, transaction_status, stream,
                    frozen_column);
}

cudaError_t CudaWb97mvNuclearJk::enqueue_private_frozen_dense(
    double* direction, const double* density, const double* zero_delta_density, double* table,
    int* transaction_status, cudaStream_t stream) noexcept {
  auto& state = *impl_;
  auto error = state.authenticate(stream);
  if (error != cudaSuccess) return error;
  const auto& plan = state.schedule.plan;
  const bool replay = state.schedule.replay_windows != 0;
  if (!state.options.prefer_fresh_direction_batches ||
      (!replay && state.schedule.windows.empty()) ||
      plan.retained_window_source_bytes)
    return cudaErrorNotSupported;
  const auto& shape = state.shape;
  const auto coordinates = 3 * shape.atoms;
  const auto& reference = state.reference;
  std::size_t table_values = plan.panels;
  std::size_t table_bytes{};
  if (!add(table_values, coordinates) || !multiply(table_values, coordinates, table_values) ||
      !multiply(table_values, sizeof(double), table_bytes) ||
      coordinates > UINT64_MAX - state.work.attempted_actions ||
      plan.components > (UINT64_MAX - state.work.component_evaluations) / coordinates ||
      plan.components > (UINT64_MAX - state.work.prepared_geometry_components) / coordinates ||
      plan.components > (UINT64_MAX - state.work.indexed_component_blocks) / coordinates ||
      plan.radial_records > UINT64_MAX - state.work.prepared_geometry_records_bound ||
      plan.projections > (UINT64_MAX - state.work.projection_terms_bound) / coordinates ||
      plan.reductions > (UINT64_MAX - state.work.reduction_terms) / coordinates)
    return cudaErrorInvalidValue;
  const void* buffers[]{direction, density, zero_delta_density, table, transaction_status,
                        reference.basis};
  const std::size_t sizes[]{coordinates * sizeof(double), plan.panels * sizeof(double),
                            plan.panels * sizeof(double), table_bytes, sizeof(int),
                            plan.basis_elements * sizeof(double)};
  for (unsigned index = 0; index < std::size(buffers); ++index) {
    if ((error = device_buffer(buffers[index], reference.device_id)) != cudaSuccess) return error;
    if (!disjoint(state.workspace, plan.bytes, buffers[index], sizes[index]))
      return cudaErrorInvalidValue;
    for (unsigned previous = 0; previous < index; ++previous)
      if (!disjoint(buffers[index], sizes[index], buffers[previous], sizes[previous]))
        return cudaErrorInvalidValue;
  }
  state.resident_valid = false;
  state.work.attempted_actions += coordinates;
  state.work.integral_source_builds += coordinates;
  auto* base = static_cast<char*>(state.workspace);
  auto* status = reinterpret_cast<int*>(base + plan.status);
  auto* hvp = table + coordinates * plan.panels;
  error = cudaMemsetAsync(status, 0, 2 * sizeof(int), stream);
  if (error == cudaSuccess) error = cudaMemsetAsync(table, 0, table_bytes, stream);
  if (error == cudaSuccess) {
    frozen_unit<<<blocks(coordinates), 128, 0, stream>>>(direction, coordinates, 0);
    error = cudaGetLastError();
  }
  if (error == cudaSuccess) {
    validate_frozen_column<<<blocks(std::max(coordinates, plan.panels)), 128, 0, stream>>>(
        direction, coordinates, zero_delta_density, plan.panels, 0, status);
    error = cudaGetLastError();
  }
  if (error == cudaSuccess) {
    symmetric<<<blocks(plan.panels), 128, 0, stream>>>(
        density, shape.nao, shape.spins, state.options.symmetry_absolute_tolerance,
        state.options.symmetry_relative_tolerance, status);
    error = cudaGetLastError();
  }
  if (error == cudaSuccess && state.work.omitted_frozen_source_groups) {
    screen_density_bound<<<blocks(plan.panels), 128, 0, stream>>>(density, plan.panels, status);
    error = cudaGetLastError();
  }
  // Stage each window once. A common-atom window uses relative coordinates:
  // simultaneous translation leaves every ERI (and its center gradient)
  // invariant. Fold the omitted column from the evaluated increments, not a
  // difference of already rounded column prefixes. No Hessian is symmetrized.
  const auto run_window = [&](const Schedule& window, cudaEvent_t upload_complete) {
    if (window.frozen_atoms.empty()) return cudaSuccess;
    auto error = upload_schedule(window, shape, reference.basis, state.workspace, status, stream);
    auto* tasks = reinterpret_cast<aot::FactorizedActiveComponent*>(base + plan.active_components);
    const bool indexed = plan.active_components && !window.active_tasks.empty();
    if (error == cudaSuccess && indexed)
      error = cudaMemcpyAsync(tasks, window.active_tasks.data(),
                              window.active_tasks.size() * sizeof(*tasks), cudaMemcpyHostToDevice,
                              stream);
    if (error == cudaSuccess && upload_complete) error = cudaEventRecord(upload_complete, stream);
    aot::FactorizedPairGeometry* prepared = nullptr;
    if (error == cudaSuccess && window.plan.paired_radial_cache && window.frozen_atoms.size() >= 2 &&
        window.plan.window_radials <= plan.prepared_count) {
      prepared = reinterpret_cast<aot::FactorizedPairGeometry*>(base + plan.prepared_geometry);
      prepare_batch_geometry<<<std::min<std::size_t>(window.groups.size(), 65535), 128, 0, stream>>>(
          reinterpret_cast<const DeviceGroup*>(base + window.plan.groups), window.groups.size(),
          reference.basis, reinterpret_cast<const double*>(base + window.plan.batch_records),
          prepared, status);
      error = cudaGetLastError();
      if (error == cudaSuccess) state.work.prepared_geometry_records_bound += window.plan.window_radials;
    }
    for (std::size_t index = 0; index < window.frozen_atoms.size(); ++index) {
      const auto atom = window.frozen_atoms[index];
      const auto* active = indexed ? &window.active_atoms[index] : nullptr;
      for (std::size_t axis = 0; axis < 3 && error == cudaSuccess; ++axis) {
        const auto column = 3 * atom + axis;
        const auto has_anchor = window.translation_anchor != SIZE_MAX;
        const auto anchor = has_anchor ? 3 * window.translation_anchor + axis : 0;
        frozen_unit<<<blocks(coordinates), 128, 0, stream>>>(direction, coordinates, column);
        error = cudaGetLastError();
        if (error != cudaSuccess) break;
        error = run_batched(window, shape, base, reference.basis, direction, density,
                            zero_delta_density, nullptr, static_cast<long long>(atom), false,
                            status, stream, state.work, table + column * plan.panels,
                            hvp + column * coordinates,
                            has_anchor ? table + anchor * plan.panels : nullptr,
                            has_anchor ? hvp + anchor * coordinates : nullptr, axis == 0, prepared,
                            active, indexed ? tasks : nullptr);
      }
    }
    return error;
  };
  if (!replay) {
    for (const auto& window : state.schedule.windows) {
      if (error != cudaSuccess) break;
      error = run_window(window, nullptr);
    }
  } else if (error == cudaSuccess) {
    CudaWb97mvJkOptions tile_options;
    if (!canonical_window_options(shape, state.options, plan.panels, tile_options))
      return drain(cudaErrorInvalidValue, stream);
    cudaEvent_t upload_complete{};
    error = cudaEventCreateWithFlags(&upload_complete, cudaEventDisableTiming);
    if (error != cudaSuccess) return drain(error, stream);
    Schedule inflight;
    bool copying{};
    try {
      std::size_t visited{};
      const char* refusal{};
      error = visit_canonical_windows(
          shape, state.replay_shells, tile_options, state.schedule.replay_source_words, refusal,
          [&](Schedule&& window) {
            if (++visited > state.schedule.replay_windows || window.plan.bytes > plan.candidates)
              return cudaErrorInvalidValue;
            if (window.frozen_atoms.empty()) return cudaSuccess;
            // Keep the previous metadata alive while planning the next window.
            // The event covers its uploads, not its complete Cartesian sweep.
            if (copying) {
              const auto ready = cudaEventSynchronize(upload_complete);
              if (ready != cudaSuccess) return ready;
              copying = false;
            }
            bind_window_tail(window, plan);
            if (plan.active_component_capacity && window.plan.paired_radial_cache &&
                window.frozen_atoms.size() >= 2 && window.plan.window_radials <= plan.prepared_count) {
              // The upload event below protects these host indices along
              // with the other replay metadata. Allocation is optional.
              try {
                build_active_components(window, plan.active_component_capacity);
              } catch (const std::bad_alloc&) {
                // Leave the ordinary per-tile dispatch in place.
              }
            }
            stage_schedule(window, state.workspace, reference.basis);
            inflight = std::move(window);
            copying = true;
            return run_window(inflight, upload_complete);
          });
      if (error == cudaSuccess && visited != state.schedule.replay_windows)
        error = cudaErrorInvalidValue;
    } catch (const std::bad_alloc&) {
      error = cudaErrorMemoryAllocation;
    } catch (...) {
      error = cudaErrorUnknown;
    }
    if (error == cudaSuccess && copying) error = cudaEventSynchronize(upload_complete);
    const auto destroy = cudaEventDestroy(upload_complete);
    if (error != cudaSuccess || destroy != cudaSuccess)
      return drain(error == cudaSuccess ? destroy : error, stream);
  }
  for (std::size_t column = 0; column < coordinates && error == cudaSuccess; ++column) {
    symmetric<<<blocks(plan.panels), 128, 0, stream>>>(
        table + column * plan.panels, shape.nao, shape.spins,
        state.options.symmetry_absolute_tolerance, state.options.symmetry_relative_tolerance,
        status);
    error = cudaGetLastError();
  }
  if (error == cudaSuccess) {
    finite_values<<<blocks(table_values), 128, 0, stream>>>(table, table_values, status);
    error = cudaGetLastError();
  }
  if (error == cudaSuccess) error = state.authenticate(stream);
  if (error == cudaSuccess) {
    accumulate_failure<<<1, 1, 0, stream>>>(status, transaction_status);
    error = cudaGetLastError();
  }
  if (error == cudaSuccess) state.work.completed_actions += coordinates;
  return error;
}

cudaError_t CudaWb97mvNuclearJk::apply_impl(const double* direction, const double* density,
                                            const double* delta_density,
                                            const CudaWb97mvJkOutput& output,
                                            int* transaction_status, cudaStream_t stream,
                                            std::size_t frozen_column) noexcept {
  auto& state = *impl_;
  if (state.work.attempted_actions == UINT64_MAX) return cudaErrorInvalidValue;
  ++state.work.attempted_actions;
  auto error = state.authenticate(stream);
  if (error != cudaSuccess) return error;
  // This schedule omits derivative sources only. It cannot publish values,
  // an arbitrary signed action, or a directional fallback column.
  if (state.work.omitted_frozen_source_groups) return cudaErrorNotSupported;
  const auto& plan = state.schedule.plan;
  const auto& shape = state.shape;
  const auto& reference = state.reference;
  const bool frozen = frozen_column != SIZE_MAX;
  if (frozen && frozen_column >= 3 * shape.atoms) return cudaErrorInvalidValue;
  const double* inputs[] = {reference.basis, direction, density, delta_density};
  const std::size_t input_counts[] = {plan.basis_elements, 3 * shape.atoms, plan.panels,
                                      plan.panels};
  double* outputs[] = {output.energy, output.fock, output.fock_direction, output.gradient,
                       output.hvp};
  const std::size_t output_counts[] = {1, plan.panels, plan.panels, 3 * shape.atoms,
                                       3 * shape.atoms};
  if (transaction_status) {
    if ((error = device_buffer(transaction_status, reference.device_id)) != cudaSuccess)
      return error;
    if (!disjoint(state.workspace, plan.bytes, transaction_status, sizeof(int)))
      return cudaErrorInvalidValue;
  }
  for (unsigned index = 0; index < 4; ++index) {
    if (!disjoint(state.workspace, plan.bytes, inputs[index], input_counts[index] * sizeof(double)))
      return cudaErrorInvalidValue;
    if (transaction_status && !disjoint(inputs[index], input_counts[index] * sizeof(double),
                                        transaction_status, sizeof(int)))
      return cudaErrorInvalidValue;
    if ((error = device_buffer(inputs[index], reference.device_id)) != cudaSuccess) return error;
  }
  for (unsigned index = 0; index < 5; ++index) {
    if (!disjoint(state.workspace, plan.bytes, outputs[index],
                  output_counts[index] * sizeof(double)))
      return cudaErrorInvalidValue;
    if (transaction_status && !disjoint(outputs[index], output_counts[index] * sizeof(double),
                                        transaction_status, sizeof(int)))
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
    if (frozen)
      validate_frozen_column<<<blocks(std::max(3 * shape.atoms, plan.panels)), 128, 0, stream>>>(
          direction, 3 * shape.atoms, delta_density, plan.panels, frozen_column, status);
    else
      finite_values<<<blocks(3 * shape.atoms), 128, 0, stream>>>(direction, 3 * shape.atoms,
                                                                 status);
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
  if (error == cudaSuccess && state.resident_valid && !transaction_status) {
    compare_direction<<<1, 128, 0, stream>>>(
        direction, reinterpret_cast<const double*>(base + plan.resident_direction), 3 * shape.atoms,
        status + 1);
    error = cudaGetLastError();
  }
  if (!transaction_status)
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
    if (!state.schedule.windows.empty() && !state.schedule.plan.retained_window_source_bytes &&
        error == cudaSuccess)
      error = upload_schedule(selected, shape, reference.basis, state.workspace, status, stream);
    if (plan.batch_bytes && error == cudaSuccess) {
      const auto atom = frozen ? static_cast<long long>(frozen_column / 3) : -1LL;
      error = run_batched(selected, shape, base, reference.basis, direction, density,
                          delta_density, candidate, atom, reuse, status, stream, state.work);
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
            error =
                aot::launch_integral_at_omega(*tile.program, tile.omega, records, count,
                                              groups[index].atoms, direction, raw, status, stream);
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
        state.work.reduction_terms +=
            group.device.quartets * quartet_reduction_bound(group.device, shape.spins);
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
  if (!transaction_status)
    error = error == cudaSuccess ? gate(status, stream) : drain(error, stream);
  if (error == cudaSuccess && plan.resident_bytes && !transaction_status)
    error = cudaMemcpyAsync(base + plan.resident_direction, direction,
                            3 * shape.atoms * sizeof(double), cudaMemcpyDeviceToDevice, stream);
  if (error == cudaSuccess) error = state.authenticate(stream);
  if (error != cudaSuccess) return transaction_status ? error : drain(error, stream);
  if (transaction_status) {
    accumulate_failure<<<1, 1, 0, stream>>>(status, transaction_status);
    error = cudaGetLastError();
  }
  std::size_t offset{};
  for (unsigned index = 0; index < 5 && error == cudaSuccess; ++index) {
    if (frozen && index != 2 && index != 4) {
      offset += output_counts[index];
      continue;
    }
    if (transaction_status) {
      publish_private_panel<<<blocks(output_counts[index]), 128, 0, stream>>>(
          candidate + offset, outputs[index], output_counts[index], status);
      error = cudaGetLastError();
    } else {
      error =
          cudaMemcpyAsync(outputs[index], candidate + offset, output_counts[index] * sizeof(double),
                          cudaMemcpyDeviceToDevice, stream);
    }
    offset += output_counts[index];
  }
  if (!transaction_status) error = drain(error, stream);
  if (error == cudaSuccess) {
    ++state.work.completed_actions;
    state.resident_valid = !transaction_status && plan.resident_bytes != 0;
  }
  return error;
}
}  // namespace generativeqc::dft
