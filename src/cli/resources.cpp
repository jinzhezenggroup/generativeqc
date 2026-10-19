#include "cli/resources.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <vector>

#include "cli/json.hpp"
#include "cli/native_basis.hpp"
#include "runtime/resource_planning.hpp"
#include "scf/hf_cuda_resources.hpp"
#include "scf/hf_df_resources.hpp"
#include "scf/hf_host_resources.hpp"
#include "scf/incremental_direct_jk_control.hpp"
#include "scf/ks_device_resources.hpp"
#include "scf/ks_host_resources.hpp"

extern "C" int generativeqc_resource_small_hf_cuda_v2(std::size_t, std::size_t, std::size_t,
                                                      std::size_t, const std::uint8_t*,
                                                      const std::size_t*, std::size_t, std::size_t,
                                                      int, double, double, std::uint64_t*,
                                                      std::size_t);
extern "C" int generativeqc_resource_ks_cuda_v1(std::size_t, std::size_t, std::size_t, std::size_t,
                                                std::size_t, std::size_t, std::size_t, std::size_t,
                                                std::size_t, std::uint64_t*, std::size_t);
extern "C" int generativeqc_resource_ks_matrix_provider_cuda_v1(std::size_t, std::uint64_t*);
extern "C" int generativeqc_resource_quadrature_cuda_v1(std::size_t, std::size_t, std::uint64_t*);
extern "C" int generativeqc_resource_ecp_workspace_v1(std::size_t, std::size_t, std::size_t,
                                                      std::size_t, std::size_t, int,
                                                      std::uint64_t*);

namespace generativeqc::cli {
namespace {
using Bytes = std::uint64_t;
namespace resource = generativeqc::runtime::resources;
constexpr Bytes maximum_bytes = std::numeric_limits<std::int64_t>::max();

Bytes sum(std::initializer_list<Bytes> values) {
  Bytes result = 0;
  for (const auto value : values) {
    if (value > maximum_bytes - result) throw UsageError("resource byte count overflows int64");
    result += value;
  }
  return result;
}
Bytes product(std::initializer_list<Bytes> values) {
  Bytes result = 1;
  for (const auto value : values) {
    if (value && result > maximum_bytes / value)
      throw UsageError("resource byte count overflows int64");
    result *= value;
  }
  return result;
}

resource::Candidate inventory_candidate(Bytes retained_host, Bytes workspace_host,
                                        Bytes retained_device, Bytes workspace_device,
                                        Bytes cost = 0, Bytes lexical_rank = 0) {
  return {cost,
          lexical_rank,
          {{retained_host, resource::kPageableSpace, 0, 0, true},
           {workspace_host, resource::kPageableSpace, 0, 0, false},
           {retained_device, resource::kFirstDeviceSpace, 0, 0, true},
           {workspace_device, resource::kFirstDeviceSpace, 0, 0, false}}};
}

resource::Composition compose_inventory(const RunOptions& options,
                                        std::vector<resource::Candidate> candidates) {
  std::vector<resource::Limit> limits;
  if (options.host_bytes)
    limits.push_back({resource::kHostSpace, *options.host_bytes, options.host_reserve_bytes});
  if (options.device_bytes)
    limits.push_back({resource::kDeviceSpace, *options.device_bytes, options.device_reserve_bytes});
  if (options.pinned_host_bytes)
    limits.push_back({resource::kPinnedSpace, *options.pinned_host_bytes, 0});
  const std::array<resource::Request, 1> requests{{{std::move(candidates)}}};
  try {
    return resource::compose(requests, limits, 5, options.headroom_fraction, 65536);
  } catch (const std::overflow_error& error) {
    throw UsageError(error.what());
  }
}

Json composed_peaks(const resource::Composition& composition, int device_id, bool include_device) {
  const auto& peaks = composition.accounting.peaks;
  Json result{{"pageable", peaks[resource::kPageableSpace].bytes},
              {"host", peaks[resource::kHostSpace].bytes},
              {"pinned", peaks[resource::kPinnedSpace].bytes},
              {"device", peaks[resource::kDeviceSpace].bytes}};
  if (include_device)
    result["device:" + std::to_string(device_id)] = peaks[resource::kFirstDeviceSpace].bytes;
  return result;
}

struct Topology {
  Bytes nbf{0}, cartesian_nbf{0}, angular{0}, shells{0}, primitives{0};
  std::vector<std::uint8_t> angular_values;
  std::vector<std::size_t> primitive_counts;
};

Topology topology(const NativeBasisData& basis) {
  Topology result;
  result.shells = basis.shells.size();
  result.primitives = basis.primitives.size();
  for (const auto& shell : basis.shells) {
    const Bytes angular = shell.angular_momentum;
    const Bytes cartesian = (angular + 1) * (angular + 2) / 2;
    result.cartesian_nbf = sum({result.cartesian_nbf, cartesian});
    result.nbf =
        sum({result.nbf,
             basis.representation == GENERATIVEQC_BASIS_SPHERICAL ? 2 * angular + 1 : cartesian});
    result.angular = std::max(result.angular, angular);
    result.angular_values.push_back(static_cast<std::uint8_t>(angular));
    result.primitive_counts.push_back(shell.primitive_count);
  }
  return result;
}

Json topology_json(const Topology& shape) {
  return {{"nbf", shape.nbf},
          {"cartesian_nbf", shape.cartesian_nbf},
          {"shells", shape.shells},
          {"primitives", shape.primitives},
          {"maximum_angular", shape.angular},
          {"shell_angular", shape.angular_values},
          {"shell_primitive_counts", shape.primitive_counts}};
}

// These capacities follow the actual Jet/preparation/SCF lifetimes, not an
// optimistic nbf^4-only estimate. Serialized workspaces do not scale with batch.
Json hf_inventory(const std::array<Bytes, 4>& inventory) {
  Json result = Json::object();
  for (std::size_t index = 0; index < inventory.size(); ++index)
    result[scf::cpu_hf_host_resource_keys[index]] = inventory[index];
  return result;
}

scf::KsHostResourceShape ks_host_shape(const Topology& shape, Bytes atoms, Bytes spins,
                                       const RunOptions& options, Bytes ecp_terms,
                                       Bytes ecp_workspace) {
  return {atoms,
          shape.nbf,
          shape.cartesian_nbf,
          shape.shells,
          shape.primitives,
          spins,
          shape.angular,
          ecp_terms,
          product({atoms, options.grid.radial_points, options.grid.angular_polar,
                   options.grid.angular_azimuth}),
          options.grid.radial_points,
          options.grid.angular_polar,
          options.backend == GENERATIVEQC_BACKEND_CUDA ? options.grid.tile_points : 256,
          options.diis_history,
          options.max_iterations.value_or(100),
          options.method_name.starts_with("pbe"),
          false,
          ecp_workspace};
}

Json ks_inventory(const std::array<Bytes, 16>& inventory) {
  Json result = Json::object();
  for (std::size_t index = 0; index < inventory.size(); ++index)
    result[scf::ks_host_resource_keys[index]] = inventory[index];
  return result;
}

// The CLI repeats topology metadata; the native owner performs the same full
// bucket, retained-fleet and detached-retry composition as the Python API.
Json cuda_df_candidates(const Topology& orbital, const Topology& auxiliary, Bytes atoms,
                        Bytes spins, Bytes nalpha, Bytes nbeta, const RunOptions& options) {
  const scf::CudaDfResourceItem item{atoms,
                                     orbital.nbf,
                                     orbital.cartesian_nbf,
                                     orbital.shells,
                                     orbital.primitives,
                                     auxiliary.nbf,
                                     auxiliary.cartesian_nbf,
                                     auxiliary.shells,
                                     auxiliary.primitives,
                                     nalpha,
                                     nbeta,
                                     spins,
                                     options.batch};
  if (options.diis_history > std::numeric_limits<unsigned>::max())
    throw UsageError("DF DIIS history exceeds the native unsigned domain");
  const char* exchange = std::getenv("GENERATIVEQC_DF_EXCHANGE");
  scf::DfPairStorageRequest storage;
  try {
    storage = scf::requested_df_pair_storage_request();
  } catch (const std::invalid_argument& error) {
    throw UsageError(error.what());
  }
  const auto native =
      scf::cuda_df_fleet_resources(std::span(&item, 1), static_cast<unsigned>(options.diis_history),
                                   options.density_fitting_memory_bytes, storage,
                                   exchange && std::string_view(exchange) == "occupied");
  Json candidates = Json::array();
  for (const auto& candidate : native) {
    const auto& tile = candidate.buckets.front().energy_tiles;
    candidates.push_back({{"name", candidate.kind == 0   ? "cuda-df-resident"
                                   : candidate.kind == 1 ? "cuda-df-source"
                                                         : "cuda-df-packed"},
                          {"retained_host", candidate.retained_host},
                          {"workspace_host", candidate.workspace_host},
                          {"retained_device", candidate.retained_device},
                          {"workspace_device", candidate.workspace_device},
                          {"runtime_allowance_device", candidate.runtime_allowance_device},
                          {"sub_budget", candidate.sub_budget},
                          {"tile_pairs", tile[1]},
                          {"tile_auxiliary", tile[2]},
                          {"automatic_rhf_rank", tile[15]},
                          {"stores_full_three_center", tile[5] != 0}});
  }
  return candidates;
}

}  // namespace

int resources_command(const RunOptions& options, std::span<const generativeqc_atom> atoms) {
  const bool hf =
      options.method == GENERATIVEQC_METHOD_RHF || options.method == GENERATIVEQC_METHOD_UHF;
  const bool ks = options.method_name == "lda-rks" || options.method_name == "lda-uks" ||
                  options.method_name == "pbe-rks" || options.method_name == "pbe-uks";
  Json budget = {
      {"schema_version", 1},
      {"host_bytes", options.host_bytes ? Json(*options.host_bytes) : Json()},
      {"device_bytes", options.device_bytes ? Json(*options.device_bytes) : Json()},
      {"pinned_host_bytes", options.pinned_host_bytes ? Json(*options.pinned_host_bytes) : Json()},
      {"per_device_bytes", Json::array()},
      {"host_reserve_bytes", options.host_reserve_bytes},
      {"device_reserve_bytes", options.device_reserve_bytes},
      {"headroom_fraction", options.headroom_fraction}};
  Json limits = Json::object();
  const auto limit = [&](const char* name, std::optional<Bytes> cap, Bytes reserve) {
    if (cap) {
      limits[name] = resource::effective_capacity(*cap, reserve, options.headroom_fraction);
    }
  };
  limit("host", options.host_bytes, options.host_reserve_bytes);
  limit("device", options.device_bytes, options.device_reserve_bytes);
  limit("pinned", options.pinned_host_bytes, 0);
  Json plan = {{"schema_version", 1},
               {"composition_inventory", "native-resource-composition-v1"},
               {"scope", "native-cli-allocation-inventory-v1"},
               {"source_identity", generativeqc_get_source_identity()},
               {"budget", budget},
               {"limits", limits},
               {"status", "unsupported"},
               {"diagnostic", nullptr},
               {"method", options.method_name},
               {"batch", options.batch},
               {"peak_bytes", Json::object()},
               {"resident_bytes", Json::object()},
               {"exclusions",
                {"caller-owned inputs and outputs",
                 "allocator bookkeeping, object/runtime overhead, CUDA contexts/modules/pools"}}};
  const auto emit = [&]() {
    plan["identity"] = canonical_hash(plan);
    std::cout << plan.dump(2) << '\n';
    return plan["status"] == "feasible" ? 0 : 2;
  };
  if (!hf && !ks) {
    plan["diagnostic"] = "native resource inventory requires RHF/UHF or LDA/PBE RKS/UKS";
    return emit();
  }
  // This inventory admits only mean-field methods, never correlated references.
  const auto convergence = resolve_scf_convergence(options, false);
  if (ks) {
    const auto incremental_requested = [](const char* name) {
      const auto control = scf::classify_incremental_direct_jk_control(std::getenv(name));
      if (control == scf::IncrementalDirectJkControl::Invalid)
        throw UsageError(std::string(name) + " must be 0/off or 1/on");
      return control == scf::IncrementalDirectJkControl::Enabled;
    };
    // Validate both eagerly, even when the first selector enables the mode.
    const bool generic = incremental_requested("GENERATIVEQC_KS_INCREMENTAL_DIRECT_JK");
    const bool legacy = incremental_requested("GENERATIVEQC_PBE0_INCREMENTAL_DIRECT_JK");
    if (generic || legacy) {
      // The scalar v1 layout has no incremental-mode input. Do not present its
      // ordinary state bytes as capacity for anchor/delta storage and closure.
      plan["diagnostic"] =
          "incremental Direct-J/K resource inventory is unsupported: anchor/delta storage and "
          "closure history lack a qualified inventory; unset or disable both "
          "GENERATIVEQC_KS_INCREMENTAL_DIRECT_JK and GENERATIVEQC_PBE0_INCREMENTAL_DIRECT_JK";
      return emit();
    }
  }
  if (options.precision != GENERATIVEQC_PRECISION_FP64) {
    plan["diagnostic"] = "this inventory covers FP64 allocation policies";
    return emit();
  }
  if (ks && (options.forces || options.density_fitting != GENERATIVEQC_DENSITY_FITTING_NONE)) {
    plan["diagnostic"] =
        "KS dry-run inventory currently covers conventional electronic energy only";
    return emit();
  }
  NativeBasisData basis;
  try {
    basis = expand_basis_input(
        options.basis_name, atoms,
        options.representation_explicit ? std::optional(options.representation) : std::nullopt);
  } catch (const std::invalid_argument& error) {
    throw UsageError(error.what());
  }
  const auto orbital = topology(basis);
  const Bytes spins =
      options.method_name.ends_with("uhf") || options.method_name.ends_with("uks") ? 2 : 1;
  std::int64_t electrons = -static_cast<std::int64_t>(options.charge);
  for (const auto& atom : atoms) electrons += atom.atomic_number;
  for (const auto core : basis.core_electrons) electrons -= core;
  const auto unpaired = static_cast<std::int64_t>(options.multiplicity) - 1;
  if (electrons < 0 || unpaired > electrons || (electrons - unpaired) % 2 ||
      (spins == 1 && unpaired) || static_cast<Bytes>((electrons + unpaired) / 2) > orbital.nbf)
    throw UsageError("invalid electron count, multiplicity or basis spin population");
  const bool fitted = options.density_fitting != GENERATIVEQC_DENSITY_FITTING_NONE;
  const bool cpu = options.backend == GENERATIVEQC_BACKEND_CPU_REFERENCE ||
                   options.density_fitting == GENERATIVEQC_DENSITY_FITTING_CPU_REFERENCE;
  if (options.density_fitting == GENERATIVEQC_DENSITY_FITTING_CUDA &&
      options.backend != GENERATIVEQC_BACKEND_CUDA)
    throw UsageError("CUDA density fitting requires --backend cuda");
  plan["backend"] = cpu ? "cpu" : "cuda";
  if (orbital.angular > (cpu ? 4u : 3u)) {
    plan["diagnostic"] = "orbital angular momentum exceeds the native inventory domain";
    return emit();
  }
  plan["representation"] =
      basis.representation == GENERATIVEQC_BASIS_SPHERICAL ? "spherical" : "cartesian";
  if (!basis.identity.empty()) plan["basis_identity"] = basis.identity;
  Json primitive_identity = Json::array(), shell_identity = Json::array(), elements = Json::array();
  for (const auto& primitive : basis.primitives)
    primitive_identity.push_back({primitive.exponent, primitive.coefficient});
  for (const auto& shell : basis.shells)
    shell_identity.push_back(
        {shell.atom_index, shell.angular_momentum, shell.primitive_offset, shell.primitive_count});
  for (const auto& atom : atoms) elements.push_back(atom.atomic_number);
  plan["topology_identity"] =
      canonical_hash(Json{{"elements", elements},
                          {"shells", shell_identity},
                          {"primitives", primitive_identity},
                          {"representation", basis.representation},
                          {"auxiliary_basis", options.auxiliary_basis_name}});
  plan["controls"] = {
      {"charge", options.charge},
      {"multiplicity", options.multiplicity},
      {"device_id", options.device_id},
      {"diis_history", options.diis_history},
      {"max_iterations", options.max_iterations.value_or(100)},
      {"precision", "fp64"},
      {"energy_tolerance", convergence.energy_tolerance},
      {"density_tolerance", convergence.density_tolerance},
      {"screening_tolerance", options.screening_tolerance.value_or(1e-14)},
      {"density_fitting", options.density_fitting},
      {"density_fitting_threshold", options.density_fitting_threshold},
      {"grid",
       {options.grid.radial_points, options.grid.angular_polar, options.grid.angular_azimuth}}};
  plan["orbital"] = topology_json(orbital);
  Bytes ecp_workspace = 0;
  if (generativeqc_resource_ecp_workspace_v1(orbital.nbf, orbital.cartesian_nbf, atoms.size(),
                                             orbital.primitives, basis.ecp_terms.size(),
                                             cpu ? 0 : 1, &ecp_workspace))
    throw UsageError("ECP allocation inventory overflows native bounds");
  if (!basis.ecp_terms.empty()) {
    plan["orbital"]["ecp_terms"] = basis.ecp_terms.size();
    plan["orbital"]["ecp_core_electrons"] = basis.core_electrons;
    plan["ecp_workspace"] = ecp_workspace;
    if (fitted) {
      plan["diagnostic"] = "ECP density fitting is not qualified";
      return emit();
    }
  }
  std::optional<Topology> auxiliary;
  if (fitted) {
    if (options.auxiliary_basis_explicit) {
      const auto expanded = expand_basis_input(options.auxiliary_basis_name, atoms,
                                               !is_local_basis(options.auxiliary_basis_name)
                                                   ? std::optional(basis.representation)
                                                   : std::nullopt);
      if (!expanded.ecp_terms.empty())
        throw UsageError("an auxiliary fitting basis cannot contain ECP potentials");
      auxiliary = topology(expanded);
      Json primitive_record = Json::array();
      for (const auto& primitive : expanded.primitives)
        primitive_record.push_back({primitive.exponent, primitive.coefficient});
      plan["auxiliary_identity"] =
          expanded.identity.empty()
              ? canonical_hash(Json{{"topology", topology_json(*auxiliary)},
                                    {"primitives", primitive_record},
                                    {"representation", expanded.representation}})
              : expanded.identity;
    } else
      auxiliary = orbital;
    if (auxiliary->angular > (cpu ? 4u : 3u)) {
      plan["diagnostic"] = "auxiliary angular momentum exceeds the native inventory domain";
      return emit();
    }
    plan["auxiliary"] = topology_json(*auxiliary);
  }
  Bytes retained = 0, workspace = 0, device = 0;
  if (cpu && hf) {
    scf::CpuHfHostFleetResources fleet;
    const auto phases =
        fleet.append({{atoms.size(), orbital.nbf, orbital.cartesian_nbf, spins, orbital.angular,
                       options.diis_history, auxiliary ? auxiliary->nbf : 0,
                       auxiliary ? auxiliary->cartesian_nbf : 0, auxiliary ? auxiliary->angular : 0,
                       ecp_workspace},
                      orbital.shells,
                      orbital.primitives,
                      auxiliary ? auxiliary->shells : 0,
                      auxiliary ? auxiliary->primitives : 0},
                     options.batch);
    plan["item_inventory"] = hf_inventory(phases);
    const auto aggregate = fleet.inventory();
    retained = aggregate[0];
    workspace = aggregate[1];
  } else if (cpu && ks) {
    scf::CpuKsHostFleetResources fleet;
    plan["item_inventory"] = ks_inventory(fleet.append(
        ks_host_shape(orbital, atoms.size(), spins, options, basis.ecp_terms.size(), ecp_workspace),
        options.batch));
    const auto aggregate = fleet.inventory();
    retained = aggregate[12];
    workspace = aggregate[13];
  } else if (hf && fitted) {
    if (auxiliary->nbf > 128) {
      plan["diagnostic"] = "CUDA DF inventory requires <=128 auxiliary AOs";
      return emit();
    }
    std::array<Bytes, 2> ordinary{};
    if (generativeqc_resource_small_hf_cuda_v2(
            orbital.nbf, orbital.cartesian_nbf, atoms.size(), orbital.shells,
            orbital.angular_values.data(), orbital.primitive_counts.data(), options.diis_history,
            spins, 0, convergence.energy_tolerance, options.screening_tolerance.value_or(1e-14),
            ordinary.data(), ordinary.size())) {
      plan["diagnostic"] =
          "CUDA DF inventory requires a CUDA library and ordinary <=16-AO topology";
      return emit();
    }
    try {
      const auto candidates = cuda_df_candidates(
          orbital, *auxiliary, atoms.size(), spins, static_cast<Bytes>((electrons + unpaired) / 2),
          static_cast<Bytes>((electrons - unpaired) / 2), options);
      plan["candidates"] = candidates;
      std::vector<std::string> names;
      for (const auto& candidate : candidates)
        names.push_back(candidate.at("name").get<std::string>());
      std::sort(names.begin(), names.end());
      std::vector<resource::Candidate> choices;
      for (std::size_t index = 0; index < candidates.size(); ++index) {
        const auto& candidate = candidates.at(index);
        const auto rank =
            std::lower_bound(names.begin(), names.end(), candidate.at("name").get<std::string>()) -
            names.begin();
        choices.push_back(inventory_candidate(
            candidate.at("retained_host").get<Bytes>(), candidate.at("workspace_host").get<Bytes>(),
            sum({candidate.at("retained_device").get<Bytes>(),
                 candidate.at("runtime_allowance_device").get<Bytes>()}),
            candidate.at("workspace_device").get<Bytes>(), index, rank));
      }
      const auto composition = compose_inventory(options, std::move(choices));
      const bool feasible = composition.status == resource::Status::feasible;
      plan["selection"] = candidates.at(composition.selections.front()).at("name");
      plan["peak_bytes"] = composed_peaks(composition, options.device_id, true);
      plan["resident_bytes"] = {
          {"pageable", composition.accounting.resident[resource::kPageableSpace].bytes},
          {"device:" + std::to_string(options.device_id),
           composition.accounting.resident[resource::kFirstDeviceSpace].bytes}};
      plan["status"] = feasible ? "feasible" : "infeasible";
      if (!feasible) plan["diagnostic"] = "no native DF allocation alternative fits all budgets";
    } catch (const UsageError&) {
      throw;
    } catch (const std::exception& error) {
      plan["status"] = "infeasible";
      plan["diagnostic"] = error.what();
    }
    return emit();
  } else if (ks) {
    const Bytes points = product({atoms.size(), options.grid.radial_points,
                                  options.grid.angular_polar, options.grid.angular_azimuth});
    std::array<Bytes, 3> capacities{};
    Bytes quadrature = 0, matrix_provider = 0;
    if (generativeqc_resource_ks_cuda_v1(
            orbital.nbf, atoms.size(), orbital.shells, orbital.primitives, points,
            options.diis_history, spins, options.method_name.starts_with("pbe") ? 1 : 0,
            options.grid.tile_points, capacities.data(), capacities.size()) ||
        generativeqc_resource_quadrature_cuda_v1(atoms.size(), points, &quadrature) ||
        generativeqc_resource_ks_matrix_provider_cuda_v1(orbital.nbf, &matrix_provider)) {
      plan["diagnostic"] =
          "KS CUDA inventory requires a CUDA library and valid launch/DIIS dimensions";
      return emit();
    }
    scf::KsHostFleetResources host_fleet;
    plan["item_inventory"] = ks_inventory(host_fleet.append(
        ks_host_shape(orbital, atoms.size(), spins, options, basis.ecp_terms.size(), ecp_workspace),
        options.batch, scf::KsHostExecution::cuda_fused));
    const auto host_inventory = host_fleet.inventory();
    retained = host_inventory[12];
    workspace = host_inventory[13];
    plan["host_fleet_composition"] = "native-declared-capacity";
    scf::CudaKsDeviceFleetResources fleet;
    std::array<Bytes, 6> device_item{};
    try {
      device_item = fleet.append(
          {orbital.cartesian_nbf, atoms.size(), orbital.shells, orbital.primitives, capacities[0],
           capacities[1], capacities[2], quadrature, matrix_provider, ecp_workspace, true, 0, 0},
          options.batch);
    } catch (const std::exception& error) {
      throw UsageError(error.what());
    }
    const auto device_fleet = fleet.inventory();
    device = device_fleet[7];
    plan["resident_bytes"]["device:" + std::to_string(options.device_id)] =
        device_fleet[7] - device_fleet[4];
    plan["device_fleet_composition"] = "native-declared-capacity";
    plan["device_inventory"] = {{"state", capacities[0]},
                                {"xc_and_grid", capacities[1]},
                                {"coulomb", capacities[2]},
                                {"setup", device_item[3]},
                                {"matrix_provider", matrix_provider}};
  } else if (hf && !fitted) {
    std::array<Bytes, 2> values{};
    if (generativeqc_resource_small_hf_cuda_v2(
            orbital.nbf, orbital.cartesian_nbf, atoms.size(), orbital.shells,
            orbital.angular_values.data(), orbital.primitive_counts.data(), options.diis_history,
            spins, 0, convergence.energy_tolerance, options.screening_tolerance.value_or(1e-14),
            values.data(), values.size())) {
      plan["diagnostic"] =
          "CUDA HF inventory requires a CUDA library and ordinary <=16-AO s/p/d/f topology";
      return emit();
    }
    scf::CudaHfFleetResources fleet;
    try {
      const auto item =
          fleet.append({atoms.size(), orbital.nbf, orbital.cartesian_nbf, orbital.shells,
                        orbital.primitives, spins, values[0], values[1], ecp_workspace},
                       options.batch);
      plan["item_inventory"] = {{"arena", item[0]}, {"host", item[1]}, {"ecp_workspace", item[2]}};
    } catch (const std::exception& error) {
      throw UsageError(error.what());
    }
    const auto aggregate = fleet.inventory();
    retained = aggregate[3];
    device = aggregate[2];
    plan["resident_bytes"]["device:" + std::to_string(options.device_id)] = aggregate[0];
    plan["cuda_hf_fleet_composition"] = "native-declared-capacity";
  } else {
    plan["diagnostic"] = "this CUDA provider has no qualified native dry-run inventory";
    return emit();
  }
  // Each native fleet already applies batch multiplicity to retained storage.
  const Bytes resident_device =
      plan["resident_bytes"].value("device:" + std::to_string(options.device_id), Bytes{0});
  if (resident_device > device)
    throw UsageError("resident device capacity exceeds the complete peak");
  const auto composition = compose_inventory(
      options,
      {inventory_candidate(retained, workspace, resident_device, device - resident_device)});
  plan["peak_bytes"] = composed_peaks(composition, options.device_id, device != 0);
  plan["resident_bytes"]["pageable"] =
      composition.accounting.resident[resource::kPageableSpace].bytes;
  plan["status"] = composition.status == resource::Status::feasible ? "feasible" : "infeasible";
  for (auto cap = limits.begin(); cap != limits.end(); ++cap) {
    if (plan["peak_bytes"].value(cap.key(), Bytes{0}) > cap.value().get<Bytes>()) {
      plan["diagnostic"] = "allocation inventory exceeds " + cap.key() + " budget";
      break;
    }
  }
  return emit();
}
}  // namespace generativeqc::cli
