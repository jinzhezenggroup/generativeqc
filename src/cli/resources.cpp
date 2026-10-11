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
#include "dft/incremental_direct_jk_control.hpp"

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
extern "C" int generativeqc_resource_df_source_bytes_v1(std::size_t, std::size_t, std::size_t,
                                                        std::size_t, std::size_t, std::size_t,
                                                        std::uint64_t*);
extern "C" int generativeqc_resource_df_diis_bytes_v1(std::size_t, std::size_t, unsigned,
                                                      std::uint64_t*);
extern "C" int generativeqc_resource_df_tiles_v3(std::size_t, std::size_t, std::size_t, std::size_t,
                                                 std::size_t, std::size_t, unsigned, std::size_t,
                                                 std::uint64_t*, std::size_t, char*, std::size_t);
extern "C" int generativeqc_resource_df_packed_tiles_v2(std::size_t, std::size_t, std::size_t,
                                                        std::size_t, std::size_t, std::size_t,
                                                        std::size_t, std::uint64_t*, std::size_t,
                                                        char*, std::size_t);
extern "C" int generativeqc_resource_ecp_workspace_v1(std::size_t, std::size_t, std::size_t,
                                                      std::size_t, std::size_t, int,
                                                      std::uint64_t*);

namespace generativeqc::cli {
namespace {
using Bytes = std::uint64_t;
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

Bytes headroom_bytes(Bytes capacity, double fraction) {
  if (fraction == 0 || capacity == 0) return 0;
  int exponent = 0;
  const double normalized = std::frexp(fraction, &exponent);
  const Bytes numerator = static_cast<Bytes>(std::ldexp(normalized, 53));
  const int shift = 53 - exponent;
  // Multiply two limbs without an extension type, then round up the exact
  // IEEE-754 rational product. Long-double multiplication is not exact here.
  Bytes high = 0, low = 0;
  for (unsigned bit = 0; bit < 53; ++bit) {
    if (!(numerator & (Bytes{1} << bit))) continue;
    const Bytes previous = low;
    low += capacity << bit;
    high += (bit ? capacity >> (64 - bit) : 0) + (low < previous ? 1 : 0);
  }
  if (shift >= 128) return high || low ? 1 : 0;
  if (shift >= 64) {
    const int upper_shift = shift - 64;
    const Bytes remainder = upper_shift ? high & ((Bytes{1} << upper_shift) - 1) : 0;
    return (high >> upper_shift) + (remainder || low ? 1 : 0);
  }
  return (low >> shift) + (high << (64 - shift)) + ((low & ((Bytes{1} << shift) - 1)) ? 1 : 0);
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
Json hf_inventory(const Topology& orbital, const Topology* auxiliary, Bytes atoms, Bytes spins,
                  Bytes history, Bytes ecp_workspace) {
  const Bytes n2 = product({orbital.nbf, orbital.nbf});
  const Bytes c2 = product({orbital.cartesian_nbf, orbital.cartesian_nbf});
  const Bytes derivatives = product({3, atoms});
  const Bytes nterms = sum({product({2, n2}), product({n2, n2})});
  const Bytes cterms = sum({product({2, c2}), product({c2, c2})});
  const Bytes conventional = sum({product({8, derivatives + 1, nterms}), 8 * derivatives});
  const Bytes cartesian = sum({product({8, derivatives + 1, cterms}), 8 * derivatives});
  const Bytes jet = sum({product({8, derivatives + 1}), 64});
  const Bytes angular = std::max(orbital.angular, auxiliary ? auxiliary->angular : 0);
  const Bytes order =
      std::max({4 * orbital.angular, 2 * orbital.angular + (auxiliary ? auxiliary->angular : 0),
                auxiliary ? 2 * auxiliary->angular : 0});
  const Bytes hermite = product({6, angular + 3, angular + 3, 2 * angular + 4});
  const Bytes recurrence = product({jet, sum({product({order + 1, order + 1, order + 1, order + 1}),
                                              hermite, 256, derivatives})});
  Bytes preparation = sum(
      {product({jet, cterms}),
       std::max(recurrence, sum({cartesian, conventional, product({8, nterms})})), ecp_workspace});
  Bytes resident = conventional;
  const Bytes matrix = sum(
      {product({8, spins, n2, 64 + 2 * (history + 1)}), product({16, history + 1, history + 1})});
  Bytes iteration = sum({resident, matrix}), finalization = iteration;
  if (auxiliary) {
    const Bytes count =
        sum({product({auxiliary->nbf, auxiliary->nbf}), product({n2, auxiliary->nbf})});
    const Bytes cart_count = sum({product({auxiliary->cartesian_nbf, auxiliary->cartesian_nbf}),
                                  product({c2, auxiliary->cartesian_nbf})});
    const Bytes raw = product({8, derivatives + 1, count});
    const Bytes cart_raw = product({8, derivatives + 1, cart_count});
    const Bytes transformed = product({8, n2, auxiliary->nbf});
    preparation = std::max(
        {preparation,
         sum({conventional, product({jet, cart_count}),
              std::max(recurrence, sum({cart_raw, raw, product({8, count})}))}),
         sum({conventional, raw, transformed, product({128, auxiliary->nbf, auxiliary->nbf})})});
    resident = sum({resident, raw, transformed});
    iteration = sum({resident, matrix, product({8, sum({auxiliary->nbf, 4 * n2})})});
    finalization = sum({resident, matrix,
                        product({8, sum({product({32, auxiliary->nbf, auxiliary->nbf}),
                                         product({2, n2, auxiliary->nbf}), 8 * derivatives})})});
  }
  return {{"preparation", preparation},
          {"iteration", iteration},
          {"finalization", finalization},
          {"integral_state", resident}};
}

Json ks_inventory(const Topology& shape, Bytes atoms, Bytes spins, const RunOptions& options,
                  Bytes ecp_terms, Bytes ecp_workspace) {
  const bool cuda = options.backend == GENERATIVEQC_BACKEND_CUDA;
  const Bytes orbital_aos = shape.nbf, cartesian_aos = shape.cartesian_nbf;
  const Bytes n2 = product({orbital_aos, orbital_aos}),
              c2 = product({cartesian_aos, cartesian_aos});
  const Bytes points = product({atoms, options.grid.radial_points, options.grid.angular_polar,
                                options.grid.angular_azimuth});
  const bool gradients = options.method_name.starts_with("pbe");
  const Bytes packed = sum({3 * atoms, 2 * shape.primitives, 16 * orbital_aos});
  const Bytes metadata = sum({8192,
                              product({512, sum({atoms, shape.shells, shape.primitives, orbital_aos,
                                                 cartesian_aos, ecp_terms})}),
                              cuda ? 16384u : 0u});
  const Bytes grid = product({36, points});
  const Bytes basis = product({8, packed});
  const Bytes warm = sum({product({8, n2, 4 + 4 * spins}), product({96, atoms})});
  const Bytes history = product({256, options.max_iterations.value_or(100)});
  const Bytes provider = product({8, sum({2 * n2, cuda ? 0 : product({n2, n2})})});
  const Bytes solver = cuda && orbital_aos > 16 ? sum({1024 * 1024, product({128, n2})}) : 0;
  const Bytes retained =
      sum({metadata, grid, basis, warm, history, provider, solver, cuda ? 16384u : 0u});
  const Bytes matrix = sum({product({8, spins, n2, 128 + 2 * (options.diis_history + 1)}),
                            product({16, options.diis_history + 1, options.diis_history + 1})});
  const Bytes tile =
      cuda ? 0
           : sum({product({8, std::min<Bytes>(points, 256), orbital_aos, gradients ? 4u : 1u}),
                  product({8, spins, n2})});
  // CPU restricted GGA retains the eligible AO grid cache for one serialized solve.
  const Bytes cache_size = product({32, points, orbital_aos});
  const Bytes cache =
      !cuda && gradients && spins == 1 && cache_size <= 64 * 1024 * 1024 ? cache_size : 0;
  const Bytes angular = shape.angular;
  const Bytes order = 4 * angular + 1;
  const Bytes recurrence =
      product({64, sum({product({order, order, order, order}),
                        product({6, angular + 3, angular + 3, 2 * angular + 4}), 256, atoms})});
  const Bytes integrals =
      cuda ? 0
           : sum({product({64, sum({2 * c2, product({c2, c2})})}),
                  std::max(recurrence, product({32, sum({product({c2, c2}), product({n2, n2}),
                                                         cartesian_aos * orbital_aos, n2})}))});
  const Bytes quadrature =
      sum({product({16, sum({options.grid.radial_points, options.grid.angular_polar})}), 8 * atoms,
           cuda ? product({8, sum({4 * atoms, 2 * (512 + 256)})}) : 0});
  const Bytes pairs = shape.shells * (shape.shells + 1) / 2;
  const Bytes setup =
      cuda ? sum({product({2048, sum({1, atoms, shape.shells, shape.primitives, orbital_aos,
                                      cartesian_aos, pairs, product({pairs, pairs})})}),
                  product({64, sum({c2, n2, product({orbital_aos, cartesian_aos})})}), quadrature,
                  basis})
           : sum({integrals, quadrature, basis});
  return {{"retained", retained},
          {"setup_workspace", sum({setup, ecp_workspace})},
          {"scf_workspace", sum({matrix, tile, cache})}};
}

struct DfTiles {
  Bytes pairs, auxiliary, peak, automatic_rank, factor, scratch;
  bool full;
};

DfTiles df_tiles(Bytes batch, Bytes nbf, Bytes naux, Bytes occupied, Bytes budget, Bytes fixed,
                 bool generated, bool packed, Bytes rhf_rank) {
  std::array<Bytes, 11> values{};
  std::array<char, 512> error{};
  const auto status = packed ? generativeqc_resource_df_packed_tiles_v2(
                                   batch, nbf, naux, occupied, budget, fixed, rhf_rank,
                                   values.data(), values.size(), error.data(), error.size())
                             : generativeqc_resource_df_tiles_v3(
                                   batch, nbf, naux, occupied, budget, fixed, generated, rhf_rank,
                                   values.data(), 7, error.data(), error.size());
  if (status) throw std::runtime_error(error[0] ? error.data() : "native DF tile query failed");
  return {values[1],
          values[2],
          values[4],
          packed ? values[10] : values[6],
          packed ? values[6] : 0,
          packed ? values[7] : 0,
          values[5] != 0};
}

// A CLI fleet repeats one topology and therefore forms one native DF bucket.
// Retained factors, spectral/solver allowances and the detached singleton cold
// retry coexist. Source/packed policies must never be charged as dense only.
Json cuda_df_candidates(const Topology& orbital, const Topology& auxiliary, Bytes atoms,
                        Bytes spins, Bytes occupied, const RunOptions& options) {
  const Bytes batch = options.batch, orbital_aos = orbital.nbf,
              cartesian_aos = orbital.cartesian_nbf;
  const Bytes aux = auxiliary.nbf, ac = auxiliary.cartesian_nbf, derivatives = 3 * atoms;
  if (orbital_aos > 16 || aux > 128)
    throw std::runtime_error("CUDA DF inventory requires <=16 orbital and <=128 auxiliary AOs");
  const char* storage_env = std::getenv("GENERATIVEQC_DF_VALUE_STORAGE");
  const std::string storage = storage_env ? storage_env : "auto";
  if (storage != "auto" && storage != "dense" && storage != "packed" && storage != "packed-single")
    throw UsageError("GENERATIVEQC_DF_VALUE_STORAGE must be auto, dense, packed or packed-single");
  const bool packed = storage == "packed" || storage == "packed-single";
  const char* exchange_env = std::getenv("GENERATIVEQC_DF_EXCHANGE");
  const bool occupied_exchange = exchange_env && std::string_view(exchange_env) == "occupied";
  const Bytes rhf_rank = spins == 1 ? occupied : 0;
  Bytes source_bytes = 0, diis = 0, retry_diis = 0;
  if (generativeqc_resource_df_source_bytes_v1(
          batch, product({batch, atoms}),
          product({batch, sum({orbital.shells, auxiliary.shells, 1})}),
          product({batch, sum({cartesian_aos, ac, 1})}),
          product({batch, sum({orbital.primitives, auxiliary.primitives, 1})}),
          product({batch, sum({orbital_aos * cartesian_aos, aux * ac})}), &source_bytes) ||
      generativeqc_resource_df_diis_bytes_v1(batch, orbital_aos, options.diis_history, &diis) ||
      generativeqc_resource_df_diis_bytes_v1(1, orbital_aos, options.diis_history, &retry_diis))
    throw std::runtime_error("invalid native DF source/DIIS capacity");
  const auto default_tile = df_tiles(batch, orbital_aos, aux, occupied, 0,
                                     sum({source_bytes, diis}), true, packed, rhf_rank);
  const Bytes shell_pairs = orbital.shells * (orbital.shells + 1) / 2;
  const Bytes item_packed =
      product({512, sum({1, atoms, orbital.shells, orbital.primitives, cartesian_aos, orbital_aos,
                         shell_pairs, auxiliary.shells, auxiliary.primitives})});
  const Bytes preparation_retained = product(
      {batch, sum({item_packed, product({8, sum({2 * orbital_aos * orbital_aos, derivatives})})})});
  const Bytes preparation_temporary =
      sum({item_packed, product({8, sum({6 * cartesian_aos * cartesian_aos, derivatives, 1})})});
  const Bytes preparation_minimum =
      sum({product({batch, item_packed}), preparation_retained, preparation_temporary});
  const Bytes combined_shells = sum({orbital.shells, auxiliary.shells, 1});
  const Bytes combined_pairs = combined_shells * (combined_shells + 1) / 2;
  const Bytes metadata = product(
      {batch, 2048,
       sum({1, atoms, combined_shells, orbital.primitives, auxiliary.primitives, cartesian_aos, ac,
            combined_pairs, product({combined_pairs, combined_pairs})})});
  const Bytes response_metadata =
      product({256, sum({1, atoms, orbital_aos, aux, orbital.primitives, auxiliary.primitives})});
  const Bytes response_minimum =
      sum({response_metadata,
           product({8, sum({4 * aux * aux, 9 * orbital_aos * orbital_aos, 6 * aux})})});
  const Bytes resident_response =
      sum({response_metadata,
           std::max(product({8, sum({12 * aux * aux, 10 * aux, orbital_aos * orbital_aos})}),
                    sum({product({8, sum({3 * aux * aux, 6 * aux, 4 * orbital_aos * orbital_aos})}),
                         product({16, orbital_aos, orbital_aos, aux})}))});
  const Bytes response =
      sum({response_metadata,
           product({8, sum({4 * aux * aux, (7 + 2 * aux) * orbital_aos * orbital_aos, 6 * aux})})});
  const Bytes response_host = sum({response_metadata, product({8, orbital_aos, orbital_aos})});
  const Bytes source_budget = options.density_fitting_memory_bytes
                                  ? options.density_fitting_memory_bytes
                                  : std::max({preparation_minimum, product({2, default_tile.peak}),
                                              product({2, response_minimum})});
  Json candidates = Json::array();
  for (const bool source : {false, true}) {
    if ((!source && options.density_fitting_memory_bytes) || (packed && !source)) continue;
    const Bytes budget = source ? source_budget : 0;
    if (source && (budget < preparation_minimum || budget / 2 < response_minimum))
      throw std::runtime_error("DF sub-budget cannot hold preparation/generated response minimum");
    const auto tile = df_tiles(batch, orbital_aos, aux, occupied, budget,
                               sum({source ? source_bytes : 0, diis}), source, packed, rhf_rank);
    if (source)
      (void)df_tiles(batch, orbital_aos, aux, occupied, std::max<Bytes>(1, budget / 2),
                     sum({source_bytes, diis}), true, packed, rhf_rank);
    const Bytes pairs =
        source ? std::min(orbital_aos * orbital_aos,
                          std::max(orbital_aos, tile.pairs / orbital_aos * orbital_aos))
               : orbital_aos * orbital_aos;
    const Bytes tile_bytes = product({8, pairs, source ? tile.auxiliary : aux});
    const Bytes matrix = product({8, batch, orbital_aos, orbital_aos}),
                metric = product({8, batch, aux, aux});
    const Bytes tensor = product({8, batch, orbital_aos, orbital_aos, aux});
    const Bytes solver = sum({64 * 1024 * 1024, 16 * matrix, 128 * aux * aux});
    const bool factors = occupied_exchange || tile.automatic_rank;
    const Bytes eigen_host = sum({1024 * 1024, 128 * orbital_aos * orbital_aos});
    const Bytes eigen_device =
        sum({eigen_host, 8 * (3 * orbital_aos * orbital_aos + orbital_aos), 5});
    Bytes device = sum({(factors ? 34u : 32u) * matrix, product({16, batch, aux}), solver,
                        1024 * batch, eigen_device,
                        product({batch, sum({16 * (orbital_aos * orbital_aos + orbital_aos), 24})}),
                        diis, 2 * metric, product({8, batch, aux})});
    device = sum(
        {device,
         packed   ? sum({2 * tile.factor, tile.scratch, source_bytes})
         : source ? sum({tile.full ? sum({tensor, 3 * tile_bytes}) : 4 * tile_bytes, source_bytes})
                  : sum({tensor, 3 * tile_bytes})});
    const Bytes setup = sum({metric, product({8, batch, aux}), solver, 4 * batch});
    const Bytes force =
        source ? std::min(response, budget / 2)
               : std::max(resident_response, std::min<Bytes>(response, 128 * 1024 * 1024));
    const Bytes generation =
        sum({source_bytes,
             product({8, batch,
                      sum({2 * cartesian_aos * cartesian_aos,
                           source ? 0 : ac * ac + cartesian_aos * cartesian_aos * ac})})});
    const Bytes one_electron =
        product({8, batch, sum({2 * orbital_aos * orbital_aos, derivatives})});
    const Bytes raw = product({8, batch, sum({aux * aux, orbital_aos * orbital_aos * aux})});
    Bytes host =
        sum({metadata, product({8, batch, sum({32 * orbital_aos * orbital_aos, 16 * derivatives})}),
             2 * matrix, product({8, batch, derivatives}), eigen_host, 64 * batch, one_electron});
    if (!source) host = sum({host, raw, tensor});
    const Bytes scf =
        product({8, batch,
                 sum({129 * orbital_aos * orbital_aos,
                      4 * (options.diis_history + 1) * orbital_aos * orbital_aos,
                      product({4, options.diis_history + 1, options.diis_history + 1})})});
    Bytes host_work = sum(
        {64 * 1024 * 1024, scf, one_electron, 64 * (metric / batch), 8 * (tensor / batch),
         2 * metadata,
         source ? 2 * tile_bytes
                : sum({2 * raw,
                       product({8, batch, sum({ac * ac, cartesian_aos * cartesian_aos * ac})})}),
         8 * (32 * orbital_aos * orbital_aos + 8 * orbital_aos)});
    host_work = sum(
        {std::max(host_work, preparation_temporary), source ? response_host : resident_response});
    const auto retry =
        df_tiles(1, orbital_aos, aux, occupied, budget,
                 sum({source ? source_bytes / batch : 0, retry_diis}), source, packed, rhf_rank);
    const Bytes device_work =
        sum({std::max({setup, force, generation}), (device - eigen_device) / batch, eigen_device,
             solver, retry.automatic_rank && !factors ? 2 * matrix / batch : 0});
    candidates.push_back({{"name", packed   ? "cuda-df-packed"
                                   : source ? "cuda-df-source"
                                            : "cuda-df-resident"},
                          {"retained_host", host},
                          {"workspace_host", host_work},
                          {"retained_device", device},
                          {"workspace_device", device_work},
                          {"runtime_allowance_device", 512 * 1024 * 1024},
                          {"sub_budget", budget},
                          {"tile_pairs", tile.pairs},
                          {"tile_auxiliary", tile.auxiliary},
                          {"automatic_rhf_rank", tile.automatic_rank},
                          {"stores_full_three_center", tile.full}});
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
      const Bytes withheld = headroom_bytes(*cap, options.headroom_fraction);
      limits[name] = *cap - withheld > reserve ? *cap - withheld - reserve : 0;
    }
  };
  limit("host", options.host_bytes, options.host_reserve_bytes);
  limit("device", options.device_bytes, options.device_reserve_bytes);
  limit("pinned", options.pinned_host_bytes, 0);
  Json plan = {{"schema_version", 1},
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
  if (ks) {
    const auto incremental_requested = [](const char* name) {
      const auto control = dft::classify_incremental_direct_jk_control(std::getenv(name));
      if (control == dft::IncrementalDirectJkControl::Invalid)
        throw UsageError(std::string(name) + " must be 0/off or 1/on");
      return control == dft::IncrementalDirectJkControl::Enabled;
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
      {"energy_tolerance", options.energy_tolerance.value_or(1e-12)},
      {"density_tolerance", options.density_tolerance.value_or(1e-10)},
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
    const auto inventory = hf_inventory(orbital, auxiliary ? &*auxiliary : nullptr, atoms.size(),
                                        spins, options.diis_history, ecp_workspace);
    plan["item_inventory"] = inventory;
    workspace =
        std::max({inventory["preparation"].get<Bytes>(), inventory["iteration"].get<Bytes>(),
                  inventory["finalization"].get<Bytes>()});
    retained =
        sum({product({8, sum({product({4, spins, orbital.nbf, orbital.nbf}),
                              product({24, atoms.size()})})}),
             product({128, sum({orbital.primitives, orbital.shells, atoms.size()})}),
             auxiliary ? product({128, sum({auxiliary->primitives, auxiliary->shells})}) : 0});
  } else if (cpu && ks) {
    const auto inventory =
        ks_inventory(orbital, atoms.size(), spins, options, basis.ecp_terms.size(), ecp_workspace);
    plan["item_inventory"] = inventory;
    retained = inventory["retained"].get<Bytes>();
    workspace = std::max(inventory["setup_workspace"].get<Bytes>(),
                         inventory["scf_workspace"].get<Bytes>());
  } else if (hf && fitted) {
    if (auxiliary->nbf > 128) {
      plan["diagnostic"] = "CUDA DF inventory requires <=128 auxiliary AOs";
      return emit();
    }
    std::array<Bytes, 2> ordinary{};
    if (generativeqc_resource_small_hf_cuda_v2(
            orbital.nbf, orbital.cartesian_nbf, atoms.size(), orbital.shells,
            orbital.angular_values.data(), orbital.primitive_counts.data(), options.diis_history,
            spins, 0, options.energy_tolerance.value_or(1e-12),
            options.screening_tolerance.value_or(1e-14), ordinary.data(), ordinary.size())) {
      plan["diagnostic"] =
          "CUDA DF inventory requires a CUDA library and ordinary <=16-AO topology";
      return emit();
    }
    try {
      const auto candidates = cuda_df_candidates(
          orbital, *auxiliary, atoms.size(), spins,
          std::max<Bytes>(1, static_cast<Bytes>((electrons + unpaired) / 2)), options);
      plan["candidates"] = candidates;
      const auto peaks = [&](const Json& candidate) {
        const Bytes host_peak = sum({candidate.at("retained_host").get<Bytes>(),
                                     candidate.at("workspace_host").get<Bytes>()});
        const Bytes device_peak = sum({candidate.at("retained_device").get<Bytes>(),
                                       candidate.at("workspace_device").get<Bytes>(),
                                       candidate.at("runtime_allowance_device").get<Bytes>()});
        return Json{{"pageable", host_peak},
                    {"host", host_peak},
                    {"pinned", 0},
                    {"device", device_peak},
                    {"device:" + std::to_string(options.device_id), device_peak}};
      };
      const Json* selected = &candidates.at(0);
      bool feasible = false;
      for (const auto& candidate : candidates) {
        const auto peak = peaks(candidate);
        bool fits = true;
        for (auto cap = limits.begin(); cap != limits.end(); ++cap)
          if (peak.value(cap.key(), Bytes{0}) > cap.value().get<Bytes>()) fits = false;
        if (fits) {
          selected = &candidate;
          feasible = true;
          break;
        }
      }
      plan["selection"] = selected->at("name");
      plan["peak_bytes"] = peaks(*selected);
      plan["resident_bytes"] = {{"pageable", selected->at("retained_host")},
                                {"device:" + std::to_string(options.device_id),
                                 sum({selected->at("retained_device").get<Bytes>(),
                                      selected->at("runtime_allowance_device").get<Bytes>()})}};
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
    const auto inventory =
        ks_inventory(orbital, atoms.size(), spins, options, basis.ecp_terms.size(), ecp_workspace);
    plan["item_inventory"] = inventory;
    retained = inventory["retained"].get<Bytes>();
    workspace = std::max(inventory["setup_workspace"].get<Bytes>(),
                         inventory["scf_workspace"].get<Bytes>());
    const Bytes live = sum({capacities[0], capacities[1], capacities[2]});
    const Bytes pairs = orbital.shells * (orbital.shells + 1) / 2;
    const Bytes setup =
        std::max({product({64, sum({1, atoms.size(), orbital.shells, orbital.primitives,
                                    orbital.cartesian_nbf, pairs,
                                    product({orbital.cartesian_nbf, orbital.cartesian_nbf})})}),
                  sum({capacities[2], quadrature}), ecp_workspace});
    device = sum(
        {product({options.batch, sum({live, matrix_provider})}), setup > live ? setup - live : 0});
    plan["resident_bytes"]["device:" + std::to_string(options.device_id)] =
        product({options.batch, sum({live, matrix_provider})});
    plan["device_inventory"] = {{"state", capacities[0]},
                                {"xc_and_grid", capacities[1]},
                                {"coulomb", capacities[2]},
                                {"setup", setup},
                                {"matrix_provider", matrix_provider}};
  } else if (hf && !fitted) {
    std::array<Bytes, 2> values{};
    if (generativeqc_resource_small_hf_cuda_v2(
            orbital.nbf, orbital.cartesian_nbf, atoms.size(), orbital.shells,
            orbital.angular_values.data(), orbital.primitive_counts.data(), options.diis_history,
            spins, 0, options.energy_tolerance.value_or(1e-12),
            options.screening_tolerance.value_or(1e-14), values.data(), values.size())) {
      plan["diagnostic"] =
          "CUDA HF inventory requires a CUDA library and ordinary <=16-AO s/p/d/f topology";
      return emit();
    }
    const Bytes pairs = orbital.shells * (orbital.shells + 1) / 2;
    retained =
        sum({4 * values[1],
             product({2048, sum({1, atoms.size(), orbital.shells, orbital.primitives, orbital.nbf,
                                 orbital.cartesian_nbf, pairs, product({pairs, pairs})})}),
             product({512, spins, orbital.nbf, orbital.nbf}), product({384, atoms.size()}),
             ecp_workspace});
    device = sum({product({options.batch, values[0]}), values[0], ecp_workspace});
    plan["resident_bytes"]["device:" + std::to_string(options.device_id)] =
        product({options.batch, values[0]});
  } else {
    plan["diagnostic"] = "this CUDA provider has no qualified native dry-run inventory";
    return emit();
  }
  retained = product({options.batch, retained});
  const Bytes host = sum({retained, workspace});
  plan["peak_bytes"] = {{"pageable", host}, {"host", host}, {"pinned", 0}, {"device", device}};
  if (device) plan["peak_bytes"]["device:" + std::to_string(options.device_id)] = device;
  plan["resident_bytes"]["pageable"] = retained;
  plan["status"] = "feasible";
  for (auto cap = limits.begin(); cap != limits.end(); ++cap) {
    if (plan["peak_bytes"].value(cap.key(), Bytes{0}) > cap.value().get<Bytes>()) {
      plan["status"] = "infeasible";
      plan["diagnostic"] = "allocation inventory exceeds " + cap.key() + " budget";
      break;
    }
  }
  return emit();
}
}  // namespace generativeqc::cli
