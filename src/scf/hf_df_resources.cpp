#include "scf/hf_df_resources.hpp"

#include <algorithm>
#include <initializer_list>
#include <limits>
#include <map>
#include <stdexcept>

#include "scf/density_fitting.hpp"

namespace generativeqc::scf {
namespace {
using Bytes = std::uint64_t;
constexpr Bytes maximum = std::min<Bytes>(std::numeric_limits<std::int64_t>::max(),
                                          std::numeric_limits<std::size_t>::max());

Bytes sum(std::initializer_list<Bytes> values) {
  Bytes result = 0;
  for (const auto value : values) {
    if (value > maximum - result) throw std::overflow_error("CUDA DF capacity sum overflows");
    result += value;
  }
  return result;
}

Bytes product(std::initializer_list<Bytes> values) {
  Bytes result = 1;
  for (const auto value : values) {
    if (value > maximum || (value && result > maximum / value))
      throw std::overflow_error("CUDA DF capacity product overflows");
    result *= value;
  }
  return result;
}

Bytes triangular(Bytes dimension) {
  const auto successor = sum({dimension, 1});
  return dimension % 2 ? product({dimension, successor / 2}) : product({dimension / 2, successor});
}

Bytes diis_bytes(Bytes batch, Bytes nbf, unsigned history) {
  return sum({density_fitting_scf_diis_device_bytes(batch, nbf, history)});
}

CudaDfResourceTile tiles(Bytes batch, Bytes nbf, Bytes naux, Bytes occupied, Bytes budget,
                         Bytes fixed, bool generated, bool packed, Bytes rhf_rank) {
  const auto plan = packed ? plan_packed_density_fitting_tiles(batch, nbf, naux, occupied, budget,
                                                               fixed, rhf_rank)
                           : plan_density_fitting_tiles(batch, nbf, naux, occupied, budget, fixed,
                                                        generated, rhf_rank);
  DfPackedValueCapacity capacity{};
  if (packed) capacity = df_packed_value_capacity(batch, nbf, naux, occupied, plan.auxiliary_tile);
  const auto dense_elements = product({nbf, nbf, naux});
  const auto materialization = packed ? product({8, nbf, nbf, plan.auxiliary_tile}) : 0;
  CudaDfResourceTile result{plan.batch_tile,
                            plan.ao_pair_tile,
                            plan.auxiliary_tile,
                            plan.occupied_tile,
                            plan.peak_workspace_bytes,
                            plan.stores_full_three_center,
                            budget,
                            fixed,
                            generated,
                            packed ? occupied : 0,
                            capacity.factor_bytes,
                            capacity.factor_bytes,
                            capacity.scratch_bytes,
                            capacity.projection_elements,
                            capacity.panel_elements,
                            plan.automatic_rhf_rank,
                            packed ? capacity.tensor_per_system : dense_elements,
                            dense_elements,
                            materialization,
                            product({2, materialization})};
  for (const auto value : result) static_cast<void>(sum({value}));
  return result;
}

struct Row {
  CudaDfResourceItem first;
  Bytes batch{}, source{}, diis{}, metadata{}, preparation_metadata{}, preparation_retained{};
  Bytes preparation_temporary{}, response_metadata{}, preparation_minimum{}, response_minimum{};
  Bytes resident_response{}, response{}, response_host{};
  CudaDfResourceTile default_tile{};
};

Row make_row(std::span<const CudaDfResourceItem> items, unsigned history, bool packed) {
  Row row{items.front()};
  const auto& first = row.first;
  const auto nbf = first.nbf, cartesian = first.cartesian_nbf;
  const auto naux = first.naux, auxiliary_cartesian = first.cartesian_naux;
  const auto square = product({nbf, nbf}), auxiliary_square = product({naux, naux});
  const auto coordinates = product({3, first.atoms});
  Bytes atoms = 0, shells = 0, primitives = 0;
  for (const auto& item : items) {
    if (item.cartesian_nbf != cartesian || item.naux != naux ||
        item.cartesian_naux != auxiliary_cartesian || item.atoms != first.atoms ||
        item.spins != first.spins)
      throw CudaDfResourceUnsupported(
          "CUDA DF bucket has incompatible Cartesian/auxiliary/atom dimensions or spin");
    const auto repetitions = item.repetitions;
    row.batch = sum({row.batch, repetitions});
    atoms = sum({atoms, product({repetitions, item.atoms})});
    const auto combined_shells = sum({item.shells, item.auxiliary_shells, 1});
    shells = sum({shells, product({repetitions, combined_shells})});
    primitives = sum(
        {primitives, product({repetitions, sum({item.primitives, item.auxiliary_primitives, 1})})});
    const auto preparation = product(
        {512, sum({1, item.atoms, item.shells, item.primitives, cartesian, nbf,
                   triangular(item.shells), item.auxiliary_shells, item.auxiliary_primitives})});
    row.preparation_metadata = sum({row.preparation_metadata, product({repetitions, preparation})});
    row.preparation_retained = sum(
        {row.preparation_retained,
         product({repetitions,
                  sum({preparation, product({8, sum({product({2, square}), coordinates})})})})});
    row.preparation_temporary =
        std::max(row.preparation_temporary,
                 sum({preparation,
                      product({8, sum({product({6, cartesian, cartesian}), coordinates, 1})})}));
    const auto pairs = triangular(combined_shells);
    row.metadata = sum(
        {row.metadata,
         product({repetitions, 2048,
                  sum({1, item.atoms, combined_shells, item.primitives, item.auxiliary_primitives,
                       cartesian, auxiliary_cartesian, pairs, product({pairs, pairs})})})});
    row.response_metadata = std::max(row.response_metadata,
                                     product({256, sum({1, item.atoms, nbf, naux, item.primitives,
                                                        item.auxiliary_primitives})}));
  }
  row.source = sum({density_fitting_source_metadata_bytes(
      row.batch, atoms, shells, product({row.batch, sum({cartesian, auxiliary_cartesian, 1})}),
      primitives,
      product(
          {row.batch, sum({product({nbf, cartesian}), product({naux, auxiliary_cartesian})})}))});
  row.diis = diis_bytes(row.batch, nbf, history);
  const auto occupied = std::max({first.nalpha, first.nbeta, Bytes{1}});
  row.default_tile = tiles(row.batch, nbf, naux, occupied, 0, sum({row.source, row.diis}), true,
                           packed, first.spins == 1 ? first.nalpha : 0);
  row.preparation_minimum =
      sum({row.preparation_metadata, row.preparation_retained, row.preparation_temporary});
  row.response_minimum =
      sum({row.response_metadata, product({8, sum({product({4, auxiliary_square}),
                                                   product({9, square}), product({6, naux})})})});
  row.resident_response = sum(
      {row.response_metadata,
       std::max(product({8, sum({product({12, auxiliary_square}), product({10, naux}), square})}),
                sum({product({8, sum({product({3, auxiliary_square}), product({6, naux}),
                                      product({4, square})})}),
                     product({16, square, naux})}))});
  row.response = sum(
      {row.response_metadata,
       product({8, sum({product({4, auxiliary_square}),
                        product({sum({7, product({2, naux})}), square}), product({6, naux})})})});
  row.response_host = sum({row.response_metadata, product({8, square})});
  return row;
}

void append_bucket(CudaDfResourceCandidate& candidate, const Row& row, unsigned history,
                   bool packed, bool occupied_exchange) {
  const auto& first = row.first;
  const auto batch = row.batch, nbf = first.nbf, naux = first.naux;
  const auto cartesian = first.cartesian_nbf, auxiliary_cartesian = first.cartesian_naux;
  const auto square = product({nbf, nbf}), coordinates = product({3, first.atoms});
  const auto budget = candidate.sub_budget;
  const bool source = budget > 0;
  const auto occupied = std::max({first.nalpha, first.nbeta, Bytes{1}});
  const auto rhf_rank = first.spins == 1 ? first.nalpha : 0;
  if (source && budget < row.preparation_minimum)
    throw std::invalid_argument(
        "DF sub-budget cannot hold the native one-electron preparation minimum");
  if (source && budget / 2 < row.response_minimum)
    throw std::invalid_argument("DF sub-budget cannot hold the generated response minimum");
  const auto tile = tiles(batch, nbf, naux, occupied, budget,
                          sum({source ? row.source : 0, row.diis}), source, packed, rhf_rank);
  const auto force_tile = source ? tiles(batch, nbf, naux, occupied, std::max<Bytes>(1, budget / 2),
                                         sum({row.source, row.diis}), true, packed, rhf_rank)
                                 : tile;
  const auto pairs =
      source ? std::min(square, std::max(nbf, product({tile[1] / nbf, nbf}))) : square;
  const auto tile_bytes = product({8, pairs, source ? tile[2] : naux});
  const auto matrix = product({8, batch, square});
  const auto metric = product({8, batch, naux, naux});
  const auto tensor = product({8, batch, square, naux});
  const auto solver = sum({64 << 20, product({16, matrix}), product({128, naux, naux})});
  const bool factors = occupied_exchange || tile[15];
  const auto eigen_host = sum({1 << 20, product({128, square})});
  const auto eigen_device = sum({eigen_host, product({8, sum({product({3, square}), nbf})}), 5});
  const auto snapshot = product({batch, sum({product({16, sum({square, nbf})}), 24})});
  auto device = sum({product({factors ? 34u : 32u, matrix}), product({16, batch, naux}), solver,
                     product({1024, batch}), eigen_device, snapshot, row.diis, product({2, metric}),
                     product({8, batch, naux})});
  device = sum({device, packed   ? sum({tile[10], tile[11], tile[12], row.source})
                        : source ? sum({tile[5] ? sum({tensor, product({3, tile_bytes})})
                                                : product({4, tile_bytes}),
                                        row.source})
                                 : sum({tensor, product({3, tile_bytes})})});
  const auto setup = sum({metric, product({8, batch, naux}), solver, product({4, batch})});
  const auto force =
      source ? std::min(row.response, budget / 2)
             : std::max(row.resident_response, std::min<Bytes>(row.response, 128 << 20));
  const auto generation =
      sum({row.source,
           product({8, batch,
                    sum({product({2, cartesian, cartesian}),
                         source ? 0
                                : sum({product({auxiliary_cartesian, auxiliary_cartesian}),
                                       product({cartesian, cartesian, auxiliary_cartesian})})})})});
  const auto overlap = sum({product({2, matrix}), product({8, batch, coordinates})});
  const auto one_electron = product({8, batch, sum({product({2, square}), coordinates})});
  const auto raw = product({8, batch, sum({product({naux, naux}), product({square, naux})})});
  auto host = sum({row.metadata,
                   product({8, batch, sum({product({32, square}), product({16, coordinates})})}),
                   overlap, eigen_host, product({64, batch}), one_electron});
  if (!source) host = sum({host, raw, tensor});
  const auto history_count = sum({history, 1});
  const auto scf = product({8, batch,
                            sum({product({129, square}), product({4, history_count, square}),
                                 product({4, history_count, history_count})})});
  const auto selection = product({8, sum({product({32, square}), product({8, nbf})})});
  auto host_work =
      sum({64 << 20, scf, one_electron, product({64, metric / batch}), product({8, tensor / batch}),
           product({2, row.metadata}), selection,
           source ? product({2, tile_bytes})
                  : sum({product({2, raw}),
                         product({8, batch,
                                  sum({product({auxiliary_cartesian, auxiliary_cartesian}),
                                       product({cartesian, cartesian, auxiliary_cartesian})})})})});
  host_work = sum({std::max(host_work, row.preparation_temporary),
                   source ? row.response_host : row.resident_response});
  const auto retry = tiles(1, nbf, naux, occupied, budget,
                           sum({source ? row.source / batch : 0, diis_bytes(1, nbf, history)}),
                           source, packed, rhf_rank);
  /** The ordinary workspace is bucket-serialized, not divisible by batch.
   * A singleton retry gets its whole workspace and its own factor decision. */
  const auto device_work =
      sum({std::max({setup, force, generation}), (device - eigen_device) / batch, eigen_device,
           solver, retry[15] && !factors ? product({2, matrix}) / batch : 0});
  candidate.retained_host = sum({candidate.retained_host, host});
  candidate.retained_device = sum({candidate.retained_device, device});
  candidate.workspace_host = std::max(candidate.workspace_host, host_work);
  candidate.workspace_device = std::max(candidate.workspace_device, device_work);
  candidate.buckets.push_back({{batch,
                                nbf,
                                naux,
                                cartesian,
                                auxiliary_cartesian,
                                coordinates,
                                row.source,
                                row.diis,
                                occupied,
                                rhf_rank,
                                row.metadata,
                                row.preparation_minimum,
                                row.preparation_retained,
                                row.preparation_temporary,
                                row.response_minimum,
                                row.resident_response,
                                row.response,
                                row.response_host,
                                host,
                                overlap,
                                eigen_device,
                                snapshot,
                                selection,
                                eigen_host,
                                device,
                                first.spins == 1},
                               tile,
                               force_tile});
}
}  // namespace

std::vector<CudaDfResourceCandidate> cuda_df_fleet_resources(
    std::span<const CudaDfResourceItem> items, unsigned diis_history, Bytes requested_budget,
    DfPairStorageRequest storage, bool occupied_exchange) {
  if (items.empty() || static_cast<unsigned>(storage) > 3)
    throw std::invalid_argument("invalid CUDA DF fleet or storage selector");
  static_cast<void>(sum({requested_budget}));
  const bool packed = storage == DfPairStorageRequest::SymmetricLower ||
                      storage == DfPairStorageRequest::SymmetricLowerSingle;
  std::map<std::array<Bytes, 4>, std::vector<CudaDfResourceItem>> groups;
  for (const auto& item : items) {
    for (const auto value : {item.atoms, item.nbf, item.cartesian_nbf, item.shells, item.primitives,
                             item.naux, item.cartesian_naux, item.auxiliary_shells,
                             item.auxiliary_primitives, item.nalpha, item.nbeta, item.repetitions})
      static_cast<void>(sum({value}));
    if (item.nbf > 16 || item.naux > 128)
      throw CudaDfResourceUnsupported(
          "CUDA DF inventory v1 requires <=16 orbital AOs and <=128 auxiliary AOs");
    if (!item.atoms || !item.nbf || !item.naux || item.cartesian_nbf < item.nbf ||
        item.cartesian_naux < item.naux || !item.shells || !item.primitives ||
        !item.auxiliary_shells || !item.auxiliary_primitives || !item.repetitions ||
        (item.spins != 1 && item.spins != 2) || item.nalpha > item.nbf || item.nbeta > item.nbf ||
        (item.spins == 1 && item.nalpha != item.nbeta))
      throw std::invalid_argument(
          "invalid CUDA DF topology: requires <=16 orbital AOs and <=128 auxiliary AOs");
    groups[{item.nbf, item.nalpha, item.nbeta, item.primitives}].push_back(item);
  }
  std::vector<Row> rows;
  Bytes source_budget = requested_budget;
  for (const auto& [key, group] : groups) {
    rows.push_back(make_row(group, diis_history, packed));
    if (!requested_budget) {
      const auto& row = rows.back();
      source_budget =
          std::max({source_budget, row.preparation_minimum, product({2, row.default_tile[4]}),
                    product({2, row.response_minimum})});
    }
  }
  std::vector<CudaDfResourceCandidate> candidates;
  if (!requested_budget && !packed) candidates.push_back({0, 0, 0});
  candidates.push_back({packed ? 2u : 1u, source_budget, packed ? 0u : 1u});
  for (auto& candidate : candidates) {
    for (const auto& row : rows)
      append_bucket(candidate, row, diis_history, packed, occupied_exchange);
    candidate.runtime_allowance_device = product({256 << 20, sum({rows.size(), 1})});
    /** Validate simultaneous-live endpoints too, not only individual buffers. */
    static_cast<void>(sum({candidate.retained_host, candidate.workspace_host}));
    static_cast<void>(sum({candidate.retained_device, candidate.workspace_device,
                           candidate.runtime_allowance_device}));
  }
  return candidates;
}
}  // namespace generativeqc::scf
