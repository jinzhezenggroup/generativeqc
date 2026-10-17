#include <algorithm>
#include <array>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <vector>

#include "scf/hf_df_resources.hpp"

extern "C" int generativeqc_resource_hf_df_cuda_fleet_v1(const std::uint64_t*, std::size_t,
                                                         unsigned, std::uint64_t, unsigned,
                                                         unsigned, std::uint64_t*, std::size_t,
                                                         char*, std::size_t);

namespace {
using Bytes = std::uint64_t;
using Item = generativeqc::scf::CudaDfResourceItem;
using Storage = generativeqc::scf::DfPairStorageRequest;
using generativeqc::scf::cuda_df_fleet_resources;

void require(bool condition, const char* message) {
  if (!condition) throw std::runtime_error(message);
}

void check_composition() {
  const Item first{2, 2, 2, 2, 6, 2, 2, 2, 6, 1, 1, 1, 2};
  const Item second{3, 5, 6, 3, 9, 9, 10, 4, 12, 3, 2, 2, 1};
  for (const auto storage : {Storage::Automatic, Storage::Dense, Storage::SymmetricLower,
                             Storage::SymmetricLowerSingle}) {
    for (const bool occupied : {false, true}) {
      const std::array<Item, 2> items{second, first};
      const auto fleet = cuda_df_fleet_resources(items, 8, 0, storage, occupied);
      for (const auto& candidate : fleet) {
        auto left = cuda_df_fleet_resources(std::span(&first, 1), 8, candidate.sub_budget, storage,
                                            occupied);
        auto right = cuda_df_fleet_resources(std::span(&second, 1), 8, candidate.sub_budget,
                                             storage, occupied);
        const auto& first_bucket = left.front();
        const auto& second_bucket = right.front();
        require(
            candidate.retained_host == first_bucket.retained_host + second_bucket.retained_host &&
                candidate.retained_device ==
                    first_bucket.retained_device + second_bucket.retained_device,
            "DF retained buckets must coexist");
        require(candidate.workspace_host ==
                        std::max(first_bucket.workspace_host, second_bucket.workspace_host) &&
                    candidate.workspace_device ==
                        std::max(first_bucket.workspace_device, second_bucket.workspace_device),
                "DF setup/force/retry workspace must be serialized");
        require(candidate.runtime_allowance_device == 3u * (256u << 20),
                "DF provider allowance must cover every warm bucket and singleton retry");
        require(candidate.buckets.front().values[1] == 2 && candidate.buckets.back().values[1] == 5,
                "DF bucket order must follow the native scientific bucket key");
        require(
            candidate.buckets.front().values[25] == 1 && candidate.buckets.back().values[25] == 0,
            "RHF authorization must not be inferred for UHF");
      }
      auto singleton = first;
      singleton.repetitions = 1;
      const std::array<Item, 2> expanded{singleton, singleton};
      const auto repeated = cuda_df_fleet_resources(std::span(&first, 1), 8, 0, storage, occupied);
      const auto individual = cuda_df_fleet_resources(expanded, 8, 0, storage, occupied);
      for (std::size_t index = 0; index < repeated.size(); ++index) {
        require(repeated[index].inventory() == individual[index].inventory() &&
                    repeated[index].buckets.front().values ==
                        individual[index].buckets.front().values &&
                    repeated[index].buckets.front().energy_tiles ==
                        individual[index].buckets.front().energy_tiles &&
                    repeated[index].buckets.front().force_tiles ==
                        individual[index].buckets.front().force_tiles,
                "repeated CLI topology and expanded Python metadata must agree");
      }
    }
  }
}

void check_wire() {
  const std::array<Bytes, 13> item{2, 2, 2, 2, 6, 2, 2, 2, 6, 1, 1, 1, 1};
  std::array<char, 512> error{};
  std::array<Bytes, 3> header{};
  const auto query = [&](const auto& input, auto& output, Bytes budget = 0, unsigned storage = 0,
                         unsigned exchange = 0) {
    return generativeqc_resource_hf_df_cuda_fleet_v1(input.data(), input.size(), 8, budget, storage,
                                                     exchange, output.data(), output.size(),
                                                     error.data(), error.size());
  };
  require(!query(item, header) && header[0] == 1 && header[1] == 2,
          "DF wire sizing must report the complete two-candidate inventory");
  std::vector<Bytes> output(header[2], 314159);
  require(!query(item, output) && std::equal(header.begin(), header.end(), output.begin()),
          "DF full publication differs from its sizing query");
  for (const Bytes budget : {Bytes{1}, std::numeric_limits<Bytes>::max()}) {
    std::fill(output.begin(), output.end(), 314159);
    require(
        query(item, output, budget) != 0 &&
            std::all_of(output.begin(), output.end(), [](auto value) { return value == 314159; }),
        "DF failed budget request partially published its inventory");
  }
  for (std::size_t index = 0; index < item.size(); ++index) {
    auto invalid = item;
    invalid[index] = std::numeric_limits<Bytes>::max();
    header.fill(314159);
    require(query(invalid, header) != 0 && header == std::array<Bytes, 3>{314159, 314159, 314159},
            "DF invalid shape or overflow changed output");
  }
  for (const auto selectors : {std::array<unsigned, 2>{4, 0}, {0, 2}}) {
    require(query(item, header, 0, selectors[0], selectors[1]) != 0 && header[0] == 314159,
            "DF invalid policy selector was admitted");
  }
  for (const auto length : {0u, 2u, 4u, 150u}) {
    require(generativeqc_resource_hf_df_cuda_fleet_v1(item.data(), item.size(), 8, 0, 0, 0,
                                                      output.data(), length, error.data(),
                                                      error.size()) != 0 &&
                output.front() == 314159,
            "DF invalid publication length was accepted");
  }
  auto alias = item;
  require(generativeqc_resource_hf_df_cuda_fleet_v1(alias.data(), alias.size(), 8, 0, 0, 0,
                                                    alias.data() + 1, 3, error.data(),
                                                    error.size()) != 0 &&
              alias == item,
          "DF overlapping input/output corrupted topology");
  require(generativeqc_resource_hf_df_cuda_fleet_v1(
              item.data(), item.size(), 8, 0, 0, 0, output.data(), output.size(),
              reinterpret_cast<char*>(output.data()), 8) != 0 &&
              output.front() == 314159,
          "DF error/output alias corrupted the publication");
  require(generativeqc_resource_hf_df_cuda_fleet_v1(nullptr, item.size(), 8, 0, 0, 0, output.data(),
                                                    output.size(), error.data(), error.size()) != 0,
          "DF null input was accepted");
  require(generativeqc_resource_hf_df_cuda_fleet_v1(item.data(), 12, 8, 0, 0, 0, output.data(),
                                                    output.size(), error.data(), error.size()) != 0,
          "DF partial shape row was accepted");
  require(generativeqc_resource_hf_df_cuda_fleet_v1(
              item.data(), item.size(), 8, 0, 0, 0, output.data(),
              std::numeric_limits<std::size_t>::max(), error.data(), error.size()) != 0,
          "DF overflowing output span was read");
}
}  // namespace

void check_additional_wire_boundaries() {
  std::array<std::uint64_t, 26> shapes{2, 2, 2, 2, 6, 2, 2, 2, 6, 1, 1, 1, 1,
                                       2, 2, 2, 2, 6, 2, 2, 2, 6, 1, 1, 1, 1};
  std::array<std::uint64_t, 3> output{314159, 314159, 314159};
  std::array<char, 128> error{};
  const auto original = shapes;
  require(generativeqc_resource_hf_df_cuda_fleet_v1(
              shapes.data(), shapes.size(), 8, 0, 0, 0, output.data(), output.size(),
              reinterpret_cast<char*>(shapes.data()), 8) != 0 &&
              shapes == original,
          "DF input/error alias corrupted immutable topology");
  constexpr auto maximum_count = std::numeric_limits<std::size_t>::max();
  require(generativeqc_resource_hf_df_cuda_fleet_v1(
              shapes.data(), maximum_count - maximum_count % 13, 8, 0, 0, 0, output.data(),
              output.size(), error.data(), error.size()) != 0 &&
              output[0] == 314159,
          "DF overflowing input range was read");
  shapes[17] = std::uint64_t{1} << 61;
  require(generativeqc_resource_hf_df_cuda_fleet_v1(shapes.data(), shapes.size(), 8, 0, 0, 0,
                                                    output.data(), output.size(), error.data(),
                                                    error.size()) != 0 &&
              output == std::array<std::uint64_t, 3>{314159, 314159, 314159},
          "DF late intermediate overflow partially published an earlier bucket");
}

int main() {
  check_composition();
  check_wire();
  check_additional_wire_boundaries();
}
