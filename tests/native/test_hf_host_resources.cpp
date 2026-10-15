#include <algorithm>
#include <array>
#include <cstdint>
#include <limits>
#include <stdexcept>

#include "scf/hf_host_resources.hpp"

extern "C" int generativeqc_resource_hf_cpu_host_v1(const std::uint64_t*, std::size_t,
                                                    std::uint64_t*, std::size_t);
extern "C" int generativeqc_resource_hf_cpu_fleet_v1(const std::uint64_t*, std::size_t,
                                                     std::uint64_t*, std::size_t);

namespace {
void require(bool condition, const char* message) {
  if (!condition) throw std::runtime_error(message);
}

void check_fleet_capacity() {
  const std::array<std::uint64_t, 14> item{2, 2, 2, 1, 0, 8, 0, 0, 0, 0, 2, 6, 0, 0};
  std::array<std::uint64_t, 28> shapes{};
  std::copy(item.begin(), item.end(), shapes.begin());
  std::copy(item.begin(), item.end(), shapes.begin() + 14);
  std::array<std::uint64_t, 11> values{};
  const auto query = [&](const auto& request) {
    return generativeqc_resource_hf_cpu_fleet_v1(request.data(), request.size(), values.data(),
                                                 values.size());
  };
  const auto unchanged = [&] {
    return std::all_of(values.begin(), values.end(), [](auto value) { return value == 314159; });
  };
  require(query(shapes) == 0 && values[0] == 3584 && values[1] == 60360 && values[2] == 63944 &&
              values[3] == 60360 && values[7] == 60360 && values[10] == 1392,
          "fleet must sum retained state but share serialized workspace");
  generativeqc::scf::CpuHfHostFleetResources fleet;
  const generativeqc::scf::CpuHfHostResourceItem native_item{
      {2, 2, 2, 1, 0, 8, 0, 0, 0, 0}, 2, 6, 0, 0};
  fleet.append(native_item, 2);
  require(fleet.inventory() == std::array<std::uint64_t, 3>{3584, 60360, 63944},
          "homogeneous CLI repetition must match the ragged fleet owner");
  for (const auto copies : {std::uint64_t{0}, std::numeric_limits<std::uint64_t>::max()}) {
    bool rejected = false;
    try {
      fleet.append(native_item, copies);
    } catch (const std::exception&) {
      rejected = true;
    }
    require(rejected && fleet.inventory() == std::array<std::uint64_t, 3>{3584, 60360, 63944},
            "invalid append changed an existing fleet aggregate");
  }
  const std::array<std::uint64_t, 14> second{3, 5, 6, 2, 2, 2, 9, 15, 4, 0, 2, 6, 3, 9};
  std::copy(second.begin(), second.end(), shapes.begin() + 14);
  require(query(shapes) == 0 && values[0] == 6912 &&
              values[1] ==
                  std::max({values[3], values[4], values[5], values[7], values[8], values[9]}) &&
              values[2] == values[0] + values[1],
          "heterogeneous direct/DF fleet lifetimes were not composed");
  for (std::size_t slot = 0; slot < second.size(); ++slot) {
    auto invalid = shapes;
    invalid[14 + slot] = std::numeric_limits<std::uint64_t>::max();
    values.fill(314159);
    require(query(invalid) != 0 && unchanged(),
            "invalid later item published partial fleet values");
  }
  for (const auto slot : {0u, 10u, 11u, 12u, 13u}) {
    auto invalid = shapes;
    invalid[14 + slot] = 0;
    values.fill(314159);
    require(query(invalid) != 0 && unchanged(), "incomplete shell/primitive topology was accepted");
  }
  auto exact = item;
  exact[9] = std::numeric_limits<std::int64_t>::max() - 60360 - 1792;
  std::array<std::uint64_t, 7> single{};
  require(generativeqc_resource_hf_cpu_fleet_v1(exact.data(), exact.size(), single.data(),
                                                single.size()) == 0 &&
              single[2] == std::numeric_limits<std::int64_t>::max(),
          "exact signed-int64 fleet peak was rejected");
  ++exact[9];
  single.fill(314159);
  require(generativeqc_resource_hf_cpu_fleet_v1(exact.data(), exact.size(), single.data(),
                                                single.size()) != 0 &&
              single[0] == 314159 && single[6] == 314159,
          "retained plus workspace overflow published a fleet");
  const auto original = shapes;
  require(generativeqc_resource_hf_cpu_fleet_v1(shapes.data(), shapes.size(), shapes.data(),
                                                values.size()) != 0 &&
              shapes == original,
          "overlapping output corrupted immutable input");
  for (const auto lengths : {std::array<std::size_t, 2>{0, 3}, {27, 11}, {28, 10}, {28, 12}}) {
    values.fill(314159);
    require(generativeqc_resource_hf_cpu_fleet_v1(shapes.data(), lengths[0], values.data(),
                                                  lengths[1]) != 0 &&
                unchanged(),
            "wrong fleet schema length published capacities");
  }
  constexpr auto maximum_count = std::numeric_limits<std::size_t>::max();
  constexpr auto oversized_shapes = maximum_count - maximum_count % 14;
  require(generativeqc_resource_hf_cpu_fleet_v1(shapes.data(), oversized_shapes, values.data(),
                                                3 + 4 * (oversized_shapes / 14)) != 0 &&
              unchanged(),
          "overflowing buffer range was read as an actual fleet");
  require(generativeqc_resource_hf_cpu_fleet_v1(nullptr, 28, values.data(), values.size()) != 0 &&
              unchanged() &&
              generativeqc_resource_hf_cpu_fleet_v1(shapes.data(), 28, nullptr, values.size()) != 0,
          "null fleet shape or publication buffer was accepted");
}
}  // namespace

int main() {
  check_fleet_capacity();
  const std::array<std::uint64_t, 10> shape{2, 2, 2, 1, 0, 8, 0, 0, 0, 0};
  std::array<std::uint64_t, 4> inventory{};
  const auto query = [&](const auto& request) {
    return generativeqc_resource_hf_cpu_host_v1(request.data(), request.size(), inventory.data(),
                                                inventory.size());
  };
  const auto unchanged = [&] {
    return std::all_of(inventory.begin(), inventory.end(),
                       [](auto value) { return value == 314159; });
  };
  require(query(shape) == 0 && inventory == std::array<std::uint64_t, 4>{60360, 5312, 5312, 1392},
          "H2 CPU HF lifetime bounds changed");
  for (const auto spins : {1u, 2u}) {
    for (const auto history : {0u, 1u, 8u, 64u}) {
      auto fitted = shape;
      fitted[3] = spins;
      fitted[5] = history;
      fitted[6] = 5;
      fitted[7] = 6;
      fitted[8] = 2;
      require(query(fitted) == 0, "DF or unrestricted topology was rejected");
      const auto expected =
          generativeqc::scf::cpu_hf_host_resources({2, 2, 2, spins, 0, history, 5, 6, 2, 0});
      require(inventory == expected && inventory[3] > 1392 && inventory[2] > inventory[1],
              "DF derivative response owners are missing");
    }
  }
  for (std::size_t slot = 0; slot < shape.size(); ++slot) {
    auto invalid = shape;
    invalid[slot] = std::numeric_limits<std::uint64_t>::max();
    inventory.fill(314159);
    require(query(invalid) != 0 && unchanged(), "overflow published a partial CPU HF inventory");
  }
  for (const auto slot : {0u, 1u, 2u, 3u}) {
    auto invalid = shape;
    invalid[slot] = 0;
    inventory.fill(314159);
    require(query(invalid) != 0 && unchanged(), "zero CPU HF topology was accepted");
  }
  for (const auto slot : {4u, 8u}) {
    auto invalid = shape;
    invalid[slot] = 5;
    inventory.fill(314159);
    require(query(invalid) != 0 && unchanged(), "unsupported angular domain was accepted");
  }
  for (const auto slot : {6u, 7u, 8u}) {
    auto invalid = shape;
    invalid[slot] = 1;
    inventory.fill(314159);
    require(query(invalid) != 0 && unchanged(), "incomplete auxiliary topology was accepted");
  }
  auto exact = shape;
  exact[9] = std::numeric_limits<std::int64_t>::max() - 60360;
  require(query(exact) == 0 && inventory[0] == std::numeric_limits<std::int64_t>::max(),
          "exact signed-int64 capacity was rejected");
  ++exact[9];
  inventory.fill(314159);
  require(query(exact) != 0 && unchanged(), "signed-int64 boundary overflow was accepted");
  for (const auto lengths : {std::array<std::size_t, 2>{9, 4}, {11, 4}, {10, 3}, {10, 5}}) {
    require(generativeqc_resource_hf_cpu_host_v1(shape.data(), lengths[0], inventory.data(),
                                                 lengths[1]) != 0 &&
                unchanged(),
            "wrong schema length published an inventory");
  }
  require(generativeqc_resource_hf_cpu_host_v1(nullptr, 10, inventory.data(), 4) != 0 &&
              unchanged() &&
              generativeqc_resource_hf_cpu_host_v1(shape.data(), 10, nullptr, 4) != 0,
          "null shape or output was accepted");
}
