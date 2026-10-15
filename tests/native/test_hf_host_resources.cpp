#include <algorithm>
#include <array>
#include <cstdint>
#include <limits>
#include <stdexcept>

#include "scf/hf_host_resources.hpp"

extern "C" int generativeqc_resource_hf_cpu_host_v1(const std::uint64_t*, std::size_t,
                                                    std::uint64_t*, std::size_t);

namespace {
void require(bool condition, const char* message) {
  if (!condition) throw std::runtime_error(message);
}
}  // namespace

int main() {
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
