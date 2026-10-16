#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <limits>

#include "scf/hf_cuda_resources.hpp"

extern "C" int generativeqc_resource_hf_cuda_fleet_v1(const std::uint64_t*, std::size_t,
                                                      std::uint64_t*, std::size_t);

namespace {
using Bytes = std::uint64_t;

void require(bool condition, const char* message) {
  if (!condition) {
    std::cerr << message << '\n';
    std::exit(1);
  }
}

void check_scalar_owner() {
  using namespace generativeqc::scf;
  const CudaHfResourceShape first{2, 2, 2, 2, 6, 1, 4096, 256, 0};
  const CudaHfResourceShape second{3, 7, 8, 4, 12, 2, 1024, 2048, 8192};
  require(cuda_hf_item_resources(first) == std::array<Bytes, 3>{4096, 59136, 0},
          "H2 host capacity changed");
  CudaHfFleetResources fleet;
  const auto first_item = fleet.append(first);
  const auto second_item = fleet.append(second);
  const auto expected = std::array<Bytes, 4>{5120, 12288, 17408, first_item[1] + second_item[1]};
  require(fleet.inventory() == expected, "fleet lost separate arena/ECP retry maxima");
  CudaHfFleetResources reversed;
  reversed.append(second);
  reversed.append(first);
  require(reversed.inventory() == expected, "fleet capacity depends on input order");
  CudaHfFleetResources repeated;
  repeated.append(first, 3);
  CudaHfFleetResources explicit_fleet;
  for (unsigned index = 0; index < 3; ++index) explicit_fleet.append(first);
  require(repeated.inventory() == explicit_fleet.inventory() &&
              repeated.inventory() == std::array<Bytes, 4>{12288, 4096, 16384, 3 * 59136},
          "homogeneous multiplicity was applied twice or to the serialized retry");
  for (const auto copies : {Bytes{0}, std::numeric_limits<Bytes>::max()}) {
    bool rejected = false;
    try {
      fleet.append(first, copies);
    } catch (const std::exception&) {
      rejected = true;
    }
    require(rejected && fleet.inventory() == expected,
            "invalid append changed already published fleet state");
  }
}

void check_bridge() {
  const std::array<Bytes, 18> shapes{2, 2, 2, 2, 6,  1, 4096, 256,  0,
                                     3, 7, 8, 4, 12, 2, 1024, 2048, 8192};
  std::array<Bytes, 10> values{};
  const auto query = [&](const auto& request) {
    return generativeqc_resource_hf_cuda_fleet_v1(request.data(), request.size(), values.data(),
                                                  values.size());
  };
  const auto unchanged = [&] {
    return std::all_of(values.begin(), values.end(), [](auto value) { return value == 314159; });
  };
  require(query(shapes) == 0 && values[0] == 5120 && values[1] == 12288 && values[2] == 17408 &&
              values[3] == values[5] + values[8] && values[4] == 4096 && values[5] == 59136 &&
              values[6] == 0 && values[7] == 1024 && values[9] == 8192,
          "compiled bridge did not publish the complete ragged fleet");
  const auto valid = values;
  auto reversed = shapes;
  std::swap_ranges(reversed.begin(), reversed.begin() + 9, reversed.begin() + 9);
  require(query(reversed) == 0 && std::equal(valid.begin(), valid.begin() + 4, values.begin()) &&
              std::equal(valid.begin() + 4, valid.begin() + 7, values.begin() + 7) &&
              std::equal(valid.begin() + 7, valid.end(), values.begin() + 4),
          "bridge lost item order while composing an order-independent fleet");
  for (std::size_t slot = 9; slot < shapes.size(); ++slot) {
    auto invalid = shapes;
    invalid[slot] = std::numeric_limits<Bytes>::max();
    values.fill(314159);
    require(query(invalid) != 0 && unchanged(), "invalid later item published partial capacities");
  }
  for (const auto slot : {9u, 10u, 12u, 13u, 14u}) {
    auto invalid = shapes;
    invalid[slot] = 0;
    values.fill(314159);
    require(query(invalid) != 0 && unchanged(), "zero later topology published capacities");
  }
  auto invalid = shapes;
  invalid[11] = invalid[10] - 1;
  values.fill(314159);
  require(query(invalid) != 0 && unchanged(), "public AOs exceeded Cartesian capacity");
  invalid = shapes;
  invalid[6] = invalid[15] = std::numeric_limits<std::int64_t>::max() / 3 + 1;
  require(query(invalid) != 0 && unchanged(), "aggregate overflow published valid item prefixes");
  for (const auto lengths : {std::array<std::size_t, 2>{0, 10}, {17, 10}, {18, 9}, {18, 11}}) {
    require(generativeqc_resource_hf_cuda_fleet_v1(shapes.data(), lengths[0], values.data(),
                                                   lengths[1]) != 0 &&
                unchanged(),
            "wrong schema lengths were accepted");
  }
  require(generativeqc_resource_hf_cuda_fleet_v1(nullptr, 18, values.data(), values.size()) != 0 &&
              unchanged() &&
              generativeqc_resource_hf_cuda_fleet_v1(shapes.data(), 18, nullptr, 10) != 0,
          "null input or output was accepted");
  auto alias = shapes;
  require(generativeqc_resource_hf_cuda_fleet_v1(alias.data(), alias.size(), alias.data() + 1,
                                                 values.size()) != 0 &&
              alias == shapes,
          "partially overlapping input/output modified the shape buffer");
  constexpr auto maximum = std::numeric_limits<std::size_t>::max();
  constexpr auto oversized_shapes = maximum - maximum % 9;
  require(generativeqc_resource_hf_cuda_fleet_v1(shapes.data(), oversized_shapes, values.data(),
                                                 4 + 3 * (oversized_shapes / 9)) != 0 &&
              unchanged(),
          "overflowing address range was treated as a real fleet");
}
}  // namespace

int main() {
  check_scalar_owner();
  check_bridge();
}
