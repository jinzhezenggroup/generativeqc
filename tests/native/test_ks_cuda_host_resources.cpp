#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>

#include "scf/ks_host_resources.hpp"

extern "C" int generativeqc_resource_ks_cuda_host_fleet_v1(const std::uint64_t*, std::size_t,
                                                           std::uint64_t, std::uint64_t,
                                                           std::uint64_t*, std::size_t);

namespace {

void require(bool condition, const char* message) {
  if (!condition) throw std::runtime_error(message);
}

constexpr generativeqc::scf::KsHostResourceShape small_shape{
    2, 2, 2, 2, 6, 1, 0, 0, 6144, 24, 8, 256, 8, 100, true, false, 0};
constexpr std::array<std::uint64_t, 18> wire_shape{2,  2, 2,   2, 6,   1, 0, 0, 6144,
                                                   24, 8, 256, 8, 100, 1, 0, 0, 0};

void test_storage_domains() {
  using namespace generativeqc::scf;
  const auto fused = ks_host_resources(small_shape, KsHostExecution::cuda_fused);
  const std::array<std::uint64_t, 16> expected{31744, 221184, 400, 448, 25600, 64,     0,     16384,
                                               16384, 32768,  0,   0,   0,     344976, 69984, 5968};
  require(fused == expected, "CUDA host components differ from independent LP64 inventory");
  const auto unfused = ks_host_resources(small_shape, KsHostExecution::cuda_host_unfused);
  require(unfused[0] == 15360 && unfused[10] == 64 && unfused[13] == 328656 && unfused[15] == 22384,
          "host-unfused XC must retain staging and use a transient AO tile");
  auto unrestricted = small_shape;
  unrestricted.spins = 2;
  const auto spin_inventory = ks_host_resources(unrestricted, KsHostExecution::cuda_host_unfused);
  require(spin_inventory[10] == 192 && spin_inventory[13] == 328912 && spin_inventory[15] == 27088,
          "UKS split XC matrices were not retained independently");
  auto large = small_shape;
  large.orbital_aos = large.cartesian_aos = 17;
  require(ks_host_resources(large, KsHostExecution::cuda_fused)[6] == 1048576 + 128 * 17 * 17,
          "large solver must reserve the existing ordinary provider workspace");
  large.orbital_aos = large.cartesian_aos = 16;
  require(ks_host_resources(large, KsHostExecution::cuda_fused)[6] == 0,
          "small native solver must not charge provider numeric workspace");
  large.maximum_angular = 5;
  large.orbital_aos = large.cartesian_aos = 100000;
  require(ks_host_resources(large, KsHostExecution::cuda_fused)[14] > 0,
          "CUDA shape composition must not evaluate CPU fourth powers or through-g admission");
  try {
    (void)cpu_ks_host_resources(large);
    require(false, "CPU through-g contract was broadened by shared host composition");
  } catch (const std::invalid_argument&) {
  }
}

void test_consumer_lifetimes_and_rollback() {
  using namespace generativeqc::scf;
  constexpr std::uint64_t allowance = (128U + 256U) << 20;
  KsHostFleetResources fleet(0, allowance);
  const auto item = fleet.append(small_shape, 3, KsHostExecution::cuda_fused);
  auto before = fleet.inventory();
  require(before[12] == allowance + 3 * item[13] && before[13] == 69984 &&
              before[14] == before[12] + before[13],
          "retained public consumers must coexist once with every replay, not per item");
  KsHostFleetResources repeated(0, allowance);
  for (std::size_t index = 0; index < 3; ++index)
    repeated.append(small_shape, 1, KsHostExecution::cuda_fused);
  require(repeated.inventory() == before, "homogeneous repetitions changed host lifetime");
  for (const auto copies : {std::uint64_t{0}, std::numeric_limits<std::uint64_t>::max()}) {
    try {
      fleet.append(small_shape, copies, KsHostExecution::cuda_fused);
      require(false, "invalid CUDA host repetition accepted");
    } catch (const std::invalid_argument&) {
    } catch (const std::overflow_error&) {
    }
    require(fleet.inventory() == before, "failed host append changed earlier retained owners");
  }
  constexpr auto maximum = static_cast<std::uint64_t>(std::numeric_limits<std::int64_t>::max());
  KsHostFleetResources exact(0, maximum - item[13] - item[14]);
  exact.append(small_shape, 1, KsHostExecution::cuda_fused);
  before = exact.inventory();
  require(before[14] == maximum, "exact int64 retained-plus-transient peak was refused");
  try {
    exact.append(small_shape, 1, KsHostExecution::cuda_fused);
    require(false, "aggregate host overflow was accepted");
  } catch (const std::overflow_error&) {
  }
  require(exact.inventory() == before, "aggregate overflow changed the fleet");
  KsHostFleetResources transient(256U << 20);
  transient.append(small_shape);
  const auto cpu = transient.inventory();
  require(cpu[12] == 263184 && cpu[13] == (256U << 20),
          "CPU force allowance must remain transient, not inherit CUDA retained policy");
}

void test_transactional_bridge() {
  constexpr std::uint64_t sentinel = 314159;
  constexpr std::uint64_t allowance = (128U + 256U) << 20;
  std::array<std::uint64_t, 36> shapes{};
  std::copy(wire_shape.begin(), wire_shape.end(), shapes.begin());
  std::copy(wire_shape.begin(), wire_shape.end(), shapes.begin() + 18);
  shapes[35] = 1;
  std::array<std::uint64_t, 51> values;
  values.fill(sentinel);
  require(generativeqc_resource_ks_cuda_host_fleet_v1(shapes.data(), shapes.size(), allowance, 0,
                                                      values.data(), 47) == 0,
          "CPU SDK must expose CUDA host shape composition without a CUDA context");
  require(values[12] == allowance + 344976 + 328656 && values[13] == 69984 &&
              values[14] == values[12] + values[13] && values[47] == sentinel,
          "ragged schedule totals or output tail are incorrect");
  const auto succeeds = values;
  for (const auto slot : {18u, 23u, 32u, 33u, 34u, 35u}) {
    auto invalid = shapes;
    invalid[slot] = slot == 23 || slot == 32 || slot == 33 || slot == 35
                        ? 3
                        : std::numeric_limits<std::uint64_t>::max();
    values.fill(sentinel);
    const auto before = values;
    require(generativeqc_resource_ks_cuda_host_fleet_v1(invalid.data(), invalid.size(), allowance,
                                                        0, values.data(), 47) != 0 &&
                values == before,
            "a malformed later descriptor partly published a host plan");
  }
  values.fill(sentinel);
  for (const auto count : {std::size_t{0}, std::size_t{46}, std::size_t{48}}) {
    const auto before = values;
    require(generativeqc_resource_ks_cuda_host_fleet_v1(shapes.data(), shapes.size(), allowance, 0,
                                                        values.data(), count) != 0 &&
                values == before,
            "wrong output schema mutated storage");
  }
  require(generativeqc_resource_ks_cuda_host_fleet_v1(nullptr, 36, allowance, 0, values.data(),
                                                      47) != 0 &&
              generativeqc_resource_ks_cuda_host_fleet_v1(shapes.data(), 36, allowance, 0, nullptr,
                                                          47) != 0,
          "null buffers were accepted");
  std::array<std::uint64_t, 128> alias{};
  std::copy(shapes.begin(), shapes.end(), alias.begin());
  const auto before_alias = alias;
  require(generativeqc_resource_ks_cuda_host_fleet_v1(alias.data(), 36, allowance, 0,
                                                      alias.data() + 18, 47) != 0 &&
              alias == before_alias,
          "overlapping shape/publication buffers were accepted");
  require(generativeqc_resource_ks_cuda_host_fleet_v1(shapes.data(), 36, allowance, 0,
                                                      values.data(), 47) == 0 &&
              values[14] == succeeds[14],
          "a refusal poisoned a subsequent valid neighbor");
}

}  // namespace

int main() {
  test_storage_domains();
  test_consumer_lifetimes_and_rollback();
  test_transactional_bridge();
}
