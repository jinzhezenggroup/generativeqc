#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>

#include "scf/ks_device_resources.hpp"

extern "C" int generativeqc_resource_ks_cuda_fleet_v1(const std::uint64_t*, std::size_t,
                                                      std::uint64_t*, std::size_t);

namespace {
void require(bool condition, const char* message) {
  if (!condition) throw std::runtime_error(message);
}

void test_fleet() {
  using generativeqc::scf::CudaKsDeviceFleetResources;
  const generativeqc::scf::CudaKsDeviceResourceItem first{2,  2,  2, 6,    100, 200, 300,
                                                          40, 50, 0, true, 700, 900};
  const generativeqc::scf::CudaKsDeviceResourceItem second{1,   1,  1,    1,     20, 40, 60,
                                                           500, 30, 4000, false, 10, 20};
  CudaKsDeviceFleetResources fleet;
  require(fleet.append(first) == std::array<std::uint64_t, 6>{100, 200, 300, 1280, 40, 50},
          "CUDA first-item phases differ from the explicit allocation bound");
  require(fleet.inventory() ==
              std::array<std::uint64_t, 9>{100, 200, 300, 50, 680, 700, 900, 2030, 2230},
          "CUDA first-item provider/stationary peak is inconsistent");
  require(fleet.append(second) == std::array<std::uint64_t, 6>{20, 0, 60, 4576, 500, 30},
          "CUDA ECP derivative scratch must coexist with retained SCF arenas");
  const auto before = fleet.inventory();
  require(before == std::array<std::uint64_t, 9>{120, 200, 360, 80, 4496, 710, 920, 5966, 6176},
          "CUDA ragged fleet must sum retention and serialize setup excess");
  for (const auto copies : {std::uint64_t{0}, std::numeric_limits<std::uint64_t>::max()}) {
    bool refused = false;
    try {
      fleet.append(first, copies);
    } catch (const std::invalid_argument&) {
      refused = true;
    } catch (const std::overflow_error&) {
      refused = true;
    }
    require(refused && fleet.inventory() == before, "invalid repetition mutated CUDA fleet");
  }
  CudaKsDeviceFleetResources repeated, individual;
  repeated.append(first, 3);
  for (std::size_t index = 0; index < 3; ++index) individual.append(first);
  require(repeated.inventory() == individual.inventory(), "CUDA repetitions changed lifetimes");
  auto invalid = first;
  invalid.atoms = 0;
  bool refused = false;
  try {
    fleet.append(invalid);
  } catch (const std::invalid_argument&) {
    refused = true;
  }
  require(refused && fleet.inventory() == before, "invalid topology mutated fleet");
  auto independent = first;
  independent.stationary_minimum = 900;
  independent.stationary_cached = 700;
  CudaKsDeviceFleetResources alternate;
  alternate.append(independent);
  require(alternate.inventory()[7] == 2230 && alternate.inventory()[8] == 2030,
          "CUDA fleet invented an ordering constraint between independent cache candidates");
  auto exact = first;
  exact.state = static_cast<std::uint64_t>(std::numeric_limits<std::int64_t>::max());
  exact.xc = exact.coulomb = exact.matrix_provider = 0;
  exact.stationary_minimum = exact.stationary_cached = 0;
  CudaKsDeviceFleetResources boundary;
  boundary.append(exact);
  const auto exact_inventory = boundary.inventory();
  require(exact_inventory[7] == exact.state && exact_inventory[8] == exact.state,
          "exact signed-int64 device peak was refused");
  refused = false;
  try {
    boundary.append(first);
  } catch (const std::overflow_error&) {
    refused = true;
  }
  require(refused && boundary.inventory() == exact_inventory,
          "overflowing device peak mutated the exact-limit fleet");
}

void test_bridge() {
  const std::array<std::uint64_t, 26> shapes{2, 2, 2, 6, 100, 200, 300, 40,  50, 0,    1, 700, 900,
                                             1, 1, 1, 1, 20,  40,  60,  500, 30, 4000, 0, 10,  20};
  std::array<std::uint64_t, 23> output;
  output.fill(0x12345678);
  const auto pristine = output;
  require(!generativeqc_resource_ks_cuda_fleet_v1(shapes.data(), shapes.size(), output.data(), 21),
          "valid CUDA fleet transport failed without a device");
  const std::array<std::uint64_t, 9> expected{120, 200, 360, 80, 4496, 710, 920, 5966, 6176};
  require(std::equal(expected.begin(), expected.end(), output.begin()),
          "CUDA transport published inconsistent aggregate capacities");
  require(output[21] == pristine[21] && output[22] == pristine[22],
          "CUDA transport overwrote its output tail");
  for (const auto slot : {13u, 23u, 24u, 25u}) {
    auto invalid = shapes;
    if (slot == 13) invalid[slot] = 0;
    if (slot == 23) invalid[slot] = 2;
    if (slot == 24) invalid[slot] = std::numeric_limits<std::uint64_t>::max();
    if (slot == 25) invalid[slot] = std::numeric_limits<std::uint64_t>::max();
    output = pristine;
    require(generativeqc_resource_ks_cuda_fleet_v1(invalid.data(), invalid.size(), output.data(),
                                                   21) != 0 &&
                output == pristine,
            "invalid later CUDA descriptor published partial output");
  }
  for (const auto length : {0u, 25u, 27u}) {
    output = pristine;
    require(generativeqc_resource_ks_cuda_fleet_v1(shapes.data(), length, output.data(), 21) != 0 &&
                output == pristine,
            "invalid CUDA shape length changed output");
  }
  output = pristine;
  require(generativeqc_resource_ks_cuda_fleet_v1(nullptr, shapes.size(), output.data(), 21) != 0 &&
              output == pristine,
          "null CUDA input changed output");
  require(generativeqc_resource_ks_cuda_fleet_v1(shapes.data(), shapes.size(), nullptr, 21) != 0,
          "null CUDA output was accepted");
  require(generativeqc_resource_ks_cuda_fleet_v1(shapes.data(), shapes.size(), output.data(), 22) !=
                  0 &&
              output == pristine,
          "invalid CUDA output length changed output");
  auto alias = shapes;
  require(generativeqc_resource_ks_cuda_fleet_v1(alias.data(), alias.size(), alias.data() + 1,
                                                 21) != 0 &&
              alias == shapes,
          "overlapping CUDA transport input/output was accepted");
}

void test_df_setup_lifetime() {
  using namespace generativeqc::scf;
  CudaKsDeviceResourceItem item{1, 1, 1, 1, 100, 200, 300, 40, 50, 0, true, 700, 900};
  item.fock_setup = 5000;
  CudaKsDeviceFleetResources fleet;
  fleet.append(item, 3);
  // During a serialized rebuild the other two items retain all their numeric
  // arenas; all three provider and lazy-force reservations remain available.
  const auto values = fleet.inventory();
  require(values[4] == 4400 && values[7] == 8450 && values[8] == 9050,
          "DF metric setup did not coexist with retained fleet/force owners");
  const auto before = values;
  item.fock_setup = std::numeric_limits<std::uint64_t>::max();
  bool refused = false;
  try {
    fleet.append(item);
  } catch (const std::overflow_error&) {
    refused = true;
  }
  require(refused && fleet.inventory() == before,
          "overflowing DF setup changed a previously admitted fleet");
  item.fock_setup = 5000;
  item.deferred_setup = 9000;
  CudaKsDeviceFleetResources late;
  late.append(item, 3);
  require(late.inventory()[7] == 13050 && late.inventory()[8] == 13650,
          "late derivative export borrowed storage from still-live SCF/Fock owners");
  require(cuda_ks_one_electron_capacity(1, 1, 1, 1) == 448,
          "shared one-electron capacity changed its existing small-shape envelope");
  // Atoms may have no orbital shells. The paired bridge still uploads their
  // charges/positions and retains both coordinate-major gradient outputs.
  // This sparse topology exposes a missing second output hidden by AO storage
  // in ordinary molecular bases.
  constexpr std::uint64_t atoms = 1000;
  const auto atom_payload = atoms * (sizeof(std::int32_t) + 9 * sizeof(double));
  require(cuda_ks_one_electron_capacity(1, atoms, 1, 1, true) >= atom_payload,
          "paired one-electron capacity omitted atoms outside the orbital space");
}
}  // namespace

int main() {
  test_fleet();
  test_bridge();
  test_df_setup_lifetime();
}
