#include <algorithm>
#include <array>
#include <cstdint>
#include <limits>
#include <stdexcept>

#include "generativeqc/generativeqc.hpp"
#include "scf/hf_host_resources.hpp"

extern "C" int generativeqc_resource_hf_cpu_host_v1(const std::uint64_t*, std::size_t,
                                                    std::uint64_t*, std::size_t);
extern "C" int generativeqc_resource_hf_cpu_fleet_v1(const std::uint64_t*, std::size_t,
                                                     std::uint64_t*, std::size_t);

namespace {
void require(bool condition, const char* message) {
  if (!condition) throw std::runtime_error(message);
}

void check_prepared_host_lease() {
  generativeqc_context_descriptor cpu{};
  cpu.struct_size = sizeof(cpu);
  cpu.abi_version = GENERATIVEQC_ABI_VERSION;
  cpu.backend = GENERATIVEQC_BACKEND_CPU_REFERENCE;
  generativeqc::Context first_context(cpu), second_context(cpu);
  const std::array<generativeqc_atom, 2> hydrogen{{{1, 0, 0, 0}, {1, .05, -.1, 1.4}}};
  const std::array<generativeqc_atom, 2> hydroxyl{{{8, 0, 0, 0}, {1, .12, .15, 1.8}}};
  generativeqc::System first(first_context, hydrogen, "sto-3g");
  generativeqc::System second(second_context, hydroxyl, "sto-3g", 0, 2,
                              GENERATIVEQC_BASIS_SPHERICAL);
  generativeqc_method_descriptor rhf{};
  rhf.struct_size = sizeof(rhf);
  rhf.abi_version = GENERATIVEQC_ABI_VERSION;
  rhf.method = GENERATIVEQC_METHOD_RHF;
  auto uhf = rhf;
  uhf.method = GENERATIVEQC_METHOD_UHF;
  uhf.diis_history = 5;

  const auto quote = [](const std::array<std::uint64_t, 14>& shape) {
    std::array<std::uint64_t, 7> capacity{};
    require(generativeqc_resource_hf_cpu_fleet_v1(shape.data(), shape.size(), capacity.data(),
                                                  capacity.size()) == 0,
            "native HF fleet rejected the SDK test topology");
    return capacity;
  };
  const auto small = quote({2, 2, 2, 1, 0, 8, 0, 0, 0, 0, 2, 6, 0, 0});
  const auto large = quote({2, 6, 6, 2, 1, 5, 0, 0, 0, 0, 4, 12, 0, 0});
  const auto combined = small[0] + large[0] + std::max(small[1], large[1]);
  const auto expect = [](const generativeqc::HostResources& cap, std::uint64_t count,
                         std::uint64_t retained, std::uint64_t workspace) {
    const auto snapshot = cap.snapshot();
    require(snapshot.owner_count == count && snapshot.retained_bytes == retained &&
                snapshot.workspace_bytes == workspace,
            "shared Direct HF lease does not match the expected owner lifetimes");
  };
  const auto refuse = [](generativeqc::Context& context, const generativeqc::System& system,
                         const generativeqc_method_descriptor& method,
                         const generativeqc::HostResources& cap, generativeqc_status status) {
    auto* published = reinterpret_cast<generativeqc_calculation*>(1);
    require(generativeqc_calculation_prepare_with_host_resources_v1(
                context.get(), system.get(), &method, cap.get(), &published) == status &&
                published == nullptr,
            "refused Direct HF preparation published a handle or the wrong status");
  };
  {
    generativeqc::HostResources short_cap(large[2] - 1);
    refuse(second_context, second, uhf, short_cap, GENERATIVEQC_STATUS_OUT_OF_MEMORY);
    expect(short_cap, 0, 0, 0);
  }
  {
    generativeqc::HostResources exact(large[2]);
    const auto properties = GENERATIVEQC_PROPERTY_ENERGY | GENERATIVEQC_PROPERTY_FORCES;
    generativeqc::Calculation ordinary(second_context, second, uhf);
    const auto reference = ordinary.execute(properties);
    generativeqc::Calculation bounded(second_context, second, uhf, exact);
    expect(exact, 1, large[0], large[1]);
    const auto energy = bounded.execute();
    const auto ordinary_energy = ordinary.execute();
    require(energy.energy == ordinary_energy.energy &&
                energy.iterations == ordinary_energy.iterations && !energy.forces,
            "leased UHF energy replay changed the scientific endpoint");
    const auto forces = bounded.execute(properties);
    require(forces.energy == reference.energy && forces.iterations == reference.iterations &&
                forces.forces == reference.forces,
            "leased UHF force replay changed the scientific endpoint");
  }
  {
    generativeqc::HostResources cap(combined - 1);
    {
      generativeqc::Calculation owner(first_context, first, rhf, cap);
      refuse(second_context, second, uhf, cap, GENERATIVEQC_STATUS_OUT_OF_MEMORY);
      expect(cap, 1, small[0], small[1]);
    }
    expect(cap, 0, 0, 0);
    generativeqc::Calculation retry(second_context, second, uhf, cap);
    expect(cap, 1, large[0], large[1]);
  }
  {
    auto cap = std::make_unique<generativeqc::HostResources>(combined);
    generativeqc::Calculation owner(first_context, first, rhf, *cap);
    refuse(second_context, second, rhf, *cap, GENERATIVEQC_STATUS_INVALID_ARGUMENT);
    auto fitted = uhf;
    fitted.density_fitting_mode = GENERATIVEQC_DENSITY_FITTING_CPU_REFERENCE;
    refuse(second_context, second, fitted, *cap, GENERATIVEQC_STATUS_NOT_IMPLEMENTED);
    auto automatic = uhf;
    automatic.precision_mode = GENERATIVEQC_PRECISION_AUTO;
    refuse(second_context, second, automatic, *cap, GENERATIVEQC_STATUS_NOT_IMPLEMENTED);
    auto seeded = rhf;
    generativeqc_initial_guess_options guess{};
    guess.struct_size = sizeof(guess);
    guess.abi_version = GENERATIVEQC_ABI_VERSION;
    guess.kind = GENERATIVEQC_INITIAL_GUESS_MINAO;
    seeded.initial_guess = &guess;
    refuse(first_context, first, seeded, *cap, GENERATIVEQC_STATUS_NOT_IMPLEMENTED);
    expect(*cap, 1, small[0], small[1]);
    auto other = std::make_unique<generativeqc::Calculation>(second_context, second, uhf, *cap);
    expect(*cap, 2, small[0] + large[0], std::max(small[1], large[1]));
    other.reset();
    expect(*cap, 1, small[0], small[1]);
    cap.reset();
    generativeqc::Calculation ordinary(first_context, first, rhf);
    const auto properties = GENERATIVEQC_PROPERTY_ENERGY | GENERATIVEQC_PROPERTY_FORCES;
    const auto result = owner.execute(properties);
    const auto reference = ordinary.execute(properties);
    require(result.energy == reference.energy && result.iterations == reference.iterations &&
                result.forces == reference.forces,
            "closing the host metadata detached a live RHF plan");
  }
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
  check_prepared_host_lease();
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
