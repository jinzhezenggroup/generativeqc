#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <vector>

namespace generativeqc::scf::cuda_execution {
struct Angular {
  unsigned x{}, y{}, z{};
};
constexpr unsigned angular_total(Angular a) { return a.x + a.y + a.z; }
struct Dual3 {
  double value{}, derivative_x{}, derivative_y{}, derivative_z{};
};
}  // namespace generativeqc::scf::cuda_execution

#include "integrals/range_exchange_runtime.hpp"

namespace {
struct SyntheticProgram {
  static generativeqc::scf::cuda_execution::Dual3 evaluate(
      const double*, const double*, const generativeqc::scf::cuda_execution::Angular*,
      unsigned mask) {
    double seeds = 0;
    for (unsigned bit = 0; bit < 4; ++bit) seeds += (mask & (1U << bit)) ? 1.0 : 0.0;
    return {0.0, seeds, 0.0, 0.0};
  }
};
void require(bool valid, const char* message) {
  if (!valid) throw std::runtime_error(message);
}
void expanded_shape_and_transaction() {
  constexpr std::size_t atoms = 9, aos = 65, primitives = 129;
  std::vector<double> packed(3 * atoms + 2 * primitives + 16 * aos);
  for (std::size_t primitive = 0; primitive < primitives; ++primitive) {
    packed[3 * atoms + 2 * primitive] = 1.0;
    packed[3 * atoms + 2 * primitive + 1] = 1.0;
  }
  auto* records = packed.data() + 3 * atoms + 2 * primitives;
  for (std::size_t orbital = 0; orbital < aos; ++orbital) {
    auto* row = records + 16 * orbital;
    row[0] = orbital == aos - 1 ? 0.0 : 8.0;
    row[1] = 128.0;
    row[2] = 1.0;
    row[3] = 1.0;
    row[7] = 1.0;
  }
  const std::array<std::int64_t, 4> quartet{0, 64, 64, 0};
  const std::array<double, 1> weights{1.0};
  std::array<std::uint64_t, 2> work{};
  std::vector<double> force(3 * atoms, 77.0);
  const auto calculate = [&](bool independent_slots, double* output) {
    return generativeqc::integrals::range_exchange_records<SyntheticProgram>(
        packed.data(), atoms, primitives, aos, quartet.data(), weights.data(), weights.size(),
        independent_slots, output, work.data());
  };
  require(calculate(false, force.data()) == 0, "wide physical force was rejected");
  require(work[0] == 1 && work[1] == 1, "wide physical primitive work is wrong");
  for (std::size_t index = 0; index < force.size(); ++index) {
    const auto expected = index == 0 ? 2.0 : index == 24 ? -2.0 : 0.0;
    require(force[index] == expected, "wide physical force publication is incorrect");
  }
  std::array<double, 12> slots{};
  require(calculate(true, slots.data()) == 0, "independent derivative slots were rejected");
  require(work[0] == 1 && work[1] == 3, "independent primitive work is wrong");
  for (std::size_t index = 0; index < slots.size(); ++index) {
    const auto expected = index == 0 || index == 3 || index == 6 ? 1.0 : index == 9 ? -3.0 : 0.0;
    require(slots[index] == expected, "independent derivative publication is incorrect");
  }
  records[0] = std::numeric_limits<double>::quiet_NaN();
  std::fill(force.begin(), force.end(), 77.0);
  require(calculate(false, force.data()) == 1, "nonfinite atom identity was admitted");
  require(std::all_of(force.begin(), force.end(), [](double v) { return v == 77.0; }),
          "failed range derivative partially overwrote the force vector");
}
}  // namespace
int main() {
  try {
    expanded_shape_and_transaction();
    std::cout << "Native range force expanded shape: PASS\n";
    return 0;
  } catch (const std::exception& failure) {
    std::cerr << failure.what() << '\n';
    return 1;
  }
}
