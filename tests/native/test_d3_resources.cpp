#include <array>
#include <cstdint>
#include <iostream>
#include <limits>
#include <stdexcept>

#include "dft/dispersion/d3_runtime.hpp"

extern "C" int generativeqc_resource_d3_layout_v1(int, std::uint64_t, std::uint64_t, std::uint64_t,
                                                  std::uint64_t*, std::size_t);

namespace {
void require(bool condition, const char* message) {
  if (!condition) throw std::runtime_error(message);
}
}  // namespace

int main() {
  using namespace generativeqc::dft::dispersion;
  try {
    for (int backend : {0, 1}) {
      for (const auto shape : {std::array<std::uint64_t, 3>{1, 2, 2},
                               {3, 7, 4},
                               {4100, 4100, 1},
                               {2, 4097, 4096},
                               {UINT32_MAX, UINT32_MAX, 1}}) {
        std::array<std::uint64_t, 6> values{};
        require(generativeqc_resource_d3_layout_v1(backend, shape[0], shape[1], shape[2],
                                                   values.data(), values.size()) == 0,
                "valid shape rejected");
        const auto systems = shape[0], atoms = shape[1], maximum = shape[2];
        const auto plan = 4 * (systems + 1) + 28 * atoms;
        const auto workspace = 128 * (backend ? atoms : maximum);
        const auto execution = 48 * atoms + 14 * systems + (backend ? 0 : workspace + 24 * maximum);
        const auto device =
            backend ? 4 * (systems + 1) + 52 * atoms + 14 * systems + workspace + values[3] : 0;
        require(values[0] == plan && values[1] == execution && values[2] == device &&
                    values[3] > 0 && values[4] == workspace &&
                    values[5] == plan + execution + device,
                "independent layout mismatch");
      }
    }
    const auto maximum = std::numeric_limits<std::uint64_t>::max();
    for (const auto shape : {std::array<std::uint64_t, 3>{0, 1, 1},
                             {1, 0, 1},
                             {1, 1, 0},
                             {2, 1, 1},
                             {2, 6, 2},
                             {2, 2, 2},
                             {1, 4097, 4097},
                             {maximum, 1, 1},
                             {1, maximum, 1},
                             {1, 1, maximum}}) {
      std::array<std::uint64_t, 6> values{11, 22, 33, 44, 55, 66};
      const auto original = values;
      require(generativeqc_resource_d3_layout_v1(0, shape[0], shape[1], shape[2], values.data(),
                                                 values.size()) != 0 &&
                  values == original,
              "invalid shape modified output");
    }
    for (const auto count : {std::size_t{0}, std::size_t{5}, std::size_t{7},
                             std::numeric_limits<std::size_t>::max()}) {
      std::array<std::uint64_t, 6> values{11, 22, 33, 44, 55, 66};
      const auto original = values;
      require(generativeqc_resource_d3_layout_v1(0, 1, 2, 2, values.data(), count) != 0 &&
                  values == original,
              "invalid schema modified output");
    }
    std::array<std::uint64_t, 6> values{11, 22, 33, 44, 55, 66};
    const auto original = values;
    for (int backend : {-1, 2, 256})
      require(
          generativeqc_resource_d3_layout_v1(backend, 1, 2, 2, values.data(), values.size()) != 0 &&
              values == original,
          "invalid backend modified output");
    require(generativeqc_resource_d3_layout_v1(0, 1, 2, 2, nullptr, 6) != 0,
            "null output accepted");

    std::string detail;
    generativeqc_status status{};
    const D3Parameters parameters{1.0, 0.95948085, 0.38574991, 4.80688534, 0.0};
    for (std::uint64_t budget : {477, 478}) {
      auto plan = D3Plan::prepare(GENERATIVEQC_BACKEND_CPU_REFERENCE, 0, {0, 2}, {1, 1},
                                  {0, 0, -0.7, 0, 0, 0.7}, parameters, budget, detail, status);
      require(bool(plan) == (budget == 478), "production exact/below capacity gate changed");
      if (plan) {
        require(plan->resources().maximum_bytes == budget &&
                    plan->resources().plan_host_bytes == 64 &&
                    plan->resources().execution_host_bytes == 414,
                "prepared owner disagrees with query");
      } else {
        require(status == GENERATIVEQC_STATUS_OUT_OF_MEMORY, "wrong below-cap status");
      }
    }
    std::cout << "D3 shared shape authority, transactional ABI and exact capacity gates passed\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
