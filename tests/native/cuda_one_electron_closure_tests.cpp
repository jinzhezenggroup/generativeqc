#include <array>
#include <iostream>
#include <limits>
#include <stdexcept>

#include "dft/hessian_aot.hpp"
#include "integrals/cuda_one_electron_response.hpp"

namespace {
namespace aot = generativeqc::dft::hessian_aot;
std::array<aot::Program, 4> selected;
std::size_t selected_count{3};
int never_launch(const double*, std::size_t, const long long*, const double*, double*, int*,
                 cudaStream_t) {
  throw std::runtime_error("host closure admission launched a CUDA leaf");
}
void require(bool condition, const char* message) {
  if (!condition) throw std::runtime_error(message);
}
}  // namespace

namespace generativeqc::dft::hessian_aot {
// Link a deliberately incomplete/ambiguous package without loading any actual
// generated kernel. Admission must depend on scientific selectors, never names.
std::span<const Program> programs() noexcept { return {selected.data(), selected_count}; }
}  // namespace generativeqc::dft::hessian_aot

int main() {
  using namespace generativeqc::integrals;
  try {
    constexpr std::array<unsigned, 2> angular{1, 1};
    constexpr std::array<std::uint32_t, 9> components{0, 1, 2, 3, 4, 5, 6, 7, 8};
    constexpr std::array<std::string_view, 3> families{"overlap", "kinetic", "nuclear_attraction"};
    for (unsigned family = 0; family < 3; ++family) {
      auto& program = selected[family];
      program.kind = aot::Kind::integral;
      program.integral = never_launch;
      program.signature = {families[family], angular, "full_range", 0};
      program.components = components;
      program.centers = family == 2 ? 3 : 2;
      program.record_stride = 3 + 3 * program.centers;
      program.outputs = 1 + 6 * program.centers;
    }
    const auto complete = selected;
    const CudaOneElectronShape shape{1, 2, 3};
    constexpr std::array<CudaOneElectronShell, 1> shells{{{0, 1, 0, 2, 0, 3}}};
    const CudaOneElectronOptions options;
    std::size_t bytes{};
    require(cuda_one_electron_workspace(shape, shells, options, bytes) == cudaSuccess && bytes,
            "complete nameless scientific closure was not admitted");
    for (unsigned kind = 0; kind < 9; ++kind) {
      selected = complete;
      selected_count = 3;
      if (kind == 0) selected[0].components = {components.data(), 8};
      if (kind == 1) {
        selected[3] = selected[0];
        selected[3].components = {components.data(), 1};
        selected_count = 4;
      }
      if (kind == 2) selected[1].signature.radial = "long_range";
      if (kind == 3) selected[1].centers = 3;
      if (kind == 4) selected[2].record_stride = 9;
      if (kind == 5) selected[2].outputs = 13;
      if (kind == 6) selected[0].integral = nullptr;
      if (kind == 7) selected[0].signature.omega = 0.3;
      if (kind == 8) selected[0].signature.omega = std::numeric_limits<double>::quiet_NaN();
      bytes = 27;
      require(cuda_one_electron_workspace(shape, shells, options, bytes) == cudaErrorNotSupported &&
                  !bytes,
              "incomplete/overlapping/malformed closure admitted");
    }
    std::cout << "one-electron host closure gates passed\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
