#include <array>
#include <iostream>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <vector>

#include "dft/cuda_wb97mv_nuclear_jk.hpp"
#include "dft/hessian_aot.hpp"

namespace {
namespace aot = generativeqc::dft::hessian_aot;
std::vector<aot::Program> selected;
int never_launch(const double*, std::size_t, const long long*, const double*, double*, int*,
                 cudaStream_t) {
  throw std::runtime_error("host closure admission launched CUDA");
}
void require(bool valid, const char* message) {
  if (!valid) throw std::runtime_error(message);
}
}  // namespace
namespace generativeqc::dft::hessian_aot {
std::span<const Program> programs() noexcept { return selected; }
}  // namespace generativeqc::dft::hessian_aot

int main() {
  using namespace generativeqc;
  using namespace generativeqc::dft;
  try {
    // Qualify the largest through-f cache and tiled closure without generating
    // kernels or initializing a CUDA context. Nameless entries are intentional.
    constexpr std::array<unsigned, 4> angular{3, 3, 3, 3};
    std::vector<std::uint32_t> components(10000);
    std::iota(components.begin(), components.end(), 0);
    for (const auto radial : {"full_range", "long_range"})
      for (std::size_t begin = 0; begin < components.size(); begin += 16) {
        aot::Program program{};
        program.kind = aot::Kind::integral;
        program.integral = never_launch;
        program.signature = {"eri", angular, radial,
                             radial == std::string_view("full_range") ? 0 : 0.3};
        program.components = {components.data() + begin, 16};
        program.centers = 4;
        program.record_stride = 17;
        program.outputs = 25;
        selected.push_back(program);
      }
    const auto complete = selected;
    const CudaWb97mvJkShape shape{1, 2, 10, 2};
    constexpr std::array<integrals::CudaOneElectronShell, 1> shells{{{0, 3, 0, 2, 0, 10}}};
    CudaWb97mvJkOptions options;
    std::size_t bytes{};
    require(cuda_wb97mv_jk_workspace(shape, shells, options, bytes) == cudaSuccess &&
                bytes > 5'000'000 && bytes < 6'000'000,
            "complete nameless through-f closure failed admission");
    for (unsigned kind = 0; kind < 10; ++kind) {
      selected = complete;
      if (kind == 0) selected.pop_back();
      if (kind == 1) selected.push_back(selected.front());
      if (kind == 2) selected[0].signature.radial = "short_range";
      if (kind == 3) selected[0].signature.omega = 0.3;
      if (kind == 4) selected[0].record_stride = 16;
      if (kind == 5) selected[0].outputs = 24;
      if (kind == 6) selected[0].centers = 3;
      if (kind == 7) selected[0].integral = nullptr;
      if (kind == 8) selected[0].components = {components.data(), 17};
      if (kind == 9) selected[0].signature.omega = std::numeric_limits<double>::quiet_NaN();
      bytes = 27;
      require(cuda_wb97mv_jk_workspace(shape, shells, options, bytes) == cudaErrorNotSupported &&
                  !bytes,
              "malformed closure admitted");
    }
    selected = complete;
    for (unsigned kind = 0; kind < 12; ++kind) {
      auto invalid = options;
      if (kind == 0) invalid.radial_tile = 0;
      if (kind == 1) invalid.radial_tile = 65;
      if (kind == 2) invalid.max_groups = 1;
      if (kind == 3) invalid.max_workspace_bytes = 1;
      if (kind == 4) invalid.max_primitive_component_evaluations = 319999;
      if (kind == 5) invalid.max_projection_terms = 1619999;
      if (kind == 6) invalid.max_reduction_terms = 1;
      if (kind == 7) invalid.symmetry_absolute_tolerance = -1;
      if (kind == 8) invalid.symmetry_relative_tolerance = std::numeric_limits<double>::infinity();
      if (kind == 9) invalid.part = static_cast<CudaWb97mvJkPart>(99);
      if (kind == 10)
        invalid.symmetry_absolute_tolerance = std::numeric_limits<double>::quiet_NaN();
      if (kind == 11) invalid.symmetry_relative_tolerance = -1;
      bytes = 27;
      require(cuda_wb97mv_jk_workspace(shape, shells, invalid, bytes) == cudaErrorInvalidValue &&
                  !bytes,
              "invalid resource/work limit admitted");
    }
    for (unsigned kind = 0; kind < 7; ++kind) {
      auto invalid = shape;
      if (kind == 0) invalid.nao = SIZE_MAX;
      if (kind == 1) invalid.atoms = SIZE_MAX;
      if (kind == 2) invalid.primitives = SIZE_MAX;
      if (kind == 3) invalid.spins = 3;
      if (kind == 4) invalid.atoms = 0;
      if (kind == 5) invalid.primitives = 0;
      if (kind == 6) invalid.nao = 0;
      bytes = 27;
      require(cuda_wb97mv_jk_workspace(invalid, shells, options, bytes) == cudaErrorInvalidValue &&
                  !bytes,
              "invalid/overflow shape admitted");
    }
    for (const auto tile : {1, 3, 64}) {
      options.radial_tile = tile;
      require(cuda_wb97mv_jk_workspace(shape, shells, options, bytes) == cudaSuccess,
              "valid radial bound rejected");
    }
    std::cout << "J/K host closure/work/arena gates passed\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
