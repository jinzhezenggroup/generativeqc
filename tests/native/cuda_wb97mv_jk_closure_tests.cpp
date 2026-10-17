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
int never_batch(const aot::FactorizedIntegralBatch*, std::size_t, const double*, int*,
                cudaStream_t) {
  throw std::runtime_error("host tiled admission launched CUDA");
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
    auto streamed = options;
    streamed.max_resident_integral_bytes = 0;
    require(cuda_wb97mv_jk_workspace(shape, shells, streamed, bytes) == cudaSuccess &&
                bytes > 5'000'000 && bytes < 6'000'000,
            "complete nameless through-f closure failed admission");
    const auto minimum = bytes;
    require(cuda_wb97mv_jk_workspace(shape, shells, options, bytes) == cudaSuccess &&
                bytes > minimum && bytes - minimum <= options.max_resident_integral_bytes,
            "resident storage exceeded its explicit cap");
    auto budgeted = options;
    budgeted.max_workspace_bytes = minimum;
    require(
        cuda_wb97mv_jk_workspace(shape, shells, budgeted, bytes) == cudaSuccess && bytes == minimum,
        "optional residency changed minimum arena admission");
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
      if (kind == 5) invalid.max_projection_terms = 19999;
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
    selected.clear();
    constexpr std::array<unsigned, 4> scalar_angular{};
    constexpr std::array<std::uint32_t, 1> scalar_component{};
    for (const auto radial : {"full_range", "long_range"}) {
      aot::Program program{};
      program.kind = aot::Kind::integral;
      program.integral = never_launch;
      program.integral_batch = never_batch;
      program.signature = {"eri", scalar_angular, radial,
                           radial == std::string_view("full_range") ? 0 : 0.3};
      program.components = scalar_component;
      program.centers = 4;
      program.record_stride = 17;
      program.outputs = 25;
      selected.push_back(program);
    }
    std::vector<integrals::CudaOneElectronShell> large_shells;
    for (std::size_t index = 0; index < 24; ++index)
      large_shells.push_back({index, 0, index, 1, index, 1});
    auto bounded = options;
    bounded.max_workspace_bytes = 64ULL << 20;
    require(
        cuda_wb97mv_jk_workspace({24, 24, 24, 2}, large_shells, bounded, bytes) == cudaSuccess &&
            bytes <= bounded.max_workspace_bytes,
        "663552 ordered groups did not fit bounded canonical tiles");
    bounded.max_source_groups = 90299;
    require(cuda_wb97mv_jk_workspace({24, 24, 24, 2}, large_shells, bounded, bytes) ==
                    cudaErrorInvalidValue &&
                !bytes,
            "global canonical source guard was bypassed");
    large_shells.resize(4);
    bounded = options;
    bounded.max_groups = 16;
    bounded.source_tile = 1;
    bounded.max_primitive_component_evaluations = 110;
    require(cuda_wb97mv_jk_workspace({4, 4, 4, 2}, large_shells, bounded, bytes) == cudaSuccess,
            "canonical source work was charged as ordered primitive replay");
    bounded.max_primitive_component_evaluations = 109;
    require(cuda_wb97mv_jk_workspace({4, 4, 4, 2}, large_shells, bounded, bytes) ==
                    cudaErrorInvalidValue &&
                !bytes,
            "source tiles reset the global primitive work guard");
    bounded.max_primitive_component_evaluations = 110;
    bounded.source_tile = 0;
    require(cuda_wb97mv_jk_workspace({4, 4, 4, 2}, large_shells, bounded, bytes) ==
                    cudaErrorInvalidValue &&
                !bytes,
            "disabled tiled fallback bypassed ordered admission");
    std::cout << "J/K host closure/work/arena gates passed\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
