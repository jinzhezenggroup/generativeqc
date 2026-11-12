#include <array>
#include <cmath>
#include <iostream>
#include <memory>
#include <numbers>
#include <stdexcept>

#include "api/handles.hpp"
#include "generativeqc/generativeqc.h"
#include "integrals/ecp.hpp"
#include "integrals/ecp_cuda.hpp"

namespace {
void require(bool condition, const char* message) {
  if (!condition) throw std::runtime_error(message);
}

void check_cuda_grid_admission() {
  using namespace generativeqc::integrals;
  std::string detail;
  for (const auto grid : {std::array<unsigned, 2>{16, 8}, {512, 96}})
    require(validate_ecp_cuda_grid(grid[0], grid[1], detail) == GENERATIVEQC_STATUS_SUCCESS,
            "CUDA raw grid boundary rejected");
  generativeqc_context context;
  context.state.executed_backend = GENERATIVEQC_BACKEND_CUDA;
  generativeqc_system system;
  system.data.atoms = {{11, {0, 0, 0}, 10}};
  system.data.shells = {{0, 0, {{0.7, 1.0}}}};
  system.data.ecp_terms = {{0, -1, 2, 0.8, -2.0}};
  for (const auto grid :
       {std::array<unsigned, 2>{0, 0}, {15, 8}, {16, 7}, {513, 8}, {16, 97}, {1536, 192}}) {
    for (bool derivatives : {false, true}) {
      EcpData native_output{1, 1, {42}, {43}, {44}, {45}};
      require(ecp_integrals_cuda(-1, system.data, grid[0], grid[1], derivatives, native_output,
                                 detail) == GENERATIVEQC_STATUS_INVALID_ARGUMENT,
              "raw CUDA grid must reject before device selection");
      require(native_output.nbf == 1 && native_output.ncoord == 1 &&
                  native_output.local == std::vector<double>{42} &&
                  native_output.nonlocal == std::vector<double>{43} &&
                  native_output.local_derivative == std::vector<double>{44} &&
                  native_output.nonlocal_derivative == std::vector<double>{45},
              "rejected direct CUDA grid changed output");
      std::vector<double> output(derivatives ? 8 : 2, 123.0);
      require(generativeqc_system_ecp_integrals(&context, &system, grid[0], grid[1], derivatives,
                                                output.data(), output.size()) ==
                  GENERATIVEQC_STATUS_INVALID_ARGUMENT,
              "public CUDA grid boundary not enforced");
      for (double value : output) require(value == 123.0, "rejected C grid changed output");
    }
  }
}

void check_grid_wrapper() {
  std::vector<generativeqc::integrals::EcpRadialPoint> radial;
  std::vector<generativeqc::integrals::EcpSpherePoint> sphere;
  generativeqc::integrals::ecp_quadrature(16, 8, radial, sphere);
  require(radial.size() == 16 && sphere.size() == 128, "native grid wrapper shape");
  double area = 0;
  for (const auto& p : sphere) area += p.weight;
  require(std::abs(area - 4 * std::numbers::pi) < 1e-12, "native grid wrapper weights");
  generativeqc::integrals::ecp_quadrature(16, 8, radial, sphere);
  require(radial.size() == 32 && sphere.size() == 256, "native grid wrapper append contract");
}

void check_c_api() {
  const generativeqc_context_descriptor context_desc{sizeof(context_desc), GENERATIVEQC_ABI_VERSION,
                                                     0, GENERATIVEQC_BACKEND_CPU_REFERENCE};
  generativeqc_context* raw{};
  require(generativeqc_context_create(&context_desc, &raw) == GENERATIVEQC_STATUS_SUCCESS,
          "context create");
  std::unique_ptr<generativeqc_context, decltype(&generativeqc_context_destroy)> context(
      raw, generativeqc_context_destroy);
  const std::array<generativeqc_atom, 2> atoms{{{11, 0, 0, 0}, {1, 0.3, 0.1, 3.0}}};
  const std::array<generativeqc_primitive, 2> primitives{{{0.55, 1}, {0.7, 1}}};
  const std::array<int32_t, 2> cores{10, 0};
  for (unsigned center : {0U, 1U})
    for (auto representation : {GENERATIVEQC_BASIS_CARTESIAN, GENERATIVEQC_BASIS_SPHERICAL})
      for (unsigned angular : {3U, 4U}) {
        std::array<generativeqc_shell, 2> shells{{{0, 0, 0, 1}, {1, 0, 1, 1}}};
        shells[center].angular_momentum = angular;
        const generativeqc_system_descriptor descriptor{sizeof(descriptor),
                                                        GENERATIVEQC_ABI_VERSION,
                                                        atoms.data(),
                                                        atoms.size(),
                                                        shells.data(),
                                                        shells.size(),
                                                        primitives.data(),
                                                        primitives.size(),
                                                        0,
                                                        1,
                                                        representation};
        const generativeqc_ecp_term term{0, -1, 2, 0.8, -2};
        generativeqc_system* system{};
        const auto status = generativeqc_system_create_ecp(context.get(), &descriptor, cores.data(),
                                                           &term, 1, &system);
        std::unique_ptr<generativeqc_system, decltype(&generativeqc_system_destroy)> owner(
            system, generativeqc_system_destroy);
        require(status == (angular == 3 ? GENERATIVEQC_STATUS_SUCCESS
                                        : GENERATIVEQC_STATUS_INVALID_ARGUMENT),
                "native ECP orbital-f/g capability mismatch");
        require((system != nullptr) == (angular == 3), "failed ECP constructor published output");
        if (angular == 3) {
          const std::array<generativeqc_ecp_term, 2> supported{{term, {0, 3, 2, 0.63, 0.74}}};
          generativeqc_system* extended{};
          require(generativeqc_system_create_ecp(context.get(), &descriptor, cores.data(),
                                                 supported.data(), supported.size(),
                                                 &extended) == GENERATIVEQC_STATUS_SUCCESS &&
                      extended,
                  "native f projector rejected");
          generativeqc_system_destroy(extended);
          const std::array<generativeqc_ecp_term, 2> unsupported{{term, {0, 4, 2, 0.8, -2}}};
          generativeqc_system* bad{};
          require(generativeqc_system_create_ecp(context.get(), &descriptor, cores.data(),
                                                 unsupported.data(), unsupported.size(),
                                                 &bad) == GENERATIVEQC_STATUS_INVALID_ARGUMENT &&
                      !bad,
                  "projector f support silently enabled projector g");
        }
      }
}
#if !GENERATIVEQC_HAS_CUDA
void check_cuda_not_built_stub() {
  generativeqc::core::System system;
  generativeqc::integrals::EcpData output;
  std::string detail;
  require(generativeqc::integrals::ecp_integrals_cuda(0, system, 160, 32, false, output, detail) ==
              GENERATIVEQC_STATUS_NOT_IMPLEMENTED,
          "CPU-only ECP CUDA stub lost not-implemented status");
  require(generativeqc::integrals::ecp_integrals_cuda(0, system, 0, 0, false, output, detail,
                                                      true) == GENERATIVEQC_STATUS_NOT_IMPLEMENTED,
          "checked CUDA defaults must not be interpreted as a raw zero grid");
  require(detail.find("cuda.scalar") != std::string::npos &&
              detail.find("not built") != std::string::npos,
          "CPU-only ECP CUDA stub lost provider-specific diagnostics");
}
#endif

}  // namespace

int main() {
  try {
    check_cuda_grid_admission();
    check_grid_wrapper();
    check_c_api();
#if !GENERATIVEQC_HAS_CUDA
    check_cuda_not_built_stub();
#endif
    std::cout << "ECP native orbital-f boundary and host-grid wrapper passed\n";
  } catch (const std::exception& e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}
