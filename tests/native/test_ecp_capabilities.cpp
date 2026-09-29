#include <array>
#include <cmath>
#include <iostream>
#include <memory>
#include <numbers>
#include <stdexcept>

#include "generativeqc/generativeqc.h"
#include "integrals/ecp.hpp"
#include "integrals/ecp_cuda.hpp"

namespace {
void require(bool condition, const char* message) {
  if (!condition) throw std::runtime_error(message);
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
  require(detail.find("cuda.scalar") != std::string::npos &&
              detail.find("not built") != std::string::npos,
          "CPU-only ECP CUDA stub lost provider-specific diagnostics");
}
#endif

}  // namespace

int main() {
  try {
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
