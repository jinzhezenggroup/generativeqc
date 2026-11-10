#include "integrals/ecp_cuda.hpp"
#include "runtime/provider_registry.hpp"
namespace generativeqc::integrals {
namespace {
struct EcpCudaDomain {
  std::uint32_t maximum_derivative_order{1};
};
constexpr runtime::ProviderDescriptor<EcpCudaDomain> kEcpCudaProvider{
    {"integrals.ecp", "cuda.scalar", 1, runtime::ProviderBackend::Cuda},
    {},
    runtime::ProviderAvailability::NotBuilt,
    runtime::ProviderFallback::None,
    runtime::ProviderRequirement::Resources,
    "CUDA support was not compiled into this build",
    "src/integrals/ecp_cuda_stub.cpp"};

generativeqc_status unavailable(std::string& detail, std::string_view operation) {
  return runtime::provider_not_implemented(kEcpCudaProvider, detail, operation);
}
}  // namespace

generativeqc_status ecp_integrals_cuda(int, const core::System&, unsigned radial, unsigned polar,
                                       bool, EcpData&, std::string& detail, bool convergence) {
  if (!convergence) {
    const auto status = validate_ecp_cuda_grid(radial, polar, detail);
    if (status != GENERATIVEQC_STATUS_SUCCESS) return status;
  }
  return unavailable(detail, "ECP integral evaluation");
}
generativeqc_status add_ecp_cuda(int, const core::System&, void*, double*, const double*, double*,
                                 std::string& detail, double) {
  return unavailable(detail, "ECP operator application");
}
}  // namespace generativeqc::integrals
