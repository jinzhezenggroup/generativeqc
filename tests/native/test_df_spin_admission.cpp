#include <iostream>
#include <stdexcept>
#include <string>

#include "api/handles.hpp"
#include "generativeqc/fock.h"
#include "scf/cuda/df_source_domain.hpp"
#include "scf/cuda/df_spin_domain.hpp"
#include "scf/mean_field.hpp"

namespace {
void require(bool condition, const char* message) {
  if (!condition) throw std::runtime_error(message);
}

void public_plan_admission() {
  // Inject only the context's backend selection, so this host-only admission
  // test never needs to initialize CUDA. The real C adapter, resolver and
  // PreparedFockPlan constructor must reject before using the invalid device.
  generativeqc_context context;
  context.state.executed_backend = GENERATIVEQC_BACKEND_CUDA;
  context.state.device_id = -1;
  for (int representation : {GENERATIVEQC_BASIS_CARTESIAN, GENERATIVEQC_BASIS_SPHERICAL}) {
    for (unsigned derivative : {0U, 1U}) {
      for (bool restricted : {false, true}) {
        for (unsigned angular : {3U, 4U}) {
          for (int fitted_term : {0, 1, 2}) {
            generativeqc_system orbital;
            orbital.data.atoms = {{2, {0, 0, 0}}};
            orbital.data.shells = {{0, angular, {{1.0, 1.0}}}};
            orbital.data.electron_count = 2;
            orbital.data.multiplicity = 1;
            orbital.data.basis_representation = representation;
            generativeqc_system auxiliary = orbital;
            auxiliary.data.shells[0].angular_momentum = 0;
            generativeqc_fock_spec spec{
                sizeof(spec),
                GENERATIVEQC_ABI_VERSION,
                1,
                restricted ? GENERATIVEQC_FOCK_RESTRICTED : GENERATIVEQC_FOCK_UNRESTRICTED,
                derivative,
                {1, 1.0, GENERATIVEQC_FOCK_FULL_RANGE, 0.0,
                 fitted_term == 1 ? GENERATIVEQC_FOCK_EXACT : GENERATIVEQC_FOCK_DENSITY_FITTED},
                {1, restricted ? -0.5 : -1.0, GENERATIVEQC_FOCK_FULL_RANGE, 0.0,
                 fitted_term == 0 ? GENERATIVEQC_FOCK_EXACT : GENERATIVEQC_FOCK_DENSITY_FITTED}};
            generativeqc_fock_plan* plan = nullptr;
            const auto status = generativeqc_fock_plan_create(&context, &orbital, &auxiliary, &spec,
                                                              nullptr, &plan);
            require(plan == nullptr, "admission failure must not publish a plan");
            if (!restricted && angular == 4) {
              require(status == GENERATIVEQC_STATUS_INVALID_ARGUMENT,
                      "unqualified unrestricted g must fail native admission");
              require(context.last_detail.find("unrestricted CUDA DF") != std::string::npos,
                      "unrestricted g must fail before provider or device setup");
            } else {
              // CUDA-enabled builds reach the invalid-device sentinel. Host
              // builds reach provider availability instead. Neither is a claim
              // that these numerical endpoints were executed by this test.
              require(status == GENERATIVEQC_STATUS_NOT_IMPLEMENTED ||
                          (status == GENERATIVEQC_STATUS_INVALID_ARGUMENT &&
                           context.last_detail == "CUDA prepared Fock requires a device"),
                      "restricted g and unrestricted f must pass spin admission");
            }
          }
        }
      }
    }
  }
}

void resident_admission() {
#if GENERATIVEQC_HAS_CUDA
  using namespace generativeqc::scf;
  generativeqc::core::System orbital;
  orbital.atoms = {{2, {0, 0, 0}}};
  orbital.shells = {{0, 4, {{1.0, 1.0}}}};
  orbital.electron_count = 2;
  orbital.multiplicity = 1;
  const ScfOptions options;
  for (int entry : {0, 1, 2}) {
    bool rejected = false;
    try {
      if (entry == 0)
        (void)run_uhf_density_fitting_cuda(orbital, orbital, options, -1);
      else if (entry == 1)
        (void)run_uhf_density_fitting_cuda_bucket({orbital}, orbital, options, {nullptr}, -1);
      else {
        CudaDensityFittingJkPlan* plan = nullptr;
        (void)run_uhf_density_fitting_cuda_bucket_cached(&plan, {orbital}, orbital, options,
                                                         {nullptr}, -1);
      }
    } catch (const std::invalid_argument& error) {
      rejected = std::string(error.what()).find("unrestricted CUDA DF") != std::string::npos;
    }
    require(rejected, "resident UHF entries must reject g before preparation");
  }
#endif
}

void independent_source_limits() {
  using namespace generativeqc::scf;
  using namespace generativeqc::scf::cuda_execution;
  generativeqc::core::System orbital, auxiliary;
  orbital.shells = {{0, 4, {{1.0, 1.0}}}};
  auxiliary.shells = {{0, 5, {{1.0, 1.0}}}};
  CudaDfSourcePolicy policy;
  std::string detail;
  require_cuda_df_spin_domain(orbital, FockSpin::Restricted);
  require(cuda_df_value_domain(orbital, auxiliary, policy, detail),
          "qualified generic full-range g/h source must remain available");
  for (unsigned math : {1U, 2U, 3U}) {
    policy.value_math = math;
    require(!cuda_df_value_domain(orbital, auxiliary, policy, detail),
            "spin admission must not widen non-generic source math");
  }
  policy.value_math = 0;
  policy.range_omega = 0.3;
  require(!cuda_df_value_domain(orbital, auxiliary, policy, detail),
          "spin admission must not widen range-separated source qualification");
}
}  // namespace

int main() {
  try {
    public_plan_admission();
    independent_source_limits();
    resident_admission();
    std::cout << "native CUDA DF spin admission passed\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
