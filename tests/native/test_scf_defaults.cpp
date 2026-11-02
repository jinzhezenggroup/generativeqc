#include <cmath>
#include <cstring>
#include <limits>
#include <string>
#include <string_view>

#include "dft/grid.hpp"
#include "generativeqc/scf_defaults.h"
#include "methods/named_ks_catalog.hpp"
#include "native_test_require.hpp"

int main() {
  generativeqc_scf_defaults_context_v1 context{sizeof(context),
                                               GENERATIVEQC_ABI_VERSION,
                                               GENERATIVEQC_BACKEND_CUDA,
                                               GENERATIVEQC_METHOD_RHF,
                                               0,
                                               0};
  generativeqc_scf_defaults_v1 output{sizeof(output), GENERATIVEQC_ABI_VERSION};
  const auto query = [&](const generativeqc_ks_options* options, std::string_view identity) {
    return generativeqc_scf_defaults_query_v1(&context, options, identity.data(), identity.size(),
                                              &output);
  };
  require_test(query(nullptr, {}) == GENERATIVEQC_STATUS_SUCCESS);
  require_test(output.qualified_defaults && output.energy_tolerance == 1e-9 &&
               output.density_tolerance == 1e-5 && output.screening_tolerance == 1e-12);
  context.method = GENERATIVEQC_METHOD_UHF;
  require_test(query(nullptr, {}) == GENERATIVEQC_STATUS_SUCCESS && !output.qualified_defaults);
  context.method = GENERATIVEQC_METHOD_RHF;
  context.has_ecp = 1;
  require_test(query(nullptr, {}) == GENERATIVEQC_STATUS_SUCCESS && !output.qualified_defaults);
  context.has_ecp = 0;
  context.backend = GENERATIVEQC_BACKEND_CPU_REFERENCE;
  require_test(query(nullptr, {}) == GENERATIVEQC_STATUS_SUCCESS && !output.qualified_defaults);
  require_test(output.energy_tolerance == 1e-10 && output.density_tolerance == 1e-8);
  context.backend = GENERATIVEQC_BACKEND_CUDA;

  const auto grid = generativeqc::dft::default_grid_spec();
  generativeqc_ks_options options{};
  options.struct_size = sizeof(options);
  options.abi_version = GENERATIVEQC_ABI_VERSION;
  options.grid_version = grid.version;
  options.radial_points = grid.radial_points;
  options.angular_polar = grid.angular_polar;
  options.angular_azimuth = grid.angular_azimuth;
  options.partition_iterations = grid.partition_iterations;
  options.coincident_tolerance = grid.coincident_tolerance;
  options.element_radii = grid.element_radii.data();
  options.element_radius_count = grid.element_radii.size();
  options.xc_execution_schedule = GENERATIVEQC_XC_EXECUTION_DEVICE_FUSED;
  for (auto name : {"pbe-rks", "pbe0-rks", "b3lyp-rks", "r2scan-rks", "m06-2x-rks"}) {
    const auto* model = generativeqc::methods::named_ks::find_method(name);
    require_test(model && query(&options, model->identity) == GENERATIVEQC_STATUS_SUCCESS);
    require_test(output.qualified_defaults);
  }
  const auto identity = generativeqc::methods::named_ks::find_method("pbe-rks")->identity;
  for (auto name : {"pbe-uks", "wb97m-v-rks", "pbe-d3(bj)-rks"}) {
    const auto* model = generativeqc::methods::named_ks::find_method(name);
    require_test(model && query(&options, model->identity) == GENERATIVEQC_STATUS_SUCCESS);
    require_test(!output.qualified_defaults);
  }
  require_test(query(&options, std::string(64, '0')) == GENERATIVEQC_STATUS_SUCCESS);
  require_test(!output.qualified_defaults);
  options.radial_points = 4;
  require_test(query(&options, identity) == GENERATIVEQC_STATUS_SUCCESS &&
               !output.qualified_defaults);
  options.radial_points = 3;
  auto radii = grid.element_radii;
  radii[103] = std::nextafter(radii[103], std::numeric_limits<double>::infinity());
  options.element_radii = radii.data();
  require_test(query(&options, identity) == GENERATIVEQC_STATUS_SUCCESS &&
               !output.qualified_defaults);
  options.element_radii = grid.element_radii.data();
  options.xc_execution_schedule = GENERATIVEQC_XC_EXECUTION_HOST_UNFUSED;
  require_test(query(&options, identity) == GENERATIVEQC_STATUS_SUCCESS &&
               !output.qualified_defaults);
  options.xc_execution_schedule = GENERATIVEQC_XC_EXECUTION_DEVICE_FUSED;

  const auto before = output;
  options.element_radii = nullptr;
  require_test(query(&options, identity) == GENERATIVEQC_STATUS_INVALID_ARGUMENT);
  require_test(std::memcmp(&before, &output, sizeof(output)) == 0);
  options.element_radii = grid.element_radii.data();
  options.abi_version += 1;
  require_test(query(&options, identity) == GENERATIVEQC_STATUS_ABI_MISMATCH);
  require_test(std::memcmp(&before, &output, sizeof(output)) == 0);
  options.abi_version -= 1;
  require_test(query(&options, "PBE") == GENERATIVEQC_STATUS_INVALID_ARGUMENT);
  require_test(std::memcmp(&before, &output, sizeof(output)) == 0);
  return 0;
}
