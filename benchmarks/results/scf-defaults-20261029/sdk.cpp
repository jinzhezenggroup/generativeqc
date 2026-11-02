// Installed-header smoke: compiler-derived model/grid facts are offline inputs.
#include <generativeqc/scf_defaults.h>

#include <array>
#include <cassert>
#include <cstring>
int main() {
  generativeqc_scf_defaults_context_v1 context{sizeof(context),
                                               GENERATIVEQC_ABI_VERSION,
                                               GENERATIVEQC_BACKEND_CUDA,
                                               GENERATIVEQC_METHOD_RHF,
                                               0,
                                               0};
  generativeqc_scf_defaults_v1 output{sizeof(output), GENERATIVEQC_ABI_VERSION};
  assert(generativeqc_scf_defaults_query_v1(&context, nullptr, nullptr, 0, &output) == 0);
  assert(output.qualified_defaults && output.energy_tolerance == 1e-9 &&
         output.density_tolerance == 1e-5);
  constexpr std::array<double, 119> radii{0.0,
                                          0.6614041435977716,
                                          2.645616574391086,
                                          2.7401028806193395,
                                          1.984212430793315,
                                          1.6062672058803025,
                                          1.322808287195543,
                                          1.2283219809672903,
                                          1.133835674739037,
                                          0.9448630622825309,
                                          2.8345891868475928,
                                          3.4015070242171115,
                                          2.8345891868475928,
                                          2.3621576557063273,
                                          2.0786987370215684,
                                          1.8897261245650618,
                                          1.8897261245650618,
                                          1.8897261245650618,
                                          3.4015070242171115,
                                          4.157397474043137,
                                          3.4015070242171115,
                                          3.0235617993040993,
                                          2.645616574391086,
                                          2.551130268162834,
                                          2.645616574391086,
                                          2.645616574391086,
                                          2.645616574391086,
                                          2.551130268162834,
                                          2.551130268162834,
                                          2.551130268162834,
                                          2.551130268162834,
                                          2.4566439619345806,
                                          2.3621576557063273,
                                          2.1731850432498208,
                                          2.1731850432498208,
                                          2.1731850432498208,
                                          3.590479636673617,
                                          4.440856392727896,
                                          3.7794522491301237,
                                          3.4015070242171115,
                                          2.929075493075846,
                                          2.7401028806193395,
                                          2.7401028806193395,
                                          2.551130268162834,
                                          2.4566439619345806,
                                          2.551130268162834,
                                          2.645616574391086,
                                          3.0235617993040993,
                                          2.929075493075846,
                                          2.929075493075846,
                                          2.7401028806193395,
                                          2.7401028806193395,
                                          2.645616574391086,
                                          2.645616574391086,
                                          3.96842486158663,
                                          4.913287923869161,
                                          4.062911167814883,
                                          3.6849659429018704,
                                          3.4959933304453648,
                                          3.4959933304453648,
                                          3.4959933304453648,
                                          3.4959933304453648,
                                          3.4959933304453648,
                                          3.4959933304453648,
                                          3.4015070242171115,
                                          3.307020717988858,
                                          3.307020717988858,
                                          3.307020717988858,
                                          3.307020717988858,
                                          3.307020717988858,
                                          3.307020717988858,
                                          3.307020717988858,
                                          2.929075493075846,
                                          2.7401028806193395,
                                          2.551130268162834,
                                          2.551130268162834,
                                          2.4566439619345806,
                                          2.551130268162834,
                                          2.551130268162834,
                                          2.551130268162834,
                                          2.8345891868475928,
                                          3.590479636673617,
                                          3.4015070242171115,
                                          3.0235617993040993,
                                          3.590479636673617,
                                          2.7401028806193395,
                                          3.96842486158663,
                                          3.4015070242171115,
                                          4.062911167814883,
                                          3.6849659429018704,
                                          3.4015070242171115,
                                          3.4015070242171115,
                                          3.307020717988858,
                                          3.307020717988858,
                                          3.307020717988858,
                                          3.307020717988858,
                                          3.307020717988858,
                                          3.307020717988858,
                                          3.307020717988858,
                                          3.307020717988858,
                                          3.307020717988858,
                                          3.307020717988858,
                                          3.307020717988858,
                                          3.307020717988858,
                                          0.0,
                                          0.0,
                                          0.0,
                                          0.0,
                                          0.0,
                                          0.0,
                                          0.0,
                                          0.0,
                                          0.0,
                                          0.0,
                                          0.0,
                                          0.0,
                                          0.0,
                                          0.0,
                                          0.0};
  generativeqc_ks_options grid{};
  grid.struct_size = sizeof(grid);
  grid.abi_version = GENERATIVEQC_ABI_VERSION;
  grid.grid_version = 4;
  grid.radial_points = 3;
  grid.angular_polar = 0;
  grid.angular_azimuth = 0;
  grid.partition_iterations = 3;
  grid.coincident_tolerance = 1e-12;
  grid.element_radii = radii.data();
  grid.element_radius_count = radii.size();
  grid.xc_execution_schedule = GENERATIVEQC_XC_EXECUTION_DEVICE_FUSED;
  const char* identity = "be2ab1065d31bce3ec9fbd06d90a79133e387b11d9e4f754eefd9d052ec6f254";
  auto query = [&] {
    return generativeqc_scf_defaults_query_v1(&context, &grid, identity, 64, &output);
  };
  assert(query() == 0 && output.qualified_defaults);
  context.has_ecp = 1;
  assert(query() == 0 && !output.qualified_defaults && output.density_tolerance == 1e-8);
  context.has_ecp = 0;
  grid.radial_points += 1;
  assert(query() == 0 && !output.qualified_defaults);
  const auto before = output;
  grid.element_radii = nullptr;
  assert(query() == GENERATIVEQC_STATUS_INVALID_ARGUMENT);
  assert(std::memcmp(&before, &output, sizeof(output)) == 0);
}
