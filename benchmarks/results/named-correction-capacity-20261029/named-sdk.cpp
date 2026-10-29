#include <array>
#include <cmath>
#include <cstdio>
#include <generativeqc/generativeqc.hpp>
#include <memory>
#include <stdexcept>
#include <string>

namespace qc = generativeqc;
void require(bool good, const char* why) {
  if (!good) throw std::runtime_error(why);
}
int main() {
  try {
    qc::Context context({sizeof(generativeqc_context_descriptor), GENERATIVEQC_ABI_VERSION, 0,
                         GENERATIVEQC_BACKEND_CUDA});
    const std::array<generativeqc_atom, 2> h2{{{1, 0, 0, -.7}, {1, 0, 0, .7}}};
    qc::System system(context, h2, "sto-3g");
    for (const auto* selector : {"pbe-d3(bj)-rks", "pbe-d3(0)-rks", "pbe-d4(bj-eeq-atm)-rks",
                                 "pbe-d3(bj)-uks", "pbe-d3(0)-uks", "pbe-d4(bj-eeq-atm)-uks"}) {
      auto method = qc::default_method_descriptor(
          std::string(selector).ends_with("uks") ? "pbe-uks" : "pbe-rks");
      method.energy_tolerance = 1e-12;
      method.density_tolerance = 1e-10;
      method.max_iterations = 250;
      const generativeqc_named_ks_descriptor named{sizeof(named), GENERATIVEQC_ABI_VERSION,
                                                   selector, nullptr, 256ULL << 20};
      const generativeqc_device_resource_buffer_v1 row{2ULL << 30, 4, 0, 0, 0, 0};
      const generativeqc_device_resource_descriptor descriptor{
          sizeof(descriptor), GENERATIVEQC_ABI_VERSION, &row, 1,
          GENERATIVEQC_DEVICE_RESOURCE_NUMERIC_CAP};
      generativeqc_device_resources* raw_resources{};
      qc::check(generativeqc_device_resources_create(&descriptor, &raw_resources));
      std::unique_ptr<generativeqc_device_resources,
                      decltype(&generativeqc_device_resources_destroy)>
          resources(raw_resources, generativeqc_device_resources_destroy);
      generativeqc_calculation* raw{};
      qc::check(generativeqc_calculation_prepare_named_ks_with_device_resources_v1(
          context.get(), system.get(), &method, &named, resources.get(), &raw));
      std::unique_ptr<generativeqc_calculation, decltype(&generativeqc_calculation_destroy)>
          calculation(raw, generativeqc_calculation_destroy);
      generativeqc_device_resource_diagnostic diagnostic{sizeof(diagnostic),
                                                         GENERATIVEQC_ABI_VERSION};
      qc::check(generativeqc_device_resources_read(resources.get(), &diagnostic));
      require(!diagnostic.capacity_reserved_bytes, "incomplete named promise admitted");
      std::array<double, 6> forces{};
      generativeqc_result_descriptor result{};
      result.struct_size = sizeof(result);
      result.abi_version = GENERATIVEQC_ABI_VERSION;
      result.forces = forces.data();
      result.force_count = forces.size();
      qc::check(generativeqc_calculation_execute(calculation.get(), &result));
      require(result.converged && result.executed_backend == GENERATIVEQC_BACKEND_CUDA &&
                  std::isfinite(result.energy),
              "named CUDA SDK endpoint incomplete");
      for (auto force : forces) require(std::isfinite(force), "nonfinite force");
      calculation.reset();
      qc::check(generativeqc_device_resources_read(resources.get(), &diagnostic));
      require(!diagnostic.numeric_live_bytes && !diagnostic.provider_reserved_bytes &&
                  !diagnostic.capacity_reserved_bytes,
              "SDK teardown retained charges");
      std::printf("%s complete E/F and cleanup passed\n", selector);
    }
  } catch (const std::exception& error) {
    std::fprintf(stderr, "%s\n", error.what());
    return 1;
  }
}
