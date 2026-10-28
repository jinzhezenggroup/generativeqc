#include <array>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <string_view>

#include "generativeqc/generativeqc.hpp"
#include "generativeqc/ks.hpp"

void require(bool ok, const char* detail) {
  if (!ok) throw std::runtime_error(detail);
}
int main(int argc, char**) {
  try {
    const bool cuda = argc > 1;
    const auto backend = cuda ? GENERATIVEQC_BACKEND_CUDA : GENERATIVEQC_BACKEND_CPU_REFERENCE;
    const std::array<std::uint32_t, 2> counts{2, 3};
    generativeqc_d3_runtime_diagnostic capacity{};
    capacity.struct_size = sizeof(capacity);
    capacity.abi_version = GENERATIVEQC_ABI_VERSION;
    generativeqc::check(generativeqc_ks_d3_resource_query_v1(backend, counts.data(), 2, &capacity));
    require(capacity.system_count == 2 && capacity.total_atoms == 5, "wrong capacity shape");
    generativeqc::Context context(
        {sizeof(generativeqc_context_descriptor), GENERATIVEQC_ABI_VERSION, 0, backend});
    const std::array<generativeqc_atom, 2> h2{{{1, 0, 0, -.7}, {1, 0, 0, .7}}};
    const std::array<generativeqc_atom, 3> water{
        {{8, 0, 0, 0}, {1, 1.8, 0, 0}, {1, -.45, 1.75, 0}}};
    generativeqc::System first(context, h2, "sto-3g"), second(context, water, "sto-3g");
    const generativeqc_system* systems[]{first.get(), second.get()};
    auto method = generativeqc::default_method_descriptor("pbe-rks");
    generativeqc_named_ks_descriptor named{sizeof(named), GENERATIVEQC_ABI_VERSION,
                                           "pbe-d3(bj)-rks", nullptr, capacity.maximum_bytes - 1};
    generativeqc_batch* batch = nullptr;
    require(generativeqc_batch_prepare_named_ks_v1(context.get(), systems, 2, &method, &named, 0,
                                                   &batch) == GENERATIVEQC_STATUS_OUT_OF_MEMORY &&
                !batch,
            "below correction cap was admitted");
    named.dispersion_maximum_bytes = capacity.maximum_bytes;
    generativeqc_device_resources* resources = nullptr;
    generativeqc_device_resources* retained = nullptr;
    generativeqc_device_resource_buffer_v1 buffer{};
    buffer.bytes = 512ull << 20;
    buffer.space = 4;
    buffer.device = 0;
    generativeqc_device_resource_descriptor descriptor{};
    descriptor.struct_size = sizeof(descriptor);
    descriptor.abi_version = GENERATIVEQC_ABI_VERSION;
    descriptor.buffers = &buffer;
    descriptor.buffer_count = 1;
    generativeqc::check(generativeqc_device_resources_create(&descriptor, &resources));
    const auto status = generativeqc_batch_prepare_named_ks_with_device_resources_v1(
        context.get(), systems, 2, &method, &named, GENERATIVEQC_BATCH_ENABLE_WARM_STARTS,
        resources, &batch);
    if (cuda) {
      generativeqc::check(status);
      generativeqc::check(generativeqc_batch_get_device_resources(batch, &retained));
    } else {
      require(status == GENERATIVEQC_STATUS_NOT_IMPLEMENTED && !batch,
              "CPU device resources did not reject");
      generativeqc::check(
          generativeqc_batch_prepare_named_ks_v1(context.get(), systems, 2, &method, &named,
                                                 GENERATIVEQC_BATCH_ENABLE_WARM_STARTS, &batch));
    }
    generativeqc_device_resources_destroy(resources);
    resources = nullptr;
    std::array<double, 6> f1{};
    std::array<double, 9> f2{};
    std::array<generativeqc_batch_item_result_descriptor, 2> output{};
    for (auto& item : output) {
      item.struct_size = sizeof(item);
      item.abi_version = GENERATIVEQC_ABI_VERSION;
    }
    output[0].forces = f1.data();
    output[0].force_count = f1.size();
    output[1].forces = f2.data();
    output[1].force_count = f2.size();
    for (int replay = 0; replay != 2; ++replay) {
      generativeqc::check(
          generativeqc_batch_execute(batch, nullptr, 0, output.data(), output.size()));
      for (const auto& item : output)
        require(item.status == GENERATIVEQC_STATUS_SUCCESS && std::isfinite(item.energy),
                "complete SDK endpoint failed");
    }
    generativeqc_device_resource_diagnostic diagnostic{};
    diagnostic.struct_size = sizeof(diagnostic);
    diagnostic.abi_version = GENERATIVEQC_ABI_VERSION;
    if (retained) {
      generativeqc::check(generativeqc_device_resources_read(retained, &diagnostic));
      require(diagnostic.numeric_live_bytes >= capacity.device_bytes,
              "correction storage not charged");
    }
    generativeqc_batch_destroy(batch);
    if (retained) {
      generativeqc::check(generativeqc_device_resources_read(retained, &diagnostic));
      require(!diagnostic.numeric_live_bytes && !diagnostic.provider_reserved_bytes,
              "SDK destruction leaked retained resources");
      generativeqc_device_resources_destroy(retained);
    }
    resources = nullptr;
    retained = nullptr;
    generativeqc::check(generativeqc_device_resources_create(&descriptor, &resources));
    generativeqc_calculation* calculation = nullptr;
    const auto single_status = generativeqc_calculation_prepare_named_ks_with_device_resources_v1(
        context.get(), first.get(), &method, &named, resources, &calculation);
    if (cuda) {
      generativeqc::check(single_status);
      generativeqc::check(generativeqc_calculation_get_device_resources(calculation, &retained));
    } else {
      require(single_status == GENERATIVEQC_STATUS_NOT_IMPLEMENTED && !calculation,
              "CPU named scalar device admission did not reject");
      generativeqc::check(generativeqc_calculation_prepare_named_ks_v1(
          context.get(), first.get(), &method, &named, &calculation));
    }
    generativeqc_device_resources_destroy(resources);
    generativeqc_result_descriptor single{};
    single.struct_size = sizeof(single);
    single.abi_version = GENERATIVEQC_ABI_VERSION;
    single.forces = f1.data();
    single.force_count = f1.size();
    generativeqc::check(generativeqc_calculation_execute(calculation, &single));
    require(single.converged && std::isfinite(single.energy), "SDK scalar execution failed");
    generativeqc_calculation_destroy(calculation);
    if (retained) {
      generativeqc::check(generativeqc_device_resources_read(retained, &diagnostic));
      require(!diagnostic.numeric_live_bytes && !diagnostic.provider_reserved_bytes,
              "SDK scalar destruction leaked retained resources");
      generativeqc_device_resources_destroy(retained);
    }
    std::cout << "Installed SDK named KS resource ownership and complete E/F passed\n";
    return 0;
  } catch (const std::exception& e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}
