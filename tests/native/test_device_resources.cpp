#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string_view>
#include <thread>

#include "generativeqc/generativeqc.hpp"

namespace {
void require(bool condition, const char* message) {
  if (!condition) throw std::runtime_error(message);
}

using Resources = std::unique_ptr<generativeqc_device_resources,
                                  decltype(&generativeqc_device_resources_destroy)>;

generativeqc_device_resource_diagnostic read(const generativeqc_device_resources* resources) {
  generativeqc_device_resource_diagnostic result{};
  result.struct_size = sizeof(result);
  result.abi_version = GENERATIVEQC_ABI_VERSION;
  require(generativeqc_device_resources_read(resources, &result) == GENERATIVEQC_STATUS_SUCCESS,
          "public native resource snapshot failed");
  return result;
}

void metadata_contracts() {
  std::array<generativeqc_device_resource_buffer_v1, 3> buffers{
      {{64, 4, 7, 0, 0, 0}, {32, 4, 7, 1, 1, 0}, {100, 4, 7, 1, 1, 1}}};
  generativeqc_device_resource_descriptor descriptor{};
  descriptor.struct_size = sizeof(descriptor);
  descriptor.abi_version = GENERATIVEQC_ABI_VERSION;
  descriptor.buffers = buffers.data();
  descriptor.buffer_count = buffers.size();
  generativeqc_device_resources* handle = nullptr;
  const auto create = [&] { return generativeqc_device_resources_create(&descriptor, &handle); };
  require(create() == GENERATIVEQC_STATUS_SUCCESS && handle,
          "public selected-inventory resource construction failed");
  Resources owner(handle, generativeqc_device_resources_destroy);
  const auto initial = read(owner.get());
  require(initial.numeric_limit == 64 && initial.declared_device_peak == 132 &&
              initial.device == 7 && initial.numeric_live_bytes == 0 &&
              initial.requested_bytes_available && initial.host_samples == 0 &&
              initial.cuda_arena_samples == 0,
          "native admission subtracted independent peaks or invented allocation samples");
  require(create() == GENERATIVEQC_STATUS_INVALID_ARGUMENT && handle == owner.get(),
          "public creation replaced a live resource metadata handle");
  handle = nullptr;
  for (const auto field : {std::size_t{1}, std::size_t{5}}) {
    const auto original = buffers;
    if (field == 1)
      buffers[0].space = 1;
    else
      buffers[0].runtime_allowance = 2;
    require(create() == GENERATIVEQC_STATUS_INVALID_ARGUMENT && !handle,
            "public inventory accepted an invalid atomic space/allowance");
    buffers = original;
  }
  buffers[0].last_phase = 0;
  buffers[0].first_phase = 1;
  require(create() == GENERATIVEQC_STATUS_INVALID_ARGUMENT && !handle,
          "reversed capacity lifetime was admitted");
  buffers[0].first_phase = 0;
  buffers[1].device = 8;
  require(create() == GENERATIVEQC_STATUS_NOT_IMPLEMENTED && !handle,
          "multiple devices were silently combined");
  buffers[1].device = 7;
  buffers[0].bytes = std::numeric_limits<std::uint64_t>::max();
  require(create() == GENERATIVEQC_STATUS_INVALID_ARGUMENT && !handle,
          "overflowing device capacities were admitted");
  buffers[0].bytes = 64;
  const auto before = buffers;
  require(generativeqc_device_resources_create(
              &descriptor, reinterpret_cast<generativeqc_device_resources**>(buffers.data())) ==
                  GENERATIVEQC_STATUS_INVALID_ARGUMENT &&
              std::memcmp(buffers.data(), before.data(), sizeof(buffers)) == 0,
          "overlapping inventory/output was modified");
  descriptor.abi_version = GENERATIVEQC_ABI_VERSION + 1;
  require(create() == GENERATIVEQC_STATUS_ABI_MISMATCH && !handle,
          "incompatible resource descriptor was accepted");
  auto invalid = initial;
  invalid.struct_size = sizeof(invalid) - 1;
  const auto saved = invalid;
  require(generativeqc_device_resources_read(owner.get(), &invalid) ==
                  GENERATIVEQC_STATUS_ABI_MISMATCH &&
              std::memcmp(&invalid, &saved, sizeof(invalid)) == 0,
          "rejected diagnostic changed caller storage");
  require(
      generativeqc_device_resources_read(nullptr, &invalid) == GENERATIVEQC_STATUS_INVALID_ARGUMENT,
      "null resource handle was accepted");
  generativeqc_device_resources_destroy(nullptr);
}

void reject_cpu_cutover_without_unbudgeted_retry() {
  const std::array<generativeqc_device_resource_buffer_v1, 1> buffers{{{4096, 4, 0, 0, 0, 0}}};
  generativeqc_device_resource_descriptor inventory{};
  inventory.struct_size = sizeof(inventory);
  inventory.abi_version = GENERATIVEQC_ABI_VERSION;
  inventory.buffers = buffers.data();
  inventory.buffer_count = buffers.size();
  generativeqc_device_resources* handle = nullptr;
  require(generativeqc_device_resources_create(&inventory, &handle) == GENERATIVEQC_STATUS_SUCCESS,
          "context-free device metadata unexpectedly required CUDA");
  Resources resources(handle, generativeqc_device_resources_destroy);
  generativeqc_context_descriptor descriptor{};
  descriptor.struct_size = sizeof(descriptor);
  descriptor.abi_version = GENERATIVEQC_ABI_VERSION;
  descriptor.backend = GENERATIVEQC_BACKEND_CPU_REFERENCE;
  generativeqc::Context context(descriptor);
  const std::array<generativeqc_atom, 2> atoms{{{1, 0, 0, -0.7}, {1, 0, 0, 0.7}}};
  generativeqc::System system(context, atoms, "sto-3g");
  auto method = generativeqc::default_method_descriptor("rhf");
  method.energy_tolerance = 1e-12;
  method.density_tolerance = 1e-10;
  method.max_iterations = 100;
  generativeqc_calculation* calculation = nullptr;
  require(generativeqc_calculation_prepare_with_device_resources(
              context.get(), system.get(), &method, resources.get(), &calculation) ==
                  GENERATIVEQC_STATUS_NOT_IMPLEMENTED &&
              !calculation,
          "CPU preparation silently retried without the requested device owner");
  const generativeqc_system* systems[]{system.get()};
  generativeqc_batch* batch = nullptr;
  require(generativeqc_batch_prepare_with_device_resources(context.get(), systems, 1, &method, 0,
                                                           resources.get(), &batch) ==
                  GENERATIVEQC_STATUS_NOT_IMPLEMENTED &&
              !batch,
          "CPU batch silently retried without the requested device owner");
  require(generativeqc_calculation_prepare(context.get(), system.get(), &method, &calculation) ==
              GENERATIVEQC_STATUS_SUCCESS,
          "ordinary public CPU preparation failed");
  std::unique_ptr<generativeqc_calculation, decltype(&generativeqc_calculation_destroy)> ordinary(
      calculation, generativeqc_calculation_destroy);
  generativeqc_device_resources* missing = nullptr;
  require(generativeqc_calculation_get_device_resources(ordinary.get(), &missing) ==
                  GENERATIVEQC_STATUS_NOT_IMPLEMENTED &&
              !missing,
          "unbudgeted calculation invented a resource owner");
  std::array<double, 6> forces{};
  generativeqc_result_descriptor result{};
  result.struct_size = sizeof(result);
  result.abi_version = GENERATIVEQC_ABI_VERSION;
  result.forces = forces.data();
  result.force_count = forces.size();
  require(
      generativeqc_calculation_execute(ordinary.get(), &result) == GENERATIVEQC_STATUS_SUCCESS &&
          std::abs(result.energy + 1.1167143251757694) < 2e-10 &&
          std::abs(forces[2] - 0.028454057162517143) < 2e-9,
      "resource metadata/refusal changed the independent ordinary CPU endpoint");
}

Resources capacity(std::uint64_t bytes, std::uint64_t device = 0) {
  const generativeqc_device_resource_buffer_v1 buffer{bytes, 4, device, 0, 0, 0};
  const generativeqc_device_resource_descriptor descriptor{sizeof(descriptor),
                                                           GENERATIVEQC_ABI_VERSION, &buffer, 1};
  generativeqc_device_resources* result = nullptr;
  generativeqc::check(generativeqc_device_resources_create(&descriptor, &result));
  return Resources(result, generativeqc_device_resources_destroy);
}

/** Scheduler-only public SDK/C ABI lifecycle checks, not a complete allocation
 * or performance certificate. H2 constants are independent reference values;
 * ragged CPU controls check dispatch/publication separately from PySCF gates. */
void cuda_prepared_lifetime() {
  generativeqc_context_descriptor descriptor{};
  descriptor.struct_size = sizeof(descriptor);
  descriptor.abi_version = GENERATIVEQC_ABI_VERSION;
  descriptor.backend = GENERATIVEQC_BACKEND_CUDA;
  generativeqc::Context context(descriptor);
  descriptor.backend = GENERATIVEQC_BACKEND_CPU_REFERENCE;
  generativeqc::Context cpu(descriptor);
  const std::array<generativeqc_atom, 2> hydrogen{{{1, 0, 0, -0.7}, {1, 0, 0, 0.7}}};
  const std::array<generativeqc_atom, 3> water{
      {{8, 0, 0, 0}, {1, 1.43, 0, 1.11}, {1, -1.43, 0, 1.11}}};
  generativeqc::System system(context, hydrogen, "sto-3g"), other(context, water, "sto-3g"),
      cpu_system(cpu, hydrogen, "sto-3g"), cpu_other(cpu, water, "sto-3g");
  for (const auto* name : {"rhf", "uhf"}) {
    auto method = generativeqc::default_method_descriptor(name);
    method.energy_tolerance = 1e-12;
    method.density_tolerance = 1e-10;
    method.max_iterations = 100;
    auto resources = capacity(64 * 1024 * 1024);
    generativeqc_calculation* handle = nullptr;
    generativeqc::check(generativeqc_calculation_prepare_with_device_resources(
        context.get(), system.get(), &method, resources.get(), &handle));
    std::unique_ptr<generativeqc_calculation, decltype(&generativeqc_calculation_destroy)> single(
        handle, generativeqc_calculation_destroy);
    generativeqc_device_resources* snapshot_handle = nullptr;
    generativeqc::check(
        generativeqc_calculation_get_device_resources(single.get(), &snapshot_handle));
    Resources monitor(snapshot_handle, generativeqc_device_resources_destroy);
    resources.reset();
    for (int replay = 0; replay < 2; ++replay) {
      std::array<double, 6> forces{};
      generativeqc_result_descriptor result{};
      result.struct_size = sizeof(result);
      result.abi_version = GENERATIVEQC_ABI_VERSION;
      result.forces = forces.data();
      result.force_count = forces.size();
      generativeqc::check(generativeqc_calculation_execute(single.get(), &result));
      require(result.executed_backend == GENERATIVEQC_BACKEND_CUDA && result.converged &&
                  std::abs(result.energy + 1.1167143251757694) < 2e-9 &&
                  std::abs(forces[2] - 0.028454057162517143) < 2e-8,
              "native resource-aware single E/F differs from independent H2 reference");
      const auto sample = read(monitor.get());
      require(sample.numeric_peak_bytes > 0 && sample.combined_peak_bytes <= sample.numeric_limit &&
                  sample.rejected_allocations == 0,
              "single native execution bypassed resource accounting");
    }
    single.reset();
    require(read(monitor.get()).numeric_live_bytes == 0 &&
                read(monitor.get()).provider_reserved_bytes == 0,
            "single close leaked retained numeric/provider charges");

    auto insufficient = capacity(1);
    handle = nullptr;
    auto status = generativeqc_calculation_prepare_with_device_resources(
        context.get(), system.get(), &method, insufficient.get(), &handle);
    single.reset(handle);
    if (status == GENERATIVEQC_STATUS_SUCCESS) {
      generativeqc_result_descriptor result{};
      result.struct_size = sizeof(result);
      result.abi_version = GENERATIVEQC_ABI_VERSION;
      status = generativeqc_calculation_execute(single.get(), &result);
    }
    require(status == GENERATIVEQC_STATUS_OUT_OF_MEMORY &&
                read(insufficient.get()).rejected_allocations > 0,
            "insufficient native single cap silently retried unbudgeted execution");
    single.reset();
    require(read(insufficient.get()).numeric_live_bytes == 0,
            "failed single preparation/execution leaked charges");

    const std::array<const generativeqc::System*, 3> cpu_systems{&cpu_system, &cpu_other,
                                                                 &cpu_system};
    generativeqc::Batch reference(cpu, cpu_systems, method);
    const auto expected = reference.execute();
    const std::array<const generativeqc_system*, 3> systems{system.get(), other.get(),
                                                            system.get()};
    resources = capacity(64 * 1024 * 1024);
    generativeqc_batch* batch_handle = nullptr;
    generativeqc::check(generativeqc_batch_prepare_with_device_resources(
        context.get(), systems.data(), systems.size(), &method,
        GENERATIVEQC_BATCH_ENABLE_WARM_STARTS, resources.get(), &batch_handle));
    std::unique_ptr<generativeqc_batch, decltype(&generativeqc_batch_destroy)> batch(
        batch_handle, generativeqc_batch_destroy);
    snapshot_handle = nullptr;
    generativeqc::check(generativeqc_batch_get_device_resources(batch.get(), &snapshot_handle));
    monitor.reset(snapshot_handle);
    resources.reset();
    for (int replay = 0; replay < 2; ++replay) {
      std::array<std::array<double, 9>, 3> forces{};
      std::array<generativeqc_batch_item_result_descriptor, 3> results{};
      for (std::size_t index = 0; index < results.size(); ++index) {
        results[index].struct_size = sizeof(results[index]);
        results[index].abi_version = GENERATIVEQC_ABI_VERSION;
        results[index].forces = forces[index].data();
        results[index].force_count = index == 1 ? 9 : 6;
      }
      generativeqc::check(
          generativeqc_batch_execute(batch.get(), nullptr, 0, results.data(), results.size()));
      for (std::size_t index = 0; index < results.size(); ++index) {
        const auto& result = results[index];
        require(result.status == GENERATIVEQC_STATUS_SUCCESS && result.converged &&
                    result.executed_backend == GENERATIVEQC_BACKEND_CUDA &&
                    std::abs(result.energy - expected[index].energy) < 2e-9,
                "native resource-aware ragged batch changed dispatch or energy publication");
        for (std::size_t component = 0; component < result.force_count; ++component)
          require(std::abs(forces[index][component] - expected[index].forces[component]) < 2e-8,
                  "native resource-aware ragged force publication differs from CPU control");
      }
      const auto sample = read(monitor.get());
      require(sample.numeric_live_bytes > 0 && sample.numeric_peak_bytes <= sample.numeric_limit &&
                  sample.rejected_allocations == 0,
              "ragged native execution lost retained charges after metadata close");
    }
    std::thread release([owner = std::move(batch)]() mutable { owner.reset(); });
    release.join();
    require(read(monitor.get()).numeric_live_bytes == 0 &&
                read(monitor.get()).provider_reserved_bytes == 0,
            "cross-thread native batch close leaked retained numeric/provider charges");
    auto mismatch = capacity(64 * 1024 * 1024, 7);
    handle = nullptr;
    require(generativeqc_calculation_prepare_with_device_resources(
                context.get(), system.get(), &method, mismatch.get(), &handle) ==
                    GENERATIVEQC_STATUS_INVALID_ARGUMENT &&
                !handle,
            "resource-aware preparation ignored visible-device mismatch");
  }
}
}  // namespace

int main(int argc, char** argv) {
  try {
    metadata_contracts();
    reject_cpu_cutover_without_unbudgeted_retry();
    if (argc == 2 && std::string_view(argv[1]) == "--cuda") cuda_prepared_lifetime();
    std::cout << "Public device resource metadata/refusal and unchanged CPU endpoint passed\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
