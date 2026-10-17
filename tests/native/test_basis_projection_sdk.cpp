#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <limits>
#include <stdexcept>

#include "generativeqc/generativeqc.h"

namespace {
void require(bool condition, const char* message) {
  if (!condition) throw std::runtime_error(message);
}

template <class Record>
Record versioned() {
  Record record{};
  record.struct_size = sizeof(record);
  record.abi_version = GENERATIVEQC_ABI_VERSION;
  return record;
}

void run() {
  auto descriptor = versioned<generativeqc_basis_projection_descriptor_v1>();
  descriptor.source_ao = 2;
  descriptor.target_ao = 3;
  descriptor.occupied = 2;
  descriptor.occupation = 2;
  descriptor.maximum_ao = 4096;
  descriptor.relative_threshold = 1e-10;
  descriptor.validation_tolerance = 1e-7;
  descriptor.maximum_residual = 1;
  auto capacity = versioned<generativeqc_basis_projection_capacity_v1>();
  require(generativeqc_basis_projection_query_v1(&descriptor, &capacity) ==
                  GENERATIVEQC_STATUS_SUCCESS &&
              capacity.coefficient_count == 6 && capacity.density_count == 9 &&
              capacity.numeric_workspace_bytes >= (6 + 9) * sizeof(double),
          "shape-only installed projection capacity differs");
  const auto original = capacity;
  descriptor.abi_version += 1;
  require(generativeqc_basis_projection_query_v1(&descriptor, &capacity) ==
                  GENERATIVEQC_STATUS_ABI_MISMATCH &&
              std::memcmp(&capacity, &original, sizeof(capacity)) == 0,
          "capacity ABI refusal modified output");
  descriptor.abi_version -= 1;
  descriptor.source_ao = std::numeric_limits<std::uint64_t>::max();
  descriptor.maximum_ao = std::numeric_limits<std::uint64_t>::max();
  require(generativeqc_basis_projection_query_v1(&descriptor, &capacity) ==
                  GENERATIVEQC_STATUS_INVALID_ARGUMENT &&
              std::memcmp(&capacity, &original, sizeof(capacity)) == 0,
          "capacity overflow was not transactional");
  descriptor.source_ao = 2;
  descriptor.maximum_ao = 4096;
  struct ExtendedDescriptor {
    generativeqc_basis_projection_descriptor_v1 descriptor;
    generativeqc_basis_projection_capacity_v1 aliased_output;
  } extended_descriptor{descriptor, versioned<generativeqc_basis_projection_capacity_v1>()};
  extended_descriptor.descriptor.struct_size = sizeof(extended_descriptor);
  require(generativeqc_basis_projection_query_v1(&extended_descriptor.descriptor,
                                                 &extended_descriptor.aliased_output) ==
              GENERATIVEQC_STATUS_INVALID_ARGUMENT,
          "query accepted output in an input descriptor extension");
  const auto context_descriptor = [] {
    auto record = versioned<generativeqc_context_descriptor>();
    record.backend = GENERATIVEQC_BACKEND_CPU_REFERENCE;
    return record;
  }();
  generativeqc_context* context = nullptr;
  require(generativeqc_context_create(&context_descriptor, &context) == GENERATIVEQC_STATUS_SUCCESS,
          "installed projection CPU context failed");
  struct ContextGuard {
    generativeqc_context* context;
    ~ContextGuard() { generativeqc_context_destroy(context); }
  } guard{context};
  const std::array<double, 4> source{{1, 0, 0, 1}};
  const std::array<double, 9> target{{1, 0, 0, 0, 1, 0, 0, 0, 1}};
  const std::array<double, 6> cross{{0.8, 0, 0, 0.6, 0, 0.8}};
  const std::array<double, 9> expected{{2, 0, 0, 0, 0.72, 0.96, 0, 0.96, 1.28}};
  descriptor.source_overlap = source.data();
  descriptor.target_overlap = target.data();
  descriptor.cross_overlap = cross.data();
  descriptor.source_state = source.data();
  descriptor.source_overlap_count = source.size();
  descriptor.target_overlap_count = target.size();
  descriptor.cross_overlap_count = cross.size();
  descriptor.source_state_count = source.size();
  std::array<double, 7> coefficients;
  std::array<double, 10> density;
  coefficients.fill(-77);
  density.fill(-88);
  auto output = versioned<generativeqc_basis_projection_output_v1>();
  output.coefficients = coefficients.data();
  output.coefficient_capacity = coefficients.size();
  output.density = density.data();
  output.density_capacity = density.size();
  const auto untouched = output;
  descriptor.maximum_workspace_bytes = capacity.numeric_workspace_bytes - 1;
  const auto good_pointer = descriptor.source_overlap;
  descriptor.source_overlap = reinterpret_cast<const double*>(1);
  require(generativeqc_basis_project_cpu_v1(context, &descriptor, &output) ==
                  GENERATIVEQC_STATUS_OUT_OF_MEMORY &&
              std::memcmp(&output, &untouched, sizeof(output)) == 0 && density[0] == -88,
          "below-limit execution read data or modified output");
  descriptor.source_overlap = good_pointer;
  descriptor.maximum_workspace_bytes = capacity.numeric_workspace_bytes;
  require(generativeqc_basis_project_cpu_v1(context, &descriptor, &output) ==
              GENERATIVEQC_STATUS_SUCCESS,
          "exact-limit installed projection failed");
  for (std::size_t index = 0; index < expected.size(); ++index)
    require(std::abs(density[index] - expected[index]) < 2e-12,
            "installed projection differs from ambient oracle");
  require(coefficients.back() == -77 && density.back() == -88 && output.source_metric_rank == 2 &&
              output.target_metric_rank == 3 && output.occupied_rank == 2 &&
              std::abs(output.electron_trace - 4) < 2e-12,
          "installed detached output/diagnostics/tails differ");
  const auto published = output;
  struct ExtendedOutput {
    generativeqc_basis_projection_output_v1 output;
    std::array<double, 8> aliased_coefficients;
  } extended_output{published, {}};
  extended_output.output.struct_size = sizeof(extended_output);
  extended_output.output.coefficients = extended_output.aliased_coefficients.data();
  require(generativeqc_basis_project_cpu_v1(context, &descriptor, &extended_output.output) ==
              GENERATIVEQC_STATUS_INVALID_ARGUMENT,
          "execution accepted output in an output descriptor extension");
  const auto published_density = density;
  const auto published_coefficients = coefficients;
  const auto failure = [&](const auto& malformed, generativeqc_status status) {
    auto sentinel = published;
    require(generativeqc_basis_project_cpu_v1(context, &malformed, &sentinel) == status &&
                std::memcmp(&sentinel, &published, sizeof(sentinel)) == 0 &&
                density == published_density && coefficients == published_coefficients,
            "projection refusal changed caller state");
  };
  auto malformed = descriptor;
  malformed.input_kind = 2;
  failure(malformed, GENERATIVEQC_STATUS_INVALID_ARGUMENT);
  malformed = descriptor;
  malformed.source_state_count -= 1;
  failure(malformed, GENERATIVEQC_STATUS_INVALID_ARGUMENT);
  malformed = descriptor;
  malformed.abi_version += 1;
  failure(malformed, GENERATIVEQC_STATUS_ABI_MISMATCH);
  malformed = descriptor;
  const std::array<double, 4> invalid_metric{{1, 0, 0, -1}};
  malformed.source_overlap = invalid_metric.data();
  failure(malformed, GENERATIVEQC_STATUS_INVALID_ARGUMENT);
  auto aliased = published;
  aliased.density = coefficients.data();
  require(generativeqc_basis_project_cpu_v1(context, &descriptor, &aliased) ==
                  GENERATIVEQC_STATUS_INVALID_ARGUMENT &&
              density == published_density && coefficients == published_coefficients,
          "overlapping output buffers were accepted");
  descriptor.input_kind = 1;
  const std::array<double, 4> source_density{{2, 0, 0, 2}};
  descriptor.source_state = source_density.data();
  require(generativeqc_basis_project_cpu_v1(context, &descriptor, &output) ==
              GENERATIVEQC_STATUS_SUCCESS,
          "installed density reconstruction failed");
  descriptor.occupied = 0;
  const std::array<double, 4> zero{};
  descriptor.source_state = zero.data();
  output.coefficients = nullptr;
  output.coefficient_capacity = 0;
  require(generativeqc_basis_project_cpu_v1(context, &descriptor, &output) ==
                  GENERATIVEQC_STATUS_SUCCESS &&
              output.electron_trace == 0 && output.occupied_rank == 0,
          "empty occupied channel invented electrons");
  for (std::size_t index = 0; index < 9; ++index)
    require(density[index] == 0, "empty spin density is nonzero");
}
}  // namespace

int main() {
  try {
    run();
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
