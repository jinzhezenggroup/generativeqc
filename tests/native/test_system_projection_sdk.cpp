#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <vector>

#include "generativeqc/generativeqc.hpp"

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

/** Public SDK operation with a normalized one-AO He source; no Python runtime. */
void run() {
  auto descriptor = versioned<generativeqc_context_descriptor>();
  descriptor.backend = GENERATIVEQC_BACKEND_CPU_REFERENCE;
  generativeqc::Context context(descriptor);
  const generativeqc_atom atom{2, 0, 0, 0};
  auto model = versioned<generativeqc_system_descriptor>();
  model.atoms = &atom;
  model.atom_count = 1;
  model.multiplicity = 1;
  model.basis_representation = GENERATIVEQC_BASIS_CARTESIAN;
  generativeqc_system *left{}, *right{};
  require(generativeqc_system_create_bundled(context.get(), &model, "sto-3g", nullptr, &right) == 0,
          "native source preparation failed");
  const std::unique_ptr<generativeqc_system, decltype(&generativeqc_system_destroy)> source(
      right, generativeqc_system_destroy);
  require(
      generativeqc_system_create_bundled(context.get(), &model, "def2-svp", nullptr, &left) == 0,
      "native target preparation failed");
  const std::unique_ptr<generativeqc_system, decltype(&generativeqc_system_destroy)> target(
      left, generativeqc_system_destroy);
  auto controls = versioned<generativeqc_hf_system_projection_controls_v1>();
  controls.unrestricted = 1;
  controls.maximum_ao = 4096;
  controls.maximum_host_bytes = 1u << 28;
  controls.relative_threshold = 1e-10;
  controls.validation_tolerance = 1e-7;
  controls.maximum_residual = 0.5;
  auto capacity = versioned<generativeqc_hf_system_projection_capacity_v1>();
  require(generativeqc_hf_system_projection_query_v1(left, right, &controls, &capacity) == 0,
          "native metadata query failed");
  require(capacity.source_ao == 1 && capacity.spin_count == 2 && capacity.alpha_occupied == 1 &&
              capacity.beta_occupied == 1,
          "native projection occupation metadata differs");
  std::array<generativeqc_basis_projection_output_v1, 2> outputs;
  std::array<std::vector<double>, 2> coefficients, density;
  for (unsigned spin = 0; spin < 2; ++spin) {
    coefficients[spin].resize(capacity.target_ao, 17);
    density[spin].resize(capacity.target_ao * capacity.target_ao, 19);
    outputs[spin] = versioned<generativeqc_basis_projection_output_v1>();
    outputs[spin].coefficients = coefficients[spin].data();
    outputs[spin].coefficient_capacity = coefficients[spin].size();
    outputs[spin].density = density[spin].data();
    outputs[spin].density_capacity = density[spin].size();
  }
  std::array<double, 2> input{1, 1};
  const auto execute = [&]() {
    return generativeqc_hf_system_project_cpu_v1(context.get(), left, right, &controls,
                                                 input.data(), input.size(), outputs.data(), 2);
  };
  std::array<unsigned char, sizeof(outputs)> original{};
  std::memcpy(original.data(), outputs.data(), sizeof(outputs));
  controls.maximum_host_bytes = capacity.numeric_workspace_bytes - 1;
  require(execute() == GENERATIVEQC_STATUS_OUT_OF_MEMORY, "below-capacity transfer was admitted");
  require(std::memcmp(original.data(), outputs.data(), sizeof(outputs)) == 0,
          "refusal changed output descriptors");
  for (const auto& values : density)
    for (double value : values) require(value == 19, "refusal partially published density");
  controls.maximum_host_bytes = capacity.numeric_workspace_bytes;
  input[1] = 0.5;
  require(execute() == GENERATIVEQC_STATUS_INVALID_ARGUMENT, "impure beta transfer was admitted");
  require(std::memcmp(original.data(), outputs.data(), sizeof(outputs)) == 0,
          "beta refusal partially published alpha descriptor");
  for (const auto& values : density)
    for (double value : values) require(value == 19, "beta refusal partially published density");
  input[1] = 1;
  require(execute() == GENERATIVEQC_STATUS_SUCCESS, "exact-capacity SDK projection failed");
  for (const auto& output : outputs) {
    require(std::abs(output.electron_trace - 1) < 1e-10, "spin electron trace differs");
    require(output.metric_idempotency_error < 1e-10, "projected spin is not idempotent");
  }
}
}  // namespace

int main() {
  try {
    run();
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
