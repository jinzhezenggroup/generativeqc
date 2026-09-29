#include <array>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include "generativeqc/generativeqc.h"

namespace {

struct Context {
  generativeqc_context* value{};
  explicit Context(generativeqc_backend backend) {
    generativeqc_context_descriptor descriptor{sizeof(generativeqc_context_descriptor),
                                               GENERATIVEQC_ABI_VERSION, 0, backend};
    const auto status = generativeqc_context_create(&descriptor, &value);
    if (status != GENERATIVEQC_STATUS_SUCCESS)
      throw std::runtime_error("failed to create D3 parity context");
  }
  ~Context() { generativeqc_context_destroy(value); }
  Context(const Context&) = delete;
  Context& operator=(const Context&) = delete;
};

struct Fleet {
  std::vector<std::vector<std::int32_t>> numbers;
  std::vector<std::vector<double>> coordinates;
  std::vector<generativeqc_d3_system_descriptor> descriptors;
};

Fleet make_fleet(std::array<std::size_t, 2> atom_counts = {64, 96}) {
  Fleet fleet;
  for (std::size_t atoms : atom_counts) {
    fleet.numbers.emplace_back(atoms);
    fleet.coordinates.emplace_back(3 * atoms);
    auto& z = fleet.numbers.back();
    auto& xyz = fleet.coordinates.back();
    for (std::size_t atom = 0; atom < atoms; ++atom) {
      constexpr std::array<std::int32_t, 4> species{6, 8, 7, 1};
      z[atom] = species[atom % species.size()];
      const double layer = static_cast<double>(atom / 16);
      const double row = static_cast<double>((atom / 4) % 4);
      const double col = static_cast<double>(atom % 4);
      xyz[3 * atom] = 2.35 * col + 0.07 * row;
      xyz[3 * atom + 1] = 2.55 * row + 0.11 * layer;
      xyz[3 * atom + 2] = 2.75 * layer + 0.05 * col;
    }
  }
  for (std::size_t i = 0; i < fleet.numbers.size(); ++i) {
    fleet.descriptors.push_back(generativeqc_d3_system_descriptor{
        sizeof(generativeqc_d3_system_descriptor), GENERATIVEQC_ABI_VERSION,
        fleet.numbers[i].data(), fleet.coordinates[i].data(),
        static_cast<std::uint32_t>(fleet.numbers[i].size())});
  }
  return fleet;
}

generativeqc_d3_bj_descriptor bj_model(double s9 = 0.0) {
  return {sizeof(generativeqc_d3_bj_descriptor),
          GENERATIVEQC_ABI_VERSION,
          GENERATIVEQC_D3_DAMPING_BJ,
          1.0,
          0.7875,
          0.4289,
          4.4407,
          s9,
          0.0,
          0.0,
          0.0,
          256u << 20,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0};
}

generativeqc_d3_bj_descriptor zero_model() {
  return {sizeof(generativeqc_d3_bj_descriptor),
          GENERATIVEQC_ABI_VERSION,
          GENERATIVEQC_D3_DAMPING_ZERO,
          1.0,
          0.722,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          256u << 20,
          1.217,
          1.0,
          14.0,
          0.0,
          0.0};
}

struct Outputs {
  std::vector<double> energy;
  std::vector<std::vector<double>> gradient;
  std::vector<generativeqc_status> status;
  std::vector<generativeqc_backend> backend;
};

Outputs execute(generativeqc_backend backend, const Fleet& fleet,
                const generativeqc_d3_bj_descriptor& model,
                const std::vector<std::vector<double>>* changed = nullptr,
                bool want_gradient = true) {
  Context context(backend);
  generativeqc_d3_batch* batch{};
  auto status = generativeqc_d3_batch_prepare(context.value, fleet.descriptors.data(),
                                              static_cast<std::uint32_t>(fleet.descriptors.size()),
                                              &model, &batch);
  if (status != GENERATIVEQC_STATUS_SUCCESS) {
    const auto* detail = generativeqc_context_get_last_detail(context.value);
    throw std::runtime_error(detail ? detail : "D3 parity prepare failed");
  }

  Outputs outputs;
  const std::size_t systems = fleet.descriptors.size();
  outputs.energy.resize(systems);
  outputs.gradient.resize(systems);
  outputs.status.resize(systems);
  outputs.backend.resize(systems);

  std::vector<generativeqc_d3_batch_item_result_descriptor> results(systems);
  std::vector<generativeqc_d3_batch_input_descriptor> inputs;
  if (changed) inputs.resize(systems);
  for (std::size_t i = 0; i < systems; ++i) {
    outputs.gradient[i].resize(want_gradient ? 3 * fleet.numbers[i].size() : 0);
    results[i] = generativeqc_d3_batch_item_result_descriptor{
        sizeof(generativeqc_d3_batch_item_result_descriptor),
        GENERATIVEQC_ABI_VERSION,
        GENERATIVEQC_STATUS_INTERNAL_ERROR,
        0.0,
        want_gradient ? outputs.gradient[i].data() : nullptr,
        want_gradient ? static_cast<std::uint32_t>(outputs.gradient[i].size()) : 0u,
        backend};
    if (changed) {
      inputs[i] = generativeqc_d3_batch_input_descriptor{
          sizeof(generativeqc_d3_batch_input_descriptor), GENERATIVEQC_ABI_VERSION,
          (*changed)[i].data(), static_cast<std::uint32_t>((*changed)[i].size())};
    }
  }

  status =
      generativeqc_d3_batch_execute(batch, changed ? inputs.data() : nullptr,
                                    changed ? static_cast<std::uint32_t>(inputs.size()) : 0u,
                                    results.data(), static_cast<std::uint32_t>(results.size()));
  generativeqc_d3_batch_destroy(batch);
  if (status != GENERATIVEQC_STATUS_SUCCESS)
    throw std::runtime_error("D3 parity batch execution failed");

  for (std::size_t i = 0; i < systems; ++i) {
    outputs.energy[i] = results[i].energy;
    outputs.status[i] = results[i].status;
    outputs.backend[i] = results[i].executed_backend;
  }
  return outputs;
}

void require_close(double actual, double expected, double atol, double rtol, const char* message) {
  if (!std::isfinite(actual) || !std::isfinite(expected)) {
    std::cerr << message << ": nonfinite comparison, expected " << expected << ", got " << actual
              << '\n';
    throw std::runtime_error(message);
  }
  const double tolerance = atol + rtol * std::abs(expected);
  if (std::abs(actual - expected) > tolerance) {
    std::cerr << message << ": expected " << expected << ", got " << actual << ", tolerance "
              << tolerance << '\n';
    throw std::runtime_error(message);
  }
}

void compare(const Outputs& cpu, const Outputs& gpu, bool gradient) {
  if (cpu.energy.size() != gpu.energy.size()) throw std::runtime_error("fleet size mismatch");
  for (std::size_t i = 0; i < cpu.energy.size(); ++i) {
    if (cpu.status[i] != GENERATIVEQC_STATUS_SUCCESS ||
        gpu.status[i] != GENERATIVEQC_STATUS_SUCCESS)
      throw std::runtime_error("D3 CPU/CUDA item failed");
    if (cpu.backend[i] != GENERATIVEQC_BACKEND_CPU_REFERENCE ||
        gpu.backend[i] != GENERATIVEQC_BACKEND_CUDA)
      throw std::runtime_error("D3 CPU/CUDA backend publication mismatch");
    require_close(gpu.energy[i], cpu.energy[i], 5.0e-12, 5.0e-11, "D3 CPU/CUDA energy mismatch");
    if (!gradient) continue;
    if (cpu.gradient[i].size() != gpu.gradient[i].size())
      throw std::runtime_error("D3 CPU/CUDA gradient size mismatch");
    for (std::size_t q = 0; q < cpu.gradient[i].size(); ++q)
      require_close(gpu.gradient[i][q], cpu.gradient[i][q], 2.0e-10, 2.0e-8,
                    "D3 CPU/CUDA gradient mismatch");
  }
}

void test_variant(const Fleet& fleet, const generativeqc_d3_bj_descriptor& model) {
  compare(execute(GENERATIVEQC_BACKEND_CPU_REFERENCE, fleet, model),
          execute(GENERATIVEQC_BACKEND_CUDA, fleet, model), true);
}

void test_changed_geometry(const Fleet& fleet) {
  auto changed = fleet.coordinates;
  changed[0][0] += 0.037;
  changed[0][11] -= 0.023;
  changed[1][6] += 0.019;
  changed[1][37] -= 0.031;
  const auto model = zero_model();
  compare(execute(GENERATIVEQC_BACKEND_CPU_REFERENCE, fleet, model, &changed),
          execute(GENERATIVEQC_BACKEND_CUDA, fleet, model, &changed), true);
}

void test_energy_only(const Fleet& fleet) {
  const auto model = bj_model();
  compare(execute(GENERATIVEQC_BACKEND_CPU_REFERENCE, fleet, model, nullptr, false),
          execute(GENERATIVEQC_BACKEND_CUDA, fleet, model, nullptr, false), false);
}

void test_energy_only_overflow_failure_isolated() {
  const auto fleet = make_fleet({2048, 8});
  auto model = zero_model();
  model.s6 = std::numeric_limits<double>::max() / 2.0;
  model.s8 = 0.0;

  const auto cpu = execute(GENERATIVEQC_BACKEND_CPU_REFERENCE, fleet, model, nullptr, false);
  const auto gpu = execute(GENERATIVEQC_BACKEND_CUDA, fleet, model, nullptr, false);
  if (cpu.status[0] != GENERATIVEQC_STATUS_NUMERICAL_FAILURE ||
      gpu.status[0] != GENERATIVEQC_STATUS_NUMERICAL_FAILURE)
    throw std::runtime_error("D3 cooperative energy-only overflow did not fail closed");
  if (!std::isnan(cpu.energy[0]) || !std::isnan(gpu.energy[0]))
    throw std::runtime_error("failed D3 overflow item published a numeric energy");
  if (cpu.status[1] != GENERATIVEQC_STATUS_SUCCESS || gpu.status[1] != GENERATIVEQC_STATUS_SUCCESS)
    throw std::runtime_error("D3 overflow poisoned the finite neighboring item");
  require_close(gpu.energy[1], cpu.energy[1], 5.0e-12, 5.0e-11,
                "D3 finite neighbor energy mismatch after overflow");
}

}  // namespace

int main() {
  try {
    const auto fleet = make_fleet();
    test_variant(fleet, bj_model());
    test_variant(fleet, zero_model());
    test_variant(fleet, bj_model(1.0));
    test_changed_geometry(fleet);
    test_energy_only(fleet);
    test_energy_only_overflow_failure_isolated();
    std::cout << "D3 cooperative CUDA CPU-parity, variants, changed-geometry and energy-only "
                 "tests passed\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
