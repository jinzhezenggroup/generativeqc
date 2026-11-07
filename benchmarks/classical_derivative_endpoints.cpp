#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "dft/dispersion/d3_runtime.hpp"
#include "dft/dispersion/d4_runtime.hpp"
#include "dft/dispersion/gcp_r2scan3c.hpp"
#if GENERATIVEQC_HAS_CUDA
#include "dft/dispersion/gcp_cuda.hpp"
#endif
#include "generated_method_parameters.hpp"

namespace {
using namespace generativeqc::dft::dispersion;
using Clock = std::chrono::steady_clock;

struct Inputs {
  std::vector<std::uint32_t> offsets{0};
  std::vector<std::int32_t> numbers;
  std::vector<double> coordinates;
  std::vector<double> charges;
  std::vector<std::uint8_t> active;
  std::vector<std::uint8_t> gradient;

  Inputs(int atoms, int systems) : charges(systems, 0.0), active(systems, 1), gradient(systems, 1) {
    for (int system = 0; system < systems; ++system) {
      for (int atom = 0; atom < atoms; ++atom) {
        numbers.push_back(atom % 4 == 0 ? 6 : atom % 4 == 1 ? 8 : 1);
        coordinates.push_back(3.1 * (atom % 5) + 0.03 * system);
        coordinates.push_back(2.9 * ((atom / 5) % 5) + 0.1 * (atom % 2));
        coordinates.push_back(3.3 * (atom / 25) + 0.13 * (atom % 3));
      }
      offsets.push_back(static_cast<std::uint32_t>(numbers.size()));
    }
  }
};

struct Result {
  std::vector<double> energies, gradients, charges;
};

struct GcpEndpoint {
  generativeqc_backend backend;
  std::vector<std::uint32_t> offsets;
  std::vector<std::int32_t> numbers;
#if GENERATIVEQC_HAS_CUDA
  std::vector<std::unique_ptr<GcpCudaPlan>> devices;
#endif
};

std::unique_ptr<GcpEndpoint> prepare_gcp(const Inputs& inputs, generativeqc_backend backend) {
  auto plan = std::make_unique<GcpEndpoint>();
  plan->backend = backend;
  plan->offsets = inputs.offsets;
  plan->numbers = inputs.numbers;
#if GENERATIVEQC_HAS_CUDA
  if (backend == GENERATIVEQC_BACKEND_CUDA) {
    for (std::size_t system = 0; system + 1 < inputs.offsets.size(); ++system) {
      const auto begin = inputs.offsets[system];
      const auto atoms = inputs.offsets[system + 1] - begin;
      plan->devices.push_back(std::make_unique<GcpCudaPlan>(
          0, std::span<const std::int32_t>(inputs.numbers.data() + begin, atoms),
          GcpCudaPlan::numeric_bytes(atoms)));
    }
  }
#endif
  return plan;
}

void execute(GcpEndpoint& plan, const Inputs& inputs, Result& result) {
  result.energies.resize(inputs.offsets.size() - 1);
  result.gradients.resize(inputs.coordinates.size());
  for (std::size_t system = 0; system + 1 < inputs.offsets.size(); ++system) {
    const auto begin = inputs.offsets[system];
    const auto atoms = inputs.offsets[system + 1] - begin;
#if GENERATIVEQC_HAS_CUDA
    if (plan.backend == GENERATIVEQC_BACKEND_CUDA) {
      plan.devices[system]->execute(
          std::span<const double>(inputs.coordinates.data() + 3 * begin, 3 * atoms),
          result.energies[system],
          std::span<double>(result.gradients.data() + 3 * begin, 3 * atoms));
      continue;
    }
#endif
    if (evaluate_r2scan3c_gcp(atoms, plan.numbers.data() + begin,
                              inputs.coordinates.data() + 3 * begin, r2scan3c_gcp_parameters(),
                              result.energies.data() + system,
                              result.gradients.data() + 3 * begin) != GCPStatus::success)
      throw std::runtime_error("gCP item failed");
  }
}

void require(generativeqc_status status, const std::string& detail) {
  if (status != GENERATIVEQC_STATUS_SUCCESS) throw std::runtime_error(detail);
}

std::unique_ptr<D3Plan> prepare_d3(const Inputs& inputs, generativeqc_backend backend, bool zero) {
  std::string detail;
  generativeqc_status status{};
  D3ModelParameters parameters{};
  parameters.damping = zero ? D3Damping::zero : D3Damping::bj;
  parameters.bj = {1.0, 0.95948085, 0.38574991, 4.80688534, 0.0};
  parameters.zero = {1.0, 0.722, 1.217, 1.0, 14.0, 0.0, 0.0, 0.0};
  parameters.atm = {zero ? 0.0 : 1.0, 0.0, 0.0, 0.0};
  parameters.atm_enabled = !zero;
  auto plan = D3Plan::prepare(backend, 0, inputs.offsets, inputs.numbers, inputs.coordinates,
                              parameters, 512ULL << 20, detail, status);
  require(status, detail);
  if (!plan) throw std::runtime_error("D3 prepare returned no plan");
  return plan;
}

std::unique_ptr<D4Plan> prepare_d4(const Inputs& inputs, generativeqc_backend backend) {
  std::string detail;
  generativeqc_status status{};
  const auto primal = generativeqc::generated::method_parameters::pbeD4();
  D4Parameters parameters{
      D4ReferenceModel::eeq, primal.s6,          primal.s8,         primal.s9, primal.a1, primal.a2,
      primal.cn_cutoff,      primal.pair_cutoff, primal.atm_cutoff, primal.ga, primal.gc};
  auto plan = D4Plan::prepare(backend, 0, inputs.offsets, inputs.numbers, inputs.charges,
                              inputs.coordinates, parameters, D4EEQProfile::standard, 512ULL << 20,
                              detail, status);
  require(status, detail);
  if (!plan) throw std::runtime_error("D4 prepare returned no plan");
  return plan;
}

void execute(D3Plan& plan, const Inputs& inputs, Result& result) {
  std::vector<D3Status> statuses;
  std::string detail;
  require(plan.execute(inputs.coordinates, inputs.active, inputs.gradient, statuses,
                       result.energies, result.gradients, detail),
          detail);
  if (!std::all_of(statuses.begin(), statuses.end(),
                   [](D3Status status) { return status == D3Status::success; }))
    throw std::runtime_error("D3 item failed");
}

void execute(D4Plan& plan, const Inputs& inputs, Result& result) {
  std::vector<D4Status> statuses;
  std::string detail;
  require(plan.execute(inputs.coordinates, inputs.active, inputs.gradient, statuses,
                       result.energies, result.gradients, result.charges, detail),
          detail);
  if (!std::all_of(statuses.begin(), statuses.end(),
                   [](D4Status status) { return status == D4Status::success; }))
    throw std::runtime_error("D4 item failed");
}

double maximum_error(const std::vector<double>& actual, const std::vector<double>& expected) {
  if (actual.size() != expected.size()) throw std::runtime_error("result extent mismatch");
  double error = 0.0;
  for (std::size_t index = 0; index < actual.size(); ++index) {
    if (!std::isfinite(actual[index])) throw std::runtime_error("nonfinite result");
    error = std::max(error, std::abs(actual[index] - expected[index]));
  }
  return error;
}

double checksum(const Result& result) {
  double sum = 0.0;
  for (std::size_t index = 0; index < result.energies.size(); ++index)
    sum += (index + 1) * result.energies[index];
  for (std::size_t index = 0; index < result.gradients.size(); ++index)
    sum += (index + 1) * result.gradients[index];
  return sum;
}

template <class Prepare>
void measure(const char* method, int atoms, int systems, generativeqc_backend backend,
             const char* revision, Prepare prepare) {
  Inputs inputs(atoms, systems);
  Result result, reference;
  auto reference_plan = prepare(inputs, GENERATIVEQC_BACKEND_CPU_REFERENCE);
  execute(*reference_plan, inputs, reference);
  auto plan = prepare(inputs, backend);
  execute(*plan, inputs, result);
  const double energy_error = maximum_error(result.energies, reference.energies);
  const double gradient_error = maximum_error(result.gradients, reference.gradients);
  if (energy_error > 3e-11 || gradient_error > 1e-9)
    throw std::runtime_error("CPU/device endpoint parity gate failed");

  // Each sample includes the native synchronous publication to host outputs.
  // Cold samples also include plan construction, table upload and destruction.
  for (const std::string mode : {"cold", "unchanged", "changed"}) {
    std::vector<double> samples;
    for (int sample = 0; sample < 7; ++sample) {
      const auto started = Clock::now();
      for (int repeat = 0; repeat < 3; ++repeat) {
        if (mode == "changed") inputs.coordinates[0] += 1e-5;
        if (mode == "cold") {
          auto cold_plan = prepare(inputs, backend);
          execute(*cold_plan, inputs, result);
        } else {
          execute(*plan, inputs, result);
        }
      }
      samples.push_back(std::chrono::duration<double>(Clock::now() - started).count() / 3);
    }
    auto ordered_samples = samples;
    std::sort(ordered_samples.begin(), ordered_samples.end());
    std::cout << std::setprecision(17) << "{\"revision\":\"" << revision << "\",\"method\":\""
              << method << "\",\"backend\":\""
              << (backend == GENERATIVEQC_BACKEND_CUDA ? "cuda" : "cpu") << "\",\"mode\":\"" << mode
              << "\",\"atoms_per_system\":" << atoms << ",\"systems\":" << systems
              << ",\"pair_candidates\":" << systems * atoms * (atoms - 1) / 2
              << ",\"triple_candidates\":"
              << (std::string(method) == "d3-zero" || std::string(method) == "gcp"
                      ? 0
                      : systems * atoms * (atoms - 1) * (atoms - 2) / 6)
              << ",\"eeq_dimension\":" << (std::string(method) == "d4-eeq" ? atoms + 1 : 0)
              << ",\"samples\":7,\"repeats\":3,\"median_seconds\":" << ordered_samples[3]
              << ",\"minimum_seconds\":" << ordered_samples.front()
              << ",\"maximum_seconds\":" << ordered_samples.back() << ",\"sample_seconds\":[";
    for (std::size_t sample = 0; sample < samples.size(); ++sample) {
      if (sample) std::cout << ',';
      std::cout << samples[sample];
    }
    std::cout << "],\"energy_error\":" << energy_error << ",\"gradient_error\":" << gradient_error
              << ",\"checksum\":" << checksum(result) << "}\n";
  }
}
}  // namespace

int main(int argc, char** argv) {
  try {
    const auto backend = argc > 1 && std::string(argv[1]) == "cuda"
                             ? GENERATIVEQC_BACKEND_CUDA
                             : GENERATIVEQC_BACKEND_CPU_REFERENCE;
    const char* revision = argc > 2 ? argv[2] : "unlabelled";
    for (const int atoms : {4, 24, 64}) {
      const int systems = atoms < 64 ? 8 : 2;
      measure("d3-bj-atm", atoms, systems, backend, revision,
              [](const Inputs& inputs, generativeqc_backend selected) {
                return prepare_d3(inputs, selected, false);
              });
      measure("d3-zero", atoms, systems, backend, revision,
              [](const Inputs& inputs, generativeqc_backend selected) {
                return prepare_d3(inputs, selected, true);
              });
      measure("d4-eeq", atoms, systems, backend, revision, prepare_d4);
      measure("gcp", atoms, systems, backend, revision, prepare_gcp);
    }
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
