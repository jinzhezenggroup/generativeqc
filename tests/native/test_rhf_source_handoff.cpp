// Detached source lifetime and values against the independent CPU evaluator.
#include <cuda_runtime_api.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "integrals/electron_interaction_source.hpp"
#include "molecule/basis.hpp"
#include "posthf/raw_source.hpp"
#include "scf/mean_field.hpp"
#include "scf/rhf_source_handoff.hpp"

namespace {
void require(bool value, const char* message) {
  if (!value) throw std::runtime_error(message);
}
void check(cudaError_t error) { require(error == cudaSuccess, cudaGetErrorString(error)); }

generativeqc::core::System helium(unsigned angular, bool spherical) {
  generativeqc::core::System system;
  system.basis_representation =
      spherical ? GENERATIVEQC_BASIS_SPHERICAL : GENERATIVEQC_BASIS_CARTESIAN;
  system.atoms.push_back({2, {0.0, 0.0, 0.0}});
  system.shells.push_back({0, 0, {{1.3, 1.0}}});
  system.shells.push_back({0, 0, {{0.35, 1.0}}});
  if (angular) system.shells.push_back({0, angular, {{0.65, 1.0}}});
  std::string detail;
  require(
      generativeqc::molecule::validate_and_normalize(system, detail) == GENERATIVEQC_STATUS_SUCCESS,
      detail.c_str());
  return system;
}

void verify(const generativeqc::integrals::ElectronInteractionSource& source,
            const generativeqc::core::System& system, bool full) {
  using Op = generativeqc::integrals::ElectronInteractionOperator;
  const auto n = source.nbf(), width = full ? n : std::min<std::size_t>(n, 2);
  const std::array<std::size_t, 4> begin =
      full ? std::array<std::size_t, 4>{} : std::array{n - width, 0UL, n - width, n - width};
  const std::array count{width, width, width, width};
  const auto size = width * width * width * width;
  std::vector<double> expected(size), actual(size);
  generativeqc::posthf::RawSource oracle(system);
  oracle.read(Op::eri, begin, count, expected.data(), size);
  cudaStream_t stream{};
  double* device{};
  check(cudaStreamCreate(&stream));
  check(cudaMalloc(reinterpret_cast<void**>(&device), size * sizeof(double)));
  try {
    source.read_device(Op::eri, begin, count, {0, stream, device, size}, size);
    check(cudaMemcpyAsync(actual.data(), device, size * sizeof(double), cudaMemcpyDeviceToHost,
                          stream));
    check(cudaStreamSynchronize(stream));
    for (std::size_t i = 0; i < size; ++i)
      require(std::isfinite(actual[i]) && std::abs(actual[i] - expected[i]) < 3e-11,
              "detached source differs from independent public-AO ERI");
    bool rejected = false;
    try {
      source.read_device(Op::eri, begin, count, {1, stream, device, size}, size);
    } catch (const std::invalid_argument&) {
      rejected = true;
    }
    require(rejected, "wrong-device source use was accepted");
  } catch (...) {
    cudaStreamSynchronize(stream);
    cudaFree(device);
    cudaStreamDestroy(stream);
    throw;
  }
  check(cudaFree(device));
  check(cudaStreamDestroy(stream));
}
}  // namespace

int main() {
  try {
    using namespace generativeqc;
    for (unsigned angular : {0U, 1U, 2U, 3U}) {
      for (bool spherical : {false, true}) {
        auto system = helium(angular, spherical);
        scf::ScfOptions options;
        options.screening_tolerance = 0.0;
        options.compute_forces = false;
        options.precision_mode = GENERATIVEQC_PRECISION_FP64;
        options.export_physical_reference = true;
        options.reference_memory_budget_bytes = 1ULL << 30;
        options.energy_tolerance = 1e-12;
        options.density_tolerance = 1e-11;
        options.max_iterations = 150;
        scf::CudaRhfSourceHandoff handoff;
        auto result = scf::run_rhf_cuda_with_source(system, options, 0, nullptr, handoff);
        require(result.converged && result.reference && handoff.source,
                "RHF did not publish a detached exact source");
        require(handoff.device_copy_bytes > 0 && !handoff.resource_fallback &&
                    handoff.numeric_peak_bytes <= options.reference_memory_budget_bytes &&
                    handoff.source->retained_numeric_bytes() == handoff.retained_numeric_bytes,
                "source handoff accounting is inconsistent");
        // The RHF arena has already died. A later RHF solve also cannot mutate
        // the retained source's geometry or its ordered public-AO metadata.
        auto moved = system;
        moved.atoms[0].position[2] += 0.125;
        scf::CudaRhfSourceHandoff other;
        auto second = scf::run_rhf_cuda_with_source(moved, options, 0, nullptr, other);
        require(second.converged && other.source, "second geometry failed");
        require(handoff.source->orbital().atoms[0].position[2] == 0.0,
                "source snapshot followed a later geometry");
        verify(*handoff.source, system, angular == 0);
        verify(*other.source, moved, false);
        const auto exact_budget = handoff.required_peak_bytes;
        handoff.source.reset();
        other.source.reset();
        if (angular == 0) {
          options.reference_memory_budget_bytes = exact_budget;
          auto admitted = scf::run_rhf_cuda_with_source(system, options, 0, nullptr, handoff);
          require(admitted.converged && handoff.source, "exact handoff budget rejected");
          options.reference_memory_budget_bytes = exact_budget - 1;
          auto fallback = scf::run_rhf_cuda_with_source(system, options, 0, nullptr, handoff);
          require(fallback.converged && !handoff.source && handoff.resource_fallback &&
                      handoff.device_copy_bytes == 0,
                  "short handoff budget did not preserve the RHF reference");
        }
        std::cout << "l=" << angular << " spherical=" << spherical
                  << " compact handoff and independent ERI passed\n";
      }
    }
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
