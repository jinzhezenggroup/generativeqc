#include <cuda_runtime_api.h>
#include <dlfcn.h>

#include <chrono>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "scf/cuda/rhf_bucket_internal.hpp"
#include "scf/mean_field.hpp"

extern "C" bool tensor_structured_scf_qualification_for_test_v1(int);

namespace {
using Clock = std::chrono::steady_clock;

void require(bool condition, const char* message) {
  if (!condition) throw std::runtime_error(message);
}

void array(const std::vector<double>& values) {
  std::cout << '[';
  for (std::size_t index = 0; index < values.size(); ++index) {
    if (index) std::cout << ',';
    if (std::isfinite(values[index]))
      std::cout << values[index];
    else
      std::cout << "null";
  }
  std::cout << ']';
}

/** Detach only after clean endpoint timing. These copies are diagnostic work,
 * not production transfers or part of the measured complete solve. */
std::vector<double> detach(const generativeqc::scf::CudaRhfBucketPlan& plan, std::size_t offset,
                           std::size_t count) {
  std::vector<double> output(count);
  const auto* source = static_cast<const char*>(plan.resources.arena_) + offset;
  require(cudaMemcpy(output.data(), source, count * sizeof(double), cudaMemcpyDeviceToHost) ==
              cudaSuccess,
          "final matrix diagnostic download failed");
  return output;
}
}  // namespace

int main(int argc, char** argv) try {
  require(argc == 4, "usage: structured-endpoint FIXTURE REGION_MASK BATCH_SIZE");
  require(std::getenv("SLURM_JOB_ID") && std::getenv("CUDA_VISIBLE_DEVICES"),
          "real-device endpoint qualification requires Slurm visibility");
  const int mask = std::stoi(argv[2]);
  const unsigned batch_size = std::stoul(argv[3]);
  require(batch_size > 0 && batch_size <= 3, "bounded probe supports one to three systems");
  require(tensor_structured_scf_qualification_for_test_v1(mask), "invalid diagnostic mask");
  std::ifstream input(argv[1]);
  generativeqc::core::System system;
  unsigned atoms{}, shells{}, spherical{}, unrestricted{};
  input >> atoms >> shells >> spherical >> unrestricted >> system.charge >> system.multiplicity;
  system.basis_representation =
      spherical ? GENERATIVEQC_BASIS_SPHERICAL : GENERATIVEQC_BASIS_CARTESIAN;
  for (unsigned index = 0; index < atoms; ++index) {
    generativeqc::core::Atom atom;
    input >> atom.atomic_number >> atom.position[0] >> atom.position[1] >> atom.position[2];
    system.atoms.push_back(atom);
  }
  for (unsigned index = 0; index < shells; ++index) {
    generativeqc::core::Shell shell;
    unsigned primitives{};
    input >> shell.atom_index >> shell.angular_momentum >> primitives;
    for (unsigned primitive = 0; primitive < primitives; ++primitive) {
      generativeqc::core::Primitive value;
      input >> value.exponent >> value.coefficient;
      shell.primitives.push_back(value);
    }
    system.shells.push_back(shell);
  }
  require(bool(input), "malformed qualification fixture");
  system.electron_count = -system.charge;
  for (const auto& atom : system.atoms) system.electron_count += atom.atomic_number;
  std::vector<generativeqc::core::System> systems(batch_size, system);
  generativeqc::scf::ScfOptions options;
  options.energy_tolerance = 1e-11;
  options.density_tolerance = 1e-9;
  options.screening_tolerance = 1e-14;
  options.max_iterations = 200;
  options.precision_mode = GENERATIVEQC_PRECISION_FP64;
  generativeqc::scf::CudaRhfBucketPlan* plan = nullptr;
  struct Owner {
    generativeqc::scf::CudaRhfBucketPlan*& plan;
    ~Owner() { generativeqc::scf::destroy_rhf_cuda_bucket_plan(plan); }
  } owner{plan};
  std::vector<std::vector<double>> warm(batch_size);
  Dl_info library_info{};
  require(dladdr(reinterpret_cast<void*>(&generativeqc_get_source_identity), &library_info) != 0,
          "cannot identify the loaded scientific library");
  std::cout << std::setprecision(17) << "{\"mask\":" << mask
            << ",\"source_identity\":" << std::quoted(generativeqc_get_source_identity())
            << ",\"library_path\":" << std::quoted(library_info.dli_fname) << ",\"rows\":[";
  for (unsigned phase = 0; phase < 5; ++phase) {
    if (phase) std::cout << ',';
    if (phase == 3)
      for (auto& item : systems) item.atoms.back().position[0] += 0.012;
    std::vector<const std::vector<double>*> seeds(batch_size, nullptr);
    if (phase)
      for (unsigned item = 0; item < batch_size; ++item)
        if (!warm[item].empty()) seeds[item] = &warm[item];
    const auto start = Clock::now();
    const auto result =
        unrestricted
            ? generativeqc::scf::run_uhf_cuda_bucket_cached(&plan, systems, options, seeds, 0)
            : generativeqc::scf::run_rhf_cuda_bucket_cached(&plan, systems, options, seeds, 0);
    const double seconds = std::chrono::duration<double>(Clock::now() - start).count();
    require(plan && result.size() == batch_size, "endpoint did not retain its prepared owner");
    const auto nbf = plan->nbf;
    const auto spins = unrestricted ? 2U : 1U;
    const auto fock = detach(*plan, plan->layout.fock, batch_size * spins * nbf * nbf);
    const auto overlap = detach(*plan, plan->layout.overlap, batch_size * nbf * nbf);
    const auto orthogonalizer = detach(*plan, plan->layout.orthogonalizer, batch_size * nbf * nbf);
    const auto coefficients =
        detach(*plan, plan->layout.coefficients, batch_size * spins * nbf * nbf);
    const auto eigenvalues = detach(*plan, plan->layout.eigenvalues, batch_size * spins * nbf);
    std::cout << "{\"phase\":" << phase << ",\"complete_seconds\":" << seconds << ",\"nbf\":" << nbf
              << ",\"direct_nbf\":" << plan->direct_nbf << ",\"arena_bytes\":" << plan->layout.bytes
              << ",\"provider_retained_bytes\":" << plan->resources.provider_retained_bytes_
              << ",\"graph_generation\":" << plan->graphs.iteration_generation() << ",\"fock\":";
    array(fock);
    std::cout << ",\"overlap\":";
    array(overlap);
    std::cout << ",\"orthogonalizer\":";
    array(orthogonalizer);
    std::cout << ",\"coefficients\":";
    array(coefficients);
    std::cout << ",\"eigenvalues\":";
    array(eigenvalues);
    std::cout << ",\"items\":[";
    for (unsigned item = 0; item < batch_size; ++item) {
      if (item) std::cout << ',';
      const auto& value = result[item];
      std::cout << "{\"status\":" << value.status
                << ",\"converged\":" << (value.scf.converged ? "true" : "false") << ",\"energy\":";
      if (std::isfinite(value.scf.energy))
        std::cout << value.scf.energy;
      else
        std::cout << "null";
      std::cout << ",\"iterations\":" << value.scf.iterations << ",\"density\":";
      array(value.scf.density);
      std::cout << ",\"forces\":";
      array(value.scf.forces);
      std::cout << ",\"execution_plan_reused\":" << (value.execution_plan_reused ? "true" : "false")
                << '}';
      if (value.status == GENERATIVEQC_STATUS_SUCCESS) warm[item] = value.scf.density;
    }
    std::cout << "],\"bindings\":[";
    for (unsigned slot = 0; slot < 7; ++slot) {
      if (slot) std::cout << ',';
      const auto operation = static_cast<generativeqc::tensor::StructuredOperation>(slot);
      const auto& binding = *plan->resources.structured_binding_;
      const auto& selected = binding.site(operation);
      std::cout << "{\"operation\":" << slot
                << ",\"provider\":" << std::quoted(std::string(selected.provider))
                << ",\"offers\":[";
      const auto offers = binding.candidates(operation);
      for (unsigned offer = 0; offer < offers.size(); ++offer) {
        if (offer) std::cout << ',';
        std::cout << "{\"provider\":" << std::quoted(std::string(offers[offer].provider))
                  << ",\"rejection\":" << std::quoted(std::string(offers[offer].rejection))
                  << ",\"temporary_bytes\":" << offers[offer].temporary_bytes
                  << ",\"host_bytes\":" << offers[offer].host_bytes
                  << ",\"provider_bytes\":" << offers[offer].provider_bytes << '}';
      }
      std::cout << "]}";
    }
    std::cout << "]}" << std::flush;
  }
  std::cout << "]}\n";
  require(tensor_structured_scf_qualification_for_test_v1(-1), "hook reset failed");
  return 0;
} catch (const std::exception& error) {
  std::cerr << error.what() << '\n';
  return 1;
}
