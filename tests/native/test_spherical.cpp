#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string>

#include "generativeqc/generativeqc.h"
#include "integrals/s_integrals.hpp"
#include "molecule/basis.hpp"
#include "scf/mean_field.hpp"

namespace {

void require(bool condition, const char* message) {
  if (!condition) throw std::runtime_error(message);
}

generativeqc::core::System helium_hydrogen_sd(int charge = 1, std::uint32_t multiplicity = 1) {
  generativeqc::core::System system;
  system.atoms = {{2, {0.0, 0.0, -0.7}}, {1, {0.0, 0.0, 0.7}}};
  system.shells = {
      {0, 0, {{1.5, 1.0}}},
      {0, 2, {{0.8, 1.0}}},
      {1, 0, {{1.2, 1.0}}},
  };
  system.charge = charge;
  system.multiplicity = multiplicity;
  system.basis_representation = GENERATIVEQC_BASIS_SPHERICAL;
  std::string detail;
  require(
      generativeqc::molecule::validate_and_normalize(system, detail) == GENERATIVEQC_STATUS_SUCCESS,
      "spherical s/d system normalization failed");
  return system;
}

generativeqc::core::System single_f_shell() {
  generativeqc::core::System system;
  system.atoms = {{2, {0.0, 0.0, 0.0}}};
  system.shells = {{0, 3, {{0.6, 1.0}}}};
  system.basis_representation = GENERATIVEQC_BASIS_SPHERICAL;
  std::string detail;
  require(
      generativeqc::molecule::validate_and_normalize(system, detail) == GENERATIVEQC_STATUS_SUCCESS,
      "spherical f system normalization failed");
  return system;
}

#if GENERATIVEQC_HAS_CUDA
generativeqc::core::System helium_sf_atom() {
  generativeqc::core::System system;
  system.atoms = {{2, {0.0, 0.0, 0.0}}};
  system.shells = {
      {0, 0, {{1.5, 1.0}}},
      {0, 3, {{0.6, 1.0}}},
  };
  system.basis_representation = GENERATIVEQC_BASIS_SPHERICAL;
  std::string detail;
  require(
      generativeqc::molecule::validate_and_normalize(system, detail) == GENERATIVEQC_STATUS_SUCCESS,
      "spherical s/f atom normalization failed");
  return system;
}

generativeqc::core::System helium_hydrogen_sd_doublet() { return helium_hydrogen_sd(2, 2); }

bool cuda_device_available() {
  // CUDA-enabled login-node builds deliberately remain testable without
  // borrowing a scheduler-owned device. Allocated workers execute this block.
  generativeqc_context_descriptor descriptor{sizeof(generativeqc_context_descriptor),
                                             GENERATIVEQC_ABI_VERSION, 0,
                                             GENERATIVEQC_BACKEND_CUDA};
  generativeqc_context* context = nullptr;
  const generativeqc_status status = generativeqc_context_create(&descriptor, &context);
  if (context != nullptr) generativeqc_context_destroy(context);
  return status == GENERATIVEQC_STATUS_SUCCESS;
}

void require_cuda_matches_cpu(const generativeqc::scf::ScfResult& cuda,
                              const generativeqc::scf::ScfResult& cpu, const char* label) {
  require(cpu.converged, "CPU spherical oracle did not converge");
  require(cuda.converged, label);
  require(std::abs(cuda.energy - cpu.energy) < 3.0e-9,
          "CUDA spherical energy differs from the CPU oracle");
  require(cuda.forces.size() == cpu.forces.size(),
          "CUDA spherical force shape differs from the CPU oracle");
  for (std::size_t coordinate = 0; coordinate < cpu.forces.size(); ++coordinate) {
    require(std::abs(cuda.forces[coordinate] - cpu.forces[coordinate]) < 3.0e-8,
            "CUDA spherical analytic force differs from the CPU oracle");
  }
}
#endif

}  // namespace

int main() {
  try {
    const generativeqc::core::System sd = helium_hydrogen_sd();
    require(generativeqc::molecule::cartesian_ao_count(sd) == 8,
            "Cartesian source count for the s/d system is incorrect");
    require(generativeqc::molecule::ao_count(sd) == 7, "spherical s/d AO count is incorrect");

    generativeqc::scf::ScfOptions options;
    options.energy_tolerance = 1.0e-12;
    options.density_tolerance = 1.0e-10;
    const generativeqc::scf::ScfResult result = generativeqc::scf::run_rhf(sd, options);
    require(result.converged, "spherical s/d RHF did not converge");
    // Independent PySCF/libcint reference with cart=False and the exact same
    // one-primitive basis definitions.
    require(std::abs(result.energy - (-2.3341870407859284)) < 3.0e-12,
            "spherical s/d energy differs from PySCF");
    require(std::abs(result.forces[2] - 0.3502792384052442) < 8.0e-12 &&
                std::abs(result.forces[5] + 0.3502792384052438) < 8.0e-12,
            "spherical s/d analytic force differs from PySCF");

    const generativeqc::integrals::IntegralData f_integrals =
        generativeqc::integrals::build_integrals(single_f_shell());
    require(f_integrals.nbf == 7, "spherical f transform produced the wrong AO count");
    for (std::size_t row = 0; row < f_integrals.nbf; ++row) {
      for (std::size_t column = 0; column < f_integrals.nbf; ++column) {
        const double expected = row == column ? 1.0 : 0.0;
        require(std::abs(f_integrals.overlap[row * f_integrals.nbf + column] - expected) < 3.0e-13,
                "real-spherical f functions are not orthonormal");
      }
    }

#if GENERATIVEQC_HAS_CUDA
    if (cuda_device_available()) {
      const generativeqc::scf::ScfResult cuda_sd = generativeqc::scf::run_rhf_cuda(sd, options, 0);
      require_cuda_matches_cpu(cuda_sd, result, "CUDA spherical s/d RHF did not converge");

      // The one-center s/f case forces the CUDA integral consumers through all
      // seven real f harmonics without making the allocated-GPU smoke test a
      // large molecular benchmark.
      const generativeqc::core::System sf = helium_sf_atom();
      const generativeqc::scf::ScfResult cpu_sf = generativeqc::scf::run_rhf(sf, options);
      const generativeqc::scf::ScfResult cuda_sf = generativeqc::scf::run_rhf_cuda(sf, options, 0);
      require_cuda_matches_cpu(cuda_sf, cpu_sf, "CUDA spherical s/f RHF did not converge");

      const generativeqc::core::System sd_doublet = helium_hydrogen_sd_doublet();
      const generativeqc::scf::ScfResult cpu_uhf = generativeqc::scf::run_uhf(sd_doublet, options);
      const generativeqc::scf::ScfResult cuda_uhf =
          generativeqc::scf::run_uhf_cuda(sd_doublet, options, 0);
      require_cuda_matches_cpu(cuda_uhf, cpu_uhf, "CUDA spherical s/d UHF did not converge");
    } else {
      std::cout << "CUDA spherical checks skipped: no allocated CUDA device\n";
    }
#endif

    std::cout << "validated real-spherical d/f transforms and RHF gradient\n";
    return EXIT_SUCCESS;
  } catch (const std::exception& error) {
    std::cerr << "test failure: " << error.what() << '\n';
    return EXIT_FAILURE;
  }
}
