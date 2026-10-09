#include <array>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "generativeqc/ks.hpp"

namespace {

void require(bool condition, const char* message) {
  if (!condition) throw std::runtime_error(message);
}

}  // namespace

int main() {
  try {
    generativeqc_context_descriptor context_descriptor{sizeof(generativeqc_context_descriptor),
                                                       GENERATIVEQC_ABI_VERSION, 0,
                                                       GENERATIVEQC_BACKEND_CPU_REFERENCE};
    generativeqc::Context context(context_descriptor);

    const std::array<generativeqc_primitive, 6> h_primitives{{
        {3.42525091, 0.15432897},
        {0.62391373, 0.53532814},
        {0.16885540, 0.44463454},
        {3.42525091, 0.15432897},
        {0.62391373, 0.53532814},
        {0.16885540, 0.44463454},
    }};
    const std::array<generativeqc_atom, 2> h_atoms{{
        {1, 0.0, 0.0, -0.7},
        {1, 0.0, 0.0, 0.7},
    }};
    const std::array<generativeqc_shell, 2> h_shells{{{0, 0, 0, 3}, {1, 0, 3, 3}}};
    generativeqc_system_descriptor h2_descriptor{sizeof(generativeqc_system_descriptor),
                                                 GENERATIVEQC_ABI_VERSION,
                                                 h_atoms.data(),
                                                 h_atoms.size(),
                                                 h_shells.data(),
                                                 h_shells.size(),
                                                 h_primitives.data(),
                                                 h_primitives.size(),
                                                 0,
                                                 1};
    generativeqc::System h2(context, h2_descriptor);

    const std::array<generativeqc_atom, 1> he_atoms{{{2, 0.0, 0.0, 0.0}}};
    const std::array<generativeqc_primitive, 3> he_primitives{{
        {6.36242139, 0.15432897},
        {1.15892300, 0.53532814},
        {0.31364979, 0.44463454},
    }};
    const std::array<generativeqc_shell, 1> he_shells{{{0, 0, 0, 3}}};
    generativeqc_system_descriptor he_descriptor{sizeof(generativeqc_system_descriptor),
                                                 GENERATIVEQC_ABI_VERSION,
                                                 he_atoms.data(),
                                                 he_atoms.size(),
                                                 he_shells.data(),
                                                 he_shells.size(),
                                                 he_primitives.data(),
                                                 he_primitives.size(),
                                                 0,
                                                 1};
    generativeqc::System helium(context, he_descriptor);

    generativeqc_method_descriptor method{sizeof(generativeqc_method_descriptor),
                                          GENERATIVEQC_ABI_VERSION,
                                          GENERATIVEQC_METHOD_RHF,
                                          100,
                                          8,
                                          1.0e-12,
                                          1.0e-10,
                                          1.0e-14};
    const std::array<const generativeqc::System*, 2> systems{{&h2, &helium}};
    generativeqc::Batch batch(context, systems, method);
    const auto cold = batch.execute();
    const auto warm = batch.execute();
    require(cold.size() == 2 && warm.size() == 2, "C++ batch result count is incorrect");
    require(std::abs(cold[0].energy - (-1.11671432506255)) < 2.0e-9,
            "C++ batch H2 energy is incorrect");
    require(std::abs(cold[1].energy - (-2.807783957539976)) < 2.0e-10,
            "C++ batch helium energy is incorrect");
    require(!cold[0].warm_start_used && warm[0].warm_start_used,
            "C++ batch did not expose warm-start state");
    require(cold[0].forces.size() == 6 && cold[1].forces.size() == 3,
            "C++ batch introduced padded force storage");
    try {
      (void)batch.last_density_fitting_metric_diagnostics();
      throw std::runtime_error("CPU C++ batch unexpectedly published CUDA DF diagnostics");
    } catch (const generativeqc::Error& error) {
      require(error.status() == GENERATIVEQC_STATUS_NOT_IMPLEMENTED,
              "C++ DF diagnostic getter returned the wrong status");
    }
    const auto hf = generativeqc::Calculation(context, h2, method).execute();
    require(!hf.physical_residual_rms && hf.density_rms == hf.reference_residual,
            "C++ HF invented a physical residual or changed its legacy alias");
    auto open_shell_descriptor = h2_descriptor;
    open_shell_descriptor.charge = -1;
    open_shell_descriptor.multiplicity = 2;
    generativeqc::System open_shell(context, open_shell_descriptor);
    for (const auto ks_method : {GENERATIVEQC_METHOD_LDA_UKS, GENERATIVEQC_METHOD_PBE_UKS}) {
      method.method = ks_method;
      const auto ks = generativeqc::Calculation(context, open_shell, method).execute();
      require(ks.physical_residual_rms && std::isfinite(*ks.physical_residual_rms) &&
                  *ks.physical_residual_rms < method.density_tolerance &&
                  ks.density_rms < method.density_tolerance &&
                  ks.density_rms == ks.reference_residual &&
                  ks.density_rms != *ks.physical_residual_rms,
              "C++ UKS conflated density update and physical residual");
    }
    // The native name resolver is backed by the same generated ABI manifest
    // as the CLI. Compiler-only names are not implicitly transformed.
    require(generativeqc::resolve_method("pbe-rks") == GENERATIVEQC_METHOD_PBE_RKS,
            "C++ native method name resolution failed");
    const char* canonical_name = nullptr;
    require(generativeqc_method_get_name(GENERATIVEQC_METHOD_PBE_RKS, &canonical_name) ==
                    GENERATIVEQC_STATUS_SUCCESS &&
                std::string(canonical_name) == "pbe-rks",
            "native name round trip failed");
    generativeqc_method unchanged = GENERATIVEQC_METHOD_RHF;
    require(generativeqc_method_from_name("pbe0-rks", &unchanged) ==
                    GENERATIVEQC_STATUS_INVALID_ARGUMENT &&
                unchanged == GENERATIVEQC_METHOD_RHF,
            "compiler-only method was incorrectly admitted as a native ABI ID");
    require(generativeqc_method_get_name(999999, &canonical_name) ==
                GENERATIVEQC_STATUS_INVALID_ARGUMENT,
            "unknown native method ID was accepted");

    // No Python compiler/runtime is involved: this is the same PBE0 physical
    // graph as the native semantic-composition C ABI qualification.
    auto pbe0_descriptor = generativeqc::default_method_descriptor("pbe-rks");
    pbe0_descriptor.max_iterations = 200;
    pbe0_descriptor.energy_tolerance = 1.0e-12;
    pbe0_descriptor.density_tolerance = 1.0e-10;
    // The builder is destroyed as soon as native preparation succeeds.
    // The returned Calculation must own the full KS snapshot independently.
    auto prepared = [&]() {
      generativeqc::KsComposition pbe0(
          GENERATIVEQC_METHOD_PBE_RKS, "semilocal-scaled-v1/pbe-spin-c2-1e-18", 1);
      pbe0.set_grid({1, 64, 12, 24, 3, 1.0e-12, 256})
          .add_semilocal("GGA_C_PBE", 1.0)
          .add_semilocal("GGA_X_PBE", 0.75)
          .add_exact_exchange(GENERATIVEQC_KS_EXCHANGE_FULL_RANGE, 0.25);
      return pbe0.prepare(context, h2, pbe0_descriptor);
    }();
    const auto pbe0_energy = prepared.execute();
    require(std::isfinite(pbe0_energy.energy) &&
                std::abs(pbe0_energy.energy - (-1.1543107969377155)) < 1.0e-6,
            "Python-free C++ PBE0 energy differs from qualified native reference");
    require(pbe0_energy.physical_residual_rms.has_value(),
            "C++ PBE0 SCF lost physical commutator diagnostic");
    try {
      (void)prepared.execute(GENERATIVEQC_PROPERTY_ENERGY | GENERATIVEQC_PROPERTY_FORCES);
      throw std::runtime_error("C++ SDK incorrectly advertised DFT forces");
    } catch (const generativeqc::Error& error) {
      require(error.status() == GENERATIVEQC_STATUS_NOT_IMPLEMENTED,
              "C++ DFT force request did not fail closed");
    }
    std::cout << "C++ ragged batch and explicit KS API: PASS\n";
    return EXIT_SUCCESS;
  } catch (const std::exception& error) {
    std::cerr << "test failure: " << error.what() << '\n';
    return EXIT_FAILURE;
  }
}
