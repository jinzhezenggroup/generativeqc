#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string>
#include <thread>

#include "generativeqc/fock.h"
#include "generativeqc/generativeqc.h"

namespace {

struct Evaluation {
  double energy{};
  std::array<double, 6> forces{};
};

void require(bool condition, const char* message) {
  if (!condition) throw std::runtime_error(message);
}

void verify_context_detail_storage() {
  const generativeqc_context_descriptor context_descriptor{sizeof(generativeqc_context_descriptor),
                                                           GENERATIVEQC_ABI_VERSION, 0,
                                                           GENERATIVEQC_BACKEND_CPU_REFERENCE};
  generativeqc_context* first = nullptr;
  generativeqc_context* second = nullptr;
  require(generativeqc_context_create(&context_descriptor, &first) == GENERATIVEQC_STATUS_SUCCESS,
          "first detail context creation failed");
  require(generativeqc_context_create(&context_descriptor, &second) == GENERATIVEQC_STATUS_SUCCESS,
          "second detail context creation failed");

  const generativeqc_atom atom{1, 0.0, 0.0, 0.0};
  generativeqc_primitive primitive{1.0, 1.0};
  generativeqc_shell shell{0, 0, 0, 1};
  generativeqc_system_descriptor descriptor{sizeof(generativeqc_system_descriptor),
                                            GENERATIVEQC_ABI_VERSION,
                                            &atom,
                                            1,
                                            &shell,
                                            1,
                                            &primitive,
                                            1,
                                            0,
                                            1};
  generativeqc_system* system = nullptr;
  primitive.exponent = -1.0;
  require(generativeqc_system_create(first, &descriptor, &system) ==
              GENERATIVEQC_STATUS_INVALID_ARGUMENT,
          "invalid primitive did not fail");
  const char* first_detail = generativeqc_context_get_last_detail(first);
  require(first_detail != nullptr &&
              std::string(first_detail).find("positive finite") != std::string::npos,
          "first context detail was not recorded");

  const char* worker_detail = nullptr;
  std::thread worker([&] { worker_detail = generativeqc_context_get_last_detail(first); });
  worker.join();
  require(worker_detail != nullptr &&
              std::string(worker_detail).find("positive finite") != std::string::npos,
          "context detail pointer did not survive worker thread exit");

  primitive.exponent = 1.0;
  shell.angular_momentum = 5;
  require(generativeqc_system_create(second, &descriptor, &system) ==
              GENERATIVEQC_STATUS_NOT_IMPLEMENTED,
          "unsupported angular momentum did not fail");
  const char* second_detail = generativeqc_context_get_last_detail(second);
  require(second_detail != nullptr &&
              std::string(second_detail).find("supports s through g") != std::string::npos,
          "second context detail was not recorded");
  require(std::string(first_detail).find("positive finite") != std::string::npos,
          "first context detail was overwritten by another context query");

  // Successful calls must preserve the last failure, including when a caller
  // queries it again after an operation that uses scratch diagnostics.
  shell.angular_momentum = 0;
  descriptor.multiplicity = 2;
  require(generativeqc_system_create(first, &descriptor, &system) == GENERATIVEQC_STATUS_SUCCESS,
          "valid system did not recover after failure");
  const generativeqc_fock_spec spec{
      sizeof(spec),
      GENERATIVEQC_ABI_VERSION,
      1,
      GENERATIVEQC_FOCK_RESTRICTED,
      0,
      {1, 1.0, GENERATIVEQC_FOCK_FULL_RANGE, 0.0, GENERATIVEQC_FOCK_EXACT},
      {1, -0.5, GENERATIVEQC_FOCK_FULL_RANGE, 0.0, GENERATIVEQC_FOCK_EXACT}};
  generativeqc_fock_plan* plan = nullptr;
  require(generativeqc_fock_plan_create(first, system, nullptr, &spec, nullptr, &plan) ==
              GENERATIVEQC_STATUS_SUCCESS,
          "valid Fock plan did not preserve successful recovery");
  const char* after_success = generativeqc_context_get_last_detail(first);
  require(after_success == first_detail &&
              std::string(first_detail).find("positive finite") != std::string::npos,
          "successful API call or later getter invalidated the last failure detail");
  generativeqc_fock_plan_destroy(plan);
  generativeqc_system_destroy(system);

  generativeqc_context_destroy(first);
  generativeqc_context_destroy(second);
}

// Exercise output selection on one retained C ABI calculation, including
// energy-only as its first execution and forces after an energy-only replay.
Evaluation h2(double distance, bool verify_energy_only = false,
              generativeqc_backend backend = GENERATIVEQC_BACKEND_CPU_REFERENCE,
              generativeqc_density_fitting_mode df_mode = GENERATIVEQC_DENSITY_FITTING_NONE,
              bool unrestricted = false, std::uint64_t df_budget = 0) {
  generativeqc_context_descriptor context_descriptor{sizeof(generativeqc_context_descriptor),
                                                     GENERATIVEQC_ABI_VERSION, 0, backend};
  generativeqc_context* context = nullptr;
  require(generativeqc_context_create(&context_descriptor, &context) == GENERATIVEQC_STATUS_SUCCESS,
          "context creation failed");

  const std::array<generativeqc_atom, 2> atoms{{
      {1, 0.0, 0.0, -0.5 * distance},
      {1, 0.0, 0.0, 0.5 * distance},
  }};
  const std::array<generativeqc_primitive, 6> primitives{{
      {3.42525091, 0.15432897},
      {0.62391373, 0.53532814},
      {0.16885540, 0.44463454},
      {3.42525091, 0.15432897},
      {0.62391373, 0.53532814},
      {0.16885540, 0.44463454},
  }};
  const std::array<generativeqc_shell, 2> shells{{
      {0, 0, 0, 3},
      {1, 0, 3, 3},
  }};
  generativeqc_system_descriptor system_descriptor{sizeof(generativeqc_system_descriptor),
                                                   GENERATIVEQC_ABI_VERSION,
                                                   atoms.data(),
                                                   static_cast<uint32_t>(atoms.size()),
                                                   shells.data(),
                                                   static_cast<uint32_t>(shells.size()),
                                                   primitives.data(),
                                                   static_cast<uint32_t>(primitives.size()),
                                                   0,
                                                   1};
  system_descriptor.charge = unrestricted ? 1 : 0;
  system_descriptor.multiplicity = unrestricted ? 2 : 1;
  generativeqc_system* system = nullptr;
  const generativeqc_status system_status =
      generativeqc_system_create(context, &system_descriptor, &system);
  require(system_status == GENERATIVEQC_STATUS_SUCCESS, "system creation failed");

  generativeqc_method_descriptor method{sizeof(generativeqc_method_descriptor),
                                        GENERATIVEQC_ABI_VERSION,
                                        GENERATIVEQC_METHOD_RHF,
                                        100,
                                        8,
                                        1.0e-12,
                                        1.0e-10,
                                        1.0e-14};
  method.method = unrestricted ? GENERATIVEQC_METHOD_UHF : GENERATIVEQC_METHOD_RHF;
  method.density_fitting_mode = df_mode;
  method.density_fitting_memory_budget_bytes = df_budget;
  generativeqc_calculation* calculation = nullptr;
  require(generativeqc_calculation_prepare(context, system, &method, &calculation) ==
              GENERATIVEQC_STATUS_SUCCESS,
          "calculation preparation failed");

  Evaluation evaluation;
  generativeqc_result_descriptor result{sizeof(generativeqc_result_descriptor),
                                        GENERATIVEQC_ABI_VERSION,
                                        0.0,
                                        evaluation.forces.data(),
                                        static_cast<uint32_t>(evaluation.forces.size()),
                                        0,
                                        0.0,
                                        0.0,
                                        0,
                                        GENERATIVEQC_BACKEND_CPU_REFERENCE};
  generativeqc_result_descriptor first_energy_only{sizeof(generativeqc_result_descriptor),
                                                   GENERATIVEQC_ABI_VERSION,
                                                   0.0,
                                                   nullptr,
                                                   0,
                                                   0,
                                                   0.0,
                                                   0.0,
                                                   0,
                                                   backend};
  if (verify_energy_only) {
    require(generativeqc_calculation_execute(calculation, &first_energy_only) ==
                GENERATIVEQC_STATUS_SUCCESS,
            "first energy-only execution failed");
    require(first_energy_only.executed_backend == backend,
            "energy-only execution used an unexpected backend");
  }
  const generativeqc_status status = generativeqc_calculation_execute(calculation, &result);
  require(status == GENERATIVEQC_STATUS_SUCCESS, "RHF execution failed");
  require(result.converged == 1, "RHF did not report convergence");
  evaluation.energy = result.energy;

  if (verify_energy_only) {
    generativeqc_result_descriptor energy_only{sizeof(generativeqc_result_descriptor),
                                               GENERATIVEQC_ABI_VERSION,
                                               0.0,
                                               nullptr,
                                               0,
                                               0,
                                               0.0,
                                               0.0,
                                               0,
                                               GENERATIVEQC_BACKEND_CPU_REFERENCE};
    require(
        generativeqc_calculation_execute(calculation, &energy_only) == GENERATIVEQC_STATUS_SUCCESS,
        "energy-only execution failed");
    const double tolerance = backend == GENERATIVEQC_BACKEND_CUDA ? 2.0e-9 : 1.0e-14;
    require(std::abs(energy_only.energy - evaluation.energy) < tolerance &&
                std::abs(first_energy_only.energy - evaluation.energy) < tolerance,
            "omitting force storage changed the energy");
    const auto expected_forces = evaluation.forces;
    evaluation.forces.fill(NAN);
    require(generativeqc_calculation_execute(calculation, &result) == GENERATIVEQC_STATUS_SUCCESS,
            "force execution after energy-only replay failed");
    for (std::size_t i = 0; i < expected_forces.size(); ++i) {
      require(std::isfinite(evaluation.forces[i]) &&
                  std::abs(evaluation.forces[i] - expected_forces[i]) < tolerance,
              "energy-only replay changed or suppressed later forces");
    }
  }

  generativeqc_calculation_destroy(calculation);
  generativeqc_system_destroy(system);
  generativeqc_context_destroy(context);
  return evaluation;
}

/** A single-atom RHF preparation with one s-shell of \p primitive_count primitives. */
struct PreparedSingleAtom {
  generativeqc_context* context;
  generativeqc_system* system;
  generativeqc_calculation* calculation;
};

PreparedSingleAtom prepare_single_atom_rhf(int atomic_number, std::size_t primitive_count) {
  generativeqc_context_descriptor context_descriptor{sizeof(generativeqc_context_descriptor),
                                                     GENERATIVEQC_ABI_VERSION, 0,
                                                     GENERATIVEQC_BACKEND_CPU_REFERENCE};
  generativeqc_context* context = nullptr;
  require(generativeqc_context_create(&context_descriptor, &context) == GENERATIVEQC_STATUS_SUCCESS,
          "single-atom context creation failed");

  const std::array<generativeqc_atom, 1> atoms{{{atomic_number, 0.0, 0.0, 0.0}}};
  const generativeqc_primitive primitives[4] = {{1.0, 1.0}, {1.0, 1.0}, {1.0, 1.0}, {1.0, 1.0}};
  const std::array<generativeqc_shell, 1> shells{
      {{0, 0, 0, static_cast<uint32_t>(primitive_count)}}};
  generativeqc_system_descriptor system_descriptor{sizeof(generativeqc_system_descriptor),
                                                   GENERATIVEQC_ABI_VERSION,
                                                   atoms.data(),
                                                   static_cast<uint32_t>(atoms.size()),
                                                   shells.data(),
                                                   static_cast<uint32_t>(shells.size()),
                                                   primitives,
                                                   static_cast<uint32_t>(primitive_count),
                                                   0,
                                                   1};
  generativeqc_system* system = nullptr;
  require(generativeqc_system_create(context, &system_descriptor, &system) ==
              GENERATIVEQC_STATUS_SUCCESS,
          "single-atom system creation failed");

  generativeqc_method_descriptor method{sizeof(generativeqc_method_descriptor),
                                        GENERATIVEQC_ABI_VERSION,
                                        GENERATIVEQC_METHOD_RHF,
                                        100,
                                        8,
                                        1.0e-12,
                                        1.0e-10,
                                        1.0e-14};
  method.density_fitting_mode = GENERATIVEQC_DENSITY_FITTING_NONE;
  method.density_fitting_memory_budget_bytes = 0;
  method.precision_mode = GENERATIVEQC_PRECISION_FP64;
  generativeqc_calculation* calculation = nullptr;
  require(generativeqc_calculation_prepare(context, system, &method, &calculation) ==
              GENERATIVEQC_STATUS_SUCCESS,
          "single-atom calculation preparation failed");

  return {context, system, calculation};
}

/**
 * The provenance getter must report availability honestly. Both the availability
 * query (a NULL \p out) and the copy-out are gated on whether a completed
 * execution has populated the record: UNAVAILABLE before a run and after one that
 * threw, SUCCESS after a normal return. This pins the \p precision_available gate
 * so a stale record can never be serialized from a failed or not-yet-run run.
 */
void verify_precision_provenance_gate() {
  // He: Z=2, two s primitives -> two AOs, one occupied pair, converges.
  {
    const PreparedSingleAtom he = prepare_single_atom_rhf(2, 2);
    generativeqc_precision_provenance prov{sizeof(generativeqc_precision_provenance),
                                           GENERATIVEQC_ABI_VERSION};
    require(generativeqc_calculation_get_precision_provenance(he.calculation, &prov) ==
                GENERATIVEQC_STATUS_PRECISION_UNAVAILABLE,
            "provenance must be unavailable before any execution");
    require(generativeqc_calculation_get_precision_provenance(he.calculation, nullptr) ==
                GENERATIVEQC_STATUS_PRECISION_UNAVAILABLE,
            "availability query must be unavailable before any execution");

    double forces[3] = {0.0, 0.0, 0.0};
    generativeqc_result_descriptor result{sizeof(generativeqc_result_descriptor),
                                          GENERATIVEQC_ABI_VERSION,
                                          0.0,
                                          forces,
                                          3,
                                          0,
                                          0.0,
                                          0.0,
                                          0,
                                          GENERATIVEQC_BACKEND_CPU_REFERENCE};
    const generativeqc_status executed = generativeqc_calculation_execute(he.calculation, &result);
    require(executed == GENERATIVEQC_STATUS_SUCCESS && result.converged == 1,
            "He RHF reference run did not converge");
    require(generativeqc_calculation_get_precision_provenance(he.calculation, &prov) ==
                GENERATIVEQC_STATUS_SUCCESS,
            "provenance must be available after a completed run");
    require(prov.requested_mode == GENERATIVEQC_PRECISION_FP64,
            "completed run reports the requested fp64 policy");
    require(generativeqc_calculation_get_precision_provenance(he.calculation, nullptr) ==
                GENERATIVEQC_STATUS_SUCCESS,
            "availability query must be available after a completed run");
    // Both descriptor fields are part of the contract: an exactly sized struct
    // that advertises a foreign ABI must be rejected and left untouched rather
    // than filled with the current layout.
    generativeqc_precision_provenance foreign_abi{sizeof(generativeqc_precision_provenance),
                                                  GENERATIVEQC_ABI_VERSION + 1U};
    foreign_abi.policy_version = 4242U;
    foreign_abi.requested_mode = 4242;
    require(generativeqc_calculation_get_precision_provenance(he.calculation, &foreign_abi) ==
                GENERATIVEQC_STATUS_ABI_MISMATCH,
            "a foreign abi_version must be rejected");
    require(foreign_abi.policy_version == 4242U && foreign_abi.requested_mode == 4242 &&
                foreign_abi.struct_size == sizeof(generativeqc_precision_provenance) &&
                foreign_abi.mixed_precision_reserved_error == 0.0 &&
                foreign_abi.refinement_iterations == 0,
            "a rejected descriptor must not be modified");
    constexpr std::uint32_t truncated_precision_size = static_cast<std::uint32_t>(
        offsetof(generativeqc_precision_provenance, mixed_stage_fock_builds));
    generativeqc_precision_provenance truncated{truncated_precision_size, GENERATIVEQC_ABI_VERSION};
    truncated.mixed_stage_fock_builds = 4242U;
    require(generativeqc_calculation_get_precision_provenance(he.calculation, &truncated) ==
                GENERATIVEQC_STATUS_ABI_MISMATCH,
            "a truncated precision-provenance descriptor must be rejected");
    require(truncated.struct_size == truncated_precision_size &&
                truncated.mixed_stage_fock_builds == 4242U,
            "a rejected truncated precision descriptor must not be modified");
    generativeqc_calculation_destroy(he.calculation);
    generativeqc_system_destroy(he.system);
    generativeqc_context_destroy(he.context);
  }
  // Be: Z=4, one s primitive -> one AO but two occupied pairs, so the cold
  // host plan throws after prepare succeeds ("basis has fewer orbitals than
  // occupied electron pairs"). A throw must reset the record back to
  // unavailable rather than leak the previous run's provenance.
  {
    const PreparedSingleAtom be = prepare_single_atom_rhf(4, 1);
    generativeqc_precision_provenance prov{sizeof(generativeqc_precision_provenance),
                                           GENERATIVEQC_ABI_VERSION};
    require(generativeqc_calculation_get_precision_provenance(be.calculation, &prov) ==
                GENERATIVEQC_STATUS_PRECISION_UNAVAILABLE,
            "provenance must be unavailable before a failing execution");

    double forces[3] = {0.0, 0.0, 0.0};
    generativeqc_result_descriptor result{sizeof(generativeqc_result_descriptor),
                                          GENERATIVEQC_ABI_VERSION,
                                          0.0,
                                          forces,
                                          3,
                                          0,
                                          0.0,
                                          0.0,
                                          0,
                                          GENERATIVEQC_BACKEND_CPU_REFERENCE};
    const generativeqc_status executed = generativeqc_calculation_execute(be.calculation, &result);
    require(executed != GENERATIVEQC_STATUS_SUCCESS, "Be/1s RHF must fail to converge");
    require(generativeqc_calculation_get_precision_provenance(be.calculation, &prov) ==
                GENERATIVEQC_STATUS_PRECISION_UNAVAILABLE,
            "provenance must fall back to unavailable after a failed execution");
    require(generativeqc_calculation_get_precision_provenance(be.calculation, nullptr) ==
                GENERATIVEQC_STATUS_PRECISION_UNAVAILABLE,
            "availability query must fall back to unavailable after a failed execution");
    generativeqc_calculation_destroy(be.calculation);
    generativeqc_system_destroy(be.system);
    generativeqc_context_destroy(be.context);
  }
}

}  // namespace

int main() {
  try {
    int available = -1;
    require(generativeqc_method_available(GENERATIVEQC_METHOD_RHF, &available) ==
                    GENERATIVEQC_STATUS_SUCCESS &&
                available == 1,
            "RHF capability query failed");
    require(generativeqc_method_available(GENERATIVEQC_METHOD_UHF, &available) ==
                    GENERATIVEQC_STATUS_SUCCESS &&
                available == 1,
            "UHF capability query failed");

    verify_precision_provenance_gate();
    verify_context_detail_storage();

    generativeqc_method_capabilities_descriptor capabilities{
        sizeof(generativeqc_method_capabilities_descriptor),
        GENERATIVEQC_ABI_VERSION,
        0,
        0,
        0,
        0,
        0};
    require(generativeqc_method_get_capabilities(GENERATIVEQC_METHOD_RHF, &capabilities) ==
                GENERATIVEQC_STATUS_SUCCESS,
            "RHF detailed capability query failed");
    require(capabilities.family == GENERATIVEQC_METHOD_FAMILY_HARTREE_FOCK &&
                capabilities.available == 1 && capabilities.supports_batch == 1 &&
                capabilities.supported_properties ==
                    (GENERATIVEQC_PROPERTY_ENERGY | GENERATIVEQC_PROPERTY_FORCES),
            "RHF detailed capabilities are incorrect");
    require(generativeqc_method_get_capabilities(GENERATIVEQC_METHOD_RCCSD_T, &capabilities) ==
                    GENERATIVEQC_STATUS_SUCCESS &&
                capabilities.family == GENERATIVEQC_METHOD_FAMILY_COUPLED_CLUSTER &&
                capabilities.available == 1 &&
                capabilities.supported_properties ==
                    (GENERATIVEQC_PROPERTY_ENERGY | GENERATIVEQC_PROPERTY_FORCES) &&
                capabilities.supports_batch == 1,
            "RCCSD(T) native energy/force capability is incorrect");

    const Evaluation center = h2(1.4, true);
    require(std::abs(center.energy - (-1.11671432506255)) < 2.0e-9,
            "H2/STO-3G RHF energy differs from the reference");
    for (int axis = 0; axis < 3; ++axis) {
      require(std::abs(center.forces[axis] + center.forces[3 + axis]) < 2.0e-10,
              "forces violate translational invariance");
    }

    const double step = 1.0e-4;
    const Evaluation plus = h2(1.4 + step);
    const Evaluation minus = h2(1.4 - step);
    const double d_energy_d_distance = (plus.energy - minus.energy) / (2.0 * step);
    // With atoms at +/-R/2, force_z(atom 1) equals -dE/dR.
    require(std::abs(center.forces[5] + d_energy_d_distance) < 2.0e-6,
            "analytic RHF force disagrees with finite differences");

    for (bool unrestricted : {false, true}) {
      h2(1.4, true, GENERATIVEQC_BACKEND_CPU_REFERENCE, GENERATIVEQC_DENSITY_FITTING_CPU_REFERENCE,
         unrestricted);
    }
#if GENERATIVEQC_HAS_CUDA
    generativeqc_context_descriptor probe{sizeof(generativeqc_context_descriptor),
                                          GENERATIVEQC_ABI_VERSION, 0, GENERATIVEQC_BACKEND_CUDA};
    generativeqc_context* cuda_context = nullptr;
    if (generativeqc_context_create(&probe, &cuda_context) == GENERATIVEQC_STATUS_SUCCESS) {
      generativeqc_context_destroy(cuda_context);
      for (bool unrestricted : {false, true}) {
        for (std::uint64_t budget : {0ULL, 8ULL * 1024ULL * 1024ULL}) {
          h2(1.4, true, GENERATIVEQC_BACKEND_CUDA, GENERATIVEQC_DENSITY_FITTING_CUDA, unrestricted,
             budget);
        }
      }
    }
#endif

    std::cout << "H2 energy: " << center.energy << '\n';
    std::cout << "H2 force z(atom 1): " << center.forces[5] << '\n';
    return EXIT_SUCCESS;
  } catch (const std::exception& error) {
    std::cerr << "test failure: " << error.what() << '\n';
    return EXIT_FAILURE;
  }
}
