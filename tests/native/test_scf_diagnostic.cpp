#include <array>
#include <cmath>
#include <iostream>
#include <stdexcept>

#include "dft/ao_grid.hpp"
#include "dft/grid.hpp"
#include "generativeqc/generativeqc.h"
#include "molecule/basis.hpp"
#include "scf/fock_prepared.hpp"
#include "scf/mean_field.hpp"

namespace {
using namespace generativeqc;

void require(bool condition, const char* message) {
  if (!condition) throw std::runtime_error(message);
}

struct Handles {
  generativeqc_context* context{};
  generativeqc_system* system{};
  generativeqc_calculation* calculation{};
  ~Handles() {
    generativeqc_calculation_destroy(calculation);
    generativeqc_system_destroy(system);
    generativeqc_context_destroy(context);
  }
};

void check_measures(bool uks, bool pbe) {
  // Asymmetric H3/H3+ is nonstationary after one iteration. Symmetric H2
  // would make both measures almost zero and hide the adapter regression.
  const std::array<generativeqc_atom, 3> atoms{
      {{1, 0, 0, 0}, {1, .15, .13, 1.5}, {1, .6, .26, 3.0}}};
  const std::array<generativeqc_primitive, 3> primitives{
      {{3.425250914, .1543289673}, {.6239137298, .5353281423}, {.168855404, .4446345422}}};
  const std::array<generativeqc_shell, 3> shells{{{0, 0, 0, 3}, {1, 0, 0, 3}, {2, 0, 0, 3}}};
  core::System native_system;
  native_system.charge = uks ? 0 : 1;
  native_system.multiplicity = uks ? 2 : 1;
  for (unsigned i = 0; i < atoms.size(); ++i) {
    native_system.atoms.push_back({1, {atoms[i].x, atoms[i].y, atoms[i].z}});
    native_system.shells.push_back({i, 0, {}});
    for (const auto& primitive : primitives)
      native_system.shells.back().primitives.push_back({primitive.exponent, primitive.coefficient});
  }
  std::string detail;
  require(molecule::validate_and_normalize(native_system, detail) == GENERATIVEQC_STATUS_SUCCESS,
          "invalid diagnostic fixture");
  scf::FockBuildSpec spec;
  spec.spin = uks ? scf::FockSpin::Unrestricted : scf::FockSpin::Restricted;
  spec.exchange.present = false;
  spec.derivative_order = 0;
  const scf::PreparedFockPlan plan(native_system, nullptr,
                                   scf::resolve_fock_build(spec, scf::FockBackend::Cpu));
  const dft::AoBasis basis(native_system);
  const dft::MolecularGrid grid(native_system);
  scf::ScfOptions options;
  options.compute_forces = false;
  options.max_iterations = 1;
  const auto run = uks ? (pbe ? scf::run_pbe_uks : scf::run_lda_uks)
                       : (pbe ? scf::run_pbe_rks : scf::run_lda_rks);
  const auto native = run(plan, basis, grid, options, nullptr);
  require(std::abs(native.density_rms - native.physical_residual_rms) > 1e-4,
          "fixture cannot distinguish density update from physical residual");

  Handles handles;
  const generativeqc_context_descriptor context{sizeof(context), GENERATIVEQC_ABI_VERSION, 0,
                                                GENERATIVEQC_BACKEND_CPU_REFERENCE};
  require(generativeqc_context_create(&context, &handles.context) == GENERATIVEQC_STATUS_SUCCESS,
          "context creation failed");
  const generativeqc_system_descriptor system{sizeof(system),
                                              GENERATIVEQC_ABI_VERSION,
                                              atoms.data(),
                                              3,
                                              shells.data(),
                                              3,
                                              primitives.data(),
                                              3,
                                              native_system.charge,
                                              native_system.multiplicity,
                                              GENERATIVEQC_BASIS_CARTESIAN};
  require(generativeqc_system_create(handles.context, &system, &handles.system) ==
              GENERATIVEQC_STATUS_SUCCESS,
          "system creation failed");
  generativeqc_method_descriptor method{};
  method.struct_size = sizeof(method);
  method.abi_version = GENERATIVEQC_ABI_VERSION;
  method.method = uks ? (pbe ? GENERATIVEQC_METHOD_PBE_UKS : GENERATIVEQC_METHOD_LDA_UKS)
                      : (pbe ? GENERATIVEQC_METHOD_PBE_RKS : GENERATIVEQC_METHOD_LDA_RKS);
  method.max_iterations = 1;
  require(generativeqc_calculation_prepare(handles.context, handles.system, &method,
                                           &handles.calculation) == GENERATIVEQC_STATUS_SUCCESS,
          "preparation failed");
  generativeqc_scf_diagnostic diagnostic{sizeof(diagnostic), GENERATIVEQC_ABI_VERSION, -7, -11};
  generativeqc_ks_diagnostic ks{};
  ks.struct_size = sizeof(ks);
  ks.abi_version = GENERATIVEQC_ABI_VERSION;
  ks.nuclear_energy = -19;
  require(generativeqc_calculation_get_ks_diagnostic(handles.calculation, &ks, nullptr, 0) ==
                  GENERATIVEQC_STATUS_NOT_IMPLEMENTED &&
              ks.nuclear_energy == -19,
          "unexecuted calculation published a KS snapshot");
  require(generativeqc_calculation_get_scf_diagnostic(nullptr, &diagnostic) ==
              GENERATIVEQC_STATUS_INVALID_ARGUMENT,
          "null calculation accepted");
  require(generativeqc_calculation_get_scf_diagnostic(handles.calculation, nullptr) ==
                  GENERATIVEQC_STATUS_NOT_IMPLEMENTED &&
              generativeqc_calculation_get_scf_diagnostic(handles.calculation, &diagnostic) ==
                  GENERATIVEQC_STATUS_NOT_IMPLEMENTED &&
              diagnostic.density_rms == -7 && diagnostic.physical_residual_rms == -11,
          "unexecuted calculation published a diagnostic");
  generativeqc_result_descriptor result{};
  result.struct_size = sizeof(result);
  result.abi_version = GENERATIVEQC_ABI_VERSION;
  require(generativeqc_calculation_execute(handles.calculation, &result) ==
              GENERATIVEQC_STATUS_NOT_CONVERGED,
          "one-iteration fixture unexpectedly converged");
  require(generativeqc_calculation_get_scf_diagnostic(handles.calculation, nullptr) ==
                  GENERATIVEQC_STATUS_SUCCESS &&
              generativeqc_calculation_get_scf_diagnostic(handles.calculation, &diagnostic) ==
                  GENERATIVEQC_STATUS_SUCCESS,
          "completed nonconverged diagnostic unavailable");
  require(std::abs(result.density_rms - native.density_rms) < 1e-13 &&
              diagnostic.density_rms == result.density_rms &&
              std::abs(diagnostic.physical_residual_rms - native.physical_residual_rms) < 1e-13,
          "C API changed the legacy density measure or conflated it with physical residual");
  generativeqc_ks_iteration row{};
  row.struct_size = sizeof(row);
  row.abi_version = GENERATIVEQC_ABI_VERSION;
  require(generativeqc_calculation_get_ks_diagnostic(handles.calculation, &ks, &row, 1) ==
              GENERATIVEQC_STATUS_SUCCESS,
          "completed valid nonconverged KS history unavailable");
  const auto& physical = native.dft_diagnostic;
  require(ks.history_count == 1 && row.iteration == 1 && ks.fock_builds == 1 &&
              ks.grid_points == grid.point_count() && ks.tile_points == 256 &&
              ks.required_ao_order == (pbe ? 1U : 0U) && !ks.initial_density_used &&
              ks.occupations[0] == physical.occupations[0] &&
              ks.occupations[1] == physical.occupations[1] &&
              std::abs(ks.xc_energy - physical.components.xc) < 1e-13 &&
              std::abs(ks.physical_residual_max - physical.physical_residual) < 1e-13 &&
              row.xc_energy == ks.xc_energy && std::isinf(row.energy_change) &&
              std::abs(row.density_change_max - physical.history[0].density_change) < 1e-13,
          "KS snapshot lost its model, occupations, physical components or iteration history");
  ks.nuclear_energy = -19;
  require(generativeqc_calculation_get_ks_diagnostic(handles.calculation, &ks, &row, 0) ==
                  GENERATIVEQC_STATUS_INVALID_ARGUMENT &&
              ks.nuclear_energy == -19,
          "short KS history buffer partially overwrote summary");
  row.abi_version += 1;
  require(generativeqc_calculation_get_ks_diagnostic(handles.calculation, &ks, &row, 1) ==
                  GENERATIVEQC_STATUS_ABI_MISMATCH &&
              ks.nuclear_energy == -19,
          "invalid KS history ABI partially overwrote summary");
  diagnostic.abi_version += 1;
  require(generativeqc_calculation_get_scf_diagnostic(handles.calculation, &diagnostic) ==
              GENERATIVEQC_STATUS_ABI_MISMATCH,
          "diagnostic ABI mismatch accepted");
  diagnostic.abi_version = GENERATIVEQC_ABI_VERSION;
  diagnostic.struct_size = sizeof(diagnostic) - 1;
  require(generativeqc_calculation_get_scf_diagnostic(handles.calculation, &diagnostic) ==
              GENERATIVEQC_STATUS_ABI_MISMATCH,
          "undersized diagnostic accepted");
  diagnostic = {sizeof(diagnostic), GENERATIVEQC_ABI_VERSION, -7, -11};
  // A valid execution request that fails in the backend must discard the
  // preceding run's values, for both availability probes and copy-out.
  std::array<double, 9> forces{};
  result.forces = forces.data();
  result.force_count = forces.size();
  require(generativeqc_calculation_execute(handles.calculation, &result) ==
              GENERATIVEQC_STATUS_NOT_IMPLEMENTED,
          "unsupported force execution did not fail");
  require(generativeqc_calculation_get_scf_diagnostic(handles.calculation, nullptr) ==
                  GENERATIVEQC_STATUS_NOT_IMPLEMENTED &&
              generativeqc_calculation_get_scf_diagnostic(handles.calculation, &diagnostic) ==
                  GENERATIVEQC_STATUS_NOT_IMPLEMENTED &&
              diagnostic.density_rms == -7 && diagnostic.physical_residual_rms == -11,
          "failed backend execution retained stale diagnostic values");
  require(generativeqc_calculation_get_ks_diagnostic(handles.calculation, &ks, nullptr, 0) ==
                  GENERATIVEQC_STATUS_NOT_IMPLEMENTED &&
              ks.nuclear_energy == -19,
          "failed execution exposed the preceding KS history");
}
}  // namespace

int main() {
  try {
    for (bool uks : {false, true})
      for (bool pbe : {false, true}) check_measures(uks, pbe);
    std::cout << "RKS/UKS public density and physical residual measures remain distinct\n";
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
