#include <cstdint>
#include <exception>

#include "api/handles.hpp"
#include "methods/execution_failure.hpp"
#include "scf/mean_field.hpp"

/** Test-only bridge to the same physical-reference owner used by MP2 and CC.
 * It stops before correlation, so a failed HF iterate can be examined without
 * allocating transformed integrals or accidentally publishing a CC result. */
extern "C" int conventional_ecp_hf_probe(generativeqc_system* system, unsigned maximum_iterations,
                                         double* diagnostics) {
  using namespace generativeqc;
  try {
    scf::ScfOptions options;
    options.max_iterations = maximum_iterations;
    options.energy_tolerance = 2.3827302298009143e-9;
    options.density_tolerance = 1e-11;
    options.screening_tolerance = 0;
    options.compute_forces = false;
    options.export_physical_reference = true;
    options.reference_memory_budget_bytes = 43ULL << 30;
    options.precision_mode = GENERATIVEQC_PRECISION_FP64;
    const auto result = scf::run_rhf_cuda(system->data, options, 0);
    diagnostics[0] = result.converged;
    diagnostics[1] = bool(result.reference);
    diagnostics[2] = result.iterations;
    diagnostics[3] = result.energy;
    diagnostics[4] = result.energy_change;
    diagnostics[5] = result.density_rms;
    diagnostics[6] = result.physical_residual_rms;
    diagnostics[7] = result.reference ? result.reference->commutator_residual : 0.0;
    return result.converged ? GENERATIVEQC_STATUS_SUCCESS : GENERATIVEQC_STATUS_NOT_CONVERGED;
  } catch (...) {
    return methods::item_exception_status();
  }
}
