#pragma once

#include <vector>

#include "cc/df_source_response.hpp"
#include "cc/df_triples.hpp"
#include "cc/lambda_response.hpp"
#include "hf/rhf_frame_response.hpp"
#include "methods/rccsd_method.hpp"

namespace generativeqc::methods::detail {
/** Complete internal endpoint, on an unchanged conventional RHF reference with
 * a DF correlation Hamiltonian. Public capability promotion is separate. */
struct DFCCSDTResult {
  double energy{}, reference_energy{}, correlation_energy{}, triples_energy{};
  std::vector<double> forces;
  std::size_t numeric_capacity_bytes{}, source_weight_values{}, metric_weight_values{};
  double total_seconds{}, triples_seconds{}, lambda_seconds{}, source_response_seconds{},
      orbital_seconds{};
  CcPerformanceDiagnostic primal;
  cc::SolverDiagnostic solver;
  cc::LambdaDiagnostic lambda;
  hf::RHFFrameResponseResult orbital;
  cc::triples::DFCudaResult triples;
  cc::triples::DFCudaFockResult triples_fock;
};

/** Cold normalized geometry -> exact native CUDA RHF -> DF source -> CCSD(T)
 * -> complete nuclear forces. Retains the exact source/frame only when forces
 * are requested, and consumes it before either input geometry can change.
 * Energy-only calls share the same scientific Hamiltonian and solver gates.
 * The optional CCSD-only mode omits triples and is an independent closure gate.
 * Disabling df_auxiliary_reduction retains expanded Lambda actions for matched
 * endpoint validation; it changes only the response schedule, not the method.
 * Disabling df_matrix_gemm selects the scalar DF residual for matched energy
 * and force endpoint comparisons with the same compiled library.
 * All phase bounds charge simultaneously live owners; no CPU integral/CC
 * reference fallback or four-index full MO Hamiltonian is used.
 */
DFCCSDTResult run_df_ccsdt_native(runtime::ExecutionContext&, const core::System& orbital,
                                  const core::System& auxiliary,
                                  const generativeqc_method_descriptor&, bool forces = true,
                                  bool with_triples = true, bool df_auxiliary_reduction = true,
                                  bool df_matrix_gemm = true, bool lambda_matrix_gemm = true,
                                  std::size_t lambda_batch_limit = 8,
                                  std::size_t ccsd_batch_limit = 8);
}  // namespace generativeqc::methods::detail
