// Validation-only adapter: cold native CCSD followed by native fixed-orbital
// Lambda/parameter response. Publish only after every requested phase passes.
#include <algorithm>
#include <array>
#include <cstdio>
#include <exception>
#include <limits>
#include <vector>

#include "cc/df_lambda.hpp"
#include "cc/lambda_response.hpp"

extern "C" int df_cc_lambda_probe(std::size_t o, std::size_t v, std::size_t q, int mode,
                                  std::size_t budget, std::size_t device_budget,
                                  const double* const* input, const double* source1,
                                  const double* source2, double* const* output, double* values,
                                  std::size_t* counts, char* error,
                                  std::size_t error_size) noexcept {
  try {
    generativeqc::cc::Problem p;
    p.nocc = o;
    p.nvir = v;
    p.naux = (mode & 1) ? q : 0;
    const auto n1 = o * v, n2 = n1 * n1;
    const std::array<std::size_t, 16> sizes{o * o,
                                            n1,
                                            v * v,
                                            n2,
                                            n2,
                                            n2,
                                            p.naux ? 0 : o * v * v * v,
                                            o * v * o * o,
                                            o * o * o * o,
                                            p.naux ? 0 : v * v * v * v,
                                            n1,
                                            n2,
                                            n1,
                                            n2,
                                            p.naux * n1,
                                            p.naux * v * v};
    const std::array<std::vector<double>*, 16> fields{
        &p.foo,  &p.fov,  &p.fvv, &p.ovov, &p.ovvo,       &p.oovv,       &p.ovvv,   &p.ovoo,
        &p.oooo, &p.vvvv, &p.d1,  &p.d2,   &p.initial_t1, &p.initial_t2, &p.df_bov, &p.df_bvv};
    for (std::size_t x = 0; x < sizes.size(); ++x)
      if (sizes[x]) fields[x]->assign(input[x], input[x] + sizes[x]);
    generativeqc::cc::SolverOptions options;
    // The requested budget specifically probes response admission after a
    // successful independent primal solve; do not fail earlier in CC storage.
    options.max_bytes = 1ULL << 30;
    options.max_iterations = 100;
    options.energy_tolerance = 1e-13;
    options.residual_tolerance = 1e-11;
    const auto cc = generativeqc::cc::solve_cuda(p, options, 0);
    if (!cc.converged()) throw std::runtime_error(cc.reason);
    generativeqc::cc::LambdaOptions response;
    response.max_bytes = budget;
    response.df_max_device_bytes = device_budget;
    response.gmres.absolute_tolerance = 1e-12;
    response.df_auxiliary_reduction = !(mode & 4);
    response.df_matrix_gemm = !(mode & 8);
    response.df_core_reuse = !(mode & 64);
    response.df_audit_matrix_gemm = !(mode & 128);
    if (!(mode & 2048)) {
      response.df_primal_matrix_gemm = mode & 1024;
      response.df_auxiliary_batch_limit = (mode & 512) ? 32 : (mode & 16) ? 2 : 8;
    }
    if (mode & 32) {
      // Large finite seeds overflow intermediate adjoints. The sticky flag
      // must reject the complete action before the adapter publishes outputs.
      generativeqc::cc::detail::DFLambdaActions actions(p, cc, response, 0, false, true);
      std::vector<double> one(n1, std::numeric_limits<double>::max()),
          two(n2, std::numeric_limits<double>::max()), out_one, out_two;
      actions.transpose(bool(mode & 256), one, two, out_one, out_two);
      throw std::logic_error("overflowing adjoint was accepted");
    }
    const auto result =
        (mode & 2) ? generativeqc::cc::solve_lambda_parameter_response_cuda_with_energy_source(
                         p, cc, {source1, n1}, {source2, n2}, 0, response)
                   : generativeqc::cc::solve_lambda_parameter_response_cuda(p, cc, 0, response);
    const std::array<const std::vector<double>*, 16> arrays{
        &cc.t1,       &cc.t2,       &result.lambda.lambda1, &result.lambda.lambda2, &result.foo,
        &result.fov,  &result.fvv,  &result.ovov,           &result.ovvo,           &result.oovv,
        &result.ovoo, &result.oooo, &result.df_bov,         &result.df_bvv,         &result.ovvv,
        &result.vvvv};
    for (std::size_t x = 0; x < arrays.size(); ++x)
      if (!arrays[x]->empty()) std::copy(arrays[x]->begin(), arrays[x]->end(), output[x]);
    const auto& d = result.lambda.diagnostic;
    const double scalars[]{cc.correlation_energy,
                           d.cc_r1_max,
                           d.cc_r2_max,
                           d.lambda_residual_norm,
                           d.independent_residual_norm,
                           d.independent_residual_max};
    const std::size_t work[]{d.iterations,
                             d.operator_actions,
                             d.numeric_capacity_bytes,
                             d.owned_device_bytes,
                             d.h2d_bytes,
                             d.d2h_bytes,
                             d.synchronizations,
                             d.df_auxiliary_slices,
                             d.df_contraction_terms,
                             d.df_generated_kernels,
                             std::size_t(d.df_auxiliary_reduction),
                             d.df_preparation_calls,
                             d.df_reduced_actions,
                             std::size_t(d.df_matrix_gemm),
                             d.df_auxiliary_batch_size,
                             d.df_auxiliary_batches,
                             d.df_gemm_calls,
                             d.df_gemm_summands,
                             d.df_packing_output_bytes,
                             d.df_provider_allowance_bytes,
                             std::size_t(d.df_core_reuse),
                             d.df_core_reuse_bytes,
                             d.df_core_reuse_preparations,
                             d.df_core_reuse_actions,
                             std::size_t(d.df_audit_matrix_gemm),
                             d.df_audit_arena_bytes,
                             std::size_t(d.df_primal_matrix_gemm),
                             d.df_available_device_bytes,
                             d.df_device_limit_bytes};
    std::copy(std::begin(scalars), std::end(scalars), values);
    std::copy(std::begin(work), std::end(work), counts);
    return 0;
  } catch (const std::exception& e) {
    if (error && error_size) std::snprintf(error, error_size, "%s", e.what());
    return 1;
  }
}
