// Validation adapter: publish outputs only after the complete native owner
// succeeds. Failed preflight or arithmetic leaves the caller's sentinels intact.
#include <algorithm>
#include <cstdio>
#include <exception>

#include "cc/df_triples.hpp"

extern "C" int df_triples_fock_probe(std::size_t o, std::size_t v, std::size_t q,
                                     const double* const* inputs, double threshold,
                                     std::size_t budget, std::size_t caller_bytes, std::size_t rows,
                                     std::size_t panels, double* const* output, double* values,
                                     std::size_t* counts, char* error,
                                     std::size_t error_size) noexcept {
  try {
    const auto r = generativeqc::cc::triples::fock_response_df_cuda(
        o, v, q, inputs[0], inputs[1], inputs[2], inputs[3], inputs[4], inputs[5], inputs[6],
        inputs[7], inputs[8], threshold, budget, 0, caller_bytes, rows, panels);
    std::copy(r.foo.begin(), r.foo.end(), output[0]);
    std::copy(r.fvv.begin(), r.fvv.end(), output[1]);
    const double scalars[]{r.seconds, r.minimum_absolute_denominator};
    const std::size_t work[]{r.numeric_capacity_bytes,
                             r.borrowed_host_bytes,
                             r.workspace_bytes,
                             r.arena_bytes,
                             r.provider_retained_bytes,
                             r.page_capacity,
                             r.page_count,
                             r.panel_capacity,
                             r.occupied_pairs,
                             r.unique_vector_cubes,
                             r.vector_cubes,
                             r.page_builds,
                             r.panel_gemms,
                             r.w_gemms,
                             r.fock_gemms,
                             r.contraction_summands,
                             r.scalar_evaluations,
                             r.audit_kernels,
                             r.scatter_kernels,
                             r.h2d_bytes,
                             r.d2h_bytes};
    std::copy(std::begin(scalars), std::end(scalars), values);
    std::copy(std::begin(work), std::end(work), counts);
    return 0;
  } catch (const std::exception& e) {
    if (error && error_size) std::snprintf(error, error_size, "%s", e.what());
    return 1;
  }
}

extern "C" int df_triples_combined_probe(std::size_t o, std::size_t v, std::size_t q,
                                         const double* const* inputs, double threshold,
                                         std::size_t budget, std::size_t caller_bytes,
                                         std::size_t rows, std::size_t panels,
                                         double* const* output, double* values, std::size_t* counts,
                                         char* error, std::size_t error_size) noexcept {
  try {
    const auto r = generativeqc::cc::triples::pullback_and_fock_df_cuda(
        o, v, q, inputs[0], inputs[1], inputs[2], inputs[3], inputs[4], inputs[5], inputs[6],
        inputs[7], inputs[8], threshold, budget, 0, caller_bytes, rows, panels, true, false);
    if (r.pullback.gap.requested || !r.pullback.eps_o.empty() || !r.pullback.eps_v.empty())
      throw std::runtime_error("combined gap-free response published epsilon cotangents");
    std::copy(r.fock.foo.begin(), r.fock.foo.end(), output[0]);
    std::copy(r.fock.fvv.begin(), r.fock.fvv.end(), output[1]);
    const std::array response{&r.pullback.bov, &r.pullback.bvv, &r.pullback.ovoo, &r.pullback.ovov,
                              &r.pullback.fov, &r.pullback.t1,  &r.pullback.t2};
    for (std::size_t x = 0; x < response.size(); ++x)
      std::copy(response[x]->begin(), response[x]->end(), output[x + 2]);
    const double scalars[]{r.pullback.diagnostic.energy, r.seconds,
                           r.fock.minimum_absolute_denominator};
    const std::size_t work[]{r.shared_input_device_bytes,
                             r.shared_h2d_bytes,
                             r.pullback.diagnostic.h2d_bytes,
                             r.fock.h2d_bytes,
                             r.pullback.numeric_capacity_bytes,
                             r.fock.numeric_capacity_bytes,
                             r.pullback.borrowed_host_bytes,
                             r.fock.borrowed_host_bytes,
                             r.shared_response_cubes,
                             r.avoided_w_gemms,
                             r.fock.vector_cubes,
                             r.fock.w_gemms,
                             r.pullback.diagnostic.occupied_tiles,
                             r.pullback.diagnostic.moment_gemms};
    std::copy(std::begin(scalars), std::end(scalars), values);
    std::copy(std::begin(work), std::end(work), counts);
    return 0;
  } catch (const std::exception& e) {
    if (error && error_size) std::snprintf(error, error_size, "%s", e.what());
    return 1;
  }
}
