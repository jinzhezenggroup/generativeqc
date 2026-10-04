// Validation-only seam: the CPU source is normalized metadata, never an oracle
// integral provider. All production response and derivative actions run CUDA.
#include <algorithm>
#include <array>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <vector>

#include "hf/rhf_frame_response.hpp"
#include "posthf/capacity.hpp"
#include "posthf/raw_source.hpp"

extern "C" int rhf_frame_response_probe(void* opaque, std::size_t occupied,
                                        const double* const* inputs, bool blas, bool relax,
                                        std::size_t budget, std::size_t max_iterations,
                                        double* const* outputs, std::size_t* counts,
                                        double* diagnostics, char* error,
                                        std::size_t error_size) noexcept {
  using namespace generativeqc;
  try {
    const auto& raw = *static_cast<posthf::RawSource*>(opaque);
    const auto n = raw.nbf();
    hf::PhysicalReference ref;
    ref.nbf = n;
    ref.nocc = occupied;
    const std::array<std::vector<double>*, 6> matrices{&ref.coefficients, &ref.hcore,
                                                       &ref.fock,         &ref.overlap,
                                                       &ref.density,      &ref.orbital_energies};
    for (std::size_t i = 0; i < matrices.size(); ++i)
      matrices[i]->assign(inputs[i], inputs[i] + (i == 5 ? n : n * n));
    hf::RHFFrameResponseOptions options;
    options.matrix_blas = blas;
    options.relax_orbitals = relax;
    options.maximum_bytes = budget;
    options.gmres.max_iterations = max_iterations;
    // Account for the test's borrowed inputs/destinations as well as the native
    // reference copies; the auxiliary metadata belongs only to this probe.
    options.caller_bytes = (12 * n * n + n + 3 * raw.orbital().atoms.size()) * sizeof(double) +
                           posthf::source_capacity(raw.auxiliary());
    hf::RHFFrameResponseRecycle recycle;
    const auto* repeat = std::getenv("GENERATIVEQC_TEST_RHF_RECYCLE_REPEAT");
    if (repeat && std::string(repeat) == "1") options.recycling = &recycle;
    auto result = hf::rhf_frame_response_cuda(raw.orbital(), ref, {inputs[6], n * n},
                                              {inputs[7], n * n}, 0, options);
    if (options.recycling) {
      if (!result.recycle_published)
        throw std::runtime_error("first response did not publish subspace");
      // Reuse exactly this owned frame/source, without retaining both result
      // payloads. This is a lifecycle/action-count test, not a cold endpoint.
      result = {};
      result = hf::rhf_frame_response_cuda(raw.orbital(), ref, {inputs[6], n * n},
                                           {inputs[7], n * n}, 0, options);
      if (!result.recycled_guess) throw std::runtime_error("same-operator subspace was not reused");
    }
    const std::array<const std::vector<double>*, 6> arrays{
        &result.gradient,        &result.hcore_weights, &result.overlap_weights,
        &result.fock_ao_weights, &result.stationarity,  &result.orbital_rhs};
    for (std::size_t i = 0; i < arrays.size(); ++i)
      std::copy(arrays[i]->begin(), arrays[i]->end(), outputs[i]);
    const std::size_t stats[]{result.numeric_capacity_bytes,
                              result.jk_actions,
                              result.derivative_passes,
                              result.orbital_actions,
                              result.gemms,
                              result.contraction_terms,
                              result.h2d_bytes,
                              result.d2h_bytes,
                              result.synchronizations,
                              result.explicit_hessian_elements,
                              result.orbital_response.iterations,
                              result.global_stability_certified};
    std::copy(std::begin(stats), std::end(stats), counts);
    diagnostics[0] = result.orbital_residual;
    diagnostics[1] = result.maximum_stationarity;
    diagnostics[2] = result.reference_residual;
    return 0;
  } catch (const std::exception& failure) {
    if (error && error_size) std::snprintf(error, error_size, "%s", failure.what());
    return 1;
  }
}
