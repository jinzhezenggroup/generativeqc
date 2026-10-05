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
#include "runtime/cuda_resources.cuh"
#include "scf/cuda_direct_jk_device.hpp"

extern "C" int rhf_frame_response_probe(void* opaque, std::size_t occupied,
                                        const double* const* inputs, bool blas, bool relax,
                                        std::size_t budget, std::size_t max_iterations,
                                        double screening, unsigned nuclear_schedule,
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
    options.orbital_screening_tolerance = screening;
    options.profile_jk = true;
    options.bilinear_derivative = nuclear_schedule == 1;
    options.symmetric_polarization = nuclear_schedule == 2;
    options.relax_orbitals = relax;
    options.maximum_bytes = budget;
    // This probe's scalar ablation uses a real resource constraint. The public
    // method has no provider selector. Small fixtures fit below the shared
    // 96 MiB provider allowance; never enlarge the caller's requested budget.
    if (!blas) options.maximum_bytes = std::min<std::size_t>(budget, (96ULL << 20) - 1);
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
                              result.global_stability_certified,
                              result.jk_census_actions,
                              result.jk_quartet_visits,
                              result.jk_eri_evaluations,
                              result.exact_refinements,
                              result.screened_jk_actions,
                              result.prepared_contractions,
                              result.contraction_binding_bytes,
                              result.prepared_contraction_summands};
    std::copy(std::begin(stats), std::end(stats), counts);
    diagnostics[0] = result.orbital_residual;
    diagnostics[1] = result.maximum_stationarity;
    diagnostics[2] = result.reference_residual;
    diagnostics[3] = result.applied_screening;
    diagnostics[4] = result.screened_residual;
    return 0;
  } catch (const std::exception& failure) {
    if (error && error_size) std::snprintf(error, error_size, "%s", failure.what());
    return 1;
  }
}

// An independent provider seam for signed-density linearity, self-adjointness
// and tiny dense integral oracles. Production molecular response uses no oracle.
extern "C" int rhf_linear_jk_probe(void* opaque, const double* density, double threshold,
                                   double* output, std::uint64_t* counts, char* error,
                                   std::size_t error_size) noexcept {
  using namespace generativeqc;
  scf::CudaDirectJkPlan* plan = nullptr;
  try {
    const auto& raw = *static_cast<posthf::RawSource*>(opaque);
    const auto n = raw.nbf(), nn = n * n;
    runtime::CudaDeviceScope scope(0);
    std::string detail;
    scf::CudaDirectJkDiagnostic stats;
    const auto check = [&](generativeqc_status code) {
      if (code != GENERATIVEQC_STATUS_SUCCESS) throw std::runtime_error(detail);
    };
    check(scf::create_cuda_direct_jk_plan(0, {raw.orbital()}, 0, 0.0, 1ULL << 30, &plan, stats,
                                          detail));
    const auto stream = scf::cuda_direct_jk_stream(plan);
    // These buffers die before their stream; a failure drains before releasing
    // the caller's host output and density pointers.
    {
      int failure = 0;
      runtime::OwnedCudaBuffer<double> storage(0, 3 * nn, stream);
      runtime::OwnedCudaBuffer<int> flag(0, 1, stream);
      runtime::OwnedCudaBuffer<std::uint64_t> census(0, 2, stream);
      auto* d = storage.get();
      auto* j = d + nn;
      auto* k = j + nn;
      runtime::cuda_resource_check(
          cudaMemcpyAsync(d, density, nn * sizeof(double), cudaMemcpyHostToDevice, stream));
      auto spec = scf::make_hf_fock_spec(scf::FockSpin::Restricted);
      spec.derivative_order = 0;
      check(scf::enqueue_cuda_direct_jk_linear_device(plan, spec, d, nn, j, k, flag.get(),
                                                      threshold, census.get(), detail));
      runtime::cuda_resource_check(
          cudaMemcpyAsync(output, j, 2 * nn * sizeof(double), cudaMemcpyDeviceToHost, stream));
      runtime::cuda_resource_check(cudaMemcpyAsync(counts, census.get(), 2 * sizeof(std::uint64_t),
                                                   cudaMemcpyDeviceToHost, stream));
      runtime::cuda_resource_check(
          cudaMemcpyAsync(&failure, flag.get(), sizeof(int), cudaMemcpyDeviceToHost, stream));
      runtime::cuda_resource_check(cudaStreamSynchronize(stream));
      if (failure) throw std::runtime_error("nonfinite fixed-mask J/K");
    }
    scf::destroy_cuda_direct_jk_plan(plan);
    return 0;
  } catch (const std::exception& failure) {
    scf::destroy_cuda_direct_jk_plan(plan);
    if (error && error_size) std::snprintf(error, error_size, "%s", failure.what());
    return 1;
  }
}

// Force the canonical bilinear API even on SPD so small dense oracles qualify
// the coefficient separately from production's specialized-shell preference.
extern "C" int rhf_bilinear_derivative_probe(void* opaque, const double* density,
                                             const double* seed, double* output,
                                             std::uint64_t* counts, char* error,
                                             std::size_t error_size) noexcept {
  using namespace generativeqc;
  scf::CudaDirectJkPlan* plan = nullptr;
  try {
    const auto& raw = *static_cast<posthf::RawSource*>(opaque);
    const auto nn = raw.nbf() * raw.nbf();
    runtime::CudaDeviceScope scope(0);
    std::string detail;
    scf::CudaDirectJkDiagnostic stats;
    const auto check = [&](generativeqc_status code) {
      if (code != GENERATIVEQC_STATUS_SUCCESS) throw std::runtime_error(detail);
    };
    check(scf::create_cuda_direct_jk_plan(0, {raw.orbital()}, 1, 0.0, 1ULL << 30, &plan, stats,
                                          detail));
    const auto stream = scf::cuda_direct_jk_stream(plan);
    {
      runtime::OwnedCudaBuffer<double> storage(0, 2 * nn, stream);
      runtime::OwnedCudaBuffer<std::uint64_t> census(0, 2, stream);
      runtime::cuda_resource_check(cudaMemcpyAsync(storage.get(), density, nn * sizeof(double),
                                                   cudaMemcpyHostToDevice, stream));
      runtime::cuda_resource_check(cudaMemcpyAsync(storage.get() + nn, seed, nn * sizeof(double),
                                                   cudaMemcpyHostToDevice, stream));
      std::vector<double> result;
      check(scf::execute_cuda_direct_bilinear_derivative_device(
          plan, storage.get(), storage.get() + nn, nn, result, census.get(), detail));
      runtime::cuda_resource_check(cudaMemcpyAsync(counts, census.get(), 2 * sizeof(std::uint64_t),
                                                   cudaMemcpyDeviceToHost, stream));
      runtime::cuda_resource_check(cudaStreamSynchronize(stream));
      std::copy(result.begin(), result.end(), output);
    }
    scf::destroy_cuda_direct_jk_plan(plan);
    return 0;
  } catch (const std::exception& failure) {
    scf::destroy_cuda_direct_jk_plan(plan);
    if (error && error_size) std::snprintf(error, error_size, "%s", failure.what());
    return 1;
  }
}
