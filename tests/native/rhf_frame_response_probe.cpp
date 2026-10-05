// Validation-only seam: the CPU source is normalized metadata, never an oracle
// integral provider. All production response and derivative actions run CUDA.
#include <algorithm>
#include <array>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <memory>
#include <vector>

#include "hf/rhf_frame_response.hpp"
#include "posthf/capacity.hpp"
#include "posthf/raw_source.hpp"
#include "runtime/cuda_resources.cuh"
#include "runtime/resource_ledger.hpp"
#include "scf/cuda/direct_jk_plan.hpp"
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
    if (const auto* resident = std::getenv("GENERATIVEQC_TEST_RHF_RESIDENT_JK_BYTES"))
      options.resident_jk_maximum_bytes = std::stoull(resident);
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
    if (options.resident_jk_maximum_bytes.value_or(0) > 0 && !result.resident_jk_bytes)
      throw std::runtime_error("resident response qualification did not exercise source reuse");
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

/** Exercise one immutable Direct source across signed restricted/unrestricted
 * inputs, and independently recompute every output through the uncached source.
 * The cache admission receipt separates retained values from executed radial
 * evaluations; CUDA events measure the action, not setup or host transfers. */
extern "C" int rhf_resident_jk_probe(void* opaque, const double* densities,
                                     std::size_t density_count, bool unrestricted,
                                     std::size_t maximum_cache_bytes, double* output,
                                     std::uint64_t* counts, double* timings, char* error,
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
    counts[0] = scf::cuda_direct_jk_resident_value_bytes(plan);
    const auto prepare_started = std::chrono::steady_clock::now();
    generativeqc_status prepared;
    {
      // Refuse the actual allocation after size admission, without exhausting
      // the GPU or changing the provider's production fallback policy.
      struct AdmissionScope {
        std::shared_ptr<runtime::DeviceResourceLedger> previous{
            runtime::active_device_resource_ledger};
        std::shared_ptr<runtime::DeviceResourceLedger> refused;
        AdmissionScope() {
          const auto* requested = std::getenv("GENERATIVEQC_TEST_RHF_RESIDENT_ALLOCATION_REFUSAL");
          if (requested && std::string(requested) == "1") {
            refused = std::make_shared<runtime::DeviceResourceLedger>();
            refused->device = 0;
            runtime::active_device_resource_ledger = refused;
          }
        }
        ~AdmissionScope() { runtime::active_device_resource_ledger = previous; }
      } admission;
      prepared = scf::prepare_cuda_direct_jk_resident_values(plan, maximum_cache_bytes, detail);
      if (admission.refused &&
          (prepared != GENERATIVEQC_STATUS_OUT_OF_MEMORY || admission.refused->rejected != 1 ||
           admission.refused->live != 0 || admission.refused->peak != 0))
        throw std::runtime_error("resident allocation refusal changed ownership");
    }
    if (prepared != GENERATIVEQC_STATUS_OUT_OF_MEMORY &&
        prepared != GENERATIVEQC_STATUS_NOT_IMPLEMENTED)
      check(prepared);
    timings[0] =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - prepare_started).count();
    stats = scf::cuda_direct_jk_plan_diagnostic(plan);
    counts[1] = stats.resident_value_bytes;
    counts[2] = stats.resident_value_count;
    counts[3] = density_count;
    counts[4] = counts[5] = 0;
    counts[6] = prepared;
    if (counts[1]) {
      check(scf::prepare_cuda_direct_jk_resident_values(plan, maximum_cache_bytes, detail));
      if (scf::cuda_direct_jk_plan_diagnostic(plan).device_bytes != stats.device_bytes)
        throw std::runtime_error("idempotent resident preparation changed ownership");
      const auto original_device = plan->device_id;
      plan->device_id = original_device + 1;
      const auto mismatched =
          scf::prepare_cuda_direct_jk_resident_values(plan, maximum_cache_bytes, detail);
      plan->device_id = original_device;
      if (mismatched != GENERATIVEQC_STATUS_INVALID_ARGUMENT)
        throw std::runtime_error("idempotent resident preparation accepted a foreign device");
    }
    const auto stream = scf::cuda_direct_jk_stream(plan);
    auto spec = scf::make_hf_fock_spec(unrestricted ? scf::FockSpin::Unrestricted
                                                    : scf::FockSpin::Restricted);
    spec.derivative_order = 0;
    counts[7] = scf::cuda_direct_jk_value_census_available(plan, spec);
    const std::size_t input_matrices = unrestricted ? 2 : 1;
    const std::size_t output_matrices = unrestricted ? 3 : 2;
    {
      runtime::OwnedCudaBuffer<double> storage(0, (input_matrices + output_matrices) * nn, stream);
      runtime::OwnedCudaBuffer<int> flag(0, 1, stream);
      runtime::OwnedCudaBuffer<std::uint64_t> census(0, 2, stream);
      struct Events {
        cudaEvent_t start{}, stop{};
        ~Events() {
          if (stop) (void)cudaEventDestroy(stop);
          if (start) (void)cudaEventDestroy(start);
        }
      } events;
      runtime::cuda_resource_check(cudaEventCreate(&events.start));
      runtime::cuda_resource_check(cudaEventCreate(&events.stop));
      auto* density = storage.get();
      auto* coulomb = density + input_matrices * nn;
      auto* exchange = coulomb + nn;
      for (std::size_t index = 0; index < density_count; ++index) {
        runtime::cuda_resource_check(
            cudaMemcpyAsync(density, densities + index * input_matrices * nn,
                            input_matrices * nn * sizeof(double), cudaMemcpyHostToDevice, stream));
        for (unsigned uncached = 0; uncached < 2; ++uncached) {
          const auto started = std::chrono::steady_clock::now();
          runtime::cuda_resource_check(cudaEventRecord(events.start, stream));
          if (uncached && !unrestricted)
            check(scf::enqueue_cuda_direct_jk_linear_device(
                plan, spec, density, nn, coulomb, exchange, flag.get(), 0.0, census.get(), detail));
          else {
            const auto retained = plan->resident_values;
            if (uncached) plan->resident_values = nullptr;
            try {
              check(scf::enqueue_cuda_direct_jk_device(
                  plan, spec, density, unrestricted ? density + nn : nullptr, nn, coulomb, exchange,
                  unrestricted ? exchange + nn : nullptr, flag.get(), detail,
                  counts[7] ? census.get() : nullptr));
            } catch (...) {
              plan->resident_values = retained;
              throw;
            }
            plan->resident_values = retained;
          }
          runtime::cuda_resource_check(cudaEventRecord(events.stop, stream));
          int failure = 0;
          std::array<std::uint64_t, 2> work{};
          struct DownloadDrain {
            cudaStream_t stream;
            ~DownloadDrain() {
              if (stream) (void)cudaStreamSynchronize(stream);
            }
          } download_fence{stream};
          runtime::cuda_resource_check(cudaMemcpyAsync(
              output + (index * 2 + uncached) * output_matrices * nn, coulomb,
              output_matrices * nn * sizeof(double), cudaMemcpyDeviceToHost, stream));
          runtime::cuda_resource_check(
              cudaMemcpyAsync(&failure, flag.get(), sizeof(int), cudaMemcpyDeviceToHost, stream));
          if (!uncached && counts[7])
            runtime::cuda_resource_check(cudaMemcpyAsync(work.data(), census.get(), sizeof(work),
                                                         cudaMemcpyDeviceToHost, stream));
          runtime::cuda_resource_check(cudaStreamSynchronize(stream));
          download_fence.stream = nullptr;
          if (failure) throw std::runtime_error("nonfinite resident J/K replay");
          float milliseconds = 0;
          runtime::cuda_resource_check(
              cudaEventElapsedTime(&milliseconds, events.start, events.stop));
          timings[1 + 2 * uncached] +=
              std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count();
          timings[2 + 2 * uncached] += 1e-3 * milliseconds;
          counts[4] += work[0];
          counts[5] += work[1];
        }
      }
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
