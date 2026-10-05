// Validation adapter for the actual native streamed CUDA source-response owner.
// Inputs may be deliberately unphysical to expose transposes and masked errors.
#include <algorithm>
#include <array>
#include <cstdio>
#include <exception>
#include <stdexcept>
#include <vector>

#include "posthf/df_mo_response.hpp"
#include "runtime/cuda_resources.cuh"
#if defined(GENERATIVEQC_TEST_HOOKS)
#include "tensor/cuda_contraction_selection.cuh"
#endif

namespace {
struct CaptureCleanup {
  cudaStream_t stream;
  ~CaptureCleanup() {
    cudaStreamCaptureStatus status{};
    if (cudaStreamIsCapturing(stream, &status) == cudaSuccess &&
        status != cudaStreamCaptureStatusNone) {
      cudaGraph_t graph{};
      (void)cudaStreamEndCapture(stream, &graph);
      if (graph) (void)cudaGraphDestroy(graph);
    }
  }
};
}  // namespace

extern "C" int df_mo_response_probe_v3(std::size_t n, std::size_t q, const double* const* input,
                                       double* const* output, std::size_t budget,
                                       std::size_t caller_bytes, int failure, int provider_test,
                                       std::size_t* counts, std::size_t counts_size, char* error,
                                       std::size_t error_size, const char* artifact) noexcept {
  using namespace generativeqc;
  try {
    if (!counts || counts_size != 16)
      throw std::invalid_argument("probe diagnostic capacity mismatch");
#if defined(GENERATIVEQC_TEST_HOOKS)
    tensor::contraction_libraries_unavailable_for_test = provider_test == 1;
    // Synthetic qualification ceilings and ranking never become production defaults.
    tensor::cutensor_reservation_for_test =
        (provider_test == 2 || provider_test == 3)
            ? tensor::ContractionProviderReservation{64ULL << 20, 256ULL << 20, 64ULL << 20}
            : tensor::ContractionProviderReservation{};
    tensor::cutensor_preparations_before_rejection_for_test = provider_test == 3 ? 2 : -1;
    tensor::cublaslt_reservation_for_test =
        (provider_test == 4 || provider_test == 5)
            ? tensor::ContractionProviderReservation{64ULL << 20, 256ULL << 20, 64ULL << 20}
            : tensor::ContractionProviderReservation{};
    tensor::cublaslt_preparations_before_rejection_for_test = provider_test == 5 ? 2 : -1;
    tensor::cutlass_region_qualification_for_test =
        (provider_test == 6 || provider_test == 7)
            ? tensor::CutlassRegionQualification{{0, 0, 1ULL << 20, 256ULL << 20},
                                                 artifact ? artifact : "",
                                                 tensor::cutlass_provider_version()}
            : tensor::CutlassRegionQualification{};
    tensor::cutlass_preparations_before_rejection_for_test = provider_test == 7 ? 2 : -1;
#else
    if (provider_test) throw std::invalid_argument("provider qualification requires test hooks");
#endif
    const auto nn = n * n, qq = q * q, full = nn * q, small = nn + qq;
    std::vector<double> bar_raw(full), bar_c(nn), bar_root(qq);
    runtime::CudaDeviceScope scope(0);
    runtime::OwnedCudaStream stream(0);
    runtime::OwnedCudaBuffer<double> data(0, 2 * full + small, stream.get());
    double* raw = data.get();
    double* c = raw + full;
    double* root = c + nn;
    double* bar = root + qq;
    const std::array<double*, 4> device{raw, c, root, bar};
    const std::array<std::size_t, 4> sizes{full, nn, qq, full};
    for (std::size_t i = 0; i < 4; ++i)
      runtime::cuda_resource_check(cudaMemcpyAsync(device[i], input[i], sizes[i] * sizeof(double),
                                                   cudaMemcpyHostToDevice, stream.get()));
    runtime::cuda_resource_check(cudaStreamSynchronize(stream.get()));
    CaptureCleanup capture{stream.get()};
    if (failure == 5)
      runtime::cuda_resource_check(
          cudaStreamBeginCapture(stream.get(), cudaStreamCaptureModeThreadLocal));
    std::size_t reads = 0, consumed = 0;
    const auto result = posthf::pullback_df_mo_source_cuda(
        {n, q, failure == 4 ? nullptr : c, root, bar}, 0, stream.get(),
        [&](std::size_t mu, double* row, cudaStream_t owner_stream) {
          if (mu != reads++ % n || owner_stream != stream.get())
            throw std::runtime_error("probe source order/stream mismatch");
          if (failure == 1 && reads == n + 1)
            throw std::runtime_error("injected second-pass source failure");
          runtime::cuda_resource_check(cudaMemcpyAsync(row, raw + mu * n * q,
                                                       n * q * sizeof(double),
                                                       cudaMemcpyDeviceToDevice, owner_stream));
        },
        [&](std::size_t mu, const double* row, cudaStream_t owner_stream) {
          if (mu != consumed++ || owner_stream != stream.get())
            throw std::runtime_error("probe response row order/stream mismatch");
          runtime::cuda_resource_check(cudaMemcpyAsync(bar_raw.data() + mu * n * q, row,
                                                       n * q * sizeof(double),
                                                       cudaMemcpyDeviceToHost, owner_stream));
          if (failure == 2)
            throw std::runtime_error("injected consume failure after queued download");
        },
        [&](const double* dc, const double* dw, cudaStream_t owner_stream) {
          runtime::cuda_resource_check(cudaMemcpyAsync(bar_c.data(), dc, nn * sizeof(double),
                                                       cudaMemcpyDeviceToHost, owner_stream));
          runtime::cuda_resource_check(cudaMemcpyAsync(bar_root.data(), dw, qq * sizeof(double),
                                                       cudaMemcpyDeviceToHost, owner_stream));
          if (failure == 3)
            throw std::runtime_error("injected finish failure after queued downloads");
        },
        budget, caller_bytes + sizeof(double) * (5 * full + 3 * small));
    const std::array<const std::vector<double>*, 3> values{&bar_raw, &bar_c, &bar_root};
    for (std::size_t i = 0; i < 3; ++i) std::copy(values[i]->begin(), values[i]->end(), output[i]);
    const std::size_t work[]{result.source_rows,
                             result.source_values,
                             result.output_rows,
                             result.output_values,
                             result.gemms,
                             result.contraction_summands,
                             result.owned_device_bytes,
                             result.numeric_capacity_bytes,
                             result.scalar_d2h_bytes,
                             result.binding_bytes,
                             result.provider_version,
                             result.prepared_contractions,
                             result.provider == "cublas"           ? 0UL
                             : result.provider == "generated.cuda" ? 1UL
                             : result.provider == "cublaslt"       ? 3UL
                             : result.provider == "cutlass-aot"    ? 4UL
                                                                   : 2UL,
                             result.preparation_ns,
                             result.optional_workspace_bytes,
                             result.observed_provider_bytes};
    if (result.candidate_identity.size() != 64)
      throw std::runtime_error("missing selected candidate identity");
    std::copy(std::begin(work), std::end(work), counts);
    return 0;
  } catch (const std::exception& e) {
    if (error && error_size) std::snprintf(error, error_size, "%s", e.what());
    return 1;
  }
}
