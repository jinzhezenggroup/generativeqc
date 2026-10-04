// Validation-only ELF interposer. The production library remains byte-identical
// to its build. Copy the metric root and raw tiles actually consumed by the
// frozen-frame source, without changing their arithmetic or device contents.
// The extra synchronization makes these runs unsuitable for timing claims.
#include <cublas_v2.h>
#include <dlfcn.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <vector>

#include "scf/cuda/df_plan_internal.hpp"

namespace {
std::size_t n = 0, q = 0, plans = 0, tiles = 0, whitening_calls = 0;
std::vector<double> metric, root, eigenvalues, raw, raw_low, bmo;
std::vector<std::size_t> visits;
std::string failure;
generativeqc::scf::CudaDensityFittingSourceCounters source_counters;

template <class Function>
Function next(const char* name) {
  auto* symbol = dlsym(RTLD_NEXT, name);
  if (!symbol) {
    std::fprintf(stderr, "missing production symbol: %s\n", dlerror());
    std::abort();
  }
  return reinterpret_cast<Function>(symbol);
}

void copy_device(double* dst, const double* src, std::size_t count, cudaStream_t stream) {
  auto status = cudaMemcpyAsync(dst, src, count * sizeof(double), cudaMemcpyDeviceToHost, stream);
  if (status == cudaSuccess) status = cudaStreamSynchronize(stream);
  if (status != cudaSuccess) throw std::runtime_error(cudaGetErrorString(status));
}
}  // namespace

namespace generativeqc::scf {
generativeqc_status create_cuda_density_fitting_jk_plan_from_source(
    int device, CudaDensityFittingIntegralSource** source, std::size_t batch, std::size_t nbf,
    std::size_t naux, const std::vector<double>& metrics, double relative_threshold,
    std::size_t aux_tile, std::size_t pair_tile, CudaDensityFittingJkPlan** plan,
    std::vector<CudaDensityFittingMetricDiagnostic>& diagnostics, std::string& detail,
    bool retain) {
  using Fn = generativeqc_status (*)(
      int, CudaDensityFittingIntegralSource**, std::size_t, std::size_t, std::size_t,
      const std::vector<double>&, double, std::size_t, std::size_t, CudaDensityFittingJkPlan**,
      std::vector<CudaDensityFittingMetricDiagnostic>&, std::string&, bool);
  Dl_info symbol{};
  if (!dladdr(reinterpret_cast<void*>(
                  static_cast<Fn>(&create_cuda_density_fitting_jk_plan_from_source)),
              &symbol))
    std::abort();
  static auto call = next<Fn>(symbol.dli_sname);
  const auto status = call(device, source, batch, nbf, naux, metrics, relative_threshold, aux_tile,
                           pair_tile, plan, diagnostics, detail, retain);
  if (status == GENERATIVEQC_STATUS_SUCCESS && batch == 1 && nbf == 230 && naux == 488) {
    try {
      if (++plans != 1) throw std::runtime_error("unexpected second metric owner");
      n = nbf;
      q = naux;
      metric = metrics;
      root.resize(q * q);
      raw.assign(n * n * q, std::numeric_limits<double>::quiet_NaN());
      raw_low.assign(n * n * q, std::numeric_limits<double>::quiet_NaN());
      visits.assign(n * n, 0);
      copy_device(root.data(), (*plan)->inverse_square_roots, q * q, (*plan)->stream);
#ifndef BASELINE_NO_RETAINED_EIGENVALUES
      eigenvalues.resize(q);
      copy_device(eigenvalues.data(), (*plan)->metric_eigenvalues, q, (*plan)->stream);
#endif
    } catch (const std::exception& e) {
      failure = e.what();
    }
  }
  return status;
}

generativeqc_status generate_cuda_density_fitting_raw_expansion(
    CudaDensityFittingIntegralSource* source, std::size_t system, std::size_t pair_begin,
    std::size_t pair_count, std::size_t aux_begin, std::size_t aux_count, void* stream,
    double* output, double* low, std::string& detail) {
  using Fn = decltype(&generate_cuda_density_fitting_raw_expansion);
  Dl_info symbol{};
  if (!dladdr(reinterpret_cast<void*>(&generate_cuda_density_fitting_raw_expansion), &symbol))
    std::abort();
  static auto call = next<Fn>(symbol.dli_sname);
  auto status = call(source, system, pair_begin, pair_count, aux_begin, aux_count, stream, output,
                     low, detail);
  if (status == GENERATIVEQC_STATUS_SUCCESS && n && failure.empty()) {
    try {
      if (system || aux_begin || aux_count != q || pair_begin + pair_count > n * n)
        throw std::runtime_error("unexpected frozen-source expansion layout");
      copy_device(raw.data() + pair_begin * q, output, pair_count * q,
                  static_cast<cudaStream_t>(stream));
      copy_device(raw_low.data() + pair_begin * q, low, pair_count * q,
                  static_cast<cudaStream_t>(stream));
      for (std::size_t i = pair_begin; i < pair_begin + pair_count; ++i) ++visits[i];
      ++tiles;
      source_counters = cuda_density_fitting_integral_source_counters(source);
      if (source_counters.generated_value_tiles != tiles ||
          source_counters.generated_value_bytes !=
              (pair_begin + pair_count) * q * 2 * sizeof(double))
        throw std::runtime_error("raw expansion traffic counters omit a component or repeat work");
    } catch (const std::exception& e) {
      failure = e.what();
    }
  }
  return status;
}
}  // namespace generativeqc::scf

// Capture the actual final transform before the pre-existing physical pair
// projection. This prevents projection from concealing a factor failure.
extern "C" cublasStatus_t cublasDgemm_v2(cublasHandle_t h, cublasOperation_t ta,
                                         cublasOperation_t tb, int m, int cols, int k,
                                         const double* alpha, const double* a, int lda,
                                         const double* b, int ldb, const double* beta, double* c,
                                         int ldc) {
  static auto call = next<decltype(&cublasDgemm_v2)>("cublasDgemm_v2");
  auto status = call(h, ta, tb, m, cols, k, alpha, a, lda, b, ldb, beta, c, ldc);
  if (status == CUBLAS_STATUS_SUCCESS && n && m == (int)q && cols == (int)(n * n) && k == (int)q) {
    try {
      ++whitening_calls;
      bmo.resize(n * n * q);
      cudaStream_t stream;
      if (cublasGetStream(h, &stream) != CUBLAS_STATUS_SUCCESS)
        throw std::runtime_error("missing BLAS stream");
      copy_device(bmo.data(), c, bmo.size(), stream);
    } catch (const std::exception& e) {
      failure = e.what();
    }
  }
  return status;
}

// Copies are explicit so Python owns snapshots after production teardown.
extern "C" int copy_frozen_capture(double* m, double* w, double* e, double* a, double* al,
                                   double* b, std::size_t* counts, char* error) {
  if (failure.empty() &&
      (plans != 1 || tiles != 230 || whitening_calls != 1 || visits.size() != 230 * 230 ||
       !std::all_of(visits.begin(), visits.end(), [](auto x) { return x == 1; }) ||
       !std::all_of(raw.begin(), raw.end(), [](double x) { return std::isfinite(x); }) ||
       !std::all_of(raw_low.begin(), raw_low.end(), [](double x) { return std::isfinite(x); })))
    failure = "incomplete or duplicate production snapshot";
  if (!failure.empty()) {
    std::snprintf(error, 4096, "%s", failure.c_str());
    return 1;
  }
  std::copy(metric.begin(), metric.end(), m);
  std::copy(root.begin(), root.end(), w);
  std::copy(eigenvalues.begin(), eigenvalues.end(), e);
  std::copy(raw.begin(), raw.end(), a);
  std::copy(raw_low.begin(), raw_low.end(), al);
  std::copy(bmo.begin(), bmo.end(), b);
  counts[0] = plans;
  counts[1] = tiles;
  counts[2] = n;
  counts[3] = q;
  counts[4] = whitening_calls;
  counts[5] = source_counters.generated_value_bytes;
  counts[6] = source_counters.generated_value_tiles;
  return 0;
}
