#pragma once

#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <mutex>

#include "runtime/resource_cuda.cuh"
#include "scf/cuda/direct_force_work.hpp"

namespace generativeqc::scf::cuda_execution {

/** Opt-in synchronous census, separate from clean endpoint timing. Storage is
 * charged to the active resource ledger and borrowed only by one dispatch.
 * Capture is refused before allocation: a diagnostic pointer may never escape
 * into a graph. Every exit drains before freeing, including a launch failure.
 */
class DirectForceDiagnostic {
 public:
  explicit DirectForceDiagnostic(cudaStream_t stream) : stream_(stream) {}
  DirectForceDiagnostic(const DirectForceDiagnostic&) = delete;
  DirectForceDiagnostic& operator=(const DirectForceDiagnostic&) = delete;
  ~DirectForceDiagnostic() {
    if (device_) {
      (void)cudaStreamSynchronize(stream_);
      (void)runtime::resource_cuda_free(device_);
    }
  }

  cudaError_t prepare() {
    path_ = std::getenv("GENERATIVEQC_DIRECT_FORCE_CENSUS");
    if (!path_ || !*path_) return cudaSuccess;
    cudaStreamCaptureStatus capture{};
    auto status = cudaStreamIsCapturing(stream_, &capture);
    if (status != cudaSuccess) return status;
    // CUDA's stable capture-unsupported status is 900. Compatibility headers
    // such as CuMetal retain the status ABI but may omit this enumerator name.
    if (capture != cudaStreamCaptureStatusNone) return static_cast<cudaError_t>(900);
    status = runtime::resource_cuda_malloc(reinterpret_cast<void**>(&device_), sizeof(*device_));
    if (status != cudaSuccess) return status;
    return cudaMemsetAsync(device_, 0, sizeof(*device_), stream_);
  }

  DirectForceWork* device() const noexcept { return device_; }
  void set_complete_domain(bool complete) noexcept { complete_ = complete; }
  void set_schedules(unsigned page, bool class_major, bool split_low) noexcept {
    page_ = page == 8 || page == 16 ? page : 32;
    class_major_ = class_major;
    split_low_ = split_low;
  }

  cudaError_t finish(cudaError_t launch_status) {
    if (!device_) return launch_status;
    DirectForceWork host{};
    auto status = cudaMemcpyAsync(&host, device_, sizeof(host), cudaMemcpyDeviceToHost, stream_);
    const auto drained = cudaStreamSynchronize(stream_);
    if (status == cudaSuccess) status = drained;
    if (launch_status != cudaSuccess) status = launch_status;
    for (const auto& row : host.classes)
      if (row.generated_owned) complete_ = false;
    write(host, status);
    // Release only after the download and every device borrower have retired.
    const auto released = runtime::resource_cuda_free(device_);
    device_ = nullptr;
    return status == cudaSuccess ? released : status;
  }

 private:
  cudaStream_t stream_{};
  DirectForceWork* device_{};
  const char* path_{};
  bool complete_{};
  unsigned page_{32};
  bool class_major_{}, split_low_{};
  inline static std::mutex mutex_;
  inline static std::atomic<unsigned long long> sequence_{};

  void write(const DirectForceWork& work, cudaError_t status) const {
    std::lock_guard<std::mutex> lock(mutex_);
    auto* out = std::fopen(path_, "a");
    if (!out) return;  // The evidence reader must reject a missing record.
    std::fprintf(out,
                 "{\"schema\":\"generativeqc.direct-force-census.v1\",\"id\":%llu,"
                 "\"diagnostic_only\":true,\"valid\":%s,\"cuda_error\":%d,"
                 "\"device_bytes\":%zu,\"complete_bounded_spd_domain\":%s,"
                 "\"order7_page_size\":%u,\"dddd_class_major\":%s,"
                 "\"split_low_enumeration\":%s,\"traversals\":[",
                 sequence_.fetch_add(1), status == cudaSuccess ? "true" : "false", int(status),
                 sizeof(work), complete_ ? "true" : "false", page_, class_major_ ? "true" : "false",
                 split_low_ ? "true" : "false");
    for (unsigned route = 0; route < 3; ++route) {
      const auto& row = work.traversal[route];
      std::fprintf(out,
                   "%s{\"route\":%u,\"pages\":%llu,\"inactive_pages\":%llu,"
                   "\"block_rejected_pages\":%llu,\"candidates\":%llu,"
                   "\"angular_rejected\":%llu}",
                   route ? "," : "", route, row.pages, row.inactive_pages, row.block_rejected_pages,
                   row.candidates, row.angular_rejected);
    }
    std::fputs("],\"classes\":[", out);
    for (unsigned c = 0; c < detail::kDirectQuartetShellClassCount; ++c) {
      const auto& row = work.classes[c];
      const auto& recurrence = row.recurrence;
      const auto order = detail::direct_quartet_shell_class_angular_order(c);
      // The retained order-six consumer has no unique shell census. Its
      // shared-recurrence alternative reports shells only after actual entry.
      // Low-order precontracted bilinears have no component execution count.
      auto count = [](unsigned long long value, bool available, char (&buffer)[32]) {
        if (available)
          std::snprintf(buffer, sizeof(buffer), "%llu", value);
        else
          std::snprintf(buffer, sizeof(buffer), "null");
        return buffer;
      };
      char shells[32], components[32], products[32], bra[32], ket[32], coulomb[32],
          contractions[32], published[32];
      // Only the cooperative/materialized consumers instrument recurrence
      // preparation sites. Zero must never stand for unavailable attribution.
      const bool shared_recurrence =
          complete_ && (order == 7 || order == 8 || recurrence.coulomb_preparations != 0U);
      std::fprintf(
          out,
          "%s{\"class\":%u,\"angular_order\":%zu,\"enumerated\":%llu,"
          "\"shell_screened\":%llu,\"generated_owned\":%llu,\"admitted\":%llu,"
          "\"executed_shells\":%s,\"executed_components\":%s,"
          "\"primitive_pair_products\":%s,\"bra_preparations\":%s,"
          "\"ket_preparations\":%s,\"coulomb_preparations\":%s,"
          "\"component_contractions\":%s,\"published_components\":%s}",
          c ? "," : "", c, order, row.enumerated, row.shell_screened, row.generated_owned,
          row.admitted,
          count(row.executed_shells, complete_ && (order != 6 || shared_recurrence), shells),
          count(row.executed_components, complete_ && order >= 6, components),
          count(row.primitive_pair_products, complete_, products),
          count(recurrence.bra_preparations, shared_recurrence, bra),
          count(recurrence.ket_preparations, shared_recurrence, ket),
          count(recurrence.coulomb_preparations, shared_recurrence, coulomb),
          count(recurrence.component_contractions, shared_recurrence, contractions),
          count(recurrence.published_components, shared_recurrence, published));
    }
    std::fputs("]}\n", out);
    std::fclose(out);
  }
};

}  // namespace generativeqc::scf::cuda_execution
