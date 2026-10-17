/** Diagnostic-only ELF interposer for RHF provider-retained memory.
 *
 * This shared object forwards every argument, output pointer and return code to
 * the next CUDA/cuBLAS/cuSOLVER definition. It adds no CUDA call, allocation,
 * synchronization, context initialization, provider warmup or retry. The only
 * side effect is a bounded host JSONL journal at the fresh path selected by
 * GENERATIVEQC_RHF_PROVIDER_TRACE. Its output is attribution evidence, never
 * scientific qualification or performance timing.
 */

#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif

#include <cublas_v2.h>
#include <cuda_runtime_api.h>
#include <cusolverDn.h>
#include <dlfcn.h>
#include <fcntl.h>
#include <sys/stat.h>
#include <sys/syscall.h>
#include <unistd.h>

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>

namespace {

constexpr std::uint64_t kMaximumEvents = 512;
constexpr int kUninitialized = -2;
constexpr int kInitializing = -3;
std::atomic<int> journal_fd{kUninitialized};
std::atomic<std::uint64_t> next_sequence{};
std::atomic<std::uint64_t> events_written{};
std::atomic<std::uint64_t> dropped_events{};
std::atomic<std::uint64_t> resolution_errors{};
std::atomic<std::uint64_t> write_errors{};
thread_local unsigned provider_depth{};

std::uint64_t process_id() noexcept { return static_cast<std::uint64_t>(::getpid()); }

std::uint64_t thread_id() noexcept { return static_cast<std::uint64_t>(::syscall(SYS_gettid)); }

int open_journal() noexcept {
  int observed = journal_fd.load(std::memory_order_acquire);
  if (observed != kUninitialized && observed != kInitializing) return observed;
  int expected = kUninitialized;
  if (journal_fd.compare_exchange_strong(expected, kInitializing, std::memory_order_acq_rel)) {
    const char* path = std::getenv("GENERATIVEQC_RHF_PROVIDER_TRACE");
    const int descriptor =
        path && *path
            ? ::open(path, O_WRONLY | O_CREAT | O_EXCL | O_APPEND | O_CLOEXEC, S_IRUSR | S_IWUSR)
            : -1;
    journal_fd.store(descriptor, std::memory_order_release);
    return descriptor;
  }
  while ((observed = journal_fd.load(std::memory_order_acquire)) == kInitializing) {
    ::syscall(SYS_sched_yield);
  }
  return observed;
}

void write_record(const char* data, std::size_t size) noexcept {
  const int descriptor = open_journal();
  if (descriptor < 0) return;
  const auto written = ::write(descriptor, data, size);
  if (written != static_cast<ssize_t>(size)) {
    write_errors.fetch_add(1, std::memory_order_relaxed);
  }
}

void event(const char* name, const char* phase, std::int64_t status, const char* origin,
           bool observed = false, std::uint64_t free_bytes = 0,
           std::uint64_t total_bytes = 0) noexcept {
  const auto sequence = next_sequence.fetch_add(1, std::memory_order_relaxed);
  if (sequence >= kMaximumEvents) {
    dropped_events.fetch_add(1, std::memory_order_relaxed);
    return;
  }
  char line[768]{};
  const int length = std::snprintf(
      line, sizeof(line),
      "{\"schema\":\"generativeqc.rhf-provider-probe.v1\",\"sequence\":%llu,"
      "\"pid\":%llu,\"tid\":%llu,\"event\":\"%s\",\"phase\":\"%s\","
      "\"status\":%lld,\"origin\":\"%s\",\"observed\":%s,"
      "\"free_bytes\":%llu,\"total_bytes\":%llu}\n",
      static_cast<unsigned long long>(sequence), static_cast<unsigned long long>(process_id()),
      static_cast<unsigned long long>(thread_id()), name, phase, static_cast<long long>(status),
      origin, observed ? "true" : "false", static_cast<unsigned long long>(free_bytes),
      static_cast<unsigned long long>(total_bytes));
  if (length <= 0 || static_cast<std::size_t>(length) >= sizeof(line)) {
    write_errors.fetch_add(1, std::memory_order_relaxed);
    return;
  }
  write_record(line, static_cast<std::size_t>(length));
  events_written.fetch_add(1, std::memory_order_relaxed);
}

template <class Function>
Function resolve(const char* name, const char* version) noexcept {
  (void)::dlerror();
  auto* symbol = ::dlvsym(RTLD_NEXT, name, version);
  if (!symbol || ::dlerror()) {
    resolution_errors.fetch_add(1, std::memory_order_relaxed);
    event("resolution_error", "end", -1, "interposer");
    return nullptr;
  }
  return reinterpret_cast<Function>(symbol);
}

struct ProviderDepth {
  ProviderDepth() noexcept { ++provider_depth; }
  ~ProviderDepth() { --provider_depth; }
};

template <class Status, class Function, class... Args>
Status provider_call(const char* name, Status unavailable, Function function,
                     Args... args) noexcept {
  event(name, "begin", 0, "application");
  if (!function) {
    event(name, "end", static_cast<std::int64_t>(unavailable), "application");
    return unavailable;
  }
  Status status{};
  {
    ProviderDepth depth;
    status = function(args...);
  }
  event(name, "end", static_cast<std::int64_t>(status), "application");
  return status;
}

__attribute__((destructor)) void finish_journal() noexcept {
  const int descriptor = open_journal();
  if (descriptor < 0) return;
  char line[512]{};
  const auto sequence = next_sequence.load(std::memory_order_relaxed);
  const int length = std::snprintf(
      line, sizeof(line),
      "{\"schema\":\"generativeqc.rhf-provider-probe.v1\",\"sequence\":%llu,"
      "\"pid\":%llu,\"tid\":%llu,\"event\":\"terminal\",\"phase\":\"end\","
      "\"status\":0,\"events_written\":%llu,\"dropped_events\":%llu,"
      "\"resolution_errors\":%llu,\"write_errors\":%llu}\n",
      static_cast<unsigned long long>(sequence), static_cast<unsigned long long>(process_id()),
      static_cast<unsigned long long>(thread_id()),
      static_cast<unsigned long long>(events_written.load(std::memory_order_relaxed)),
      static_cast<unsigned long long>(dropped_events.load(std::memory_order_relaxed)),
      static_cast<unsigned long long>(resolution_errors.load(std::memory_order_relaxed)),
      static_cast<unsigned long long>(write_errors.load(std::memory_order_relaxed)));
  if (length > 0 && static_cast<std::size_t>(length) < sizeof(line)) {
    write_record(line, static_cast<std::size_t>(length));
  }
  (void)::close(descriptor);
  journal_fd.store(-1, std::memory_order_release);
}

}  // namespace

extern "C" cudaError_t cudaMemGetInfo(std::size_t* free_bytes, std::size_t* total_bytes) {
  using Function = cudaError_t (*)(std::size_t*, std::size_t*);
  static Function real = resolve<Function>("cudaMemGetInfo", "libcudart.so.12");
  if (!real) return cudaErrorUnknown;
  const auto status = real(free_bytes, total_bytes);
  const bool observed = status == cudaSuccess && free_bytes && total_bytes;
  event("cuda_mem_get_info", "end", static_cast<std::int64_t>(status),
        provider_depth ? "provider_internal" : "application", observed, observed ? *free_bytes : 0,
        observed ? *total_bytes : 0);
  return status;
}

extern "C" cublasStatus_t cublasCreate_v2(cublasHandle_t* handle) {
  using Function = cublasStatus_t (*)(cublasHandle_t*);
  static Function real = resolve<Function>("cublasCreate_v2", "libcublas.so.12");
  return provider_call("cublas_create", CUBLAS_STATUS_INTERNAL_ERROR, real, handle);
}

extern "C" cublasStatus_t cublasSetStream_v2(cublasHandle_t handle, cudaStream_t stream) {
  using Function = cublasStatus_t (*)(cublasHandle_t, cudaStream_t);
  static Function real = resolve<Function>("cublasSetStream_v2", "libcublas.so.12");
  return provider_call("cublas_set_stream", CUBLAS_STATUS_INTERNAL_ERROR, real, handle, stream);
}

extern "C" cublasStatus_t cublasSetPointerMode_v2(cublasHandle_t handle, cublasPointerMode_t mode) {
  using Function = cublasStatus_t (*)(cublasHandle_t, cublasPointerMode_t);
  static Function real = resolve<Function>("cublasSetPointerMode_v2", "libcublas.so.12");
  return provider_call("cublas_set_pointer_mode", CUBLAS_STATUS_INTERNAL_ERROR, real, handle, mode);
}

extern "C" cusolverStatus_t cusolverDnCreate(cusolverDnHandle_t* handle) {
  using Function = cusolverStatus_t (*)(cusolverDnHandle_t*);
  static Function real = resolve<Function>("cusolverDnCreate", "libcusolver.so.11");
  return provider_call("cusolver_create", CUSOLVER_STATUS_INTERNAL_ERROR, real, handle);
}

extern "C" cusolverStatus_t cusolverDnSetStream(cusolverDnHandle_t handle, cudaStream_t stream) {
  using Function = cusolverStatus_t (*)(cusolverDnHandle_t, cudaStream_t);
  static Function real = resolve<Function>("cusolverDnSetStream", "libcusolver.so.11");
  return provider_call("cusolver_set_stream", CUSOLVER_STATUS_INTERNAL_ERROR, real, handle, stream);
}

extern "C" cusolverStatus_t cusolverDnCreateParams(cusolverDnParams_t* parameters) {
  using Function = cusolverStatus_t (*)(cusolverDnParams_t*);
  static Function real = resolve<Function>("cusolverDnCreateParams", "libcusolver.so.11");
  return provider_call("cusolver_create_params", CUSOLVER_STATUS_INTERNAL_ERROR, real, parameters);
}
