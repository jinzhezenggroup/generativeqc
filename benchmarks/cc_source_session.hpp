#pragma once
#include <cuda_runtime_api.h>
#include <dlfcn.h>
#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>

#include <array>
#include <cerrno>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>
#include <utility>

#include "cc/phase_observation_contract.hpp"

extern "C" int generativeqc_residency_observer_bind_v1(
    generativeqc::cc::phase_observation::Observer, void*);
extern "C" int generativeqc_residency_observer_unbind_v1(
    generativeqc::cc::phase_observation::Observer, void*, std::uint64_t*);

namespace generativeqc::benchmarks::cc_trace {
namespace phase_observation = generativeqc::cc::phase_observation;
class AtomicFile {
 public:
  explicit AtomicFile(std::string target)
      : target_(std::move(target)), pending_(target_ + ".pending-" + std::to_string(getpid())) {
    fd_ = open(pending_.c_str(), O_WRONLY | O_CREAT | O_EXCL | O_NOFOLLOW | O_CLOEXEC, 0600);
    if (fd_ < 0) throw std::runtime_error("cannot create exclusive CC trace output");
  }
  ~AtomicFile() {
    if (fd_ >= 0) close(fd_);
  }  // Preserve failed pending data, never overwrite/delete old evidence.
  int descriptor() const noexcept { return fd_; }
  bool commit() noexcept {
    if (fsync(fd_) != 0) return false;
    const int closing = close(fd_);
    fd_ = -1;
    if (closing != 0) return false;
    if (link(pending_.c_str(), target_.c_str()) != 0) return false;
    if (unlink(pending_.c_str()) != 0) return false;
    const auto slash = target_.rfind('/');
    const auto directory = target_.substr(0, slash);
    const int dir = open(directory.c_str(), O_RDONLY | O_DIRECTORY | O_CLOEXEC);
    if (dir < 0) return false;
    const bool ok = fsync(dir) == 0;
    close(dir);
    return ok;
  }

 private:
  std::string target_, pending_;
  int fd_ = -1;
};

/** Optional standalone collector. Production defaults do not load this DSO.
 * The module is pinned until process exit because CUPTI may retain callback
 * code addresses after stop. No source context/code pointer is guessed. */
class Session {
 public:
  Session(const char* backend, const char* prefix) {
    if (!backend && !prefix) return;
    if (!backend || !prefix || backend[0] != '/' || prefix[0] != '/')
      throw std::invalid_argument("CC trace needs absolute backend/output paths");
    prefix_ = prefix;
    if (lstat(backend, &backend_identity_) != 0 || !S_ISREG(backend_identity_.st_mode))
      throw std::invalid_argument("CC backend must be an exact regular file");
    module_ = dlopen(backend, RTLD_NOW | RTLD_LOCAL | RTLD_NODELETE);
    if (!module_) throw std::runtime_error("cannot load pinned CC trace backend");
    const auto activity_schema =
        symbol<std::uint64_t (*)(std::uint64_t)>("generativeqc_cc_cupti_schema_v1");
    const auto source_schema =
        symbol<std::uint64_t (*)(std::uint64_t)>("generativeqc_cc_source_schema_v1");
    if (activity_schema(0) != 1 || activity_schema(1) != 20 || activity_schema(2) != 14 ||
        activity_schema(3) != 28 || source_schema(0) != 1 || source_schema(1) != 19 ||
        source_schema(2) != 24 || source_schema(3) != phase_observation::kDetailedContract)
      throw std::runtime_error("CC backend schema/version mismatch");

    begin_ = symbol<Begin>("generativeqc_cc_cupti_begin_v1");
    stop_ = symbol<Stop>("generativeqc_cc_cupti_stop_v1");
    read_ = symbol<Read>("generativeqc_cc_cupti_read_v1");
    activity_export_ = symbol<Export>("generativeqc_cc_cupti_export_v1");
    source_begin_ = symbol<SourceBegin>("generativeqc_cc_source_begin_v1");
    set_permit_ = symbol<Begin>("generativeqc_cc_source_permit_v1");
    source_seal_ = symbol<Seal>("generativeqc_cc_source_seal_v1");
    source_read_ = symbol<SourceRead>("generativeqc_cc_source_read_v1");
    source_export_ = symbol<Export>("generativeqc_cc_source_export_v1");
    legacy_export_ = symbol<LegacyExport>("generativeqc_cc_legacy_export_v1");
    legacy_source_read_ = symbol<Read>("generativeqc_cupti_source_read_v1");
    legacy_graph_read_ = symbol<Read>("generativeqc_cupti_graph_read_v1");
    callback_ = symbol<phase_observation::Observer>("generativeqc_cc_source_observe_v1");
    push_ = symbol<Begin>("generativeqc_cc_cupti_push_v1");
    pop_ = symbol<Begin>("generativeqc_cc_cupti_pop_v1");
    if (begin_(1u << 21) != 0) throw std::runtime_error("CC activity initialization failed");
    try {
      if (source_begin_(1u << 18, phase_observation::kDetailedContract) != 0)
        throw std::runtime_error("CC source initialization failed");
      if (generativeqc_residency_observer_bind_v1(callback_, nullptr) != 0)
        throw std::runtime_error("CC observer owner unavailable");
      bound_ = true;
      if (generativeqc_cc_phase_enable_v1(phase_observation::kDetailedContract, callback_, nullptr,
                                          &permit_) != 0)
        throw std::runtime_error("CC detailed source contract unavailable");
      if (set_permit_(permit_) != 0) throw std::runtime_error("CC source permit mismatch");
      active_ = true;
    } catch (...) {
      // A throwing constructor has no Session destructor. Release this
      // partially acquired binding before propagating the original error.
      std::uint64_t errors = 0;
      if (permit_) (void)generativeqc_cc_phase_finish_v1(permit_, &errors);
      if (bound_) {
        (void)generativeqc_residency_observer_unbind_v1(callback_, nullptr, &errors);
        bound_ = false;
      }
      (void)stop_();
      throw;
    }
  }
  Session(const Session&) = delete;
  Session& operator=(const Session&) = delete;
  ~Session() {
    if (active_) try {
        (void)finish(1);
      } catch (...) {
      }
  }
  bool finish(std::uint64_t science_result = 0) noexcept {
    if (!active_) return true;
    active_ = false;
    try {
      std::uint64_t errors = 0, dispatch_errors = 0;
      const int phase_status = generativeqc_cc_phase_finish_v1(permit_, &errors);
      const int detach_status =
          generativeqc_residency_observer_unbind_v1(callback_, nullptr, &dispatch_errors);
      if (!detach_status) bound_ = false;
      // Caller has left the scientific resource/timing block. This is an
      // explicitly observer-owned completion fence, never hidden in science.
      constexpr std::uint64_t fence_region = 0x8000000000000001ULL;
      const int pushed = push_(fence_region);
      auto synchronize = reinterpret_cast<decltype(&cudaDeviceSynchronize)>(
          dlsym(RTLD_DEFAULT, "cudaDeviceSynchronize"));
      const int fenced = synchronize ? static_cast<int>(synchronize()) : -1;
      const int popped = pop_(fence_region);
      if (pushed || fenced || popped) errors |= 1;
      const int seal_status =
          source_seal_(phase_status, errors, detach_status, dispatch_errors, science_result);
      const int stop_status = stop_();
      std::array<std::uint64_t, 14> activity_before{}, activity_after{};
      std::array<std::uint64_t, 24> source_before{}, source_after{};
      if (read_(nullptr, 0, activity_before.data(), 14) || source_read_(source_before.data(), 24))
        return false;
      bool written = publish(".activity.bin", [&](int fd) { return activity_export_(fd); });
      written = publish(".source.bin", [&](int fd) { return source_export_(fd); }) && written;
      written = publish(".graph.bin", [&](int fd) { return legacy_export_(fd, 1); }) && written;
      written = publish(".legacy.bin", [&](int fd) { return legacy_export_(fd, 2); }) && written;
      if (read_(nullptr, 0, activity_after.data(), 14) || source_read_(source_after.data(), 24) ||
          activity_before != activity_after || source_before != source_after)
        return false;
      std::array<std::uint64_t, 8> source_legacy{}, graph_legacy{};
      if (legacy_source_read_(nullptr, 0, source_legacy.data(), 8) ||
          legacy_graph_read_(nullptr, 0, graph_legacy.data(), 8))
        return false;
      bool clean = written && !phase_status && !detach_status && !errors && !dispatch_errors &&
                   !science_result && !seal_status && !stop_status;
      for (auto index : {2, 3, 4, 5, 6, 9, 12, 13}) clean = clean && activity_after[index] == 0;
      for (auto index : {2, 3, 4, 5, 6, 12, 13, 14, 15, 16, 20, 21})
        clean = clean && source_after[index] == 0;
      clean = clean && activity_after[7] == 28 && activity_after[8] == 1 && source_after[19] == 1;
      clean = clean && activity_after[10] > 0;
      clean = clean && !source_legacy[2] && !source_legacy[3] && !source_legacy[4] &&
              source_legacy[5] == 1;
      clean =
          clean && !graph_legacy[2] && !graph_legacy[3] && !graph_legacy[7] && graph_legacy[6] == 1;
      return clean;
    } catch (...) {
      return false;
    }
  }

 private:
  using Begin = int (*)(std::uint64_t);
  using Stop = int (*)();
  using Read = int (*)(std::uint64_t*, std::uint64_t, std::uint64_t*, std::uint64_t);
  using SourceBegin = int (*)(std::uint64_t, std::uint64_t);
  using SourceRead = int (*)(std::uint64_t*, std::uint64_t);
  using Seal = int (*)(std::uint64_t, std::uint64_t, std::uint64_t, std::uint64_t, std::uint64_t);
  using Export = int (*)(int);
  using LegacyExport = int (*)(int, std::uint64_t);
  template <class Function>
  Function symbol(const char* name) {
    auto* pointer = dlsym(module_, name);
    if (!pointer) throw std::runtime_error("missing CC backend ABI");
    Dl_info info{};
    struct stat actual{};
    if (!dladdr(pointer, &info) || !info.dli_fname || stat(info.dli_fname, &actual) != 0 ||
        actual.st_dev != backend_identity_.st_dev || actual.st_ino != backend_identity_.st_ino)
      throw std::runtime_error("CC backend symbol resolved to another module");
    return reinterpret_cast<Function>(pointer);
  }
  template <class Write>
  bool publish(const char* suffix, Write write) {
    AtomicFile file(prefix_ + suffix);
    return write(file.descriptor()) == 0 && file.commit();
  }
  void* module_{};
  struct stat backend_identity_{};
  std::string prefix_;
  std::uint64_t permit_{};
  bool bound_{}, active_{};
  Begin begin_{}, set_permit_{}, push_{}, pop_{};
  Stop stop_{};
  Read read_{}, legacy_source_read_{}, legacy_graph_read_{};
  SourceBegin source_begin_{};
  SourceRead source_read_{};
  Seal source_seal_{};
  Export activity_export_{}, source_export_{};
  LegacyExport legacy_export_{};
  phase_observation::Observer callback_{};
};
}  // namespace generativeqc::benchmarks::cc_trace
