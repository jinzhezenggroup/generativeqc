#include <dlfcn.h>
#include <link.h>

#include <array>
#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <thread>

#include "model/gfn2/eigensolver.hpp"
#include "tensor/cpu/lp64_provider.hpp"
namespace cpu = generativeqc::tensor::cpu;
static std::atomic<int> opens{}, closes{}, namespace_opens{};
static const char* first_path;
static const char* fallback_path;
static bool configured_first, private_mode;
static void require(bool ok, const char* detail) {
  if (!ok) {
    std::fprintf(stderr, "%s\n", detail);
    std::abort();
  }
}
extern "C" void* __real_dlopen(const char*, int);
extern "C" int __real_dlclose(void*);
extern "C" void* __real_dlmopen(Lmid_t, const char*, int);
extern "C" void* __wrap_dlopen(const char* name, int flags) {
  int call = opens++;
  require(flags == (RTLD_NOW | RTLD_LOCAL), "runtime dlopen flags changed");
  if (configured_first && call == 0) {
    require(std::strcmp(name, "/configured/reviewed-runtime.so") == 0,
            "configured runtime lost precedence");
    return __real_dlopen(first_path, flags);
  }
  return __real_dlopen(fallback_path, flags);
}
extern "C" int __wrap_dlclose(void* handle) {
  ++closes;
  return __real_dlclose(handle);
}
extern "C" void* __wrap_dlmopen(Lmid_t id, const char* name, int flags) {
  ++namespace_opens;
  require(id == LM_ID_NEWLM && flags == (RTLD_NOW | RTLD_LOCAL),
          "private namespace admission changed");
  void* handle = __real_dlmopen(id, name, flags);
  if (handle) {
    Lmid_t actual = LM_ID_BASE;
    require(dlinfo(handle, RTLD_DI_LMID, &actual) == 0 && actual != LM_ID_BASE,
            "private namespace not isolated");
  }
  return handle;
}
int main(int argc, char** argv) {
  require(argc == 6, "arguments: configured/private expected first fallback");
  configured_first = std::strcmp(argv[1], "configured") == 0;
  private_mode = std::strcmp(argv[1], "private") == 0;
  const bool expected = std::atoi(argv[2]) != 0;
  first_path = argv[3];
  fallback_path = argv[4];
  void* host = nullptr;
  if (std::strcmp(argv[5], "none") != 0) {
    host = __real_dlopen(argv[5], RTLD_NOW | RTLD_GLOBAL);
    require(host != nullptr, "host poison load failed");
  }
  cpu::CpuLinearAlgebraBackend outputs[12];
  std::array<std::thread, 12> threads;
  for (int i = 0; i < 12; ++i)
    threads[i] = std::thread([&, i] {
      std::string error;
      const auto status = cpu::prepare_lp64_runtime_backend(outputs[i], error);
      require((status == cpu::Lp64BackendStatus::success) == expected, "cohort admission changed");
      require(error.empty() == expected, "factory error publication changed");
    });
  for (auto& thread : threads) thread.join();
  const auto before = opens.load(), namespaces = namespace_opens.load(), closed = closes.load();
  std::string error;
  auto backend = outputs[0];
  const auto method_status =
      generativeqc::xtb::detail::gfn2::make_mkl_rt_lp64_backend(backend, error);
  require(method_status == (expected ? GENERATIVEQC_XTB_STATUS_SUCCESS
                                     : GENERATIVEQC_XTB_STATUS_BACKEND_UNAVAILABLE),
          "method status mapping");
  require(opens == before && namespace_opens == namespaces && closes == closed,
          "factory reloaded after cache");
  for (auto& item : outputs)
    require(item.ready() == expected, "inconsistent concurrent publication");
  if (private_mode) {
    require(namespaces == 1 && opens == 0, "configured private provider fell back");
    require(closed == (expected ? 0 : 1), "private handle lifetime changed");
  } else {
    require(namespaces == 0 && before >= 1, "ordinary runtime discovery changed");
    require(closed == before - (expected ? 1 : 0),
            "successful handle not retained/rejected handle not closed");
  }
  if (expected) {
    require(backend.production() && !backend.production_mkl(), "wrong production origin");
    require(backend.production_openblas_isolated() == private_mode, "wrong namespace provenance");
    auto saved = backend;
    {
      auto copied = saved;
      require(copied.ready(), "copy dropped dispatch");
    }
    require(closes == closed, "backend copy lifetime unloaded provider");
    double a = 4, w = 0, work[9]{};
    std::int32_t iw[8]{};
    auto eigen = cpu::bind_symmetric_eigen(saved);
    require(eigen(102, 'V', 'L', 1, &a, 1, &w, work, 9, iw, 8) == 0 && a == 1 && w == 4,
            "retained dispatch failed");
    if (host) {
      auto host_call = dlsym(host, "scipy_LAPACKE_dpotrf_work");
      require(host_call != nullptr, "missing host poison symbol");
      // Private success despite a globally visible rejecting cohort, plus the
      // observed non-base link map, proves the current Linux isolation route.
    }
  }
  std::printf("admitted=%d opens=%d closes=%d namespaces=%d cached=12+1\n", int(expected), before,
              closed, namespaces);
}
