#include "tensor/cpu/lp64_provider.hpp"
// Adapted from xTBloom's retained CPU provider. The scoped CUDA/MKL linking
// permission in src/xtb/native/CUDA_MKL_LINKING_EXCEPTION still applies.

// Preserve the inherited platform guards and loader body for auditable transfer.
// clang-format off
#if defined(GENERATIVEQC_XTB_CONFIGURED_CPU_LINALG_SHIM) && defined(__linux__)
#include "runtime/mkl_pthread_tss_bridge.h"
#endif

#if defined(_WIN32)
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#else
#include <dlfcn.h>
#endif

#if (defined(GENERATIVEQC_XTB_CONFIGURED_CPU_LINALG_SHIM) || defined(GENERATIVEQC_XTB_CONFIGURED_WHEEL_OPENBLAS)) && \
    defined(__linux__)
#include <link.h>
#endif


#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <new>
#include <type_traits>
#include <utility>

namespace generativeqc::tensor::cpu {
namespace {
static_assert(sizeof(LapackInt) == 4u, "the CPU provider requires an LP64 LAPACK ABI");
constexpr int kCblasColMajor = 102;
constexpr int kCblasNoTrans = 111;
constexpr int kCblasTrans = 112;
constexpr int kCblasLower = 122;
constexpr int kCblasNonUnit = 131;
constexpr int kCblasLeft = 141;
constexpr char kLower = 'L';
constexpr char kEigenvectors = 'V';

template <typename Function>
bool load_symbol(void* handle, const char* name, Function& function) {
#if defined(_WIN32)
  static_assert(sizeof(Function) == sizeof(FARPROC));
  const FARPROC symbol = GetProcAddress(static_cast<HMODULE>(handle), name);
  if (symbol == nullptr) {
    return false;
  }
#else
  static_assert(sizeof(Function) == sizeof(void*));
  dlerror();
  void* symbol = dlsym(handle, name);
  if (symbol == nullptr || dlerror() != nullptr) {
    return false;
  }
#endif
  std::memcpy(&function, &symbol, sizeof(function));
  return true;
}

bool load_lapacke_cblas_symbols(void* handle, bool scipy_prefix, LapackDpotrfWork& dpotrf_work,
                                LapackDpoconWork& dpocon_work, LapackDsyevdWork& dsyevd_work,
                                CblasDtrsm& dtrsm, CblasDgemm& dgemm) {
  /* Load one coherent ABI. scipy-openblas32 prefixes every public symbol to
   * coexist safely with another BLAS; mixing standard and prefixed functions
   * from the same handle could combine incompatible providers accidentally. */
  dpotrf_work = nullptr;
  dpocon_work = nullptr;
  dsyevd_work = nullptr;
  dtrsm = nullptr;
  dgemm = nullptr;
  if (scipy_prefix) {
    return load_symbol(handle, "scipy_LAPACKE_dpotrf_work", dpotrf_work) &&
           load_symbol(handle, "scipy_LAPACKE_dpocon_work", dpocon_work) &&
           load_symbol(handle, "scipy_LAPACKE_dsyevd_work", dsyevd_work) &&
           load_symbol(handle, "scipy_cblas_dtrsm", dtrsm) &&
           load_symbol(handle, "scipy_cblas_dgemm", dgemm);
  }
  return load_symbol(handle, "LAPACKE_dpotrf_work", dpotrf_work) &&
         load_symbol(handle, "LAPACKE_dpocon_work", dpocon_work) &&
         load_symbol(handle, "LAPACKE_dsyevd_work", dsyevd_work) &&
         load_symbol(handle, "cblas_dtrsm", dtrsm) && load_symbol(handle, "cblas_dgemm", dgemm);
}

#ifdef GENERATIVEQC_XTB_CONFIGURED_PYODIDE_OPENBLAS
bool load_pyodide_lapacke_cblas_symbols(void* handle, LapackDpotrfWork& dpotrf_work,
                                        LapackDpoconWork& dpocon_work,
                                        LapackDsyevdWork& dsyevd_work, CblasDtrsm& dtrsm,
                                        CblasDgemm& dgemm) {
  /* These wrapper names are private to GenerativeQC. The adapter itself resolves
   * every raw OpenBLAS entry point from the exact absolute provider handle, so
   * a SciPy-first load cannot interpose on computation through global names. */
  dpotrf_work = nullptr;
  dpocon_work = nullptr;
  dsyevd_work = nullptr;
  dtrsm = nullptr;
  dgemm = nullptr;
  return load_symbol(handle, "generativeqc_xtb_pyodide_LAPACKE_dpotrf_work", dpotrf_work) &&
         load_symbol(handle, "generativeqc_xtb_pyodide_LAPACKE_dpocon_work", dpocon_work) &&
         load_symbol(handle, "generativeqc_xtb_pyodide_LAPACKE_dsyevd_work", dsyevd_work) &&
         load_symbol(handle, "generativeqc_xtb_pyodide_cblas_dtrsm", dtrsm) &&
         load_symbol(handle, "generativeqc_xtb_pyodide_cblas_dgemm", dgemm);
}
#endif

bool backend_self_test(const CpuLinearAlgebraBackend& backend) {
  double factor[1]{1.0};
  double reciprocal_condition = 0.0;
  double work[9]{};
  LapackInt integer_work[8]{};
  if (CpuLinearAlgebraAccess::dpotrf(backend)(kCblasColMajor, kLower, 1, factor, 1) != 0 ||
      CpuLinearAlgebraAccess::dpocon(backend)(kCblasColMajor, kLower, 1, factor, 1, 1.0,
                                              &reciprocal_condition, work, integer_work) != 0 ||
      !(reciprocal_condition > 0.0) || !std::isfinite(reciprocal_condition)) {
    return false;
  }
  double eigenvectors[1]{2.0};
  double eigenvalues[1]{};
  if (CpuLinearAlgebraAccess::dsyevd(backend)(kCblasColMajor, kEigenvectors, kLower, 1,
                                              eigenvectors, 1, eigenvalues, work, 9, integer_work,
                                              8) != 0 ||
      eigenvalues[0] != 2.0 || eigenvectors[0] != 1.0) {
    return false;
  }
  double rhs[1]{4.0};
  const double triangular[1]{2.0};
  CpuLinearAlgebraAccess::dtrsm(backend)(kCblasColMajor, kCblasLeft, kCblasLower, kCblasNoTrans,
                                         kCblasNonUnit, 1, 1, 1.0, triangular, 1, rhs, 1);
  double product[1]{};
  CpuLinearAlgebraAccess::dgemm(backend)(kCblasColMajor, kCblasNoTrans, kCblasTrans, 1, 1, 1, 1.0,
                                         rhs, 1, rhs, 1, 0.0, product, 1);
  return rhs[0] == 2.0 && product[0] == 4.0;
}

#if defined(GENERATIVEQC_XTB_CONFIGURED_PYODIDE_OPENBLAS)
void* open_pyodide_private_adapter() {
  /* Emscripten 5's dladdr is a stub and it has no link-map namespaces. The
   * Python wheel loader therefore supplies the installed absolute path. Accept
   * only the compile-time-reviewed basename and never search by generic name. */
  const char* path = std::getenv("GENERATIVEQC_XTB_PYODIDE_LAPACKE_SHIM");
  if (path == nullptr || path[0] != '/') {
    return nullptr;
  }
  const char* separator = std::strrchr(path, '/');
  const char* basename = separator == nullptr ? path : separator + 1;
  if (std::strcmp(basename, GENERATIVEQC_XTB_CONFIGURED_PYODIDE_OPENBLAS_ADAPTER) != 0) {
    return nullptr;
  }
  return dlopen(path, RTLD_NOW | RTLD_LOCAL);
}
#elif defined(GENERATIVEQC_XTB_CONFIGURED_WHEEL_OPENBLAS) && defined(_WIN32)
void* open_private_bundled_sibling(const char* filename) {
  /* Resolve relative to generativeqc.dll itself, never the process working
   * directory or PATH. LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR keeps any provider
   * dependencies in the same private wheel directory, while the default
   * directories retain Windows system-runtime resolution. */
  static const unsigned char kModuleAnchor = 0u;
  HMODULE module = nullptr;
  const DWORD flags =
      GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT;
  if (GetModuleHandleExW(flags, reinterpret_cast<LPCWSTR>(&kModuleAnchor), &module) == 0) {
    return nullptr;
  }

  std::array<wchar_t, 32768> module_path{};
  const DWORD length =
      GetModuleFileNameW(module, module_path.data(), static_cast<DWORD>(module_path.size()));
  if (length == 0u || length >= module_path.size()) {
    return nullptr;
  }
  std::wstring path(module_path.data(), length);
  const std::size_t separator = path.find_last_of(L"\\/");
  if (separator == std::wstring::npos) {
    return nullptr;
  }
  path.resize(separator + 1u);
  for (const char* character = filename; *character != '\0'; ++character) {
    path.push_back(static_cast<wchar_t>(static_cast<unsigned char>(*character)));
  }
  return static_cast<void*>(LoadLibraryExW(
      path.c_str(), nullptr, LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR | LOAD_LIBRARY_SEARCH_DEFAULT_DIRS));
}

void close_dynamic_library(void* handle) {
  if (handle != nullptr) {
    static_cast<void>(FreeLibrary(static_cast<HMODULE>(handle)));
  }
}
#elif defined(GENERATIVEQC_XTB_CONFIGURED_WHEEL_OPENBLAS) && defined(__APPLE__)
std::string canonical_path(const char* path) {
  char* resolved = realpath(path, nullptr);
  if (resolved == nullptr) {
    return {};
  }
  std::string result(resolved);
  std::free(resolved);
  return result;
}

void* open_private_bundled_sibling(const char* filename, std::string& expected_path) {
  /* The provider has an GenerativeQC-private LC_ID and is opened by the absolute
   * path beside libgenerativeqc. This prevents name-based discovery and host SciPy
   * reuse without claiming Linux dlmopen-style namespace isolation. */
  static const unsigned char kModuleAnchor = 0u;
  Dl_info module{};
  if (dladdr(&kModuleAnchor, &module) == 0 || module.dli_fname == nullptr) {
    return nullptr;
  }
  std::string path(module.dli_fname);
  const std::size_t separator = path.find_last_of('/');
  if (separator == std::string::npos) {
    return nullptr;
  }
  path.resize(separator + 1u);
  path += filename;
  expected_path = canonical_path(path.c_str());
  if (expected_path.empty()) {
    return nullptr;
  }
  return dlopen(expected_path.c_str(), RTLD_NOW | RTLD_LOCAL);
}

void close_dynamic_library(void* handle) {
  if (handle != nullptr) {
    static_cast<void>(dlclose(handle));
  }
}
#elif defined(GENERATIVEQC_XTB_CONFIGURED_CPU_LINALG_SHIM) || defined(GENERATIVEQC_XTB_CONFIGURED_WHEEL_OPENBLAS)
std::string host_isolated_sibling_path(const char* soname) {
  static const unsigned char kModuleAnchor = 0u;
  Dl_info module{};
  if (dladdr(&kModuleAnchor, &module) == 0 || module.dli_fname == nullptr) {
    return {};
  }
  std::string path(module.dli_fname);
  const std::size_t separator = path.find_last_of('/');
  if (separator == std::string::npos) {
    return {};
  }
  path.resize(separator + 1u);
  path += soname;
  return path;
}

void* open_host_isolated_sibling(const char* soname) {
  /* A LOCAL handle still resolves relocations against already-global objects.
   * A new link-map namespace is required to keep a host BLAS implementation
   * from interposing on GenerativeQC's private LP64 provider cohort. */
  const std::string path = host_isolated_sibling_path(soname);
  if (path.empty()) {
    return nullptr;
  }

  void* handle = dlmopen(LM_ID_NEWLM, path.c_str(), RTLD_NOW | RTLD_LOCAL);
  if (handle == nullptr) {
    return nullptr;
  }
  Lmid_t namespace_id = LM_ID_BASE;
  if (dlinfo(handle, RTLD_DI_LMID, &namespace_id) != 0 || namespace_id == LM_ID_BASE) {
    static_cast<void>(dlclose(handle));
    return nullptr;
  }
  return handle;
}

#ifdef GENERATIVEQC_XTB_CONFIGURED_CPU_LINALG_SHIM
struct MklIsolatedProviderHandles {
  void* provider = nullptr;
  void* bridge = nullptr;
  void* base_pthread = nullptr;
  bool provider_ready = false;
};

void close_mkl_bootstrap_handles(MklIsolatedProviderHandles& handles) {
  /* This cleanup is valid only before the provider enters the namespace. Once
   * provider code can create a base-registry pthread key, its destructor may
   * point back into the private namespace and every handle must remain loaded
   * for process life, including on a later verification failure. */
  if (handles.bridge != nullptr) {
    static_cast<void>(dlclose(handles.bridge));
    handles.bridge = nullptr;
  }
  if (handles.base_pthread != nullptr) {
    static_cast<void>(dlclose(handles.base_pthread));
    handles.base_pthread = nullptr;
  }
}

bool load_base_pthread_tss_api(generativeqc_xtb_mkl_pthread_tss_api& api, void*& retained_handle) {
  /* Resolve from a specific base-namespace DSO before creating the private
   * namespace. On glibc <2.34 pthread lives in libpthread; on newer glibc the
   * compatibility DSO forwards to libc. If libpthread is not loaded yet,
   * loading it here adds one base implementation rather than a second private
   * allocator. The successful handle is retained for the bridge lifetime. */
  const char* const libraries[] = {"libpthread.so.0", "libc.so.6"};
  for (const char* library : libraries) {
    void* handle = dlopen(library, RTLD_NOW | RTLD_LOCAL | RTLD_NOLOAD);
    if (handle == nullptr && std::strcmp(library, "libpthread.so.0") == 0) {
      handle = dlopen(library, RTLD_NOW | RTLD_LOCAL);
    }
    if (handle == nullptr) {
      continue;
    }

    generativeqc_xtb_mkl_pthread_tss_api candidate{};
    if (load_symbol(handle, "pthread_key_create", candidate.key_create) &&
        load_symbol(handle, "pthread_key_delete", candidate.key_delete) &&
        load_symbol(handle, "pthread_getspecific", candidate.getspecific) &&
        load_symbol(handle, "pthread_setspecific", candidate.setspecific)) {
      api = candidate;
      retained_handle = handle;
      return true;
    }
    static_cast<void>(dlclose(handle));
  }
  return false;
}

MklIsolatedProviderHandles open_mkl_isolated_provider() {
  MklIsolatedProviderHandles handles;
  generativeqc_xtb_mkl_pthread_tss_api api{};
  if (!load_base_pthread_tss_api(api, handles.base_pthread)) {
    return handles;
  }

  const std::string bridge_path =
      host_isolated_sibling_path("libgenerativeqc_xtb_mkl_pthread_tss_bridge.so");
  const std::string provider_path = host_isolated_sibling_path("libgenerativeqc_xtb_mkl_lp64_shim.so");
  if (bridge_path.empty() || provider_path.empty()) {
    close_mkl_bootstrap_handles(handles);
    return handles;
  }

  /* The first dlmopen admits only the dependency-free bridge. It cannot load
   * or execute a private libc/libdl before the base function table is set. */
  handles.bridge = dlmopen(LM_ID_NEWLM, bridge_path.c_str(), RTLD_NOW | RTLD_LOCAL);
  if (handles.bridge == nullptr) {
    close_mkl_bootstrap_handles(handles);
    return handles;
  }
  Lmid_t namespace_id = LM_ID_BASE;
  generativeqc_xtb_mkl_pthread_tss_bridge_initialize_fn initialize = nullptr;
  if (dlinfo(handles.bridge, RTLD_DI_LMID, &namespace_id) != 0 || namespace_id == LM_ID_BASE ||
      !load_symbol(handles.bridge, GENERATIVEQC_XTB_MKL_PTHREAD_TSS_BRIDGE_INITIALIZE_SYMBOL, initialize) ||
      initialize(&api) != 0) {
    close_mkl_bootstrap_handles(handles);
    return handles;
  }

  /* The provider shim records the bridge as its first DT_NEEDED dependency.
   * Loading it into the initialized namespace reuses that image, so public
   * pthread TSS calls and glibc <=2.33 libdl's weak __pthread_* aliases bind
   * before any private dependency relocation or constructor can execute. */
  handles.provider = dlmopen(namespace_id, provider_path.c_str(), RTLD_NOW | RTLD_LOCAL);
  if (handles.provider == nullptr) {
    close_mkl_bootstrap_handles(handles);
    return handles;
  }
  Lmid_t provider_namespace_id = LM_ID_BASE;
  handles.provider_ready = dlinfo(handles.provider, RTLD_DI_LMID, &provider_namespace_id) == 0 &&
                           provider_namespace_id == namespace_id;
  return handles;
}
#endif
#endif

#if defined(GENERATIVEQC_XTB_CONFIGURED_WHEEL_OPENBLAS) && defined(_WIN32)
template <typename Function>
HMODULE dynamic_symbol_module(Function function) {
  static_assert(sizeof(Function) == sizeof(void*));
  void* address = nullptr;
  std::memcpy(&address, &function, sizeof(address));
  HMODULE module = nullptr;
  const DWORD flags =
      GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT;
  if (GetModuleHandleExW(flags, static_cast<LPCWSTR>(address), &module) == 0) {
    return nullptr;
  }
  return module;
}

template <typename First, typename... Rest>
bool symbols_belong_to_private_provider(void* opened_handle, First first, Rest... rest) {
  const HMODULE module = static_cast<HMODULE>(opened_handle);
  if (module == nullptr || dynamic_symbol_module(first) != module ||
      ((dynamic_symbol_module(rest) != module) || ...)) {
    return false;
  }
  return true;
}
#elif defined(GENERATIVEQC_XTB_CONFIGURED_WHEEL_OPENBLAS) && defined(__APPLE__)
template <typename Function>
bool dynamic_symbol_info(Function function, Dl_info& info) {
  static_assert(sizeof(Function) == sizeof(void*));
  void* address = nullptr;
  std::memcpy(&address, &function, sizeof(address));
  return dladdr(address, &info) != 0 && info.dli_fbase != nullptr && info.dli_fname != nullptr;
}

template <typename First, typename... Rest>
bool symbols_belong_to_private_provider(const std::string& expected_path, First first,
                                        Rest... rest) {
  Dl_info first_info{};
  if (!dynamic_symbol_info(first, first_info)) {
    return false;
  }
  bool same_image = true;
  const auto check = [&](auto function) {
    Dl_info info{};
    same_image =
        same_image && dynamic_symbol_info(function, info) && info.dli_fbase == first_info.dli_fbase;
  };
  (check(rest), ...);
  return same_image && canonical_path(first_info.dli_fname) == expected_path;
}
#endif


}  // namespace

bool CpuLinearAlgebraBackend::ready() const noexcept {
  return origin_ != Origin::kNone && dpotrf_work_ != nullptr && dpocon_work_ != nullptr &&
         dsyevd_work_ != nullptr && dtrsm_ != nullptr && dgemm_ != nullptr;
}

bool CpuLinearAlgebraBackend::production() const noexcept {
  return origin_ == Origin::kMklShimLp64 || origin_ == Origin::kOpenBlasIsolatedLp64 ||
         origin_ == Origin::kBundledOpenBlasLp64 || origin_ == Origin::kOpenBlasLp64;
}

bool CpuLinearAlgebraBackend::production_mkl() const noexcept {
  return origin_ == Origin::kMklShimLp64;
}

bool CpuLinearAlgebraBackend::production_mkl_isolated() const noexcept {
  return origin_ == Origin::kMklShimLp64;
}

bool CpuLinearAlgebraBackend::production_openblas_isolated() const noexcept {
  return origin_ == Origin::kOpenBlasIsolatedLp64;
}

void CpuLinearAlgebraBackend::release_thread_resources() const noexcept {
  if (thread_cleanup_ != nullptr) {
    thread_cleanup_();
  }
}

Lp64BackendStatus prepare_internal_test_lp64_backend(
    LapackDpotrfWork dpotrf_work, LapackDpoconWork dpocon_work, LapackDsyevdWork dsyevd_work,
    CblasDtrsm dtrsm, CblasDgemm dgemm, BlasSetNumThreadsLocal set_num_threads_local,
    CpuLinearAlgebraBackend& backend, std::string& error, BlasThreadCleanup thread_cleanup) {
  CpuLinearAlgebraBackend created = CpuLinearAlgebraAccess::make(
      CpuLinearAlgebraBackend::Origin::kInternalTestLp64, dpotrf_work, dpocon_work, dsyevd_work,
      dtrsm, dgemm, set_num_threads_local, thread_cleanup);
  if (!created.ready() || !backend_self_test(created)) {
    error = "internal LP64 test backend failed its column-major preflight";
    return Lp64BackendStatus::unavailable;
  }
  backend = created;
  error.clear();
  return Lp64BackendStatus::success;
}

Lp64BackendStatus prepare_lp64_runtime_backend(CpuLinearAlgebraBackend& backend, std::string& error) {
  struct LinalgRuntimeState {
    CpuLinearAlgebraBackend backend;
    Lp64BackendStatus status = Lp64BackendStatus::unavailable;
    std::string message;
    std::array<void*, 3> retained_loader_handles{};
  };
  static const LinalgRuntimeState runtime = [] {
    LinalgRuntimeState state;
#ifdef GENERATIVEQC_XTB_CONFIGURED_PYODIDE_OPENBLAS
    /* Pyodide wheels carry a content-qualified OpenBLAS side module plus a
     * narrow native adapter. Emscripten cannot isolate dynamic-linker
     * namespaces, so Python supplies exact installed paths and the adapter
     * dlsyms every provider function from that absolute private handle. */
    {
      dlerror();
      void* handle = open_pyodide_private_adapter();
      if (handle != nullptr) {
        LapackDpotrfWork dpotrf_work = nullptr;
        LapackDpoconWork dpocon_work = nullptr;
        LapackDsyevdWork dsyevd_work = nullptr;
        CblasDtrsm dtrsm = nullptr;
        CblasDgemm dgemm = nullptr;
        BlasSetNumThreadsLocal set_threads = nullptr;
        using OpenBlasGetConfig = const char* (*)();
        OpenBlasGetConfig get_config = nullptr;
        if (load_pyodide_lapacke_cblas_symbols(handle, dpotrf_work, dpocon_work, dsyevd_work, dtrsm,
                                               dgemm) &&
            load_symbol(handle, "generativeqc_xtb_pyodide_openblas_get_config", get_config) &&
            load_symbol(handle, "generativeqc_xtb_pyodide_openblas_set_num_threads_local", set_threads)) {
          const char* config = get_config();
          constexpr const char* kExpectedConfigPrefix =
              GENERATIVEQC_XTB_CONFIGURED_PYODIDE_OPENBLAS_CONFIG_PREFIX;
          if (config != nullptr &&
              std::strncmp(config, kExpectedConfigPrefix, std::strlen(kExpectedConfigPrefix)) ==
                  0 &&
              std::strstr(config, "USE64BITINT") == nullptr) {
            CpuLinearAlgebraBackend created = CpuLinearAlgebraAccess::make(
                CpuLinearAlgebraBackend::Origin::kBundledOpenBlasLp64, dpotrf_work, dpocon_work,
                dsyevd_work, dtrsm, dgemm, set_threads);
            if (backend_self_test(created)) {
              /* Retain the adapter and exact provider handles for process life. */
              state.backend = created;
              state.status = Lp64BackendStatus::success;
              return state;
            }
          }
        }
        static_cast<void>(dlclose(handle));
      }
      state.message =
          "private Pyodide OpenBLAS provider or LAPACKE adapter is missing or failed "
          "verification";
      return state;
    }
#endif
#ifdef GENERATIVEQC_XTB_CONFIGURED_WHEEL_OPENBLAS
    /* Python wheels carry one hash-verified scipy-openblas32 provider cohort
     * as a private sibling. Linux loads an auditwheel-repaired shim in a fresh
     * glibc link-map namespace. macOS/Windows load a renamed provider image by
     * absolute path and verify that every dispatch symbol comes from that
     * image. A configured bundle is an all-or-nothing contract, so a failure
     * never falls back to an unrelated system provider. */
    {
#if defined(_WIN32)
      void* handle = open_private_bundled_sibling(GENERATIVEQC_XTB_CONFIGURED_WHEEL_OPENBLAS_FILENAME);
      constexpr CpuLinearAlgebraBackend::Origin kOrigin =
          CpuLinearAlgebraBackend::Origin::kBundledOpenBlasLp64;
#elif defined(__APPLE__)
      dlerror();
      std::string provider_path;
      void* handle =
          open_private_bundled_sibling(GENERATIVEQC_XTB_CONFIGURED_WHEEL_OPENBLAS_FILENAME, provider_path);
      constexpr CpuLinearAlgebraBackend::Origin kOrigin =
          CpuLinearAlgebraBackend::Origin::kBundledOpenBlasLp64;
#else
      dlerror();
      void* handle = open_host_isolated_sibling("libgenerativeqc_xtb_openblas_lp64_shim.so");
      constexpr CpuLinearAlgebraBackend::Origin kOrigin =
          CpuLinearAlgebraBackend::Origin::kOpenBlasIsolatedLp64;
#endif
      if (handle != nullptr) {
        LapackDpotrfWork dpotrf_work = nullptr;
        LapackDpoconWork dpocon_work = nullptr;
        LapackDsyevdWork dsyevd_work = nullptr;
        CblasDtrsm dtrsm = nullptr;
        CblasDgemm dgemm = nullptr;
        BlasSetNumThreadsLocal set_threads = nullptr;
        using OpenBlasGetConfig = const char* (*)();
        OpenBlasGetConfig get_config = nullptr;
        if (load_lapacke_cblas_symbols(handle, true, dpotrf_work, dpocon_work, dsyevd_work, dtrsm,
                                       dgemm) &&
            load_symbol(handle, "scipy_openblas_get_config", get_config)) {
          const char* config = get_config();
          constexpr const char* kExpectedConfigPrefix =
              GENERATIVEQC_XTB_CONFIGURED_WHEEL_OPENBLAS_CONFIG_PREFIX;
          if (config != nullptr &&
              std::strncmp(config, kExpectedConfigPrefix, std::strlen(kExpectedConfigPrefix)) ==
                  0 &&
              std::strstr(config, "USE64BITINT") == nullptr) {
#if defined(_WIN32) || defined(__APPLE__)
            using OpenBlasSetNumThreadsGlobal = void (*)(int);
            using OpenBlasGetNumThreads = int (*)();
            OpenBlasSetNumThreadsGlobal set_threads_global = nullptr;
            OpenBlasGetNumThreads get_threads = nullptr;
            if (load_symbol(handle, "scipy_openblas_set_num_threads", set_threads_global) &&
                load_symbol(handle, "scipy_openblas_get_num_threads", get_threads)
#if defined(_WIN32)
                && symbols_belong_to_private_provider(handle, dpotrf_work, dpocon_work, dsyevd_work,
                                                      dtrsm, dgemm, get_config, set_threads_global,
                                                      get_threads)
#else
                && symbols_belong_to_private_provider(provider_path, dpotrf_work, dpocon_work,
                                                      dsyevd_work, dtrsm, dgemm, get_config,
                                                      set_threads_global, get_threads)
#endif
            ) {
              /* Desktop providers do not export local thread control. This
               * renamed private image is initialized exactly once by the
               * thread-safe function-static factory, so fixing its global
               * setting cannot alter an unrelated host OpenBLAS instance. */
              set_threads_global(1);
              if (get_threads() == 1) {
                CpuLinearAlgebraBackend created = CpuLinearAlgebraAccess::make(
                    kOrigin, dpotrf_work, dpocon_work, dsyevd_work, dtrsm, dgemm, nullptr);
                if (backend_self_test(created)) {
                  state.backend = created;
                  state.status = Lp64BackendStatus::success;
                  return state;
                }
              }
            }
#else
            if (!load_symbol(handle, "openblas_set_num_threads_local", set_threads)) {
              static_cast<void>(
                  load_symbol(handle, "scipy_openblas_set_num_threads_local", set_threads));
            }
            if (set_threads != nullptr) {
              CpuLinearAlgebraBackend created = CpuLinearAlgebraAccess::make(
                  kOrigin, dpotrf_work, dpocon_work, dsyevd_work, dtrsm, dgemm, set_threads);
              if (backend_self_test(created)) {
                state.backend = created;
                state.status = Lp64BackendStatus::success;
                return state;
              }
            }
#endif
          }
        }
#if defined(_WIN32) || defined(__APPLE__)
        close_dynamic_library(handle);
#else
        static_cast<void>(dlclose(handle));
#endif
      }
#if defined(_WIN32) || defined(__APPLE__)
      constexpr const char* kPrivateProviderName = GENERATIVEQC_XTB_CONFIGURED_WHEEL_OPENBLAS_FILENAME;
#else
      constexpr const char* kPrivateProviderName = "libgenerativeqc_xtb_openblas_lp64_shim.so";
#endif
      state.message = std::string("private wheel OpenBLAS provider is missing or failed ") +
                      "verification (" + kPrivateProviderName + ")";
      return state;
    }
#endif

#if defined(_WIN32)
    /* Native system-provider discovery remains POSIX-only. Windows wheels use
     * the private provider above; a non-wheel Windows build must configure a
     * future explicit LoadLibrary provider path instead of searching PATH. */
    state.message = "CPU linear-algebra runtime is unavailable in this Windows build";
    return state;
#else

#ifdef GENERATIVEQC_XTB_CONFIGURED_CPU_LINALG_SHIM
    /* Preferred isolated MKL provider: initialize the dependency-free pthread
     * TSS bridge alone in a new namespace, then add the private shim with fixed
     * DT_NEEDED dependencies on libmkl_intel_lp64, libmkl_sequential, and
     * libmkl_core. This orders TSS bridging before private libc/libdl/MKL while
     * preserving #30 namespace isolation. We never load libmkl_rt, call
     * MKL_Set_Interface_Layer, or read MKL interface-layer state. */
    {
      dlerror();
      MklIsolatedProviderHandles handles = open_mkl_isolated_provider();
      void* handle = handles.provider_ready ? handles.provider : nullptr;
      if (handles.provider != nullptr) {
        /* Provider relocation or constructors may already have registered a
         * base pthread key whose destructor lives in the private namespace.
         * Retain all three handles before any further verification, even if
         * symbol resolution or the backend self-test subsequently fails. */
        state.retained_loader_handles = {handles.provider, handles.bridge, handles.base_pthread};
      }
      if (handle != nullptr) {
        LapackDpotrfWork dpotrf_work = nullptr;
        LapackDpoconWork dpocon_work = nullptr;
        LapackDsyevdWork dsyevd_work = nullptr;
        CblasDtrsm dtrsm = nullptr;
        CblasDgemm dgemm = nullptr;
        BlasSetNumThreadsLocal set_threads = nullptr;
        BlasThreadCleanup thread_cleanup = nullptr;
        if (load_lapacke_cblas_symbols(handle, false, dpotrf_work, dpocon_work, dsyevd_work, dtrsm,
                                       dgemm) &&
            load_symbol(handle, "MKL_Set_Num_Threads_Local", set_threads) &&
            load_symbol(handle, "MKL_Thread_Free_Buffers", thread_cleanup)) {
          CpuLinearAlgebraBackend created = CpuLinearAlgebraAccess::make(
              CpuLinearAlgebraBackend::Origin::kMklShimLp64, dpotrf_work, dpocon_work, dsyevd_work,
              dtrsm, dgemm, set_threads, thread_cleanup);
          if (backend_self_test(created)) {
            /* Retain the provider, initialized bridge, and exact base pthread
             * handle for process life so dispatch and TSS destructor pointers
             * remain valid through worker/interpreter teardown. */
            state.backend = created;
            state.status = Lp64BackendStatus::success;
            return state;
          }
        }
      }
      state.message =
          "host-isolated MKL pthread bridge/provider shim is configured but did not verify "
          "(libgenerativeqc_xtb_mkl_pthread_tss_bridge, libgenerativeqc_xtb_mkl_lp64_shim)";
      return state;
    }
#endif

#if !defined(GENERATIVEQC_XTB_CONFIGURED_CPU_LINALG_MKL)
#ifdef GENERATIVEQC_XTB_CONFIGURED_CPU_LINALG_RUNTIME
    constexpr const char* kConfiguredRuntime = GENERATIVEQC_XTB_CONFIGURED_CPU_LINALG_RUNTIME;
#else
    constexpr const char* kConfiguredRuntime = nullptr;
#endif

    using OpenBlasGetConfig = const char* (*)();
    /* dlopen candidates in preference order: the absolute path CMake baked in
     * (GENERATIVEQC_XTB_CPU_LINALG_LIBRARY / find_package(BLAS) in CMakeLists.txt) first,
     * then known OpenBLAS sonames that the Python layer may already have
     * preloaded. MKL is never accepted through this base-namespace fallback;
     * its only production path is the isolated component shim above. */
    const char* const runtime_names[] = {
        kConfiguredRuntime, "libscipy_openblas.so", "libscipy_openblas32_.so",
        "libopenblas.so.0", "libopenblas.so",       "libopenblas.so.3",
    };

    for (const char* name : runtime_names) {
      if (name == nullptr || *name == '\0') {
        continue;
      }
      void* handle = dlopen(name, RTLD_NOW | RTLD_LOCAL);
      if (handle == nullptr) {
        continue;
      }
      LapackDpotrfWork dpotrf_work = nullptr;
      LapackDpoconWork dpocon_work = nullptr;
      LapackDsyevdWork dsyevd_work = nullptr;
      CblasDtrsm dtrsm = nullptr;
      CblasDgemm dgemm = nullptr;
      BlasSetNumThreadsLocal set_threads = nullptr;
      bool scipy_prefix = false;
      if (!load_lapacke_cblas_symbols(handle, false, dpotrf_work, dpocon_work, dsyevd_work, dtrsm,
                                      dgemm)) {
        scipy_prefix = true;
        if (!load_lapacke_cblas_symbols(handle, true, dpotrf_work, dpocon_work, dsyevd_work, dtrsm,
                                        dgemm)) {
          static_cast<void>(dlclose(handle));
          continue;
        }
      }
      /* INTERFACE64 OpenBLAS builds may retain unsuffixed function names, so
       * symbol spelling alone cannot prove the 32-bit LapackInt ABI. Reject
       * providers that cannot identify themselves or report USE64BITINT. */
      OpenBlasGetConfig get_config = nullptr;
      if (!load_symbol(handle, scipy_prefix ? "scipy_openblas_get_config" : "openblas_get_config",
                       get_config)) {
        static_cast<void>(dlclose(handle));
        continue;
      }
      const char* config = get_config();
      if (config == nullptr || std::strstr(config, "USE64BITINT") != nullptr) {
        static_cast<void>(dlclose(handle));
        continue;
      }
      if (!load_symbol(handle, "openblas_set_num_threads_local", set_threads)) {
        /* scipy-openblas32 currently retains the unprefixed local-control
         * symbol, while other prefixed builds may follow the public header. */
        static_cast<void>(load_symbol(handle, "scipy_openblas_set_num_threads_local", set_threads));
      }
      if (set_threads == nullptr) {
        static_cast<void>(dlclose(handle));
        continue;
      }
      CpuLinearAlgebraBackend created =
          CpuLinearAlgebraAccess::make(CpuLinearAlgebraBackend::Origin::kOpenBlasLp64, dpotrf_work,
                                       dpocon_work, dsyevd_work, dtrsm, dgemm, set_threads);
      if (!backend_self_test(created)) {
        static_cast<void>(dlclose(handle));
        continue;
      }
      /* Retain one process-lifetime loader reference so all dispatch pointers stay valid. */
      state.backend = created;
      state.status = Lp64BackendStatus::success;
      return state;
    }
    state.message =
        "failed to load an LP64 OpenBLAS runtime (libopenblas*.so, "
        "libscipy_openblas.so, or the CMake-configured path)";
    return state;
#else
    state.message =
        "host-isolated MKL provider shim is unavailable; configure the adjacent MKL "
        "LP64/sequential components or select an LP64 OpenBLAS runtime";
    return state;
#endif
#endif
  }();

  if (runtime.status != Lp64BackendStatus::success) {
    error = runtime.message.empty() ? "LP64 CPU linear-algebra backend initialization failed"
                                    : runtime.message;
    return runtime.status;
  }
  backend = runtime.backend;
  error.clear();
  return Lp64BackendStatus::success;
}


// clang-format on
LapackInt cholesky_lower(const CpuLinearAlgebraBackend& backend, LapackInt n, double* matrix) {
  return CpuLinearAlgebraAccess::dpotrf(backend)(102, 'L', n, matrix, n);
}
LapackInt reciprocal_condition_lower(const CpuLinearAlgebraBackend& backend, LapackInt n,
                                     const double* factor, double norm,
                                     double* reciprocal_condition, double* work,
                                     LapackInt* integer_work) {
  return CpuLinearAlgebraAccess::dpocon(backend)(102, 'L', n, factor, n, norm, reciprocal_condition,
                                                 work, integer_work);
}
void solve_lower_triangular(const CpuLinearAlgebraBackend& backend, TriangularSide side,
                            Transpose transpose, LapackInt n, const double* factor, double* rhs) {
  CpuLinearAlgebraAccess::dtrsm(backend)(102, side == TriangularSide::left ? 141 : 142, 122,
                                         transpose == Transpose::none ? 111 : 112, 131, n, n, 1.0,
                                         factor, n, rhs, n);
}
SymmetricEigenBinding bind_symmetric_eigen(const CpuLinearAlgebraBackend& backend) noexcept {
  return SymmetricEigenBinding(CpuLinearAlgebraAccess::dsyevd(backend));
}
GemmBinding bind_gemm(const CpuLinearAlgebraBackend& backend) noexcept {
  return GemmBinding(CpuLinearAlgebraAccess::dgemm(backend));
}

}  // namespace generativeqc::tensor::cpu
