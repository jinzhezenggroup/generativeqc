#pragma once

#include <cstdint>
#include <string>

#include "tensor/cpu/lp64_abi.hpp"

namespace generativeqc::tensor::cpu {

/** Runtime-LP64 admission is separate from canonical CpuLinalgPlan admission.
 * No scalar/build-linked fallback is supplied here. Existing integrated CMake
 * admits Linux bundled OpenBLAS or configured/system POSIX runtime discovery.
 * Inherited MKL, desktop-private and Pyodide arms remain inactive/unqualified;
 * extracting their source does not add build wiring or platform support. */
enum class Lp64BackendStatus : std::uint8_t { success, unavailable };

/*
 * Verified LP64 linear-algebra dispatch.
 *
 * Production code obtains this handle from prepare_lp64_runtime_backend. The
 * factory loads and verifies all required symbols from a private bundled
 * OpenBLAS provider in native wheels, the configured native LP64 runtime, or
 * common system SONAMEs. System providers must expose local thread control so
 * xtbloom's outer batch workers can keep BLAS sequential. The macOS/Windows
 * wheel provider is a renamed private image instead: initialization fixes that
 * image globally to one thread once, without mutating an unrelated host BLAS.
 *
 * The MKL path is host-isolated. CMake builds a private shim with fixed
 * DT_NEEDED dependencies on
 * libmkl_intel_lp64, libmkl_sequential, and libmkl_core, and the factory loads
 * the adjacent shim with RTLD_LOCAL in a new glibc link-map namespace. The
 * namespace is required because RTLD_LOCAL alone still permits pre-existing
 * global host symbols to interpose on new dependencies. The components are
 * intrinsically LP64 and sequential, so xtbloom never loads libmkl_rt, calls
 * MKL_Set_Interface_Layer, reads MKL interface-layer state, or mutates an
 * embedding process's MKL state. A missing or invalid shim fails
 * deterministically; MKL never falls back to the base namespace. Plain LP64
 * Linux wheels apply the same namespace isolation to a hash-verified private
 * shim loaded by absolute sibling path. auditwheel vendors and collision-
 * renames the shim's scipy-openblas32 dependency closure. macOS and Windows
 * instead load a renamed provider by absolute sibling path; that is a private
 * payload boundary but not Linux-style link-map isolation. The upstream Python
 * distribution is a build input only and is never imported or required at
 * runtime. Pyodide wheels use the official content-pinned WebAssembly
 * OpenBLAS artifact and a narrow LAPACKE adapter. Because Emscripten has no
 * isolated namespace or deep binding, the Python loader supplies exact
 * installed paths and the adapter resolves raw functions only from that
 * provider handle, never through global SciPy/NumPy symbols. Native system
 * OpenBLAS remains a separate production provider. The testing factory is
 * kept in this internal namespace so tests can install
 * spies and deterministic LAPACK failures without making ABI claims on behalf
 * of an external provider.
 */
class CpuLinearAlgebraBackend {
 public:
  CpuLinearAlgebraBackend() noexcept = default;

  [[nodiscard]] bool ready() const noexcept;
  /* True for any verified lazily-loaded production backend (MKL or OpenBLAS). */
  [[nodiscard]] bool production() const noexcept;
  /* True only when the loaded production backend is the isolated MKL shim. */
  [[nodiscard]] bool production_mkl() const noexcept;
  /* True only for the host-isolated MKL shim provider, which never mutates the
   * embedding process's MKL interface/threading state. */
  [[nodiscard]] bool production_mkl_isolated() const noexcept;
  /* True only for the private OpenBLAS cohort bundled in Linux wheels and
   * loaded in its own glibc link-map namespace. Desktop private providers do
   * not claim this stronger isolation property. */
  [[nodiscard]] bool production_openblas_isolated() const noexcept;
  /* Release provider-owned state for the calling thread. Only the isolated
   * MKL backend supplies this hook; persistent runtime workers invoke it
   * before pthread teardown so oneMKL never leaves cleanup to glibc TSD
   * destruction after the worker has returned. */
  void release_thread_resources() const noexcept;

 private:
  enum class Origin : std::uint8_t {
    kNone,
    kMklShimLp64,
    kOpenBlasIsolatedLp64,
    kBundledOpenBlasLp64,
    kOpenBlasLp64,
    kInternalTestLp64,
  };

  Origin origin_ = Origin::kNone;
  LapackDpotrfWork dpotrf_work_ = nullptr;
  LapackDpoconWork dpocon_work_ = nullptr;
  LapackDsyevdWork dsyevd_work_ = nullptr;
  CblasDtrsm dtrsm_ = nullptr;
  CblasDgemm dgemm_ = nullptr;
  BlasSetNumThreadsLocal set_num_threads_local_ = nullptr;
  BlasThreadCleanup thread_cleanup_ = nullptr;

  friend Lp64BackendStatus prepare_lp64_runtime_backend(CpuLinearAlgebraBackend& backend,
                                                        std::string& error);
  friend Lp64BackendStatus prepare_internal_test_lp64_backend(
      LapackDpotrfWork dpotrf_work, LapackDpoconWork dpocon_work, LapackDsyevdWork dsyevd_work,
      CblasDtrsm dtrsm, CblasDgemm dgemm, BlasSetNumThreadsLocal set_num_threads_local,
      CpuLinearAlgebraBackend& backend, std::string& error, BlasThreadCleanup thread_cleanup);
  friend struct CpuLinearAlgebraAccess;
};

Lp64BackendStatus prepare_lp64_runtime_backend(CpuLinearAlgebraBackend& backend,
                                               std::string& error);

/* Internal test-only dependency injection; production must use the runtime factory. */
Lp64BackendStatus prepare_internal_test_lp64_backend(
    LapackDpotrfWork dpotrf_work, LapackDpoconWork dpocon_work, LapackDsyevdWork dsyevd_work,
    CblasDtrsm dtrsm, CblasDgemm dgemm, BlasSetNumThreadsLocal set_num_threads_local,
    CpuLinearAlgebraBackend& backend, std::string& error,
    BlasThreadCleanup thread_cleanup = nullptr);

struct CpuLinearAlgebraAccess {
  static CpuLinearAlgebraBackend make(CpuLinearAlgebraBackend::Origin origin,
                                      LapackDpotrfWork dpotrf_work, LapackDpoconWork dpocon_work,
                                      LapackDsyevdWork dsyevd_work, CblasDtrsm dtrsm,
                                      CblasDgemm dgemm,
                                      BlasSetNumThreadsLocal set_num_threads_local,
                                      BlasThreadCleanup thread_cleanup = nullptr) noexcept {
    CpuLinearAlgebraBackend backend;
    backend.origin_ = origin;
    backend.dpotrf_work_ = dpotrf_work;
    backend.dpocon_work_ = dpocon_work;
    backend.dsyevd_work_ = dsyevd_work;
    backend.dtrsm_ = dtrsm;
    backend.dgemm_ = dgemm;
    backend.set_num_threads_local_ = set_num_threads_local;
    backend.thread_cleanup_ = thread_cleanup;
    return backend;
  }

  static LapackDpotrfWork dpotrf(const CpuLinearAlgebraBackend& backend) noexcept {
    return backend.dpotrf_work_;
  }
  static LapackDpoconWork dpocon(const CpuLinearAlgebraBackend& backend) noexcept {
    return backend.dpocon_work_;
  }
  static LapackDsyevdWork dsyevd(const CpuLinearAlgebraBackend& backend) noexcept {
    return backend.dsyevd_work_;
  }
  static CblasDtrsm dtrsm(const CpuLinearAlgebraBackend& backend) noexcept {
    return backend.dtrsm_;
  }
  static CblasDgemm dgemm(const CpuLinearAlgebraBackend& backend) noexcept {
    return backend.dgemm_;
  }
  static BlasSetNumThreadsLocal set_threads(const CpuLinearAlgebraBackend& backend) noexcept {
    return backend.set_num_threads_local_;
  }
};

class ScopedSequentialBlas final {
 public:
  explicit ScopedSequentialBlas(const CpuLinearAlgebraBackend& backend)
      : setter_(CpuLinearAlgebraAccess::set_threads(backend)) {
    if (setter_ != nullptr) {
      previous_ = setter_(1);
    }
  }

  ~ScopedSequentialBlas() {
    if (setter_ != nullptr) {
      static_cast<void>(setter_(previous_));
    }
  }

  ScopedSequentialBlas(const ScopedSequentialBlas&) = delete;
  ScopedSequentialBlas& operator=(const ScopedSequentialBlas&) = delete;

 private:
  BlasSetNumThreadsLocal setter_ = nullptr;
  int previous_ = 0;
};

enum class TriangularSide { left, right };
enum class Transpose { none, transpose };

/** Typed column-major primitive operations. The admitted owner controls the
 * surrounding thread scope; these calls neither select a provider nor retry. */
LapackInt cholesky_lower(const CpuLinearAlgebraBackend& backend, LapackInt n, double* matrix);
LapackInt reciprocal_condition_lower(const CpuLinearAlgebraBackend& backend, LapackInt n,
                                     const double* factor, double norm,
                                     double* reciprocal_condition, double* work,
                                     LapackInt* integer_work);
void solve_lower_triangular(const CpuLinearAlgebraBackend& backend, TriangularSide side,
                            Transpose transpose, LapackInt n, const double* factor, double* rhs);

/** Opaque borrowed primitive bindings. Method consumers cannot unwrap the ABI
 * entry. The existing provider lifetime still owns every function pointer. */
class SymmetricEigenBinding {
 public:
  explicit operator bool() const noexcept { return call_ != nullptr; }
  LapackInt operator()(LapackInt layout, char vectors, char triangle, LapackInt n, double* matrix,
                       LapackInt lda, double* values, double* work, LapackInt lwork,
                       LapackInt* integer_work, LapackInt liwork) const {
    return call_(layout, vectors, triangle, n, matrix, lda, values, work, lwork, integer_work,
                 liwork);
  }

 private:
  explicit SymmetricEigenBinding(LapackDsyevdWork call) noexcept : call_(call) {}
  LapackDsyevdWork call_;
  friend SymmetricEigenBinding bind_symmetric_eigen(const CpuLinearAlgebraBackend&) noexcept;
};

class GemmBinding {
 public:
  void operator()(int layout, int transpose_left, int transpose_right, LapackInt rows,
                  LapackInt columns, LapackInt inner, double alpha, const double* left,
                  LapackInt leading_left, const double* right, LapackInt leading_right, double beta,
                  double* result, LapackInt leading_result) const {
    call_(layout, transpose_left, transpose_right, rows, columns, inner, alpha, left, leading_left,
          right, leading_right, beta, result, leading_result);
  }

 private:
  explicit GemmBinding(CblasDgemm call) noexcept : call_(call) {}
  CblasDgemm call_;
  friend GemmBinding bind_gemm(const CpuLinearAlgebraBackend&) noexcept;
};

SymmetricEigenBinding bind_symmetric_eigen(const CpuLinearAlgebraBackend& backend) noexcept;
GemmBinding bind_gemm(const CpuLinearAlgebraBackend& backend) noexcept;

}  // namespace generativeqc::tensor::cpu
