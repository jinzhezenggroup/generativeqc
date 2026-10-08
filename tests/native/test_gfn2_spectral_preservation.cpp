// The same real-API probe is linked independently to frozen and current owners.
// Provider callbacks use stack-only arithmetic and publish a deterministic trace.
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <memory>
#include <new>
#include <string>
#include <utility>
#include <vector>

#include "model/gfn2/eigensolver.hpp"
#include "solver/cpu/symmetric_eigen.hpp"

namespace {
bool counting = false;
std::size_t allocation_count = 0;
std::size_t allocation_bytes = 0;
void* allocation(std::size_t n, std::size_t alignment = 0) {
  if (counting) {
    ++allocation_count;
    allocation_bytes += n;
  }
  void* p = nullptr;
  if (alignment) {
    if (posix_memalign(&p, alignment, n ? n : alignment)) throw std::bad_alloc{};
  } else {
    p = std::malloc(n ? n : 1);
  }
  if (!p) throw std::bad_alloc{};
  return p;
}
void* occupation_address = nullptr;
std::size_t occupation_calls = 0;
std::array<void*, 4> scan_addresses{};
std::size_t scan_calls = 0, threaded_scan_calls = 0;
int thread_count = 17;
}  // namespace
void* operator new(std::size_t n) { return allocation(n); }
void* operator new[](std::size_t n) { return allocation(n); }
void* operator new(std::size_t n, std::align_val_t a) {
  return allocation(n, static_cast<std::size_t>(a));
}
void* operator new[](std::size_t n, std::align_val_t a) {
  return allocation(n, static_cast<std::size_t>(a));
}
void operator delete(void* p) noexcept { std::free(p); }
void operator delete[](void* p) noexcept { std::free(p); }
void operator delete(void* p, std::size_t) noexcept { std::free(p); }
void operator delete[](void* p, std::size_t) noexcept { std::free(p); }
void operator delete(void* p, std::align_val_t) noexcept { std::free(p); }
void operator delete[](void* p, std::align_val_t) noexcept { std::free(p); }
void operator delete(void* p, std::size_t, std::align_val_t) noexcept { std::free(p); }
void operator delete[](void* p, std::size_t, std::align_val_t) noexcept { std::free(p); }
extern "C" void __cyg_profile_func_enter(void* function, void*) {
  if (function == occupation_address) ++occupation_calls;
  for (const auto* address : scan_addresses)
    if (function == address) {
      ++scan_calls;
      if (thread_count != 17) ++threaded_scan_calls;
    }
}
extern "C" void __cyg_profile_func_exit(void*, void*) {}

// The existing cohort mock handles provider preflight, unchanged in both builds.
extern "C" {
std::int32_t LAPACKE_dpotrf_work(std::int32_t, char, std::int32_t, double*, std::int32_t);
std::int32_t LAPACKE_dpocon_work(std::int32_t, char, std::int32_t, const double*, std::int32_t,
                                 double, double*, double*, std::int32_t*);
std::int32_t LAPACKE_dsyevd_work(std::int32_t, char, char, std::int32_t, double*, std::int32_t,
                                 double*, double*, std::int32_t, std::int32_t*, std::int32_t);
void cblas_dtrsm(int, int, int, int, int, std::int32_t, std::int32_t, double, const double*,
                 std::int32_t, double*, std::int32_t);
void cblas_dgemm(int, int, int, std::int32_t, std::int32_t, std::int32_t, double, const double*,
                 std::int32_t, const double*, std::int32_t, double, double*, std::int32_t);
}
namespace {
namespace gfn = generativeqc::xtb::detail::gfn2;
namespace cpu = generativeqc::tensor::cpu;
namespace eigen = generativeqc::solver::cpu;
using Status = generativeqc_xtb_status_t;
constexpr Status ok = GENERATIVEQC_XTB_STATUS_SUCCESS;
constexpr Status invalid = GENERATIVEQC_XTB_STATUS_INVALID_ARGUMENT;
constexpr Status numerical = GENERATIVEQC_XTB_STATUS_EIGENSOLVER_FAILED;
constexpr Status internal = GENERATIVEQC_XTB_STATUS_INTERNAL_ERROR;
void require(bool v, const char* message) {
  if (!v) {
    std::fprintf(stderr, "spectral preservation: %s\n", message);
    std::abort();
  }
}
void near(long double a, long double b, const char* message, long double tol = 3e-11L) {
  require(std::abs(a - b) <= tol * (1 + std::abs(b)), message);
}
std::uint64_t digest(const void* p, std::size_t n) {
  auto* bytes = static_cast<const unsigned char*>(p);
  std::uint64_t hash = 1469598103934665603ull;
  for (std::size_t i = 0; i < n; ++i) hash = (hash ^ bytes[i]) * 1099511628211ull;
  return hash;
}
void bytes(const char* name, const void* p, std::size_t n) {
  // Every published byte, including caller-owned padding, participates.
  std::printf("%s %zu %016llx\n", name, n, static_cast<unsigned long long>(digest(p, n)));
}
using Snapshot = std::vector<unsigned char>;
Snapshot save(const void* p, std::size_t n) {
  const auto* b = static_cast<const unsigned char*>(p);
  return {b, b + n};
}
void unchanged(const Snapshot& saved, const void* p, const char* message) {
  require(std::memcmp(saved.data(), p, saved.size()) == 0, message);
}
struct Arena {
  const std::size_t size;
  unsigned char* raw;
  unsigned char* data;
  explicit Arena(std::size_t n)
      : size(n),
        raw(static_cast<unsigned char*>(allocation(n + 128, 64))),
        data(raw + 64) {
    std::memset(raw, 0xA7, n + 128);
  }
  ~Arena() { std::free(raw); }
  Arena(const Arena&) = delete;
  void guards() const {
    for (std::size_t i = 0; i < 64; ++i)
      require(raw[i] == 0xA7 && data[size + i] == 0xA7, "arena boundary overwritten");
  }
};
struct Calls {
  int potrf = 0, pocon = 0, eigen = 0, trsm = 0, gemm = 0, threads = 0;
  std::uint64_t trace = 1469598103934665603ull;
} calls;
bool preflight = false;
int fail_potrf = 0, fail_pocon = 0, fail_eigen = 0, fail_gemm = 0;
int injected_info = -7;
bool poor_condition = false;
const void* watched_wave = nullptr;
std::size_t watched_wave_size = 0;
std::uint64_t watched_wave_digest = 0;
const void* watched_results = nullptr;
std::size_t watched_results_size = 0;
std::uint64_t watched_results_digest = 0;
void watch_publication() {
  if (!watched_wave) return;
  require(digest(watched_wave, watched_wave_size) == watched_wave_digest,
          "wavefunction published before all backend calls completed");
  require(digest(watched_results, watched_results_size) == watched_results_digest,
          "thermodynamics published before all backend calls completed");
}
void trace(int operation, int n, int detail = 0) {
  watch_publication();
  for (int v : {operation, n, detail})
    calls.trace = (calls.trace ^ std::uint64_t(v)) * 1099511628211ull;
}
void reset() {
  calls = {};
  fail_potrf = fail_pocon = fail_eigen = fail_gemm = 0;
  injected_info = -7;
  poor_condition = false;
  occupation_calls = scan_calls = threaded_scan_calls = 0;
}
void summary() {
  std::printf(
      "calls %d %d %d %d %d %d occupations %zu scans %zu threaded-scans %zu trace %016llx\n",
      calls.potrf, calls.pocon, calls.eigen, calls.trsm, calls.gemm, calls.threads,
      occupation_calls, scan_calls, threaded_scan_calls,
      static_cast<unsigned long long>(calls.trace));
  require(thread_count == 17, "BLAS local thread scope was not restored");
}
int set_threads(int n) {
  const int old = thread_count;
  thread_count = n;
  if (!preflight) ++calls.threads;
  return old;
}
std::int32_t potrf(std::int32_t layout, char uplo, std::int32_t n, double* a, std::int32_t lda) {
  if (preflight) return LAPACKE_dpotrf_work(layout, uplo, n, a, lda);
  require(layout == 102 && uplo == 'L' && lda == n && thread_count == 1, "DPOTRF ABI/scope");
  trace(1, n);
  if (++calls.potrf == fail_potrf) return injected_info;
  for (int j = 0; j < n; ++j) {
    double sum = a[j + j * n];
    for (int k = 0; k < j; ++k) sum -= a[j + k * n] * a[j + k * n];
    if (sum <= 0) return j + 1;
    a[j + j * n] = std::sqrt(sum);
    for (int i = j + 1; i < n; ++i) {
      sum = a[i + j * n];
      for (int k = 0; k < j; ++k) sum -= a[i + k * n] * a[j + k * n];
      a[i + j * n] = sum / a[j + j * n];
    }
  }
  return 0;
}
std::int32_t pocon(std::int32_t layout, char uplo, std::int32_t n, const double* a,
                   std::int32_t lda, double norm, double* rcond, double* work, std::int32_t* iw) {
  if (preflight) return LAPACKE_dpocon_work(layout, uplo, n, a, lda, norm, rcond, work, iw);
  require(layout == 102 && uplo == 'L' && lda == n && norm > 0 && thread_count == 1,
          "DPOCON ABI/scope");
  trace(2, n);
  if (++calls.pocon == fail_pocon) {
    *rcond = poor_condition ? 1e-20 : 0.5;
    return poor_condition ? 0 : injected_info;
  }
  *rcond = 0.5;
  return 0;
}
std::int32_t syevd(std::int32_t layout, char vectors, char uplo, std::int32_t n, double* a,
                   std::int32_t lda, double* w, double* work, std::int32_t lwork, std::int32_t* iw,
                   std::int32_t liwork) {
  if (preflight)
    return LAPACKE_dsyevd_work(layout, vectors, uplo, n, a, lda, w, work, lwork, iw, liwork);
  require(layout == 102 && vectors == 'V' && uplo == 'L' && lda == n && n <= 3 &&
              thread_count == 1 && lwork == 37 && liwork == 18,
          "DSYEVD borrowed pointers/counts/scope");
  trace(3, n, lwork * 100 + liwork);
  if (++calls.eigen == fail_eigen) return injected_info;
  // A stack-only analytic eigensystem for the prescribed 2x2 block plus
  // independent diagonal orbitals. No production eigensolver is an oracle.
  double q[9]{};
  for (int i = 0; i < n; ++i) q[i + i * n] = 1;
  if (n >= 2) {
    const double theta = 0.5 * std::atan2(2 * a[1], a[0] - a[n + 1]);
    const double c = std::cos(theta), s = std::sin(theta);
    const double center = (a[0] + a[n + 1]) / 2;
    const double radius = std::hypot((a[0] - a[n + 1]) / 2, a[1]);
    w[0] = center - radius;
    w[1] = center + radius;
    q[0] = -s;
    q[1] = c;
    q[n] = c;
    q[n + 1] = s;
    for (int i = 2; i < n; ++i) {
      w[i] = a[i + i * n];
      for (int j = 0; j < i; ++j) near(a[i + j * n], 0, "prescribed transformed block", 1e-13L);
    }
  } else {
    w[0] = a[0];
  }
  std::copy_n(q, n * n, a);
  return 0;
}
void trsm(int layout, int side, int uplo, int trans, int diag, std::int32_t rows, std::int32_t cols,
          double alpha, const double* l, std::int32_t lda, double* b, std::int32_t ldb) {
  if (preflight)
    return cblas_dtrsm(layout, side, uplo, trans, diag, rows, cols, alpha, l, lda, b, ldb);
  require(layout == 102 && uplo == 122 && diag == 131 && rows == cols && lda == rows &&
              ldb == rows && alpha == 1 && thread_count == 1,
          "TRSM ABI/scope");
  const int n = rows;
  trace(4, n, side * 1000 + trans);
  ++calls.trsm;
  if (side == 141 && trans == 111) {
    for (int col = 0; col < n; ++col)
      for (int row = 0; row < n; ++row) {
        for (int k = 0; k < row; ++k) b[row + col * n] -= l[row + k * n] * b[k + col * n];
        b[row + col * n] /= l[row + row * n];
      }
  } else if (side == 142 && trans == 112) {
    for (int row = 0; row < n; ++row)
      for (int col = 0; col < n; ++col) {
        for (int k = 0; k < col; ++k) b[row + col * n] -= b[row + k * n] * l[col + k * n];
        b[row + col * n] /= l[col + col * n];
      }
  } else {
    require(side == 141 && trans == 112, "unexpected triangular operation");
    for (int col = 0; col < n; ++col)
      for (int row = n - 1; row >= 0; --row) {
        for (int k = row + 1; k < n; ++k) b[row + col * n] -= l[k + row * n] * b[k + col * n];
        b[row + col * n] /= l[row + row * n];
      }
  }
}
void gemm(int layout, int ta, int tb, std::int32_t m, std::int32_t n, std::int32_t k, double alpha,
          const double* a, std::int32_t lda, const double* b, std::int32_t ldb, double beta,
          double* c, std::int32_t ldc) {
  if (preflight) return cblas_dgemm(layout, ta, tb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
  require(layout == 102 && ta == 111 && tb == 112 && m == n && n == k && lda == n && ldb == n &&
              ldc == n && alpha == 1 && beta == 0 && thread_count == 1,
          "GEMM ABI/scope");
  trace(5, n);
  const bool corrupt = ++calls.gemm == fail_gemm;
  for (int col = 0; col < n; ++col)
    for (int row = 0; row < n; ++row) {
      double sum = 0;
      for (int p = 0; p < n; ++p) sum += a[row + p * n] * b[col + p * n];
      c[row + col * n] = corrupt ? std::numeric_limits<double>::quiet_NaN() : sum;
    }
}
gfn::CpuLinearAlgebraBackend provider() {
  gfn::CpuLinearAlgebraBackend result;
  std::string error;
  preflight = true;
  require(gfn::make_internal_test_lp64_backend(potrf, pocon, syevd, trsm, gemm, set_threads, result,
                                               error) == ok,
          "real shared LP64 provider admission failed");
  preflight = false;
  reset();
  return result;
}

constexpr std::array<std::int64_t, 4> orbital_offsets{0, 2, 3, 6};
constexpr std::array<std::int64_t, 4> overlap_offsets{0, 4, 5, 14};
constexpr std::array<std::int64_t, 4> matrix_offsets{0, 4, 6, 15};
constexpr std::array<std::int64_t, 4> value_offsets{0, 2, 4, 7};
constexpr std::array<std::int64_t, 4> occupation_offsets{0, 4, 6, 12};
constexpr std::array<std::int32_t, 3> spins{1, 2, 1};
constexpr std::array<double, 3> alpha{0.8, 0.7, 1.25}, beta{0.8, 0.2, 0.75};
constexpr std::array<std::size_t, 5> field_bytes{0, 192, 320, 512, 704};
constexpr std::size_t wave_bytes = 896;
struct Results {
  std::array<Status, 3> statuses;
  std::array<double, 6> chemical;
  std::array<double, 3> entropy, band, free;
};
struct Fixture {
  gfn::EigensolverWavefunctionLayout layout;
  gfn::EigensolverPlan plan;
  gfn::EigensolverOverlapCache cache;
  gfn::EigensolverWorkspace full, worker;
  gfn::EigensolverWavefunctionView wave;
  gfn::EigensolverThermodynamicsView thermo;
  gfn::CpuLinearAlgebraBackend backend = provider();
  Arena wave_arena{wave_bytes};
  std::unique_ptr<Arena> cache_arena, full_arena, worker_arena;
  Results result;
  std::array<double, 14> overlaps{};
  std::array<double, 15> hamiltonians{};
  std::string error;
  Fixture() {
    error.reserve(1024);
    std::memset(&result, 0xB6, sizeof(result));
    layout.batch_size = 3;
    layout.workspace_size_bytes = wave_bytes;
    layout.orbital_offsets = orbital_offsets.data();
    layout.orbital_offset_count = orbital_offsets.size();
    layout.spin_channels = spins.data();
    layout.spin_channel_count = spins.size();
    layout.alpha_electron_counts = alpha.data();
    layout.beta_electron_counts = beta.data();
    layout.electron_count_count = alpha.size();
    const std::array<const std::int64_t*, 5> offsets{matrix_offsets.data(), value_offsets.data(),
                                                     occupation_offsets.data(),
                                                     matrix_offsets.data(), matrix_offsets.data()};
    for (std::size_t field = 0; field < 5; ++field)
      layout.fields[field] = {field_bytes[field], offsets[field][3], offsets[field], 4};
    require(gfn::make_eigensolver_plan(layout, plan, error) == ok, "ragged plan preparation");
    cache_arena = std::make_unique<Arena>(plan.overlap_cache_size_bytes());
    full_arena = std::make_unique<Arena>(plan.workspace_size_bytes());
    worker_arena = std::make_unique<Arena>(plan.worker_workspace_size_bytes());
    require(gfn::bind_eigensolver_overlap_cache(plan, cache_arena->data, cache_arena->size, cache,
                                                error) == ok,
            "cache bind");
    require(gfn::bind_eigensolver_workspace(plan, full_arena->data, full_arena->size, full,
                                            error) == ok,
            "full workspace bind");
    require(gfn::bind_eigensolver_worker_workspace(plan, worker_arena->data, worker_arena->size,
                                                   worker, error) == ok,
            "worker bind");
    auto pointer = [&](std::size_t i) {
      return reinterpret_cast<double*>(wave_arena.data + field_bytes[i]);
    };
    wave = {wave_arena.data, wave_arena.size, pointer(0), pointer(1),
            pointer(2),      pointer(3),      pointer(4)};
    thermo = {result.statuses.data(), 3, result.chemical.data(), 6, result.entropy.data(), 3,
              result.band.data(),     3, result.free.data(),     3};
    inputs();
  }
  ~Fixture() {
    watched_wave = nullptr;
    wave_arena.guards();
    cache_arena->guards();
    full_arena->guards();
    worker_arena->guards();
  }
  void inputs(double scale = 1) {
    for (int system = 0; system < 3; ++system) {
      const int n = static_cast<int>(orbital_offsets[system + 1] - orbital_offsets[system]);
      double l[9]{}, q[9]{};
      for (int i = 0; i < n; ++i) {
        l[i * n + i] = scale * (1.25 + 0.25 * i);
        q[i * n + i] = 1;
        if (i > 0) l[i * n + i - 1] = scale * 0.125;
      }
      if (n >= 2) {
        q[0] = 0.8;
        q[1] = 0.6;
        q[n] = -0.6;
        q[n + 1] = 0.8;
      }
      double lq[9]{};
      for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j)
          for (int k = 0; k < n; ++k) lq[i * n + j] += l[i * n + k] * q[k * n + j];
      for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j) {
          double s = 0;
          for (int k = 0; k < n; ++k) s += l[i * n + k] * l[j * n + k];
          overlaps[overlap_offsets[system] + static_cast<std::int64_t>(i) * n + j] = s;
          for (int spin = 0; spin < spins[system]; ++spin) {
            double h = 0;
            for (int k = 0; k < n; ++k)
              h += lq[i * n + k] * lq[j * n + k] * (-1.0 + 1.5 * k + .2 * system + .3 * spin);
            hamiltonians[matrix_offsets[system] + static_cast<std::int64_t>(spin) * n * n +
                         static_cast<std::int64_t>(i) * n + j] = h;
          }
        }
    }
  }
  Status factor(std::uint64_t generation = 9) {
    allocation_count = 0;
    counting = true;
    const auto status =
        gfn::factor_overlap_cpu(plan, overlaps.data(), generation, backend, full, cache, error);
    counting = false;
    require(allocation_count == 0, "factor call allocated");
    return status;
  }
  Status solve(std::uint64_t generation = 9, double temperature = .2, int system = -1) {
    allocation_count = 0;
    counting = true;
    const auto status =
        system < 0 ? gfn::solve_eigensystems_cpu(plan, cache, generation, hamiltonians.data(),
                                                 temperature, backend, full, wave, thermo, error)
                   : gfn::solve_eigensystem_cpu(plan, system, cache, generation,
                                                hamiltonians.data() + matrix_offsets[system],
                                                temperature, backend, worker, wave, thermo, error);
    counting = false;
    require(allocation_count == 0, "solve call allocated");
    return status;
  }
  void watch() {
    watched_wave = wave_arena.data;
    watched_wave_size = wave_arena.size;
    watched_wave_digest = digest(watched_wave, watched_wave_size);
    watched_results = &result;
    watched_results_size = sizeof(result);
    watched_results_digest = digest(watched_results, watched_results_size);
  }
  void report(const char* tag, Status status) {
    watched_wave = nullptr;
    std::printf("%s status %d error %s\n", tag, status, error.c_str());
    bytes("cache", cache_arena->data, cache_arena->size);
    bytes("wave", wave_arena.data, wave_arena.size);
    bytes("results", &result, sizeof(result));
    summary();
  }
};

void check_failed_system(const Fixture& f, int system, const Snapshot& wave_before,
                         const Results& before) {
  require(f.result.statuses[system] == numerical, "failed peer status not published");
  for (int field = 0; field < 5; ++field) {
    const auto& layout = f.layout.fields[field];
    const auto offset = layout.offset_bytes + layout.system_offsets[system] * sizeof(double);
    const auto count =
        (layout.system_offsets[system + 1] - layout.system_offsets[system]) * sizeof(double);
    require(std::memcmp(wave_before.data() + offset, f.wave_arena.data + offset, count) == 0,
            "failed system published electronic fields");
  }
  require(std::memcmp(f.result.chemical.data() + 2 * system, before.chemical.data() + 2 * system,
                      2 * sizeof(double)) == 0,
          "failed system published chemical potentials");
  for (auto member : {&Results::entropy, &Results::band, &Results::free})
    require(
        std::memcmp(&(f.result.*member)[system], &(before.*member)[system], sizeof(double)) == 0,
        "failed system published scalar thermodynamics");
}

void test_plan() {
  Fixture f;
  require(f.plan.batch_size() == 3 && f.plan.maximum_orbitals() == 3 &&
              f.plan.total_matrix_elements() == 14 && f.plan.resident_bytes() > 0,
          "ragged dimensions");
  require(f.plan.matrix_offsets() ==
              std::vector<std::int64_t>(overlap_offsets.begin(), overlap_offsets.end()),
          "physical matrix offsets");
  std::printf("sizes %zu %zu %zu\n", f.plan.overlap_cache_size_bytes(),
              f.plan.worker_workspace_size_bytes(), f.plan.workspace_size_bytes());
  auto offset = [](const void* base, const void* p) -> std::ptrdiff_t {
    return p ? static_cast<const unsigned char*>(p) - static_cast<const unsigned char*>(base) : -1;
  };
  std::printf("cache-offsets %td %td %td\n",
              offset(f.cache.workspace_base, f.cache.cholesky_factors),
              offset(f.cache.workspace_base, f.cache.geometry_generations),
              offset(f.cache.workspace_base, f.cache.system_statuses));
  for (auto* view : {&f.full, &f.worker}) {
    for (const void* p : {static_cast<void*>(view->coefficients),
                          static_cast<void*>(view->densities),
                          static_cast<void*>(view->energy_weighted_densities),
                          static_cast<void*>(view->eigenvalues),
                          static_cast<void*>(view->occupations),
                          static_cast<void*>(view->lapack_work),
                          static_cast<void*>(view->lapack_integer_work),
                          static_cast<void*>(view->factor_staging),
                          static_cast<void*>(view->factor_generation_staging),
                          static_cast<void*>(view->factor_status_staging),
                          static_cast<void*>(view->batch_coefficients),
                          static_cast<void*>(view->batch_densities),
                          static_cast<void*>(view->batch_energy_weighted_densities),
                          static_cast<void*>(view->batch_eigenvalues),
                          static_cast<void*>(view->batch_occupations),
                          static_cast<void*>(view->batch_system_statuses),
                          static_cast<void*>(view->batch_chemical_potentials),
                          static_cast<void*>(view->batch_entropies),
                          static_cast<void*>(view->batch_band_energies),
                          static_cast<void*>(view->batch_free_energies)})
      std::printf("%td ", offset(view->workspace_base, p));
    std::puts("");
  }
  require(f.plan.overlap_cache_size_bytes() == 256 &&
              f.plan.worker_workspace_size_bytes() == 1152 && f.plan.workspace_size_bytes() == 2304,
          "exact packed sizes changed");
  auto copy = f.plan;
  auto moved = std::move(copy);
  require(moved.identity() == f.plan.identity() && !copy.sealed(), "plan lifetime identity");
  require(gfn::validate_eigensolver_overlap_cache_binding(moved, f.cache, f.error) == ok &&
              gfn::validate_eigensolver_worker_workspace_binding(moved, f.worker, f.error) == ok,
          "copied plan cannot bind original resources");
  require(f.plan.overlaps_storage(f.plan.identity(), 1) &&
              f.plan.overlaps_storage(f.plan.matrix_offsets().data(), sizeof(std::int64_t)) &&
              !f.plan.overlaps_storage(f.wave_arena.data, wave_bytes),
          "plan storage overlap ownership");
  const auto before = save(f.cache_arena->data, f.cache_arena->size);
  auto other = f.plan;
  require(gfn::make_eigensolver_plan(f.layout, other, f.error) == ok, "second plan");
  require(gfn::validate_eigensolver_overlap_cache_binding(other, f.cache, f.error) == invalid,
          "foreign plan cache admitted");
  unchanged(before, f.cache_arena->data, "cache changed on foreign identity");
  // A one-system plan with the same maximum orbital extent must have the
  // identical bounded worker requirement, independent of batch size and spin.
  const std::array<std::int64_t, 2> one_orbitals{0, 3}, one_matrices{0, 9}, one_values{0, 3},
      one_occupations{0, 6};
  auto one_layout = f.layout;
  one_layout.batch_size = 1;
  one_layout.orbital_offsets = one_orbitals.data();
  one_layout.orbital_offset_count = 2;
  one_layout.spin_channel_count = one_layout.electron_count_count = 1;
  const std::array<const std::int64_t*, 5> one_offsets{one_matrices.data(), one_values.data(),
                                                       one_occupations.data(), one_matrices.data(),
                                                       one_matrices.data()};
  for (int field = 0; field < 5; ++field) {
    one_layout.fields[field].element_count = one_offsets[field][1];
    one_layout.fields[field].system_offsets = one_offsets[field];
    one_layout.fields[field].system_offset_count = 2;
  }
  gfn::EigensolverPlan one;
  require(gfn::make_eigensolver_plan(one_layout, one, f.error) == ok &&
              one.worker_workspace_size_bytes() == f.plan.worker_workspace_size_bytes(),
          "worker storage grew with batch extent");
  f.report("plan", ok);
}

void test_setup() {
  Fixture f;
  gfn::EigensolverPlan plan;
  allocation_count = allocation_bytes = 0;
  counting = true;
  const auto status = gfn::make_eigensolver_plan(f.layout, plan, f.error);
  counting = false;
  require(status == ok && allocation_count > 0, "setup allocation accounting");
  std::printf("setup %zu bytes %zu resident %zu\n", allocation_count, allocation_bytes,
              plan.resident_bytes());
  allocation_count = 0;
  counting = true;
  auto copy = plan;
  auto moved = std::move(copy);
  counting = false;
  require(allocation_count == 0 && moved.identity() == plan.identity(),
          "copying immutable resource ownership allocated");
}

void test_binding() {
  Fixture f;
  const auto before = save(f.cache_arena->data, f.cache_arena->size);
  const auto cache_before = f.cache;
  const auto full_before = f.full;
  const auto worker_before = f.worker;
  for (int bad = 0; bad < 2; ++bad) {
    require(gfn::bind_eigensolver_overlap_cache(f.plan, f.cache_arena->data + bad,
                                                f.cache_arena->size - (bad == 0), f.cache,
                                                f.error) == invalid,
            "short/unaligned cache accepted");
    require(std::memcmp(&f.cache, &cache_before, sizeof(f.cache)) == 0,
            "failed cache bind changed descriptor");
    require(gfn::bind_eigensolver_workspace(f.plan, f.full_arena->data + bad,
                                            f.full_arena->size - (bad == 0), f.full,
                                            f.error) == invalid,
            "short/unaligned full workspace accepted");
    require(std::memcmp(&f.full, &full_before, sizeof(f.full)) == 0,
            "failed full bind changed descriptor");
    require(gfn::bind_eigensolver_worker_workspace(f.plan, f.worker_arena->data + bad,
                                                   f.worker_arena->size - (bad == 0), f.worker,
                                                   f.error) == invalid,
            "short/unaligned worker accepted");
    require(std::memcmp(&f.worker, &worker_before, sizeof(f.worker)) == 0,
            "failed worker bind changed descriptor");
  }
  unchanged(before, f.cache_arena->data, "failed binds changed persistent bytes");
  auto forged = f.cache;
  ++forged.cholesky_factors;
  require(gfn::validate_eigensolver_overlap_cache_binding(f.plan, forged, f.error) == invalid,
          "forged cache accepted");
  auto scratch = f.worker;
  ++scratch.lapack_work;
  require(gfn::validate_eigensolver_worker_workspace_binding(f.plan, scratch, f.error) == invalid,
          "forged work accepted");
  allocation_count = 0;
  counting = true;
  require(gfn::bind_eigensolver_overlap_cache(f.plan, f.cache_arena->data, f.cache_arena->size,
                                              f.cache, f.error) == ok,
          "allocation-free cache rebind");
  require(gfn::bind_eigensolver_worker_workspace(f.plan, f.worker_arena->data, f.worker_arena->size,
                                                 f.worker, f.error) == ok,
          "allocation-free worker rebind");
  counting = false;
  require(allocation_count == 0, "binding allocated");
  Arena descriptor_storage(f.full_arena->size);
  auto* overlapping = new (descriptor_storage.data) gfn::EigensolverWorkspace;
  const auto overlap_before = save(descriptor_storage.data, descriptor_storage.size);
  require(gfn::bind_eigensolver_workspace(f.plan, descriptor_storage.data, descriptor_storage.size,
                                          *overlapping, f.error) == invalid,
          "full workspace descriptor inside its numerical storage accepted");
  unchanged(overlap_before, descriptor_storage.data, "failed control-alias bind mutated storage");
  overlapping->~EigensolverWorkspace();
  descriptor_storage.guards();
  f.report("binding", ok);
}

void test_generations() {
  Fixture f;
  for (std::uint64_t generation :
       std::array<std::uint64_t, 4>{9, 9, 3, std::numeric_limits<std::uint64_t>::max()}) {
    reset();
    f.inputs(generation == 3 ? 2 : 1);
    require(f.factor(generation) == ok, "nonzero generation rejected");
    require(calls.potrf == 3 && calls.pocon == 3, "same/decreasing stamp skipped factor work");
    require(scan_calls == 3 && threaded_scan_calls == 0, "factor preflight scan work changed");
    for (int system = 0; system < 3; ++system)
      require(f.cache.geometry_generations[system] == generation &&
                  f.cache.system_statuses[system] == ok,
              "generation/status publication");
    f.report("generation", ok);
  }
  const auto before = save(f.cache_arena->data, f.cache_arena->size);
  reset();
  require(f.factor(0) == invalid, "zero generation accepted");
  unchanged(before, f.cache_arena->data, "zero generation changed cache");
  f.report("zero-generation", invalid);
}

void test_factor_backend() {
  for (int operation : {1, 2}) {
    Fixture f;
    require(f.factor() == ok, "initial factor");
    const auto before = save(f.cache_arena->data, f.cache_arena->size);
    f.inputs(2);
    reset();
    if (operation == 1)
      fail_potrf = 2;
    else
      fail_pocon = 2;
    require(f.factor(8) == internal, "negative factor info not whole-call failure");
    require(calls.potrf == 2, "failure was not after a healthy peer");
    unchanged(before, f.cache_arena->data,
              "backend failure changed cache, generation, status or padding");
    f.report("factor-backend", internal);
  }
}

void test_factor_numerical() {
  for (int mode : {0, 1, 2}) {
    Fixture f;
    require(f.factor() == ok, "initial factor");
    const auto before = save(f.cache_arena->data, f.cache_arena->size);
    f.inputs(2);
    reset();
    if (mode == 0) {
      fail_potrf = 2;
      injected_info = 1;
    }
    if (mode == 1) {
      fail_pocon = 2;
      poor_condition = true;
    }
    if (mode == 2) f.overlaps[4] = -1;
    require(f.factor(2) == ok, "numerical factor failure became call failure");
    for (int system = 0; system < 3; ++system)
      require(f.cache.geometry_generations[system] == 2 &&
                  f.cache.system_statuses[system] == (system == 1 ? numerical : ok),
              "numerical factor peer generation/status semantics");
    const auto begin = overlap_offsets[1] * sizeof(double);
    require(std::memcmp(before.data() + begin,
                        reinterpret_cast<const unsigned char*>(f.cache.cholesky_factors) + begin,
                        sizeof(double)) == 0,
            "failed factor replaced old healthy Cholesky factor");
    require(f.cache.cholesky_factors[0] == 2.5, "healthy peer factor not committed");
    f.report("factor-numerical", ok);
  }
}

void test_solve_backend() {
  Fixture f;
  require(f.factor() == ok, "factor");
  const auto wave_before = save(f.wave_arena.data, f.wave_arena.size);
  const auto result_before = save(&f.result, sizeof(f.result));
  const auto cache_before = save(f.cache_arena->data, f.cache_arena->size);
  reset();
  fail_eigen = 2;
  f.watch();
  require(f.solve() == internal, "negative DSYEVD info not whole-call failure");
  require(calls.eigen == 2 && occupation_calls == 2 && calls.gemm == 2,
          "backend failure did not follow healthy restricted peer");
  unchanged(wave_before, f.wave_arena.data, "backend failure published wavefunction");
  unchanged(result_before, &f.result, "backend failure published thermodynamics");
  unchanged(cache_before, f.cache_arena->data, "solve mutated factor cache");
  f.report("solve-backend", internal);
}

void test_solve_numerical() {
  for (int mode : {0, 1, 2, 3}) {
    Fixture f;
    require(f.factor() == ok, "factor");
    const auto wave_before = save(f.wave_arena.data, f.wave_arena.size);
    const Results result_before = f.result;
    reset();
    if (mode == 0) {
      fail_eigen = 2;
      injected_info = 1;
    }
    if (mode == 1) fail_gemm = 3;
    if (mode == 2) f.cache.geometry_generations[1] = 8;
    if (mode == 3) f.cache.system_statuses[1] = numerical;
    f.watch();
    require(f.solve() == ok, "system-local numerical failure became call failure");
    check_failed_system(f, 1, wave_before, result_before);
    require(f.result.statuses[0] == ok && f.result.statuses[2] == ok,
            "healthy peers not committed after numerical failure");
    f.report("solve-numerical", ok);
  }
}

void test_workers() {
  Fixture full, one;
  require(full.factor() == ok && one.factor() == ok, "factor");
  reset();
  full.watch();
  require(full.solve() == ok, "full batch solve");
  watched_wave = nullptr;
  require(occupation_calls == 6, "full batch must solve both spin populations for every system");
  bytes("full-wave", full.wave_arena.data, full.wave_arena.size);
  bytes("full-results", &full.result, sizeof(full.result));
  reset();
  for (int system : {2, 0, 1}) {
    const auto before = save(one.wave_arena.data, one.wave_arena.size);
    require(one.solve(9, .2, system) == ok, "bounded worker solve");
    for (int field = 0; field < 5; ++field) {
      const auto& layout = one.layout.fields[field];
      for (int peer = 0; peer < 3; ++peer) {
        if (peer == system) continue;
        const auto start = layout.offset_bytes + layout.system_offsets[peer] * sizeof(double);
        const auto count =
            (layout.system_offsets[peer + 1] - layout.system_offsets[peer]) * sizeof(double);
        require(std::memcmp(before.data() + start, one.wave_arena.data + start, count) == 0,
                "worker touched peer electronic fields");
      }
    }
  }
  require(occupation_calls == 6, "worker omitted a spin population solve");
  require(std::memcmp(full.wave_arena.data, one.wave_arena.data, wave_bytes) == 0 &&
              std::memcmp(&full.result, &one.result, sizeof(Results)) == 0,
          "worker and batch publications differ");
  one.report("workers", ok);
  const auto before = save(one.wave_arena.data, one.wave_arena.size);
  const Results result_before = one.result;
  reset();
  fail_eigen = 1;
  injected_info = 1;
  require(one.solve(9, .2, 1) == ok, "worker numerical failure contract");
  check_failed_system(one, 1, before, result_before);
  one.report("worker-failed", ok);
  const auto failed_wave = save(one.wave_arena.data, one.wave_arena.size);
  const auto failed_results = save(&one.result, sizeof(one.result));
  reset();
  fail_eigen = 1;
  require(one.solve(9, .2, 1) == internal, "worker backend failure contract");
  unchanged(failed_wave, one.wave_arena.data, "worker backend failure published electronic fields");
  unchanged(failed_results, &one.result, "worker backend failure published thermodynamics");
  one.report("worker-backend", internal);

  // Worker validation is local: an unrelated malformed input must not force a
  // batch scan or prevent an eligible system from progressing.
  reset();
  one.hamiltonians[matrix_offsets[2]] = std::numeric_limits<double>::quiet_NaN();
  require(one.solve(9, .2, 0) == ok && occupation_calls == 2 && calls.eigen == 1,
          "worker inspected unrelated Hamiltonians or merged equal populations");
  one.report("worker-local", ok);
}

void test_aliases() {
  Fixture f;
  require(f.factor() == ok, "factor");
  auto wave_before = save(f.wave_arena.data, f.wave_arena.size);
  auto cache_before = save(f.cache_arena->data, f.cache_arena->size);
  reset();
  require(gfn::factor_overlap_cpu(f.plan, f.cache.cholesky_factors, 4, f.backend, f.full, f.cache,
                                  f.error) == invalid,
          "overlap/cache alias accepted");
  require(gfn::factor_overlap_cpu(f.plan, f.overlaps.data(), 4, f.backend, f.worker, f.cache,
                                  f.error) == invalid,
          "worker-only factor scratch accepted");
  require(gfn::solve_eigensystems_cpu(f.plan, f.cache, 9, f.hamiltonians.data(), .2, f.backend,
                                      f.worker, f.wave, f.thermo, f.error) == invalid,
          "worker-only batch scratch accepted");
  require(gfn::solve_eigensystems_cpu(f.plan, f.cache, 9, f.wave.coefficients, .2, f.backend,
                                      f.full, f.wave, f.thermo, f.error) == invalid,
          "Hamiltonian/output alias accepted");
  auto output = f.wave;
  ++output.occupations;
  require(gfn::solve_eigensystems_cpu(f.plan, f.cache, 9, f.hamiltonians.data(), .2, f.backend,
                                      f.full, output, f.thermo, f.error) == invalid,
          "noncanonical output field accepted");
  auto results = f.thermo;
  results.entropies = const_cast<double*>(f.plan.alpha_electron_counts().data());
  require(gfn::solve_eigensystems_cpu(f.plan, f.cache, 9, f.hamiltonians.data(), .2, f.backend,
                                      f.full, f.wave, results, f.error) == invalid,
          "immutable plan/output alias accepted");
  results = f.thermo;
  --results.chemical_potential_capacity;
  require(gfn::solve_eigensystems_cpu(f.plan, f.cache, 9, f.hamiltonians.data(), .2, f.backend,
                                      f.full, f.wave, results, f.error) == invalid,
          "short scalar capacity accepted");
  require(f.solve(0) == invalid, "solve zero stamp accepted");
  require(f.solve(9, -.1) == invalid, "negative temperature accepted");
  auto bad = f.hamiltonians;
  bad[1] += 1;
  require(gfn::solve_eigensystems_cpu(f.plan, f.cache, 9, bad.data(), .2, f.backend, f.full, f.wave,
                                      f.thermo, f.error) == invalid,
          "asymmetric input accepted");
  unchanged(wave_before, f.wave_arena.data, "invalid solve changed output bytes");
  unchanged(cache_before, f.cache_arena->data, "invalid solve changed cache");
  require(calls.potrf == 0 && calls.eigen == 0 && calls.gemm == 0,
          "invalid admission submitted backend work");
  f.report("aliases", invalid);

  // Check exact active borrowed spans and adjacent boundaries independently of
  // the full owner; capacities larger than active spans do not create aliases.
  eigen::PreparedSymmetricEigen prepared;
  require(eigen::prepare_borrowed_symmetric_eigen(3, prepared), "borrowed preparation");
  alignas(64) double storage[64]{};
  alignas(64) std::int32_t integer_work[18]{};
  eigen::SymmetricEigenWorkBinding binding;
  require(eigen::bind_symmetric_eigen_work(prepared, {storage + 16, 37, integer_work, 18}, binding),
          "borrowed exact counts");
  eigen::BorrowedSymmetricEigenProblem problem{1, storage, 64, storage + 1, 63};
  {
    cpu::ScopedSequentialBlas scope(f.backend);
    require(eigen::execute_symmetric_eigen(prepared, cpu::bind_symmetric_eigen(f.backend), problem,
                                           binding)
                .submitted(),
            "adjacent borrowed arrays rejected");
  }
  problem.values = storage;
  require(!eigen::execute_symmetric_eigen(prepared, cpu::bind_symmetric_eigen(f.backend), problem,
                                          binding)
               .submitted(),
          "borrowed eigen matrix/value alias accepted");
  require(
      !eigen::bind_symmetric_eigen_work(prepared, {storage + 16, 36, integer_work, 18}, binding),
      "short borrowed capacity accepted");
  std::puts("borrowed boundaries passed");
}

// Independent scalar occupations: bracket fixed prescribed spectra and bisect
// directly in absolute energy with long-double arithmetic, unlike production.
std::array<long double, 3> occupations(int n, const double* energies, double count, double t,
                                       long double& mu, long double& entropy) {
  std::array<long double, 3> result{};
  entropy = mu = 0;
  if (t == 0) {
    for (int i = 0; i < n; ++i)
      result[i] = std::max(0.L, std::min(1.L, static_cast<long double>(count) - i));
    return result;
  }
  long double low = -100, high = 100;
  for (int iteration = 0; iteration < 300; ++iteration) {
    mu = (low + high) / 2;
    long double sum = 0;
    for (int i = 0; i < n; ++i) sum += 1 / (1 + std::exp((energies[i] - mu) / t));
    if (sum < count)
      low = mu;
    else
      high = mu;
  }
  for (int i = 0; i < n; ++i) {
    const long double v = 1 / (1 + std::exp((energies[i] - mu) / t));
    result[i] = v;
    if (v > 0 && v < 1) entropy -= v * std::log(v) + (1 - v) * std::log1p(-v);
  }
  return result;
}
void test_oracle() {
  for (double temperature : {0., .2}) {
    Fixture f;
    require(f.factor() == ok, "oracle factor");
    reset();
    f.watch();
    require(f.solve(9, temperature) == ok, "oracle solve");
    watched_wave = nullptr;
    require(occupation_calls == 6,
            "restricted equal populations must retain two occupation solves");
    require(scan_calls == 4 && threaded_scan_calls == 0, "solve preflight scan work changed");
    for (int system = 0; system < 3; ++system) {
      const int n = static_cast<int>(orbital_offsets[system + 1] - orbital_offsets[system]);
      const double* s = f.overlaps.data() + overlap_offsets[system];
      long double band = 0, entropy = 0;
      std::array<std::array<long double, 3>, 2> occ{};
      for (int spin = 0; spin < 2; ++spin) {
        const int channel = spins[system] == 1 ? 0 : spin;
        const double* e = f.wave.eigenvalues + value_offsets[system] + static_cast<std::int64_t>(channel) * n;
        long double mu = 0, ent = 0;
        occ[spin] =
            occupations(n, e, spin == 0 ? alpha[system] : beta[system], temperature, mu, ent);
        near(f.result.chemical[2 * system + spin], mu, "independent Fermi chemical potential");
        entropy += ent;
        long double count = 0;
        for (int i = 0; i < n; ++i) {
          near(e[i], -1 + 1.5 * i + .2 * system + .3 * channel, "prescribed generalized spectrum");
          near(f.wave.occupations[occupation_offsets[system] + static_cast<std::int64_t>(spin) * n + i], occ[spin][i],
               "independent occupations");
          count += occ[spin][i];
          band += occ[spin][i] * e[i];
        }
        near(count, spin == 0 ? alpha[system] : beta[system], "independent population");
      }
      for (int channel = 0; channel < spins[system]; ++channel) {
        const double* c = f.wave.coefficients + matrix_offsets[system] + static_cast<std::int64_t>(channel) * n * n;
        const double* h = f.hamiltonians.data() + matrix_offsets[system] + static_cast<std::int64_t>(channel) * n * n;
        const double* e = f.wave.eigenvalues + value_offsets[system] + static_cast<std::int64_t>(channel) * n;
        const double* d = f.wave.density + matrix_offsets[system] + static_cast<std::int64_t>(channel) * n * n;
        const double* wd =
            f.wave.energy_weighted_density + matrix_offsets[system] + static_cast<std::int64_t>(channel) * n * n;
        for (int row = 0; row < n; ++row)
          for (int col = 0; col < n; ++col) {
            long double hc = 0, sc = 0, metric = 0, density = 0, weighted = 0;
            for (int k = 0; k < n; ++k) {
              hc += static_cast<long double>(h[row * n + k]) * c[k * n + col];
              sc += static_cast<long double>(s[row * n + k]) * c[k * n + col];
              const auto weight = spins[system] == 1 ? occ[0][k] + occ[1][k] : occ[channel][k];
              density += static_cast<long double>(c[row * n + k]) * c[col * n + k] * weight;
              weighted += static_cast<long double>(c[row * n + k]) * c[col * n + k] * weight * e[k];
              for (int j = 0; j < n; ++j)
                metric += static_cast<long double>(c[k * n + row]) * s[k * n + j] * c[j * n + col];
            }
            near(hc, sc * e[col], "independent generalized residual");
            near(metric, row == col ? 1 : 0, "independent metric orthogonality");
            near(d[row * n + col], density, "independent density contraction");
            near(wd[row * n + col], weighted, "independent energy density contraction");
          }
      }
      near(f.result.entropy[system], entropy, "independent entropy");
      near(f.result.band[system], band, "independent band energy");
      near(f.result.free[system], band - temperature * entropy, "independent free energy");
    }
    f.report("oracle", ok);
  }
}
}  // namespace
int main(int argc, char** argv) {
  require(argc == 4, "expected scenario and instrumented occupation/scan symbol offsets");
  occupation_address = reinterpret_cast<void*>(reinterpret_cast<std::uintptr_t>(&main) +
                                               std::strtoll(argv[2], nullptr, 16));
  require(occupation_address != nullptr, "missing real occupation instrumentation");
  char* scan = argv[3];
  std::size_t index = 0;
  while (*scan) {
    require(index < scan_addresses.size(), "too many symmetry scan symbols");
    char* end = nullptr;
    const auto offset = std::strtoll(scan, &end, 16);
    require(end != scan, "invalid symmetry scan offset");
    scan_addresses[index++] =
        reinterpret_cast<void*>(reinterpret_cast<std::uintptr_t>(&main) + offset);
    scan = *end == ',' ? end + 1 : end;
  }
  require(index > 0, "missing symmetry scan instrumentation");
  const std::string scenario = argv[1];
  if (scenario == "plan")
    test_plan();
  else if (scenario == "setup")
    test_setup();
  else if (scenario == "binding")
    test_binding();
  else if (scenario == "generations")
    test_generations();
  else if (scenario == "factor-backend")
    test_factor_backend();
  else if (scenario == "factor-numerical")
    test_factor_numerical();
  else if (scenario == "solve-backend")
    test_solve_backend();
  else if (scenario == "solve-numerical")
    test_solve_numerical();
  else if (scenario == "workers")
    test_workers();
  else if (scenario == "aliases")
    test_aliases();
  else if (scenario == "oracle")
    test_oracle();
  else
    require(false, "unknown scenario");
}
