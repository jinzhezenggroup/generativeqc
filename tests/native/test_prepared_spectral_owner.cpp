// Link this consumer against only the shared owner and LP64 provider.
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <new>
#include <string>
#include <utility>

#include "solver/cpu/prepared_spectral.hpp"

namespace spectral = generativeqc::solver::cpu;
namespace cpu = generativeqc::tensor::cpu;
using Result = spectral::SpectralResult;
using Int = cpu::LapackInt;

static bool forbid_allocation = false;
static bool fail_next_allocation = false;
static std::size_t allocation_count = 0;

static void require(bool condition, const char* message) {
  if (!condition) {
    std::fprintf(stderr, "prepared spectral contract: %s\n", message);
    std::abort();
  }
}

void* operator new(std::size_t size) {
  require(!forbid_allocation, "execution/copy allocated");
  if (fail_next_allocation) {
    fail_next_allocation = false;
    throw std::bad_alloc();
  }
  ++allocation_count;
  if (void* result = std::malloc(size ? size : 1)) return result;
  throw std::bad_alloc();
}
void* operator new[](std::size_t size) { return ::operator new(size); }
void operator delete(void* value) noexcept { std::free(value); }
void operator delete[](void* value) noexcept { std::free(value); }
void operator delete(void* value, std::size_t) noexcept { std::free(value); }
void operator delete[](void* value, std::size_t) noexcept { std::free(value); }
void* operator new(std::size_t size, std::align_val_t alignment) {
  require(!forbid_allocation, "execution/copy allocated aligned memory");
  if (fail_next_allocation) {
    fail_next_allocation = false;
    throw std::bad_alloc();
  }
  ++allocation_count;
  void* result = nullptr;
  if (posix_memalign(&result, static_cast<std::size_t>(alignment), size ? size : 1) == 0)
    return result;
  throw std::bad_alloc();
}
void* operator new[](std::size_t size, std::align_val_t alignment) {
  return ::operator new(size, alignment);
}
void operator delete(void* value, std::align_val_t) noexcept { std::free(value); }
void operator delete[](void* value, std::align_val_t) noexcept { std::free(value); }
void operator delete(void* value, std::size_t, std::align_val_t) noexcept { std::free(value); }
void operator delete[](void* value, std::size_t, std::align_val_t) noexcept { std::free(value); }

struct Event {
  char operation;
  Int n;
  const double* factor;
  double* matrix;
  double* work;
  Int* integer_work;
  Int doubles;
  Int integers;
  int side;
  int transpose;
};
static std::array<Event, 128> events;
static std::size_t event_count = 0;
static bool preflight = false;
static int threads = 17, thread_changes = 0, cleanup_calls = 0;
static Int potrf_failure_order = 0, potrf_info = 0;
static Int pocon_failure_order = 0, pocon_info = 0;
static Int condition_order = 0;
static double condition = .5;
static Int eigen_info = 0;
static bool nonfinite_eigen = false;

static void reset_events() { event_count = 0; }
static void reset_faults() {
  potrf_failure_order = potrf_info = pocon_failure_order = pocon_info = 0;
  condition_order = eigen_info = 0;
  condition = .5;
  nonfinite_eigen = false;
}
static void record(Event event) {
  if (!preflight) {
    require(event_count < events.size(), "mock trace capacity");
    events[event_count++] = event;
  }
}
static int set_threads(int count) {
  ++thread_changes;
  return std::exchange(threads, count);
}
static void cleanup() { ++cleanup_calls; }

// Deliberately small independent arithmetic primitives. The provider factory still
// admits this complete cohort through its real ABI self-test before owner use.
static Int potrf(Int layout, char uplo, Int n, double* a, Int lda) {
  require(layout == 102 && uplo == 'L' && lda == n && n <= 5, "POTRF ABI");
  record({'P', n, nullptr, a, nullptr, nullptr, 0, 0, 0, 0});
  if (!preflight && n == potrf_failure_order) return potrf_info;
  for (Int column = 0; column < n; ++column) {
    for (Int row = column; row < n; ++row) {
      long double value = a[row + column * n];
      for (Int k = 0; k < column; ++k)
        value -= static_cast<long double>(a[row + k * n]) * a[column + k * n];
      if (row == column) {
        if (value <= 0) return column + 1;
        a[row + column * n] = std::sqrt(value);
      } else {
        a[row + column * n] = value / a[column + column * n];
      }
    }
  }
  return 0;
}
static Int pocon(Int layout, char uplo, Int n, const double* a, Int lda, double norm, double* rcond,
                 double* work, Int* iwork) {
  require(layout == 102 && uplo == 'L' && lda == n && norm > 0, "POCON ABI");
  record({'C', n, a, nullptr, work, iwork, 0, 0, 0, 0});
  *rcond = !preflight && n == condition_order ? condition : .5;
  return !preflight && n == pocon_failure_order ? pocon_info : 0;
}
static void trsm(int layout, int side, int uplo, int transpose, int diagonal, Int rows, Int columns,
                 double alpha, const double* a, Int lda, double* b, Int ldb) {
  require(layout == 102 && uplo == 122 && diagonal == 131 && rows == columns && lda == rows &&
              ldb == rows && alpha == 1.,
          "TRSM ABI");
  require(
      (side == 141 && (transpose == 111 || transpose == 112)) || (side == 142 && transpose == 112),
      "unexpected TRSM operation");
  record({'T', rows, a, b, nullptr, nullptr, 0, 0, side, transpose});
  const Int n = rows;
  if (side == 142) {
    // X L^T = B is a forward substitution for each row.
    for (Int row = 0; row < n; ++row)
      for (Int column = 0; column < n; ++column) {
        long double value = b[row + column * n];
        for (Int k = 0; k < column; ++k)
          value -= static_cast<long double>(b[row + k * n]) * a[column + k * n];
        b[row + column * n] = value / a[column + column * n];
      }
  } else {
    for (Int column = 0; column < n; ++column)
      for (Int step = 0; step < n; ++step) {
        const Int row = transpose == 111 ? step : n - 1 - step;
        long double value = b[row + column * n];
        if (transpose == 111)
          for (Int k = 0; k < row; ++k)
            value -= static_cast<long double>(a[row + k * n]) * b[k + column * n];
        else
          for (Int k = row + 1; k < n; ++k)
            value -= static_cast<long double>(a[k + row * n]) * b[k + column * n];
        b[row + column * n] = value / a[row + row * n];
      }
  }
}
static Int syevd(Int layout, char vectors, char uplo, Int n, double* a, Int lda, double* w,
                 double* work, Int lwork, Int* iwork, Int liwork) {
  require(layout == 102 && vectors == 'V' && uplo == 'L' && lda == n && n <= 5, "SYEVD ABI");
  require(lwork == (preflight ? 9 : 81) && liwork == (preflight ? 8 : 28),
          "query or per-member work counts replaced N=5 counts");
  record({'E', n, nullptr, a, work, iwork, lwork, liwork, 0, 0});
  if (!preflight && eigen_info) return eigen_info;
  // Maximum-pivot Jacobi, with fixed stack storage, independently diagonalizes
  // the incoming ordinary symmetric matrix. No production scalar solver is used.
  std::array<long double, 25> matrix{}, vectors_out{};
  for (Int i = 0; i < n * n; ++i) matrix[i] = a[i];
  for (Int i = 0; i < n; ++i) vectors_out[i + i * n] = 1;
  bool converged = false;
  for (int iteration = 0; iteration < 500; ++iteration) {
    Int p = 0, q = 0;
    long double largest = 0;
    for (Int j = 1; j < n; ++j)
      for (Int i = 0; i < j; ++i)
        if (std::abs(matrix[i + j * n]) > largest) {
          largest = std::abs(matrix[i + j * n]);
          p = i;
          q = j;
        }
    if (largest < 1e-18L) {
      converged = true;
      break;
    }
    const long double apq = matrix[p + q * n];
    const long double tau = (matrix[q + q * n] - matrix[p + p * n]) / (2 * apq);
    const long double t = std::copysign(1.L, tau) / (std::abs(tau) + std::sqrt(1 + tau * tau));
    const long double c = 1 / std::sqrt(1 + t * t), s = t * c;
    matrix[p + p * n] -= t * apq;
    matrix[q + q * n] += t * apq;
    matrix[p + q * n] = matrix[q + p * n] = 0;
    for (Int k = 0; k < n; ++k) {
      if (k != p && k != q) {
        const long double mkp = matrix[k + p * n], mkq = matrix[k + q * n];
        matrix[k + p * n] = matrix[p + k * n] = c * mkp - s * mkq;
        matrix[k + q * n] = matrix[q + k * n] = s * mkp + c * mkq;
      }
      const long double vkp = vectors_out[k + p * n], vkq = vectors_out[k + q * n];
      vectors_out[k + p * n] = c * vkp - s * vkq;
      vectors_out[k + q * n] = s * vkp + c * vkq;
    }
  }
  require(converged, "independent mock eigensolver convergence");
  std::array<Int, 5> order{0, 1, 2, 3, 4};
  std::sort(order.begin(), order.begin() + n,
            [&](Int i, Int j) { return matrix[i + i * n] < matrix[j + j * n]; });
  for (Int j = 0; j < n; ++j) {
    w[j] = matrix[order[j] + order[j] * n];
    for (Int i = 0; i < n; ++i) a[i + j * n] = vectors_out[i + order[j] * n];
  }
  if (!preflight && nonfinite_eigen) w[0] = std::numeric_limits<double>::quiet_NaN();
  return 0;
}
static void gemm(int layout, int ta, int tb, Int m, Int n, Int k, double alpha, const double* a,
                 Int lda, const double* b, Int ldb, double beta, double* c, Int ldc) {
  require(preflight && layout == 102 && ta == 111 && tb == 112 && m == 1 && n == 1 && k == 1 &&
              lda == 1 && ldb == 1 && ldc == 1 && alpha == 1 && beta == 0,
          "owner unexpectedly used GEMM or provider preflight changed");
  c[0] = a[0] * b[0];
}
static cpu::CpuLinearAlgebraBackend make_backend() {
  cpu::CpuLinearAlgebraBackend result;
  std::string error;
  preflight = true;
  require(
      cpu::prepare_internal_test_lp64_backend(potrf, pocon, syevd, trsm, gemm, set_threads, result,
                                              error, cleanup) == cpu::Lp64BackendStatus::success,
      "LP64 preflight");
  preflight = false;
  require(result.ready() && !result.production(), "mock provider provenance");
  return result;
}

struct Fixture {
  spectral::PreparedSpectralPlan plan;
  cpu::CpuLinearAlgebraBackend backend = make_backend();
  alignas(64) std::array<unsigned char, 448> storage;
  alignas(64) spectral::SpectralOverlapCache cache;
  alignas(64) std::array<double, 30> overlap{}, staged_factors{};
  std::array<std::uint64_t, 3> staged_generations{};
  std::array<std::int32_t, 3> staged_statuses{};
  std::array<double, 25> h{}, coefficients{};
  std::array<double, 5> eigenvalues{};
  std::array<double, 81> lapack_work{};
  std::array<Int, 28> lapack_integers{};
  spectral::SpectralFactorWorkspace factor_work;
  spectral::SpectralWorkspace solve_work;
  std::string error;

  Fixture() {
    reset_faults();
    error.reserve(512);
    const std::int64_t offsets[]{0, 2, 7, 8};
    require(
        spectral::prepare_spectral_plan(offsets, 4, 1e-9, {7, 41}, plan, error) == Result::success,
        "prepare ragged [2,5,1]");
    storage.fill(0xa5);
    require(spectral::bind_spectral_overlap_cache(plan, storage.data(), storage.size(), cache,
                                                  error) == Result::success,
            "bind fixture");
    factor_work = {staged_factors.data(),  30, staged_generations.data(), 3,
                   staged_statuses.data(), 3,  lapack_work.data(),        81,
                   lapack_integers.data(), 28};
    solve_work = {coefficients.data(), 25, eigenvalues.data(),     5,
                  lapack_work.data(),  81, lapack_integers.data(), 28};
    for (int system = 0; system < 3; ++system) {
      const int n = dimension(system);
      for (int i = 0; i < n; ++i) overlap[plan.matrix_offsets()[system] + i * n + i] = 1;
    }
    h[0] = 2;
    h[1] = 1;
    h[2] = 1;
    h[3] = 2;
    reset_events();
  }
  int dimension(int system) const {
    return plan.orbital_offsets()[system + 1] - plan.orbital_offsets()[system];
  }
  Result factor(std::uint64_t generation = 19) {
    return spectral::factor_spectral_overlaps(plan, overlap.data(), generation, backend,
                                              factor_work, cache, error);
  }
  Result solve(int system = 0, std::uint64_t generation = 19) {
    return spectral::solve_prepared_spectrum(plan, system, cache, generation, h.data(), backend,
                                             solve_work);
  }
};

template <class T>
static T* wrapping_pointer() {
  return reinterpret_cast<T*>(std::numeric_limits<std::uintptr_t>::max() &
                              ~(std::uintptr_t(alignof(T)) - 1));
}
static void* wrapping_cache() {
  return reinterpret_cast<void*>(std::numeric_limits<std::uintptr_t>::max() & ~std::uintptr_t(63));
}
static bool same_cache(const spectral::SpectralOverlapCache& a,
                       const spectral::SpectralOverlapCache& b) {
  return a.workspace_base == b.workspace_base && a.workspace_size_bytes == b.workspace_size_bytes &&
         a.factors == b.factors && a.generations == b.generations && a.statuses == b.statuses &&
         a.plan_identity == b.plan_identity;
}

static void test_plan() {
  spectral::PreparedSpectralPlan plan;
  require(!plan.sealed() && !plan.identity() && plan.batch_size() == 0 &&
              plan.orbital_offsets().empty() && plan.matrix_offsets().empty() &&
              plan.cache_size_bytes() == 0 && plan.resident_bytes() == 0 &&
              !plan.symmetric_eigen().ready(),
          "default plan inspection");
  std::string error;
  error.reserve(512);
  std::array<std::int64_t, 4> offsets{0, 2, 7, 8};
  reset_events();
  require(spectral::prepare_spectral_plan(offsets.data(), offsets.size(), 1e-9, {7, 41}, plan,
                                          error) == Result::success,
          "plan preparation");
  require(event_count == 0 && thread_changes == 0, "prepare initialized/called provider");
  require(plan.batch_size() == 3 && plan.total_matrix_elements() == 30 && plan.maximum_order() == 5,
          "ragged dimensions");
  require(plan.factor_offset_bytes() == 0 && plan.generation_offset_bytes() == 256 &&
              plan.status_offset_bytes() == 320 && plan.cache_size_bytes() == 384,
          "exact 64-byte cache packing");
  require(plan.status_encoding().success == 7 && plan.status_encoding().numerical_failure == 41 &&
              plan.minimum_overlap_rcond() == 1e-9,
          "caller numerical/status policy copied");
  require(plan.symmetric_eigen().required().doubles == 81 &&
              plan.symmetric_eigen().required().integers == 28 &&
              plan.symmetric_eigen().maximum_order() == 5,
          "max-system work dimensions");
  offsets.fill(-99);
  const std::array<std::int64_t, 4> expected_orbitals{0, 2, 7, 8}, expected_matrices{0, 4, 29, 30};
  require(std::equal(plan.orbital_offsets().begin(), plan.orbital_offsets().end(),
                     expected_orbitals.begin()) &&
              std::equal(plan.matrix_offsets().begin(), plan.matrix_offsets().end(),
                         expected_matrices.begin()),
          "input metadata was borrowed");
  const auto* identity = plan.identity();
  const auto* data = plan.orbital_offsets().data();
  const auto allocations = allocation_count;
  forbid_allocation = true;
  auto copy = plan;
  spectral::PreparedSpectralPlan assigned;
  assigned = copy;
  auto moved = std::move(assigned);
  plan = {};
  copy = {};
  forbid_allocation = false;
  require(allocation_count == allocations && moved.identity() == identity &&
              moved.orbital_offsets().data() == data && !assigned.sealed(),
          "O(1) shared plan lifetime");
  plan = moved;
  require(plan.overlaps_storage(plan.identity(), 1) &&
              plan.overlaps_storage(plan.orbital_offsets().data(), sizeof(std::int64_t)) &&
              plan.overlaps_storage(plan.matrix_offsets().data(), sizeof(std::int64_t)) &&
              !plan.overlaps_storage(expected_orbitals.data(), sizeof(expected_orbitals)) &&
              plan.overlaps_storage(wrapping_pointer<double>(), sizeof(double)),
          "metadata range ownership");
  auto rejected = [&](const std::int64_t* input, std::size_t count, double rcond = 1e-9,
                      spectral::SpectralStatusEncoding encoding = {7, 41}) {
    require(spectral::prepare_spectral_plan(input, count, rcond, encoding, plan, error) ==
                Result::invalid_argument,
            "invalid plan admitted");
    require(plan.identity() == identity && plan.orbital_offsets().data() == data && !error.empty(),
            "failed plan replacement changed identity/metadata");
  };
  rejected(nullptr, 4);
  rejected(expected_orbitals.data(), 1);
  rejected(wrapping_pointer<std::int64_t>(), 4);
  rejected(reinterpret_cast<const std::int64_t*>(
               reinterpret_cast<const char*>(expected_orbitals.data()) + 1),
           4);
  for (const auto bad : {std::array<std::int64_t, 4>{1, 2, 7, 8},
                         {0, 2, 2, 8},
                         {0, 2, 1, 8},
                         {0, -1, 7, 8},
                         {0, 2, 7, -1}})
    rejected(bad.data(), bad.size());
  const std::int64_t too_large[]{0, std::int64_t(std::numeric_limits<Int>::max()) + 1};
  const std::int64_t work_overflow[]{0, 32768};
  rejected(too_large, 2);
  rejected(work_overflow, 2);
  rejected(expected_orbitals.data(), std::numeric_limits<std::size_t>::max());
  for (double rcond : {0., -1., 1., std::numeric_limits<double>::infinity(),
                       std::numeric_limits<double>::quiet_NaN()})
    rejected(expected_orbitals.data(), 4, rcond);
  rejected(expected_orbitals.data(), 4, 1e-9, {7, 7});
  fail_next_allocation = true;
  require(spectral::prepare_spectral_plan(expected_orbitals.data(), 4, 1e-9, {7, 41}, plan,
                                          error) == Result::allocation_failure &&
              !fail_next_allocation && plan.identity() == identity,
          "allocation-failed replacement");
  spectral::PreparedSpectralPlan small, large;
  const std::int64_t small_offsets[]{0, 1}, large_offsets[]{0, 30000};
  require(spectral::prepare_spectral_plan(small_offsets, 2, 1e-9, {7, 41}, small, error) ==
                  Result::success &&
              spectral::prepare_spectral_plan(large_offsets, 2, 1e-9, {7, 41}, large, error) ==
                  Result::success &&
              small.resident_bytes() == large.resident_bytes() && large.resident_bytes() < 1024,
          "plan allocated numerical work instead of metadata only");
}

static void test_binding() {
  Fixture f;
  require(f.cache.factors == reinterpret_cast<double*>(f.storage.data()) &&
              f.cache.generations == reinterpret_cast<std::uint64_t*>(f.storage.data() + 256) &&
              f.cache.statuses == reinterpret_cast<std::int32_t*>(f.storage.data() + 320) &&
              f.cache.plan_identity == f.plan.identity(),
          "canonical binding pointers");
  for (int i = 0; i < 30; ++i) require(f.cache.factors[i] == 0, "factor initialization");
  for (int i = 0; i < 3; ++i)
    require(f.cache.generations[i] == 0 && f.cache.statuses[i] == 41,
            "custom unready initialization");
  for (std::size_t i = 0; i < f.storage.size(); ++i)
    if ((i >= 240 && i < 256) || (i >= 280 && i < 320) || i >= 332)
      require(f.storage[i] == 0xa5, "binding wrote cache padding or spare capacity");
  const auto retained = f.cache;
  const auto bytes = f.storage;
  auto rejected = [&](void* storage, std::size_t size) {
    require(spectral::bind_spectral_overlap_cache(f.plan, storage, size, f.cache, f.error) ==
                    Result::invalid_argument &&
                same_cache(f.cache, retained) && f.storage == bytes,
            "failed bind changed old binding/storage");
  };
  rejected(nullptr, 384);
  rejected(f.storage.data(), 383);
  rejected(f.storage.data() + 1, 384);
  rejected(wrapping_cache(), 384);
  rejected(&f.cache, 384);
  for (const void* metadata : {static_cast<const void*>(f.plan.identity()),
                               static_cast<const void*>(f.plan.orbital_offsets().data()),
                               static_cast<const void*>(f.plan.matrix_offsets().data())})
    rejected(
        reinterpret_cast<void*>(reinterpret_cast<std::uintptr_t>(metadata) & ~std::uintptr_t(63)),
        384);
  alignas(64) spectral::PreparedSpectralPlan aligned_plan = f.plan;
  require(spectral::bind_spectral_overlap_cache(aligned_plan, &aligned_plan, 384, f.cache,
                                                f.error) == Result::invalid_argument &&
              same_cache(f.cache, retained),
          "bind aliases plan descriptor");
  alignas(64) std::string aligned_error;
  aligned_error.reserve(512);
  require(spectral::bind_spectral_overlap_cache(f.plan, &aligned_error, 384, f.cache,
                                                aligned_error) == Result::invalid_argument &&
              same_cache(f.cache, retained),
          "bind aliases error descriptor");
  auto copy = f.plan;
  f.plan = {};
  require(spectral::validate_spectral_overlap_cache(copy, f.cache, f.error) == Result::success,
          "retained plan copy lost cache lifetime");
  f.plan = copy;
  spectral::PreparedSpectralPlan equivalent;
  require(spectral::prepare_spectral_plan(copy.orbital_offsets().data(), 4, 1e-9, {7, 41},
                                          equivalent, f.error) == Result::success,
          "second independent plan");
  require(spectral::validate_spectral_overlap_cache(equivalent, f.cache, f.error) ==
              Result::invalid_argument,
          "equivalent dimensions substituted for plan identity");
  for (int mutation = 0; mutation < 7; ++mutation) {
    auto bad = f.cache;
    if (mutation == 0) ++bad.factors;
    if (mutation == 1) ++bad.generations;
    if (mutation == 2) ++bad.statuses;
    if (mutation == 3) bad.workspace_size_bytes = 383;
    if (mutation == 4) bad.workspace_base = f.storage.data() + 1;
    if (mutation == 5) bad.plan_identity = nullptr;
    if (mutation == 6) bad.workspace_base = wrapping_cache();
    require(
        spectral::validate_spectral_overlap_cache(f.plan, bad, f.error) == Result::invalid_argument,
        "forged cache accepted");
  }
  require(f.storage == bytes && event_count == 0, "binding validation performed work");
  // A canonical-looking descriptor embedded in its own claimed factor storage
  // must be rejected before bind/validation/execution can overwrite the control.
  alignas(64) std::array<unsigned char, 384> aliased{};
  auto* embedded = new (aliased.data() + 64)
      spectral::SpectralOverlapCache{aliased.data(),
                                     aliased.size(),
                                     reinterpret_cast<double*>(aliased.data()),
                                     reinterpret_cast<std::uint64_t*>(aliased.data() + 256),
                                     reinterpret_cast<std::int32_t*>(aliased.data() + 320),
                                     f.plan.identity()};
  const auto aliased_bytes = aliased;
  require(spectral::validate_spectral_overlap_cache(f.plan, *embedded, f.error) ==
              Result::invalid_argument,
          "canonical cache overlaps its descriptor");
  require(spectral::factor_spectral_overlaps(f.plan, f.overlap.data(), 19, f.backend, f.factor_work,
                                             *embedded, f.error) == Result::invalid_argument,
          "factor accepted embedded cache descriptor");
  require(spectral::solve_prepared_spectrum(f.plan, 0, *embedded, 19, f.h.data(), f.backend,
                                            f.solve_work) == Result::invalid_argument &&
              aliased == aliased_bytes && event_count == 0,
          "solve accepted or changed embedded cache descriptor");
  embedded->~SpectralOverlapCache();
}

static void test_factor_admission() {
  Fixture f;
  require(f.factor() == Result::success, "initial factor");
  const auto cache = f.storage;
  auto rejected = [&](const spectral::SpectralFactorWorkspace& work, const double* input,
                      const spectral::SpectralOverlapCache& binding,
                      std::uint64_t generation = 19) {
    const auto factors = f.staged_factors;
    const auto generations = f.staged_generations;
    const auto statuses = f.staged_statuses;
    const auto lapack = f.lapack_work;
    const auto integers = f.lapack_integers;
    reset_events();
    require(spectral::factor_spectral_overlaps(f.plan, input, generation, f.backend, work, binding,
                                               f.error) == Result::invalid_argument,
            "invalid factor spans admitted");
    require(event_count == 0 && f.storage == cache && f.staged_factors == factors &&
                f.staged_generations == generations && f.staged_statuses == statuses &&
                f.lapack_work == lapack && f.lapack_integers == integers,
            "rejected factor changed cache/scratch or called provider");
  };
  rejected(f.factor_work, nullptr, f.cache);
  rejected(f.factor_work, wrapping_pointer<double>(), f.cache);
  rejected(f.factor_work,
           reinterpret_cast<const double*>(reinterpret_cast<const char*>(f.overlap.data()) + 1),
           f.cache);
  rejected(f.factor_work, f.overlap.data(), f.cache, 0);
  for (int mutation = 0; mutation < 27; ++mutation) {
    auto bad = f.factor_work;
    if (mutation == 0) --bad.factor_capacity;
    if (mutation == 1) --bad.generation_capacity;
    if (mutation == 2) --bad.status_capacity;
    if (mutation == 3) --bad.lapack_work_capacity;
    if (mutation == 4) --bad.lapack_integer_work_capacity;
    if (mutation == 5) bad.factors = nullptr;
    if (mutation == 6) bad.generations = nullptr;
    if (mutation == 7) bad.statuses = nullptr;
    if (mutation == 8) bad.lapack_work = nullptr;
    if (mutation == 9) bad.lapack_integer_work = nullptr;
    if (mutation == 10) bad.factors = wrapping_pointer<double>();
    if (mutation == 11) bad.generations = wrapping_pointer<std::uint64_t>();
    if (mutation == 12) bad.statuses = wrapping_pointer<std::int32_t>();
    if (mutation == 13) bad.lapack_work = wrapping_pointer<double>();
    if (mutation == 14) bad.lapack_integer_work = wrapping_pointer<Int>();
    if (mutation == 15) bad.factors = f.overlap.data();
    if (mutation == 16) bad.factors = f.cache.factors;
    if (mutation == 17) bad.generations = reinterpret_cast<std::uint64_t*>(bad.factors);
    if (mutation == 18) bad.statuses = reinterpret_cast<std::int32_t*>(bad.generations);
    if (mutation == 19) bad.lapack_work = bad.factors;
    if (mutation == 20) bad.lapack_integer_work = reinterpret_cast<Int*>(bad.lapack_work);
    if (mutation == 21) bad.factors = reinterpret_cast<double*>(&bad);
    if (mutation == 22)
      bad.factors = reinterpret_cast<double*>(reinterpret_cast<char*>(bad.factors) + 1);
    if (mutation == 23)
      bad.generations =
          reinterpret_cast<std::uint64_t*>(reinterpret_cast<char*>(bad.generations) + 1);
    if (mutation == 24)
      bad.statuses = reinterpret_cast<std::int32_t*>(reinterpret_cast<char*>(bad.statuses) + 1);
    if (mutation == 25)
      bad.lapack_work = reinterpret_cast<double*>(reinterpret_cast<char*>(bad.lapack_work) + 1);
    if (mutation == 26)
      bad.lapack_integer_work =
          reinterpret_cast<Int*>(reinterpret_cast<char*>(bad.lapack_integer_work) + 1);
    rejected(bad, f.overlap.data(), f.cache);
  }
  rejected(f.factor_work, f.cache.factors, f.cache);
  rejected(f.factor_work, reinterpret_cast<const double*>(&f.backend), f.cache);
  rejected(f.factor_work, reinterpret_cast<const double*>(&f.plan), f.cache);
  rejected(f.factor_work, reinterpret_cast<const double*>(&f.cache), f.cache);
  rejected(f.factor_work, reinterpret_cast<const double*>(&f.error), f.cache);
  rejected(f.factor_work, reinterpret_cast<const double*>(f.plan.orbital_offsets().data()),
           f.cache);
  rejected(f.factor_work, reinterpret_cast<const double*>(f.plan.matrix_offsets().data()), f.cache);
  rejected(f.factor_work, reinterpret_cast<const double*>(f.plan.identity()), f.cache);
  auto forged = f.cache;
  forged.generations = reinterpret_cast<std::uint64_t*>(forged.factors);
  rejected(f.factor_work, f.overlap.data(), forged);
  for (double value :
       {std::numeric_limits<double>::quiet_NaN(), std::numeric_limits<double>::infinity()}) {
    f.overlap[29] = value;
    rejected(f.factor_work, f.overlap.data(), f.cache);
  }
  f.overlap[29] = 1;
  f.overlap[5] = .1;
  rejected(f.factor_work, f.overlap.data(), f.cache);
  f.overlap[5] = 0;
  cpu::CpuLinearAlgebraBackend unavailable;
  reset_events();
  require(
      spectral::factor_spectral_overlaps(f.plan, f.overlap.data(), 19, unavailable, f.factor_work,
                                         f.cache, f.error) == Result::backend_unavailable &&
          event_count == 0 && f.storage == cache,
      "unready factor backend");
  spectral::PreparedSpectralPlan empty;
  require(spectral::factor_spectral_overlaps(empty, f.overlap.data(), 19, f.backend, f.factor_work,
                                             f.cache, f.error) == Result::invalid_argument &&
              event_count == 0 && f.storage == cache,
          "unsealed factor plan");
}

static void test_solve_admission() {
  Fixture f;
  require(f.factor() == Result::success, "initial factor");
  const auto cache = f.storage;
  auto rejected = [&](const spectral::SpectralWorkspace& work, const double* h,
                      const spectral::SpectralOverlapCache& binding, int system = 0,
                      std::uint64_t generation = 19, Result expected = Result::invalid_argument) {
    const auto coefficients = f.coefficients;
    const auto values = f.eigenvalues;
    const auto lapack = f.lapack_work;
    const auto integers = f.lapack_integers;
    reset_events();
    require(spectral::solve_prepared_spectrum(f.plan, system, binding, generation, h, f.backend,
                                              work) == expected,
            "invalid/stale solve admitted");
    require(event_count == 0 && f.storage == cache && f.coefficients == coefficients &&
                f.eigenvalues == values && f.lapack_work == lapack && f.lapack_integers == integers,
            "rejected solve changed output/cache or called provider");
  };
  for (int mutation = 0; mutation < 25; ++mutation) {
    auto bad = f.solve_work;
    if (mutation == 0) bad.coefficient_capacity = 3;
    if (mutation == 1) bad.eigenvalue_capacity = 1;
    if (mutation == 2) bad.lapack_work_capacity = 80;
    if (mutation == 3) bad.lapack_integer_work_capacity = 27;
    if (mutation == 4) bad.coefficients = nullptr;
    if (mutation == 5) bad.eigenvalues = nullptr;
    if (mutation == 6) bad.lapack_work = nullptr;
    if (mutation == 7) bad.lapack_integer_work = nullptr;
    if (mutation == 8) bad.coefficients = wrapping_pointer<double>();
    if (mutation == 9) bad.eigenvalues = wrapping_pointer<double>();
    if (mutation == 10) bad.lapack_work = wrapping_pointer<double>();
    if (mutation == 11) bad.lapack_integer_work = wrapping_pointer<Int>();
    if (mutation == 12) bad.coefficients = f.h.data();
    if (mutation == 13) bad.coefficients = f.cache.factors;
    if (mutation == 14) bad.eigenvalues = bad.coefficients;
    if (mutation == 15) bad.lapack_work = bad.coefficients;
    if (mutation == 16) bad.lapack_integer_work = reinterpret_cast<Int*>(bad.lapack_work);
    if (mutation == 17) bad.coefficients = reinterpret_cast<double*>(&bad);
    if (mutation == 18) bad.coefficients = reinterpret_cast<double*>(&f.plan);
    if (mutation == 19) bad.coefficients = reinterpret_cast<double*>(&f.backend);
    if (mutation == 20) bad.coefficients = reinterpret_cast<double*>(&f.cache);
    if (mutation == 21)
      bad.coefficients = reinterpret_cast<double*>(reinterpret_cast<char*>(bad.coefficients) + 1);
    if (mutation == 22)
      bad.eigenvalues = reinterpret_cast<double*>(reinterpret_cast<char*>(bad.eigenvalues) + 1);
    if (mutation == 23)
      bad.lapack_work = reinterpret_cast<double*>(reinterpret_cast<char*>(bad.lapack_work) + 1);
    if (mutation == 24)
      bad.lapack_integer_work =
          reinterpret_cast<Int*>(reinterpret_cast<char*>(bad.lapack_integer_work) + 1);
    rejected(bad, f.h.data(), f.cache);
  }
  rejected(f.solve_work, nullptr, f.cache);
  rejected(f.solve_work, wrapping_pointer<double>(), f.cache);
  rejected(f.solve_work,
           reinterpret_cast<const double*>(reinterpret_cast<const char*>(f.h.data()) + 1), f.cache);
  rejected(f.solve_work, f.cache.factors, f.cache);
  rejected(f.solve_work, reinterpret_cast<const double*>(f.plan.orbital_offsets().data()), f.cache);
  rejected(f.solve_work, reinterpret_cast<const double*>(f.plan.identity()), f.cache);
  rejected(f.solve_work, f.h.data(), f.cache, -1);
  rejected(f.solve_work, f.h.data(), f.cache, 3);
  rejected(f.solve_work, f.h.data(), f.cache, 0, 0);
  auto forged = f.cache;
  ++forged.factors;
  rejected(f.solve_work, f.h.data(), forged);
  f.h[0] = std::numeric_limits<double>::quiet_NaN();
  rejected(f.solve_work, f.h.data(), f.cache);
  // Generation rejection precedes even reading the Hamiltonian values.
  rejected(f.solve_work, f.h.data(), f.cache, 0, 18, Result::numerical_failure);
  f.h[0] = 2;
  f.h[1] = 5;
  rejected(f.solve_work, f.h.data(), f.cache);
  f.h[1] = 1;
  cpu::CpuLinearAlgebraBackend unavailable;
  reset_events();
  require(spectral::solve_prepared_spectrum(f.plan, 0, f.cache, 19, f.h.data(), unavailable,
                                            f.solve_work) == Result::backend_unavailable &&
              event_count == 0 && f.storage == cache,
          "unready solve backend");
  spectral::PreparedSpectralPlan empty;
  require(spectral::solve_prepared_spectrum(empty, 0, f.cache, 19, f.h.data(), f.backend,
                                            f.solve_work) == Result::invalid_argument &&
              event_count == 0 && f.storage == cache,
          "unsealed solve plan");
}

// Construct H=B^T diag(lambda) B, S=B^T B using a dense, invertible,
// nontriangular B. Thus the spectrum is prescribed before the factor/solve
// algorithm is called; no Cholesky reduction or eigensolver produces the oracle.
static void prescribed_problem(int n, int variant, double* h, double* s,
                               std::array<double, 5>& spectrum) {
  std::array<long double, 25> b{};
  for (int row = 0; row < n; ++row) {
    spectrum[row] = -1.75 + .85 * row + 1.3 * variant;
    for (int column = 0; column < n; ++column)
      b[row * n + column] = row == column
                                ? 1.4L + .2L * row
                                : .09L * (row + 1) / (column + 2) + .025L * (column - row);
  }
  for (int row = 0; row < n; ++row)
    for (int column = 0; column < n; ++column) {
      long double overlap = 0, hamiltonian = 0;
      for (int k = 0; k < n; ++k) {
        const auto product = b[k * n + row] * b[k * n + column];
        overlap += product;
        hamiltonian += spectrum[k] * product;
      }
      h[row * n + column] = hamiltonian;
      s[row * n + column] = overlap;
    }
}
static void verify_oracle(int n, const double* h, const double* s, const double* c,
                          const double* values, const std::array<double, 5>& expected) {
  for (int column = 0; column < n; ++column) {
    require(std::abs(values[column] - expected[column]) < 2e-11, "prescribed eigenvalue oracle");
    for (int row = 0; row < n; ++row) {
      long double hc = 0, sc = 0, metric = 0;
      for (int k = 0; k < n; ++k) {
        hc += static_cast<long double>(h[row * n + k]) * c[k + column * n];
        sc += static_cast<long double>(s[row * n + k]) * c[k + column * n];
        for (int l = 0; l < n; ++l)
          metric += static_cast<long double>(c[k + row * n]) * s[k * n + l] * c[l + column * n];
      }
      require(std::abs(hc - sc * values[column]) < 3e-11L, "independent H C = S C lambda residual");
      require(std::abs(metric - (row == column ? 1 : 0)) < 3e-12L,
              "independent C^T S C = I residual");
    }
  }
}
static void verify_trace(const Fixture& f, int system, std::size_t expected_count = 4) {
  require(event_count == expected_count && events[0].operation == 'T' &&
              events[1].operation == 'T' && events[2].operation == 'E',
          "TRSM/TRSM/SYEVD trace");
  require(events[0].side == 141 && events[0].transpose == 111 && events[1].side == 142 &&
              events[1].transpose == 112,
          "reduction dispatch arguments");
  if (expected_count == 4)
    require(events[3].operation == 'T' && events[3].side == 141 && events[3].transpose == 112,
            "TRSM recovery trace");
  for (std::size_t i = 0; i < expected_count; ++i) {
    require(events[i].n == f.dimension(system) && events[i].matrix == f.coefficients.data(),
            "borrowed matrix identity/dimension");
    if (events[i].operation == 'T')
      require(events[i].factor == f.cache.factors + f.plan.matrix_offsets()[system],
              "cached factor pointer replaced");
  }
  require(events[2].work == f.lapack_work.data() &&
              events[2].integer_work == f.lapack_integers.data() && events[2].doubles == 81 &&
              events[2].integers == 28,
          "borrowed max-system scratch");
}
static void test_numerics() {
  Fixture f;
  std::array<std::array<double, 25>, 3> hamiltonians{};
  std::array<std::array<double, 5>, 3> spectra{};
  for (int system = 0; system < 3; ++system)
    prescribed_problem(f.dimension(system), 0, hamiltonians[system].data(),
                       f.overlap.data() + f.plan.matrix_offsets()[system], spectra[system]);
  const auto input = f.overlap;
  const auto allocations = allocation_count;
  const auto setters = thread_changes;
  forbid_allocation = true;
  const auto factor_result = f.factor();
  forbid_allocation = false;
  require(factor_result == Result::success && allocation_count == allocations &&
              thread_changes == setters && threads == 17 && f.overlap == input,
          "factor allocated/mutated input or thread state");
  require(event_count == 6, "batch factor work count");
  for (int system = 0; system < 3; ++system) {
    const int n = f.dimension(system);
    const int offset = f.plan.matrix_offsets()[system];
    require(f.cache.generations[system] == 19 && f.cache.statuses[system] == 7,
            "caller status/generation success encoding");
    require(events[2 * system].operation == 'P' && events[2 * system + 1].operation == 'C' &&
                events[2 * system].n == n &&
                events[2 * system].matrix == f.staged_factors.data() + offset &&
                events[2 * system + 1].work == f.lapack_work.data() &&
                events[2 * system + 1].integer_work == f.lapack_integers.data(),
            "factor staging dispatch");
    for (int column = 0; column < n; ++column)
      for (int row = 0; row < n; ++row) {
        long double reconstructed = 0;
        for (int k = 0; k < n; ++k)
          reconstructed += static_cast<long double>(f.cache.factors[offset + row + k * n]) *
                           f.cache.factors[offset + column + k * n];
        require(std::abs(reconstructed - f.overlap[offset + row * n + column]) < 3e-12L,
                "cached L L^T factor residual");
        if (row < column)
          require(f.cache.factors[offset + row + column * n] == 0, "upper factor not cleared");
      }
  }
  const auto committed = f.storage;
  // Three independently prescribed spectra for every member, sharing one factor.
  for (int system = 0; system < 3; ++system)
    for (int variant = 0; variant < 3; ++variant) {
      std::array<double, 25> repeated_overlap{};
      std::array<double, 5> expected{};
      prescribed_problem(f.dimension(system), variant, f.h.data(), repeated_overlap.data(),
                         expected);
      const auto h = f.h;
      f.coefficients.fill(-987.);
      f.eigenvalues.fill(-654.);
      reset_events();
      const auto count = allocation_count;
      forbid_allocation = true;
      const auto result = f.solve(system);
      forbid_allocation = false;
      require(result == Result::success && allocation_count == count && f.storage == committed &&
                  f.h == h && thread_changes == setters && threads == 17,
              "independent spectrum altered cache/input, allocated, or changed threads");
      verify_trace(f, system);
      verify_oracle(f.dimension(system), f.h.data(),
                    f.overlap.data() + f.plan.matrix_offsets()[system], f.coefficients.data(),
                    f.eigenvalues.data(), expected);
      for (int i = f.dimension(system) * f.dimension(system); i < 25; ++i)
        require(f.coefficients[i] == -987., "solve wrote beyond member matrix");
      for (int i = f.dimension(system); i < 5; ++i)
        require(f.eigenvalues[i] == -654., "solve wrote beyond member values");
    }
  // Thread scoping remains entirely caller-owned, including restoration.
  f.h = hamiltonians[0];
  reset_events();
  {
    cpu::ScopedSequentialBlas scope(f.backend);
    require(threads == 1 && thread_changes == setters + 1, "caller thread scope entry");
    forbid_allocation = true;
    const auto result = f.solve();
    forbid_allocation = false;
    require(result == Result::success && threads == 1 && thread_changes == setters + 1,
            "owner nested/changed caller thread scope");
  }
  require(threads == 17 && thread_changes == setters + 2 && cleanup_calls == 0,
          "caller thread scope restoration/cleanup ownership");
  for (std::uint64_t generation : {std::uint64_t(3), std::uint64_t(3),
                                   std::numeric_limits<std::uint64_t>::max(), std::uint64_t(1)}) {
    reset_events();
    forbid_allocation = true;
    const auto result = f.factor(generation);
    forbid_allocation = false;
    require(result == Result::success, "nonmonotone/repeated generation rejected");
    for (int i = 0; i < 3; ++i)
      require(f.cache.generations[i] == generation, "generation not published");
    reset_events();
    require(f.solve(0, generation) == Result::success, "nonmonotone generation could not solve");
  }
}

static void test_failures() {
  Fixture f;
  require(f.factor() == Result::success, "initial factor");
  for (int kind = 0; kind < 2; ++kind) {
    reset_faults();
    const auto cache = f.storage;
    // The middle member fails only after a peer's candidate was fully staged.
    if (kind == 0) {
      potrf_failure_order = 5;
      potrf_info = -3;
    } else {
      pocon_failure_order = 5;
      pocon_info = -6;
    }
    reset_events();
    require(f.factor(20) == Result::backend_failure && f.storage == cache && event_count >= 3,
            "backend error partially committed the batch");
    require(f.staged_generations[0] == 20 && f.cache.generations[0] == 19,
            "backend-error test did not exercise staged peer");
  }
  for (int kind = 0; kind < 5; ++kind) {
    reset_faults();
    const auto cache = f.storage;
    const double old_peer_factor = f.cache.factors[0];
    if (kind == 0) {
      potrf_failure_order = 5;
      potrf_info = 2;
    }
    if (kind == 1) {
      pocon_failure_order = 5;
      pocon_info = 1;
    }
    if (kind == 2) {
      condition_order = 5;
      condition = 1e-12;
    }
    if (kind == 3) {
      condition_order = 5;
      condition = std::numeric_limits<double>::quiet_NaN();
    }
    if (kind == 4) {
      condition_order = 5;
      condition = std::numeric_limits<double>::infinity();
    }
    f.overlap[0] = 4 + kind;
    reset_events();
    const std::uint64_t generation = 30 + kind;
    forbid_allocation = true;
    const auto result = f.factor(generation);
    forbid_allocation = false;
    require(result == Result::success && f.cache.statuses[0] == 7 && f.cache.statuses[1] == 41 &&
                f.cache.statuses[2] == 7,
            "numerical failure poisoned a peer or returned whole-batch failure");
    for (int i = 0; i < 3; ++i)
      require(f.cache.generations[i] == generation, "peer-local failure generation missing");
    require(std::memcmp(f.storage.data() + 4 * sizeof(double), cache.data() + 4 * sizeof(double),
                        25 * sizeof(double)) == 0 &&
                f.cache.factors[0] != old_peer_factor,
            "numerical failure changed old factor or lost valid peer");
    f.coefficients.fill(876);
    f.eigenvalues.fill(987);
    const auto coefficients = f.coefficients;
    const auto values = f.eigenvalues;
    const auto lapack = f.lapack_work;
    reset_events();
    require(f.solve(1, generation) == Result::numerical_failure && event_count == 0 &&
                f.coefficients == coefficients && f.eigenvalues == values &&
                f.lapack_work == lapack,
            "failed factor solve did work");
    reset_events();
    require(f.solve(0, generation) == Result::success, "usable peer could not solve");
  }
  reset_faults();
  f.overlap[4] = -1;
  require(f.factor(40) == Result::success && f.cache.statuses[1] == 41 && f.cache.statuses[0] == 7,
          "actual non-positive-definite member acceptance");
  f.overlap[4] = 1;
  require(f.factor(41) == Result::success && f.cache.statuses[1] == 7, "failed member recovery");
  const auto cache = f.storage;
  for (Int info : {Int(-7), Int(2), Int(0)}) {
    reset_faults();
    eigen_info = info;
    nonfinite_eigen = info == 0;
    reset_events();
    forbid_allocation = true;
    const auto result = f.solve(0, 41);
    forbid_allocation = false;
    require(result == (info < 0 ? Result::backend_failure : Result::numerical_failure) &&
                f.storage == cache,
            "eigen provider/numerical failure mapping or cache changed");
    verify_trace(f, 0, info == 0 ? 4 : 3);
  }
}

static void test_token_admission() {
  using Token = spectral::AdmittedSpectralMatrices;
  Fixture f;
  Token token;
  require(!token.matrix(f.plan, 0, 0) && !token.is_overlap_batch(f.plan),
          "default admission token");
  const auto allocations = allocation_count;
  const auto setters = thread_changes;
  forbid_allocation = true;
  const auto admitted =
      spectral::admit_spectral_matrices(f.plan, 0, 3, f.overlap.data(), nullptr, nullptr, token);
  forbid_allocation = false;
  require(admitted == Result::success && token.is_overlap_batch(f.plan) &&
              token.matrix(f.plan, 0, 0) == f.overlap.data() &&
              token.matrix(f.plan, 1, 0) == f.overlap.data() + 4 &&
              token.matrix(f.plan, 2, 0) == f.overlap.data() + 29 &&
              allocation_count == allocations && thread_changes == setters && event_count == 0,
          "allocation/provider-free full overlap admission");
  for (const auto indexes : {std::array<int, 2>{-1, 0}, {3, 0}, {0, -1}, {0, 1}})
    require(!token.matrix(f.plan, indexes[0], indexes[1]), "invalid token matrix index");
  Fixture other;
  require(!token.matrix(other.plan, 0, 0) && !token.is_overlap_batch(other.plan),
          "equal-shaped independent plan accepted a token");
  auto shared = f.plan;
  require(token.matrix(shared, 2, 0) == f.overlap.data() + 29 && token.is_overlap_batch(shared),
          "plan copy rejected admitted input");
  Token slice;
  require(spectral::admit_spectral_matrices(f.plan, 1, 2, f.overlap.data() + 4, nullptr, nullptr,
                                            slice) == Result::success &&
              !slice.is_overlap_batch(f.plan) && !slice.matrix(f.plan, 0, 0) &&
              slice.matrix(f.plan, 1, 0) == f.overlap.data() + 4 &&
              slice.matrix(f.plan, 2, 0) == f.overlap.data() + 29,
          "partial ragged slice did not normalize its plan offset");
  std::array<unsigned char, sizeof(Token)> retained;
  std::memcpy(retained.data(), &token, retained.size());
  auto rejected = [&](std::int64_t first, std::size_t count, const double* values,
                      const std::int64_t* offsets, const std::int32_t* counts) {
    reset_events();
    require(spectral::admit_spectral_matrices(f.plan, first, count, values, offsets, counts,
                                              token) == Result::invalid_argument &&
                std::memcmp(retained.data(), &token, retained.size()) == 0 &&
                token.matrix(f.plan, 2, 0) == f.overlap.data() + 29 && event_count == 0,
            "failed matrix admission changed existing token or called provider");
  };
  rejected(-1, 3, f.overlap.data(), nullptr, nullptr);
  rejected(3, 1, f.overlap.data(), nullptr, nullptr);
  rejected(0, 0, f.overlap.data(), nullptr, nullptr);
  rejected(1, 3, f.overlap.data(), nullptr, nullptr);
  rejected(0, std::numeric_limits<std::size_t>::max(), f.overlap.data(), nullptr, nullptr);
  rejected(0, 3, nullptr, nullptr, nullptr);
  rejected(0, 3, wrapping_pointer<double>(), nullptr, nullptr);
  rejected(0, 3,
           reinterpret_cast<const double*>(reinterpret_cast<const char*>(f.overlap.data()) + 1),
           nullptr, nullptr);
  alignas(64) std::array<std::int64_t, 4> offsets{19, 23, 48, 49};
  alignas(64) std::array<std::int32_t, 3> counts{1, 1, 1};
  rejected(0, 3, f.overlap.data(), offsets.data(), nullptr);
  rejected(0, 3, f.overlap.data(), nullptr, counts.data());
  rejected(0, 3, f.overlap.data(), wrapping_pointer<std::int64_t>(), counts.data());
  rejected(0, 3, f.overlap.data(), offsets.data(), wrapping_pointer<std::int32_t>());
  rejected(0, 3, f.overlap.data(),
           reinterpret_cast<const std::int64_t*>(reinterpret_cast<const char*>(offsets.data()) + 1),
           counts.data());
  rejected(0, 3, f.overlap.data(), offsets.data(),
           reinterpret_cast<const std::int32_t*>(reinterpret_cast<const char*>(counts.data()) + 1));
  for (auto bad : {std::array<std::int64_t, 4>{-1, 3, 28, 29},
                   {19, 23, 47, 49},
                   {19, 23, 23, 49},
                   {19, 23, 22, 49},
                   {19, 23, 48, -1}})
    rejected(0, 3, f.overlap.data(), bad.data(), counts.data());
  for (auto bad : {std::array<std::int32_t, 3>{0, 1, 1}, {1, -1, 1}, {1, 2, 1}})
    rejected(0, 3, f.overlap.data(), offsets.data(), bad.data());
  auto invalid_values = f.overlap;
  invalid_values.back() = std::numeric_limits<double>::quiet_NaN();
  rejected(0, 3, invalid_values.data(), nullptr, nullptr);
  invalid_values.back() = 1;
  invalid_values[5] = .25;
  rejected(0, 3, invalid_values.data(), nullptr, nullptr);
  spectral::PreparedSpectralPlan empty;
  require(spectral::admit_spectral_matrices(empty, 0, 3, f.overlap.data(), nullptr, nullptr,
                                            token) == Result::invalid_argument &&
              std::memcmp(retained.data(), &token, retained.size()) == 0,
          "unsealed admission plan");
  // No aliased output is accessed as a Token: admission must reject it before
  // scanning values or assigning its candidate, leaving the real objects intact.
  const auto values_before = f.overlap;
  const auto offsets_before = offsets;
  const auto counts_before = counts;
  const auto* identity = f.plan.identity();
  for (void* output :
       {static_cast<void*>(f.overlap.data()), static_cast<void*>(offsets.data()),
        static_cast<void*>(counts.data()), static_cast<void*>(&f.plan),
        const_cast<void*>(static_cast<const void*>(f.plan.identity())),
        const_cast<void*>(static_cast<const void*>(f.plan.orbital_offsets().data())),
        const_cast<void*>(static_cast<const void*>(f.plan.matrix_offsets().data()))}) {
    require(spectral::admit_spectral_matrices(f.plan, 0, 3, f.overlap.data(), offsets.data(),
                                              counts.data(), *reinterpret_cast<Token*>(output)) ==
                    Result::invalid_argument &&
                f.overlap == values_before && offsets == offsets_before &&
                counts == counts_before && f.plan.identity() == identity,
            "output token aliases source/layout/plan storage");
  }
  require(
      token.overlaps_borrowed_storage(&token, sizeof(token)) &&
          token.overlaps_borrowed_storage(f.overlap.data() + 29, sizeof(double)) &&
          token.overlaps_borrowed_storage(f.plan.matrix_offsets().data(), sizeof(std::int64_t)) &&
          !token.overlaps_borrowed_storage(f.h.data(), sizeof(f.h)) &&
          token.overlaps_borrowed_storage(wrapping_pointer<double>(), sizeof(double)),
      "borrowed admission storage ranges");
}

static void test_token_execution() {
  using Token = spectral::AdmittedSpectralMatrices;
  Fixture f;
  // Arbitrary spectrum counts and a nonzero first storage offset are independent
  // of method-specific spin cardinalities. All borrowed arrays remain immutable.
  alignas(64) const std::array<std::int64_t, 4> offsets{19, 31, 81, 85};
  alignas(64) const std::array<std::int32_t, 3> counts{3, 2, 4};
  std::array<double, 66> values{};
  std::array<std::array<std::array<double, 5>, 4>, 3> prescribed{};
  for (int system = 0; system < 3; ++system)
    for (int spectrum = 0; spectrum < counts[system]; ++spectrum) {
      const auto n = f.dimension(system);
      prescribed_problem(
          n, spectrum, values.data() + offsets[system] - offsets[0] + spectrum * n * n,
          f.overlap.data() + f.plan.matrix_offsets()[system], prescribed[system][spectrum]);
    }
  const auto original_values = values;
  const auto original_overlap = f.overlap;
  Token overlaps, spectra;
  const auto allocations = allocation_count;
  const auto setters = thread_changes;
  forbid_allocation = true;
  const auto overlap_admission =
      spectral::admit_spectral_matrices(f.plan, 0, 3, f.overlap.data(), nullptr, nullptr, overlaps);
  const auto spectra_admission = spectral::admit_spectral_matrices(
      f.plan, 0, 3, values.data(), offsets.data(), counts.data(), spectra);
  const auto factor_result = spectral::factor_admitted_spectral_overlaps(
      f.plan, overlaps, 19, f.backend, f.factor_work, f.cache, f.error);
  forbid_allocation = false;
  require(overlap_admission == Result::success && spectra_admission == Result::success &&
              factor_result == Result::success && overlaps.is_overlap_batch(f.plan) &&
              !spectra.is_overlap_batch(f.plan) && allocation_count == allocations &&
              thread_changes == setters && event_count == 6,
          "admitted factor resources/work count");
  const auto committed = f.storage;
  for (int system = 0; system < 3; ++system)
    for (int spectrum = 0; spectrum < counts[system]; ++spectrum) {
      const int n = f.dimension(system);
      const auto* h = values.data() + offsets[system] - offsets[0] + spectrum * n * n;
      require(spectra.matrix(f.plan, system, spectrum) == h,
              "arbitrary multiplicity pointer calculation");
      reset_events();
      forbid_allocation = true;
      const auto result = spectral::solve_admitted_spectrum(f.plan, system, spectrum, spectra,
                                                            f.cache, 19, f.backend, f.solve_work);
      forbid_allocation = false;
      require(result == Result::success && f.storage == committed && values == original_values &&
                  f.overlap == original_overlap && allocation_count == allocations &&
                  thread_changes == setters,
              "admitted spectrum changed immutable inputs/cache/resources");
      verify_trace(f, system);
      verify_oracle(n, h, f.overlap.data() + f.plan.matrix_offsets()[system], f.coefficients.data(),
                    f.eigenvalues.data(), prescribed[system][spectrum]);
    }
  Token partial;
  const std::array<std::int64_t, 3> partial_offsets{409, 459, 463};
  const std::array<std::int32_t, 2> partial_counts{2, 4};
  require(
      spectral::admit_spectral_matrices(f.plan, 1, 2, values.data() + 12, partial_offsets.data(),
                                        partial_counts.data(), partial) == Result::success &&
          !partial.matrix(f.plan, 0, 0) && partial.matrix(f.plan, 1, 1) == values.data() + 37 &&
          partial.matrix(f.plan, 2, 3) == values.data() + 65,
      "custom partial ragged slice");
  reset_events();
  require(spectral::solve_admitted_spectrum(f.plan, 2, 3, partial, f.cache, 19, f.backend,
                                            f.solve_work) == Result::success,
          "partial token execution");
  verify_oracle(1, values.data() + 65, f.overlap.data() + 29, f.coefficients.data(),
                f.eigenvalues.data(), prescribed[2][3]);
  auto rejected_solve = [&](const spectral::PreparedSpectralPlan& plan, const Token& token,
                            int system, int spectrum, const spectral::SpectralWorkspace& work,
                            std::uint64_t generation = 19,
                            Result expected = Result::invalid_argument) {
    const auto coefficients = f.coefficients;
    const auto eigenvalues = f.eigenvalues;
    const auto lapack = f.lapack_work;
    const auto integers = f.lapack_integers;
    reset_events();
    require(spectral::solve_admitted_spectrum(plan, system, spectrum, token, f.cache, generation,
                                              f.backend, work) == expected &&
                event_count == 0 && f.storage == committed && values == original_values &&
                f.coefficients == coefficients && f.eigenvalues == eigenvalues &&
                f.lapack_work == lapack && f.lapack_integers == integers,
            "rejected admitted solve changed borrowed input, outputs, cache or dispatched work");
  };
  Token empty;
  Fixture other;
  rejected_solve(f.plan, empty, 0, 0, f.solve_work);
  rejected_solve(other.plan, spectra, 0, 0, f.solve_work);
  rejected_solve(f.plan, spectra, -1, 0, f.solve_work);
  rejected_solve(f.plan, spectra, 3, 0, f.solve_work);
  rejected_solve(f.plan, spectra, 0, -1, f.solve_work);
  for (int system = 0; system < 3; ++system)
    rejected_solve(f.plan, spectra, system, counts[system], f.solve_work);
  rejected_solve(f.plan, partial, 0, 0, f.solve_work);
  rejected_solve(f.plan, spectra, 0, 0, f.solve_work, 18, Result::numerical_failure);
  rejected_solve(f.plan, spectra, 0, 0, f.solve_work, 0);
  // Chosen matrix is values[0:4]. Every target below is otherwise disjoint
  // from it, so rejecting these requires the token's complete borrowed ranges.
  for (void* target :
       {static_cast<void*>(&spectra), const_cast<void*>(static_cast<const void*>(offsets.data())),
        const_cast<void*>(static_cast<const void*>(counts.data())),
        static_cast<void*>(values.data() + 4), static_cast<void*>(values.data() + 12),
        static_cast<void*>(values.data() + 62)}) {
    auto bad = f.solve_work;
    bad.coefficients = static_cast<double*>(target);
    rejected_solve(f.plan, spectra, 0, 0, bad);
  }
  for (int which = 0; which < 3; ++which) {
    auto bad = f.solve_work;
    if (which == 0) bad.eigenvalues = values.data() + 4;
    if (which == 1) bad.lapack_work = values.data() + 4;
    if (which == 2) bad.lapack_integer_work = reinterpret_cast<Int*>(values.data() + 4);
    rejected_solve(f.plan, spectra, 0, 0, bad);
  }
  auto rejected_factor = [&](const spectral::PreparedSpectralPlan& plan, const Token& token,
                             const spectral::SpectralFactorWorkspace& work) {
    const auto staged = f.staged_factors;
    const auto generations = f.staged_generations;
    const auto statuses = f.staged_statuses;
    reset_events();
    require(spectral::factor_admitted_spectral_overlaps(plan, token, 19, f.backend, work, f.cache,
                                                        f.error) == Result::invalid_argument &&
                event_count == 0 && f.storage == committed && f.staged_factors == staged &&
                f.staged_generations == generations && f.staged_statuses == statuses,
            "rejected admitted factor changed cache/scratch or called provider");
  };
  rejected_factor(f.plan, empty, f.factor_work);
  rejected_factor(other.plan, overlaps, f.factor_work);
  rejected_factor(f.plan, partial, f.factor_work);
  rejected_factor(f.plan, spectra, f.factor_work);
  auto alias_token = f.factor_work;
  alias_token.factors = reinterpret_cast<double*>(&overlaps);
  rejected_factor(f.plan, overlaps, alias_token);
  auto alias_offsets = f.factor_work;
  alias_offsets.factors =
      reinterpret_cast<double*>(const_cast<std::int64_t*>(f.plan.matrix_offsets().data()));
  rejected_factor(f.plan, overlaps, alias_offsets);
  // A surviving immutable plan copy preserves identity and the internally
  // borrowed default partition after the original handle has been destroyed.
  auto surviving_plan = f.plan;
  forbid_allocation = true;
  auto surviving_overlap = overlaps;
  auto surviving_spectra = spectra;
  overlaps = {};
  spectra = {};
  f.plan = {};
  forbid_allocation = false;
  require(surviving_overlap.is_overlap_batch(surviving_plan) &&
              surviving_spectra.matrix(surviving_plan, 1, 1) == values.data() + 37,
          "plan-copy/token-copy lifetime lost borrowed partitions");
  reset_events();
  forbid_allocation = true;
  const auto refactor = spectral::factor_admitted_spectral_overlaps(
      surviving_plan, surviving_overlap, 3, f.backend, f.factor_work, f.cache, f.error);
  const auto resolve = spectral::solve_admitted_spectrum(surviving_plan, 1, 1, surviving_spectra,
                                                         f.cache, 3, f.backend, f.solve_work);
  forbid_allocation = false;
  require(refactor == Result::success && resolve == Result::success && values == original_values &&
              f.overlap == original_overlap,
          "retained immutable plan/token execution");
  verify_oracle(5, values.data() + 37, f.overlap.data() + 4, f.coefficients.data(),
                f.eigenvalues.data(), prescribed[1][1]);
}

int main(int argc, char** argv) {
  require(argc == 2, "expected one contract case");
  if (std::strcmp(argv[1], "plan") == 0)
    test_plan();
  else if (std::strcmp(argv[1], "binding") == 0)
    test_binding();
  else if (std::strcmp(argv[1], "factor_admission") == 0)
    test_factor_admission();
  else if (std::strcmp(argv[1], "solve_admission") == 0)
    test_solve_admission();
  else if (std::strcmp(argv[1], "numerics") == 0)
    test_numerics();
  else if (std::strcmp(argv[1], "failures") == 0)
    test_failures();
  else if (std::strcmp(argv[1], "token_admission") == 0)
    test_token_admission();
  else if (std::strcmp(argv[1], "token_execution") == 0)
    test_token_execution();
  else
    require(false, "unknown contract case");
  std::printf("%s passed\n", argv[1]);
}
