#include <cstdio>
#include <cstdlib>

#include "methods/gfn2_electronic_update.cpp"
#include "scf/solver/cpu_target_eigen.hpp"
namespace cpu = generativeqc::tensor::cpu;
namespace eigen = generativeqc::solver::cpu;
namespace gfn = generativeqc::xtb::detail::gfn2;
static bool forbid_allocation, preflight = true;
static int thread_count = 17, set_calls, cleanup_calls, raw_info, stage, eigen_calls;
static const double* expected_factor;
static double* expected_matrix;
static double* expected_values;
static double* expected_work;
static std::int32_t* expected_iwork;
void* operator new(std::size_t n) {
  if (forbid_allocation) std::abort();
  if (void* p = std::malloc(n ? n : 1)) return p;
  throw std::bad_alloc();
}
void operator delete(void* p) noexcept { std::free(p); }
void operator delete(void* p, std::size_t) noexcept { std::free(p); }
static void require(bool value, const char* detail) {
  if (!value) {
    std::fprintf(stderr, "%s\n", detail);
    std::abort();
  }
}
static int set_threads(int n) {
  ++set_calls;
  int old = thread_count;
  thread_count = n;
  return old;
}
static void cleanup() { ++cleanup_calls; }
static std::int32_t potrf(std::int32_t layout, char uplo, std::int32_t n, double* a,
                          std::int32_t lda) {
  require(layout == 102 && uplo == 'L' && lda == n, "DPOTRF arguments");
  if (!preflight) require(a == expected_matrix && n == 2, "DPOTRF pointer/dimension");
  return preflight ? 0 : raw_info;
}
static std::int32_t pocon(std::int32_t layout, char uplo, std::int32_t n, const double* factor,
                          std::int32_t lda, double norm, double* rcond, double* work,
                          std::int32_t* iwork) {
  require(layout == 102 && uplo == 'L' && lda == n, "DPOCON arguments");
  if (!preflight)
    require(n == 2 && factor == expected_factor && norm == 4.5 && work == expected_work &&
                iwork == expected_iwork,
            "DPOCON borrowed binding");
  *rcond = 1.;
  return preflight ? 0 : raw_info;
}
static std::int32_t syevd(std::int32_t layout, char vectors, char triangle, std::int32_t n,
                          double* a, std::int32_t lda, double* w, double* work, std::int32_t lwork,
                          std::int32_t* iwork, std::int32_t liwork) {
  require(layout == 102 && vectors == 'V' && triangle == 'L' && lda == n, "DSYEVD ABI formatting");
  if (preflight) {
    require(n == 1 && lwork == 9 && liwork == 8, "preflight exact work");
    w[0] = a[0];
    a[0] = 1.;
    return 0;
  }
  require(n == 2 && a == expected_matrix && w == expected_values && work == expected_work &&
              iwork == expected_iwork && lwork == 81 && liwork == 28 && stage == 2 &&
              thread_count == 1,
          "DSYEVD n<N pointers/counts/order/scope");
  require(a[0] == 2 && a[1] == 1 && a[2] == 1 && a[3] == 2, "transformed matrix changed");
  ++eigen_calls;
  stage = 3;
  if (raw_info) return raw_info;
  const double q = 1 / std::sqrt(2.);
  a[0] = q;
  a[1] = -q;
  a[2] = q;
  a[3] = q;
  w[0] = 1;
  w[1] = 3;
  return 0;
}
static void trsm(int layout, int side, int triangle, int trans, int diagonal, std::int32_t rows,
                 std::int32_t columns, double alpha, const double* factor, std::int32_t lda,
                 double* rhs, std::int32_t ldb) {
  require(layout == 102 && triangle == 122 && diagonal == 131 && rows == columns && lda == rows &&
              ldb == rows && alpha == 1.,
          "TRSM fixed ABI arguments");
  if (preflight) {
    rhs[0] /= factor[0];
    return;
  }
  require(rows == 2 && factor == expected_factor && rhs == expected_matrix && thread_count == 1,
          "TRSM pointers/scope");
  if (stage == 0)
    require(side == 141 && trans == 111, "first transform");
  else if (stage == 1)
    require(side == 142 && trans == 112, "second transform");
  else
    require(stage == 3 && side == 141 && trans == 112, "back transform");
  ++stage;
}
static void gemm(int layout, int ta, int tb, std::int32_t m, std::int32_t n, std::int32_t k,
                 double alpha, const double* a, std::int32_t lda, const double* b, std::int32_t ldb,
                 double beta, double* c, std::int32_t ldc) {
  require(layout == 102 && ta == 111 && tb == 112 && m == n && n == k && lda == n && ldb == n &&
              ldc == n && alpha == 1. && beta == 0.,
          "weighted Gram ABI arguments");
  for (int col = 0; col < n; ++col)
    for (int row = 0; row < n; ++row) {
      double value = 0;
      for (int p = 0; p < n; ++p) value += a[row + p * n] * b[col + p * n];
      c[row + col * n] = value;
    }
}
static cpu::CpuLinearAlgebraBackend backend() {
  cpu::CpuLinearAlgebraBackend result;
  std::string error;
  preflight = true;
  require(
      cpu::prepare_internal_test_lp64_backend(potrf, pocon, syevd, trsm, gemm, set_threads, result,
                                              error, cleanup) == cpu::Lp64BackendStatus::success,
      "shared complete cohort preflight");
  require(result.ready() && !result.production(), "internal origin misclassified");
  for (int missing = 0; missing < 5; ++missing) {
    auto retained = result;
    require(gfn::make_internal_test_lp64_backend(
                missing == 0 ? nullptr : potrf, missing == 1 ? nullptr : pocon,
                missing == 2 ? nullptr : syevd, missing == 3 ? nullptr : trsm,
                missing == 4 ? nullptr : gemm, set_threads, retained,
                error) == GENERATIVEQC_XTB_STATUS_BACKEND_UNAVAILABLE,
            "partial cohort admitted");
    require(retained.ready(), "failed preparation mutated output");
  }
  preflight = false;
  return result;
}
static void prepare_ragged_plan(gfn::EigensolverPlanData& plan, void* memory, std::size_t bytes,
                                gfn::EigensolverOverlapCache& cache) {
  const std::int64_t offsets[]{0, 2, 7};
  std::string error;
  require(eigen::prepare_spectral_plan(
              offsets, 3, 1e-12,
              {GENERATIVEQC_XTB_STATUS_SUCCESS, GENERATIVEQC_XTB_STATUS_EIGENSOLVER_FAILED},
              plan.spectral, error) == eigen::SpectralResult::success,
          "prepared ragged n2/N5 spectral plan");
  eigen::SpectralOverlapCache bound;
  require(eigen::bind_spectral_overlap_cache(plan.spectral, memory, bytes, bound, error) ==
              eigen::SpectralResult::success,
          "canonical shared cache binding");
  cache = {bound.workspace_base, bound.workspace_size_bytes,
           bound.factors,        bound.generations,
           bound.statuses,       &plan};
  cache.geometry_generations[0] = 1;
  cache.system_statuses[0] = GENERATIVEQC_XTB_STATUS_SUCCESS;
  plan.spin_channels = {1, 1};
  plan.alpha_electron_counts = {0., 0.};
  plan.beta_electron_counts = {0., 0.};
  for (auto& field : plan.wavefunction_fields) field.system_offsets = {0, 4, 29};
  plan.wavefunction_fields[1].system_offsets = {0, 2, 7};
  plan.wavefunction_fields[2].system_offsets = {0, 4, 14};
}
static gfn::NumericalResult solve_small_system(const cpu::CpuLinearAlgebraBackend& provider,
                                               const gfn::EigensolverPlanData& plan,
                                               const gfn::EigensolverOverlapCache& cache,
                                               const double* h, double* a, double* values,
                                               double* work, std::int32_t* iw) {
  // Exercise the real method's shared-spectrum call and status mapping, retaining
  // independently borrowed matrix/value/LAPACK buffers for the ABI trace.
  double occupations[4]{}, densities[4]{}, weighted_densities[4]{};
  double staged_coefficients[4]{}, staged_values[2]{}, staged_occupations[4]{};
  double staged_densities[4]{}, staged_weighted_densities[4]{};
  double potentials[4]{}, entropies[2]{}, bands[2]{}, free_energies[2]{};
  std::int32_t statuses[2]{};
  gfn::EigensolverWorkspace scratch;
  scratch.coefficients = a;
  scratch.eigenvalues = values;
  scratch.occupations = occupations;
  scratch.densities = densities;
  scratch.energy_weighted_densities = weighted_densities;
  scratch.lapack_work = work;
  scratch.lapack_integer_work = iw;
  gfn::EigensolverWavefunctionView output;
  output.coefficients = staged_coefficients;
  output.eigenvalues = staged_values;
  output.occupations = staged_occupations;
  output.density = staged_densities;
  output.energy_weighted_density = staged_weighted_densities;
  const gfn::EigensolverThermodynamicsView thermodynamics{
      statuses, 2, potentials, 4, entropies, 2, bands, 2, free_energies, 2};
  return gfn::solve_system_unchecked(plan, 0, cache, 1, h, 0., provider, scratch, output,
                                     thermodynamics);
}
static void test_dispatch() {
  auto provider = backend();
  alignas(64) double a[4]{}, values[2]{}, factor[4]{1, 0, 0, 1}, work[81]{};
  alignas(64) std::int32_t iw[28]{};
  expected_matrix = a;
  expected_values = values;
  expected_factor = factor;
  expected_work = work;
  expected_iwork = iw;
  for (int info : {0, -9, 5}) {
    raw_info = info;
    require(cpu::cholesky_lower(provider, 2, a) == info, "DPOTRF raw status mapping");
    double rcond = 0;
    require(cpu::reciprocal_condition_lower(provider, 2, factor, 4.5, &rcond, work, iw) == info &&
                rcond == 1.,
            "DPOCON raw status mapping");
  }
  gfn::EigensolverPlanData plan;
  alignas(64) std::byte cache_memory[512]{};
  gfn::EigensolverOverlapCache cache;
  prepare_ragged_plan(plan, cache_memory, sizeof(cache_memory), cache);
  std::copy_n(factor, 4, cache.cholesky_factors);
  expected_factor = cache.cholesky_factors;
  const double h[]{2, 1, 1, 2};
  for (int info : {0, -7, 4}) {
    stage = eigen_calls = set_calls = 0;
    raw_info = info;
    forbid_allocation = true;
    gfn::NumericalResult result;
    {
      cpu::ScopedSequentialBlas scope(provider);
      result = solve_small_system(provider, plan, cache, h, a, values, work, iw);
    }
    forbid_allocation = false;
    require(eigen_calls == 1 && set_calls == 2 && thread_count == 17, "nested scope or retry");
    require(stage == (info == 0 ? 4 : 3), "back-transform after failure/order changed");
    require(result == (info < 0   ? gfn::NumericalResult::kBackendFailure
                       : info > 0 ? gfn::NumericalResult::kDataFailure
                                  : gfn::NumericalResult::kSuccess),
            "actual GFN leaf status mapping changed");
  }
  raw_info = 0;
  const double c[]{1, 0, 0, 1}, weights[]{.25, .75};
  double panel[4]{}, density[4]{};
  forbid_allocation = true;
  require(generativeqc::tensor::weighted_gram::execute_column_major(cpu::bind_gemm(provider), 2, c,
                                                                    weights, panel, density),
          "opaque weighted Gram binding failed");
  forbid_allocation = false;
  require(density[0] == .25 && density[1] == 0 && density[2] == 0 && density[3] == .75,
          "weighted Gram arithmetic changed");
  set_calls = 0;
  try {
    cpu::ScopedSequentialBlas scope(provider);
    throw 1;
  } catch (int) {
  }
  require(thread_count == 17 && set_calls == 2, "exception did not restore thread scope");
  {
    auto copy = provider;
    require(copy.ready(), "copy lost capability");
  }
  require(cleanup_calls == 0, "backend destruction ran thread cleanup");
  provider.release_thread_resources();
  require(cleanup_calls == 1, "explicit cleanup hook lost");
  std::puts("typed primitive pointers/order/status, GFN n2/N5, opaque Gram and lifetime passed");
}
static void residual(const double* h, const double* s, const double* c, const double* w,
                     bool column) {
  auto coefficient = [&](int row, int col) { return c[column ? row + 2 * col : 2 * row + col]; };
  for (int row = 0; row < 2; ++row)
    for (int col = 0; col < 2; ++col) {
      long double hc = 0, sc = 0, gram = 0;
      for (int k = 0; k < 2; ++k) {
        hc += (long double)h[2 * row + k] * coefficient(k, col);
        sc += (long double)s[2 * row + k] * coefficient(k, col);
        for (int j = 0; j < 2; ++j)
          gram += (long double)coefficient(k, row) * s[2 * k + j] * coefficient(j, col);
      }
      require(std::abs(hc - sc * w[col]) < 2e-11L && std::abs(gram - (row == col ? 1 : 0)) < 2e-12L,
              "independent generalized residual/metric oracle");
    }
}
static void test_real() {
  cpu::CpuLinearAlgebraBackend provider;
  std::string error;
  require(gfn::make_mkl_rt_lp64_backend(provider, error) == 0, error.c_str());
  require(provider.production(), "real oracle did not use admitted production provider");
  gfn::EigensolverPlanData plan;
  alignas(64) std::byte cache_memory[512]{};
  gfn::EigensolverOverlapCache cache;
  prepare_ragged_plan(plan, cache_memory, sizeof(cache_memory), cache);
  double h[]{8, 10, 10, 26}, s[]{4, 2, 2, 10}, factor[]{2, 1, 0, 3}, a[4]{}, w[2]{}, work[81]{};
  std::int32_t iw[28]{};
  std::copy_n(factor, 4, cache.cholesky_factors);
  {
    cpu::ScopedSequentialBlas scope(provider);
    require(solve_small_system(provider, plan, cache, h, a, w, work, iw) ==
                gfn::NumericalResult::kSuccess,
            "actual GFN production provider solve failed");
  }
  require(std::abs(w[0] - 1) < 2e-12 && std::abs(w[1] - 3) < 2e-12,
          "independent prescribed spectrum");
  residual(h, s, a, w, true);
  using Matrix = generativeqc::scf::reference::Matrix;
  Matrix hm(h, h + 4), sm(s, s + 4);
  const double scale = std::sqrt(26.) / 156.;
  Matrix x{16 * scale, -2 * scale, -2 * scale, 10 * scale};
  auto gaussian = generativeqc::scf::solver::cpu_target_eigen(hm, &sm, &x, 2);
  residual(h, s, gaussian.vectors.data(), gaussian.values.data(), false);
  require(std::abs(gaussian.values[0] - 1) < 2e-12 && std::abs(gaussian.values[1] - 3) < 2e-12,
          "Gaussian prescribed spectrum");
  std::puts("actual GFN runtime-LP64 and Gaussian scalar independent oracles passed");
}
int main(int argc, char** argv) {
  require(argc == 2, "expected dispatch or real");
  if (std::strcmp(argv[1], "dispatch") == 0)
    test_dispatch();
  else
    test_real();
}
