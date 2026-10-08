#include <cstdint>
#ifndef PREFIXED
#define PREFIXED 0
#endif
#ifndef OMIT
#define OMIT 0
#endif
#ifndef FAIL
#define FAIL 0
#endif
#ifndef PREFIX_THREADS
#define PREFIX_THREADS 0
#endif
#if PREFIXED
#define NAME_IMPL(x) scipy_##x
#else
#define NAME_IMPL(x) x
#endif
#define NAME(x) NAME_IMPL(x)
extern "C" {
#if OMIT != 1
std::int32_t NAME(LAPACKE_dpotrf_work)(std::int32_t layout, char uplo, std::int32_t n, double* a,
                                       std::int32_t lda) {
  return FAIL == 1 || layout != 102 || uplo != 'L' || n < 1 || lda != n || !a ? -1 : 0;
}
#endif
#if OMIT != 2
std::int32_t NAME(LAPACKE_dpocon_work)(std::int32_t, char, std::int32_t, const double*,
                                       std::int32_t, double, double* rcond, double*,
                                       std::int32_t*) {
  *rcond = FAIL == 3 ? 0. : 1.;
  return FAIL == 2 ? -2 : 0;
}
#endif
#if OMIT != 3
#ifdef MIXED
std::int32_t scipy_LAPACKE_dsyevd_work(std::int32_t, char, char, std::int32_t, double*,
                                       std::int32_t, double*, double*, std::int32_t, std::int32_t*,
                                       std::int32_t) {
  return 0;
}
#else
std::int32_t NAME(LAPACKE_dsyevd_work)(std::int32_t layout, char vectors, char uplo, std::int32_t n,
                                       double* a, std::int32_t lda, double* w, double*,
                                       std::int32_t lwork, std::int32_t*, std::int32_t liwork) {
  if (FAIL == 4 || layout != 102 || vectors != 'V' || uplo != 'L' || n < 1 || lda != n ||
      lwork < 1 || liwork < 1)
    return -3;
  for (int i = 0; i < n; ++i) w[i] = a[i + i * n] + (FAIL == 5 ? 1. : 0.);
  for (int i = 0; i < n * n; ++i) a[i] = 0;
  for (int i = 0; i < n; ++i) a[i + i * n] = FAIL == 6 ? 2. : 1.;
  return 0;
}
#endif
#endif
#if OMIT != 4
void NAME(cblas_dtrsm)(int, int, int, int, int, std::int32_t, std::int32_t, double, const double* a,
                       std::int32_t, double* b, std::int32_t) {
  if (FAIL != 7) b[0] /= a[0];
}
#endif
#if OMIT != 5
void NAME(cblas_dgemm)(int, int, int, std::int32_t, std::int32_t, std::int32_t, double,
                       const double* a, std::int32_t, const double* b, std::int32_t, double,
                       double* c, std::int32_t) {
  c[0] = FAIL == 8 ? 0. : a[0] * b[0];
}
#endif
#if OMIT != 6
const char* NAME(openblas_get_config)() {
#if FAIL == 9
  return nullptr;
#elif FAIL == 10
  return "OpenBLAS 0.test USE64BITINT";
#else
  return "OpenBLAS 0.test LP64";
#endif
}
#endif
static thread_local int current_threads = 17;
#if OMIT != 7
#if PREFIX_THREADS
int scipy_openblas_set_num_threads_local(int n) {
#else
int openblas_set_num_threads_local(int n) {
#endif
  int old = current_threads;
  current_threads = n;
  return old;
}
#endif
int gate_threads() { return current_threads; }
}
