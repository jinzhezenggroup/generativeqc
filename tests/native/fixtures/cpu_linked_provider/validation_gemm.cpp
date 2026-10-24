// An independent extended-precision BLAS double for physical validation tests.
// It checks thread ownership and supports injected numerical failure. Its output
// is qualified by the real SCF admission gates, not a mirrored implementation.
#include <cstdio>
#include <cstdlib>
#include <string_view>

namespace {
int threads = 7;
bool used = false;
void require(bool value) {
  if (!value) std::abort();
}
struct Audit {
  ~Audit() {
    const auto* setting = std::getenv("GENERATIVEQC_SCF_VALIDATION_GEMM");
    const bool requested = setting && std::string_view(setting) == "1";
    require(threads == 7);
    require(used ==
            (requested && GENERATIVEQC_HAS_OPENBLAS && GENERATIVEQC_OPENBLAS_HAS_LOCAL_THREADS));
  }
} audit;
}  // namespace
extern "C" int openblas_set_num_threads_local(int next) {
  int previous = threads;
  threads = next;
  return previous;
}
extern "C" int openblas_get_num_threads() { return threads; }
extern "C" void openblas_set_num_threads(int) { std::abort(); }
extern "C" void cblas_dgemm(int layout, int ta, int tb, int m, int n, int k, double alpha,
                            const double* a, int lda, const double* b, int ldb, double beta,
                            double* c, int ldc) {
  require(layout == 101 && threads == 1 && (ta == 111 || ta == 112) && (tb == 111 || tb == 112));
  used = true;
  for (int i = 0; i < m; ++i)
    for (int j = 0; j < n; ++j) {
      long double sum = 0;
      for (int p = 0; p < k; ++p)
        sum += static_cast<long double>(ta == 112 ? a[p * lda + i] : a[i * lda + p]) *
               (tb == 112 ? b[j * ldb + p] : b[p * ldb + j]);
      c[i * ldc + j] = alpha * sum + (beta == 0 ? 0 : beta * c[i * ldc + j]);
    }
  if (std::getenv("GENERATIVEQC_TEST_VALIDATION_GEMM_CORRUPT")) c[0] += .1;
}
