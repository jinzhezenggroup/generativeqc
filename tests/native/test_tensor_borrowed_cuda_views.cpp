// Host recording provider: verify the production borrowed views in supported C++ modes.
#include <cassert>
#include <climits>
#include <cstdint>
#include <type_traits>
#include <utility>
#include <vector>
using cublasStatus_t = int;
struct Handle {
  int pointer_mode = 1;
  int stream = 71;
};
using cublasHandle_t = Handle*;
enum cublasOperation_t { CUBLAS_OP_N = 0, CUBLAS_OP_T = 1 };
std::vector<std::uint64_t> record;
int result = 0;
template <class T>
std::uint64_t bits(T v) {
  if constexpr (std::is_pointer_v<T>)
    return reinterpret_cast<std::uintptr_t>(v);
  else
    return static_cast<std::uint64_t>(v);
}
template <class... T>
int capture(T... args) {
  record = {bits(args)...};
  return result;
}
int cublasDgemm(cublasHandle_t h, cublasOperation_t ta, cublasOperation_t tb, int m, int n, int k,
                const double* alpha, const double* a, int lda, const double* b, int ldb,
                const double* beta, double* c, int ldc) {
  return capture(h, ta, tb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
}
int cublasDgemmStridedBatched(cublasHandle_t h, cublasOperation_t ta, cublasOperation_t tb, int m,
                              int n, int k, const double* alpha, const double* a, int lda,
                              long long sa, const double* b, int ldb, long long sb,
                              const double* beta, double* c, int ldc, long long sc, int count) {
  return capture(h, ta, tb, m, n, k, alpha, a, lda, sa, b, ldb, sb, beta, c, ldc, sc, count);
}
int cublasDgemv(cublasHandle_t h, cublasOperation_t t, int m, int n, const double* alpha,
                const double* a, int lda, const double* x, int ix, const double* beta, double* y,
                int iy) {
  return capture(h, t, m, n, alpha, a, lda, x, ix, beta, y, iy);
}

#ifdef GENERATIVEQC_TEST_HAS_GEAM
int cublasDgeam(cublasHandle_t h, cublasOperation_t ta, cublasOperation_t tb, int m, int n,
                const double* alpha, const double* a, int lda, const double* beta, const double* b,
                int ldb, double* c, int ldc) {
  return capture(h, ta, tb, m, n, alpha, a, lda, beta, b, ldb, c, ldc);
}

#endif

namespace generativeqc_tensor {
#include "borrowed_tensor_views.inc"
}
using namespace generativeqc_tensor;

#ifdef GENERATIVEQC_TEST_HAS_GEAM
static_assert(column_geam_available<cublasHandle_t>);
#else
static_assert(!column_geam_available<cublasHandle_t>);
#endif

template <class H>
void geam(H h, bool ta, bool tb, int v) {
  const auto* alpha = reinterpret_cast<const double*>(0x1);
  const auto* beta = reinterpret_cast<const double*>(0x3);
  auto* data = reinterpret_cast<double*>(0x9);
  if constexpr (column_geam_available<H>) {
    auto expected = cublasDgeam(h, ta ? CUBLAS_OP_T : CUBLAS_OP_N, tb ? CUBLAS_OP_T : CUBLAS_OP_N,
                                v, INT_MAX, alpha, data, INT_MIN, beta, data, -1, data, 0);
    auto args = record;
    record.clear();
    assert(submit_column_geam(ColumnGeamView<H>{h, ta, tb, v, INT_MAX, alpha, data, INT_MIN, beta,
                                                data, -1, data, 0}) == expected);
    assert(record == args);
  } else {
    record.clear();
  }  // The owner can retain its generated transpose fallback.
}
int main() {
  Handle handle;
  auto* data = reinterpret_cast<double*>(0x9);
  const auto* alpha = reinterpret_cast<const double*>(0x1);
  const auto* beta = reinterpret_cast<const double*>(0x3);
  for (auto* h : {&handle, static_cast<Handle*>(nullptr)})
    for (int mode : {0, 1}) {
      handle.pointer_mode = mode;
      for (int status : {0, 1, 3, 7, 13}) {
        result = status;
        for (int ta : {0, 1})
          for (int tb : {0, 1})
            for (int v : {INT_MIN, -3, 0, 2, INT_MAX})
              for (long long stride : {LLONG_MIN, -3LL, 0LL, 7LL, LLONG_MAX}) {
                auto oa = ta ? CUBLAS_OP_T : CUBLAS_OP_N, ob = tb ? CUBLAS_OP_T : CUBLAS_OP_N;
                auto expected = cublasDgemm(h, oa, ob, v, INT_MAX, INT_MIN, alpha, data, -1, data,
                                            0, beta, data, INT_MAX);
                auto args = record;
                record.clear();
                assert(submit_column_gemm({h, ta != 0, tb != 0, v, INT_MAX, INT_MIN, alpha, data,
                                           -1, data, 0, beta, data, INT_MAX}) == expected);
                assert(record == args);
                expected = cublasDgemmStridedBatched(h, oa, ob, v, INT_MAX, INT_MIN, alpha, data,
                                                     -1, stride, data, 0, 0, beta, data, INT_MAX,
                                                     LLONG_MAX, v);
                args = record;
                record.clear();
                assert(submit_column_batched_gemm({h, ta != 0, tb != 0, v, INT_MAX, INT_MIN, alpha,
                                                   data, -1, stride, data, 0, 0, beta, data,
                                                   INT_MAX, LLONG_MAX, v}) == expected);
                assert(record == args);
                expected =
                    cublasDgemv(h, oa, v, INT_MAX, alpha, data, INT_MIN, data, v, beta, data, -3);
                args = record;
                record.clear();
                assert(submit_column_gemv({h, ta != 0, v, INT_MAX, alpha, data, INT_MIN, data, v,
                                           beta, data, -3}) == expected);
                assert(record == args);
                geam(h, ta != 0, tb != 0, v);
                assert(handle.pointer_mode == mode && handle.stream == 71);
              }
      }
    }
}
