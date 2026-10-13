#include <cublas_v2.h>

#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <limits>
#include <string>
#include <vector>

#include "generativeqc/generativeqc.h"

// Prepared handles are borrowed unchanged. No stream/policy/lifetime mutator is
// implemented by this double: adding one to the adapter fails to link.
struct cublasContext {
  cudaStream_t stream;
  cublasStatus_t result{CUBLAS_STATUS_SUCCESS};
  bool submitted{};
};

cublasStatus_t cublasDgemm(cublasHandle_t handle, cublasOperation_t ta, cublasOperation_t tb, int m,
                           int n, int k, const double* alpha, const double* a, int lda,
                           const double* b, int ldb, const double* beta, double* c, int ldc) {
  assert(handle && alpha && beta);
  handle->submitted = true;
  if (handle->result != CUBLAS_STATUS_SUCCESS) return handle->result;
  for (int j = 0; j < n; ++j) {
    for (int i = 0; i < m; ++i) {
      double sum = 0;
      for (int p = 0; p < k; ++p) {
        const auto av = ta == CUBLAS_OP_N ? a[i + p * lda] : a[p + i * lda];
        const auto bv = tb == CUBLAS_OP_N ? b[p + j * ldb] : b[j + p * ldb];
        sum += av * bv;
      }
      c[i + j * ldc] = *alpha * sum + (*beta == 0 ? 0 : *beta * c[i + j * ldc]);
    }
  }
  return CUBLAS_STATUS_SUCCESS;
}

// FP64 strided provider double: preserve input broadcast and transpose semantics.
cublasStatus_t cublasDgemmStridedBatched(
    cublasHandle_t handle, cublasOperation_t ta, cublasOperation_t tb, int m, int n, int k,
    const double* alpha, const double* a, int lda, long long stride_a, const double* b, int ldb,
    long long stride_b, const double* beta, double* c, int ldc, long long stride_c, int batches) {
  assert(handle && alpha && beta && batches > 0);
  for (int batch = 0; batch < batches; ++batch) {
    const auto status = cublasDgemm(handle, ta, tb, m, n, k, alpha, a + batch * stride_a, lda,
                                   b + batch * stride_b, ldb, beta, c + batch * stride_c, ldc);
    if (status != CUBLAS_STATUS_SUCCESS) return status;
  }
  return CUBLAS_STATUS_SUCCESS;
}

struct CudaDensityFittingJkPlan {
  cublasHandle_t blas;
  cudaStream_t stream;
};
#include "df_matrix_product.inc"

namespace {
constexpr double sentinel = -987654;

void products(CudaDensityFittingJkPlan& plan) {
  // Independent row-major mathematical operands. Physical storage is built
  // separately for all four transpose combinations and padded leading strides.
  constexpr int m = 2, n = 3, k = 4;
  const double left[m][k] = {{1, -2, 3, 0.5}, {4, 2, -1, 3}};
  const double right[k][n] = {{2, 1, -3}, {-1, 4, 2}, {3, 0.5, 1}, {2, -2, 4}};
  for (bool ta : {false, true}) {
    for (bool tb : {false, true}) {
      const int lda = (ta ? k : m) + 2, ldb = (tb ? n : k) + 3, ldc = m + 4;
      std::vector<double> a(lda * (ta ? m : k) + 2, sentinel);
      std::vector<double> b(ldb * (tb ? k : n) + 2, sentinel);
      for (int i = 0; i < m; ++i)
        for (int p = 0; p < k; ++p) a[1 + (ta ? p + i * lda : i + p * lda)] = left[i][p];
      for (int p = 0; p < k; ++p)
        for (int j = 0; j < n; ++j) b[1 + (tb ? j + p * ldb : p + j * ldb)] = right[p][j];
      const auto original_a = a, original_b = b;
      for (const auto scalars : {std::array<double, 2>{1, 0}, {-1.25, 0.5}}) {
        std::vector<double> c(ldc * n + 2, sentinel), expected = c;
        for (int i = 0; i < m; ++i) {
          for (int j = 0; j < n; ++j) {
            const double initial = 10 + i + j;
            c[1 + i + j * ldc] =
                scalars[1] == 0 ? std::numeric_limits<double>::quiet_NaN() : initial;
            double value = 0;
            for (int p = 0; p < k; ++p) value += left[i][p] * right[p][j];
            expected[1 + i + j * ldc] = scalars[0] * value + scalars[1] * initial;
          }
        }
        std::string detail = "unchanged";
        assert(scf_matrix_product_view(plan, ta, tb, m, n, k, a.data() + 1, lda, b.data() + 1, ldb,
                                       c.data() + 1, ldc, scalars[0], scalars[1], "product",
                                       detail) == GENERATIVEQC_STATUS_SUCCESS);
        assert(c == expected && a == original_a && b == original_b && detail == "unchanged");
      }
    }
  }
  // A rank-two Gram reconstruction borrows the same allocation twice.
  double factor[] = {1, 2, 3, -1, 4, 2}, gram[9]{};
  std::string detail;
  assert(scf_matrix_product_view(plan, false, true, 3, 3, 2, factor, 3, factor, 3, gram, 3, 1, 0,
                                 "gram", detail) == GENERATIVEQC_STATUS_SUCCESS);
  for (int i = 0; i < 3; ++i)
    for (int j = 0; j < 3; ++j)
      assert(gram[i + j * 3] == factor[i] * factor[j] + factor[i + 3] * factor[j + 3]);
  // Split reduction panels preserve physical row stride and beta accumulation.
  double a[] = {1, 4, -2, 2, 3, -1, 0.5, 3};
  double b0[] = {2, -1, 1, 4, -3, 2}, b1[] = {3, 2, 0.5, -2, 1, 4}, c[6]{};
  assert(scf_matrix_product_view(plan, false, false, m, n, 2, a, m, b0, 2, c, m, 1, 0, "panel",
                                 detail) == GENERATIVEQC_STATUS_SUCCESS);
  assert(scf_matrix_product_view(plan, false, false, m, n, 2, a + 4, m, b1, 2, c, m, 1, 1, "panel",
                                 detail) == GENERATIVEQC_STATUS_SUCCESS);
  for (int i = 0; i < m; ++i) {
    for (int j = 0; j < n; ++j) {
      double expected = 0;
      for (int p = 0; p < k; ++p) expected += left[i][p] * right[p][j];
      assert(c[i + j * m] == expected);
    }
  }
}

void invalid(CudaDensityFittingJkPlan& plan) {
  double data[64]{};
  const auto saved = plan.blas;
  const auto over = static_cast<std::size_t>(std::numeric_limits<int>::max()) + 1;
  for (bool ta : {false, true}) {
    for (bool tb : {false, true}) {
      // M,N,K,LDA,LDB,LDC. Reject invalid views before provider submission.
      const std::array<std::size_t, 6> valid{2, 3, 4, ta ? 4U : 2U, tb ? 3U : 4U, 2};
      for (int slot = 0; slot < 6; ++slot) {
        for (const auto value : {std::size_t{0}, over, std::numeric_limits<std::size_t>::max()}) {
          auto shape = valid;
          shape[slot] = value;
          saved->submitted = false;
          std::string detail;
          assert(scf_matrix_product_view(plan, ta, tb, shape[0], shape[1], shape[2], data, shape[3],
                                         data, shape[4], data, shape[5], 1, 0, "bad",
                                         detail) == GENERATIVEQC_STATUS_INVALID_ARGUMENT);
          assert(!saved->submitted && !detail.empty());
        }
      }
      for (int slot = 3; slot < 6; ++slot) {
        auto shape = valid;
        --shape[slot];
        saved->submitted = false;
        std::string detail;
        assert(scf_matrix_product_view(plan, ta, tb, shape[0], shape[1], shape[2], data, shape[3],
                                       data, shape[4], data, shape[5], 1, 0, "bad",
                                       detail) == GENERATIVEQC_STATUS_INVALID_ARGUMENT);
        assert(!saved->submitted);
      }
    }
  }
  for (int missing = 0; missing < 4; ++missing) {
    saved->submitted = false;
    plan.blas = missing == 0 ? nullptr : saved;
    std::string detail;
    assert(scf_matrix_product_view(plan, false, false, 2, 3, 4, missing == 1 ? nullptr : data, 2,
                                   missing == 2 ? nullptr : data, 4, missing == 3 ? nullptr : data,
                                   2, 1, 0, "bad", detail) == GENERATIVEQC_STATUS_INVALID_ARGUMENT);
    assert(!saved->submitted);
  }
  plan.blas = saved;
}

void batched(CudaDensityFittingJkPlan& plan) {
  // Exercise input broadcasting, independent batches, all transposes,
  // padded leading dimensions, alpha/beta and invalid view admission.
  constexpr int m = 2, n = 3, k = 4, batches = 3;
  for (bool ta : {false, true}) {
    for (bool tb : {false, true}) {
      const int lda = (ta ? k : m) + 2, ldb = (tb ? n : k) + 2, ldc = m + 3;
      const auto stride_a = ta ? (lda * m + 7) : 0;
      const auto stride_b = tb ? 0 : (ldb * n + 6);
      const auto stride_c = ldc * n + 5;
      std::vector<double> a(1 + (batches - 1) * stride_a + lda * (ta ? m : k), sentinel);
      std::vector<double> b(1 + (batches - 1) * stride_b + ldb * (tb ? k : n), sentinel);
      std::vector<double> c(1 + batches * stride_c, sentinel), expected = c;
      for (int batch = 0; batch < batches; ++batch) {
        if (stride_a || batch == 0) {
          for (int i = 0; i < m; ++i)
            for (int p = 0; p < k; ++p)
              a[1 + batch * stride_a + (ta ? p + i * lda : i + p * lda)] =
                  2.0 * i + p + 0.25 * (ta ? batch : 0);
        }
        if (stride_b || batch == 0) {
          for (int p = 0; p < k; ++p)
            for (int j = 0; j < n; ++j)
              b[1 + batch * stride_b + (tb ? j + p * ldb : p + j * ldb)] =
                  p - 2.0 * j + 0.5 * (tb ? 0 : batch);
        }
        for (int j = 0; j < n; ++j)
          for (int i = 0; i < m; ++i) {
            double sum = 0;
            for (int p = 0; p < k; ++p) {
              const double av = 2.0 * i + p + 0.25 * (ta ? batch : 0);
              const double bv = p - 2.0 * j + 0.5 * (tb ? 0 : batch);
              sum += av * bv;
            }
            const double seed = 3 + batch + i + j;
            c[1 + batch * stride_c + i + j * ldc] = seed;
            expected[1 + batch * stride_c + i + j * ldc] = -1.25 * sum + 0.5 * seed;
          }
      }
      std::string detail = "unchanged";
      const ScfMatrixBatchView view{
          {a.data() + 1, static_cast<std::size_t>(lda), static_cast<std::size_t>(stride_a)},
          {b.data() + 1, static_cast<std::size_t>(ldb), static_cast<std::size_t>(stride_b)},
          {c.data() + 1, static_cast<std::size_t>(ldc), static_cast<std::size_t>(stride_c)},
          m, n, k, batches, ta, tb, -1.25, 0.5};
      assert(scf_matrix_product_batched_view(plan, view, "batch", detail) ==
             GENERATIVEQC_STATUS_SUCCESS);
      for (std::size_t i = 0; i < c.size(); ++i)
        assert(std::abs(c[i] - expected[i]) < 1e-12);
      assert(detail == "unchanged");
    }
  }
  double data[64]{};
  const ScfMatrixBatchView valid{{data, 4, 0}, {data, 4, 0}, {data, 4, 16},
                                 2, 3, 4, 2, false, false};
  for (int slot = 0; slot < 5; ++slot) {
    auto view = valid;
    switch (slot) {
      case 0: view.batches = 0; break;
      case 1: view.rows = 0; break;
      case 2: view.columns = std::numeric_limits<std::size_t>::max(); break;
      case 3: view.left.leading_dimension = 1; break;
      case 4: view.right.batch_stride = std::numeric_limits<std::size_t>::max(); break;
    }
    plan.blas->submitted = false;
    std::string detail;
    assert(scf_matrix_product_batched_view(plan, view, "invalid", detail) ==
           GENERATIVEQC_STATUS_INVALID_ARGUMENT);
    assert(!plan.blas->submitted && !detail.empty());
  }
  for (const auto failure : {CUBLAS_STATUS_ALLOC_FAILED, CUBLAS_STATUS_EXECUTION_FAILED}) {
    plan.blas->result = failure;
    std::string detail;
    assert(scf_matrix_product_batched_view(plan, valid, "batched failure", detail) ==
           (failure == CUBLAS_STATUS_ALLOC_FAILED ? GENERATIVEQC_STATUS_OUT_OF_MEMORY
                                                  : GENERATIVEQC_STATUS_CUDA_ERROR));
    assert(detail == "batched failure failed with cuBLAS status " +
                         std::to_string(static_cast<int>(failure)));
  }
  plan.blas->result = CUBLAS_STATUS_SUCCESS;
}

void failures(CudaDensityFittingJkPlan& plan) {
  double input[4]{}, output[4] = {1, 2, 3, 4};
  for (const auto failure : {CUBLAS_STATUS_ALLOC_FAILED, CUBLAS_STATUS_NOT_INITIALIZED,
                             CUBLAS_STATUS_INVALID_VALUE, CUBLAS_STATUS_EXECUTION_FAILED}) {
    plan.blas->result = failure;
    std::string detail;
    assert(scf_matrix_product_view(plan, false, false, 2, 2, 2, input, 2, input, 2, output, 2, 1, 0,
                                   "retained failure context", detail) ==
           (failure == CUBLAS_STATUS_ALLOC_FAILED ? GENERATIVEQC_STATUS_OUT_OF_MEMORY
                                                  : GENERATIVEQC_STATUS_CUDA_ERROR));
    assert(detail == "retained failure context failed with cuBLAS status " +
                         std::to_string(static_cast<int>(failure)));
    for (int i = 0; i < 4; ++i) assert(output[i] == i + 1);
  }
}
}  // namespace

int main(int argc, char** argv) {
  assert(argc == 2);
  int stream_sentinel;
  const auto stream = reinterpret_cast<cudaStream_t>(&stream_sentinel);
  cublasContext handle{stream};
  CudaDensityFittingJkPlan plan{&handle, stream};
  const std::string mode = argv[1];
  if (mode == "products")
    products(plan);
  else if (mode == "invalid")
    invalid(plan);
  else if (mode == "failures")
    failures(plan);
  else if (mode == "batched")
    batched(plan);
  else
    return 1;
  assert(plan.blas == &handle && plan.stream == stream && handle.stream == stream);
}
