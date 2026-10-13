#include <cublas_v2.h>

#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstdint>
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

namespace {
cublasStatus_t create_error = CUBLAS_STATUS_SUCCESS;
cublasStatus_t stream_error = CUBLAS_STATUS_SUCCESS;
cublasStatus_t mode_error = CUBLAS_STATUS_SUCCESS;
int handles_created = 0, handles_destroyed = 0;
}  // namespace
cublasStatus_t cublasCreate(cublasHandle_t* handle) {
  if (create_error != CUBLAS_STATUS_SUCCESS) return create_error;
  *handle = new cublasContext{};
  ++handles_created;
  return CUBLAS_STATUS_SUCCESS;
}
cublasStatus_t cublasDestroy(cublasHandle_t handle) {
  if (handle) {
    ++handles_destroyed;
    delete handle;
  }
  return CUBLAS_STATUS_SUCCESS;
}
cublasStatus_t cublasSetStream(cublasHandle_t handle, cudaStream_t stream) {
  if (stream_error != CUBLAS_STATUS_SUCCESS) return stream_error;
  assert(handle);
  handle->stream = stream;
  return CUBLAS_STATUS_SUCCESS;
}
cublasStatus_t cublasSetPointerMode(cublasHandle_t handle, cublasPointerMode_t mode) {
  assert(handle && mode == CUBLAS_POINTER_MODE_HOST);
  return mode_error;
}
cublasStatus_t cublasDsyrk(cublasHandle_t handle, cublasFillMode_t triangle,
                           cublasOperation_t transpose, int n, int k, const double* alpha,
                           const double* u, int ldu, const double* beta, double* output,
                           int ldout) {
  assert(handle && triangle == CUBLAS_FILL_MODE_LOWER && transpose == CUBLAS_OP_T);
  handle->submitted = true;
  if (handle->result != CUBLAS_STATUS_SUCCESS) return handle->result;
  for (int column = 0; column < n; ++column) {
    for (int row = column; row < n; ++row) {
      double product = 0;
      for (int p = 0; p < k; ++p) product += u[p + row * ldu] * u[p + column * ldu];
      output[row + column * ldout] =
          *alpha * product + (*beta == 0 ? 0 : *beta * output[row + column * ldout]);
    }
  }
  return CUBLAS_STATUS_SUCCESS;
}

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
cublasStatus_t cublasDgemmStridedBatched(cublasHandle_t handle, cublasOperation_t ta,
                                         cublasOperation_t tb, int m, int n, int k,
                                         const double* alpha, const double* a, int lda,
                                         long long stride_a, const double* b, int ldb,
                                         long long stride_b, const double* beta, double* c, int ldc,
                                         long long stride_c, int batches) {
  assert(handle && alpha && beta && batches > 0);
  for (int batch = 0; batch < batches; ++batch) {
    const auto status = cublasDgemm(handle, ta, tb, m, n, k, alpha, a + batch * stride_a, lda,
                                    b + batch * stride_b, ldb, beta, c + batch * stride_c, ldc);
    if (status != CUBLAS_STATUS_SUCCESS) return status;
  }
  return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasDgemv(cublasHandle_t handle, cublasOperation_t trans, int m, int n,
                           const double* alpha, const double* a, int lda, const double* x,
                           int incx, const double* beta, double* y, int incy) {
  handle->submitted = true;
  if (handle->result != CUBLAS_STATUS_SUCCESS) return handle->result;
  for (int out = 0; out < (trans == CUBLAS_OP_T ? n : m); ++out) {
    double value = 0;
    for (int k = 0; k < (trans == CUBLAS_OP_T ? m : n); ++k)
      value += (trans == CUBLAS_OP_T ? a[k + out * lda] : a[out + k * lda]) * x[k * incx];
    y[out * incy] = *alpha * value + (*beta == 0 ? 0 : *beta * y[out * incy]);
  }
  return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasDgeam(cublasHandle_t handle, cublasOperation_t ta, cublasOperation_t tb,
                           int m, int n, const double* alpha, const double* a, int lda,
                           const double* beta, const double* b, int ldb, double* c, int ldc) {
  handle->submitted = true;
  if (handle->result != CUBLAS_STATUS_SUCCESS) return handle->result;
  for (int j = 0; j < n; ++j)
    for (int i = 0; i < m; ++i) {
      const auto left = ta == CUBLAS_OP_T ? a[j + i * lda] : a[i + j * lda];
      const auto right = tb == CUBLAS_OP_T ? b[j + i * ldb] : b[i + j * ldb];
      c[i + j * ldc] = *alpha * left + (*beta == 0 ? 0 : *beta * right);
    }
  return CUBLAS_STATUS_SUCCESS;
}

struct CudaDensityFittingJkPlan {
  cublasHandle_t blas;
  cudaStream_t stream;
};
#include "df_matrix_product.inc"
#include "borrowed_response_ops.inc"

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
      const auto original_a = a, original_b = b;
      std::string detail = "unchanged";
      const ScfMatrixBatchView view{
          {a.data() + 1, static_cast<std::size_t>(lda), static_cast<std::size_t>(stride_a)},
          {b.data() + 1, static_cast<std::size_t>(ldb), static_cast<std::size_t>(stride_b)},
          {c.data() + 1, static_cast<std::size_t>(ldc), static_cast<std::size_t>(stride_c)},
          m,
          n,
          k,
          batches,
          ta,
          tb,
          -1.25,
          0.5};
      assert(scf_matrix_product_batched_view(plan, view, "batch", detail) ==
             GENERATIVEQC_STATUS_SUCCESS);
      for (std::size_t i = 0; i < c.size(); ++i) assert(std::abs(c[i] - expected[i]) < 1e-12);
      assert(a == original_a && b == original_b && detail == "unchanged");
    }
  }
  double data[64]{};
  const ScfMatrixBatchView valid{{data, 4, 0}, {data, 4, 0}, {data, 4, 16}, 2, 3, 4, 2,
                                 false,        false};
  for (int slot = 0; slot < 5; ++slot) {
    auto view = valid;
    switch (slot) {
      case 0:
        view.batches = 0;
        break;
      case 1:
        view.rows = 0;
        break;
      case 2:
        view.columns = std::numeric_limits<std::size_t>::max();
        break;
      case 3:
        view.left.leading_dimension = 1;
        break;
      case 4:
        view.right.batch_stride = std::numeric_limits<std::size_t>::max();
        break;
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

void batched_admission(CudaDensityFittingJkPlan& plan, const std::string& mode) {
  double input[64]{}, output[64]{};
  const ScfMatrixBatchView valid{{input, 4, 0}, {input, 4, 0}, {output, 4, 16}, 2, 3, 4, 2,
                                 false,         false};
  const auto saved = plan.blas;
  // Fail injected vendor calls before any invalid address can be dereferenced.
  saved->result = CUBLAS_STATUS_EXECUTION_FAILED;
  const auto rejected = [&](const ScfMatrixBatchView& view) {
    saved->submitted = false;
    std::string detail;
    assert(scf_matrix_product_batched_view(plan, view, "admission", detail) ==
           GENERATIVEQC_STATUS_INVALID_ARGUMENT);
    assert(!saved->submitted && !detail.empty());
  };
  if (mode == "batched_overlap") {
    for (std::size_t stride : {0U, 1U, 9U}) {
      auto view = valid;
      view.output.batch_stride = stride;
      rejected(view);
    }
  } else if (mode == "batched_overflow") {
    for (int slot = 0; slot != 3; ++slot) {
      auto view = valid;
      view.batches = 3;
      const auto huge = static_cast<std::size_t>(std::numeric_limits<long long>::max());
      if (slot == 0) view.left.batch_stride = huge;
      if (slot == 1) view.right.batch_stride = huge;
      if (slot == 2) view.output.batch_stride = huge;
      rejected(view);
    }
    // Each vendor dimension fits int, but the physical matrix's byte span does not.
    const auto huge = static_cast<std::size_t>(std::numeric_limits<int>::max());
    for (int slot = 0; slot != 3; ++slot) {
      auto view = valid;
      view.batches = 1;
      if (slot == 0) {
        view.reduction = huge;
        view.left.leading_dimension = huge;
        view.right.leading_dimension = huge;
      } else if (slot == 1) {
        view.columns = huge;
        view.right.leading_dimension = huge;
      } else {
        view.columns = huge;
        view.output.leading_dimension = huge;
      }
      rejected(view);
    }
    const auto end = std::numeric_limits<std::uintptr_t>::max() - sizeof(double) + 1;
    for (int slot = 0; slot != 3; ++slot) {
      auto view = valid;
      if (slot == 0) view.left.data = reinterpret_cast<const double*>(end);
      if (slot == 1) view.right.data = reinterpret_cast<const double*>(end);
      if (slot == 2) view.output.data = reinterpret_cast<double*>(end);
      rejected(view);
    }
  } else {
    for (int slot = 0; slot != 4; ++slot) {
      auto view = valid;
      plan.blas = slot == 0 ? nullptr : saved;
      if (slot == 1) view.left.data = nullptr;
      if (slot == 2) view.right.data = nullptr;
      if (slot == 3) view.output.data = nullptr;
      rejected(view);
    }
    plan.blas = saved;
    for (int slot = 0; slot != 7; ++slot) {
      for (std::size_t value :
           {std::size_t{0}, static_cast<std::size_t>(std::numeric_limits<int>::max()) + 1}) {
        auto view = valid;
        if (slot == 0) view.rows = value;
        if (slot == 1) view.columns = value;
        if (slot == 2) view.reduction = value;
        if (slot == 3) view.batches = value;
        if (slot == 4) view.left.leading_dimension = value;
        if (slot == 5) view.right.leading_dimension = value;
        if (slot == 6) view.output.leading_dimension = value;
        rejected(view);
      }
    }
  }
  saved->result = CUBLAS_STATUS_SUCCESS;
  // A single output may have zero stride. Both inputs may share read-only storage.
  auto single = valid;
  single.batches = 1;
  single.output.batch_stride = 0;
  std::string detail;
  assert(scf_matrix_product_batched_view(plan, single, "single", detail) ==
         GENERATIVEQC_STATUS_SUCCESS);
  // The last column's unused padding is not part of a writable matrix's span.
  auto adjacent = valid;
  adjacent.output.batch_stride = 10;
  assert(scf_matrix_product_batched_view(plan, adjacent, "adjacent", detail) ==
         GENERATIVEQC_STATUS_SUCCESS);
}

void gram(CudaDensityFittingJkPlan& plan) {
  constexpr int n = 3, k = 2;
  double coefficients[] = {1, 2, 3, -1, 4, 2};
  double output[n * n];
  for (bool lower : {false, true}) {
    std::fill(std::begin(output), std::end(output), sentinel);
    bool wrote_lower = false;
    std::string detail = "unchanged";
    assert(scf_occupied_gram(plan, n, k, 1.25, coefficients, output, lower, wrote_lower,
                             "occupied gram", detail) == GENERATIVEQC_STATUS_SUCCESS);
    assert(wrote_lower == lower && detail == "unchanged");
    for (int i = 0; i < n; ++i)
      for (int j = 0; j < n; ++j) {
        const double expected = 1.25 * (coefficients[i * k] * coefficients[j * k] +
                                        coefficients[i * k + 1] * coefficients[j * k + 1]);
        if (lower && i < j)
          assert(output[i + j * n] == sentinel);
        else
          assert(output[i + j * n] == expected);
      }
  }
  bool wrote_lower = true;
  std::string detail;
  plan.blas->submitted = false;
  assert(scf_occupied_gram(plan, 0, k, 1, coefficients, output, false, wrote_lower, "bad gram",
                           detail) == GENERATIVEQC_STATUS_INVALID_ARGUMENT);
  assert(!plan.blas->submitted && !wrote_lower);
  for (const auto error : {CUBLAS_STATUS_ALLOC_FAILED, CUBLAS_STATUS_EXECUTION_FAILED}) {
    plan.blas->result = error;
    assert(scf_occupied_gram(plan, n, k, 1, coefficients, output, true, wrote_lower, "failed gram",
                             detail) == (error == CUBLAS_STATUS_ALLOC_FAILED
                                             ? GENERATIVEQC_STATUS_OUT_OF_MEMORY
                                             : GENERATIVEQC_STATUS_CUDA_ERROR));
    assert(detail ==
           "failed gram failed with cuBLAS status " + std::to_string(static_cast<int>(error)));
  }
  plan.blas->result = CUBLAS_STATUS_SUCCESS;
}

void owner(CudaDensityFittingJkPlan& plan) {
  const auto existing = plan.blas;
  plan.blas = nullptr;
  const auto created = handles_created, destroyed = handles_destroyed;
  std::string detail;
  assert(prepare_df_blas(plan, detail) == GENERATIVEQC_STATUS_SUCCESS);
  assert(plan.blas && plan.blas->stream == plan.stream && detail.empty());
  assert(prepare_df_blas(plan, detail) == GENERATIVEQC_STATUS_INVALID_ARGUMENT);
  release_df_blas(plan);
  assert(!plan.blas && handles_created == created + 1 && handles_destroyed == destroyed + 1);
  release_df_blas(plan);
  assert(handles_destroyed == destroyed + 1);
  for (int phase = 0; phase < 3; ++phase) {
    create_error = phase == 0 ? CUBLAS_STATUS_ALLOC_FAILED : CUBLAS_STATUS_SUCCESS;
    stream_error = phase == 1 ? CUBLAS_STATUS_EXECUTION_FAILED : CUBLAS_STATUS_SUCCESS;
    mode_error = phase == 2 ? CUBLAS_STATUS_INVALID_VALUE : CUBLAS_STATUS_SUCCESS;
    const auto expected =
        phase == 0 ? GENERATIVEQC_STATUS_OUT_OF_MEMORY : GENERATIVEQC_STATUS_CUDA_ERROR;
    assert(prepare_df_blas(plan, detail) == expected);
    assert(detail.find("initialize CUDA DF cuBLAS") != std::string::npos);
    release_df_blas(plan);
    assert(!plan.blas && handles_created == handles_destroyed);
  }
  create_error = stream_error = mode_error = CUBLAS_STATUS_SUCCESS;
  plan.blas = existing;
}

void borrowed_response(cublasHandle_t handle) {
  double a[6] = {1, 2, 3, 4, 5, 6};
  double b[6] = {2, -1, 3, 0, 1, 4};
  double c[6]{};
  const double one = 1.0, zero = 0.0;
  handle->submitted = false;
  assert(generativeqc_tensor::submit_column_gemm(
             {handle, false, false, 2, 3, 2, &one, a, 2, b, 2, &zero, c, 2}) ==
         CUBLAS_STATUS_SUCCESS);
  assert(handle->submitted);
  for (int j = 0; j < 3; ++j)
    for (int i = 0; i < 2; ++i)
      assert(c[i + j * 2] == a[i] * b[j * 2] + a[i + 2] * b[1 + j * 2]);

  double y[6] = {7, sentinel, 8, sentinel, 9, sentinel};
  double x[2] = {2, 3};
  const double half = 0.5;
  assert(generativeqc_tensor::submit_column_gemv(
             {handle, true, 2, 3, &one, a, 2, x, 1, &half, y, 2}) ==
         CUBLAS_STATUS_SUCCESS);
  assert(y[0] == 8 + half * 7 && y[2] == 18 + half * 8 && y[4] == 28 + half * 9);
  assert(y[1] == sentinel && y[3] == sentinel && y[5] == sentinel);

  double transpose[6]{};
  assert(generativeqc_tensor::submit_column_geam(
             generativeqc_tensor::ColumnGeamView<cublasHandle_t>{
                 handle, true, true, 3, 2, &one, a, 2, &zero, a, 2, transpose, 3}) ==
         CUBLAS_STATUS_SUCCESS);
  for (int j = 0; j < 2; ++j)
    for (int i = 0; i < 3; ++i) assert(transpose[i + j * 3] == a[j + i * 2]);

  double alpha[] = {1, 0, 0, 1}, right[] = {2, 3, 4, 5, 6, 7, 8, 9};
  double output[8]{};
  assert(generativeqc_tensor::submit_column_batched_gemm(
             {handle, false, false, 2, 2, 2, &one, alpha, 2, 0, right, 2, 4,
              &zero, output, 2, 4, 2}) == CUBLAS_STATUS_SUCCESS);
  for (int i = 0; i < 8; ++i) assert(output[i] == right[i]);

  handle->result = CUBLAS_STATUS_EXECUTION_FAILED;
  assert(generativeqc_tensor::submit_column_gemm(
             {handle, false, false, 2, 2, 2, &one, a, 2, b, 2, &zero, output, 2}) ==
         CUBLAS_STATUS_EXECUTION_FAILED);
  assert(generativeqc_tensor::submit_column_gemv(
             {handle, false, 2, 2, &one, a, 2, x, 1, &zero, y, 1}) ==
         CUBLAS_STATUS_EXECUTION_FAILED);
  handle->result = CUBLAS_STATUS_SUCCESS;
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
  else if (mode == "batched_overlap" || mode == "batched_overflow" || mode == "batched_invalid")
    batched_admission(plan, mode);
  else if (mode == "gram")
    gram(plan);
  else if (mode == "owner")
    owner(plan);
  else if (mode == "borrowed_response")
    borrowed_response(plan.blas);
  else
    return 1;
  assert(plan.blas == &handle && plan.stream == stream && handle.stream == stream);
}
