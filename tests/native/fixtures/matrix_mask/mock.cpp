// Host execution/queue doubles only: this does not qualify CUDA or cuBLAS numerics.
#include <cassert>
#include <functional>
#include <vector>

#include "scf/cuda/matrix_library.hpp"
#include "scf/cuda/scf_matrix_kernels.hpp"

namespace {
std::vector<std::function<void()>> queue;
int gemms{}, batched{}, copies{}, natives{}, fail_gemm{}, copy_error{}, synchronizations{};
cudaStream_t expected_stream = reinterpret_cast<void*>(1);

long long matrix_offset(int row, int col, int leading_dimension) {
  // cuBLAS dimensions are int, but matrix offsets may exceed INT_MAX.
  return static_cast<long long>(col) * leading_dimension + row;
}
}  // namespace

cudaError_t cudaStreamIsCapturing(cudaStream_t, cudaStreamCaptureStatus* c) {
  *c = 0;
  return 0;
}
cudaError_t cudaGetDevice(int* d) {
  *d = 0;
  return 0;
}
cudaError_t cudaSetDevice(int) { return 0; }
cudaError_t cudaMemGetInfo(std::size_t* f, std::size_t* t) {
  *f = *t = 1U << 30;
  return 0;
}
cudaError_t cudaStreamSynchronize(cudaStream_t) {
  ++synchronizations;
  return 0;
}
cudaError_t cudaGetLastError() { return 0; }
cudaError_t cudaPeekAtLastError() { return copies ? copy_error : 0; }
cublasStatus_t cublasCreate(cublasHandle_t* h) {
  *h = reinterpret_cast<void*>(2);
  return 0;
}
cublasStatus_t cublasDestroy(cublasHandle_t) { return 0; }
cublasStatus_t cublasSetStream(cublasHandle_t, cudaStream_t s) {
  assert(s == expected_stream);
  return 0;
}
cublasStatus_t cublasSetPointerMode(cublasHandle_t, int) { return 0; }
cublasStatus_t cublasSetMathMode(cublasHandle_t, int) { return 0; }
cublasStatus_t cublasSetWorkspace(cublasHandle_t, void*, std::size_t) { return 0; }

cublasStatus_t cublasDgemmStridedBatched(cublasHandle_t h, cublasOperation_t op,
                                         cublasOperation_t opb, int m, int n, int k,
                                         const double* alpha, const double* a, int lda,
                                         long long sa, const double* b, int ldb, long long sb,
                                         const double* beta, double* c, int ldc, long long sc,
                                         int batch) {
  assert(h == reinterpret_cast<void*>(2) && opb == CUBLAS_OP_N);
  ++gemms;
  if (fail_gemm == gemms) return 13;
  const double av = *alpha, bv = *beta;
  assert(bv == 0.0);
  queue.emplace_back([=] {
    for (int s = 0; s < batch; ++s)
      for (int col = 0; col < n; ++col)
        for (int row = 0; row < m; ++row) {
          double value = 0;
          for (int q = 0; q < k; ++q)
            value += a[s * sa + (op == CUBLAS_OP_T ? matrix_offset(q, row, lda)
                                                   : matrix_offset(row, q, lda))] *
                     b[s * sb + matrix_offset(q, col, ldb)];
          c[s * sc + matrix_offset(row, col, ldc)] = av * value;
        }
  });
  if (batch > 1) ++batched;
  return 0;
}

cublasStatus_t cublasDgemm(cublasHandle_t h, cublasOperation_t opa, cublasOperation_t opb, int m,
                           int n, int k, const double* alpha, const double* a, int lda,
                           const double* b, int ldb, const double* beta, double* c, int ldc) {
  return cublasDgemmStridedBatched(h, opa, opb, m, n, k, alpha, a, lda, 0, b, ldb, 0, beta, c, ldc,
                                   0, 1);
}

// Implemented by the test-generated host execution of the unchanged production
// copy_selected_matrices_kernel body. CUDA block scheduling is not modeled.
void host_selected_copy(dim3, dim3, int, int, int, const std::uint8_t*, const double*, double*);

namespace generativeqc::scf::cuda_execution {
generativeqc_status cuda_status(cudaError_t s) { return s ? GENERATIVEQC_STATUS_CUDA_ERROR : 0; }
generativeqc_status blas_status(cublasStatus_t s) { return s ? GENERATIVEQC_STATUS_CUDA_ERROR : 0; }
void launch_copy_selected_matrices_kernel(dim3 grid, dim3 block, std::size_t shared,
                                          cudaStream_t stream, std::int32_t batch,
                                          std::int32_t spins, std::int32_t n,
                                          const std::uint8_t* active, const double* source,
                                          double* output) {
  assert(stream == expected_stream && shared == 0);
  ++copies;
  if (!copy_error)
    queue.emplace_back(
        [=] { host_selected_copy(grid, block, batch, spins, n, active, source, output); });
}
void launch_matrix_product_kernel(dim3, dim3, std::size_t, cudaStream_t stream, std::int32_t,
                                  std::int32_t, const double*, bool, const double*,
                                  const std::uint8_t*, double*, double) {
  assert(stream == expected_stream);
  ++natives;
}
void launch_spin_matrix_product_kernel(dim3, dim3, std::size_t, cudaStream_t stream, std::int32_t,
                                       std::int32_t, std::int32_t, const double*, bool, bool,
                                       const double*, bool, const std::uint8_t*, double*) {
  assert(stream == expected_stream);
  ++natives;
}
}  // namespace generativeqc::scf::cuda_execution

extern "C" {
long long probe_matrix_offset(int row, int col, int leading_dimension) {
  return matrix_offset(row, col, leading_dimension);
}
void probe_reset(int fail, int copy_failure) {
  queue.clear();
  gemms = batched = copies = natives = synchronizations = 0;
  fail_gemm = fail;
  copy_error = copy_failure;
}
int probe_stat(int i) {
  const int values[] = {gemms,   batched,          copies,
                        natives, synchronizations, static_cast<int>(queue.size())};
  return values[i];
}
void probe_replay() {
  for (const auto& op : queue) op();
}
int probe_submit(int spin_api, int batch, int spins, int n, int left_spin, int transpose,
                 int right_spin, const double* left, const double* right,
                 const std::uint8_t* active, double* output, double* scratch, std::size_t capacity,
                 int library, double scale) {
  using namespace generativeqc::scf::cuda_execution;
  MatrixLibraryResources r{expected_stream, reinterpret_cast<void*>(2), scratch, capacity};
  if (spin_api)
    return launch_spin_matrix_product(r, batch, spins, n, left, left_spin, transpose, right,
                                      right_spin, active, output, library);
  return launch_matrix_product(r, batch, n, left, transpose, right, active, output, library, scale);
}
}
