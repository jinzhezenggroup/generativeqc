#pragma once

#include <cuda_runtime_api.h>

#include <cstddef>

namespace generativeqc::response {

enum class CudaGmresStatus : int {
  running,
  converged,
  iteration_limit,
  breakdown,
  nonfinite_input,
  invalid_preconditioner,
  nonfinite_operator,
  numerical_failure,
};

struct CudaGmresOptions {
  std::size_t restart{32};
  std::size_t max_iterations{256};
  unsigned reorthogonalizations{2};
  double relative_tolerance{1e-10};
  double absolute_tolerance{1e-12};
  double breakdown_relative_tolerance{1e-14};
  double diagonal_floor{1e-12};
  std::size_t max_workspace_bytes{256ULL << 20};
};

struct CudaGmresResult {
  CudaGmresStatus status{CudaGmresStatus::running};
  std::size_t iterations{};
  std::size_t operator_actions{};
  std::size_t preconditioner_actions{};
  std::size_t restarts{};
  std::size_t workspace_bytes{};
  std::size_t checkpoint_bytes{};
  double rhs_norm{};
  double residual_norm{};
  double tolerance{};
  bool converged() const noexcept { return status == CudaGmresStatus::converged; }
};

/** Private, stream-ordered result prefix of the linked GMRES arena. Read it
 * on device after enqueue, before the next replay overwrites the arena. */
struct CudaGmresDeviceReceipt {
  int code;
  std::size_t iterations, restarts;
  double rhs_norm, residual_norm, tolerance;
};

/** Enqueue a native CUDA action on exactly the borrowed stream.
 * Inputs/outputs are distinct dimension-element device panels. The callback
 * must not retain them, change device/stream, transfer host data, or perform
 * host numerical work. The stream captures a reusable action topology, so a
 * callback invocation is not a count of actions executed.
 * Its owner remains responsible for prepared-state authentication and any
 * additional workspace. A failed action never authorizes solution publication.
 */
using CudaGmresAction = cudaError_t (*)(void*, const double*, double*, std::size_t,
                                        cudaStream_t) noexcept;

/** Pure host admission: exact caller-owned workspace, including device scalars.
 * No GPU access or allocation. Invalid options, overflow and budget refusal
 * return cudaErrorInvalidValue and set bytes to zero.
 */
cudaError_t cuda_gmres_workspace(std::size_t dimension, const CudaGmresOptions& options,
                                 std::size_t& bytes) noexcept;

/** One reusable graph for identical unseeded strict solves. Bound pointers
 * and the provider's captured storage must outlive replay and cannot change
 * identity. The caller owns any setup/terminal transfer and outer transaction.
 * A nonzero upstream audit suppresses the iteration body, and only converged
 * candidates reach the private solution. Noncopyable, single-stream owner. */
class CudaGmresReplay {
 public:
  CudaGmresReplay() = default;
  ~CudaGmresReplay();
  CudaGmresReplay(const CudaGmresReplay&) = delete;
  CudaGmresReplay& operator=(const CudaGmresReplay&) = delete;

  cudaError_t prepare(std::size_t dimension, const CudaGmresOptions& options,
                      CudaGmresAction action, void* context, const double* rhs,
                      const double* diagonal, double* private_solution, void* workspace,
                      std::size_t workspace_bytes, cudaStream_t stream) noexcept;
  cudaError_t enqueue(const int* upstream_status = nullptr) const noexcept;
  const CudaGmresDeviceReceipt* receipt() const noexcept { return receipt_; }

 private:
  cudaGraphExec_t execution_{};
  cudaStream_t stream_{};
  CudaGmresOptions options_{};
  const double *rhs_{}, *diagonal_{};
  double* solution_{};
  void* workspace_{};
  std::size_t dimension_{}, required_{};
  const CudaGmresDeviceReceipt* receipt_{};
};

/** Restarted, right-preconditioned FP64 GMRES with device-only numerics.
 * The optional diagonal is the divisor, not its inverse; signed entries are
 * allowed, but every magnitude must exceed diagonal_floor. Modified Gram
 * Schmidt, stable norms, Givens rotations, triangular solves, candidates and
 * true UNPRECONDITIONED residuals all execute in linked CUDA code. A bounded
 * conditional graph controls the iterations; the host reads one setup status
 * and the terminal receipt, never per-iteration status. No interpreter, runtime
 * compiler, BLAS handle, extra scientific workspace or CPU fallback is used.
 *
 * The workspace must be double-aligned, resident on the current CUDA device,
 * and disjoint from rhs, diagonal and solution. The solution must be disjoint
 * from inputs. It is modified ONLY after a true-residual convergence gate.
 * An optional initial guess is copied into private workspace before solving;
 * it may exactly alias solution. Tolerance is still relative to the original
 * RHS and the initial residual uses the supplied operator, not the guess's
 * original proposal operator.
 * This call synchronizes the borrowed stream at setup and before return;
 * the enclosing owner must not use these panels concurrently on other streams.
 * Numerical failure is reported in result; CUDA/contract failure is the return
 * value. Empty systems need no action or input/output pointers, but still use
 * the reported workspace. This is a solver, not a molecular CPKS capability.
 */
cudaError_t solve_cuda_gmres(std::size_t dimension, const CudaGmresOptions& options,
                             CudaGmresAction action, void* context, const double* rhs,
                             const double* diagonal, double* solution, void* workspace,
                             std::size_t workspace_bytes, cudaStream_t stream,
                             CudaGmresResult& result,
                             const double* initial_guess = nullptr) noexcept;

}  // namespace generativeqc::response
