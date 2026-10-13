#pragma once

#include "scf/cuda/df_scf_state.hpp"
#include "scf/cuda_density_fitting.hpp"

namespace generativeqc::scf::cuda_df {

/** cuBLAS/cuSOLVER integration for retained DF SCF state.
 * Borrow plan streams and own solver workspace setup independently of J/K selection.
 */
/** Borrowed strict-FP64 column-major matrix product for DF SCF owners.
 * The caller specifies mathematical transposes, M/N/K, physical leading
 * dimensions and accumulation scalars. This adapter alone submits to the
 * retained provider; it never allocates, changes the stream/math policy,
 * reselects a backend, or prepares a handle during CUDA Graph capture.
 */
/** Borrowed DF BLAS handle preparation and disposal. The DF plan owns the
 * stream and lifetime; this provider implementation owns NVIDIA setup/modes.
 * There is no allocation, provider search or handle state mutation at replay.
 * A partially created handle is retained for failure cleanup by release().
 */
generativeqc_status prepare_df_blas(CudaDensityFittingJkPlan& plan, std::string& detail);
void release_df_blas(CudaDensityFittingJkPlan& plan) noexcept;

/** Explicit Gram operation. Lower-triangle publication is a semantic request,
 * not a vendor selector: if SYRK is unavailable the incumbent full GEMM is
 * preserved. The caller mirrors the lower triangle only when actually used.
 */
generativeqc_status scf_occupied_gram(CudaDensityFittingJkPlan& plan, std::size_t columns,
                                      std::size_t reduction, double weight, const double* factor,
                                      double* output, bool lower_triangle, bool& wrote_lower,
                                      const char* failure_context, std::string& detail);

generativeqc_status scf_matrix_product_view(
    CudaDensityFittingJkPlan& plan, bool transpose_left, bool transpose_right, std::size_t rows,
    std::size_t columns, std::size_t reduction, const double* left, std::size_t left_ld,
    const double* right, std::size_t right_ld, double* output, std::size_t output_ld, double alpha,
    double beta, const char* failure_context, std::string& detail);

/** Explicit physical batched matrix views in column-major storage.
 * Zero input batch stride denotes read-only broadcasting. All strides are
 * element counts (not bytes); provider selection and resource lifetime stay
 * outside scientific callers. Writable batch spans must be disjoint; single
 * batches may use any representable stride. Callers retain allocation capacity
 * and lifetime ownership. No handles or device storage are created here.
 */
struct ScfMatrixBatchView {
  struct Input {
    const double* data{};
    std::size_t leading_dimension{}, batch_stride{};
  };
  struct Output {
    double* data{};
    std::size_t leading_dimension{}, batch_stride{};
  };
  Input left{}, right{};
  Output output{};
  std::size_t rows{}, columns{}, reduction{}, batches{};
  bool transpose_left{}, transpose_right{};
  double alpha{1.0}, beta{};
};

generativeqc_status scf_matrix_product_batched_view(CudaDensityFittingJkPlan& plan,
                                                    const ScfMatrixBatchView& view,
                                                    const char* failure_context,
                                                    std::string& detail);

/** Submit an explicit strided batched GEMM through the borrowed DF SCF BLAS provider.
 * Scientific callers own equation/order semantics; this boundary owns only
 * library submission, strides, accumulation coefficients, and failure mapping.
 */
generativeqc_status scf_gemm_strided(CudaDensityFittingJkPlan& plan, bool transpose_left,
                                     std::size_t batch_size, std::size_t nbf, const double* left,
                                     std::size_t left_stride, const double* right,
                                     std::size_t right_stride, double* output,
                                     std::size_t output_stride, double alpha, double beta,
                                     const char* failure_context, std::string& detail);

generativeqc_status scf_gemm(CudaDensityFittingJkPlan& plan, bool transpose_left,
                             std::size_t batch_size, std::size_t nbf, const double* left,
                             const double* right, double* output, std::string& detail);

/** Execute the shared canonical-X reduction or recovery on borrowed square
 * column-major storage. The selected spectral phase remains separately callable. */
generativeqc_status scf_generalized_transform(CudaDensityFittingJkPlan& plan, bool recovery,
                                              std::size_t batch_size, std::size_t nbf,
                                              double* matrix, const double* orthogonalizer,
                                              double* temporary, std::string& detail);

generativeqc_status setup_device_solver(CudaDensityFittingJkPlan& plan, std::size_t nbf,
                                        std::size_t batch_size, double* eigensystem,
                                        double* eigenvalues, DeviceSolver& solver,
                                        std::string& detail);

/** Submit one eigensystem per system/spin on the owning plan stream.
 * Diagnostic tracing distinguishes graph construction from ordinary execution. */
generativeqc_status solve_device_batch(CudaDensityFittingJkPlan& plan, DeviceSolver& solver,
                                       std::size_t nbf, std::size_t batch_size, double* eigensystem,
                                       double* eigenvalues, int* info, std::string& detail);

/** After ending a failed capture, clear only expected capture/mode errors.
 * Ordinary launches must start on a noncapturing stream with no stale CUDA
 * error. Unrelated runtime failures remain failures; this is not a CPU retry. */
generativeqc_status recover_scf_capture(cudaStream_t stream, cudaError_t capture_error,
                                        generativeqc_status iteration_status,
                                        bool& capture_rejected, std::string& detail);

}  // namespace generativeqc::scf::cuda_df
