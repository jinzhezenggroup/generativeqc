#pragma once

#include "scf/cuda/df_scf_state.hpp"
#include "scf/cuda_density_fitting.hpp"

namespace generativeqc::scf::cuda_df {

/** cuBLAS/cuSOLVER integration for retained DF SCF state.
 * Borrow plan streams and own solver workspace setup independently of J/K selection.
 */
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
