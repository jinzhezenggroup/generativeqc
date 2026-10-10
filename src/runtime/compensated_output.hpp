#pragma once

namespace generativeqc::runtime {

/** Runtime-owned scatter planes shared by host launchers and CUDA kernels.
 * A pointer-only call preserves ordinary accumulation. The optional correction
 * belongs to the same stream and matrix layout as sum, and is folded by its
 * owner only after every generated and compatibility writer completes.
 */
struct CompensatedOutput {
  double* sum;
  double* correction;

#ifdef __CUDACC__
  __host__ __device__
#endif
      constexpr CompensatedOutput(double* sum_plane, double* correction_plane = nullptr)
      : sum(sum_plane), correction(correction_plane) {
  }
};

}  // namespace generativeqc::runtime
