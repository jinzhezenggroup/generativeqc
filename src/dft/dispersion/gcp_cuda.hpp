#pragma once

#include <cuda_runtime_api.h>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>

namespace generativeqc::dft::dispersion {
/** Retained, bounded CUDA gCP owner for the canonical H-Ar r2SCAN-3c model.
 * Coordinates are host inputs; every pair equation and the energy reduction
 * execute on the selected device. Output publication follows stream completion.
 */
class GcpCudaPlan {
 public:
  static std::size_t numeric_bytes(std::size_t atoms, bool second_order = false);
  GcpCudaPlan(int device, std::span<const std::int32_t> numbers, std::size_t maximum_bytes,
              bool second_order = false);
  ~GcpCudaPlan();
  GcpCudaPlan(const GcpCudaPlan&) = delete;
  GcpCudaPlan& operator=(const GcpCudaPlan&) = delete;
  void execute(std::span<const double> xyz, double& energy, std::span<double> gradient);
  /** Apply the analytic Cartesian Hessian of gCP, without storing a full matrix.
   * Requires a second-order capacity reservation at construction. */
  void apply_hessian(std::span<const double> xyz, std::span<const double> direction,
                     std::span<double> output);
  /** Borrow only immutable tables. The locked Hessian owner provides all
   * device panels, including one status per atom, on its own stream. */
  cudaError_t launch_hessian_device(const double* xyz, const double* direction, double* output,
                                    int* statuses, cudaStream_t stream) const noexcept;

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};
}  // namespace generativeqc::dft::dispersion
