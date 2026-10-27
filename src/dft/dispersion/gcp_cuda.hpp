#pragma once

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
  static std::size_t numeric_bytes(std::size_t atoms);
  GcpCudaPlan(int device, std::span<const std::int32_t> numbers, std::size_t maximum_bytes);
  ~GcpCudaPlan();
  GcpCudaPlan(const GcpCudaPlan&) = delete;
  GcpCudaPlan& operator=(const GcpCudaPlan&) = delete;
  void execute(std::span<const double> xyz, double& energy, std::span<double> gradient);

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};
}  // namespace generativeqc::dft::dispersion
