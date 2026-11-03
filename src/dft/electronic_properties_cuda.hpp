#pragma once

#include <cstddef>
#include <memory>

#include "core/types.hpp"

namespace generativeqc::dft {

/** Detached, device-bound FP64 property state. No system/context pointers are
 * borrowed. Implementations synchronize publication and restore the caller's
 * CUDA device; callers serialize operations and destruction on one owner. */
class CudaElectronicProperties {
 public:
  virtual ~CudaElectronicProperties() = default;
  virtual void analyze(double* dipole, double* mulliken, double* lowdin) = 0;
  virtual void evaluate(const double* points, std::size_t count, bool spin, double* output) = 0;
  virtual std::size_t atoms() const noexcept = 0;
  virtual int device() const noexcept = 0;
};

/** Upload one accepted restricted total or alpha/beta density. Packing uses
 * native normalized through-f AO expansions; no host integrals are evaluated. */
std::unique_ptr<CudaElectronicProperties> make_cuda_electronic_properties(
    const core::System& system, int device, const double* density, std::size_t count,
    std::size_t point_capacity);

}  // namespace generativeqc::dft
