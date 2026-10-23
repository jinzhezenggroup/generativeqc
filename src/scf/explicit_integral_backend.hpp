#pragma once
#include <memory>

#include "scf/fock_build.hpp"
#include "scf/initial_guess/eigen_operation.hpp"

namespace generativeqc::scf {
/** Backend operations on caller-supplied integrals. The prepared owner retains
 * this object, its device ERIs and stream for the complete SCF/response solve. */
class ExplicitIntegralBackend {
 public:
  virtual ~ExplicitIntegralBackend() = default;
  virtual DirectJkMatrices build(const FockBuildSpec&, const std::vector<double>&,
                                 const std::vector<double>&) = 0;
  virtual reference::EigenResult eigen(const std::vector<double>&, const std::vector<double>*,
                                       const std::vector<double>*, std::size_t) = 0;
};
std::unique_ptr<ExplicitIntegralBackend> make_cuda_explicit_integrals(
    int device, std::size_t n, const std::vector<double>& eri, std::size_t budget);
}  // namespace generativeqc::scf
