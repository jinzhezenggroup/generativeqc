#include "scf/explicit_integral_backend.hpp"

#include <stdexcept>
#if !GENERATIVEQC_HAS_CUDA
namespace generativeqc::scf {
std::unique_ptr<ExplicitIntegralBackend> make_cuda_explicit_integrals(int, std::size_t,
                                                                      const std::vector<double>&,
                                                                      std::size_t) {
  throw std::runtime_error("CUDA explicit-integral backend is unavailable in this build");
}
}  // namespace generativeqc::scf
#endif
