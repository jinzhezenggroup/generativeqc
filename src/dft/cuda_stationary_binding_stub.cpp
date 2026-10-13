#include "dft/cuda_stationary_binding.hpp"

namespace generativeqc::dft {
const CudaStationaryBinding* cuda_stationary_binding(std::string_view) noexcept { return nullptr; }
const CudaStationaryBinding* cuda_stationary_binding(const CudaStationaryModel&) noexcept {
  return nullptr;
}
std::span<const CudaStationaryBinding* const> cuda_stationary_bindings() noexcept { return {}; }
}  // namespace generativeqc::dft
