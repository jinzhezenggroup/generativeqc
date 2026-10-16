#pragma once

#include <cuda_runtime_api.h>

#include <array>
#include <cstddef>
#include <cstdint>

namespace generativeqc::dft::hessian_aot {

/** Internal build-time binding to the shared compiler-emitted factor library.
 * The descriptor is copied into the kernel launch; no host metadata is borrowed
 * by queued device work. Every selected component remains explicit in Program.
 * The enclosing owner retains all numerical buffers and gates device status.
 */
int launch_factorized_integral(unsigned family, unsigned radial, const unsigned* angular,
                               const std::uint32_t* components, std::size_t component_count,
                               const double* records, std::size_t count, const long long* atoms,
                               const double* direction, double* output, int* status,
                               cudaStream_t stream) noexcept;

/** Distinct constant selectors retain the established context-free leaf ABI.
 * Host-only metadata units instantiate these small forwarders instead of
 * compiling the same high-angular CUDA derivative algebra for every tile.
 */
template <unsigned Family, unsigned Radial, unsigned First, unsigned Second, unsigned Third,
          unsigned Fourth, unsigned... Components>
int factorized_integral_callback(const double* records, std::size_t count, const long long* atoms,
                                 const double* direction, double* output, int* status,
                                 cudaStream_t stream) {
  static constexpr std::array<unsigned, 4> angular{First, Second, Third, Fourth};
  static constexpr std::array<std::uint32_t, sizeof...(Components)> components{Components...};
  return launch_factorized_integral(Family, Radial, angular.data(), components.data(),
                                    components.size(), records, count, atoms, direction, output,
                                    status, stream);
}

}  // namespace generativeqc::dft::hessian_aot
