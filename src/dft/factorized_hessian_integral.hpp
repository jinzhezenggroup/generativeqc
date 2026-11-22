#pragma once

#include <cuda_runtime_api.h>

#include <array>
#include <cstddef>
#include <cstdint>

namespace generativeqc::dft::hessian_aot {

/** One checked, owner-resident primitive chunk. Batched launches share only
 * scientific selectors; records, center incidences and output remain distinct.
 * The numerical owner validates extents and retains every pointee on stream. */
struct FactorizedIntegralBatch {
  const double* records{};
  std::size_t count{};
  long long atoms[4]{};
  double* output{};
  /** Optional shared molecular coordinates. Non-null selects compact records
   * [center exponents,scale]; atoms supplies each mathematical center position.
   * Null retains the ordinary complete primitive-record layout. */
  const double* coordinates{};
};

/** An ERI's directional derivatives vanish if the selected atom is absent
 * or if it translates all four centers together. Negative selects all. */
inline __host__ __device__ bool active_eri_centers(const long long* atoms, long long atom) {
  if (atom < 0) return true;
  unsigned matches = 0;
  for (unsigned center = 0; center < 4; ++center) matches += atoms[center] == atom;
  return matches != 0 && matches != 4;
}

/** Internal build-time binding to the shared compiler-emitted factor library.
 * The descriptor is copied into the kernel launch; no host metadata is borrowed
 * by queued device work. Every selected component remains explicit in Program.
 * The enclosing owner retains all numerical buffers and gates device status.
 */
int launch_factorized_integral(unsigned family, unsigned radial, double omega,
                               const unsigned* angular, const std::uint32_t* components,
                               std::size_t component_count, const double* records,
                               std::size_t count, const long long* atoms, const double* direction,
                               double* output, int* status, cudaStream_t stream,
                               const FactorizedIntegralBatch* batches = nullptr,
                               std::size_t batch_count = 1, long long frozen_atom = -1) noexcept;

/** Distinct constant selectors retain the established context-free leaf ABI.
 * Host-only metadata units instantiate these small forwarders instead of
 * compiling the same high-angular CUDA derivative algebra for every tile.
 */
template <unsigned Family, unsigned Radial, unsigned First, unsigned Second, unsigned Third,
          unsigned Fourth, unsigned... Components>
int factorized_range_integral_callback(double omega, const double* records, std::size_t count,
                                       const long long* atoms, const double* direction,
                                       double* output, int* status, cudaStream_t stream) {
  static constexpr std::array<unsigned, 4> angular{First, Second, Third, Fourth};
  static constexpr std::array<std::uint32_t, sizeof...(Components)> components{Components...};
  return launch_factorized_integral(Family, Radial, omega, angular.data(), components.data(),
                                    components.size(), records, count, atoms, direction, output,
                                    status, stream);
}

template <unsigned Family, unsigned Radial, unsigned First, unsigned Second, unsigned Third,
          unsigned Fourth, unsigned... Components>
int factorized_integral_callback(const double* records, std::size_t count, const long long* atoms,
                                 const double* direction, double* output, int* status,
                                 cudaStream_t stream) {
  return factorized_range_integral_callback<Family, Radial, First, Second, Third, Fourth,
                                            Components...>(
      Radial == 0 ? 0.0 : 0.3, records, count, atoms, direction, output, status, stream);
}

/** Optional batching keeps the same selectors and compiler primitive as the
 * scalar callback. Inventories without this binding retain scalar execution. */
template <unsigned Family, unsigned Radial, unsigned First, unsigned Second, unsigned Third,
          unsigned Fourth, unsigned... Components>
int factorized_integral_batch_callback(double omega, const FactorizedIntegralBatch* batches,
                                       std::size_t count, const double* direction,
                                       long long frozen_atom, int* status, cudaStream_t stream) {
  static constexpr std::array<unsigned, 4> angular{First, Second, Third, Fourth};
  static constexpr std::array<std::uint32_t, sizeof...(Components)> components{Components...};
  return launch_factorized_integral(Family, Radial, omega, angular.data(), components.data(),
                                    components.size(), nullptr, 0, nullptr, direction, nullptr,
                                    status, stream, batches, count, frozen_atom);
}

}  // namespace generativeqc::dft::hessian_aot
