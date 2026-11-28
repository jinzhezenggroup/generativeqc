#pragma once

#include <cuda_runtime_api.h>

#include <cstddef>
#include <cstdint>

namespace generativeqc::dft::hessian_aot {

struct FactorizedIntegralBatch;

/** Direction-independent Gaussian geometry and Coulomb moments. A dense
 * window may bind these once for all evaluated nuclear directions. */
struct FactorizedPairGeometry {
  double inverse_two_p, inverse_two_q, rho;
  double difference[3], shifts[4][3], product_scales[4], decay[4][3];
  double prefactor, boys[15];
};

/** One independent component writer. The low seven bits of selector are
 * the Cartesian component; the next four are the s/p shell mask. */
struct FactorizedActiveComponent {
  std::uint32_t batch, selector;
};

/** Dispatch a bounded, degree-homogeneous slice of a prepared window.
 * max_radials bounds the primitive producers per component in the slice. */
int launch_factorized_active_components(unsigned degree, const FactorizedActiveComponent* tasks,
                                        std::size_t count, std::size_t max_radials,
                                        const FactorizedIntegralBatch* batches,
                                        const double* direction, int* status, cudaStream_t stream,
                                        const FactorizedPairGeometry* prepared,
                                        const double* window_records) noexcept;

/** Full-range s/p resident batches of up to two 64-record chunks. Each
 * descriptor supplies the whole radial interval and a field-major cache;
 * the shell orders give the stride, independent of the selected components. */
int launch_factorized_radial_pair(const unsigned* angular, const std::uint32_t* components,
                                  std::size_t component_count,
                                  const FactorizedIntegralBatch* batches, std::size_t batch_count,
                                  const double* direction, long long frozen_atom, int* status,
                                  cudaStream_t stream,
                                  const FactorizedPairGeometry* prepared = nullptr,
                                  const double* window_records = nullptr) noexcept;

}  // namespace generativeqc::dft::hessian_aot
