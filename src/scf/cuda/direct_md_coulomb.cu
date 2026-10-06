#include "generated_direct_md_coulomb.cuh"
#include "scf/cuda/direct_md_coulomb.hpp"

namespace generativeqc::scf::cuda_execution {

std::uint64_t direct_md_coulomb_shell_class_mask() noexcept { return kDirectMDCoulombMask; }
unsigned direct_md_coulomb_hermite_width() noexcept { return kDirectMDHermiteWidth; }

cudaError_t prepare_direct_md_coulomb_density(cudaStream_t stream, DeviceBatch batch,
                                              const double* density, double* density_hermite,
                                              double* potential_hermite,
                                              const std::uint8_t* active) {
  if (batch.total_shell_pairs == 0) return cudaSuccess;
  direct_md_pair_transform_kernel<true><<<batch.total_shell_pairs, 32, 0, stream>>>(
      batch, density, density_hermite, potential_hermite, nullptr, active);
  return cudaPeekAtLastError();
}

cudaError_t project_direct_md_coulomb(cudaStream_t stream, DeviceBatch batch,
                                      double* potential_hermite, double* coulomb,
                                      const std::uint8_t* active) {
  if (batch.total_shell_pairs == 0) return cudaSuccess;
  direct_md_pair_transform_kernel<false><<<batch.total_shell_pairs, 32, 0, stream>>>(
      batch, nullptr, nullptr, potential_hermite, coulomb, active);
  return cudaPeekAtLastError();
}

cudaError_t enqueue_direct_md_coulomb_class(unsigned shell_class, unsigned workers,
                                            cudaStream_t stream,
                                            const GeneratedShellPairStream* topology,
                                            DeviceBatch batch, const double* density_hermite,
                                            double* potential_hermite, double screening,
                                            std::uint32_t* head, unsigned long long* work_count) {
  return launch_direct_md_j_class(shell_class, workers, stream, topology,
                                  batch.shell_primitive_pairs, batch.shell_pair_primitive_offsets,
                                  density_hermite, potential_hermite, screening, head, work_count);
}

}  // namespace generativeqc::scf::cuda_execution
