#include "generated_one_electron_derivatives.cuh"
#include "generated_one_electron_family_metadata.cuh"
#include "scf/cuda/one_electron_gradient_launch.cuh"

namespace generativeqc::scf {

cudaError_t launch_generated_one_electron_gradient(
    const OneElectronDeviceView& batch, const std::int32_t* pair_first,
    const std::int32_t* pair_second, std::size_t pair_count, const OneElectronWeightView& weights,
    const std::uint8_t* active, unsigned schedule, double output_sign, double* gradient,
    cudaStream_t stream) {
  if (batch.maximum_angular_momentum <=
      generated_one_electron_families::through_f_maximum_angular_momentum)
    return launch_through_f_one_electron_gradient(batch, pair_first, pair_second, pair_count,
                                                  weights, active, schedule, output_sign, gradient,
                                                  stream);
  return launch_gradient_family<generated_one_electron_derivatives::Program>(
      batch, pair_first, pair_second, pair_count, weights, active, schedule, output_sign, gradient,
      stream);
}

}  // namespace generativeqc::scf
