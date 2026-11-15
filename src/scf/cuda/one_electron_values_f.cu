#include "generated_one_electron_value_family.cuh"
#include "scf/cuda/one_electron_value_launch.cuh"

namespace generativeqc::scf {

cudaError_t launch_through_f_one_electron_values(
    const OneElectronDeviceView& batch, const std::int32_t* pair_first,
    const std::int32_t* pair_second, std::size_t pair_count, unsigned schedule, double* overlap,
    double* hcore, cudaStream_t stream, double* kinetic, double* attraction) {
  return launch_value_family<generated_one_electron_through_f::ValueFamily>(
      batch, pair_first, pair_second, pair_count, schedule, overlap, hcore, stream, kinetic,
      attraction);
}

}  // namespace generativeqc::scf
