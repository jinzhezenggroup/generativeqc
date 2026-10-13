#include "methods/dft_method_owner_port.hpp"

namespace generativeqc::methods::detail {

// Derivative bridge: the native owner checks token freshness and resource leases.
// Keep fallback statuses/output invalidation identical to the original API.

generativeqc_status dft_prepared_integral_gradient_cached(
    PreparedCalculation& calculation, const dft::CudaKsFinalStateToken& expected,
    const std::vector<scf::reference::Matrix>& density,
    const std::vector<scf::reference::Matrix>& weighted_density, std::vector<double>& output,
    std::size_t maximum_bytes, std::array<std::uint64_t, 9>& work, std::string& detail) {
  auto* ks = dynamic_cast<KsSingleOwnerPort*>(&calculation);
  if (ks)
    return ks->prepared_integral_gradient(expected, density, weighted_density, output,
                                          maximum_bytes, work, detail);
  output.clear();
  work = {};
  detail = "stationary integral sources require a native prepared KS calculation";
  return GENERATIVEQC_STATUS_NOT_IMPLEMENTED;
}

generativeqc_status dft_cuda_full_range_integral_derivatives(
    PreparedBatch& batch, std::size_t index, const dft::CudaKsFinalStateToken& expected,
    std::vector<double>& output, std::string& detail) {
  auto* ks = dynamic_cast<KsBatchOwnerPort*>(&batch);
  if (ks) return ks->cuda_full_range_integral_derivatives(index, expected, output, detail);
  output.clear();
  detail = "CUDA full-range stationary shell derivatives require a native KS batch";
  return GENERATIVEQC_STATUS_NOT_IMPLEMENTED;
}

generativeqc_status dft_cuda_integral_gradient_additional_device_bytes(
    PreparedBatch& batch, std::size_t index, const dft::CudaKsFinalStateToken& expected,
    std::size_t& bytes, std::string& detail) {
  bytes = 0;
  auto* ks = dynamic_cast<KsBatchOwnerPort*>(&batch);
  if (ks) return ks->cuda_integral_gradient_additional_device_bytes(index, expected, bytes, detail);
  detail = "CUDA integral derivative capacity requires a native KS batch";
  return GENERATIVEQC_STATUS_NOT_IMPLEMENTED;
}

generativeqc_status dft_cuda_integral_gradient(PreparedBatch& batch, std::size_t index,
                                               const dft::CudaKsFinalStateToken& expected,
                                               std::vector<double>& output,
                                               std::size_t maximum_bytes,
                                               std::array<std::uint64_t, 9>& work,
                                               std::string& detail) {
  auto* ks = dynamic_cast<KsBatchOwnerPort*>(&batch);
  if (ks) return ks->cuda_integral_gradient(index, expected, output, maximum_bytes, work, detail);
  detail = "CUDA integral gradient requires a native KS batch";
  return GENERATIVEQC_STATUS_NOT_IMPLEMENTED;
}

generativeqc_status dft_density_fitted_integral_gradient_cached(
    PreparedBatch& batch, std::size_t index, const dft::CudaKsFinalStateToken& expected,
    const std::vector<scf::reference::Matrix>& density,
    const std::vector<scf::reference::Matrix>& weighted_density, std::vector<double>& output,
    std::size_t maximum_bytes, std::array<std::uint64_t, 9>& work, std::string& detail) {
  auto* ks = dynamic_cast<KsBatchOwnerPort*>(&batch);
  if (ks)
    return ks->density_fitted_integral_gradient(index, expected, density, weighted_density, output,
                                                maximum_bytes, work, detail);
  output.clear();
  detail = "density-fitted integral gradient requires a native KS batch";
  return GENERATIVEQC_STATUS_NOT_IMPLEMENTED;
}

generativeqc_status dft_mixed_density_fitted_integral_gradient_cached(
    PreparedBatch& batch, std::size_t index, const dft::CudaKsFinalStateToken& expected,
    const std::vector<scf::reference::Matrix>& density,
    const std::vector<scf::reference::Matrix>& weighted_density,
    std::span<const double> lr_correction, std::vector<double>& output, std::size_t maximum_bytes,
    std::array<std::uint64_t, 9>& work, std::string& detail) {
  auto* ks = dynamic_cast<KsBatchOwnerPort*>(&batch);
  if (ks)
    return ks->mixed_density_fitted_integral_gradient(index, expected, density, weighted_density,
                                                      lr_correction, output, maximum_bytes, work,
                                                      detail);
  detail = "mixed density-fitted integral gradient requires a native KS batch";
  return GENERATIVEQC_STATUS_NOT_IMPLEMENTED;
}

generativeqc_status dft_cuda_integral_gradient_cached(
    PreparedBatch& batch, std::size_t index, const dft::CudaKsFinalStateToken& expected,
    const std::vector<scf::reference::Matrix>& density,
    const std::vector<scf::reference::Matrix>& weighted_density, std::vector<double>& output,
    std::size_t maximum_bytes, std::array<std::uint64_t, 9>& work, std::string& detail,
    bool combined_two_electron) {
  auto* ks = dynamic_cast<KsBatchOwnerPort*>(&batch);
  if (ks)
    return ks->cuda_integral_gradient(index, expected, output, maximum_bytes, work, detail,
                                      &density, &weighted_density, combined_two_electron);
  detail = "CUDA integral gradient requires a native KS batch";
  return GENERATIVEQC_STATUS_NOT_IMPLEMENTED;
}

generativeqc_status dft_cuda_fixed_density_profile(PreparedBatch& batch, std::size_t index,
                                                   const dft::CudaKsFinalStateToken& expected,
                                                   KsFixedDensityProfile& profile,
                                                   std::string& detail) {
  profile = {};
  auto* ks = dynamic_cast<KsBatchOwnerPort*>(&batch);
  if (ks) return ks->fixed_density_profile(index, expected, profile, detail);
  detail = "fixed-density component profiling requires a native KS batch";
  return GENERATIVEQC_STATUS_NOT_IMPLEMENTED;
}

}  // namespace generativeqc::methods::detail
