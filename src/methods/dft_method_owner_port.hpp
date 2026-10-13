#pragma once

#include "methods/dft_method.hpp"

namespace generativeqc::methods::detail {

// Private cross-cast ports keep derivative/snapshot ABI bridges independent of
// concrete execution-owner layout. They neither own a plan nor publish a token.
class KsSingleOwnerPort {
 public:
  virtual ~KsSingleOwnerPort() = default;
  virtual generativeqc_status final_state_token(dft::CudaKsFinalStateToken& token,
                                                std::string& detail) const = 0;

  virtual generativeqc_status read_final_state(const dft::CudaKsFinalStateToken& expected,
                                               bool compute_weighted_density,
                                               dft::VerifiedKsFinalState& state,
                                               std::string& detail) = 0;

  virtual generativeqc_status prepared_integral_gradient(
      const dft::CudaKsFinalStateToken& expected,
      const std::vector<scf::reference::Matrix>& density,
      const std::vector<scf::reference::Matrix>& weighted_density, std::vector<double>& output,
      std::size_t maximum_bytes, std::array<std::uint64_t, 9>& work, std::string& detail) = 0;
};

class KsBatchOwnerPort {
 public:
  virtual ~KsBatchOwnerPort() = default;
  virtual generativeqc_status final_state_token(std::size_t index,
                                                dft::CudaKsFinalStateToken& token,
                                                std::string& detail) const = 0;

  virtual generativeqc_status read_final_state(std::size_t index,
                                               const dft::CudaKsFinalStateToken& expected,
                                               bool compute_weighted_density,
                                               dft::VerifiedKsFinalState& state,
                                               std::string& detail) = 0;

  virtual generativeqc_status cuda_full_range_integral_derivatives(
      std::size_t index, const dft::CudaKsFinalStateToken& expected, std::vector<double>& output,
      std::string& detail) = 0;

  virtual generativeqc_status cuda_prepare_force_memory(std::size_t index,
                                                        const dft::CudaKsFinalStateToken& expected,
                                                        std::size_t additional_device_bytes,
                                                        std::array<std::uint64_t, 4>& receipt,
                                                        std::string& detail) = 0;

  virtual generativeqc_status cuda_integral_gradient_additional_device_bytes(
      std::size_t index, const dft::CudaKsFinalStateToken& expected, std::size_t& bytes,
      std::string& detail) = 0;

  virtual generativeqc_status cuda_integral_gradient(
      std::size_t index, const dft::CudaKsFinalStateToken& expected, std::vector<double>& output,
      std::size_t maximum_bytes, std::array<std::uint64_t, 9>& work, std::string& detail,
      const std::vector<scf::reference::Matrix>* cached_density = nullptr,
      const std::vector<scf::reference::Matrix>* cached_weighted_density = nullptr,
      bool combined_two_electron = false) = 0;

  virtual generativeqc_status density_fitted_integral_gradient(
      std::size_t index, const dft::CudaKsFinalStateToken& expected,
      const std::vector<scf::reference::Matrix>& density,
      const std::vector<scf::reference::Matrix>& weighted_density, std::vector<double>& output,
      std::size_t maximum_bytes, std::array<std::uint64_t, 9>& work, std::string& detail) = 0;

  virtual generativeqc_status mixed_density_fitted_integral_gradient(
      std::size_t index, const dft::CudaKsFinalStateToken& expected,
      const std::vector<scf::reference::Matrix>& density,
      const std::vector<scf::reference::Matrix>& weighted_density,
      std::span<const double> lr_correction, std::vector<double>& output, std::size_t maximum_bytes,
      std::array<std::uint64_t, 9>& work, std::string& detail) = 0;

  // Native stationary consumers remain owned by the prepared KS batch. These
  // ports forward the exact token and per-item bound without retaining storage.
  virtual generativeqc_status native_cuda_stationary_gradient(
      std::size_t index, const dft::CudaKsFinalStateToken& expected,
      std::size_t maximum_device_bytes, std::vector<double>& output,
      dft::CudaStationaryResources& resources, std::string& detail) = 0;

  virtual generativeqc_status native_cuda_stationary_capacity(std::size_t index,
                                                              dft::CudaStationaryCapacity& capacity,
                                                              std::string& detail) const = 0;

  virtual generativeqc_status native_cuda_stationary_limit(std::size_t index,
                                                           std::size_t maximum_device_bytes,
                                                           std::string& detail) = 0;

  virtual generativeqc_status resident_density(std::size_t index,
                                               const dft::CudaKsFinalStateToken& expected,
                                               int& device, const double*& alpha,
                                               const double*& beta, std::size_t& matrix_elements,
                                               unsigned& spins, void*& source_stream,
                                               std::string& detail) = 0;

  virtual generativeqc_status fixed_density_profile(std::size_t index,
                                                    const dft::CudaKsFinalStateToken& expected,
                                                    KsFixedDensityProfile& profile,
                                                    std::string& detail) = 0;

  virtual generativeqc_status resident_grid(std::size_t index,
                                            const dft::CudaKsFinalStateToken& expected, int& device,
                                            const double*& points, const double*& weights,
                                            const double*& atomic_weights, std::size_t& point_count,
                                            std::string& detail) = 0;

  virtual generativeqc_status resident_nonlocal_features(
      std::size_t index, const dft::CudaKsFinalStateToken& expected, int& device,
      const double*& density, const double*& gradient, std::size_t& point_count,
      void*& source_stream, std::string& detail) = 0;

  virtual generativeqc_status read_derivative_state(std::size_t index,
                                                    const dft::CudaKsFinalStateToken& expected,
                                                    KsDerivativeSnapshot& output,
                                                    std::string& detail) = 0;

  virtual generativeqc_status read_derivative_sources(std::size_t index,
                                                      const dft::CudaKsFinalStateToken& expected,
                                                      KsDerivativeSnapshot& output,
                                                      std::string& detail) = 0;
};

}  // namespace generativeqc::methods::detail
