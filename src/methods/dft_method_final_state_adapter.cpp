#include "methods/dft_method_owner_port.hpp"

namespace generativeqc::methods::detail {

// Final-state bridge: the native owner checks token freshness and resource leases.
// Keep fallback statuses/output invalidation identical to the original API.

generativeqc_status dft_final_state_token(const PreparedCalculation& calculation,
                                          dft::CudaKsFinalStateToken& token, std::string& detail) {
  const auto* ks = dynamic_cast<const KsSingleOwnerPort*>(&calculation);
  if (ks) return ks->final_state_token(token, detail);
  token = {};
  detail = "prepared calculation is not a KS final-state owner";
  return GENERATIVEQC_STATUS_INVALID_ARGUMENT;
}

generativeqc_status read_dft_final_state(PreparedCalculation& calculation,
                                         const dft::CudaKsFinalStateToken& expected,
                                         bool compute_weighted_density,
                                         dft::VerifiedKsFinalState& state, std::string& detail) {
  auto* ks = dynamic_cast<KsSingleOwnerPort*>(&calculation);
  if (ks) return ks->read_final_state(expected, compute_weighted_density, state, detail);
  state = {};
  detail = "prepared calculation is not a KS final-state owner";
  return GENERATIVEQC_STATUS_INVALID_ARGUMENT;
}

generativeqc_status dft_final_state_token(const PreparedBatch& batch, std::size_t index,
                                          dft::CudaKsFinalStateToken& token, std::string& detail) {
  const auto* ks = dynamic_cast<const KsBatchOwnerPort*>(&batch);
  if (ks) return ks->final_state_token(index, token, detail);
  token = {};
  detail = "prepared batch is not a KS final-state owner";
  return GENERATIVEQC_STATUS_INVALID_ARGUMENT;
}

generativeqc_status read_dft_final_state(PreparedBatch& batch, std::size_t index,
                                         const dft::CudaKsFinalStateToken& expected,
                                         bool compute_weighted_density,
                                         dft::VerifiedKsFinalState& state, std::string& detail) {
  auto* ks = dynamic_cast<KsBatchOwnerPort*>(&batch);
  if (ks) return ks->read_final_state(index, expected, compute_weighted_density, state, detail);
  state = {};
  detail = "prepared batch is not a KS final-state owner";
  return GENERATIVEQC_STATUS_INVALID_ARGUMENT;
}

generativeqc_status dft_cuda_resident_orbital_frame(PreparedBatch& batch, std::size_t index,
                                                    const dft::CudaKsFinalStateToken& expected,
                                                    dft::CudaKsResidentOrbitalFrameBinding& binding,
                                                    std::string& detail, bool prepare) {
  if (auto* ks = dynamic_cast<KsBatchOwnerPort*>(&batch))
    return ks->resident_orbital_frame(index, expected, binding, detail, prepare);
  binding = {};
  detail = "resident orbital frame requires a native KS batch";
  return GENERATIVEQC_STATUS_NOT_IMPLEMENTED;
}

generativeqc_status dft_cuda_resident_fock_sources(PreparedBatch& batch, std::size_t index,
                                                   const dft::CudaKsFinalStateToken& expected,
                                                   dft::CudaKsResidentFockBinding& binding,
                                                   std::string& detail) {
  if (auto* ks = dynamic_cast<KsBatchOwnerPort*>(&batch))
    return ks->resident_fock_sources(index, expected, binding, detail);
  binding = {};
  detail = "resident Fock sources require a native KS batch";
  return GENERATIVEQC_STATUS_NOT_IMPLEMENTED;
}

generativeqc_status dft_cuda_resident_grid_reference(PreparedBatch& batch, std::size_t index,
                                                     const dft::CudaKsFinalStateToken& expected,
                                                     dft::CudaKsResidentGridBinding& binding,
                                                     std::string& detail) {
  if (auto* ks = dynamic_cast<KsBatchOwnerPort*>(&batch))
    return ks->resident_grid_reference(index, expected, binding, detail);
  binding = {};
  detail = "resident basis/grid requires a native KS batch";
  return GENERATIVEQC_STATUS_NOT_IMPLEMENTED;
}

generativeqc_status dft_cuda_resident_density(PreparedBatch& batch, std::size_t index,
                                              const dft::CudaKsFinalStateToken& expected,
                                              int& device, const double*& alpha,
                                              const double*& beta, std::size_t& matrix_elements,
                                              unsigned& spins, void*& source_stream,
                                              std::string& detail) {
  auto* ks = dynamic_cast<KsBatchOwnerPort*>(&batch);
  if (ks)
    return ks->resident_density(index, expected, device, alpha, beta, matrix_elements, spins,
                                source_stream, detail);
  device = -1;
  alpha = nullptr;
  beta = nullptr;
  matrix_elements = 0;
  spins = 0;
  source_stream = nullptr;
  detail = "resident final density requires a native KS batch";
  return GENERATIVEQC_STATUS_NOT_IMPLEMENTED;
}

generativeqc_status dft_cuda_resident_grid(PreparedBatch& batch, std::size_t index,
                                           const dft::CudaKsFinalStateToken& expected, int& device,
                                           const double*& points, const double*& weights,
                                           const double*& atomic_weights, std::size_t& point_count,
                                           std::string& detail) {
  auto* ks = dynamic_cast<KsBatchOwnerPort*>(&batch);
  if (ks)
    return ks->resident_grid(index, expected, device, points, weights, atomic_weights, point_count,
                             detail);
  device = -1;
  points = nullptr;
  weights = nullptr;
  atomic_weights = nullptr;
  point_count = 0;
  detail = "resident molecular grid requires a native KS batch";
  return GENERATIVEQC_STATUS_NOT_IMPLEMENTED;
}

generativeqc_status dft_cuda_resident_nonlocal_features(PreparedBatch& batch, std::size_t index,
                                                        const dft::CudaKsFinalStateToken& expected,
                                                        int& device, const double*& density,
                                                        const double*& gradient,
                                                        std::size_t& point_count,
                                                        void*& source_stream, std::string& detail) {
  auto* ks = dynamic_cast<KsBatchOwnerPort*>(&batch);
  if (ks)
    return ks->resident_nonlocal_features(index, expected, device, density, gradient, point_count,
                                          source_stream, detail);
  device = -1;
  density = nullptr;
  gradient = nullptr;
  point_count = 0;
  source_stream = nullptr;
  detail = "resident nonlocal features require a native KS batch";
  return GENERATIVEQC_STATUS_NOT_IMPLEMENTED;
}

generativeqc_status read_dft_derivative_state(PreparedBatch& batch, std::size_t index,
                                              const dft::CudaKsFinalStateToken& expected,
                                              KsDerivativeSnapshot& output, std::string& detail) {
  auto* ks = dynamic_cast<KsBatchOwnerPort*>(&batch);
  if (ks) return ks->read_derivative_state(index, expected, output, detail);
  output = {};
  detail = "prepared batch is not a KS final-state owner";
  return GENERATIVEQC_STATUS_INVALID_ARGUMENT;
}

generativeqc_status read_dft_derivative_sources(PreparedBatch& batch, std::size_t index,
                                                const dft::CudaKsFinalStateToken& expected,
                                                KsDerivativeSnapshot& output, std::string& detail) {
  auto* ks = dynamic_cast<KsBatchOwnerPort*>(&batch);
  if (ks) return ks->read_derivative_sources(index, expected, output, detail);
  output = {};
  detail = "prepared batch is not a KS final-state owner";
  return GENERATIVEQC_STATUS_INVALID_ARGUMENT;
}

}  // namespace generativeqc::methods::detail
