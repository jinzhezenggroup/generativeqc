#ifndef GENERATIVEQC_METHODS_DFT_METHOD_HPP
#define GENERATIVEQC_METHODS_DFT_METHOD_HPP

#include <array>
#include <memory>
#include <span>
#include <string>

#include "dft/cuda_ks_final_state.hpp"
#include "dft/cuda_stationary_capacity.hpp"
#include "dft/cuda_stationary_controller.hpp"
#include "methods/method.hpp"

namespace generativeqc::methods::detail {

/** Context-qualified additional CUDA force capacity without any CUDA operation.
 * Shapes are atoms, AOs, primitives, grid points, maximum shell l, ECP terms
 * and total ECP core electrons. Model admission is shared with prepared owners;
 * this metadata alone neither supplies a final token nor evaluates science. */
generativeqc_status dft_cuda_stationary_capacity(const generativeqc_method_descriptor& descriptor,
                                                 std::span<const std::uint64_t> shape,
                                                 dft::CudaStationaryCapacity& capacity,
                                                 std::string& detail);

/** Read the same capacity from an authenticated prepared item, before SCF. */
generativeqc_status dft_cuda_stationary_capacity(PreparedBatch& batch, std::size_t index,
                                                 dft::CudaStationaryCapacity& capacity,
                                                 std::string& detail);

/** Bound one retained consumer, including future geometry rebuilds. Admission
 * remains scientific/context-specific; this only selects optional AO caching. */
generativeqc_status dft_cuda_stationary_limit(PreparedBatch& batch, std::size_t index,
                                              std::size_t maximum_device_bytes,
                                              std::string& detail);

/** Detached derivative inputs copied from the same owner as the #162 state.
 * The token remains authoritative: this copy alone never proves freshness. */
struct KsFixedDensityProfile {
  std::array<double, 4> milliseconds{};
  std::uint32_t present_mask{};
};

struct KsDerivativeSnapshot {
  dft::VerifiedKsFinalState state;
  core::System system;
  std::vector<double> overlap, packed_basis, points, weights;
  std::vector<std::uint32_t> grid_owners;
  std::vector<double> atomic_weights;
  std::uint64_t export_d2h_bytes{}, export_reads{}, export_synchronizations{};
};

generativeqc_status read_dft_derivative_state(PreparedBatch& batch, std::size_t index,
                                              const dft::CudaKsFinalStateToken& expected,
                                              KsDerivativeSnapshot& output, std::string& detail);

/** Private #2151 single-calculation stationary integral-source bridge.
 * Reuses the exact converged owner and the already qualified DF (CPU/CUDA)
 * or Direct (CUDA) derivative provider. The output is source-major
 * [H', overlap/Pulay, Coulomb J', exchange K'], four times 3*Natom.
 * These are +dE/dR integral terms, NOT a complete molecular gradient or force:
 * XC, Becke moving-grid and nuclear terms belong to a later consumer.
 * Unsupported Hamiltonians (ECP, range, nonlocal, D4) fail closed.
 * All supplied D/W blocks must belong to expected and are validated by the
 * caller's verified final-state export; live token is rechecked by the owner.
 * Failure clears output/work and never authorizes GENERATIVEQC_PROPERTY_FORCES.
 */
generativeqc_status dft_prepared_integral_gradient_cached(
    PreparedCalculation& calculation, const dft::CudaKsFinalStateToken& expected,
    const std::vector<scf::reference::Matrix>& density,
    const std::vector<scf::reference::Matrix>& weighted_density, std::vector<double>& output,
    std::size_t maximum_bytes, std::array<std::uint64_t, 9>& work, std::string& detail);

/** Dry capacity for lazy CUDA integral derivative owners under a live token.
 * Zero means the derivative owner was already charged to SCF. This excludes
 * DF response storage and does not allocate or grant force capability. */
generativeqc_status dft_cuda_integral_gradient_additional_device_bytes(
    PreparedBatch& batch, std::size_t index, const dft::CudaKsFinalStateToken& expected,
    std::size_t& bytes, std::string& detail);

/** CUDA stationary integral sources: hcore, overlap/Pulay, J and full-range K,
 * or separate SR-K/LR-K for range-separated models. Direct consumers validate
 * and stage resident D/W without exporting a detached frame; the retained DF
 * response contract still requires its explicit host snapshot. The live token
 * binds geometry, radial parameters and spin. XC, nonlocal correlation and
 * nuclear repulsion remain separate consumers, not a complete force. */
generativeqc_status dft_cuda_integral_gradient(PreparedBatch& batch, std::size_t index,
                                               const dft::CudaKsFinalStateToken& expected,
                                               std::vector<double>& output,
                                               std::size_t maximum_bytes,
                                               std::array<std::uint64_t, 9>& work,
                                               std::string& detail);

/** Private complete-source CUDA Direct qualification endpoint, not public
 * force admission. PBE/PBE0 RKS/UKS only. The additional device allowance
 * excludes the already charged SCF/Fock frame. Failure preserves output; the
 * exact final token gates both submission and terminal publication. */
generativeqc_status dft_cuda_native_stationary_gradient(PreparedBatch& batch, std::size_t index,
                                                        const dft::CudaKsFinalStateToken& expected,
                                                        std::size_t maximum_device_bytes,
                                                        std::vector<double>& output,
                                                        dft::CudaStationaryResources& resources,
                                                        std::string& detail);

/** Snapshot-backed variant for a consumer that already exported and validated
 * D/W under the exact same live token. The native owner still revalidates the
 * resident device density before two-electron derivative execution. */
generativeqc_status dft_cuda_integral_gradient_cached(
    PreparedBatch& batch, std::size_t index, const dft::CudaKsFinalStateToken& expected,
    const std::vector<scf::reference::Matrix>& density,
    const std::vector<scf::reference::Matrix>& weighted_density, std::vector<double>& output,
    std::size_t maximum_bytes, std::array<std::uint64_t, 9>& work, std::string& detail,
    bool combined_two_electron = false);

/** Token-checked density-fitted stationary integral sources for CPU or CUDA.
 * Output is source-major [hcore, overlap/Pulay, J_DF, K_DF], each block
 * containing 3*Natom values. D/W must be the already validated snapshot frame. */
generativeqc_status dft_density_fitted_integral_gradient_cached(
    PreparedBatch& batch, std::size_t index, const dft::CudaKsFinalStateToken& expected,
    const std::vector<scf::reference::Matrix>& density,
    const std::vector<scf::reference::Matrix>& weighted_density, std::vector<double>& output,
    std::size_t maximum_bytes, std::array<std::uint64_t, 9>& work, std::string& detail);

/** Mixed fitted-primary/exact-LR sources through the shared native composer.
 * CPU binds a compiler-generated compact weighted LR correction; CUDA owns
 * its resident derivative execution. Output is [H, Pulay, J_DF, K_SR, K_LR]. */
generativeqc_status dft_mixed_density_fitted_integral_gradient_cached(
    PreparedBatch& batch, std::size_t index, const dft::CudaKsFinalStateToken& expected,
    const std::vector<scf::reference::Matrix>& density,
    const std::vector<scf::reference::Matrix>& weighted_density,
    std::span<const double> lr_correction, std::vector<double>& output, std::size_t maximum_bytes,
    std::array<std::uint64_t, 9>& work, std::string& detail);

/** Token-checked full-range J'/K' through the prepared Direct shell owner.
 * Output is source-major [J,K], each block containing 3*Natom values. */
generativeqc_status dft_cuda_full_range_integral_derivatives(
    PreparedBatch& batch, std::size_t index, const dft::CudaKsFinalStateToken& expected,
    std::vector<double>& output, std::string& detail);

/** Borrow the accepted spin density directly from the exact CUDA KS
 * final-state owner. No transfer or synchronization is performed. source_stream
 * owns the accepted density generation; downstream device consumers must order
 * any cross-stream copy/read against it before the source can be reused. */
generativeqc_status dft_cuda_resident_density(PreparedBatch& batch, std::size_t index,
                                              const dft::CudaKsFinalStateToken& expected,
                                              int& device, const double*& alpha,
                                              const double*& beta, std::size_t& matrix_elements,
                                              unsigned& spins, void*& source_stream,
                                              std::string& detail);

/** Intrusive CUDA-event timings for J/K/XC replay at the exact resident final
 * density. No SCF step or new final-state generation is executed. */
generativeqc_status dft_cuda_fixed_density_profile(PreparedBatch& batch, std::size_t index,
                                                   const dft::CudaKsFinalStateToken& expected,
                                                   KsFixedDensityProfile& profile,
                                                   std::string& detail);

/** Borrow the final device-resident total rho/grad-rho for the exact KS
 * token. Pointers remain owned by the prepared CUDA KS plan and are valid only
 * while that owner and token remain current. source_stream identifies the CUDA
 * stream that owns the final feature generation so a downstream D2D handoff can
 * establish a device-side dependency before source reuse. This helper performs
 * no transfer or synchronization. */
generativeqc_status dft_cuda_resident_grid(PreparedBatch& batch, std::size_t index,
                                           const dft::CudaKsFinalStateToken& expected, int& device,
                                           const double*& points, const double*& weights,
                                           const double*& atomic_weights, std::size_t& point_count,
                                           std::string& detail);

generativeqc_status dft_cuda_resident_nonlocal_features(PreparedBatch& batch, std::size_t index,
                                                        const dft::CudaKsFinalStateToken& expected,
                                                        int& device, const double*& density,
                                                        const double*& gradient,
                                                        std::size_t& point_count,
                                                        void*& source_stream, std::string& detail);

generativeqc_status validate_dft_system(generativeqc_method method, const core::System& system,
                                        std::string& detail);

/** Convert one completed native KS solve into the method-neutral publication record.
 * Kept in detail scope so adapter field coverage can be tested without a backend execution. */
Result adapt_dft_result(scf::ScfResult native, generativeqc_backend backend);

/** Internal #163 handoff. These helpers accept prepared CPU RKS/CUDA KS owners;
 * they do not extend public result layouts or authorize force execution. */
generativeqc_status dft_final_state_token(const PreparedCalculation& calculation,
                                          dft::CudaKsFinalStateToken& token, std::string& detail);
generativeqc_status read_dft_final_state(PreparedCalculation& calculation,
                                         const dft::CudaKsFinalStateToken& expected,
                                         bool compute_weighted_density,
                                         dft::VerifiedKsFinalState& state, std::string& detail);
generativeqc_status dft_final_state_token(const PreparedBatch& batch, std::size_t index,
                                          dft::CudaKsFinalStateToken& token, std::string& detail);
generativeqc_status read_dft_final_state(PreparedBatch& batch, std::size_t index,
                                         const dft::CudaKsFinalStateToken& expected,
                                         bool compute_weighted_density,
                                         dft::VerifiedKsFinalState& state, std::string& detail);

std::unique_ptr<PreparedCalculation> prepare_dft_calculation(
    const Capabilities& capabilities, core::ContextState& context, const core::System& system,
    const generativeqc_method_descriptor& descriptor);

std::unique_ptr<PreparedBatch> prepare_dft_batch(const Capabilities& capabilities,
                                                 core::ContextState& context,
                                                 std::vector<core::System> systems,
                                                 const generativeqc_method_descriptor& descriptor,
                                                 generativeqc_batch_flags flags);

}  // namespace generativeqc::methods::detail

#endif
