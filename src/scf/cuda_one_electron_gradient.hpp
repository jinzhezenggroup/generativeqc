#ifndef GENERATIVEQC_SCF_CUDA_ONE_ELECTRON_GRADIENT_HPP
#define GENERATIVEQC_SCF_CUDA_ONE_ELECTRON_GRADIENT_HPP

#include <cstddef>
#include <span>
#include <string>
#include <vector>

#include "core/types.hpp"

namespace generativeqc::scf {

struct CudaDirectJkPlan;

/** Explicit staging boundary of the standalone generic gradient operation. */
struct OneElectronGradientResources {
  std::size_t device_bytes{}, host_numeric_bytes{}, host_to_device_bytes{}, device_to_host_bytes{};
  std::size_t synchronous_uploads{}, stream_synchronizations{};
};

/** Contract arbitrary full public-AO S/T/V weights on CUDA with bounded storage.
 * Empty spans mean zero. Coefficients are already normalized by the system
 * validator; both Cartesian and real-spherical s/p/d/f AOs are supported.
 * Only metadata and weights cross H2D; only 3*Natom gradient scalars cross D2H.
 * Maximum_bytes bounds this operation's numeric host staging and device arena
 * separately, excluding caller-owned weights, system and existing HF plans.
 * Output is replaced only on success; it may alias a caller-owned weight span.
 * This synchronous host bridge is used by standalone DF and diagnostics;
 * prepared Direct HF invokes the same kernel on its existing owning stream.
 */
generativeqc_status execute_cuda_one_electron_gradient(
    int device_id, const core::System& system, std::span<const double> overlap_weights,
    std::span<const double> kinetic_weights, std::span<const double> attraction_weights,
    unsigned schedule, std::size_t maximum_bytes, std::vector<double>& gradient,
    std::string& detail, OneElectronGradientResources* resources = nullptr,
    double overlap_scale = 1.0);

/** Evaluate the stationary hcore and overlap/Pulay sources under one prepared
 * topology/metadata upload and one stream drain. The two scientific outputs
 * remain separate: hcore uses D for T/V, while Pulay uses -W for S.
 * Prepared native callers may pass a matched pair of resident device pointers
 * with empty host spans; those weights are borrowed and never copied H2D. */
generativeqc_status execute_cuda_stationary_one_electron_pair(
    int device_id, const core::System& system, std::span<const double> density,
    std::span<const double> weighted_density, unsigned schedule, std::size_t maximum_bytes,
    std::vector<double>& hcore_gradient, std::vector<double>& pulay_gradient, std::string& detail,
    OneElectronGradientResources* resources = nullptr, const double* resident_density = nullptr,
    const double* resident_weighted_density = nullptr);

/** Evaluate stationary hcore/Pulay by borrowing a derivative-capable Direct
 * owner's normalized device metadata, ordinary stream and force scratch.
 * The prepared path performs no allocation, packing or H2D transfer. It uses
 * the resident shell-pair schedule and copies only the two 3*Natom results to
 * host before one stream drain. The Direct owner must outlive this call. */
generativeqc_status execute_prepared_cuda_stationary_one_electron_pair(
    CudaDirectJkPlan* source, const double* resident_density,
    const double* resident_weighted_density, std::size_t matrix_elements, std::size_t maximum_bytes,
    std::vector<double>& hcore_gradient, std::vector<double>& pulay_gradient, std::string& detail,
    OneElectronGradientResources* resources = nullptr);

}  // namespace generativeqc::scf

#endif
