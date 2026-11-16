#pragma once

#include <array>
#include <cstddef>
#include <cstdlib>
#include <optional>
#include <string_view>

#include "runtime/cuda_provider.hpp"
#include "runtime/cuda_target_info.hpp"
#include "tensor/cuda_matrix_product_admission.hpp"

namespace generativeqc::scf::cuda_policy {

/** Exact execution-stack identity, not a product-name profitability selector.
 * Resource facts constrain compatibility, while only the measured rates rank
 * candidates. Unknown facts cannot qualify an opt-in calibration.
 */
struct HfMatrixProfileContext {
  runtime::CudaTargetInfo target;
  runtime::CudaProviderKind provider{runtime::CudaProviderKind::None};
  std::array<char, 65> source_identity{};
  std::array<char, 129> native_build_id{}, runtime_build_id{}, driver_build_id{};
  std::array<char, 129> cublas_build_id{};
  int driver_version{}, runtime_version{}, toolkit_version{};
  std::array<int, 3> cublas_version{};
  bool release_build{}, fast_compile{};
  bool operator==(const HfMatrixProfileContext&) const = default;
};

/** Explicit endpoint-qualified topology choices, never inferred from GEMM rates.
 * The AO crossover may disable residency (zero) or replace the incumbent only
 * inside the record's qualified workload domain. Runtime memory admission and
 * the compiler's maximum queue domains remain independent safety bounds.
 */
struct DirectJkTopologyProfile {
  std::size_t persistent_eri_ao_limit{};
  std::size_t maximum_persistent_eri_bytes{};
  unsigned force_angular_order_count{}, fock_angular_order_count{};
  bool operator==(const DirectJkTopologyProfile&) const = default;
};

/** A bounded FP64 product calibration plus retained endpoint evidence provenance.
 * The complete admitted record participates in topology identity: editing
 * rates or provenance invalidates captured recipes even if the winner agrees.
 * Topology selections have their own explicit evidence-bound fields; product
 * calibration alone cannot promote resident ERIs or a different queue schedule.
 */
struct HfMatrixProfile {
  HfMatrixProfileContext context;
  std::array<char, 65> evidence_sha256{};
  std::size_t minimum_dimension{}, maximum_dimension{};
  std::size_t minimum_matrices{}, maximum_matrices{};
  tensor::CudaMatrixProductCalibration calibration;
  double provider_setup_nanoseconds{};
  DirectJkTopologyProfile direct_jk;

  bool covers(std::size_t dimension, std::size_t matrices) const noexcept {
    return dimension != 0 && matrices != 0 && dimension >= minimum_dimension &&
           dimension <= maximum_dimension && matrices >= minimum_matrices &&
           matrices <= maximum_matrices;
  }
  tensor::CudaMatrixProductProfile matrix_profile() const noexcept {
    tensor::CudaMatrixProductProfile profile;
    profile.calibration = calibration;
    profile.provider_setup_nanoseconds = provider_setup_nanoseconds;
    return profile;
  }
  bool operator==(const HfMatrixProfile&) const = default;
};

enum class HfMatrixProfileStatus { Absent, Unreadable, Invalid, Incompatible, Admitted };
struct HfMatrixProfileAdmission {
  HfMatrixProfileStatus status{HfMatrixProfileStatus::Absent};
  std::optional<HfMatrixProfile> profile;

  /** Borrow only an effective record for this actual target and spin workload.
   * Packing and retained capacity fallbacks must agree: an admitted file
   * outside the workload domain is still the incumbent, not a new selection.
   */
  const HfMatrixProfile* for_workload(const runtime::CudaTargetInfo& target, std::size_t dimension,
                                      std::size_t systems, std::size_t fock_states) const noexcept {
    return status == HfMatrixProfileStatus::Admitted && profile &&
                   profile->context.target == target && profile->covers(dimension, systems) &&
                   profile->covers(dimension, fock_states)
               ? &*profile
               : nullptr;
  }
};

/** Strict, size-bounded, device-inert parser. Schema/source/stack/domain/rate
 * rejection is all-or-nothing; no partial record or fallback override escapes.
 */
HfMatrixProfileAdmission admit_hf_matrix_profile(std::string_view payload,
                                                 const HfMatrixProfileContext& context);

/** Read an explicitly selected regular file on every preparation/replay.
 * Absence, failed IO and invalid/incompatible records retain the incumbent.
 * Resource-only preflights inspect only whether a path was requested, never
 * read it or query a device/provider to manufacture a calibration identity.
 */
HfMatrixProfileAdmission load_hf_matrix_profile(const char* path,
                                                const HfMatrixProfileContext& context);
inline bool hf_matrix_profile_requested() noexcept {
  const char* path = std::getenv("GENERATIVEQC_HF_MATRIX_PROFILE");
  return path && *path;
}

}  // namespace generativeqc::scf::cuda_policy
