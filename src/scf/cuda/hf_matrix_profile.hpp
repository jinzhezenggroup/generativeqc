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

/** A bounded FP64 product calibration plus retained evidence provenance.
 * The complete admitted record participates in topology identity: editing
 * rates or provenance invalidates captured recipes even if the winner agrees.
 * No persistent-ERI or angular-queue policy is inferred from product rates.
 */
struct HfMatrixProfile {
  HfMatrixProfileContext context;
  std::array<char, 65> evidence_sha256{};
  std::size_t minimum_dimension{}, maximum_dimension{};
  std::size_t minimum_matrices{}, maximum_matrices{};
  tensor::CudaMatrixProductCalibration calibration;
  double provider_setup_nanoseconds{};

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
