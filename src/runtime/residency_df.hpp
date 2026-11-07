#pragma once

#include "residency_boundaries.hpp"

namespace generativeqc::runtime {

/** DF owns this extension of the private source catalogue. Keep the common
 * observer layout unchanged: older consumers fail closed on unfamiliar tags. */
inline constexpr auto df_metric_owner = static_cast<ResidencyOwner>(64);
inline constexpr auto df_metric_input_site = static_cast<ResidencySite>(64);
inline constexpr auto df_metric_eigenvalues_site = static_cast<ResidencySite>(65);
inline constexpr auto df_metric_status_site = static_cast<ResidencySite>(66);
inline constexpr auto df_metric_fence_site = static_cast<ResidencySite>(67);
inline constexpr auto df_metric_transform_site = static_cast<ResidencySite>(68);
inline constexpr auto df_metric_scales_site = static_cast<ResidencySite>(69);
inline constexpr auto df_metric_input_payload = static_cast<ResidencyPayload>(128);
inline constexpr auto df_metric_eigenvalues_payload = static_cast<ResidencyPayload>(129);
inline constexpr auto df_metric_status_payload = static_cast<ResidencyPayload>(130);
inline constexpr auto df_metric_scales_payload = static_cast<ResidencyPayload>(131);
inline constexpr auto residency_host_transform = static_cast<ResidencyOperationKind>(4);

inline const char* df_residency_boundary_name(std::uint64_t category,
                                              std::uint64_t value) noexcept {
  if (category == 1 && value == 64) return "df_metric_setup";
  if (category == 3) {
    switch (value) {
      case 64:
        return "df.metric.input";
      case 65:
        return "df.metric.eigenvalues";
      case 66:
        return "df.metric.solver_status";
      case 67:
        return "df.metric.host_ready";
      case 68:
        return "df.metric.inverse_sqrt_transform";
      case 69:
        return "df.metric.scales_upload";
    }
  }
  if (category == 4) {
    switch (value) {
      case 128:
        return "df.metric.raw";
      case 129:
        return "df.metric.eigenvalues";
      case 130:
        return "df.metric.solver_status";
      case 131:
        return "df.metric.inverse_sqrt_scales";
    }
  }
  return nullptr;
}

}  // namespace generativeqc::runtime
