#pragma once

#include <algorithm>
#include <string_view>

#include "dft/grid.hpp"
#include "generativeqc/scf_defaults.h"
#include "methods/named_ks_catalog.hpp"

namespace generativeqc::methods::scf_defaults {

/** Reuse the native grid owner; do not copy its radii or build a second grid. */
inline bool standard_grid(const generativeqc_ks_options& options) noexcept {
  const auto expected = generativeqc::dft::default_grid_spec();
  return options.xc_execution_schedule == GENERATIVEQC_XC_EXECUTION_DEVICE_FUSED &&
         options.grid_version == expected.version &&
         options.radial_points == expected.radial_points &&
         options.angular_polar == expected.angular_polar &&
         options.angular_azimuth == expected.angular_azimuth &&
         options.partition_iterations == expected.partition_iterations &&
         options.coincident_tolerance == expected.coincident_tolerance &&
         options.element_radius_count == expected.element_radii.size() &&
         std::equal(expected.element_radii.begin(), expected.element_radii.end(),
                    options.element_radii);
}

inline bool standard_model(std::string_view identity) noexcept {
  // These are the incumbent independently qualified complete electronic models.
  // Resolve their current identities from the compiler-owned native catalog;
  // never pin historical hashes or admit arbitrary mixtures by display name.
  for (auto name : {"pbe-rks", "pbe0-rks", "b3lyp-rks", "r2scan-rks", "m06-2x-rks"}) {
    const auto* entry = generativeqc::methods::named_ks::find_method(name);
    if (entry && entry->identity == identity) return true;
  }
  return false;
}

/** Preserve qualified method defaults without changing explicit solver controls. */
inline generativeqc_scf_defaults_v1 resolve(const generativeqc_scf_defaults_context_v1& context,
                                            const generativeqc_ks_options* ks_options,
                                            std::string_view identity) noexcept {
  const bool qualified = context.backend == GENERATIVEQC_BACKEND_CUDA && !context.has_ecp &&
                         (ks_options ? standard_grid(*ks_options) && standard_model(identity)
                                     : context.method == GENERATIVEQC_METHOD_RHF);
  return {sizeof(generativeqc_scf_defaults_v1),
          GENERATIVEQC_ABI_VERSION,
          1,
          qualified,
          qualified ? 1e-9 : 1e-10,
          qualified ? 1e-5 : 1e-8,
          1e-12};
}

}  // namespace generativeqc::methods::scf_defaults
