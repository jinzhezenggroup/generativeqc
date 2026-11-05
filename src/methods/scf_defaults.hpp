#pragma once

#include "generativeqc/scf_defaults.h"
#include "scf/default_controls.hpp"

namespace generativeqc::methods::scf_defaults {

/** Defaults never grant a method, provider or physical-final-state capability. */
inline generativeqc_scf_defaults_v1 resolve(generativeqc_method method) noexcept {
  const auto controls = scf::defaults::for_method(method);
  return {sizeof(generativeqc_scf_defaults_v1),
          GENERATIVEQC_ABI_VERSION,
          scf::defaults::policy_version,
          method != GENERATIVEQC_METHOD_GFN2_XTB,
          controls.energy,
          controls.density,
          scf::defaults::screening_tolerance};
}

}  // namespace generativeqc::methods::scf_defaults
