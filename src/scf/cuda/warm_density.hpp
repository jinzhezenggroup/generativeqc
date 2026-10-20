#pragma once

#include "core/types.hpp"
#include "generativeqc/generativeqc.h"

namespace generativeqc::scf {
/** Validate an imported HF density with a temporary, bounded ordinary-stream
 * eigensolver. It owns no Fock source and never touches the fleet's live seeds.
 * The common physical admission gates and near-asymmetric fallback still apply.
 */
void validate_hf_warm_density_cuda(const core::System& source, generativeqc_method method,
                                   const std::vector<double>& density, int device);
}  // namespace generativeqc::scf
