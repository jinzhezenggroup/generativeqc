#pragma once

#include <cstddef>
#include <cstring>
#include <stdexcept>

namespace generativeqc::dft {

/** Only a profitable restricted factor is proposed. Provider capacity,
 * native-cold provenance and the full FP64 density proof remain mandatory.
 * Small explicit experiments retain the same rank/work bound. */
inline bool cold_fitted_factor_requested(const char* setting, std::size_t aos, std::size_t rank) {
  if (setting && std::strcmp(setting, "0") != 0 && std::strcmp(setting, "1") != 0)
    throw std::invalid_argument("GENERATIVEQC_CUDA_KS_COLD_FACTOR requires 0 or 1");
  return aos >= 2 && rank > 0 && rank <= aos / 2 &&
         (setting ? std::strcmp(setting, "1") == 0 : aos >= 384);
}

}  // namespace generativeqc::dft
