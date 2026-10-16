#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <vector>

namespace generativeqc::scf {

using MdTransformComponent = std::vector<std::array<std::uint8_t, 3>>;
using MdTransformShell = std::vector<MdTransformComponent>;

/** Geometry-independent support of the actual public AO expansions.
 * Slot zero retains weight * 0.0 for every primitive, including its IEEE sign
 * and nonfinite behavior. Remaining slots never alias: spherical cancellations
 * and geometry-dependent zeros must still be evaluated in the original order.
 */
struct MdTransformLayout {
  std::vector<std::uint16_t> slots;
  std::uint32_t values{1};
};

/** Enumerate the same x/y/z simplex as the device Hermite index ABI.
 * Support is a union over expansion terms, not coefficient-magnitude screening.
 * The optional through-d owner rejects larger angular domains before planning.
 */
inline MdTransformLayout md_transform_layout(const MdTransformShell& first,
                                             const MdTransformShell& second, unsigned angular) {
  if (angular > 4 || first.empty() || second.empty())
    throw std::invalid_argument("invalid MD transform layout");
  MdTransformLayout layout;
  for (const auto& first_component : first) {
    for (const auto& second_component : second) {
      for (unsigned derivative_x = 0; derivative_x <= angular; ++derivative_x) {
        for (unsigned derivative_y = 0; derivative_y <= angular - derivative_x; ++derivative_y) {
          for (unsigned derivative_z = 0; derivative_z <= angular - derivative_x - derivative_y;
               ++derivative_z) {
            bool supported = false;
            for (const auto& first_term : first_component)
              for (const auto& second_term : second_component)
                supported |= derivative_x <= first_term[0] + second_term[0] &&
                             derivative_y <= first_term[1] + second_term[1] &&
                             derivative_z <= first_term[2] + second_term[2];
            if (supported && layout.values > std::numeric_limits<std::uint16_t>::max())
              throw std::overflow_error("MD transform support exceeds slot index capacity");
            layout.slots.push_back(supported ? static_cast<std::uint16_t>(layout.values++) : 0);
          }
        }
      }
    }
  }
  return layout;
}

}  // namespace generativeqc::scf
