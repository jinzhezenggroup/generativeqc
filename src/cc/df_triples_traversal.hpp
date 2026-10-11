#pragma once

#include <cstddef>

namespace generativeqc::cc::triples::detail {

/** Visit each triangular occupied tile once without materializing a tile list.
 * The serpentine schedule preserves the maximum occupied index while sharing
 * the lower-index row between adjacent tiles in the three-panel LRU. Return
 * each tile's original flat index so the final energy reduction stays ordered.
 * The caller must preflight occupied*(occupied+1)*(occupied+2) for overflow.
 * The ordinary schedule is retained for unqualified precision/storage domains.
 */
template <class Visitor>
void visit_occupied_tiles(std::size_t occupied, bool serpentine, Visitor&& visit) {
  std::size_t sequential_tile = 0;
  for (std::size_t first = 0; first < occupied; ++first) {
    if (serpentine) {
      const auto first_base = first * (first + 1) * (first + 2) / 6;
      for (std::size_t third_offset = 0; third_offset <= first; ++third_offset) {
        const auto third = first % 2 ? first - third_offset : third_offset;
        for (std::size_t second_offset = 0; second_offset <= first - third; ++second_offset) {
          const auto second = third % 2 ? first - second_offset : third + second_offset;
          visit(first, second, third, first_base + second * (second + 1) / 2 + third);
        }
      }
    } else {
      for (std::size_t second = 0; second <= first; ++second)
        for (std::size_t third = 0; third <= second; ++third)
          visit(first, second, third, sequential_tile++);
    }
  }
}

}  // namespace generativeqc::cc::triples::detail
