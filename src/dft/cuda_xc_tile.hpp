#pragma once

#include <cstddef>

namespace generativeqc::dft {
/** Immutable offsets into one reusable batch arena. AO labels borrow the
 * original CSR map and every tile retains its own channel-major strides. */
struct CudaXcCompactTile {
  std::size_t begin{}, count{}, active{}, map_offset{}, ao_offset{}, work_offset{},
      potential_offset{};
};

}  // namespace generativeqc::dft
