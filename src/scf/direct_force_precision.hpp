#pragma once

#include <array>
#include <cstdint>

namespace generativeqc::scf {

/** Internal per-invocation arithmetic request, not an admission decision.
 * Method policy admits public budgets; private qualification callers own an
 * independent FP64/oracle comparison. Neither changes the scientific model.
 * SCF, screening, DF and higher-order force policy are unchanged. Optional
 * intrusive work collection fences the producer and must not supply clean
 * timing. Counts are attempted mixed tasks, primitive/source contractions and
 * nonfinite tasks retried once in FP64; zero work must not be called FP32. */
struct DirectForcePrecisionRequest {
  bool mixed_weighted{};
  bool collect_work{};
  std::array<std::uint64_t, 3> work{};
  bool mixed_geometry{};
  std::uint64_t geometry_points{};
  bool mixed_grid{};
  std::uint64_t grid_points{};
  std::uint64_t geometry_retries{};
};

}  // namespace generativeqc::scf
