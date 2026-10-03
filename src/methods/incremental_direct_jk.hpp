#pragma once

#include <charconv>
#include <cstdlib>
#include <cstring>
#include <optional>

#include "methods/method.hpp"

namespace generativeqc::methods::detail {

/** Shared HF/KS experiment selector, not a new public method ABI or default.
 * Resource queries and prepared owners must observe the same process setting. */
inline bool incremental_direct_jk_benchmark_requested() {
  const char* value = std::getenv("GENERATIVEQC_INCREMENTAL_DIRECT_JK");
  if (value == nullptr || std::strcmp(value, "0") == 0 || std::strcmp(value, "off") == 0)
    return false;
  if (std::strcmp(value, "1") == 0 || std::strcmp(value, "on") == 0) return true;
  throw MethodError(GENERATIVEQC_STATUS_INVALID_ARGUMENT,
                    "GENERATIVEQC_INCREMENTAL_DIRECT_JK must be 0/off or 1/on");
}

/** Parse accepted-update intervals without weakening the existing HF policy. */
inline std::optional<unsigned> incremental_direct_jk_benchmark_rebuild_interval() {
  const char* value = std::getenv("GENERATIVEQC_INCREMENTAL_DIRECT_JK_REBUILD_INTERVAL");
  if (value == nullptr) return std::nullopt;
  unsigned parsed = 0U;
  const char* end = value + std::strlen(value);
  const auto result = std::from_chars(value, end, parsed);
  if (result.ec != std::errc{} || result.ptr != end)
    throw MethodError(
        GENERATIVEQC_STATUS_INVALID_ARGUMENT,
        "GENERATIVEQC_INCREMENTAL_DIRECT_JK_REBUILD_INTERVAL must be an unsigned integer");
  return parsed;
}

}  // namespace generativeqc::methods::detail
