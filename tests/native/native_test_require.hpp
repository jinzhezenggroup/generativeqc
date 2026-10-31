#pragma once

#include <source_location>
#include <stdexcept>
#include <string>

/** Always evaluate executable contract checks, including Release/NDEBUG builds.
 * Assertions must not erase the API calls whose effects a fixture validates. */
inline void require_test(bool condition,
                         std::source_location location = std::source_location::current()) {
  if (!condition)
    throw std::runtime_error(std::string("native contract check failed: ") + location.file_name() +
                             ":" + std::to_string(location.line()));
}
