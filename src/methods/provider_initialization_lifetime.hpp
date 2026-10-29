#pragma once

#include <string>

#include "generativeqc/generativeqc.h"

namespace generativeqc::methods {
/** Native opt-in lifetime port independent of the ordinary method vtable.
 * The C ABI asks this before deleting a calculation or failed candidate. */
class ProviderInitializationCloseable {
 public:
  virtual ~ProviderInitializationCloseable() = default;
  virtual generativeqc_status close_provider_initialization_current(std::string& detail) = 0;
};
}  // namespace generativeqc::methods
