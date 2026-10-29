#pragma once

#include "generativeqc/generativeqc.h"

namespace generativeqc::api {
generativeqc_status close_provider_initialization_current(generativeqc_calculation*) noexcept;
generativeqc_status close_provider_initialization_context(generativeqc_context*) noexcept;
}  // namespace generativeqc::api
