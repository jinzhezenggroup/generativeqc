#pragma once

#include "integrals/ecp.hpp"

namespace generativeqc::testing {
using integrals::EcpData;
EcpData ecp_integrals(const core::System& system, unsigned radial = 160, unsigned angular = 32,
                      bool derivatives = true);
EcpData checked_ecp_integrals(const core::System& system, bool derivatives);
}  // namespace generativeqc::testing
