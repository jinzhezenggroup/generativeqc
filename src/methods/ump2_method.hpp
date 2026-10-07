#pragma once

#include "methods/method.hpp"

namespace generativeqc::methods::detail {
generativeqc_status validate_ump2_system(generativeqc_method, const core::System&, std::string&);
std::unique_ptr<PreparedCalculation> prepare_ump2_calculation(
    const Capabilities&, core::ContextState&, const core::System&,
    const generativeqc_method_descriptor&);
std::unique_ptr<PreparedBatch> prepare_ump2_batch(const Capabilities&, core::ContextState&,
                                                  std::vector<core::System>,
                                                  const generativeqc_method_descriptor&,
                                                  generativeqc_batch_flags);
}  // namespace generativeqc::methods::detail
