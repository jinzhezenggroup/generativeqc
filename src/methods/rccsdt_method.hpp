#pragma once

#include "methods/method.hpp"

namespace generativeqc::methods::detail {
generativeqc_status validate_rccsdt_system(generativeqc_method, const core::System&, std::string&);
std::unique_ptr<PreparedCalculation> prepare_rccsdt_calculation(
    const Capabilities&, core::ContextState&, const core::System&,
    const generativeqc_method_descriptor&);
std::unique_ptr<PreparedBatch> prepare_rccsdt_batch(const Capabilities&, core::ContextState&,
                                                    std::vector<core::System>,
                                                    const generativeqc_method_descriptor&,
                                                    generativeqc_batch_flags);
}  // namespace generativeqc::methods::detail
