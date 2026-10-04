#pragma once

#include "methods/method.hpp"

namespace generativeqc::methods::detail {

generativeqc_status validate_df_rccsdt_system(generativeqc_method, const core::System&,
                                              std::string&);
std::unique_ptr<PreparedCalculation> prepare_df_rccsdt_calculation(
    const Capabilities&, core::ContextState&, const core::System&,
    const generativeqc_method_descriptor&);

}  // namespace generativeqc::methods::detail
