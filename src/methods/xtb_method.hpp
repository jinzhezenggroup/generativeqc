#ifndef GENERATIVEQC_METHODS_XTB_METHOD_HPP
#define GENERATIVEQC_METHODS_XTB_METHOD_HPP

#include <memory>
#include <string>

#include "core/types.hpp"
#include "methods/method.hpp"

namespace generativeqc::methods::detail {

generativeqc_status validate_xtb_system(generativeqc_method method, const core::System& system,
                                        std::string& detail);

std::unique_ptr<PreparedCalculation> prepare_xtb_calculation(
    const Capabilities& capabilities, core::ContextState& context, const core::System& system,
    const generativeqc_method_descriptor& descriptor);

}  // namespace generativeqc::methods::detail

#endif
