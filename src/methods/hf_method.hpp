#ifndef GENERATIVEQC_METHODS_HF_METHOD_HPP
#define GENERATIVEQC_METHODS_HF_METHOD_HPP

#include <memory>
#include <string>
#include <vector>

#include "methods/method.hpp"

namespace generativeqc::methods::detail {

generativeqc_status validate_hf_system(generativeqc_method method, const core::System& system,
                                       std::string& detail);

std::unique_ptr<PreparedCalculation> prepare_hf_calculation(
    const Capabilities& capabilities, core::ContextState& context, const core::System& system,
    const generativeqc_method_descriptor& descriptor);

std::unique_ptr<PreparedBatch> prepare_hf_batch(const Capabilities& capabilities,
                                                core::ContextState& context,
                                                std::vector<core::System> systems,
                                                const generativeqc_method_descriptor& descriptor,
                                                generativeqc_batch_flags flags);

}  // namespace generativeqc::methods::detail

#endif
