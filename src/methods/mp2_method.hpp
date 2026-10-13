#pragma once
#include "methods/method.hpp"
namespace generativeqc::methods::detail {
generativeqc_status validate_mp2_system(generativeqc_method, const core::System&, std::string&);
std::unique_ptr<PreparedCalculation> prepare_mp2_calculation(
    const Capabilities&, core::ContextState&, const core::System&,
    const generativeqc_method_descriptor&, const core::System* auxiliary_basis = nullptr);
std::unique_ptr<PreparedBatch> prepare_mp2_batch(const Capabilities&, core::ContextState&,
                                                 std::vector<core::System>,
                                                 const generativeqc_method_descriptor&,
                                                 generativeqc_batch_flags,
                                                 const core::System* auxiliary_basis = nullptr);
}  // namespace generativeqc::methods::detail
