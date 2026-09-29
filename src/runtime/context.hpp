#ifndef GENERATIVEQC_RUNTIME_CONTEXT_HPP
#define GENERATIVEQC_RUNTIME_CONTEXT_HPP

#include <string>

#include "core/types.hpp"

namespace generativeqc::runtime {

generativeqc_status initialize_context(core::ContextState& state, std::string& detail);

#if GENERATIVEQC_HAS_CUDA
generativeqc_status initialize_cuda_context(core::ContextState& state, std::string& detail);
#endif

}  // namespace generativeqc::runtime

#endif
