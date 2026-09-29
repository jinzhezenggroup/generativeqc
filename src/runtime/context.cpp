#include "runtime/context.hpp"

namespace generativeqc::runtime {

generativeqc_status initialize_context(core::ContextState& state, std::string& detail) {
  if (state.requested_backend == GENERATIVEQC_BACKEND_CPU_REFERENCE) {
    state.executed_backend = GENERATIVEQC_BACKEND_CPU_REFERENCE;
    state.device_name = "CPU reference backend";
    return GENERATIVEQC_STATUS_SUCCESS;
  }
  if (state.requested_backend != GENERATIVEQC_BACKEND_CUDA) {
    detail = "unknown execution backend";
    return GENERATIVEQC_STATUS_INVALID_ARGUMENT;
  }
#if GENERATIVEQC_HAS_CUDA
  return initialize_cuda_context(state, detail);
#else
  detail = "the library was built without CUDA support";
  return GENERATIVEQC_STATUS_NOT_IMPLEMENTED;
#endif
}

}  // namespace generativeqc::runtime
