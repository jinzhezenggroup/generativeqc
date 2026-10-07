#pragma once

#include <cstdlib>
#include <cstring>
#include <stdexcept>

#include "scf/aot_shell_registry.hpp"
#include "scf/generated_shell_task.hpp"

namespace generativeqc::scf::cuda_execution {

/** Freeze the bounded raw-K experiment independently of recurrence selection.
 * No environment reads occur during execution or after output accumulation.
 */
inline detail::GeneratedExchangeTaskSchedule prepare_direct_exchange_task_schedule() {
  const char* value = std::getenv("GENERATIVEQC_DIRECT_K_TASK_SCHEDULE");
  if (value == nullptr || *value == '\0' || std::strcmp(value, "incumbent") == 0)
    return detail::GeneratedExchangeTaskSchedule::Incumbent;
  if (std::strcmp(value, "fill") == 0) return detail::GeneratedExchangeTaskSchedule::Fill;
  if (std::strcmp(value, "primitive") == 0)
    return detail::GeneratedExchangeTaskSchedule::Primitive;
  throw std::invalid_argument("Direct K task schedule must be incumbent, fill or primitive");
}

/** Optional lowering is frozen by the prepared owner, independently for J/K.
 * Empty/incumbent retains the qualified default. A Rys request intersects the
 * compiler inventory; absent classes keep their incumbent exact recurrence.
 * These controls qualify a candidate, rather than assert a performance win.
 */
inline std::uint64_t prepare_direct_fock_rys_mask(bool exchange) {
  const char* value = std::getenv(exchange ? "GENERATIVEQC_DIRECT_K_FOCK_LOWERING"
                                           : "GENERATIVEQC_DIRECT_J_FOCK_LOWERING");
  if (value == nullptr || *value == '\0' || std::strcmp(value, "incumbent") == 0) return 0;
  if (std::strcmp(value, "rys") != 0)
    throw std::invalid_argument("Direct Fock lowering must be incumbent or rys");
  return generated::enabled_rys_fock_shell_class_mask();
}

/** Select before launch; never retry a failed launch into partially written output. */
inline auto direct_fock_streaming_launcher(std::uint64_t rys_mask, unsigned shell_class) {
  return (rys_mask & (std::uint64_t{1} << shell_class))
             ? generated::launch_shell_class_rys_streaming_fock
             : generated::launch_shell_class_streaming_fock;
}

}  // namespace generativeqc::scf::cuda_execution
