#pragma once

#include <cstdlib>
#include <cstring>
#include <stdexcept>

#include "scf/aot_shell_registry.hpp"
#include "scf/generated_shell_task.hpp"

namespace generativeqc::scf::cuda_execution {

/** Freeze raw-K scheduling independently of recurrence selection.
 * Primitive-work buckets are the default; fill is an explicit rollback.
 * No environment reads occur during execution or after output accumulation.
 */
inline detail::GeneratedExchangeTaskSchedule prepare_direct_exchange_task_schedule() {
  const char* value = std::getenv("GENERATIVEQC_DIRECT_K_TASK_SCHEDULE");
  if (value == nullptr || *value == '\0' || std::strcmp(value, "work") == 0)
    return detail::GeneratedExchangeTaskSchedule::Work;
  if (std::strcmp(value, "fill") == 0) return detail::GeneratedExchangeTaskSchedule::Fill;
  if (std::strcmp(value, "incumbent") == 0) return detail::GeneratedExchangeTaskSchedule::Incumbent;
  if (std::strcmp(value, "primitive") == 0) return detail::GeneratedExchangeTaskSchedule::Primitive;
  throw std::invalid_argument("Direct K task schedule must be incumbent, fill, primitive or work");
}

/** Optional lowering is frozen by the prepared owner, independently for J/K.
 * Empty/incumbent retains the qualified default. Alternative requests intersect
 * the compiler inventory; absent classes keep their incumbent exact recurrence.
 * These controls qualify candidates rather than assert a performance win.
 */
inline std::uint64_t prepare_direct_fock_rys_mask(bool exchange) {
  const char* value = std::getenv(exchange ? "GENERATIVEQC_DIRECT_K_FOCK_LOWERING"
                                           : "GENERATIVEQC_DIRECT_J_FOCK_LOWERING");
  if (value == nullptr || *value == '\0' || std::strcmp(value, "incumbent") == 0) return 0;
  if (std::strcmp(value, "rys") == 0) return generated::enabled_rys_fock_shell_class_mask();
  if (exchange && std::strcmp(value, "block") == 0) return 0;
  throw std::invalid_argument(exchange ? "Direct K Fock lowering must be incumbent, rys, or block"
                                       : "Direct J Fock lowering must be incumbent or rys");
}

/** K-only block contraction is a separate compiled owner; J never reserves it. */
inline std::uint64_t prepare_direct_fock_k_block_mask() {
  const char* value = std::getenv("GENERATIVEQC_DIRECT_K_FOCK_LOWERING");
  if (value == nullptr || *value == '\0' || std::strcmp(value, "incumbent") == 0 ||
      std::strcmp(value, "rys") == 0)
    return 0;
  if (std::strcmp(value, "block") != 0)
    throw std::invalid_argument("Direct K Fock lowering must be incumbent, rys, or block");
  return generated::enabled_k_block_fock_shell_class_mask();
}

/** Select before launch; never retry a failed launch into partially written output. */
inline auto direct_fock_streaming_launcher(std::uint64_t rys_mask, std::uint64_t k_block_mask,
                                           unsigned shell_class, bool work_aware = false) {
  const auto bit = std::uint64_t{1} << shell_class;
  return (k_block_mask & bit)
             ? generated::launch_shell_class_k_block_streaming_fock
             : ((rys_mask & bit) ? generated::launch_shell_class_rys_streaming_fock
                                 : (work_aware ? generated::launch_shell_class_work_streaming_fock
                                               : generated::launch_shell_class_streaming_fock));
}

/** Compatibility overload for J/HF callers without a K-only alternative. */
inline auto direct_fock_streaming_launcher(std::uint64_t rys_mask, unsigned shell_class) {
  return direct_fock_streaming_launcher(rys_mask, 0, shell_class);
}

}  // namespace generativeqc::scf::cuda_execution
