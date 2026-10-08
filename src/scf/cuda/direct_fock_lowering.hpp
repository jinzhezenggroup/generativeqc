#pragma once

#include <cstdlib>
#include <cstring>
#include <stdexcept>

#include "scf/aot_shell_registry.hpp"
#include "scf/generated_shell_task.hpp"

namespace generativeqc::scf::cuda_execution {

/** Freeze raw-K scheduling independently of recurrence selection.
 * Cross-chunk filling is the default; incumbent is an explicit rollback.
 * No environment reads occur during execution or after output accumulation.
 */
inline detail::GeneratedExchangeTaskSchedule prepare_direct_exchange_task_schedule() {
  const char* value = std::getenv("GENERATIVEQC_DIRECT_K_TASK_SCHEDULE");
  if (value == nullptr || *value == '\0' || std::strcmp(value, "fill") == 0)
    return detail::GeneratedExchangeTaskSchedule::Fill;
  if (std::strcmp(value, "incumbent") == 0) return detail::GeneratedExchangeTaskSchedule::Incumbent;
  if (std::strcmp(value, "primitive") == 0) return detail::GeneratedExchangeTaskSchedule::Primitive;
  throw std::invalid_argument("Direct K task schedule must be incumbent, fill or primitive");
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
  if (exchange && (std::strcmp(value, "block") == 0 || std::strcmp(value, "rys-task") == 0)) return 0;
  throw std::invalid_argument(exchange ? "Direct K Fock lowering must be incumbent, rys, block or rys-task"
                                       : "Direct J Fock lowering must be incumbent or rys");
}

/** K-only block contraction is a separate compiled owner; J never reserves it. */
inline std::uint64_t prepare_direct_fock_k_block_mask() {
  const char* value = std::getenv("GENERATIVEQC_DIRECT_K_FOCK_LOWERING");
  if (value == nullptr || *value == '\0' || std::strcmp(value, "incumbent") == 0 ||
      std::strcmp(value, "rys") == 0 || std::strcmp(value, "rys-task") == 0)
    return 0;
  if (std::strcmp(value, "block") != 0)
    throw std::invalid_argument("Direct K Fock lowering must be incumbent, rys, block or rys-task");
  return generated::enabled_k_block_fock_shell_class_mask();
}

/** Freeze the qualified K-only task preference; incumbent explicitly rolls back.
 * Explicit rys-task requests retain full capability for further experiments.
 * The registry owns target/class qualification and intersects enabled coverage.
 */
inline std::uint64_t prepare_direct_fock_rys_task_mask() {
  const char* value = std::getenv("GENERATIVEQC_DIRECT_K_FOCK_LOWERING");
  if (value == nullptr || *value == '\0')
    return generated::preferred_rys_task_fock_shell_class_mask();
  return value != nullptr && std::strcmp(value, "rys-task") == 0
             ? generated::enabled_rys_task_fock_shell_class_mask() : 0;
}

/** Select before launch; never retry a failed launch into partially written output. */
inline auto direct_fock_streaming_launcher(std::uint64_t rys_mask, std::uint64_t k_block_mask,
                                           std::uint64_t rys_task_mask, unsigned shell_class) {
  const auto bit = std::uint64_t{1} << shell_class;
  return (rys_task_mask & bit) ? generated::launch_shell_class_rys_task_streaming_fock
      : ((k_block_mask & bit) ? generated::launch_shell_class_k_block_streaming_fock
      : ((rys_mask & bit) ? generated::launch_shell_class_rys_streaming_fock
                         : generated::launch_shell_class_streaming_fock));
}

/** Compatibility for prepared owners without a task-parallel alternative. */
inline auto direct_fock_streaming_launcher(std::uint64_t rys_mask, std::uint64_t k_block_mask,
                                           unsigned shell_class) {
  return direct_fock_streaming_launcher(rys_mask, k_block_mask, 0, shell_class);
}

/** Compatibility overload for J/HF callers without a K-only alternative. */
inline auto direct_fock_streaming_launcher(std::uint64_t rys_mask, unsigned shell_class) {
  return direct_fock_streaming_launcher(rys_mask, 0, shell_class);
}

}  // namespace generativeqc::scf::cuda_execution
