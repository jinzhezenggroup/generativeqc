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

/** Lowering preference and task scheduling are independent preparation controls. */
enum class DirectFockLowering { Default, Incumbent, Rys, Block, RysTask };

/** Parse once before allocation/launch; K-only alternatives are invalid for J. */
inline DirectFockLowering prepare_direct_fock_lowering(bool exchange) {
  const char* value = std::getenv(exchange ? "GENERATIVEQC_DIRECT_K_FOCK_LOWERING"
                                           : "GENERATIVEQC_DIRECT_J_FOCK_LOWERING");
  if (value == nullptr || *value == '\0')
    return exchange ? DirectFockLowering::Default : DirectFockLowering::Incumbent;
  if (std::strcmp(value, "incumbent") == 0) return DirectFockLowering::Incumbent;
  if (std::strcmp(value, "rys") == 0) return DirectFockLowering::Rys;
  if (exchange && std::strcmp(value, "block") == 0) return DirectFockLowering::Block;
  if (exchange && std::strcmp(value, "rys-task") == 0) return DirectFockLowering::RysTask;
  throw std::invalid_argument(
      exchange ? "Direct K Fock lowering must be incumbent, rys, block or rys-task"
               : "Direct J Fock lowering must be incumbent or rys");
}

/** Compatibility adapter for J/HF and existing explicit component-Rys callers. */
inline std::uint64_t prepare_direct_fock_rys_mask(bool exchange) {
  return prepare_direct_fock_lowering(exchange) == DirectFockLowering::Rys
             ? generated::enabled_rys_fock_shell_class_mask()
             : 0;
}

/** K-only block contraction is a separate compiled owner; J never reserves it. */
inline std::uint64_t prepare_direct_fock_k_block_mask() {
  return prepare_direct_fock_lowering(true) == DirectFockLowering::Block
             ? generated::enabled_k_block_fock_shell_class_mask()
             : 0;
}

/** Freeze the qualified K-only task preference; incumbent explicitly rolls back.
 * Explicit rys-task requests retain full capability for further experiments.
 * The registry owns target/class qualification and intersects enabled coverage.
 */
inline std::uint64_t prepare_direct_fock_rys_task_mask() {
  const auto lowering = prepare_direct_fock_lowering(true);
  if (lowering == DirectFockLowering::Default)
    return generated::preferred_rys_task_fock_shell_class_mask();
  return lowering == DirectFockLowering::RysTask
             ? generated::enabled_rys_task_fock_shell_class_mask()
             : 0;
}

/** Frozen prepared K policy; value initialization keeps non-K callers incumbent. */
struct DirectExchangeSelection {
  std::uint64_t rys_fock_mask{}, k_block_fock_mask{}, rys_task_fock_mask{};
  detail::GeneratedExchangeTaskSchedule task_schedule{};
};

/** Resolve both axes once and intersect every alternative with reachable values.
 * Only the registry's qualified task preference is automatic; work kernels cover
 * remaining eligible classes. Explicit lowerings do not enable task preference.
 */
inline DirectExchangeSelection prepare_direct_exchange_selection(std::uint64_t class_mask) {
  DirectExchangeSelection selection;
  const auto lowering = prepare_direct_fock_lowering(true);
  selection.task_schedule = prepare_direct_exchange_task_schedule();
  switch (lowering) {
    case DirectFockLowering::Default:
      selection.rys_task_fock_mask = generated::preferred_rys_task_fock_shell_class_mask();
      break;
    case DirectFockLowering::RysTask:
      selection.rys_task_fock_mask = generated::enabled_rys_task_fock_shell_class_mask();
      break;
    case DirectFockLowering::Rys:
      selection.rys_fock_mask = generated::enabled_rys_fock_shell_class_mask();
      break;
    case DirectFockLowering::Block:
      selection.k_block_fock_mask = generated::enabled_k_block_fock_shell_class_mask();
      break;
    case DirectFockLowering::Incumbent:
      break;
  }
  selection.rys_fock_mask &= class_mask;
  selection.k_block_fock_mask &= class_mask;
  selection.rys_task_fock_mask &= class_mask;
  return selection;
}

/** Select without environment reads or retries into partially accumulated output.
 * Rys-task keeps its value producer with the frozen task/work queue. Block falls
 * back to work/incumbent for UHF; unsupported work classes fall back in registry.
 */
inline auto direct_fock_streaming_launcher(const DirectExchangeSelection& selection,
                                           unsigned shell_class, bool unrestricted) {
  const auto bit = std::uint64_t{1} << shell_class;
  return (selection.rys_task_fock_mask & bit)
             ? (selection.task_schedule == detail::GeneratedExchangeTaskSchedule::Work
                    ? generated::launch_shell_class_rys_task_work_streaming_fock
                    : generated::launch_shell_class_rys_task_streaming_fock)
             : ((!unrestricted && (selection.k_block_fock_mask & bit))
                    ? generated::launch_shell_class_k_block_streaming_fock
                    : ((selection.rys_fock_mask & bit)
                           ? generated::launch_shell_class_rys_streaming_fock
                           : (selection.task_schedule == detail::GeneratedExchangeTaskSchedule::Work
                                  ? generated::launch_shell_class_work_streaming_fock
                                  : generated::launch_shell_class_streaming_fock)));
}

/** J/HF callers cannot accidentally opt into K-only lowerings or work scheduling. */
inline auto direct_fock_streaming_launcher(std::uint64_t rys_mask, unsigned shell_class) {
  return direct_fock_streaming_launcher(DirectExchangeSelection{rys_mask}, shell_class, false);
}

}  // namespace generativeqc::scf::cuda_execution
