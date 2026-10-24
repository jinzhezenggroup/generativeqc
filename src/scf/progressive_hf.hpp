#pragma once

#include <cstdint>

#include "generativeqc/generativeqc.h"

namespace generativeqc::scf {

/** The transport executes one synchronous operation against its own prepared
 * owners. No handle, density, provider or Python object crosses this boundary.
 * A callback must not throw; a nonzero return stops scientific work, but never
 * skips closing a successfully prepared owner. */
using ProgressiveHFCallback = int (*)(void*, std::uint32_t, std::uint32_t*);

enum class ProgressiveHFEvent : std::uint32_t {
  prepare_source,
  execute_source,
  prepare_target,
  project,
  execute_target,
  retain_target,
  close_target,
  close_source,
  audit_target,
};

/** Drive the bounded two-stage HF execution policy, not its numerical kernels.
 * Source success admits projection, never target acceptance. Only a successful,
 * converged target may export a density, and its independent physical audit
 * runs after both prepared owners close. Receipt publication includes cleanup
 * even on failure; invalid inputs leave caller storage untouched. */
inline generativeqc_status run_progressive_hf(ProgressiveHFCallback callback, void* state,
                                              std::uint64_t* receipt) noexcept {
  if (!callback || !receipt) return GENERATIVEQC_STATUS_INVALID_ARGUMENT;
  std::uint64_t completed{};
  std::uint64_t flags{};
  generativeqc_status status = GENERATIVEQC_STATUS_SUCCESS;
  const auto invoke = [&](ProgressiveHFEvent event, std::uint32_t allowed) {
    std::uint32_t result{};
    int callback_status{};
    try {
      callback_status = callback(state, static_cast<std::uint32_t>(event), &result);
    } catch (...) {
      callback_status = 1;
    }
    if (callback_status != 0) {
      if (status == GENERATIVEQC_STATUS_SUCCESS) status = GENERATIVEQC_STATUS_INTERNAL_ERROR;
      return std::uint32_t{};
    }
    completed |= std::uint64_t{1} << static_cast<std::uint32_t>(event);
    if ((result & ~allowed) != 0 || (event == ProgressiveHFEvent::execute_target && result == 2)) {
      if (status == GENERATIVEQC_STATUS_SUCCESS) status = GENERATIVEQC_STATUS_INVALID_ARGUMENT;
      return std::uint32_t{};
    }
    return result;
  };
  invoke(ProgressiveHFEvent::prepare_source, 0);
  const bool source_live =
      (completed &
       (std::uint64_t{1} << static_cast<std::uint32_t>(ProgressiveHFEvent::prepare_source))) != 0;
  if (source_live) {
    const auto source =
        status == GENERATIVEQC_STATUS_SUCCESS ? invoke(ProgressiveHFEvent::execute_source, 1) : 0;
    flags |= source;
    if (status == GENERATIVEQC_STATUS_SUCCESS) {
      invoke(ProgressiveHFEvent::prepare_target, 0);
      const bool target_live = (completed & (std::uint64_t{1} << static_cast<std::uint32_t>(
                                                 ProgressiveHFEvent::prepare_target))) != 0;
      if (target_live) {
        if (status == GENERATIVEQC_STATUS_SUCCESS && source != 0)
          invoke(ProgressiveHFEvent::project, 0);
        if (status == GENERATIVEQC_STATUS_SUCCESS) {
          const auto target = invoke(ProgressiveHFEvent::execute_target, 3);
          flags |= static_cast<std::uint64_t>(target) << 1;
          if (status == GENERATIVEQC_STATUS_SUCCESS && target == 3)
            flags |= static_cast<std::uint64_t>(invoke(ProgressiveHFEvent::retain_target, 1)) << 3;
        }
        invoke(ProgressiveHFEvent::close_target, 0);
      }
    }
    invoke(ProgressiveHFEvent::close_source, 0);
  }
  if (status == GENERATIVEQC_STATUS_SUCCESS && (flags & 8) != 0)
    invoke(ProgressiveHFEvent::audit_target, 0);
  receipt[0] = completed;
  receipt[1] = flags;
  return status;
}

}  // namespace generativeqc::scf
