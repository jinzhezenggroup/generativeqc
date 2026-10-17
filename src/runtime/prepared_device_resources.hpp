#pragma once

#include <limits>
#include <memory>
#include <span>
#include <stdexcept>

#include "runtime/resource_ledger.hpp"
#include "runtime/resource_ledger_plan.hpp"

namespace generativeqc::runtime::resources {

/** Share the selected-inventory admission authority across private/public ABIs.
 * This creates metadata only; no CUDA operation, allocation receipt, method
 * capability or global-plan identity is implied by the declared capacities. */
inline std::shared_ptr<DeviceResourceLedger> prepare_device_resources(
    std::span<const DeviceLedgerBuffer> buffers) {
  const auto admission = device_ledger_admission(buffers);
  if (!admission) return {};
  if (admission->numeric_limit > std::numeric_limits<std::size_t>::max())
    throw std::overflow_error("native device ledger capacity exceeds size_t");
  auto ledger = std::make_shared<DeviceResourceLedger>();
  ledger->limit = static_cast<std::size_t>(admission->numeric_limit);
  ledger->device = admission->device;
  ledger->declared_peak = admission->declared_peak;
  return ledger;
}

}  // namespace generativeqc::runtime::resources
