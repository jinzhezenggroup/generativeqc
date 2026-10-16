#pragma once

#include <cstddef>
#include <cstdint>
#include <optional>
#include <span>

#include "runtime/resource_planning.hpp"

namespace generativeqc::runtime::resources {

/** One selected capacity, before binding any scientific provider. Device slots
 * use kFirstDeviceSpace with a separate visible ordinal, not a dense allocation
 * indexed by that ordinal. Host rows require a zero device field.
 */
struct DeviceLedgerBuffer {
  std::uint64_t bytes{};
  std::size_t space{};
  std::uint64_t device{}, first_phase{}, last_phase{};
  bool runtime_allowance{};
};

struct DeviceLedgerAdmission {
  int device{};
  std::uint64_t numeric_limit{}, declared_peak{};
};

/** Derive the explicit numeric ledger cap from a caller-selected inventory.
 * Runtime/library allowances participate in the declared peak but cannot fund
 * tracked allocations. Sweep lifetimes before taking peaks: subtracting two
 * independent peaks is incorrect when their lifetimes differ. Missing/multiple
 * devices are an unsupported owner domain, not a guessed device or allocation.
 * This checks declared metadata, not global-plan identity or physical coverage.
 */
std::optional<DeviceLedgerAdmission> device_ledger_admission(
    std::span<const DeviceLedgerBuffer> buffers);

}  // namespace generativeqc::runtime::resources
