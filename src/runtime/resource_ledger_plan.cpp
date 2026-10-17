#include "runtime/resource_ledger_plan.hpp"

#include <limits>
#include <map>
#include <stdexcept>
#include <vector>

#include "runtime/bounded_workspace.hpp"

namespace generativeqc::runtime::resources {

std::optional<DeviceLedgerAdmission> device_ledger_admission(
    std::span<const DeviceLedgerBuffer> buffers) {
  if (buffers.empty()) throw std::invalid_argument("empty device ledger inventory");
  std::map<std::uint64_t, std::size_t> devices;
  std::vector<Buffer> all, numeric;
  for (const auto& buffer : buffers) {
    auto space = buffer.space;
    if (space == kFirstDeviceSpace) {
      if (buffer.device > static_cast<std::uint64_t>(std::numeric_limits<int>::max()))
        throw std::invalid_argument("device ledger ordinal exceeds the native device domain");
      space = devices.try_emplace(buffer.device, size_add(kFirstDeviceSpace, devices.size()))
                  .first->second;
    } else if ((space != kPinnedSpace && space != kPageableSpace) || buffer.device != 0) {
      throw std::invalid_argument("invalid device ledger memory space");
    }
    const Buffer capacity{buffer.bytes, space, buffer.first_phase, buffer.last_phase, false};
    all.push_back(capacity);
    if (!buffer.runtime_allowance) numeric.push_back(capacity);
  }
  const auto spaces = size_add(kFirstDeviceSpace, devices.size());
  const auto declared = account(all, spaces);
  const auto admitted = account(numeric, spaces);
  if (devices.size() != 1) return std::nullopt;
  const auto slot = devices.begin()->second;
  return DeviceLedgerAdmission{static_cast<int>(devices.begin()->first), admitted.peaks[slot].bytes,
                               declared.peaks[slot].bytes};
}

}  // namespace generativeqc::runtime::resources
