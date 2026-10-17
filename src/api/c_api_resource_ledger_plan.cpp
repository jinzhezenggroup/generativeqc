#include <cstddef>
#include <cstdint>
#include <exception>
#include <limits>
#include <memory>
#include <new>
#include <stdexcept>
#include <vector>

#include "runtime/bounded_workspace.hpp"
#include "runtime/resource_ledger.hpp"
#include "runtime/resource_ledger_plan.hpp"

extern "C" {

/** Private metadata-only selected-inventory to persistent-ledger binding.
 * Rows are (bytes, atomic space, visible device, first phase, last phase,
 * runtime allowance). Atomic spaces are 2 pinned, 3 pageable, 4 device;
 * host rows require device=0. Allowances never fund explicit numeric storage.
 * The caller has already admitted the global budget and selected one owner.
 * No CUDA/context/provider operation is performed.
 *
 * Success publishes a new v1-compatible handle and three receipt words:
 * device, numeric limit, full declared device peak. The output handle must be
 * null on entry; both outputs must be disjoint from each other and input.
 * Rejection touches neither output: 1 invalid, 2 overflow, 3 metadata allocation
 * failure, 4 unsupported missing/multiple-device owner. Existing handles are
 * never replaced. Destroy/read/bind use the existing ledger APIs.
 */
int generativeqc_resource_ledger_create_from_buffers_v1(const std::uint64_t* buffers,
                                                        std::size_t buffer_value_count,
                                                        void** handle, std::uint64_t* receipt,
                                                        std::size_t receipt_count) {
  using namespace generativeqc::runtime;
  try {
    if (!buffers || !buffer_value_count || buffer_value_count % 6 || !handle || !receipt ||
        receipt_count != 3)
      return 1;
    const auto input_bytes = size_mul(buffer_value_count, sizeof(std::uint64_t));
    const auto receipt_bytes = size_mul(receipt_count, sizeof(std::uint64_t));
    if (ranges_overlap(buffers, input_bytes, handle, sizeof(*handle)) ||
        ranges_overlap(buffers, input_bytes, receipt, receipt_bytes) ||
        ranges_overlap(handle, sizeof(*handle), receipt, receipt_bytes) || *handle)
      return 1;
    std::vector<resources::DeviceLedgerBuffer> capacities;
    for (std::size_t index = 0; index < buffer_value_count; index += 6) {
      const auto* row = buffers + index;
      if ((row[1] != resources::kPinnedSpace && row[1] != resources::kPageableSpace &&
           row[1] != resources::kFirstDeviceSpace) ||
          row[5] > 1)
        return 1;
      capacities.push_back(
          {row[0], static_cast<std::size_t>(row[1]), row[2], row[3], row[4], bool(row[5])});
    }
    const auto admission = resources::device_ledger_admission(capacities);
    if (!admission) return 4;
    if (admission->numeric_limit > std::numeric_limits<std::size_t>::max())
      throw std::overflow_error("native device ledger capacity exceeds size_t");
    auto ledger = std::make_unique<std::shared_ptr<DeviceResourceLedger>>(
        std::make_shared<DeviceResourceLedger>());
    (*ledger)->limit = static_cast<std::size_t>(admission->numeric_limit);
    (*ledger)->device = admission->device;
    receipt[0] = static_cast<std::uint64_t>(admission->device);
    receipt[1] = admission->numeric_limit;
    receipt[2] = admission->declared_peak;
    *handle = ledger.release();
    return 0;
  } catch (const std::bad_alloc&) {
    return 3;
  } catch (const std::overflow_error&) {
    return 2;
  } catch (const std::exception&) {
    return 1;
  }
}

}  // extern "C"
