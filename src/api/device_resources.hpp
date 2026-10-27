#pragma once

#include <memory>

#include "generativeqc/generativeqc.h"
#include "runtime/resource_ledger.hpp"

namespace generativeqc::api {

using DeviceResourceHandle = std::shared_ptr<runtime::DeviceResourceLedger>;

/** Public handles and the older opaque private ABI share the exact allocated
 * shared_ptr object, not a guessed wrapper layout. Only cast back to its real
 * allocated type; the public incomplete C tag is never dereferenced. */
inline const DeviceResourceHandle& device_resource_owner(
    const generativeqc_device_resources* resources) {
  return *reinterpret_cast<const DeviceResourceHandle*>(resources);
}

inline generativeqc_device_resources* device_resource_handle(DeviceResourceHandle* owner) {
  return reinterpret_cast<generativeqc_device_resources*>(owner);
}

}  // namespace generativeqc::api
