#pragma once

#include <memory>

#include "runtime/host_resource_ledger.hpp"

struct generativeqc_host_resources {
  std::shared_ptr<generativeqc::runtime::HostResourceLedger> ledger;
};
