#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>

#include "runtime/host_resource_ledger.hpp"

struct generativeqc_host_resources {
  std::shared_ptr<generativeqc::runtime::HostResourceLedger> ledger;
};

/** Conservative ECP setup bytes used by both planning and native admission. */
extern "C" int generativeqc_resource_ecp_workspace_v1(std::size_t nao, std::size_t cartesian_nao,
                                                      std::size_t atoms, std::size_t primitives,
                                                      std::size_t terms, int cuda,
                                                      std::uint64_t* output);
