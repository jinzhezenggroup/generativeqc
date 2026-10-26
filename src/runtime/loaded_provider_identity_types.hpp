#pragma once

#include <memory>
#include <string>

namespace generativeqc::runtime {

enum class LoadedIdentityStatus {
  available,
  unsupported_platform,
  not_loaded,
  missing_symbol,
  wrong_symbol_owner,
  missing_build_id,
  malformed_notes,
  ambiguous_build_id,
};

struct LoadedBuildIdentity {
  LoadedIdentityStatus status{LoadedIdentityStatus::missing_build_id};
  std::string build_id;
  std::shared_ptr<void> library_pin;
};

}  // namespace generativeqc::runtime
