#ifndef GENERATIVEQC_CLI_PROFILES_HPP
#define GENERATIVEQC_CLI_PROFILES_HPP

#include <filesystem>

#include "cli/json.hpp"

namespace generativeqc::cli {
std::filesystem::path profile_cache_root();
// Export and validation never probe a GPU; install/diagnose use visible ordinals.
int profile_bundle_command(int argc, char** argv);
// CUDA execution can select a validated complete binary. CPU never enters this path.
void activate_local_profile(int argc, char** argv, int device_id);
Json validate_profile_bundle(const std::filesystem::path& directory);
}  // namespace generativeqc::cli
#endif
