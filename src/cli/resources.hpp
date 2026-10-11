#ifndef GENERATIVEQC_CLI_RESOURCES_HPP
#define GENERATIVEQC_CLI_RESOURCES_HPP

#include <span>

#include "cli/options.hpp"

namespace generativeqc::cli {
// Metadata-only planning: no Context/System construction, CUDA probe or scientific arrays.
int resources_command(const RunOptions& options, std::span<const generativeqc_atom> atoms);
}  // namespace generativeqc::cli
#endif
