#ifndef GENERATIVEQC_CLI_GEOMETRY_COMMAND_HPP
#define GENERATIVEQC_CLI_GEOMETRY_COMMAND_HPP

#include <vector>

#include "cli/options.hpp"
#include "generativeqc/generativeqc.h"

namespace generativeqc::cli {
// An execution-only entry; scientific optimizers never require Python.
int geometry_command(const RunOptions& options, std::vector<generativeqc_atom> atoms);
}  // namespace generativeqc::cli

#endif
