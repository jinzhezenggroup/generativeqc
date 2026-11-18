#pragma once

#include "cli/hessian.hpp"
#include "generativeqc/saddle.h"

namespace generativeqc::cli {

/** Fixed-chemistry molecular task: no minimum escape or cheaper stages.
 * Masses are mandatory ordered physical/isotopic dalton values, not guessed
 * from a mass table. Hessian controls have exactly their endpoint meanings.
 */
struct SaddleInvocation {
  HessianInvocation derivatives;
  generativeqc_saddle_options_v1 controls;
  std::string masses_file, initial_mode_file;
  std::uint64_t root_bytes{128ULL << 20};
  double minimum_occupied_overlap{0.7};
  bool progress_json{};
};

SaddleInvocation parse_saddle_invocation(int argc, char** argv);
int saddle_command(const RunOptions& options, const std::vector<generativeqc_atom>& atoms,
                   const SaddleInvocation& invocation);

}  // namespace generativeqc::cli
