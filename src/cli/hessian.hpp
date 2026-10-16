#ifndef GENERATIVEQC_CLI_HESSIAN_HPP
#define GENERATIVEQC_CLI_HESSIAN_HPP

#include <string>
#include <vector>

#include "cli/json.hpp"
#include "cli/options.hpp"

namespace generativeqc::cli {

// Build-time compiler AD; production Hessian execution never starts Python/JIT.
struct HessianInvocation {
  std::vector<std::string> run_arguments;
  std::string operation;
  std::string direction_file;
  std::uint64_t device_bytes{2ULL << 30};
  std::uint64_t integral_bytes{64ULL << 20};
  std::uint64_t grid_bytes{128ULL << 20};
  std::uint64_t output_bytes{64ULL << 20};
  std::uint64_t response_bytes{64ULL << 20};
  std::uint64_t max_nonlocal_pairs{16777216};
  double response_rtol{1e-10};
  double response_atol{0};
  std::uint32_t response_restart{30};
  std::uint32_t response_iterations{200};
};

// Split endpoint-only controls from the ordinary, centrally validated run flags.
HessianInvocation parse_hessian_invocation(int argc, char** argv);

// Execute linked CUDA compiler AD and publish a complete accepted result only.
int hessian_command(const RunOptions& options, const std::vector<generativeqc_atom>& atoms,
                    const HessianInvocation& invocation);

}  // namespace generativeqc::cli
#endif
