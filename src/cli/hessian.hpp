#ifndef GENERATIVEQC_CLI_HESSIAN_HPP
#define GENERATIVEQC_CLI_HESSIAN_HPP

#include <array>
#include <optional>
#include <string>
#include <vector>

#include "cli/json.hpp"
#include "cli/options.hpp"

namespace generativeqc::cli {

/** Authenticated PBE base plus the user-specified, canonical SR/LR primitives. */
struct RangePbeMethodIR {
  std::string identifier;
  std::string identity;
  unsigned spins{};
  std::array<generativeqc_ks_exchange_term, 2> exchange{};
};

// Build-time compiler AD; production Hessian execution never starts Python/JIT.
struct HessianInvocation {
  std::vector<std::string> run_arguments;
  std::string operation;
  std::string direction_file;
  std::optional<RangePbeMethodIR> range_pbe;
  std::uint64_t device_bytes{2ULL << 30};
  std::uint64_t integral_bytes{64ULL << 20};
  std::uint64_t grid_bytes{128ULL << 20};
  std::uint64_t output_bytes{64ULL << 20};
  std::uint64_t response_bytes{64ULL << 20};
  std::uint64_t max_nonlocal_pairs{16777216};
  std::uint64_t numeric_bytes{512ULL << 20}, ecp_bytes{128ULL << 20};
  std::uint64_t dispersion_bytes{64ULL << 20}, max_eri_primitives{100'000'000};
  std::uint64_t cpu_packed_eri_bytes{};
  std::uint64_t max_d3_pairs{16'777'216}, max_map_terms{1ULL << 40};
  bool cpu_controls_explicit{}, cuda_controls_explicit{}, response_atol_explicit{};
  double response_rtol{1e-10};
  double response_atol{1e-12};
  std::uint32_t response_restart{30};
  std::uint32_t response_iterations{200};
  bool response_fp32_proposal{};
  std::uint32_t nonlocal_radial{}, nonlocal_polar{}, nonlocal_azimuth{};
  std::optional<std::uint32_t> nonlocal_level;
};

// Split endpoint-only controls from the ordinary, centrally validated run flags.
HessianInvocation parse_hessian_invocation(int argc, char** argv);

// Execute linked CPU/CUDA compiler AD and publish a complete accepted result only.
int hessian_command(const RunOptions& options, const std::vector<generativeqc_atom>& atoms,
                    const HessianInvocation& invocation);

}  // namespace generativeqc::cli
#endif
