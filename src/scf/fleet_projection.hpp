#pragma once

#include <array>
#include <cstddef>
#include <optional>
#include <span>
#include <string>
#include <vector>

#include "scf/occupied_projection.hpp"

namespace generativeqc::scf::fleet_projection {

/** Synchronously borrowed coordinate requests in prepared input order. */
using Coordinates = std::vector<std::optional<std::span<const double>>>;

/** Explicit per-item rejection policy and numeric workspace budget. Prepared
 * HF arenas, topology metadata and provider-private memory remain separate. */
struct Controls {
  projection::Policy policy;
  std::size_t maximum_host_bytes{};
  bool strict{true};
};

/** Conservative simultaneously-live numeric bound for eligible fleet items.
 * Rejected numerical projections can reduce the actual retained candidate set. */
struct Capacity {
  std::size_t numeric_peak_bytes{}, retained_bytes{}, largest_density_bytes{};
};

/** Detached diagnostics only: source densities never leave their native owner.
 * A projected seed is a proposal and cannot establish target convergence. */
struct Item {
  bool accepted{};
  std::string reason;
  std::size_t spins{}, numeric_workspace_bytes{};
  double source_energy{};
  int source_iterations{};
  std::array<projection::Diagnostics, 2> diagnostics{};
};

struct Result {
  Capacity capacity;
  std::vector<Item> items;
};

}  // namespace generativeqc::scf::fleet_projection
