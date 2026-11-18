#pragma once

#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <span>
#include <string_view>

#include "dft/dispersion/d3_parameters.hpp"

namespace generativeqc::dft::dispersion {

struct HostD3ColumnLimits {
  std::size_t max_numeric_bytes{}, max_pair_visits{}, max_reference_terms{};
};

struct HostD3ColumnPlan {
  std::size_t numeric_bytes{}, table_bytes{}, pair_visits{}, reference_terms{};
};

struct HostD3ColumnWork {
  std::size_t coordination_pairs{}, reference_atoms{}, energy_pairs{}, reverse_pairs{};
  std::size_t reference_terms{};
};

struct HostD3ColumnValue {
  double energy{}, right{};
};

/** AOT, FP64 two-body D3(BJ) energy/gradient/HVP for one owned geometry.
 * Reuses the canonical dispersion Graphs and pinned production tables. No
 * electronic response, finite differences, runtime compiler or Python is used.
 * Only uncut BJ with ATM off is admitted, matching the qualified TensorIR
 * second-order domain; finite cutoffs and normalization underflow are refused.
 * Three pair traversals reuse one physical CN jet per pair for all left axes.
 * The shape-only plan includes owned input snapshots, pair jets, leaf scratch
 * and simultaneous transactional outputs; static tables are reported separately.
 * Allocator/object/callback overhead, caller storage and total RSS are not caps.
 * The optional validator runs before work and immediately before publication.
 * Calls on the same owner may not overlap or reenter.
 */
class HostD3Column {
 public:
  using ValidateCurrent = std::function<void()>;
  static HostD3ColumnPlan plan(std::size_t atoms);
  HostD3Column(std::span<const std::int32_t> atomic_numbers, std::span<const double> coordinates,
               D3Parameters parameters, HostD3ColumnLimits limits,
               ValidateCurrent validate_current = {});
  ~HostD3Column();
  HostD3Column(const HostD3Column&) = delete;
  HostD3Column& operator=(const HostD3Column&) = delete;

  /** Both atom/xyz outputs publish only after all work and input authentication.
   * Direction may alias an output; gradient and HVP may not alias each other.
   */
  HostD3ColumnValue apply(std::span<const double> direction, std::span<double> gradient,
                          std::span<double> action);
  [[nodiscard]] HostD3ColumnWork work() const noexcept;
  [[nodiscard]] static std::string_view table_identity() noexcept;
  [[nodiscard]] static std::string_view radii_identity() noexcept;

 private:
  class Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace generativeqc::dft::dispersion
