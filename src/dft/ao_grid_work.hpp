#pragma once

#include <cstdint>
#include <limits>
#include <stdexcept>

namespace generativeqc::dft {
/** Cumulative executed indexed-grid schedules, not mask admission estimates.
 *
 * A deriv=2 pass includes its lower-order jets; it is not also a deriv=1 pass.
 * Traffic counts describe explicit panel writes/copies, not measured DRAM
 * transactions. Projection/gather fields cover the density-matrix route;
 * orbital_feature_tiles explicitly identifies the separately scheduled MO path.
 * Density gathering counts only executed matrices: the owned restricted-spin
 * witness permits one gather, while unqualified or unrestricted inputs use two.
 * Optional stage timings are intrusive and never endpoint timers.
 */
struct AoGridWork {
  std::uint64_t evaluation_passes{}, deriv0_passes{}, deriv1_passes{}, deriv2_passes{},
      deriv3_passes{}, deriv0_point_ao{}, deriv1_point_ao{}, deriv2_point_ao{}, deriv3_point_ao{},
      evaluation_tiles{}, evaluation_points{}, active_point_ao{}, dense_point_ao{}, ao_jet_values{};
  std::uint64_t discovery_passes{}, discovery_point_ao{}, discovery_ao_jet_values{};
  std::uint64_t density_gather_passes{}, density_gather_elements{}, projection_passes{},
      projection_matrices{}, projection_fma_pairs{}, projection_output_values{},
      identical_spin_copy_bytes{}, feature_passes{}, ao_map_h2d_bytes{}, scatter_passes{},
      scatter_elements{}, orbital_feature_tiles{};
  double ao_stage_ms{}, density_gather_ms{}, projection_ms{}, feature_ms{};

  static std::uint64_t product(std::uint64_t left, std::uint64_t right) {
    if (right && left > std::numeric_limits<std::uint64_t>::max() / right)
      throw std::overflow_error("AO/grid work product overflow");
    return left * right;
  }
  static void accumulate(std::uint64_t& counter, std::uint64_t value) {
    if (value > std::numeric_limits<std::uint64_t>::max() - counter)
      throw std::overflow_error("AO/grid work counter overflow");
    counter += value;
  }
  void record_ao(std::uint64_t points, std::uint64_t active, std::uint64_t global,
                 std::uint64_t jets, bool discovery = false) {
    if (!points) return;
    if (active > global || (jets != 1 && jets != 4 && jets != 10 && jets != 20))
      throw std::invalid_argument("invalid AO/grid work domain");
    const auto visits = product(points, active);
    const auto values = product(visits, jets);
    if (discovery) {
      if (!active) return;
      accumulate(discovery_passes, 1);
      accumulate(discovery_point_ao, visits);
      accumulate(discovery_ao_jet_values, values);
      return;
    }
    accumulate(evaluation_tiles, 1);
    accumulate(evaluation_points, points);
    accumulate(dense_point_ao, product(points, global));
    if (!active) return;
    accumulate(evaluation_passes, 1);
    auto& order_passes = jets == 1    ? deriv0_passes
                         : jets == 4  ? deriv1_passes
                         : jets == 10 ? deriv2_passes
                                      : deriv3_passes;
    accumulate(order_passes, 1);
    auto& order_visits = jets == 1    ? deriv0_point_ao
                         : jets == 4  ? deriv1_point_ao
                         : jets == 10 ? deriv2_point_ao
                                      : deriv3_point_ao;
    accumulate(order_visits, visits);
    accumulate(active_point_ao, visits);
    accumulate(ao_jet_values, values);
  }
  void record_projection(std::uint64_t points, std::uint64_t active, std::uint64_t spins,
                         std::uint64_t jets, bool identical) {
    if (!points || !active) return;
    const auto matrices = product(spins, jets);
    const auto values = product(product(points, active), matrices);
    accumulate(projection_passes, spins);
    accumulate(projection_matrices, matrices);
    accumulate(projection_output_values, values);
    accumulate(projection_fma_pairs, product(values, active));
    if (identical) accumulate(identical_spin_copy_bytes, product(values, sizeof(double)));
  }
};
}  // namespace generativeqc::dft
