// Resource admission checks must execute in Release qualification builds too.
#ifdef NDEBUG
#undef NDEBUG
#endif

// Packing calls inside assertions must execute in Release qualification too.
#ifdef NDEBUG
#undef NDEBUG
#endif

#include <algorithm>
#include <array>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <cstdlib>

#include "generativeqc/generativeqc.h"
#include "molecule/basis.hpp"
#include "scf/cuda/arena.hpp"
#include "scf/cuda/topology.hpp"
#include "scf/cuda_batch.hpp"

namespace {

void check_incremental_diis_capacity() {
  using generativeqc::scf::small_hf_cuda_resource_layout;
  using generativeqc::scf::small_hf_cuda_resource_layout_v2;
  constexpr auto selection = "GENERATIVEQC_SCF_INCREMENTAL_DIIS_GRAM";
  constexpr auto reduction = "GENERATIVEQC_SCF_INCREMENTAL_DIIS_GRAM_REDUCTION";
  assert(unsetenv(reduction) == 0);
  const std::array<std::uint8_t, 2> angular{0, 0};
  const std::array<std::size_t, 2> primitives{3, 3};
  for (const unsigned history : {1U, 2U, 8U, 64U}) {
    for (const unsigned spins : {1U, 2U}) {
      std::size_t v1[2]{}, v2[2]{}, plan_bytes = 0;
      for (const unsigned incremental : {0U, 1U}) {
        assert(setenv(selection, incremental ? "1" : "0", 1) == 0);
        assert(small_hf_cuda_resource_layout(2, 2, 2, 2, 6, history, spins, v1[incremental],
                                             plan_bytes));
        assert(small_hf_cuda_resource_layout_v2(
            2, 2, 2, angular.data(), primitives.data(), angular.size(), history, spins,
            GENERATIVEQC_PRECISION_FP64, 1e-10, 1e-12, v2[incremental], plan_bytes));
      }
      for (const char* reducer : {"cooperative", "ordered"}) {
        assert(setenv(reduction, reducer, 1) == 0);
        std::size_t ordered_v1{}, ordered_v2{};
        assert(
            small_hf_cuda_resource_layout(2, 2, 2, 2, 6, history, spins, ordered_v1, plan_bytes));
        assert(small_hf_cuda_resource_layout_v2(
            2, 2, 2, angular.data(), primitives.data(), angular.size(), history, spins,
            GENERATIVEQC_PRECISION_FP64, 1e-10, 1e-12, ordered_v2, plan_bytes));
        assert(ordered_v1 == v1[1] && ordered_v2 == v2[1]);
      }
      assert(setenv(reduction, "invalid", 1) == 0);
      std::size_t refused_v1{}, refused_v2{};
      assert(small_hf_cuda_resource_layout(2, 2, 2, 2, 6, history, spins, refused_v1, plan_bytes) ==
             (history < 2));
      assert(small_hf_cuda_resource_layout_v2(2, 2, 2, angular.data(), primitives.data(),
                                              angular.size(), history, spins,
                                              GENERATIVEQC_PRECISION_FP64, 1e-10, 1e-12, refused_v2,
                                              plan_bytes) == (history < 2));
      assert(unsetenv(reduction) == 0);
      const auto gram_bytes = history >= 2 ? sizeof(double) * history * history : 0;
      assert(v1[1] == v1[0] + gram_bytes);
      assert(v2[1] == v2[0] + gram_bytes);
    }
  }
  assert(setenv(selection, "invalid", 1) == 0);
  std::size_t arena_bytes = 0, plan_bytes = 0;
  assert(!small_hf_cuda_resource_layout(2, 2, 2, 2, 6, 8, 1, arena_bytes, plan_bytes));
  assert(!small_hf_cuda_resource_layout_v2(2, 2, 2, angular.data(), primitives.data(),
                                           angular.size(), 8, 1, GENERATIVEQC_PRECISION_FP64, 1e-10,
                                           1e-12, arena_bytes, plan_bytes));
  assert(unsetenv(selection) == 0);
}

void check_h2_cartesian_rhf() {
  using generativeqc::scf::small_hf_cuda_resource_layout_v2;
  using generativeqc::scf::cuda_execution::ArenaLayout;
  using generativeqc::scf::cuda_execution::make_layout;

  const std::array<std::uint8_t, 2> angular{0, 0};
  const std::array<std::size_t, 2> primitives{3, 3};
  std::size_t queried = 0, plan_bytes = 0;
  assert(small_hf_cuda_resource_layout_v2(2, 2, 2, angular.data(), primitives.data(),
                                          angular.size(), 8, 1, GENERATIVEQC_PRECISION_FP64, 1e-10,
                                          1e-12, queried, plan_bytes));

  ArenaLayout energy{}, force{};
  assert(make_layout(1, 2, 2, 2, 2, 3, 1, 0, 27, 0, 0, 0, 0, 0, 0, 0, 6, 8, 0, 1, true, false,
                     false, false, false, false, false, true, energy));
  // Three s-s pairs are retained as the possible PSSS ket list even though
  // this all-s topology has no PSSS bra tasks. Six exact ssss quartets each
  // occupy one compiler-valid tile.
  assert(make_layout(1, 2, 2, 2, 2, 3, 1, 0, 27, 0, 3, 6, 0, 6, 0, 0, 6, 8, 0, 1, false, false,
                     false, false, false, false, false, true, force));
  assert(queried == std::max(energy.bytes, force.bytes));
  assert(plan_bytes != 0);
}

void check_spherical_d_uhf() {
  using generativeqc::scf::small_hf_cuda_resource_layout_v2;
  using generativeqc::scf::cuda_execution::ArenaLayout;
  using generativeqc::scf::cuda_execution::make_layout;

  const std::array<std::uint8_t, 1> angular{2};
  const std::array<std::size_t, 1> primitives{1};
  std::size_t queried = 0, plan_bytes = 0;
  assert(small_hf_cuda_resource_layout_v2(5, 6, 1, angular.data(), primitives.data(),
                                          angular.size(), 8, 2, GENERATIVEQC_PRECISION_FP64, 1e-10,
                                          1e-12, queried, plan_bytes));

  ArenaLayout energy{}, force{};
  assert(make_layout(1, 5, 6, 1, 1, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 8, 0, 2, true, false, false,
                     false, false, false, false, true, energy));
  // One spherical d shell expands to six Cartesian AOs. Its single dddd
  // quartet is one exact tile and requires transformed-direct storage.
  assert(make_layout(1, 5, 6, 1, 1, 1, 1, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 8, 0, 2, false, true, false,
                     false, false, false, false, true, force));
  assert(queried == std::max(energy.bytes, force.bytes));
  assert(plan_bytes != 0);
}

void check_small_spherical_force_packs_direct_transform() {
  generativeqc::core::System system;
  system.atoms = {{2, {0.0, 0.0, -0.7}}, {1, {0.0, 0.0, 0.7}}};
  system.shells = {
      {0, 0, {{1.5, 1.0}}},
      {0, 2, {{0.8, 1.0}}},
      {1, 0, {{1.2, 1.0}}},
  };
  system.charge = 1;
  system.multiplicity = 1;
  system.basis_representation = GENERATIVEQC_BASIS_SPHERICAL;
  std::string detail;
  assert(generativeqc::molecule::validate_and_normalize(system, detail) ==
         GENERATIVEQC_STATUS_SUCCESS);

  std::vector<const std::vector<double>*> no_warm(1, nullptr);
  generativeqc::scf::cuda_execution::HostBatch energy_host;
  assert(generativeqc::scf::cuda_execution::pack_host_batch({system}, no_warm, energy_host, false,
                                                            false, false));
  assert(energy_host.nbf == 7 && energy_host.direct_nbf == 8);
  assert(energy_host.ao_to_direct_transform.empty());

  generativeqc::scf::cuda_execution::HostBatch force_host;
  assert(generativeqc::scf::cuda_execution::pack_host_batch({system}, no_warm, force_host, false,
                                                            false, true));
  assert(force_host.nbf == 7 && force_host.direct_nbf == 8);
  assert(force_host.ao_to_direct_transform.size() == 7 * 8);
}

void check_optional_psss_catalog(unsigned s_shells, unsigned p_shells, unsigned batch_size,
                                 bool unrestricted) {
  using namespace generativeqc::scf::cuda_execution;
  generativeqc::core::System system;
  system.atoms = {{2, {0.0, 0.0, 0.0}}};
  for (unsigned i = 0; i < s_shells; ++i) system.shells.push_back({0, 0, {{1.0 + 0.01 * i, 1.0}}});
  for (unsigned i = 0; i < p_shells; ++i) system.shells.push_back({0, 1, {{0.8 + 0.01 * i, 1.0}}});
  // A d shell makes the public/direct AO dimensions different. Skipping the
  // unused task table must not accidentally select the matrix-only packer.
  system.shells.push_back({0, 2, {{0.7, 1.0}}});
  system.multiplicity = 1;
  system.basis_representation = GENERATIVEQC_BASIS_SPHERICAL;
  std::string detail;
  assert(generativeqc::molecule::validate_and_normalize(system, detail) ==
         GENERATIVEQC_STATUS_SUCCESS);
  const std::vector<generativeqc::core::System> systems(batch_size, system);
  const auto n = generativeqc::molecule::ao_count(system);
  std::vector<double> density(n * n * (unrestricted ? 2 : 1), 0.125);
  const std::vector<const std::vector<double>*> warm(batch_size, &density);
  HostBatch legacy, source;
  // The default still builds the Direct-HF schedule, including per-item offsets.
  assert(pack_host_batch(systems, warm, legacy, unrestricted, false, true));
  assert(
      pack_host_batch(systems, warm, source, unrestricted, false, true, ResidentPsssPolicy::Skip));
  const std::size_t ket_pairs = s_shells * (s_shells + 1ULL) / 2;
  const std::size_t bra_pairs = s_shells * static_cast<std::size_t>(p_shells);
  assert(legacy.psss_resident_ket_pairs.size() == batch_size * ket_pairs);
  assert(!legacy.psss_resident_tasks.empty());
  std::size_t legacy_pair_visits = 0;
  for (const auto& task : legacy.psss_resident_tasks) legacy_pair_visits += task.ket_count;
  assert(legacy_pair_visits == batch_size * bra_pairs * ket_pairs);
  assert(source.psss_resident_tasks.empty() && source.psss_resident_ket_pairs.empty());
  assert(source.psss_resident_tasks.capacity() == 0);
  assert(source.psss_resident_ket_pairs.capacity() == 0);
  assert(same_topology(legacy, source));
  assert(legacy.warm_density == source.warm_density);
  assert(legacy.warm_mask == source.warm_mask);
  assert(source.ao_to_direct_transform.size() == batch_size * n * source.direct_nbf);
  assert(legacy.ao_to_direct_transform == source.ao_to_direct_transform);
}

void check_df_values_pack_g_metadata_without_scf_work() {
  using namespace generativeqc::scf::cuda_execution;
  generativeqc::core::System system;
  system.atoms = {{2, {0.0, 0.0, 0.0}}};
  system.shells = {{0, 0, {{0.5, 1.0}}}, {0, 4, {{0.8, 1.0}}}};
  system.basis_representation = GENERATIVEQC_BASIS_CARTESIAN;
  std::string detail;
  assert(generativeqc::molecule::validate_and_normalize(system, detail) ==
         GENERATIVEQC_STATUS_SUCCESS);
  const std::vector<const std::vector<double>*> no_warm(1, nullptr);
  HostBatch scf, values;
  // g metadata is admitted only for the explicit DF value owner. Ordinary
  // SCF remains bounded by its existing f recurrences and three-term AO ABI.
  assert(!pack_host_batch({system}, no_warm, scf, false, true));
  assert(pack_host_batch({system}, no_warm, values, false, true, false, ResidentPsssPolicy::Skip,
                         HostBasisPacking::DfValues));
  assert(values.nbf == 16 && values.direct_nbf == 16);
  assert(values.ao_term_counts.size() == 16);
  assert(std::all_of(values.ao_term_counts.begin(), values.ao_term_counts.end(),
                     [](auto count) { return count == 1; }));
  assert(values.shell_pair_first.empty() && values.psss_resident_tasks.empty());
  assert(values.psss_resident_ket_pairs.empty() && values.warm_density.empty());
  assert(values.occupied.empty() && values.warm_mask.empty());
  system.basis_representation = GENERATIVEQC_BASIS_SPHERICAL;
  HostBatch invalid;
  assert(!pack_host_batch({system}, no_warm, invalid, false, true, false, ResidentPsssPolicy::Skip,
                          HostBasisPacking::DfValues));
}

}  // namespace

int main() {
  check_incremental_diis_capacity();
  check_h2_cartesian_rhf();
  check_spherical_d_uhf();
  check_small_spherical_force_packs_direct_transform();
  check_df_values_pack_g_metadata_without_scf_work();
  check_optional_psss_catalog(2, 1, 2, false);
  check_optional_psss_catalog(128, 64, 1, true);
  return 0;
}
