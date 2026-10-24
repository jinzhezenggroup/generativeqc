"""Preparation bounds distinguish properties, providers, copies and tiny budgets."""

import json
import os
import subprocess
import typing
from pathlib import Path

import pytest


def test_preparation_shapes_and_resource_policy_envelopes(
    tmp_path: typing.Any, native_cxx: object
) -> None:
    source = tmp_path / "budget.cpp"
    source.write_text(
        r"""
#include "scf/df_preparation_budget.hpp"
#include <iostream>
int main() {
  using namespace generativeqc::scf;
  // 96 atoms: the measured 800 Cartesian / 768 public AO workload. This
  // shape-only check must not allocate its multi-gigabyte derivative tensors.
  DfPreparationShape shape{800, 768, 96, 448, 736, 448, 736, false, false};
  const auto energy = df_preparation_storage(shape);
  shape.include_derivatives = true;
  const auto force = df_preparation_storage(shape);
  shape.generated_one_electron = true;
  const auto generated = df_preparation_storage(shape);
  shape.public_nbf = shape.cartesian_nbf;
  shape.generated_one_electron = false;
  const auto cartesian = df_preparation_storage(shape);
  shape.cartesian_nbf = std::numeric_limits<std::size_t>::max();
  const auto overflow = df_preparation_storage(shape);
  const auto explicit_force =
      resolve_df_budget({24, 60, 5, 2, 6, true}, {}, 17);
  const auto explicit_energy =
      resolve_df_budget({24, 60, 5, 2, 6, false}, {}, 17);
  const auto small =
      resolve_df_budget({12, 24, 3, 1, 4, true}, {}, 0);
  const auto large =
      resolve_df_budget({300, 700, 80, 8, 8, true}, {}, 0);
  const auto small_repeat =
      resolve_df_budget({12, 24, 3, 1, 4, true}, {}, 0);
  const auto constrained =
      resolve_df_budget({300, 700, 80, 8, 8, true},
                        {400U << 20, 8ULL << 30, true}, 0);
  const auto roomy =
      resolve_df_budget({300, 700, 80, 8, 8, true},
                        {8ULL << 30, 16ULL << 30, true}, 0);
  const auto impossible =
      resolve_df_budget({24, 60, 5, 2, 6, true}, {}, 1);
  std::cout << "[" << energy.peak_bytes << "," << force.peak_bytes << ","
            << generated.peak_bytes << "," << cartesian.peak_bytes << ","
            << force.retained_bytes << "," << overflow.peak_bytes << ","
            << explicit_force.total_bytes << "," << explicit_force.value_bytes << ","
            << explicit_force.response_bytes << "," << explicit_energy.value_bytes << ","
            << explicit_energy.response_bytes << "," << small.total_bytes << ","
            << large.total_bytes << "," << small_repeat.total_bytes << ","
            << constrained.total_bytes << "," << constrained.reserved_headroom_bytes << ","
            << roomy.total_bytes << "," << roomy.reserved_headroom_bytes << ","
            << impossible.feasible << "," << impossible.value_bytes << ","
            << impossible.response_bytes << "]\n";
}
"""
    )
    root = Path(__file__).resolve().parents[2]
    executable = tmp_path / "budget"
    native_cxx.build_executable(
        [source],
        executable,
        compile_args=("-std=c++20", "-I" + str(root / "src")),
    )
    values = json.loads(subprocess.check_output([str(executable)], text=True))
    energy, force, generated, cartesian, retained, overflow = values[:6]
    assert energy < 1 << 30
    assert generated < 1 << 30
    # Native input and transformed output both remain live during conversion.
    assert force > 8 * 2 * 288 * (800**2 + 768**2) > 5 << 30
    assert retained > 8 * 2 * 288 * 768**2
    assert cartesian > force  # Identity conversion still returns a deep copy.
    assert force - energy > 8 * 2 * 288 * (800**2 + 768**2)
    assert generated - energy == 2 * 8 * 288  # Nuclear derivatives only.
    assert overflow == 2**64 - 1
    (
        explicit_total,
        explicit_value,
        explicit_response,
        energy_value,
        energy_response,
        small,
        large,
        small_repeat,
        constrained,
        constrained_headroom,
        roomy,
        roomy_headroom,
        impossible,
        impossible_value,
        impossible_response,
    ) = values[6:]
    assert explicit_total == 17
    assert explicit_value + explicit_response == explicit_total
    assert 0 < explicit_response < explicit_total
    assert energy_value == 17 and energy_response == 0
    assert small == small_repeat
    assert 32 << 20 <= small < large <= 1 << 30
    assert constrained <= (400 << 20)
    assert constrained_headroom == 200 << 20  # Actual half-free fallback reservation.
    assert roomy >= constrained
    assert roomy_headroom >= 256 << 20
    assert [impossible, impossible_value, impossible_response] == [0, 1, 0]


def test_constrained_headroom_reports_the_actual_reservation(
    tmp_path: Path, native_cxx: object
) -> None:
    source = tmp_path / "headroom.cpp"
    source.write_text(r"""
#include "scf/df_preparation_budget.hpp"
int main() {
  using namespace generativeqc::scf;
  const DfBudgetWorkload shape{300,700,80,8,8,true};
  const auto tight = resolve_df_budget(shape,{400ULL<<20,8ULL<<30,true},0);
  if (tight.reserved_headroom_bytes > tight.observed_free_bytes) return 1;
  if (tight.total_bytes != (150ULL<<20)) return 2;
  if (tight.reserved_headroom_bytes != (200ULL<<20)) return 3;
  for (std::size_t free : {std::size_t(0),std::size_t(1),std::size_t(2),std::size_t(7)}) {
    const auto budget = resolve_df_budget(shape,{free,8ULL<<30,true},0);
    if (budget.reserved_headroom_bytes > free) return 4;
    if (budget.total_bytes > free-budget.reserved_headroom_bytes) return 5;
  }
  return 0;
}
""")
    output = tmp_path / "headroom"
    root = Path(__file__).resolve().parents[2]
    native_cxx.build_executable(
        [source],
        output,
        compile_args=("-std=c++20", "-I" + str(root / "src")),
    )
    subprocess.run([str(output)], check=True)


def test_live_auto_budget_can_retain_large_values_without_widening_caps(
    tmp_path: Path, native_cxx: object
) -> None:
    """The no-probe safety ceiling must not force roomy devices to stream."""
    source = tmp_path / "resident_budget.cpp"
    source.write_text(r"""
#include "scf/df_preparation_budget.hpp"
int main() {
  using namespace generativeqc::scf;
  const DfResourceEnvelope roomy{30ULL<<30,32ULL<<30,true};
  for (bool forces : {false,true}) {
    const DfBudgetWorkload shape{768,768,96,1,6,forces};
    const auto live = resolve_df_budget(shape,roomy,0);
    const auto fallback = resolve_df_budget(shape,{},0);
    const auto tensor = 768ULL*768ULL*768ULL*sizeof(double);
    if (!live.feasible || live.value_bytes < 4*tensor) return 1;
    if (live.total_bytes != live.value_bytes+live.response_bytes) return 2;
    const auto available = roomy.free_bytes-live.reserved_headroom_bytes;
    if (live.total_bytes > available-available/4) return 3;
    if (fallback.total_bytes > (1ULL<<30)) return 4;
    if (fallback != resolve_df_budget(shape,{},0)) return 5;
    for (std::size_t cap : {std::size_t(1),std::size_t(32ULL<<20),
                            std::size_t(1ULL<<30),std::size_t(2ULL<<30)}) {
      const auto bounded = resolve_df_budget(shape,roomy,cap);
      if (bounded.total_bytes != cap) return 6;
      if (bounded.value_bytes+bounded.response_bytes != cap) return 7;
    }
    const auto tight = resolve_df_budget(shape,{400ULL<<20,8ULL<<30,true},0);
    if (tight.total_bytes != (150ULL<<20)) return 8;
  }
  const auto huge = std::numeric_limits<std::size_t>::max();
  const auto saturated = resolve_df_budget({huge,huge,huge,huge,12,true},roomy,0);
  if (!saturated.feasible || saturated.total_bytes > roomy.free_bytes) return 9;
  if (saturated.value_bytes+saturated.response_bytes != saturated.total_bytes) return 10;
}
""")
    executable = tmp_path / "resident_budget"
    root = Path(__file__).resolve().parents[2]
    native_cxx.build_executable(
        [source],
        executable,
        compile_args=("-std=c++20", "-I" + str(root / "src")),
    )
    subprocess.run([str(executable)], check=True)


def test_roomy_live_budget_admits_the_batch_resident_value_owner(
    tmp_path: Path, native_cxx: object
) -> None:
    """A roomy device must not stream a full resident batch for lack of budget."""
    source = tmp_path / "resident_batch_budget.cpp"
    source.write_text(r"""
#include "scf/df_preparation_budget.hpp"
int main() {
  using namespace generativeqc::scf;
  const DfBudgetWorkload work{96, 96, 12, 4, 8, true};
  const auto floor = df_resident_value_admission_floor(work);
  const auto roomy = resolve_df_budget(work, {30ULL << 30, 32ULL << 30, true}, 0);
  const auto tight = resolve_df_budget(work, {400ULL << 20, 8ULL << 30, true}, 0);
  if (!roomy.feasible || roomy.value_bytes < floor) return 1;
  // The same shape on a constrained live envelope retains the streamed
  // workload target instead of borrowing the response owner for residency.
  if (tight.value_bytes >= floor || tight.total_bytes >= roomy.total_bytes) return 2;
  if (roomy.value_bytes + roomy.response_bytes != roomy.total_bytes) return 3;
  return 0;
}
""")
    executable = tmp_path / "resident_batch_budget"
    root = Path(__file__).resolve().parents[2]
    native_cxx.build_executable(
        [source],
        executable,
        compile_args=(
            "-std=c++20",
            "-Wall",
            "-Wextra",
            "-Werror",
            "-I" + str(root / "src"),
        ),
    )
    subprocess.run([str(executable)], check=True)


def test_single_packed_value_owner_has_distinct_capacity_and_identity(
    tmp_path: Path, native_cxx: object
) -> None:
    """A single fitted owner cannot silently reserve or advertise raw storage."""
    source = tmp_path / "single_packed.cpp"
    source.write_text(r"""
#include "scf/df_value_storage.hpp"
#include <cstdlib>
int main() {
  using namespace generativeqc::scf;
  const auto full = df_packed_value_capacity(1,768,3712,160,8);
  const auto single = df_packed_value_capacity(1,768,3712,160,8,false);
  if (single.factor_bytes != 8769110016ULL ||
      single.factor_bytes != full.factor_bytes ||
      single.scratch_bytes != full.scratch_bytes) return 1;
  if (setenv("GENERATIVEQC_DF_VALUE_STORAGE","packed-single",1)) return 2;
  const auto selected = requested_df_pair_storage();
  if (!df_packed_pairs(selected) || df_retains_packed_raw(selected) ||
      selected == DfPairStorage::SymmetricLower) return 3;
  if (setenv("GENERATIVEQC_DF_VALUE_STORAGE","auto",1)) return 4;
  if (requested_df_pair_storage_request() != DfPairStorageRequest::Automatic ||
      requested_df_pair_storage() != DfPairStorage::Dense) return 5;
  if (df_packed_pairs(static_cast<DfPairStorage>(123))) return 6;
  return 0;
}
""")
    executable = tmp_path / "single_packed"
    root = Path(__file__).resolve().parents[2]
    native_cxx.build_executable(
        [source],
        executable,
        compile_args=(
            "-std=c++20",
            "-Wall",
            "-Wextra",
            "-Werror",
            "-I" + str(root / "src"),
        ),
    )
    subprocess.run([str(executable)], check=True)


def test_prepared_storage_request_is_distinct_from_resolved_layout(
    tmp_path: Path, native_cxx: object
) -> None:
    """Changing auto to an explicit layout invalidates even equal-sized owners."""
    root = Path(__file__).resolve().parents[2]
    source = tmp_path / "prepared_storage_identity.cpp"
    source.write_text(r"""
#include "scf/fock_prepared.hpp"
int main() {
  using namespace generativeqc::scf;
  FockExecutionVariant automatic;
  automatic.df_pair_storage = DfPairStorage::SymmetricLowerSingle;
  auto explicit_single = automatic;
  explicit_single.df_pair_storage_request = DfPairStorageRequest::SymmetricLowerSingle;
  if (automatic == explicit_single) return 1;
  automatic.df_pair_storage = DfPairStorage::Dense;
  auto explicit_dense = automatic;
  explicit_dense.df_pair_storage_request = DfPairStorageRequest::Dense;
  if (automatic == explicit_dense) return 2;
  auto changed_provider = automatic;
  changed_provider.df_value_mapping = 1;
  if (automatic == changed_provider) return 3;
  return 0;
}
""")
    executable = tmp_path / "prepared_storage_identity"
    native_cxx.build_executable(
        [source],
        executable,
        compile_args=(
            "-std=c++20",
            "-I" + str(root / "src"),
            "-I" + str(root / "include"),
        ),
    )
    subprocess.run([str(executable)], check=True)


def test_single_packed_96_atom_plan_keeps_values_when_occupied_scratch_does_not_fit(
    tmp_path: Path, native_cxx: object
) -> None:
    """A 96-atom default allowance admits B without forcing raw regeneration."""
    root = Path(__file__).resolve().parents[2]
    # CI passes the binary directory of its CPU build. Local preset/out-of-tree
    # builds can supply the same variable; never borrow another preset's output.
    binary_dir = Path(os.environ.get("GENERATIVEQC_BUILD_DIR", root / "build"))
    generated = binary_dir / "generated"
    if not (generated / "generated_df_exchange_schedule.hpp").exists():
        pytest.skip(f"requires generated DF source schedule in {generated}")
    source = tmp_path / "single_packed_plan.cpp"
    source.write_text(r"""
#include <iostream>
#include "src/scf/density_fitting.cpp"
int main() {
  using namespace generativeqc::scf;
  const auto default_budget = plan_packed_density_fitting_tiles(
      1,768,3712,160,13685173124ULL,0,160,false);
  if (!default_budget.stores_full_three_center ||
      default_budget.value_storage.pairs != DfPairStorage::SymmetricLowerSingle ||
      default_budget.value_storage.rank_capacity != 0 ||
      default_budget.peak_workspace_bytes > 13685173124ULL) return 1;
  const auto more_values = plan_packed_density_fitting_tiles(
      1,768,3712,160,16421977600ULL,0,160,false);
  if (more_values.value_storage.rank_capacity != 160 ||
      more_values.automatic_rhf_rank != 160 ||
      more_values.peak_workspace_bytes > 16421977600ULL) return 2;
  try {
    (void)plan_packed_density_fitting_tiles(1,768,3712,160,1,0,160,false);
    return 3;
  } catch (const DensityFittingBudgetError&) {
  }
  const auto narrow = plan_packed_density_fitting_tiles(1,1,256,0,0,0,0,false);
  if (narrow.auxiliary_tile != 256) return 4;
  try {
    (void)plan_packed_density_fitting_tiles(
        1,1,256,0,narrow.peak_workspace_bytes-1,0,0,false);
    return 5;
  } catch (const DensityFittingBudgetError&) {
  }

  const auto auto_cross = plan_requested_density_fitting_tiles(
      DfPairStorageRequest::Automatic,1,768,3712,160,160,13685173124ULL,0,true,160);
  if (!auto_cross.stores_full_three_center ||
      auto_cross.value_storage.pairs != DfPairStorage::SymmetricLowerSingle) return 6;
  const auto explicit_dense = plan_requested_density_fitting_tiles(
      DfPairStorageRequest::Dense,1,768,3712,160,160,13685173124ULL,0,true,160);
  if (explicit_dense.value_storage.pairs != DfPairStorage::Dense ||
      explicit_dense.stores_full_three_center) return 7;
  const auto auto_resident = plan_requested_density_fitting_tiles(
      DfPairStorageRequest::Automatic,1,96,96,24,24,1ULL<<30,0,true,24);
  if (!auto_resident.stores_full_three_center ||
      auto_resident.value_storage.pairs != DfPairStorage::Dense) return 8;
  const auto auto_uhf = plan_requested_density_fitting_tiles(
      DfPairStorageRequest::Automatic,1,768,3712,160,160,13685173124ULL,0,true,0);
  if (auto_uhf.value_storage.pairs != DfPairStorage::Dense) return 9;
  const auto method_owned = plan_requested_density_fitting_tiles(
      DfPairStorageRequest::Automatic,1,768,3712,768,160,13685173124ULL,0,true,0,true);
  if (!method_owned.stores_full_three_center || method_owned.automatic_rhf_rank != 0 ||
      method_owned.value_storage.pairs != DfPairStorage::SymmetricLowerSingle ||
      method_owned.peak_workspace_bytes > 13685173124ULL) return 10;
  const auto method_dense = plan_requested_density_fitting_tiles(
      DfPairStorageRequest::Dense,1,768,3712,768,160,13685173124ULL,0,true,0,true);
  if (method_dense.value_storage.pairs != DfPairStorage::Dense ||
      method_dense.stores_full_three_center) return 11;
  const auto method_bounded = plan_requested_density_fitting_tiles(
      DfPairStorageRequest::Automatic,1,768,3712,768,160,4ULL<<30,0,true,0,true);
  if (method_bounded.value_storage.pairs != DfPairStorage::Dense ||
      method_bounded.stores_full_three_center ||
      method_bounded.peak_workspace_bytes > (4ULL<<30)) return 12;
  const auto method_small = plan_requested_density_fitting_tiles(
      DfPairStorageRequest::Automatic,1,96,96,96,24,1ULL<<30,0,true,0,true);
  if (!method_small.stores_full_three_center ||
      method_small.value_storage.pairs != DfPairStorage::Dense ||
      method_small.automatic_rhf_rank != 0) return 13;
  const auto method_batch = plan_requested_density_fitting_tiles(
      DfPairStorageRequest::Automatic,2,768,3712,768,160,13685173124ULL,0,true,0,true);
  if (method_batch.value_storage.pairs != DfPairStorage::Dense) return 14;
  const auto split = resolve_df_budget(
      {768,3712,96,1,0,true}, {24ULL<<30,32ULL<<30,true}, 0);
  const auto resident_split = resolve_method_owned_df_resident_budget(
      split,768,3712,160,100ULL<<20);
  const auto resident_values = plan_requested_density_fitting_tiles(
      DfPairStorageRequest::Automatic,1,768,3712,768,160,
      resident_split.value_bytes,100ULL<<20,true,0,true);
  // Complete-U admission may borrow fractional slack, but never the absolute
  // live headroom or the shape-derived complete response reservation.
  const auto response_floor = 64ULL*1024*1024 + 2ULL*(100ULL<<20) +
      sizeof(double)*(4ULL*3712*3712 + 69ULL*768*768 + 2ULL*3712 +
                      2ULL*3712*160*160);
  if (resident_split.value_bytes + resident_split.response_bytes != resident_split.total_bytes ||
      resident_split.total_bytes > split.observed_free_bytes-split.reserved_headroom_bytes ||
      resident_split.reserved_headroom_bytes != split.reserved_headroom_bytes ||
      resident_split.response_bytes < response_floor ||
      !resident_values.stores_full_three_center ||
      resident_values.value_storage.rank_capacity != 160 ||
      resident_values.value_storage.pairs != DfPairStorage::SymmetricLowerSingle) return 15;
  for (const auto unchanged : {
      resolve_df_budget({768,3712,96,1,0,true}, {}, 0),
      resolve_df_budget({768,3712,96,1,0,true}, {24ULL<<30,32ULL<<30,true}, 4ULL<<30),
      resolve_df_budget({768,3712,96,1,0,true}, {8ULL<<30,32ULL<<30,true}, 0)}) {
    if (resolve_method_owned_df_resident_budget(unchanged,768,3712,160,100ULL<<20)
        != unchanged) return 16;
  }
  if (resolve_method_owned_df_resident_budget(split,768,3712,0,100ULL<<20) != split)
    return 17;
  // #2411: only an explicitly validated restricted J-only method may
  // rebalance a rank-zero packed owner. No occupation is inferred.
  const auto j_only_split = resolve_method_owned_df_resident_budget(
      split,768,3712,0,100ULL<<20,true);
  const auto j_only_values = plan_requested_density_fitting_tiles(
      DfPairStorageRequest::Automatic,1,768,3712,768,0,
      j_only_split.value_bytes,100ULL<<20,true,0,true);
  if (!j_only_values.stores_full_three_center ||
      j_only_values.value_storage.pairs != DfPairStorage::SymmetricLowerSingle ||
      j_only_values.value_storage.rank_capacity != 0 ||
      j_only_split.response_bytes < split.total_bytes/5 ||
      j_only_split.total_bytes != split.total_bytes) return 19;
  const auto j_only_explicit = resolve_df_budget(
      {768,3712,96,1,0,true},{24ULL<<30,32ULL<<30,true},4ULL<<30);
  if (resolve_method_owned_df_resident_budget(
          j_only_explicit,768,3712,0,100ULL<<20,true) != j_only_explicit) return 20;
  // Rank-zero eligibility cannot bypass the physical-source, singleton,
  // explicit-layout, capacity, or live-resource boundaries.
  const auto nonphysical = plan_requested_density_fitting_tiles(
      DfPairStorageRequest::Automatic,1,768,3712,768,0,
      j_only_split.value_bytes,100ULL<<20,false,0,true);
  if (nonphysical.value_storage.pairs != DfPairStorage::Dense) return 21;
  // Give a batch enough capacity for packed values, so exclusion cannot pass
  // merely because packing would have failed its memory admission anyway.
  const auto batch_packed = plan_packed_density_fitting_tiles(
      2,768,3712,0,0,0,0,false);
  const auto batch_excluded = plan_requested_density_fitting_tiles(
      DfPairStorageRequest::Automatic,2,768,3712,768,0,
      batch_packed.peak_workspace_bytes,0,true,0,true);
  if (batch_excluded.value_storage.pairs != DfPairStorage::Dense ||
      batch_excluded.stores_full_three_center) return 26;
  const auto j_dense = plan_requested_density_fitting_tiles(
      DfPairStorageRequest::Dense,1,768,3712,768,0,
      j_only_split.value_bytes,100ULL<<20,true,0,true);
  if (j_dense.value_storage.pairs != DfPairStorage::Dense) return 22;
  const auto j_small = plan_requested_density_fitting_tiles(
      DfPairStorageRequest::Automatic,1,96,96,96,0,1ULL<<30,0,true,0,true);
  if (!j_small.stores_full_three_center ||
      j_small.value_storage.pairs != DfPairStorage::Dense) return 23;
  const auto j_bounded = plan_requested_density_fitting_tiles(
      DfPairStorageRequest::Automatic,1,768,3712,768,0,4ULL<<30,0,true,0,true);
  if (j_bounded.stores_full_three_center ||
      j_bounded.value_storage.pairs != DfPairStorage::Dense ||
      j_bounded.peak_workspace_bytes > (4ULL<<30)) return 24;
  for (const auto unchanged : {
      resolve_df_budget({768,3712,96,1,0,true}, {}, 0),
      resolve_df_budget({768,3712,96,1,0,true}, {8ULL<<30,32ULL<<30,true}, 0),
      resolve_df_budget({768,3712,96,1,0,false}, {24ULL<<30,32ULL<<30,true}, 0)}) {
    if (resolve_method_owned_df_resident_budget(
            unchanged,768,3712,0,100ULL<<20,true) != unchanged) return 25;
  }
  const auto force_ready_split = resolve_method_owned_df_resident_budget(
      resolve_df_budget({768,3712,96,1,0,true},{30ULL<<30,32ULL<<30,true},0),
      768,3712,160,100ULL<<20);
  const auto force_ready_values = plan_requested_density_fitting_tiles(
      DfPairStorageRequest::Automatic,1,768,3712,768,160,
      force_ready_split.value_bytes,100ULL<<20,true,0,true);
  if (force_ready_values.value_storage.rank_capacity != 160 ||
      force_ready_values.automatic_rhf_rank != 0) return 18;
  // Diagnostic response caps are part of owner identity, even with a zero
  // public total request. Automatic resident promotion must preserve them.
  auto explicit_response = split;
  explicit_response.response_override_bytes = explicit_response.response_bytes;
  if (resolve_method_owned_df_resident_budget(
          explicit_response,768,3712,160,100ULL<<20) != explicit_response) return 27;
  // Setup-only metric scratch and lazily created SCF state never coexist.
  // The real wide basis must fit the original live envelope, without changing
  // free memory, its headroom, or the complete response allocation contract.
  const auto wide_budget = resolve_df_budget(
      {1088,3712,96,1,0,true},{31ULL<<30,32ULL<<30,true},0);
  const auto wide_split = resolve_method_owned_df_resident_budget(
      wide_budget,1088,3712,0,100ULL<<20,true);
  const auto wide_values = plan_requested_density_fitting_tiles(
      DfPairStorageRequest::Automatic,1,1088,3712,1088,0,
      wide_split.value_bytes,100ULL<<20,true,0,true);
  if (wide_split == wide_budget ||
      wide_split.value_bytes + wide_split.response_bytes != wide_budget.total_bytes ||
      wide_split.reserved_headroom_bytes != wide_budget.reserved_headroom_bytes ||
      !wide_values.stores_full_three_center ||
      wide_values.value_storage.pairs != DfPairStorage::SymmetricLowerSingle ||
      wide_values.peak_workspace_bytes > wide_split.value_bytes) return 28;
  auto wide_explicit = wide_budget;
  wide_explicit.response_override_bytes = wide_explicit.response_bytes;
  if (resolve_method_owned_df_resident_budget(
          wide_explicit,1088,3712,0,100ULL<<20,true) != wide_explicit) return 29;
  const auto tight_wide = resolve_df_budget(
      {1088,3712,96,1,0,true},{23ULL<<30,32ULL<<30,true},0);
  if (resolve_method_owned_df_resident_budget(
          tight_wide,1088,3712,0,100ULL<<20,true) != tight_wide) return 30;
  // Persistent force arenas/modules reduce observed free memory on a moved
  // geometry. Reclaim only the minimum packed allowance from the optional
  // fractional margin, while retaining the absolute live headroom.
  const auto rebuilt_budget = resolve_df_budget(
      {1088,3712,96,1,0,true},{29ULL<<30,32ULL<<30,true},0);
  const auto rebuilt = resolve_method_owned_df_resident_budget(
      rebuilt_budget,1088,3712,0,100ULL<<20,true);
  if (rebuilt.total_bytes <= rebuilt_budget.total_bytes ||
      rebuilt.total_bytes != rebuilt.value_bytes + rebuilt.response_bytes ||
      rebuilt.reserved_headroom_bytes != rebuilt_budget.reserved_headroom_bytes ||
      rebuilt.total_bytes > rebuilt.observed_free_bytes-rebuilt.reserved_headroom_bytes)
    return 31;
  const auto rebuilt_values = plan_packed_density_fitting_tiles(
      1,1088,3712,0,rebuilt.value_bytes,100ULL<<20,0,false);
  if (!rebuilt_values.stores_full_three_center ||
      rebuilt_values.peak_workspace_bytes > rebuilt.value_bytes) return 32;
  try {
    (void)plan_packed_density_fitting_tiles(
        1,1088,3712,0,rebuilt.value_bytes-1,100ULL<<20,0,false);
    return 33;
  } catch (const DensityFittingBudgetError&) {}
  // A moved restricted owner needs both B and complete U. The optional-U
  // planner may otherwise return a successful rank-zero plan, which is safe
  // in memory but causes repeated AO projections throughout force response.
  const auto occupied_rebuild_budget = resolve_df_budget(
      {768,3712,96,1,0,true},{25ULL<<30,32ULL<<30,true},0);
  const auto occupied_rebuild = resolve_method_owned_df_resident_budget(
      occupied_rebuild_budget,768,3712,160,100ULL<<20);
  const auto occupied_values = plan_packed_density_fitting_tiles(
      1,768,3712,160,occupied_rebuild.value_bytes,100ULL<<20,0,false);
  if (occupied_values.value_storage.rank_capacity != 160 ||
      occupied_values.peak_workspace_bytes > occupied_rebuild.value_bytes ||
      occupied_rebuild.total_bytes != occupied_rebuild.value_bytes+occupied_rebuild.response_bytes ||
      occupied_rebuild.reserved_headroom_bytes != occupied_rebuild_budget.reserved_headroom_bytes ||
      occupied_rebuild.total_bytes > occupied_rebuild.observed_free_bytes-
          occupied_rebuild.reserved_headroom_bytes) return 34;
  const auto smaller = plan_packed_density_fitting_tiles(
      1,768,3712,160,occupied_rebuild.value_bytes-1,100ULL<<20,0,false);
  if (smaller.value_storage.rank_capacity == 160) return 35;
  auto capped = occupied_rebuild_budget;
  capped.requested_bytes = capped.total_bytes;
  if (resolve_method_owned_df_resident_budget(capped,768,3712,160,100ULL<<20)
      != capped) return 36;
  capped = occupied_rebuild_budget;
  capped.response_override_bytes = capped.response_bytes;
  if (resolve_method_owned_df_resident_budget(capped,768,3712,160,100ULL<<20)
      != capped) return 37;
  const auto constrained_budget = resolve_df_budget(
      {768,3712,96,1,0,true},{16ULL<<30,32ULL<<30,true},0);
  const auto constrained = resolve_method_owned_df_resident_budget(
      constrained_budget,768,3712,160,100ULL<<20);
  if (constrained.total_bytes > constrained.observed_free_bytes-
      constrained.reserved_headroom_bytes) return 38;
  try {
    const auto constrained_values = plan_packed_density_fitting_tiles(
        1,768,3712,160,constrained.value_bytes,100ULL<<20,0,false);
    if (constrained_values.value_storage.rank_capacity == 160) return 39;
  } catch (const DensityFittingBudgetError&) {}
  return 0;
}
""")
    executable = tmp_path / "single_packed_plan"
    native_cxx.build_executable(
        [source],
        executable,
        compile_args=(
            "-std=c++20",
            "-O0",
            "-ffunction-sections",
            "-fdata-sections",
            "-I" + str(root),
            "-I" + str(root / "include"),
            "-I" + str(root / "src"),
            "-I" + str(generated),
        ),
        link_args=("-Wl,--gc-sections",),
    )
    subprocess.run([str(executable)], check=True)
