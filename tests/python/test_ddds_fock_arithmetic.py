"""Check lowered coefficient work independently of quartet numerical oracles."""

from __future__ import annotations

import subprocess
from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING

import pytest
from generativeqc_compiler.integral.cuda_schedule import PairOrientation, PairStorage
from generativeqc_compiler.integral.production_emission import emit_production_shard
from generativeqc_compiler.integral.production_profile import resolve_production_profile

if TYPE_CHECKING:
    from conftest import NativeCxx

ROOT = Path(__file__).resolve().parents[2]
MANIFEST = ROOT / "python/generativeqc_compiler/integral/production_shell_classes.json"


def test_sm120_ddds_materializes_smaller_value_pair_with_unrolling() -> None:
    """Cache d-s terms without changing the independent force/lane schedule."""
    profile = resolve_production_profile(MANIFEST, "sm_120")
    selection = next(item for item in profile.selections if item.spec.name == "ddds")
    assert selection.schedule.pair_orientation == PairOrientation.SWAPPED
    assert selection.schedule.pair_storage == PairStorage.RECOMPUTED
    assert selection.fock_schedule is not None
    assert selection.fock_schedule.pair_orientation == PairOrientation.CANONICAL
    assert selection.fock_schedule.pair_storage == PairStorage.MATERIALIZED
    assert selection.fock_schedule.unroll_pair_terms
    assert selection.fock_schedule.block_threads == 128
    assert selection.fock_schedule.tasks_per_warp == 2
    assert selection.fock_schedule.shared_coulomb


def test_sm120_dppp_materializes_smaller_value_pair() -> None:
    """Halve the cached table, not the source term or Coulomb-product count."""
    profile = resolve_production_profile(MANIFEST, "sm_120")
    selection = next(item for item in profile.selections if item.spec.name == "dppp")
    assert selection.schedule.pair_orientation == PairOrientation.SWAPPED
    assert selection.schedule.pair_storage == PairStorage.MATERIALIZED
    assert selection.fock_schedule is not None
    assert selection.fock_schedule.pair_orientation == PairOrientation.CANONICAL
    assert selection.fock_schedule.pair_storage == PairStorage.MATERIALIZED
    assert selection.fock_schedule.unroll_pair_terms
    assert selection.fock_schedule.block_threads == 128
    assert selection.fock_schedule.tasks_per_warp == 4
    assert selection.fock_schedule.shared_coulomb


@pytest.mark.parametrize(
    ("shell_class", "orientation", "storage", "expected_first", "expected_second"),
    (
        ("ddds", PairOrientation.SWAPPED, PairStorage.RECOMPUTED, 64, 4),
        ("ddds", PairOrientation.CANONICAL, PairStorage.MATERIALIZED, 16, 4),
        ("ddds", PairOrientation.SWAPPED, PairStorage.MATERIALIZED, 16, 4),
        ("dppp", PairOrientation.SWAPPED, PairStorage.MATERIALIZED, 8, 4),
        ("dppp", PairOrientation.CANONICAL, PairStorage.MATERIALIZED, 8, 4),
    ),
)
def test_emitted_ddds_coefficient_work(
    tmp_path: Path,
    native_cxx: NativeCxx,
    shell_class: str,
    orientation: PairOrientation,
    storage: PairStorage,
    expected_first: int,
    expected_second: int,
) -> None:
    """Execute the actual emitted loops for every Cartesian component.

    Recurrence arithmetic is stubbed only to count its evaluations; scientific
    acceptance separately compares the real production bundle against Libcint.
    Coulomb products per retained component must remain unchanged.
    Unique stub state labels also expose the original contraction traversal.
    """
    profile = resolve_production_profile(MANIFEST, "sm_120")
    selection = next(
        item for item in profile.selections if item.spec.name == shell_class
    )
    first_order = 4 if shell_class == "ddds" else 3
    first_subsets = 1 << first_order
    contractions = first_subsets * 4
    components = 216 if shell_class == "ddds" else 162
    class_name = shell_class.capitalize()
    assert selection.fock_schedule is not None
    selection = replace(
        selection,
        fock_schedule=replace(
            selection.fock_schedule, pair_orientation=orientation, pair_storage=storage
        ),
    )
    source = emit_production_shard((selection,))
    start = source.index(f"double generated_{shell_class}_component_value(")
    worker = source[start : source.index("\n}", start) + 2]
    driver = tmp_path / "work.cpp"
    driver.write_text(
        f"""
#include <cassert>
#include <cstdint>
unsigned first_calls, second_calls, coulomb_calls;
unsigned coulomb_states[{contractions}];
constexpr unsigned generated_{shell_class}_d_axes[6][2]{{
    {{0, 0}}, {{0, 1}}, {{0, 2}}, {{1, 1}}, {{1, 2}}, {{2, 2}}}};
struct Generated{class_name}PrimitiveGeometry {{
  double pair_shifts[4][3]{{}};
  double inverse_two_p = 1, inverse_two_q = 1, prefactor = 1;
}};
struct Generated{class_name}ValueTerm {{ unsigned derivative_state; double coefficient; }};
template<unsigned PairOrder>
Generated{class_name}ValueTerm generated_{shell_class}_pair_value_term(
    const unsigned*, const double*, double, unsigned subset) {{
  if constexpr (PairOrder == {first_order}) ++first_calls;
  else {{ static_assert(PairOrder == 2); ++second_calls; }}
  return {{PairOrder == {first_order} ? 4 * subset : subset, 1}};
}}
unsigned generated_{shell_class}_state_total(unsigned) {{ return 0; }}
template<bool SharedCoulomb>
double generated_{shell_class}_component_coulomb(
    const Generated{class_name}PrimitiveGeometry&, const double*, unsigned state) {{
  coulomb_states[coulomb_calls++] = state;
  return 1;
}}
template<bool SharedCoulomb>
{worker}
int main() {{
  Generated{class_name}PrimitiveGeometry geometry;
  for (unsigned component = 0; component < {components}; ++component) {{
    first_calls = second_calls = coulomb_calls = 0;
    assert(generated_{shell_class}_component_value<true>(component, geometry, nullptr) == {contractions});
    assert(first_calls == {expected_first});
    assert(second_calls == {expected_second});
    assert(coulomb_calls == {contractions});
    for (unsigned position = 0; position < {contractions}; ++position) {{
      assert(coulomb_states[position] ==
          {f"4 * (position % {first_subsets}) + position / {first_subsets}" if orientation == PairOrientation.SWAPPED else "position"});
    }}
  }}
}}
"""
    )
    executable = tmp_path / "work"
    native_cxx.build_executable(
        (driver,), executable, compile_args=("-std=c++20",), compile_timeout=30
    )
    subprocess.run([str(executable)], check=True, timeout=10)
