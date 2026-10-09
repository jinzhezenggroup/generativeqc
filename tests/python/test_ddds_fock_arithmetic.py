"""Check lowered coefficient work independently of quartet numerical oracles."""

from __future__ import annotations

import json
import struct
import subprocess
from dataclasses import replace
from decimal import Decimal, localcontext
from pathlib import Path
from typing import TYPE_CHECKING

import pytest
from generativeqc_compiler.integral.cuda_schedule import (
    PairOrientation,
    PairStorage,
    ScheduleKind,
)
from generativeqc_compiler.integral.production_emission import (
    _streaming_fock_schedule,
    emit_production_shard,
)
from generativeqc_compiler.integral.production_profile import (
    _schedule_from_payload,
    resolve_production_profile,
)
from generativeqc_compiler.integral.tuning.policy import ScheduleTrial, schedule_payload

if TYPE_CHECKING:
    from conftest import NativeCxx

ROOT = Path(__file__).resolve().parents[2]
MANIFEST = ROOT / "python/generativeqc_compiler/integral/production_shell_classes.json"


def test_sm120_ddds_materializes_smaller_value_pair_with_rolled_loops() -> None:
    """Cache d-s terms without changing the independent force/lane schedule."""
    profile = resolve_production_profile(MANIFEST, "sm_120")
    selection = next(item for item in profile.selections if item.spec.name == "ddds")
    assert selection.schedule.pair_orientation == PairOrientation.SWAPPED
    assert selection.schedule.pair_storage == PairStorage.RECOMPUTED
    assert not selection.schedule.mixed_pair_products_fp64
    assert selection.fock_schedule is not None
    assert selection.fock_schedule.pair_orientation == PairOrientation.CANONICAL
    assert selection.fock_schedule.pair_storage == PairStorage.MATERIALIZED
    assert not selection.fock_schedule.unroll_pair_terms
    assert selection.fock_schedule.mixed_pair_products_fp64
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


def test_mixed_pair_precision_survives_schedule_roundtrip_and_fallback() -> None:
    profile = resolve_production_profile(MANIFEST, "sm_120")
    selection = next(item for item in profile.selections if item.spec.name == "ddds")
    schedule = selection.fock_schedule
    assert schedule is not None
    assert _schedule_from_payload(schedule_payload(schedule)) == schedule
    ordinary = replace(schedule, mixed_pair_products_fp64=False)
    assert "mixed_pair_products_fp64" not in schedule_payload(ordinary)
    assert _schedule_from_payload(schedule_payload(ordinary)) == ordinary
    for invalid in (1, "false", None):
        payload = {**schedule_payload(ordinary), "mixed_pair_products_fp64": invalid}
        with pytest.raises(ValueError, match="invalid production kernel schedule"):
            _schedule_from_payload(payload)
    # The derived component-lane fallback must not discard the precision choice.
    fallback = replace(
        selection,
        schedule=replace(schedule, tasks_per_warp=8),
        fock_schedule=None,
    )
    derived = _streaming_fock_schedule(fallback)
    assert derived.kind == ScheduleKind.COMPONENT_LANES
    assert derived.mixed_pair_products_fp64
    emitted = emit_production_shard((fallback,))
    start = emitted.index("float generated_ddds_mixed_component_value(")
    worker = emitted[start : emitted.index("\n}", start) + 2]
    assert "const double sign =" in worker
    wide_trial = ScheduleTrial(selection.spec, schedule, profile.target)
    ordinary_trial = replace(wide_trial, schedule=ordinary)
    assert wide_trial.schedule_id != ordinary_trial.schedule_id


def test_thread_schedule_derivation_preserves_mixed_pair_precision(
    tmp_path: Path,
) -> None:
    manifest = json.loads(MANIFEST.read_text())
    row = next(
        row
        for row in manifest["architectures"]["sm_120"]["kernels"]
        if row["shell_class"] == "dsps"
    )
    row.pop("fock_schedule")
    row["schedule"]["mixed_pair_products_fp64"] = True
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest))
    profile = resolve_production_profile(path, "sm_120")
    selection = next(item for item in profile.selections if item.spec.name == "dsps")
    assert selection.fock_schedule is not None
    assert selection.fock_schedule.mixed_pair_products_fp64


@pytest.mark.parametrize("orientation", tuple(PairOrientation))
@pytest.mark.parametrize("storage", tuple(PairStorage))
def test_mixed_ddds_preserves_wide_products_across_pair_schedules(
    tmp_path: Path,
    native_cxx: NativeCxx,
    orientation: PairOrientation,
    storage: PairStorage,
) -> None:
    """Execute the emitted evaluator against exact binary32-input products.

    One nonzero term isolates product precision from the intentional change in
    contraction traversal. The real recurrence is covered by separate oracles.
    """
    profile = resolve_production_profile(MANIFEST, "sm_120")
    selection = next(item for item in profile.selections if item.spec.name == "ddds")
    assert selection.fock_schedule is not None
    selection = replace(
        selection,
        fock_schedule=replace(
            selection.fock_schedule, pair_orientation=orientation, pair_storage=storage
        ),
    )
    source = emit_production_shard((selection,))
    start = source.index("float generated_ddds_mixed_component_value(")
    worker = source[start : source.index("\n}", start) + 2]
    assert "const double sign =" in worker
    assert "const float sign =" not in worker
    assert "#pragma unroll 1" in worker
    driver = tmp_path / "mixed-product.cpp"
    driver.write_text(
        """
#include <cassert>
#include <cmath>
#include <cstdlib>
#include <iostream>
float first_coefficient, second_coefficient, coulomb_value;
constexpr unsigned generated_ddds_d_axes[6][2]{
    {0, 0}, {0, 1}, {0, 2}, {1, 1}, {1, 2}, {2, 2}};
struct GeneratedDddsMixedPrimitiveGeometry {
  float pair_shifts[4][3]{};
  float inverse_two_p = 1, inverse_two_q = 1, prefactor = 1;
};
struct GeneratedDddsMixedValueTerm { unsigned derivative_state; float coefficient; };
template<unsigned PairOrder>
GeneratedDddsMixedValueTerm generated_ddds_mixed_pair_value_term(
    const unsigned*, const float*, float, unsigned subset) {
  return {0, subset == 0 ? (PairOrder == 4 ? first_coefficient : second_coefficient) : 0};
}
unsigned generated_ddds_state_total(unsigned) { return 0; }
template<bool SharedCoulomb>
float generated_ddds_mixed_component_coulomb(
    const GeneratedDddsMixedPrimitiveGeometry&, const float*, unsigned) {
  return coulomb_value;
}
template<bool SharedCoulomb>
"""
        + worker
        + """
int main(int, char** arguments) {
  first_coefficient = std::strtof(arguments[1], nullptr);
  second_coefficient = std::strtof(arguments[2], nullptr);
  coulomb_value = std::strtof(arguments[3], nullptr);
  GeneratedDddsMixedPrimitiveGeometry geometry;
  const float first = generated_ddds_mixed_component_value<true>(0, geometry, nullptr);
  for (unsigned component = 1; component < 216; ++component)
    assert(generated_ddds_mixed_component_value<true>(component, geometry, nullptr) == first);
  std::cout << std::hexfloat << first;
}
"""
    )
    executable = tmp_path / "mixed-product"
    native_cxx.build_executable(
        (driver,),
        executable,
        compile_args=("-std=c++20", "-O2", "-ffp-contract=off"),
    )

    def binary32(value: float) -> float:
        return struct.unpack("f", struct.pack("f", value))[0]

    cases = (
        (1.9785739183425903, 0.11637665331363678, 0.4990340769290924),
        (0.7930068969726562, 0.3068036437034607, 0.9509599208831787),
        (0.11389146745204926, 0.5465744137763977, 1.8747915029525757),
        (1.8955563306808472, 0.5539439916610718, 1.2930752038955688),
    )
    for positive in cases:
        for sign in (-1, 1):
            values = (sign * positive[0], *positive[1:])
            assert tuple(map(binary32, values)) == values
            with localcontext() as context:
                context.prec = 100
                first, second, third = map(Decimal.from_float, values)
                expected = binary32(float(first * second * third))
            narrowed = binary32(binary32(values[0] * values[1]) * values[2])
            assert narrowed != expected  # This population detects the old defect.
            completed = subprocess.run(
                [str(executable), *map(repr, values)],
                check=True,
                capture_output=True,
                text=True,
                timeout=10,
            )
            assert float.fromhex(completed.stdout) == expected


def test_custom_canonical_mixed_schedule_keeps_its_existing_precision() -> None:
    profile = resolve_production_profile(MANIFEST, "sm_120")
    selection = next(item for item in profile.selections if item.spec.name == "ddds")
    assert selection.fock_schedule is not None
    selection = replace(
        selection,
        fock_schedule=replace(selection.fock_schedule, mixed_pair_products_fp64=False),
    )
    source = emit_production_shard((selection,))
    start = source.index("float generated_ddds_mixed_component_value(")
    worker = source[start : source.index("\n}", start) + 2]
    assert "const float sign =" in worker
