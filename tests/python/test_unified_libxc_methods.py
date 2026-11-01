"""Semantic selection and independent physical-point gates for Libxc recipes."""

from __future__ import annotations

from dataclasses import replace
from fractions import Fraction
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pytest
from generativeqc.ks import (
    _native_semilocal_code,
    ks_coefficients,
    native_xc_functional_code,
    resolve_ks_method,
    scf_domain_for_method,
)
from generativeqc_compiler.method import resolve_method
from generativeqc_compiler.xc.automatic_semilocal import (
    automatic_components,
    automatic_functional_name,
)
from test_libxc_automatic_point_native import (
    _evaluate,
    _inputs,
    _oracle,
    backend,  # noqa: F401 -- shared CPU/Slurm fixture
    native_point,  # noqa: F401 -- compiled production point fixture
)

if TYPE_CHECKING:
    import ctypes

    from conftest import NativeCxx
    from generativeqc_compiler.xc.automatic_point import AutomaticPointSource


@pytest.mark.parametrize("name", ("PBE0", "R2SCAN", "MN15", "RSCAN0", "R2SCAN0"))
@pytest.mark.parametrize("spin", ("rks", "uks"))
def test_libxc_and_named_selectors_share_the_execution_contract(
    name: str, spin: str
) -> None:
    ordinary, _ = resolve_ks_method(f"{name}-{spin}")
    imported, _ = resolve_ks_method(f"libxc-{spin}:{name}")
    assert ordinary.semantic_payload() == imported.semantic_payload()
    assert native_xc_functional_code(f"{name}-{spin}") == native_xc_functional_code(
        f"libxc-{spin}:{name}"
    )
    assert scf_domain_for_method(ordinary) == scf_domain_for_method(imported)
    assert ks_coefficients(ordinary) == ks_coefficients(imported)
    assert _native_semilocal_code(
        replace(imported, identifier="renamed")
    ) == _native_semilocal_code(ordinary)


@pytest.mark.parametrize("component", ("GGA_X_PBE", "GGA_X_B88", "MGGA_X_RSCAN"))
def test_atomic_libxc_selectors_are_available_without_a_curated_exclusion(
    component: str,
) -> None:
    method, _ = resolve_ks_method(f"libxc:{component}")
    assert method.primitives[0].functional.components == ((component, Fraction(1)),)
    assert native_xc_functional_code(f"libxc:{component}") == _native_semilocal_code(
        method
    )
    assert scf_domain_for_method(f"libxc:{component}") == scf_domain_for_method(method)


def test_composition_registry_reproduces_from_method_recipes() -> None:
    from generativeqc_compiler.xc._generated_semilocal_compositions import COMPOSITIONS

    from tools.generate_semilocal_compositions import registrations

    assert registrations() == COMPOSITIONS
    assert (
        resolve_method("RSCAN0").semantic_payload()
        != resolve_method("R2SCAN0").semantic_payload()
    )


COMPOSED_NAMES = tuple(
    automatic_functional_name(native_xc_functional_code(f"{name}-rks"))
    for name in ("RSCAN0", "R2SCAN0")
)


@pytest.mark.parametrize("native_point", COMPOSED_NAMES, indirect=True)
@pytest.mark.parametrize("spin", (0, 1))
def test_composed_physical_points_match_independent_libxc(
    native_point: tuple[AutomaticPointSource, ctypes.CDLL, Path],  # noqa: F811
    spin: int,
) -> None:
    """Cover vacuum, spin tails, FHC, and interior derivatives for each sum."""
    point, library, _ = native_point
    components = automatic_components(point.name)
    inputs = _inputs(components[0][0], spin)
    actual = _evaluate(library, inputs)
    assert np.all(actual[:, 11] == 1)
    observed = (
        actual[:, :11]
        if spin
        else np.column_stack(
            (
                actual[:, 0],
                actual[:, 1:3].mean(axis=1),
                0.5 * (actual[:, 3:6] + actual[:, 6:9]),
                actual[:, 9:11].mean(axis=1),
            )
        )
    )
    expected = sum(
        float(weight) * _oracle(component, inputs, spin)
        for component, weight in components
    )
    np.testing.assert_allclose(observed, expected, atol=2e-11, rtol=2e-9)
    interior = inputs[10:]
    direction = (
        np.random.default_rng(11212026).normal(size=interior.shape)
        * np.abs(interior)
        * 0.1
    )
    if spin == 0:
        direction[:, 1] = direction[:, 0]
        direction[:, 5:8] = direction[:, 2:5]
        direction[:, 9] = direction[:, 8]
    step = 1e-5
    finite_difference = sum(
        float(weight)
        * (
            _oracle(component, interior + step * direction, spin)[:, 0]
            - _oracle(component, interior - step * direction, spin)[:, 0]
        )
        / (2 * step)
        for component, weight in components
    )
    coefficients = np.column_stack((actual[10:, 1:9], 2 * actual[10:, 9:11]))
    np.testing.assert_allclose(
        np.sum(coefficients * direction, axis=1),
        finite_difference,
        atol=2e-10,
        rtol=2e-7,
    )


def test_composed_force_reservation_matches_the_complete_exchange_key(
    tmp_path: Path, native_cxx: NativeCxx
) -> None:
    """Both the allocator and prepared owner require the packaged EXX derivative."""
    import subprocess

    root = Path(__file__).resolve().parents[2]
    code = native_xc_functional_code("rscan0-rks")
    source = tmp_path / "reservation.cpp"
    source.write_text(f"""
#include <cassert>
#include "dft/cuda_stationary_binding.hpp"
namespace generativeqc::dft {{
bool packaged = false;
const CudaStationaryBinding* cuda_stationary_binding(const CudaStationaryModel& model) noexcept {{
  static const CudaStationaryBinding binding;
  return packaged && model.functional == {code}U && model.full_range_exchange == 0.25 &&
         model.domain_version == 4 && model.partition_iterations == 3 &&
         model.semilocal_exchange_scale == 1.0 && model.semilocal_correlation_scale == 1.0
         ? &binding : nullptr;
}}
}}
int main() {{
  using namespace generativeqc::dft;
  for (auto spin : {{1U, 2U}}) {{
    packaged = false;
    assert(automatic_stationary_derivative_order({code}U, 4, spin) == 0);
    packaged = true;
    assert(automatic_stationary_derivative_order({code}U, 4, spin) == 1);
    assert(automatic_stationary_derivative_order({code}U, 5, spin) == 0);
  }}
}}
""")
    executable = tmp_path / "reservation"
    native_cxx.build_executable(
        [source],
        executable,
        compile_args=("-std=c++20", f"-I{root / 'src'}", f"-I{root / 'include'}"),
    )
    subprocess.run([str(executable)], check=True)


def test_generated_recipe_metadata_coexists_with_atomic_grid(
    tmp_path: Path, required_native_cxx: NativeCxx
) -> None:
    """Recipe metadata must not shadow the independent atomic-grid namespace."""
    import subprocess

    from generativeqc_compiler.xc.atomic_grid_native import emit_atomic_grid_header

    root = Path(__file__).resolve().parents[2]
    (tmp_path / "generated_atomic_grid.hpp").write_text(emit_atomic_grid_header())
    source = tmp_path / "grid_recipe.cpp"
    source.write_text(
        """
#include <cassert>
#include "dft/global_hybrid_composition.hpp"
#include "dft/grid.cpp"
int main() {
  const auto grid = generativeqc::dft::default_grid_spec();
  assert(grid.version == 4 && grid.radial_points == 3);
  for (unsigned z = 1; z < std::size(generativeqc::generated::atomic_grid::xi); ++z)
    assert(grid.element_radii[z] == generativeqc::generated::atomic_grid::bragg[z]);
}
"""
    )
    executable = tmp_path / "grid_recipe"
    required_native_cxx.build_executable(
        [source],
        executable,
        compile_args=(
            "-std=c++20",
            f"-I{root / 'src'}",
            f"-I{root / 'include'}",
            f"-I{tmp_path}",
        ),
    )
    subprocess.run([str(executable)], check=True)
