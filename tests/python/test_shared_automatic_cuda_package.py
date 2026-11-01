"""Executable sharing preserves scientific identities and exact ABI boundaries."""

from __future__ import annotations

import ctypes
from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING

import pytest
from generativeqc_compiler.method.stationary_native_cuda import (
    emit_native_stationary_profile_cpp,
    emit_native_stationary_profile_cuda,
    native_stationary_package_sources,
    native_stationary_shared_group,
)
from generativeqc_compiler.xc._generated_semilocal_compositions import COMPOSITIONS
from generativeqc_compiler.xc.automatic_cuda_package import (
    _executable,
    automatic_cuda_package,
)
from generativeqc_compiler.xc.automatic_point import emit_automatic_point
from generativeqc_compiler.xc.automatic_semilocal import automatic_functional_code

ROOT = Path(__file__).resolve().parents[2]

if TYPE_CHECKING:
    from conftest import NativeCxx


def _profiles(names: tuple[str, ...]) -> tuple[str, ...]:
    return tuple(
        f"libxc_{automatic_functional_code(name)}_{spin}"
        for name in names
        for spin in ("rks", "uks")
    )


def test_force_consumers_do_not_scale_with_registration_count() -> None:
    """The required ownership boundary is executable ABI, not functional identity."""
    narrow = _profiles(("LDA_C_VWN_4", "GGA_X_APBE", "MGGA_X_LTA"))
    expanded = narrow + _profiles(("LDA_C_PZ_MOD", "GGA_X_RPBE", "MGGA_C_TPSS"))
    small = native_stationary_package_sources(narrow)
    large = native_stationary_package_sources(expanded)
    assert {p for p in small if p.endswith(".cu")} == {
        p for p in large if p.endswith(".cu")
    }
    assert large == native_stationary_package_sources(tuple(reversed(expanded)))
    for first, second in zip(narrow, expanded[len(narrow) :], strict=True):
        assert native_stationary_shared_group(first) == native_stationary_shared_group(
            second
        )
        # A collision in the sharing key must not merge different scientific
        # consumers. This checks generated asset equivalence, with no pinned
        # fingerprint or spelling requirement on production function bodies.
        assert emit_native_stationary_profile_cuda(
            first, staged_point=True
        ) == emit_native_stationary_profile_cuda(second, staged_point=True)


def test_fused_package_choice_is_explicit_and_bounded() -> None:
    profiles = _profiles(("GGA_X_APBE", "GGA_X_RPBE"))
    ordinary = native_stationary_package_sources(profiles)
    selected = native_stationary_package_sources(
        profiles, fused_profiles=(profiles[0],)
    )
    assert f"native_{profiles[0]}.cu" not in ordinary
    assert f"native_{profiles[0]}.cu" in selected
    with pytest.raises(ValueError, match="selected force package"):
        native_stationary_package_sources(profiles, fused_profiles=("pbe_rks",))


def test_real_vwn_aliases_share_only_the_identical_worker() -> None:
    names = tuple(f"LDA_C_VWN_{index}" for index in range(1, 5))
    workers, registrations = automatic_cuda_package(names)
    assert len(registrations) == 4
    # Four different polarized workers and one common U1 worker. The public
    # registration bodies retain their original codes and scientific identity.
    assert len(workers) == 5
    assert len(set(registrations.values())) == 4
    ordinary = [emit_automatic_point(name, backend="cuda") for name in names]
    restricted = [
        emit_automatic_point(name, backend="cuda", spin="unpolarized") for name in names
    ]
    assert (
        len({_executable(point, stationary=True).identity for point in ordinary}) == 4
    )
    assert (
        len({_executable(point, stationary=True).identity for point in restricted}) == 1
    )
    assert len({point.expression_identity for point in restricted}) == 4


@pytest.mark.parametrize(
    "field,value",
    [
        ("domain_version", 999),
        ("outputs", ((),)),
        ("spin", "polarized"),
        ("domain", "a-different-work-policy"),
        ("ingredient_mask", 7),
    ],
)
def test_executable_identity_includes_domain_output_and_spin(
    field: str, value: object
) -> None:
    point = emit_automatic_point("LDA_C_VWN_4", backend="cuda", spin="unpolarized")
    original = _executable(point, stationary=True).identity
    assert (
        original
        != _executable(replace(point, **{field: value}), stationary=True).identity
    )
    assert original != _executable(point, stationary=False).identity


def test_worker_source_and_backend_changes_cannot_borrow_an_executable() -> None:
    point = emit_automatic_point("LDA_C_VWN_4", backend="cuda", spin="unpolarized")
    original = _executable(point, stationary=True).identity
    assert (
        original
        != _executable(
            replace(point, source=point.source + "\n// changed emission\n"),
            stationary=True,
        ).identity
    )
    with pytest.raises(ValueError, match="CUDA point source"):
        _executable(replace(point, backend="cpu"), stationary=True)


def test_similar_formulas_with_different_work_floors_do_not_alias() -> None:
    original, modified = (
        emit_automatic_point(name, backend="cuda", spin="unpolarized")
        for name in ("LDA_C_CHACHIYO", "LDA_C_CHACHIYO_MOD")
    )
    assert (
        _executable(original, stationary=True).identity
        != _executable(modified, stationary=True).identity
    )


@pytest.mark.parametrize("spin", ("rks", "uks"))
@pytest.mark.parametrize(
    "names,fraction",
    (
        (("GGA_X_APBE", "GGA_X_RPBE"), 0.0),
        (
            tuple(
                next(
                    key
                    for key, record in COMPOSITIONS.items()
                    if name in record["methods"]
                )
                for name in ("RSCAN0", "R2SCAN0")
            ),
            0.25,
        ),
    ),
)
def test_shared_host_bindings_preserve_science_and_reuse_callbacks(
    tmp_path: Path,
    native_cxx: NativeCxx,
    names: tuple[str, str],
    fraction: float,
    spin: str,
) -> None:
    """Link generated registrations to one consumer and execute real dispatch."""
    first, second = tuple(
        f"libxc_{automatic_functional_code(name)}_{spin}" for name in names
    )
    group = native_stationary_shared_group(first)
    assert group == native_stationary_shared_group(second)
    (tmp_path / "generated_libxc_semilocal_cuda_registry.hpp").write_text("""
#pragma once
#include "dft/cuda_stationary_binding.hpp"
namespace generativeqc::dft::generated {
const CudaXcPointProgram* automatic_cuda_point_entry(std::uint32_t) noexcept;
}
""")
    source = tmp_path / "bindings.cpp"
    source.write_text(
        emit_native_stationary_profile_cpp(first)
        + emit_native_stationary_profile_cpp(second)
        + f"""
#include <cstring>
namespace generativeqc::dft {{
struct CudaXcPointProgram {{ unsigned code; }};
const CudaXcPointProgram* generated::automatic_cuda_point_entry(std::uint32_t code) noexcept {{
  static CudaXcPointProgram points[]{{{{{automatic_functional_code(names[0])}}}, {{{automatic_functional_code(names[1])}}}}};
  return code == points[0].code ? points : points + 1;
}}
static int metrics(void*, std::uint64_t*, std::size_t) {{ return 0; }}
const CudaStationaryBinding& native_stationary_profile_{group}() {{
  static const auto shared = [] {{ CudaStationaryBinding v; v.metrics = metrics; return v; }}();
  return shared;
}}
extern "C" bool probe() {{
  const auto& a = native_stationary_profile_{first}();
  const auto& b = native_stationary_profile_{second}();
  return a.metrics == b.metrics && a.metrics == metrics && a.point_program != b.point_program &&
         a.model.functional == a.point_program->code && b.model.functional == b.point_program->code &&
         a.model.spin_blocks == {1 if spin == "rks" else 2} &&
         b.model.spin_blocks == {1 if spin == "rks" else 2} &&
         a.model.full_range_exchange == {fraction} && b.model.full_range_exchange == {fraction} &&
         std::strcmp(a.plan_identity, b.plan_identity) != 0 &&
         std::strcmp(a.method_identity, b.method_identity) != 0 &&
         std::strcmp(a.source_identity, b.source_identity) != 0;
}}
}}
"""
    )
    binary = native_cxx.build_shared(
        [source],
        tmp_path / "bindings.so",
        compile_args=("-std=c++20", f"-I{ROOT / 'src'}", f"-I{tmp_path}"),
    )
    loaded = ctypes.CDLL(str(binary))
    loaded.probe.restype = ctypes.c_bool
    assert loaded.probe()
