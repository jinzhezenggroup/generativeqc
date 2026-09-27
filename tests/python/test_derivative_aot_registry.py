"""Method-neutral derivative AOT registry identity and selection."""

from __future__ import annotations

from vibeqc_compiler.integral.derivative_aot_registry import (
    DerivativeAotKey,
    component_group,
    component_groups,
    entry_prefix_for_key,
    make_key,
    select_packaged_derivative_aot,
)
from vibeqc_compiler.integral.range_separation import CoulombKernel
from vibeqc_compiler.integral.rsh_cpu_aot import entry_prefix as legacy_rsh_prefix


def test_registry_identity_covers_backend_radial_shell_and_group() -> None:
    angular = (1, 1, 1, 1)
    full = make_key(CoulombKernel("full_range", 0.0), angular, 0, backend="cpu")
    short = make_key(CoulombKernel("short_range", 0.3), angular, 0, backend="cpu")
    long = make_key(CoulombKernel("long_range", 0.3), angular, 0, backend="cpu")
    cuda_short = make_key(
        CoulombKernel("short_range", 0.3), angular, 0, backend="cuda"
    )

    assert len({full.identity, short.identity, long.identity, cuda_short.identity}) == 4
    assert full.to_payload()["derivative_order"] == 1
    assert full.to_payload()["radial"]["family"] == "full_range"
    assert short.to_payload()["radial"]["omega"] == 0.3
    assert short.component_indices == component_groups(angular)[0]


def test_registry_selection_is_operator_based_not_method_named() -> None:
    radial = CoulombKernel("short_range", 0.3)
    angular = (0, 1, 0, 1)
    group_index, selected = component_group(angular, 0)
    key = make_key(radial, angular, group_index, backend="cpu")
    prefix = entry_prefix_for_key(key)

    library = type("Library", (), {})()
    assert (
        select_packaged_derivative_aot(
            library,
            backend="cpu",
            radial=radial,
            angular=angular,
            component=0,
        )
        is None
    )
    setattr(library, f"{prefix}_identity_v2", object())
    resolved = select_packaged_derivative_aot(
        library,
        backend="cpu",
        radial=radial,
        angular=angular,
        component=0,
    )
    assert resolved is not None
    assert resolved.key == key
    assert resolved.component_indices == selected
    assert resolved.entry_prefix == prefix
    assert "wb97" not in prefix.lower()
    assert "pbe" not in prefix.lower()
    assert "b3lyp" not in prefix.lower()


def test_existing_cpu_range_symbol_layout_remains_compatible() -> None:
    radial = CoulombKernel("long_range", 0.3)
    angular = (1, 0, 1, 0)
    key = DerivativeAotKey(
        backend="cpu",
        radial=radial,
        angular=angular,
        group_index=0,
    )
    assert entry_prefix_for_key(key) == legacy_rsh_prefix(radial, angular, 0)


def test_full_range_has_method_neutral_registry_identity() -> None:
    radial = CoulombKernel("full_range", 0.0)
    key = make_key(radial, (0, 0, 0, 0), 0, backend="cpu")
    prefix = entry_prefix_for_key(key)
    assert prefix.startswith("vibeqc_derivative_cpu_d1_full_")
    assert "rsh" not in prefix
