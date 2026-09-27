"""Method-neutral derivative AOT registry identity and selection."""

from __future__ import annotations

from vibeqc_compiler.integral.derivative_aot_registry import (
    DerivativeAotKey,
    component_group,
    component_groups,
    entry_prefix_for_key,
    make_key,
    radial_inventory_from_payload,
    select_packaged_component_derivative_aot,
    select_packaged_derivative_aot,
)
from vibeqc_compiler.integral.first_derivative_schedule import (
    COMPONENT_LABELS,
    CPU_AOT_SHARDS,
    cpu_aot_symbol,
)
from vibeqc_compiler.integral.range_separation import CoulombKernel
from vibeqc_compiler.integral.rsh_cpu_aot import entry_prefix as legacy_rsh_prefix


def test_registry_identity_covers_backend_radial_shell_and_group() -> None:
    angular = (1, 1, 1, 1)
    full = make_key(CoulombKernel("full_range", 0.0), angular, 0, backend="cpu")
    short = make_key(CoulombKernel("short_range", 0.3), angular, 0, backend="cpu")
    long = make_key(CoulombKernel("long_range", 0.3), angular, 0, backend="cpu")
    cuda_short = make_key(CoulombKernel("short_range", 0.3), angular, 0, backend="cuda")

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


def test_shared_full_range_component_bundle_uses_same_registry() -> None:
    class Symbol:
        pass

    library = type("Library", (), {})()
    for shard in range(CPU_AOT_SHARDS):
        setattr(library, cpu_aot_symbol(shard), Symbol())

    radial = CoulombKernel("full_range", 0.0)
    selected = select_packaged_component_derivative_aot(
        library,
        backend="cpu",
        radial=radial,
    )
    assert selected is not None
    assert selected.key.backend == "cpu"
    assert selected.key.radial == radial
    assert selected.key.component_domain == COMPONENT_LABELS
    assert len(selected.symbols) == CPU_AOT_SHARDS
    assert selected.key.to_payload()["radial"]["family"] == "full_range"

    assert (
        select_packaged_component_derivative_aot(
            library,
            backend="cpu",
            radial=CoulombKernel("short_range", 0.3),
        )
        is None
    )
    assert (
        select_packaged_component_derivative_aot(
            library,
            backend="cuda",
            radial=radial,
        )
        is None
    )


def test_component_bundle_fails_closed_when_one_shard_is_missing() -> None:
    library = type("Library", (), {})()
    for shard in range(CPU_AOT_SHARDS - 1):
        setattr(library, cpu_aot_symbol(shard), object())
    assert (
        select_packaged_component_derivative_aot(
            library,
            backend="cpu",
            radial=CoulombKernel("full_range", 0.0),
        )
        is None
    )


def test_radial_inventory_is_backend_scoped_and_rejects_duplicates() -> None:
    payload = {
        "schema": "vibeqc.derivative-aot.radials.v1",
        "entries": [
            {"backend": "cpu", "family": "short_range", "omega": 0.3},
            {"backend": "cuda", "family": "short_range", "omega": 0.3},
            {"backend": "cpu", "family": "long_range", "omega": 0.3},
        ],
    }
    assert radial_inventory_from_payload(payload, backend="cpu") == (
        CoulombKernel("short_range", 0.3),
        CoulombKernel("long_range", 0.3),
    )
    assert radial_inventory_from_payload(payload, backend="cuda") == (
        CoulombKernel("short_range", 0.3),
    )

    duplicate = {
        **payload,
        "entries": [
            {"backend": "cpu", "family": "short_range", "omega": 0.3},
            {"backend": "cpu", "family": "short_range", "omega": 0.3},
        ],
    }
    import pytest

    with pytest.raises(ValueError, match="duplicate"):
        radial_inventory_from_payload(duplicate, backend="cpu")


def test_radial_inventory_rejects_undeclared_schema_or_entry_shape() -> None:
    import pytest

    with pytest.raises(ValueError, match="schema"):
        radial_inventory_from_payload(
            {"schema": "other", "entries": []}, backend="cpu"
        )
    with pytest.raises(ValueError, match="entry"):
        radial_inventory_from_payload(
            {
                "schema": "vibeqc.derivative-aot.radials.v1",
                "entries": [
                    {
                        "backend": "cpu",
                        "family": "short_range",
                        "omega": 0.3,
                        "method": "WB97M-V",
                    }
                ],
            },
            backend="cpu",
        )
