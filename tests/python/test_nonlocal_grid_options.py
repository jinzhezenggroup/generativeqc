"""Independent quadrature transport, including equal-count nonidentical rules."""

import ctypes
from dataclasses import asdict, replace
from types import SimpleNamespace

import pytest
from generativeqc import Calculator, GridSpec, KsOptions, _native
from generativeqc.ks import native_ks_options, resolve_ks_options

PRIMARY = GridSpec(12, 6, 12)
SECONDARY = GridSpec(8, 6, 12)


def test_nonlocal_grid_requires_a_grid_spec() -> None:
    with pytest.raises(TypeError, match="nonlocal_grid"):
        KsOptions(nonlocal_grid=(8, 6, 12))


def test_nonlocal_grid_payload_keeps_both_prescriptions() -> None:
    options = resolve_ks_options(
        "wb97m-v", KsOptions(grid=PRIMARY, nonlocal_grid=SECONDARY)
    )
    assert options.to_payload()["grid"] == asdict(PRIMARY)
    assert options.to_payload()["nonlocal_grid"] == asdict(SECONDARY)


@pytest.mark.parametrize(
    "secondary", (None, PRIMARY, replace(PRIMARY, element_radii=((1, 1.0),)))
)
def test_equal_grid_preserves_the_legacy_descriptor(secondary: GridSpec | None) -> None:
    options = resolve_ks_options(
        "wb97m-v", KsOptions(grid=PRIMARY, nonlocal_grid=secondary)
    )
    descriptor = native_ks_options(options, library=SimpleNamespace())
    assert descriptor.struct_size == ctypes.sizeof(_native.KsOptionsDescriptor) == 160


@pytest.mark.parametrize("secondary", (SECONDARY, GridSpec(8, 9, 12)))
def test_distinct_grid_has_a_retained_exact_layout(secondary: GridSpec) -> None:
    options = resolve_ks_options(
        "wb97m-v", KsOptions(grid=PRIMARY, nonlocal_grid=secondary)
    )
    descriptor = native_ks_options(options)
    assert (
        descriptor.struct_size
        == ctypes.sizeof(_native.NonlocalGridKsOptionsDescriptor)
        == 200
    )
    extension = descriptor._extension_owner
    assert extension.nonlocal_grid_version == 1
    assert extension.nonlocal_grid_reserved == 0
    actual = extension.nonlocal_grid.contents
    assert (actual.radial_points, actual.angular_polar, actual.angular_azimuth) == (
        secondary.radial_points,
        secondary.angular_polar,
        secondary.angular_azimuth,
    )
    assert actual.struct_size == ctypes.sizeof(_native.KsOptionsDescriptor)
    assert descriptor.radial_points == PRIMARY.radial_points


def test_distinct_grid_copies_its_own_partition_and_radii() -> None:
    secondary = replace(SECONDARY, partition_iterations=2, element_radii=((1, 1.3),))
    options = resolve_ks_options(
        "wb97m-v", KsOptions(grid=PRIMARY, nonlocal_grid=secondary)
    )
    extension = native_ks_options(options)._extension_owner
    assert extension.nonlocal_grid.contents.partition_iterations == 2
    assert extension.nonlocal_grid.contents.element_radii[1] == 1.3


def test_unavailable_independent_grid_capability_fails_closed() -> None:
    options = resolve_ks_options(
        "wb97m-v", KsOptions(grid=PRIMARY, nonlocal_grid=SECONDARY)
    )
    with pytest.raises(NotImplementedError, match="independent VV10 grids"):
        native_ks_options(options, library=SimpleNamespace())


def test_nonlocal_grid_on_semilocal_only_model_is_not_silently_ignored() -> None:
    options = resolve_ks_options(
        "pbe-rks", KsOptions(grid=PRIMARY, nonlocal_grid=PRIMARY)
    )
    with pytest.raises(ValueError, match="requires a nonlocal"):
        native_ks_options(options)


@pytest.mark.parametrize("endpoint", ("hvp", "hessian"))
def test_split_grid_second_order_is_rejected_before_native_work(endpoint: str) -> None:
    calculator = SimpleNamespace(
        _ks_options=KsOptions(grid=PRIMARY, nonlocal_grid=SECONDARY)
    )
    with pytest.raises(NotImplementedError, match="Hessians or HVPs"):
        Calculator._checked_rks_second_order_atoms(
            calculator, (), multiplicity=1, endpoint=endpoint
        )
