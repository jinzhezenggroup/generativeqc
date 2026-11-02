"""Independent force scheduling preserves science and every descriptor extension."""

import ctypes
from dataclasses import replace

import pytest
from generativeqc import GridSpec, KsOptions, _native
from generativeqc.ks import native_ks_options, resolve_ks_options


@pytest.mark.parametrize("value", [0, -1, 4097, True, 2.0, "1024"])
def test_force_tile_rejects_invalid_bounds(value: object) -> None:
    with pytest.raises(ValueError, match="force_tile_points"):
        KsOptions(force_tile_points=value)


@pytest.mark.parametrize("tile", [1, 1024, 4096])
@pytest.mark.parametrize("precision", [None, (1e-7, 2e-7, 3e-7)])
@pytest.mark.parametrize("separate_grid", [False, True])
def test_force_tile_transport_preserves_scf_and_scientific_controls(
    tile: int, precision: tuple[float, float, float] | None, separate_grid: bool
) -> None:
    original = resolve_ks_options(
        "wb97m-v",
        KsOptions(
            grid=GridSpec(12, 4, 8),
            nonlocal_grid=GridSpec(8, 4, 8) if separate_grid else None,
            tile_points=37,
            _precision_budget=precision,
        ),
    )
    selected = resolve_ks_options("wb97m-v", replace(original, force_tile_points=tile))
    assert original.to_payload().get("force_tile_points") is None
    payload = selected.to_payload()
    assert payload.pop("force_tile_points") == tile
    assert payload == original.to_payload()
    before = native_ks_options(original)
    after = native_ks_options(selected)
    assert before.force_tile_points == 0 and after.force_tile_points == tile
    assert before.tile_points == after.tile_points == 37
    assert before.struct_size == after.struct_size
    for field in (
        "grid_version",
        "radial_points",
        "angular_polar",
        "angular_azimuth",
        "nonlocal_b",
        "nonlocal_c",
        "nonlocal_maximum_bytes",
    ):
        assert getattr(before, field) == getattr(after, field)
    if precision is not None or separate_grid:
        extension = ctypes.cast(
            ctypes.byref(after),
            ctypes.POINTER(_native.ApplicationPrecisionKsOptionsDescriptor),
        ).contents
        assert (
            extension.application_energy_budget,
            extension.application_max_force_budget,
            extension.application_rms_force_budget,
        ) == (precision or (0.0, 0.0, 0.0))
    if separate_grid:
        grid = after._extension_owner.nonlocal_grid.contents
        assert grid.radial_points == 8 and grid.force_tile_points == tile
