"""r2SCAN shares the default grid policy and preserves explicit legacy grids."""

import pytest
from generativeqc import KsOptions
from generativeqc.ks import resolve_ks_options
from generativeqc_compiler.dft.grid import GridPolicy, GridSpec


@pytest.mark.parametrize("method", ["r2scan-rks", "r2scan-uks"])
def test_r2scan_default_uses_treutler(method: str) -> None:
    options = resolve_ks_options(method)
    assert options.grid == GridSpec.treutler(3)


@pytest.mark.parametrize("method", ["r2scan-rks", "r2scan-uks"])
def test_r2scan_does_not_silently_ignore_tight_grid_request(method: str) -> None:
    options = resolve_ks_options(method, KsOptions(grid_accuracy="tight"))
    assert options.grid == GridSpec.treutler(4)


@pytest.mark.parametrize("method", ["r2scan-rks", "r2scan-uks"])
def test_r2scan_explicit_grid_remains_authoritative(method: str) -> None:
    grid = GridSpec(radial_points=32, angular_polar=12, angular_azimuth=24)
    options = resolve_ks_options(method, KsOptions(grid=grid, grid_accuracy="tight"))
    assert options.grid == grid


@pytest.mark.parametrize("method", ["lda-rks", "pbe-rks", "lda-uks", "pbe-uks"])
def test_qualified_families_keep_new_production_defaults(method: str) -> None:
    assert resolve_ks_options(method).grid == GridPolicy().resolve(method)
