"""Compatibility import for DFT-owned native semilocal execution lookup."""

from generativeqc_compiler.dft.native_semilocal import (
    device_feature_ingredients,
    legacy_grid_xc_selector,
    native_semilocal_record,
)

__all__ = [
    "device_feature_ingredients",
    "legacy_grid_xc_selector",
    "native_semilocal_record",
]
