"""Compatibility re-export for native semilocal execution metadata."""

from generativeqc_compiler.dft._generated_native_semilocal import (
    SCF_DOMAIN_BY_VERSION,
    SEMILOCAL_FAMILIES,
    SEMILOCAL_FAMILY_BY_CODE,
    SEMILOCAL_FAMILY_CODES,
)

__all__ = [
    "SCF_DOMAIN_BY_VERSION",
    "SEMILOCAL_FAMILIES",
    "SEMILOCAL_FAMILY_BY_CODE",
    "SEMILOCAL_FAMILY_CODES",
]
