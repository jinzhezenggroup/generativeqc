"""Explicit functional-specific exceptions to default-allow Libxc admission.

Automatic semilocal admission is structural. Shared numerical boundary classes
such as zero-spin channels, zero gradients, density tails, and Libxc work-domain
regularization are compiler/runtime policy and must not be encoded as a list of
functional names.

Keep this map empty unless a reproducible defect is genuinely specific to one
functional and cannot be expressed as a generic capability or domain rule.
"""

from types import MappingProxyType

LIBXC_SEMILOCAL_BLACKLIST = MappingProxyType({})


def blacklist_reason(name: str) -> str | None:
    """Return the explicit functional-specific blocker, if any."""
    if not isinstance(name, str) or not name.strip():
        raise ValueError("Libxc functional name must be a nonempty string")
    return LIBXC_SEMILOCAL_BLACKLIST.get(name.upper())


__all__ = ["LIBXC_SEMILOCAL_BLACKLIST", "blacklist_reason"]
