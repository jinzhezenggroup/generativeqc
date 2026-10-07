"""Structural bounds for the optional restricted exact-K block contraction."""

from __future__ import annotations

from .shell_spec import ShellClassSpec

PACKED_RESTRICTED_K_BLOCK_MAX_DOUBLES = 32


def packed_restricted_k_block_doubles(spec: ShellClassSpec) -> int:
    """Return the complete forward/transposed raw-K block footprint."""

    first, second, third, fourth = map(len, spec.center_components)
    return 2 * (
        first * third
        + second * third
        + first * fourth
        + second * fourth
    )


def packed_restricted_k_block_eligible(spec: ShellClassSpec) -> bool:
    """Keep the qualification alternative inside one bounded shared footprint."""

    return (
        packed_restricted_k_block_doubles(spec)
        <= PACKED_RESTRICTED_K_BLOCK_MAX_DOUBLES
    )
