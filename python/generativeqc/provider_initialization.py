"""Immutable pending provider-initialization intent for bounded native preparation.

An allowance is a reservation request, not context/provider qualification or a
measured peak. The native physical-reference owner captures its actual identity
later. Omitting intent preserves the existing default path.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass

_MAX_ALLOWANCE = min((1 << 63) - 1, (1 << (8 * struct.calcsize("P"))) - 1)


@dataclass(frozen=True, slots=True)
class ProviderInitializationIntent:
    """Request a finite byte reservation without supplying any identity evidence.

    ``allowance_bytes`` is positive and representable by native ``size_t`` and
    signed-64-bit capacity arithmetic. Acceptance still requires native domain,
    budget, accounting-basis and actual-owner checks. No general allowance or
    peak bound follows from constructing this value.
    """

    allowance_bytes: int

    def __post_init__(self) -> None:
        if type(self.allowance_bytes) is not int:
            raise TypeError(
                "provider initialization allowance must be an integer byte count"
            )
        if not 0 < self.allowance_bytes <= _MAX_ALLOWANCE:
            raise ValueError(
                "provider initialization allowance is outside finite native capacity"
            )


def provider_initialization_owner_pending() -> bool:
    """Report this interpreter's one managed optional transaction slot.

    This does not describe arbitrary C ABI users or other interpreters.
    """
    from ._provider_initialization_lifetime import has_pending_owner

    return has_pending_owner()


def close_provider_initialization_owner() -> None:
    """Explicitly close the retained owner in its original ambient context.

    A failed status raises with the original owner still retained. There is no
    force-clear, context override, replacement generation or automatic retry.
    """
    from ._provider_initialization_lifetime import close_retained_owner

    close_retained_owner()
