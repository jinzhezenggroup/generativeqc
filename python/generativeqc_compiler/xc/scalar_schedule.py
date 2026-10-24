"""Consumer-owned XC scalar schedules, independent of method and spin.

Bounded power expansion is useful in the stationary derivative program, but a
different rounding path in the self-consistency map can add whole Fock builds.
Keep that distinction at the consumer boundary, not in a functional registry.
"""

from dataclasses import dataclass
from enum import Enum


class XcScalarConsumer(str, Enum):
    """Distinguish a stationary derivative from the iterative SCF map."""

    SCF_POINT = "scf_point"
    STATIONARY_FORCE = "stationary_force"


@dataclass(frozen=True, slots=True)
class XcScalarSchedule:
    """Bounded arithmetic transformations for one scalar consumer/precision."""

    lower_integer_powers: bool = False


def select_xc_scalar_schedule(
    *,
    consumer: XcScalarConsumer,
    scalar_type: str,
) -> XcScalarSchedule:
    """Expand bounded FP64 force powers without modifying any SCF map.

    Method identity and spin are deliberately absent: the shared IR pass acts
    after independent polarized AD and before either spin binding. Float keeps
    its original schedule; changing precision requires separate qualification.
    """
    if not isinstance(consumer, XcScalarConsumer):
        raise TypeError("scalar schedule requires an XC consumer")
    if scalar_type not in ("double", "float"):
        raise ValueError("scalar schedule requires double or float precision")
    return XcScalarSchedule(
        lower_integer_powers=(
            consumer is XcScalarConsumer.STATIONARY_FORCE and scalar_type == "double"
        )
    )
