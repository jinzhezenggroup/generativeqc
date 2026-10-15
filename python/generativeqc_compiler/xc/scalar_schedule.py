"""Qualified XC scalar schedules, independent of functional spelling.

The generic graph lowering is mathematical IR; only evidence-backed
consumer/spin/precision combinations may enable a changed arithmetic order.
"""

from dataclasses import dataclass
from enum import Enum

from ._generated_split_hybrids import CUDA_SMALL_INTEGER_POWER_SCHEDULES


class XcScalarConsumer(str, Enum):
    SCF_POINT = "scf_point"
    STATIONARY_FORCE = "stationary_force"


@dataclass(frozen=True, slots=True)
class XcScalarSchedule:
    lower_integer_powers: bool = False


def select_xc_scalar_schedule(
    functional_code: int,
    *,
    consumer: XcScalarConsumer,
    spin: str,
    scalar_type: str,
) -> XcScalarSchedule:
    """Select an admitted arithmetic schedule; otherwise preserve old ordering."""
    if type(functional_code) is not int or functional_code < 0:
        raise ValueError("scalar schedule requires a nonnegative functional code")
    if not isinstance(consumer, XcScalarConsumer):
        raise TypeError("scalar schedule requires an XC consumer")
    if spin not in ("unpolarized", "polarized"):
        raise ValueError("scalar schedule requires a supported spin mode")
    if scalar_type not in ("double", "float"):
        raise ValueError("scalar schedule requires double or float precision")
    return XcScalarSchedule(
        lower_integer_powers=(
            functional_code,
            consumer.value,
            spin,
            scalar_type,
        )
        in CUDA_SMALL_INTEGER_POWER_SCHEDULES
    )
