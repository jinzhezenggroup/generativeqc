"""Explicit build and native execution contracts, independent of profile labels."""

from __future__ import annotations

from dataclasses import dataclass

FOCK_VARIANTS = frozenset({"rys_value", "k_block", "rys_task"})
RYS_TASK_FOCK_CLASS_SETS = {
    "bounded_spd": (
        "psps",
        "ppps",
        "dsss",
        "dpss",
        "dsps",
        "ddss",
        "dsds",
        "dpps",
        "dspp",
    ),
}


@dataclass(frozen=True, slots=True)
class AotPackaging:
    """Select alternative families independently of their runtime preference.

    Alternatives share primary CUDA modules, so even unused families contribute
    to binary size and first use. Compilation is not scientific qualification.
    """

    fock_variants: frozenset[str] = FOCK_VARIANTS
    derivative_shells: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.fock_variants, frozenset) or not all(
            isinstance(name, str) for name in self.fock_variants
        ):
            raise TypeError("fock_variants must be a frozenset of family names")
        if self.fock_variants - FOCK_VARIANTS:
            raise ValueError("unknown optional Fock family")
        if not isinstance(self.derivative_shells, bool):
            raise TypeError("derivative_shells must be a boolean")

    def to_payload(self) -> dict[str, object]:
        """Serialize effective packaging, not the selected GPU or profile name."""
        return {
            "fock_variants": sorted(self.fock_variants),
            "derivative_shells": self.derivative_shells,
        }


@dataclass(frozen=True, slots=True)
class ForceExecutionPolicy:
    """Declare native queue selection and mandatory bounded-force ownership.

    The native owner applies the workload/resource crossover and fail-closed
    coverage checks. Tuning/display metadata must not authorize either policy.
    """

    small_work_angular_queue: bool = True
    require_bounded_spd_coverage: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.small_work_angular_queue, bool) or not isinstance(
            self.require_bounded_spd_coverage, bool
        ):
            raise TypeError("force execution policies must be booleans")

    def to_payload(self) -> dict[str, bool]:
        """Expose the execution contract separately from compiled availability."""
        return {
            "small_work_angular_queue": self.small_work_angular_queue,
            "require_bounded_spd_coverage": self.require_bounded_spd_coverage,
        }
