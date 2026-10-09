"""Finite numeric ownership for the exact resident AO bitmask producer."""

from dataclasses import dataclass

from generativeqc_compiler.dft.grid import checked_int


@dataclass(frozen=True, slots=True)
class ExactAoMapResources:
    """Charge bitmasks, both offset mirrors and one same-stream AO span.

    The full-capacity collocation arena is already owned by the grid plan.
    Labels/occupancy and derivative order do not change this reservation. Driver
    objects and compiler storage are outside the numeric-capacity contract.
    """

    aos: int
    points: int
    tile_points: int

    def __post_init__(self) -> None:
        for name in ("aos", "points", "tile_points"):
            checked_int(getattr(self, name), name, low=1, high=(1 << 63) - 1)
        checked_int(
            self.numeric_peak_bound_bytes, "AO map bytes", low=1, high=(1 << 63) - 1
        )

    @property
    def tiles(self) -> int:
        return 1 + (self.points - 1) // self.tile_points

    @property
    def numeric_peak_bound_bytes(self) -> int:
        """Match the native producer, including its host offset mirror."""
        return (
            4 * self.tiles * ((self.aos + 31) // 32)
            + 16 * (self.tiles + 1)
            + 8 * self.aos
        )

    def admitted_bytes(self, allowance: int) -> int:
        """Reserve the exact finite owner or zero for the dense fallback.

        Spending the caller's entire optional allowance would reduce the native
        derivative provider's useful workspace without storing another AO.
        """
        checked_int(allowance, "AO map allowance", low=0, high=(1 << 63) - 1)
        return (
            self.numeric_peak_bound_bytes
            if self.numeric_peak_bound_bytes <= allowance
            else 0
        )
