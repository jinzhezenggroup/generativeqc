"""Typed grid/XC extension boundary, independent of functional mathematics.

XC supplies generated scalar source and an exact expression identity. DFT owns
the tile ABI and required feature mask; it never imports the XC compiler.
"""

from dataclasses import dataclass

from .native_semilocal import legacy_grid_xc_selector


@dataclass(frozen=True)
class GridXcConsumer:
    """A built-in or artifact-bound evaluator and its collocation requirements."""

    evaluator: int
    ingredients: tuple[str, ...]
    identity: str = ""

    def __post_init__(self) -> None:
        if self.evaluator not in (0, 1, 2):
            raise ValueError("unknown grid XC evaluator")
        allowed = (("rho",), ("rho", "gradient"), ("rho", "gradient", "tau"))
        if self.ingredients not in allowed:
            raise ValueError("unsupported grid XC ingredients")
        if self.evaluator < 2:
            if self.identity or self.ingredients != allowed[self.evaluator]:
                raise ValueError("invalid built-in grid XC consumer")
        elif len(self.identity) != 64 or any(
            c not in "0123456789abcdef" for c in self.identity
        ):
            raise ValueError("compiled grid XC requires an expression identity")


@dataclass(frozen=True)
class GridXcKernel:
    """Generated point adapter bound to the identity exported by its source."""

    source: str
    consumer: GridXcConsumer

    def __post_init__(self) -> None:
        if (
            not isinstance(self.consumer, GridXcConsumer)
            or self.consumer.evaluator != 2
        ):
            raise ValueError("grid XC extension requires a compiled consumer")
        if not isinstance(self.source, str) or not self.source:
            raise ValueError("grid XC extension requires generated source")


def grid_xc_consumer(
    value: object, compiled_identity: str = "", *, spin: str | None = None
) -> GridXcConsumer | None:
    """Resolve a grid consumer without guessing mathematics from a name."""
    if not isinstance(value, GridXcConsumer):
        # Preserve canonical metadata and spin validation for existing callers;
        # exact generated expressions cross this boundary as typed consumers.
        try:
            evaluator = legacy_grid_xc_selector(value, spin=spin)
        except ValueError:
            return None
        return GridXcConsumer(
            evaluator, ("rho",) if evaluator == 0 else ("rho", "gradient")
        )
    if value.evaluator == 2 and value.identity != compiled_identity:
        return None
    return value


GRID_XC_POINT_TYPE = """struct GridXcPoint {
  double energy{}, rho[2]{}, gradient[2][3]{}, kinetic[2]{};
  bool valid = true;
};
"""


def grid_xc_extension(kernel: GridXcKernel | None = None) -> str:
    """Supply the stable tile coefficient ABI and optional generated evaluator."""
    if kernel is not None:
        if not isinstance(kernel, GridXcKernel):
            raise TypeError("expected a generated GridXcKernel")
        return GRID_XC_POINT_TYPE + kernel.source
    return (
        GRID_XC_POINT_TYPE
        + """
constexpr unsigned grid_custom_xc_mask = 0;
constexpr const char* grid_custom_xc_identity = "";
__device__ inline GridXcPoint grid_custom_xc_point(bool, const double*, I, I) {
  GridXcPoint out;
  out.valid = false;
  return out;
}
"""
    )
