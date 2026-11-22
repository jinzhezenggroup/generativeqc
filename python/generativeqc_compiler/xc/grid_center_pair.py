"""One emitted storage contract for stationary and response center geometry."""


def emit_center_pair_layout() -> str:
    """Emit in an ABI namespace unaffected by generated profile isolation."""
    return r"""#ifndef GENERATIVEQC_GRID_CENTER_PAIR_DEFINED
#define GENERATIVEQC_GRID_CENTER_PAIR_DEFINED
namespace generativeqc_grid_pair_geometry {
// Exact triangular geometry storage. The ratio fields are emitted from the same
// local AD graph, not a separately implemented force formula.
struct CenterPair {
  std::array<double, 4> distance;
  std::array<double, 2> ratio;
};
static_assert(sizeof(CenterPair) == 6 * sizeof(double));
GENERATIVEQC_GRID_HD inline size_t center_pair_index(size_t a, size_t b) {
  return a * (a - 1) / 2 + b; // canonical a > b orientation
}
}
#endif"""
