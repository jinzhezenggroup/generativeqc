"""Generated atom-adjoint cuts for the recognized Becke partition derivative.

Move exp/log-product pullbacks from incident pairs to the normalization boundary.
The ordinary four-word reverse/gather remains intact: no coefficient expansion,
new allocation, pair pruning or floating-point atomics are introduced. This
qualification-only schedule changes rounding and requires independent gates.
"""

from generativeqc_compiler.integral.expr import Graph
from generativeqc_compiler.integral.scalar_c import ScalarCEmitter
from generativeqc_compiler.xc.becke_partition import (
    BeckePartitionDerivativeOp,
    validate_becke_partition_derivative,
)
from generativeqc_compiler.xc.grid_response_ir import grid_response_primal


def _emit_atom_weight() -> str:
    """Cut the canonical exp JVP at its already normalized primal product."""
    graph = Graph()
    logarithm = graph.variable("logarithm")
    maximum = graph.variable("maximum")
    primal = graph.exponential(logarithm - maximum)
    pullback = graph.variable("seed") * graph.differentiate(primal, logarithm)
    emitter = ScalarCEmitter(graph, {"seed": "seed"})
    for identifier in graph.topological_order((primal,)):
        emitter.names[identifier] = "unused_product_ancestor"
    emitter.names[primal.identifier] = "product"
    emitter.emit((pullback,))
    return "\n".join(
        (
            "GENERATIVEQC_NORMALIZED_HD double atom_weight(double product, double seed) {",
            *emitter.lines,
            f"return {emitter.reference(pullback)};",
            "}",
        )
    )


def _emit_pair_term() -> str:
    """Emit only reachable log-AD pullback nodes, not an unused primal log."""
    graph = Graph()
    factor = graph.variable("factor")
    primal = grid_response_primal(graph, "log", {"p": factor})
    pullback = (
        graph.variable("weight")
        * graph.differentiate(primal, factor)
        * graph.variable("slope")
    )
    emitter = ScalarCEmitter(
        graph, {name: name for name in ("weight", "factor", "slope")}
    )
    emitter.emit((pullback,))
    return "\n".join(
        (
            "GENERATIVEQC_NORMALIZED_HD double pair_term(double weight, double factor, double slope) {",
            *emitter.lines,
            f"return {emitter.reference(pullback)};",
            "}",
        )
    )


_SOURCE = r"""
GENERATIVEQC_NORMALIZED_HD void prepare_atom_weight(Workspace work, size_t point,
    size_t atom) {
  const size_t zeros = work.zero_counts(point)[atom];
  const double product = zeros == 1
      ? normalized_product_value(work.field(4, point)[atom], 0, work.maximum[point])
      : zeros == 0 ? work.field(5, point)[atom] : 0;
  const double seed = work.field(6, point)[atom];
  const double weight = atom_weight(product, seed);
  // Preserve the ordinary arithmetic for overflow or a prematurely underflowed
  // cut. Field 7 is dead until gather; it does not add to the scratch bound.
  work.field(7, point)[atom] = !std::isfinite(weight) ||
      (weight == 0 && product != 0 && seed != 0)
      ? std::numeric_limits<double>::quiet_NaN() : weight;
}

template <class Geometry, class Log>
GENERATIVEQC_NORMALIZED_HD bool pair_reverse_phase(Workspace work, size_t point,
    size_t first, size_t second, Geometry geometry, Log logarithm) {
  const double first_weight = work.field(7, point)[first];
  const double second_weight = work.field(7, point)[second];
  if (!std::isfinite(first_weight) || !std::isfinite(second_weight))
    return generativeqc_grid_phased::pair_reverse_phase(
        work, point, first, second, geometry, logarithm);
  const size_t index = center_pair_index(first, second);
  const double factor = work.pair(2, index, point);
  const double slope = work.pair(3, index, point);
  std::array<double, 4> separation{};
  std::array<double, 2> coefficients{};
  bool valid = true;
  if (slope != 0) {
    separation = geometry.separation(first, second, valid);
    const auto coordinate = geometry.coordinate(work.field(0, point)[first] -
        work.field(0, point)[second], first, second, separation[0]);
    double bar_mu = 0;
    for (size_t side = 0; side < 2; ++side) {
      const size_t atom = side ? second : first;
      const double value = side ? 1 - factor : factor;
      const double weight = side ? second_weight : first_weight;
      const size_t zeros = work.zero_counts(point)[atom];
      if (!zeros) bar_mu += pair_term(side ? -weight : weight, value, slope);
      else if (zeros == 1 && value == 0)
        bar_mu += atom_weight(side ? -weight : weight, slope);
    }
    coefficients = {bar_mu * coordinate[1], bar_mu * coordinate[2]};
  }
  for (size_t word = 0; word < 4; ++word) {
    const double value = pair_pullback_component(coefficients[0], coefficients[1],
                                                 word, separation[word]);
    work.pair(word, index, point) = value;
    valid = valid && std::isfinite(value);
  }
  return valid;
}
"""


def emit_becke_normalized_adjoint(operation: BeckePartitionDerivativeOp) -> str:
    """Require authenticated canonical AD before emitting the new graph cuts.

    The caller emits shared phases first and must finish normalization/weight
    production before reverse, then all reverse pairs before the ordered gather.
    Field 7's alias is safe only at these same-stream phase boundaries.
    """
    validate_becke_partition_derivative(operation)
    return (
        "#if defined(__CUDACC__)\n"
        "#define GENERATIVEQC_NORMALIZED_HD __host__ __device__\n"
        "#else\n#define GENERATIVEQC_NORMALIZED_HD\n#endif\n"
        "namespace generativeqc_grid_normalized {\n"
        "using namespace generativeqc_grid_phased;\n"
        f"// Canonical Becke normalized-adjoint operation: {operation.identity}\n"
        + _emit_atom_weight()
        + "\n"
        + _emit_pair_term()
        + _SOURCE
        + "}\n#undef GENERATIVEQC_NORMALIZED_HD\n"
    )
