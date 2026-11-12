"""Closed Gaussian ECP projectors lowered through the canonical scalar Graph.

This is the power-two residual-radial specialization of ``ecp_projector``.
Cartesian raising differentiates the same seed as the primal operator. Native
execution supplies bounded radial leaves and contracts nuclear derivatives;
neither a second AD engine nor an independent runtime algebra is introduced.
"""

import functools
import math
import re
from fractions import Fraction
from pathlib import Path

from .ecp_policy import GAUSSIAN_MAX_SCRATCH_DOUBLES
from .expr import Expr, Graph
from .scalar_c import ScalarCEmitter
from .shell_spec import cartesian_components


def gaussian_ecp_specs() -> tuple[tuple[int, int, int, bool], ...]:
    """Enumerate s--f pairs and every shift reachable by two derivatives."""
    return tuple(
        (first, second, channel, scaled)
        for first in range(6)
        for second in range(min(first, 8 - first) + 1)
        for channel in (-1, 0, 1, 2)
        for scaled in ((False,) if channel == -1 else (False, True))
    )


def gaussian_ecp_name(spec: tuple[int, int, int, bool]) -> str:
    """Return a deterministic, installation-independent kernel symbol."""
    first, second, channel, scaled = spec
    label = "local" if channel == -1 else f"projector{channel}"
    return f"ecp_gaussian_{first}_{second}_{label}_{int(scaled)}"


def gaussian_ecp_support_cpp() -> str:
    """Emit the bounded external leaves and Cartesian derivative schedule.

    The seed's radial leaves are scaled modified spherical Bessels divided by
    argument**l. The positive regular series avoids cancellation at the origin;
    upward recurrence is used only beyond 50, above every required order.
    Derivative terms implement the Gaussian raising/lowering identity used by
    ``gaussian_roots``; contraction and physical-atom scatter remain native.
    """
    return r"""
inline bool ecp_gaussian_frontier(double alpha, double beta, double zeta,
    const double* centers, unsigned order, double* radial) {
  const double argument = (2 * alpha * beta / (alpha + beta + zeta)) *
      std::hypot(centers[0], centers[1], centers[2]) *
      std::hypot(centers[3], centers[4], centers[5]);
  if (!std::isfinite(argument) || order > 10)
    throw std::invalid_argument("Gaussian ECP radial frontier outside bounded domain");
  if (argument > 50) {
    const double tail = std::exp(-2 * argument);
    double previous = -std::expm1(-2 * argument) / (2 * argument);
    radial[0] = previous;
    double current = ((argument - 1) + (argument + 1) * tail) /
        (2 * argument * argument);
    if (order) radial[1] = current / argument;
    double denominator = argument;
    for (unsigned index = 1; index < order; ++index) {
      const double following = previous - (2 * index + 1) * current / argument;
      previous = current;
      current = following;
      denominator *= argument;
      radial[index + 1] = following / denominator;
    }
    return true;
  }
  for (unsigned index = 0; index <= order; ++index) {
    double term = 1;
    for (unsigned factor = 1; factor <= 2 * index + 1; factor += 2) term /= factor;
    double sum = term, correction = 0;
    bool converged = false;
    for (unsigned iteration = 1; iteration <= 512; ++iteration) {
      term *= argument * argument / (2 * iteration * (2 * index + 2 * iteration + 1));
      const double adjusted = term - correction;
      const double next = sum + adjusted;
      correction = (next - sum) - adjusted;
      sum = next;
      if (term <= 2e-17 * sum) { converged = true; break; }
    }
    if (!converged || !std::isfinite(sum))
      throw std::runtime_error("Gaussian ECP radial series did not converge");
    radial[index] = sum;
  }
  return false;
}
using EcpGaussianPowers = std::array<unsigned, 6>;
struct EcpGaussianMonomial { EcpGaussianPowers powers; double weight; };
inline unsigned ecp_gaussian_differentiate(
    const EcpGaussianMonomial* terms, unsigned count, unsigned axis,
    double alpha, double beta, EcpGaussianMonomial* output) {
  unsigned written = 0;
  for (unsigned index = 0; index < count; ++index) {
    const auto& term = terms[index];
    auto raised = term;
    ++raised.powers[axis];
    raised.weight *= 2 * (axis < 3 ? alpha : beta);
    output[written++] = raised;
    if (term.powers[axis]) {
      auto lowered = term;
      --lowered.powers[axis];
      lowered.weight *= -static_cast<double>(term.powers[axis]);
      output[written++] = lowered;
    }
  }
  return written;
}
template<class Getter>
inline std::array<double, 43> ecp_gaussian_component_jet(
    const EcpGaussianPowers& powers, unsigned order, double alpha, double beta,
    Getter&& get) {
  std::array<double, 43> result{};
  result[0] = get(powers);
  const EcpGaussianMonomial base{powers, 1};
  const auto sum = [&](const EcpGaussianMonomial* terms, unsigned count) {
    double value = 0, correction = 0;
    for (unsigned index = 0; index < count; ++index) {
      const double adjusted = terms[index].weight * get(terms[index].powers) - correction;
      const double next = value + adjusted;
      correction = (next - value) - adjusted;
      value = next;
    }
    return value;
  };
  if (order) for (unsigned axis = 0; axis < 6; ++axis) {
    EcpGaussianMonomial first[2];
    const unsigned count = ecp_gaussian_differentiate(&base, 1, axis, alpha, beta, first);
    result[1 + axis] = sum(first, count);
    if (order == 2) for (unsigned other = axis; other < 6; ++other) {
      EcpGaussianMonomial second[4];
      const unsigned size = ecp_gaussian_differentiate(first, count, other, alpha, beta, second);
      const double entry = sum(second, size);
      result[7 + 6 * axis + other] = entry;
      result[7 + 6 * other + axis] = entry;
    }
  }
  return result;
}
"""


def gaussian_ecp_block(
    first: int, second: int, channel: int, scaled: bool
) -> tuple[ScalarCEmitter, tuple[Expr, ...]]:
    """Build unnormalized Cartesian pairs from the analytic angular projector.

    The regular external leaf is F_l(y), y = |2ab A B / (a+b+z)|**2,
    with dF_l/dy = F_(l+1)/2. Above the scaled-Bessel boundary the leaf
    is exp(-sqrt(y)) F_l(y); its exponential is absorbed into a stable
    relative-radius Gaussian. Only that branch differentiates the radii,
    so coincident centers never evaluate singular radius derivatives.
    """
    if (first, second, channel, scaled) not in gaussian_ecp_specs():
        raise ValueError("unsupported Gaussian ECP Cartesian block")
    graph = Graph()
    alpha, beta, zeta = (graph.variable(name) for name in ("alpha", "beta", "zeta"))
    centers = tuple(graph.variable(f"center{index}") for index in range(6))
    first_square = graph.sum(value * value for value in centers[:3])
    second_square = graph.sum(value * value for value in centers[3:])
    dot = graph.sum(
        left * right for left, right in zip(centers[:3], centers[3:], strict=True)
    )
    total = alpha + beta + zeta
    scale = 2 * alpha * beta / total
    radial = tuple(graph.variable(f"radial{index}") for index in range(11))
    square = scale * scale * first_square * second_square
    argument = None
    if channel == -1:
        difference = graph.sum(
            (left - right).pow(2)
            for left, right in zip(centers[:3], centers[3:], strict=True)
        )
        exponent = (
            -(
                alpha * beta * difference
                + alpha * zeta * first_square
                + beta * zeta * second_square
            )
            / total
        )
        angular_factor = graph.constant(1)
    else:
        angular_factor = (
            (
                graph.constant(1),
                scale * dot,
                scale * scale * (3 * dot * dot - first_square * second_square) / 2,
            )[channel]
            * (2 * channel + 1)
            * radial[channel]
        )
        if scaled:
            first_radius = first_square.pow(Fraction(1, 2))
            second_radius = second_square.pow(Fraction(1, 2))
            argument = scale * first_radius * second_radius
            exponent = (
                -(
                    alpha * zeta * first_square
                    + beta * zeta * second_square
                    + alpha * beta * (first_radius - second_radius).pow(2)
                )
                / total
            )
        else:
            exponent = (
                -(
                    alpha * (beta + zeta) * first_square
                    + beta * (alpha + zeta) * second_square
                )
                / total
            )
    seed = (
        (graph.approximate_constant(math.pi) / total).pow(Fraction(3, 2))
        * graph.exponential(exponent)
        * angular_factor
    )
    leaves = []
    for center in centers:
        square_derivative = graph.differentiate(square, center)
        if scaled:
            assert argument is not None
            argument_derivative = graph.differentiate(argument, center)
        else:
            argument_derivative = None
        leaves.append(
            {
                f"radial{order}": radial[order + 1] * square_derivative / 2
                - (
                    radial[order] * argument_derivative
                    if argument_derivative is not None
                    else 0
                )
                for order in range(10)
            }
            if channel != -1
            else {}
        )

    @functools.cache
    def derivative(identifier: int, axis: int) -> Expr:
        result = graph.differentiate(
            Expr(graph, identifier), centers[axis], leaves[axis]
        )
        if len(graph.nodes) > 2_000_000:
            raise MemoryError("Gaussian ECP block exceeds its Graph node budget")
        return result

    @functools.cache
    def raised(powers: tuple[int, ...]) -> Expr:
        if not any(powers):
            return seed
        axis = next(index for index, power in enumerate(powers) if power)
        previous = list(powers)
        previous[axis] -= 1
        value = derivative(raised(tuple(previous)).identifier, axis)
        if previous[axis]:
            lower = previous.copy()
            lower[axis] -= 1
            value += previous[axis] * raised(tuple(lower))
        return value / (2 * (alpha if axis < 3 else beta))

    roots = tuple(
        raised(tuple(label.count(axis) for label in labels for axis in "xyz"))
        for labels in (
            (left, right)
            for left in cartesian_components(first)
            for right in cartesian_components(second)
        )
    )
    variables = {f"center{index}": f"centers[{index}]" for index in range(6)}
    variables.update({f"radial{index}": f"radial[{index}]" for index in range(11)})
    emitter = ScalarCEmitter(graph, variables)
    emitter.emit(roots)
    return emitter, roots


def emit_gaussian_ecp_sources(directory: Path) -> None:
    """Write bounded optimization units without changing Graph arithmetic.

    Large straight-line functions overwhelm optimizer resources even at O0.
    Stages preserve the original assignment/evaluation order, use caller-owned
    scratch, and bound each optimization unit to 512 scalar assignments. Scratch
    is allocated once by the native endpoint, not once per primitive/channel.
    """
    directory.mkdir(parents=True, exist_ok=True)
    arguments = (
        "double alpha, double beta, double zeta, const double* centers, "
        "const double* radial, double* scratch, double* out"
    )
    declarations, dispatch = [], []
    maximum_scratch = 0
    for spec in gaussian_ecp_specs():
        name = gaussian_ecp_name(spec)
        emitter, roots = gaussian_ecp_block(*spec)
        assignments = []
        for line in emitter.lines:
            match = re.match(r"\s*const double v(\d+) =", line)
            if match is None or int(match[1]) != len(assignments):
                raise ValueError(
                    "Gaussian ECP lowering requires ordered scalar assignments"
                )
            assignment = re.sub(r"const double v(\d+) =", r"scratch[\1] =", line)
            assignments.append(re.sub(r"\bv(\d+)\b", r"scratch[\1]", assignment))
        maximum_scratch = max(maximum_scratch, len(assignments))
        if maximum_scratch > GAUSSIAN_MAX_SCRATCH_DOUBLES:
            raise ValueError(
                "Gaussian ECP lowering exceeds the scratch capacity policy"
            )
        prototype = f"void {name}({arguments})"
        declarations.append(prototype + ";")
        lines = [
            "#include <cmath>",
            "#if defined(_MSC_VER)",
            "#define GQC_ECP_NOINLINE __declspec(noinline)",
            "#else",
            "#define GQC_ECP_NOINLINE __attribute__((noinline))",
            "#endif",
            "namespace generativeqc::generated {",
        ]
        calls = []
        for begin in range(0, len(assignments), 512):
            stage = f"{name}_stage_{begin // 512}"
            lines += [
                f"GQC_ECP_NOINLINE static void {stage}({arguments}) {{",
                *assignments[begin : begin + 512],
                "}",
            ]
            calls.append(f"{stage}(alpha, beta, zeta, centers, radial, scratch, out);")
        lines += [prototype + " {", *calls]
        lines += [
            f"out[{index}] = "
            + re.sub(r"\bv(\d+)\b", r"scratch[\1]", emitter.reference(root))
            + ";"
            for index, root in enumerate(roots)
        ]
        lines += ["}", "}", "#undef GQC_ECP_NOINLINE", ""]
        path = directory / f"{name}.cpp"
        text = "\n".join(lines)
        if not path.exists() or path.read_text() != text:
            path.write_text(text)
        first, second, channel, scaled = spec
        code = (first * 6 + second) * 8 + (channel + 1) * 2 + int(scaled)
        dispatch.append(f"case {code}: return &{name};")
        if channel == -1:
            dispatch.append(f"case {code + 1}: return &{name};")
    header = "\n".join(
        [
            "#pragma once",
            "#include <cstddef>",
            "#include <array>",
            "#include <cmath>",
            "#include <stdexcept>",
            "namespace generativeqc::generated {",
            *declarations,
            (
                "using EcpGaussianKernel = void (*)(double, double, double, "
                "const double*, const double*, double*, double*);"
            ),
            f"inline constexpr std::size_t ecp_gaussian_scratch_doubles = {maximum_scratch};",
            (
                "inline EcpGaussianKernel ecp_gaussian_kernel(unsigned first, "
                "unsigned second, int channel, bool scaled) {"
            ),
            "switch ((first * 6 + second) * 8 + (channel + 1) * 2 + unsigned(scaled)) {",
            *dispatch,
            "default: return nullptr;",
            "}",
            "}",
            gaussian_ecp_support_cpp(),
            "}",
            "",
        ]
    )
    path = directory / "generated_ecp_gaussian.hpp"
    if not path.exists() or path.read_text() != header:
        path.write_text(header)
