"""Physical DF primitive values, gradients and gradient JVPs from scalar AD.

The raw metric and three-center primals are the existing Hermite/Boys DAGs.
Only one component tile and one nuclear direction are emitted at a time; the
shared native contraction template owns primitive streaming and publication.
"""

from functools import cache
from itertools import product

from .cuda import CudaEmitter
from .df_derivatives import build_df_derivative_ir, build_df_derivative_kernel
from .shell_spec import cartesian_components


@cache
def emit_df_second(
    family: str,
    angular: tuple[int, ...],
    indices: tuple[int, ...],
    *,
    backend: str = "cpu",
) -> str:
    """Emit a bounded Cartesian component tile, including the physical Boys JVP."""
    if backend not in ("cpu", "cuda"):
        raise ValueError("DF second-order backend must be cpu or cuda")
    qualifier = "static" if backend == "cpu" else "__device__ __noinline__"
    integral = build_df_derivative_ir(family, angular)
    components = tuple(product(*(cartesian_components(order) for order in angular)))
    if not indices or len(indices) > 16 or len(set(indices)) != len(indices):
        raise ValueError("DF second-order tiles require one to 16 distinct components")
    if any(index < 0 or index >= len(components) for index in indices):
        raise ValueError("DF second-order component is out of range")
    count = len(angular)
    functions = []
    for packed, index in enumerate(indices):
        kernel = build_df_derivative_kernel(integral, components[index])
        graph = kernel.graph
        leaves = {
            f"center_{center}_{axis}": graph.variable(f"direction_{center}_{axis}")
            for center in range(count)
            for axis in "xyz"
        }
        variable = graph.variable("direction")
        argument_jvp = graph.differentiate(kernel.boys_argument, variable, leaves)
        leaves.update(
            {
                f"boys_{moment}": -graph.variable(f"boys_{moment + 1}") * argument_jvp
                for moment in range(kernel.boys_count)
            }
        )
        gradients = tuple(value for row in kernel.gradients for value in row)
        roots = (kernel.value, *gradients) + tuple(
            graph.differentiate(value, variable, leaves) for value in gradients
        )
        variables = {
            **{f"exponent_{center}": f"input[{center}]" for center in range(count)},
            **{
                f"center_{center}_{axis}": f"input[{count + 3 * center + coordinate}]"
                for center in range(count)
                for coordinate, axis in enumerate("xyz")
            },
            **{
                f"direction_{center}_{axis}": f"input[{4 * count + 3 * center + coordinate}]"
                for center in range(count)
                for coordinate, axis in enumerate("xyz")
            },
        }
        argument = CudaEmitter(graph, variables)
        argument.emit((kernel.boys_argument,))
        moments = kernel.boys_count + 1
        emitter = CudaEmitter(
            graph,
            {
                **variables,
                **{f"boys_{order}": f"boys[{order}]" for order in range(moments)},
            },
        )
        emitter.emit(roots)
        functions.append(
            f"{qualifier} bool component_{packed}(const double* input, double* output) {{\n"
            + "\n".join(argument.lines)
            + f"\n double boys[{moments}]{{}};\n"
            + f" if (!generativeqc::integrals::bounded_range_moments<14>({moments - 1}, "
            + f"{argument.reference(kernel.boys_argument)}, 1, "
            + "generativeqc::integrals::CoulombRange::Full, 0, boys)) return false;\n {\n"
            + "\n".join(emitter.lines)
            + "\n"
            + "\n".join(
                f" output[{position}] = {emitter.reference(value)};"
                for position, value in enumerate(roots)
            )
            + "\n } return true;\n}"
        )
    cases = "\n".join(
        f"case {index}: return component_{index}(input, output);"
        for index in range(len(indices))
    )
    if backend == "cuda":
        return '#include "integrals/range_moments.hpp"\n' + "\n".join(functions)
    return (
        '#include "integrals/first_component_runtime.hpp"\n'
        '#include "integrals/range_moments.hpp"\n'
        + "\n".join(functions)
        + f"""
struct DFSecondProgram {{
  static constexpr std::size_t exponents = {count};
  static constexpr std::size_t inputs = {7 * count};
  static constexpr std::size_t outputs = {1 + 6 * count};
  static constexpr std::size_t components = {len(indices)};
  static bool evaluate(const double* input, std::size_t component, double* output) {{
    switch (component) {{ {cases} default: return false; }}
  }}
}};
extern "C" int generativeqc_df_second_v1(const double* records, std::size_t count,
    std::size_t stride, double* output, std::size_t size) {{
  return generativeqc::integrals::contract_first_components<DFSecondProgram>(
      records, count, stride, output, size);
}}
"""
    )
