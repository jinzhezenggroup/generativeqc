"""Device-pointer lowering of existing integral value/gradient/gradient-JVP AD.

This deliberately bounded schedule streams radial products and component tiles.
Independent primitive evaluations run concurrently, but each component retains
the original ordered FP64 contraction. It introduces no integral recurrence,
derivative rule, host numerical contract, or molecular derivative tensor. FP64
outputs remain on the caller's stream.
"""

from typing import Any

from .df_second import emit_df_second
from .second_derivatives import build_second_derivative_kernel
from .second_derivatives_native import emit_second_derivative_primitive


def emit_resident_second(integral: Any, indices: tuple[int, ...]) -> str:
    """Lower selected raw AO components with unit cotangents before scalar AD."""
    if not indices or len(indices) > 16 or len(set(indices)) != len(indices):
        raise ValueError(
            "resident integral tiles require one to 16 distinct components"
        )
    centers = len(integral.operator.centers)
    exponents = len(integral.signature.shells)
    sources, cases = [], []
    for packed, component in enumerate(indices):
        kernel = build_second_derivative_kernel(integral, (component,))
        source = emit_second_derivative_primitive(
            kernel, backend="cuda", include_first_order=True
        ).replace("generated_second", f"resident_component_{packed}")
        sources.append(source)
        cases.append(
            f"case {packed}: return generativeqc::scf::resident_component_{packed}"
            "::primitive(record, record + "
            f"{exponents}, &unit, direction, output);"
        )
    evaluate = (
        "double unit = 1; switch (component) {" + "".join(cases) + "} return false;"
    )
    return _runtime("\n".join(sources), evaluate, exponents, centers, len(indices))


def emit_resident_df_second(family: str, angular: tuple, indices: tuple) -> str:
    """Use the same physical DF Hermite/Boys gradient-JVP graph as CPU."""
    source = emit_df_second(family, angular, indices, backend="cuda")
    centers = len(angular)
    cases = "".join(
        f"case {component}: return component_{component}(input, output);"
        for component in range(len(indices))
    )
    evaluate = (
        f"double input[{7 * centers}];"
        f"for (int index=0; index<{4 * centers}; ++index) input[index]=record[index];"
        f"for (int index=0; index<{3 * centers}; ++index) input[{4 * centers}+index]=direction[index];"
        "switch(component) {" + cases + "} return false;"
    )
    return _runtime(source, evaluate, centers, centers, len(indices))


def _runtime(
    source: str, evaluate: str, exponents: int, centers: int, components: int
) -> str:
    outputs = 1 + 6 * centers
    stride = exponents + 3 * centers + 1
    return (
        source
        + f"""
#include <cuda_runtime.h>
__device__ bool resident_evaluate(int component, const double* record,
    const double* direction, double* output) {{ {evaluate} }}
__global__ void resident_integral_kernel(const double* records, size_t count,
    const long long* atoms, const double* molecular_direction, double* output, int* status) {{
  const int component = blockIdx.x;
  if (component >= {components}) return;
  __shared__ double terms[{outputs}][64];
  double direction[{3 * centers}];
  for (int center=0; center<{centers}; ++center)
    for (int axis=0; axis<3; ++axis)
      direction[3*center+axis] = molecular_direction[3*atoms[center]+axis];
  double result[{outputs}]{{}};
  for (size_t begin=0; begin<count;) {{
    const size_t active = count-begin < 64 ? count-begin : 64;
    bool valid = true;
    if (threadIdx.x < active) {{
      const double* record=records+(begin+threadIdx.x)*{stride};
      double values[{outputs}]{{}};
      valid = resident_evaluate(component, record, direction, values);
      for (int index=0; index<{outputs}; ++index)
        terms[index][threadIdx.x]=record[{stride - 1}]*values[index];
    }}
    if (__syncthreads_or(!valid)) {{
      if (threadIdx.x == 0) atomicExch(status, 1);
      return;
    }}
    // Parallelize expensive primitive AD, not the contraction: preserve radial
    // addition order, including cancellation, without floating-point atomics.
    if (threadIdx.x == 0)
      for (int index=0; index<{outputs}; ++index)
        for (size_t radial=0; radial<active; ++radial)
          result[index]+=terms[index][radial];
    __syncthreads();
    begin+=active;
  }}
  if (threadIdx.x != 0) return;
  for (int index=0; index<{outputs}; ++index) {{
    if (!isfinite(result[index])) atomicExch(status, 2);
    output[component*{outputs}+index]=result[index];
  }}
}}
extern "C" int generativeqc_resident_integral(const double* records, size_t count,
    const long long* atoms, const double* direction, double* output, int* status, cudaStream_t stream) {{
  resident_integral_kernel<<<{components},64,0,stream>>>(records,count,atoms,direction,output,status);
  return static_cast<int>(cudaGetLastError());
}}
"""
    )
