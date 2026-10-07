"""Regression coverage for #1502 native RCCSD scalar CUDA reductions."""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import TYPE_CHECKING

import pytest
from generativeqc_compiler.tensor.ir import einsum, input_tensor, reduce_sum
from generativeqc_compiler.tensor.types import Index, IndexSpace, TensorSpec

from tools import generate_df_ccsd_core, generate_df_ccsd_hoisted
from tools import generate_rccsd_native as codegen

if TYPE_CHECKING:
    from conftest import NativeCxx


def _kernels(source: str) -> tuple[str, ...]:
    return tuple(source.split("__global__ void ")[1:])


def _iteration_scalar_reductions() -> tuple[tuple[int, object], ...]:
    program = codegen._prepare_production(
        codegen.iteration_program(*codegen.REPRESENTATIVE), "cuda"
    )
    return tuple(
        (number, node)
        for number, node in enumerate(codegen._execution_nodes(program))
        if not node.spec.indices and node.op in ("reduce", "einsum")
    )


def test_scalar_iteration_reduction_has_fixed_parallel_tree_and_serial_fallback() -> (
    None
):
    reductions = _iteration_scalar_reductions()
    assert reductions
    for number, node in reductions:
        parallel = codegen._cuda_kernel(
            node,
            number,
            "probe",
            {},
            parallel_scalar_reductions=True,
        )
        serial = codegen._cuda_kernel(node, number, "probe_serial", {})

        assert "const std::size_t reduction_count=" in parallel
        assert "if(reduction_count<32)" in parallel
        assert "#if GENERATIVEQC_TENSOR_HAS_STRICT_FP64_BLOCK_REDUCE" in parallel
        assert "for(std::size_t r=0;r<reduction_count;++r)" in parallel
        assert (
            "for(std::size_t r=threadIdx.x;r<reduction_count;r+=blockDim.x)" in parallel
        )
        assert "generativeqc::tensor::StrictFp64BlockReduce<256>" in parallel
        assert "BlockReduce::sum(sum,temp_storage)" in parallel
        assert "__dadd_rn" in parallel
        assert "StrictFp64BlockReduce" not in serial


def test_production_iteration_uses_parallel_scalar_reduction_but_replay_stays_serial() -> (
    None
):
    kernels = _kernels(codegen.cuda_source())
    iteration = tuple(
        kernel for kernel in kernels if kernel.startswith("iteration_node_")
    )
    replay = tuple(kernel for kernel in kernels if kernel.startswith("replay_"))

    assert '#include "tensor/cuda_reduction.cuh"' in codegen.cuda_source()
    assert iteration
    assert any("StrictFp64BlockReduce<256>" in kernel for kernel in iteration)
    assert replay
    assert all("StrictFp64BlockReduce" not in kernel for kernel in replay)


def test_df_iteration_paths_enable_parallel_scalar_reduction() -> None:
    core = _kernels(generate_df_ccsd_core.cuda_source())
    hoisted = _kernels(generate_df_ccsd_hoisted.cuda_source())

    assert any(
        kernel.startswith("iteration_node_") and "StrictFp64BlockReduce<256>" in kernel
        for kernel in core
    )
    assert any(
        kernel.startswith("iteration_packed_node_")
        and "StrictFp64BlockReduce<256>" in kernel
        for kernel in hoisted
    )
    assert any(
        kernel.startswith("iteration_scalar_node_")
        and "StrictFp64BlockReduce<256>" in kernel
        for kernel in hoisted
    )


@pytest.mark.parametrize("operation", ("reduce", "einsum"))
def test_complete_orbital_scalar_kernel_declares_runtime_extent(
    tmp_path: Path, native_cxx: NativeCxx, operation: str
) -> None:
    """Compile the CuMetal branch and execute its real full-domain serial fallback."""
    space = IndexSpace("complete", "orbital", codegen.REPRESENTATIVE_ORBITALS)
    source = input_tensor("x", TensorSpec((Index("p", space),), role="input"))
    node = (
        reduce_sum(source, (0,))
        if operation == "reduce"
        else einsum("p,p->", source, source)
    )
    kernel = codegen._cuda_kernel(
        node, 0, "complete", {}, parallel_scalar_reductions=True
    )
    arguments = ",".join([*("values" for _ in node.inputs), "&out", "o", "v", "&error"])
    unit = tmp_path / "complete.cpp"
    unit.write_text(
        """
#include <cassert>
#include <cstddef>
#include <initializer_list>
#define __global__
#define __shared__
#define __device__
#define __forceinline__ inline
struct Dim { unsigned x; };
Dim threadIdx{0}, blockDim{256};
double __dadd_rn(double a, double b) { return a+b; }
double __dmul_rn(double a, double b) { return a*b; }
#define GENERATIVEQC_CUDA_PROVIDER_CUMETAL 1
#include "tensor/cuda_reduction.cuh"
static_assert(!GENERATIVEQC_TENSOR_HAS_STRICT_FP64_BLOCK_REDUCE);
namespace generativeqc_tensor {
double finite(double value, int*, int) { return value; }
}
"""
        + kernel
        + f"""
int main() {{
  double values[300];
  for (int i=0;i<300;++i) values[i]=(i%7)-3;
  for (std::size_t o : {{0,1,2}}) {{
    for (std::size_t v : {{0,1,5,31,32,33,255,256,257}}) {{
      const auto n=o+v;
      double expected=0.0;
      for (std::size_t i=0;i<n;++i)
        expected += {"values[i]" if operation == "reduce" else "values[i]*values[i]"};
      double out=-1.0;
      int error=0;
      complete_node_0({arguments});
      assert(error==0 && out==expected);
    }}
  }}
}}
"""
    )
    executable = native_cxx.build_executable(
        [unit],
        tmp_path / "complete",
        compile_args=(
            "-std=c++17",
            "-Wall",
            "-Wextra",
            "-Werror",
            "-I",
            str(Path(__file__).resolve().parents[2] / "src"),
        ),
    )
    subprocess.run([str(executable)], check=True, timeout=10)
