"""Regression coverage for #1502 native RCCSD scalar CUDA reductions."""

from __future__ import annotations

from tools import generate_df_ccsd_core
from tools import generate_df_ccsd_hoisted
from tools import generate_rccsd_native as codegen


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


def test_scalar_iteration_reduction_has_fixed_parallel_tree_and_serial_fallback() -> None:
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
        assert "for(std::size_t r=0;r<reduction_count;++r)" in parallel
        assert "for(std::size_t r=threadIdx.x;r<reduction_count;r+=blockDim.x)" in parallel
        assert "__shfl_down_sync(0xffffffffu,sum,offset)" in parallel
        assert "__shared__ double partial[8]" in parallel
        assert "__dadd_rn" in parallel
        assert "__shfl_down_sync" not in serial


def test_production_iteration_uses_parallel_scalar_reduction_but_replay_stays_serial() -> None:
    kernels = _kernels(codegen.cuda_source())
    iteration = tuple(kernel for kernel in kernels if kernel.startswith("iteration_node_"))
    replay = tuple(kernel for kernel in kernels if kernel.startswith("replay_"))

    assert iteration
    assert any("__shfl_down_sync" in kernel for kernel in iteration)
    assert replay
    assert all("__shfl_down_sync" not in kernel for kernel in replay)


def test_df_iteration_paths_enable_parallel_scalar_reduction() -> None:
    core = _kernels(generate_df_ccsd_core.cuda_source())
    hoisted = _kernels(generate_df_ccsd_hoisted.cuda_source())

    assert any(
        kernel.startswith("iteration_node_") and "__shfl_down_sync" in kernel
        for kernel in core
    )
    assert any(
        kernel.startswith("iteration_packed_node_") and "__shfl_down_sync" in kernel
        for kernel in hoisted
    )
    assert any(
        kernel.startswith("iteration_scalar_node_") and "__shfl_down_sync" in kernel
        for kernel in hoisted
    )
