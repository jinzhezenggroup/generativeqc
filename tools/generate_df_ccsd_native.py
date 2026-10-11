"""Emit bounded native DF virtual residual/response actions from TensorIR.

This generator shares the conventional runtime-shape emitter. The compiler
derives the equations and AD; native consumers own auxiliary accumulation,
allocations, execution state, and admission of the complete endpoint.
"""

from __future__ import annotations

import argparse
import sys
from functools import cache
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if __package__ in (None, ""):
    sys.path.insert(0, str(ROOT))
    sys.path.insert(0, str(ROOT / "python"))

from generativeqc_compiler.cc.df_equations import build_df_virtual_response_programs
from generativeqc_compiler.cc.df_gemm import pack_df_contractions
from generativeqc_compiler.cc.df_lambda_matrix import matrix_program
from generativeqc_compiler.tensor import Program, prepare_for_backend

from tools.generate_rccsd_native import (
    REPRESENTATIVE,
    _cpu_function,
    _cuda_program,
    _packed_batched_matrix_gemm,
    _packed_matrix_gemm,
    _required_function,
    _size,
    contraction_query,
    ordered_batch_accumulation,
)

INPUTS = (
    "bov",
    "bvv",
    "t1",
    "t2",
    "d_t1",
    "d_t2",
    "bar_df_virtual_singles",
    "bar_df_virtual_doubles",
)
OUTPUTS = {
    "virtual": ("VirtualOutputs", ("df_virtual_singles", "df_virtual_doubles")),
    "amplitude_jvp": (
        "VirtualOutputs",
        ("d_df_virtual_singles", "d_df_virtual_doubles"),
    ),
    "amplitude_vjp": ("AmplitudeOutputs", ("bar_t1", "bar_t2")),
    "factor_vjp": ("FactorOutputs", ("bar_bov", "bar_bvv")),
}


def programs(backend: str) -> dict[str, Program]:
    """Prepare the fixed symbolic schedule without changing auxiliary scope."""
    source = build_df_virtual_response_programs(*REPRESENTATIVE)
    result = {}
    for name, program in (
        ("virtual", source.primal),
        ("amplitude_jvp", source.amplitude_jvp.program),
        ("amplitude_vjp", source.amplitude_vjp.program),
        ("factor_vjp", source.factor_vjp.program),
    ):
        prepared = prepare_for_backend(program, backend, preserve_reduction_order=True)
        if any(
            sum(index.space.kind == "virtual" for index in node.spec.indices) > 2
            for node in prepared.live_nodes
        ):
            raise ValueError(
                "native DF action reconstructed an unbounded virtual block"
            )
        result[name] = Program(
            prepared.outputs,
            provenance={
                **prepared.provenance,
                "native_execution_order": "dependencies",
            },
        )
    return result


@cache
def packed_replay_program() -> Program:
    """Pack the original expanded one-Q graph, independently of primal cuts.

    Native replay still consumes every original factor and accepted amplitude.
    Packing changes layouts and library dispatch and deduplicates one shared
    contraction. No hoisted primal intermediate or previous replay output
    supplies an audit input.
    """
    return pack_df_contractions(programs("cuda")["virtual"])


@cache
def batched_replay_program() -> Program:
    """Q-lift the original expanded graph, with shared accepted amplitudes.

    The symbolic extent supplies layout recognition; generated queries and
    actions use the runtime Q extent. No primal cuts or retained action values
    are inputs. The native owner admits only tiles fitting existing scratch.
    """
    return matrix_program(programs("cuda")["virtual"], batch_size=3)


def _batch_accumulation() -> tuple[str, str]:
    """Carry each physical residual lane through Q in the original order."""
    return ordered_batch_accumulation(
        batched_replay_program(),
        "virtual_batch",
        "BatchedReplayCudaState",
        "VirtualOutputs",
        "s.q",
        {"df_virtual_singles": "singles", "df_virtual_doubles": "doubles"},
    )


def cpu_header() -> str:
    """CPU actions and shared exact runtime-shape arena queries."""
    lines = [
        "// Generated DF RCCSD one-auxiliary-slice equations; do not edit.",
        "#pragma once",
        "#include <algorithm>",
        "#include <cmath>",
        "#include <initializer_list>",
        '#include "posthf/capacity.hpp"',
        "namespace generativeqc::cc::generated::df {",
        "using posthf::checked_add;",
        "using posthf::checked_mul;",
        "inline std::size_t checked_product(std::initializer_list<std::size_t> factors) {",
        "  std::size_t count=1; for (auto factor : factors) count=checked_mul(count,factor); return count;",
        "}",
        "struct Inputs {",
        *[f"  const double* {name}{{}};" for name in INPUTS],
        "};",
        "struct VirtualOutputs { const double *singles, *doubles; };",
        "struct AmplitudeOutputs { const double *t1, *t2; };",
        "struct FactorOutputs { const double *bov, *bvv; };",
    ]
    for backend in ("cpu", "cuda"):
        for name, program in programs(backend).items():
            lines.append(
                f'inline constexpr const char* {name}_{backend}_equation_hash = "{program.logical_hash}";'
            )
            lines.append(
                f"inline constexpr std::size_t {name}_{backend}_operation_count = {sum(node.op != 'input' for node in program.live_nodes)};"
            )
            lines.append(
                _required_function(program, f"{name}_{backend}_arena_elements")
            )
            if backend == "cpu":
                output_type, fields = OUTPUTS[name]
                lines.append(
                    _cpu_function(
                        program,
                        f"run_{name}_cpu",
                        output_type,
                        input_overrides={key: f"inputs.{key}" for key in INPUTS},
                        output_fields=fields,
                    )
                )
    packed = packed_replay_program()
    dimensions = sorted(
        {
            dimension
            for node in packed.live_nodes
            if (binding := _packed_matrix_gemm(node)) is not None
            for dimension in binding[2:]
        }
    )
    lines += [
        f'inline constexpr const char* virtual_replay_hash="{packed.logical_hash}";',
        f"inline constexpr std::size_t virtual_replay_operations={sum(node.op != 'input' for node in packed.live_nodes)};",
        f"inline constexpr std::size_t virtual_replay_gemms={sum(_packed_matrix_gemm(node) is not None for node in packed.live_nodes)};",
        _required_function(packed, "virtual_replay_arena_elements"),
        contraction_query(packed, "virtual_replay_contraction_terms"),
        "inline bool virtual_replay_dimensions_fit(std::size_t o,std::size_t v) { return "
        + " && ".join(
            f"{dimension} <= 2147483647ULL" for dimension in dimensions or ("0",)
        )
        + "; }",
        "inline std::size_t virtual_replay_packing_elements(std::size_t o,std::size_t v) { std::size_t total=0;",
        *(
            f"total=checked_add(total,{_size(node.spec)});"
            for node in packed.live_nodes
            if node.op == "transpose"
        ),
        "return total; }",
    ]
    batched = batched_replay_program()
    bindings = [
        _packed_matrix_gemm(node) or _packed_batched_matrix_gemm(node)
        for node in batched.live_nodes
    ]
    dimensions = sorted(
        {dimension for binding in bindings if binding for dimension in binding[2:]}
    )
    lines += [
        f'inline constexpr const char* virtual_batch_hash="{batched.logical_hash}";',
        f"inline constexpr std::size_t virtual_batch_operations={sum(node.op != 'input' for node in batched.live_nodes)};",
        f"inline constexpr std::size_t virtual_batch_gemms={sum(binding is not None for binding in bindings)};",
        _required_function(batched, "virtual_batch_arena_elements", batch_dim=True),
        contraction_query(batched, "virtual_batch_contraction_terms", batch_dim=True),
        "inline bool virtual_batch_dimensions_fit(std::size_t o,std::size_t v,std::size_t q) { return "
        + " && ".join(f"{dimension} <= 2147483647ULL" for dimension in dimensions)
        + "; }",
        "inline std::size_t virtual_batch_packing_elements(std::size_t o,std::size_t v,std::size_t q) { std::size_t total=0;",
        *(
            f"total=checked_add(total,{_size(node.spec)});"
            for node in batched.live_nodes
            if node.op == "transpose"
        ),
        "return total; }",
        _cpu_function(
            batched,
            "run_virtual_batch_cpu",
            "VirtualOutputs",
            batch_dim=True,
            input_overrides={key: f"inputs.{key}" for key in INPUTS},
            output_fields=OUTPUTS["virtual"][1],
        ),
        "}  // namespace generativeqc::cc::generated::df",
        "",
    ]
    return "\n".join(lines)


def cuda_header() -> str:
    """Borrowed device state; allocation/lifetime stay with its native owner."""
    return "\n".join(
        [
            "// Generated DF RCCSD device declarations; do not edit.",
            "#pragma once",
            "#include <cuda_runtime.h>",
            '#include "tensor/cuda_contraction.cuh"',
            '#include "generated_df_ccsd_cpu.hpp"',
            "namespace generativeqc::cc::generated::df {",
            "struct CudaState : Inputs {",
            "  std::size_t o{}, v{};",
            "  double* response_arena{};",
            "  int* error{};",
            "  cudaStream_t stream{};",
            "};",
            "struct ReplayCudaState : CudaState { generativeqc::tensor::PreparedContractions contractions; };",
            "inline constexpr std::size_t replay_binding_host_bytes() { return generativeqc::tensor::PreparedContractions::storage_bytes(virtual_replay_gemms); }",
            "void prepare_virtual_replay(ReplayCudaState&,generativeqc::tensor::CudaContractionContext&,std::size_t&,std::size_t&);",
            "VirtualOutputs run_virtual_replay_cuda(ReplayCudaState&);",
            "struct BatchedReplayCudaState : CudaState { std::size_t q{}; generativeqc::tensor::PreparedContractions contractions; };",
            "inline std::size_t batch_replay_binding_host_bytes(std::size_t variants) { return generativeqc::tensor::PreparedContractions::storage_bytes(virtual_batch_gemms,variants); }",
            "void prepare_virtual_batch(BatchedReplayCudaState&,generativeqc::tensor::CudaContractionContext&,std::size_t,std::size_t,std::size_t&,std::size_t&);",
            "VirtualOutputs run_virtual_batch_cuda(BatchedReplayCudaState&);",
            _batch_accumulation()[0] + ";",
            *[
                f"{output_type} run_{name}_cuda(CudaState& state);"
                for name, (output_type, _) in OUTPUTS.items()
            ],
            *[
                f"{output_type} run_{name}_accumulate_cuda(CudaState& state);"
                for name, (output_type, _) in OUTPUTS.items()
                if name != "virtual"
            ],
            "// Composition entry: caller clears error once before all Q slices.",
            "VirtualOutputs run_virtual_accumulate_cuda(CudaState& state);",
            "}  // namespace generativeqc::cc::generated::df",
            "",
        ]
    )


def cuda_source() -> str:
    """Emit CUDA primal and response arithmetic using the common native owner."""
    lines = [
        '#include "generated_df_ccsd_cuda.cuh"',
        '#include "tensor/cuda_runtime.cuh"',
        "namespace generativeqc::cc::generated::df {",
    ]
    for name, program in programs("cuda").items():
        output_type, fields = OUTPUTS[name]
        lines.append(
            _cuda_program(
                program,
                f"df_{name}",
                output_type,
                input_overrides={key: f"s.{key}" for key in INPUTS},
                output_fields=fields,
                reset_error=False,
            )
        )
        if name == "virtual":
            lines.append(
                "VirtualOutputs run_virtual_accumulate_cuda(CudaState& state) { return run_df_virtual(state); }"
            )
            lines.append(
                "VirtualOutputs run_virtual_cuda(CudaState& state) { generativeqc_tensor::cuda_check(cudaMemsetAsync(state.error,0,sizeof(int),state.stream)); return run_df_virtual(state); }"
            )
            continue
        lines.append(
            f"{output_type} run_{name}_accumulate_cuda(CudaState& state) {{ return run_df_{name}(state); }}"
        )
        lines.append(
            f"{output_type} run_{name}_cuda(CudaState& state) {{ generativeqc_tensor::cuda_check(cudaMemsetAsync(state.error,0,sizeof(int),state.stream)); return run_df_{name}(state); }}"
        )
    lines += [
        _cuda_program(
            packed_replay_program(),
            "df_virtual_replay",
            "VirtualOutputs",
            state_type="ReplayCudaState",
            input_overrides={key: f"s.{key}" for key in INPUTS},
            output_fields=OUTPUTS["virtual"][1],
            reset_error=False,
            prepared_contractions="s.contractions",
        ),
        "void prepare_virtual_replay(ReplayCudaState& s,generativeqc::tensor::CudaContractionContext& context,std::size_t& calls,std::size_t& summands){bind_df_virtual_replay(s,context,1,calls,summands);}",
        "VirtualOutputs run_virtual_replay_cuda(ReplayCudaState& s){return run_df_virtual_replay(s);}",
        _cuda_program(
            batched_replay_program(),
            "df_virtual_batch",
            "VirtualOutputs",
            state_type="BatchedReplayCudaState",
            batch_dim=True,
            input_overrides={key: f"s.{key}" for key in INPUTS},
            output_fields=OUTPUTS["virtual"][1],
            reset_error=False,
            prepared_contractions="s.contractions",
        ),
        "void prepare_virtual_batch(BatchedReplayCudaState& s,generativeqc::tensor::CudaContractionContext& context,std::size_t batch,std::size_t tail,std::size_t& calls,std::size_t& summands){bind_df_virtual_batch(s,context,batch,calls,summands); if(tail>1&&tail!=batch) bind_df_virtual_batch(s,context,tail,calls,summands);}",
        "VirtualOutputs run_virtual_batch_cuda(BatchedReplayCudaState& s){return run_df_virtual_batch(s);}",
        _batch_accumulation()[1],
        "}  // namespace generativeqc::cc::generated::df",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for name, producer in (
        ("generated_df_ccsd_cpu.hpp", cpu_header),
        ("generated_df_ccsd_cuda.cuh", cuda_header),
        ("generated_df_ccsd_cuda.cu", cuda_source),
    ):
        (args.output_dir / name).write_text(producer())


if __name__ == "__main__":
    main()
