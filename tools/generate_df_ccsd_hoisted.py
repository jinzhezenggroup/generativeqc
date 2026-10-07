"""Emit the prepare/auxiliary-reduction/retained-core DF RCCSD schedule."""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from functools import cache
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if __package__ in (None, ""):
    sys.path[:0] = [str(ROOT), str(ROOT / "python")]

from generativeqc_compiler.cc.df_gemm import pack_df_contractions
from generativeqc_compiler.cc.df_hoist import (
    AUXILIARY_OUTPUTS,
    build_df_auxiliary_reduction_programs,
)
from generativeqc_compiler.cc.df_lambda_matrix import matrix_program
from generativeqc_compiler.tensor import Program

from tools.generate_df_ccsd_core import INPUTS as CORE_INPUTS
from tools.generate_rccsd_native import (
    REPRESENTATIVE,
    _cpu_function,
    _cuda_program,
    _label_dims,
    _packed_batched_matrix_gemm,
    _packed_matrix_gemm,
    _required_function,
    _size,
    ordered_batch_accumulation,
    with_jacobi_update,
)

EXTRA_INPUTS = ("bov", "bvv", "df_tau", *(f"df_{name}" for name in AUXILIARY_OUTPUTS))
INPUTS = (*CORE_INPUTS, *EXTRA_INPUTS)
OUTPUTS = {
    "prepare": ("PreparedOutputs", ("df_tau",)),
    "auxiliary": (
        "AuxiliaryOutputs",
        tuple(f"df_{name}" for name in AUXILIARY_OUTPUTS),
    ),
    "iteration": ("IterationOutputs", None),
}


@cache
def programs() -> dict[str, Program]:
    """One fixed symbolic schedule; generated dimensions are runtime values."""
    pipeline = build_df_auxiliary_reduction_programs(*REPRESENTATIVE)
    iteration = with_jacobi_update(pipeline.core)
    iteration = Program(
        iteration.outputs,
        provenance={**iteration.provenance, "native_execution_order": "dependencies"},
    )
    return {
        "prepare": pipeline.prepare,
        "auxiliary": pipeline.auxiliary,
        "iteration": iteration,
    }


@cache
def packed_programs() -> dict[str, Program]:
    """Explicit packing nodes remain visible to the existing arena allocator."""
    return {name: pack_df_contractions(program) for name, program in programs().items()}


@cache
def batched_auxiliary_program() -> Program:
    """Reuse Lambda's Q lifting; the nonunit symbolic extent becomes runtime q."""
    return matrix_program(programs()["auxiliary"], batch_size=3)


def auxiliary_accumulation() -> tuple[str, str]:
    """The six retained primal cuts share the adjoint's ordered Q consumer."""
    return ordered_batch_accumulation(
        batched_auxiliary_program(),
        "auxiliary",
        "CudaState",
        "AuxiliaryOutputs",
        "s.auxiliary_contractions?s.q:1",
        dict(
            zip(
                OUTPUTS["auxiliary"][1],
                ("lvv", "wvoov", "wvovo", "xv", "ladder", "singles"),
                strict=True,
            )
        ),
    )


def contraction_query(program: Program, name: str, *, batch_dim: bool = False) -> str:
    """Exact scalar summand count; includes output and all reduction labels.

    This is semantic contraction work, not a hardware FLOP or wall-time model.
    Callers multiply the per-Q value by every actually evaluated auxiliary slice.
    """
    terms: Counter[tuple[str, ...]] = Counter()
    for node in program.live_nodes:
        if node.op == "einsum":
            terms[tuple(sorted(_label_dims(node).values()))] += 1
    lines = [
        f"inline std::size_t {name}(std::size_t o,std::size_t v{',std::size_t q' if batch_dim else ''}) {{",
        "  std::size_t total=0;",
    ]
    if any("n" in dimensions for dimensions in terms):
        lines.append("  const auto n=checked_add(o,v);")
    for dimensions, count in sorted(terms.items()):
        factors = ",".join((str(count), *dimensions))
        lines.append(f"  total=checked_add(total,checked_product({{{factors}}}));")
    return "\n".join([*lines, "  return total;", "}"])


def cpu_header() -> str:
    """Queries and borrowed CPU actions for complete owner admission."""
    lines = [
        "// Generated DF auxiliary-reduction schedule; do not edit.",
        "#pragma once",
        '#include "generated_df_ccsd_core_cpu.hpp"',
        "namespace generativeqc::cc::generated::dfhoist {",
        "using df::checked_add; using df::checked_mul; using df::checked_product;",
        "using dfcore::IterationOutputs;",
        "struct Inputs : dfcore::Inputs {",
        *[f"  const double* {name}{{}};" for name in EXTRA_INPUTS],
        "};",
        "struct PreparedOutputs { const double* tau; };",
        "struct AuxiliaryOutputs { const double *lvv, *wvoov, *wvovo, *xv, *ladder, *singles; };",
    ]
    for name, program in programs().items():
        kind, fields = OUTPUTS[name]
        lines += [
            f'inline constexpr const char* {name}_equation_hash = "{program.logical_hash}";',
            f"inline constexpr std::size_t {name}_operation_count = {sum(n.op != 'input' for n in program.live_nodes)};",
            _required_function(program, f"{name}_arena_elements"),
            contraction_query(program, f"{name}_contraction_terms"),
            _cpu_function(
                program,
                f"run_{name}_cpu",
                kind,
                input_overrides={key: f"inputs.{key}" for key in INPUTS},
                output_fields=fields,
            ),
        ]
        packed = packed_programs()[name]
        dimensions = sorted(
            {
                dim
                for n in packed.live_nodes
                if (g := _packed_matrix_gemm(n)) is not None
                for dim in g[2:]
            }
        )
        lines += [
            f'inline constexpr const char* {name}_packed_hash = "{packed.logical_hash}";',
            f"inline constexpr std::size_t {name}_packed_operations = {sum(n.op != 'input' for n in packed.live_nodes)};",
            f"inline constexpr std::size_t {name}_packed_gemms = {sum(_packed_matrix_gemm(n) is not None for n in packed.live_nodes)};",
            _required_function(packed, f"{name}_packed_arena_elements"),
            contraction_query(packed, f"{name}_packed_contraction_terms"),
            f"inline std::size_t {name}_packing_elements(std::size_t o,std::size_t v) {{",
            "  std::size_t total=0;",
            *(
                f"  total=checked_add(total,{_size(n.spec)});"
                for n in packed.live_nodes
                if n.op == "transpose"
            ),
            "  return total; }",
            f"inline bool {name}_packed_dimensions_fit(std::size_t o,std::size_t v) {{",
            "  return "
            + " && ".join(f"{dim} <= 2147483647ULL" for dim in dimensions or ("0",))
            + "; }",
        ]
    # Q packing/broadcast arrays enter the same exact runtime liveness arena.
    batched = batched_auxiliary_program()
    dimensions = sorted(
        {
            dim
            for node in batched.live_nodes
            if (g := _packed_matrix_gemm(node) or _packed_batched_matrix_gemm(node))
            is not None
            for dim in g[2:]
        }
    )
    lines += [
        f'inline constexpr const char* auxiliary_batched_hash = "{batched.logical_hash}";',
        f"inline constexpr std::size_t auxiliary_batched_operations = {sum(n.op != 'input' for n in batched.live_nodes)};",
        _required_function(batched, "auxiliary_batched_arena_elements", batch_dim=True),
        contraction_query(
            batched, "auxiliary_batched_contraction_terms", batch_dim=True
        ),
        "inline bool auxiliary_batched_dimensions_fit(std::size_t o,std::size_t v,std::size_t q) { return "
        + " && ".join(dim + "<=2147483647ULL" for dim in dimensions or ("0",))
        + "; }",
        "inline std::size_t auxiliary_batched_packing_elements(std::size_t o,std::size_t v,std::size_t q) { std::size_t total=0;",
        *(
            f"total=checked_add(total,{_size(n.spec)});"
            for n in batched.live_nodes
            if n.op in ("transpose", "broadcast")
        ),
        "return total; }",
    ]
    # The old bounded schedule is retained for replay and resource/work fallback.
    from tools.generate_df_ccsd_core import programs as core_programs
    from tools.generate_df_ccsd_native import programs as virtual_programs

    lines += [
        contraction_query(
            core_programs()["iteration"], "fallback_core_contraction_terms"
        ),
        contraction_query(
            core_programs()["replay"], "fallback_replay_contraction_terms"
        ),
        contraction_query(
            virtual_programs("cpu")["virtual"], "fallback_virtual_cpu_contraction_terms"
        ),
        contraction_query(
            virtual_programs("cuda")["virtual"],
            "fallback_virtual_cuda_contraction_terms",
        ),
    ]
    return "\n".join(
        [*lines, "}  // namespace generativeqc::cc::generated::dfhoist", ""]
    )


def cuda_header() -> str:
    """Owner-provided state; all operations share one sticky arithmetic flag."""
    return "\n".join(
        [
            "// Generated DF auxiliary-reduction CUDA declarations; do not edit.",
            "#pragma once",
            '#include "tensor/cuda_contraction.cuh"',
            '#include "generated_df_ccsd_hoisted_cpu.hpp"',
            '#include "generated_df_ccsd_core_cuda.cuh"',
            "namespace generativeqc::cc::generated::dfhoist {",
            "struct CudaState : dfcore::CudaState {",
            *[f"  const double* {name}{{}};" for name in EXTRA_INPUTS],
            "  double *prepare_arena{}, *auxiliary_arena{};",
            "  std::size_t q{1};",
            "  generativeqc::tensor::PreparedContractions prepare_contractions, auxiliary_contractions, auxiliary_batched_contractions, iteration_contractions;",
            "};",
            "inline constexpr std::size_t contraction_host_bytes(std::size_t variants) { return "
            + "+".join(
                f"generativeqc::tensor::PreparedContractions::storage_bytes({sum(_packed_matrix_gemm(n) is not None for n in p.live_nodes)})"
                for p in packed_programs().values()
            )
            + f"+generativeqc::tensor::PreparedContractions::storage_bytes({sum(_packed_matrix_gemm(n) is not None or _packed_batched_matrix_gemm(n) is not None for n in batched_auxiliary_program().live_nodes)},variants); }}",
            "void prepare_contractions(CudaState&,generativeqc::tensor::CudaContractionContext&,std::size_t batch,std::size_t tail,std::size_t& calls,std::size_t& summands);",
            "PreparedOutputs run_prepare_cuda(CudaState& state);",
            "AuxiliaryOutputs run_auxiliary_cuda(CudaState& state);",
            "DeviceIterationOutputs run_iteration_cuda(CudaState& state);",
            auxiliary_accumulation()[0] + ";",
            "}  // namespace generativeqc::cc::generated::dfhoist",
            "",
        ]
    )


def cuda_source() -> str:
    """Emit an optional matrix schedule and retain the original scalar fallback."""
    lines = [
        "#include <algorithm>",
        "#include <cub/block/block_reduce.cuh>",
        '#include "generated_df_ccsd_hoisted_cuda.cuh"',
        "namespace generativeqc::cc::generated::dfhoist {",
    ]
    for name, program in programs().items():
        kind, fields = OUTPUTS[name]
        if name == "iteration":
            kind = "DeviceIterationOutputs"
        for suffix, p, binding in (
            ("scalar", program, None),
            ("packed", packed_programs()[name], f"s.{name}_contractions"),
        ):
            lines.append(
                _cuda_program(
                    p,
                    name + "_" + suffix,
                    kind,
                    input_overrides={key: f"s.{key}" for key in INPUTS},
                    arena_field=f"{name}_arena",
                    output_fields=fields,
                    reset_error=False,
                    prepared_contractions=binding,
                    parallel_scalar_reductions=name == "iteration",
                )
            )
        if name == "auxiliary":
            lines.append(
                _cuda_program(
                    batched_auxiliary_program(),
                    "auxiliary_batched",
                    kind,
                    input_overrides={key: f"s.{key}" for key in INPUTS},
                    arena_field="auxiliary_arena",
                    output_fields=fields,
                    reset_error=False,
                    prepared_contractions="s.auxiliary_batched_contractions",
                    batch_dim=True,
                )
            )
            lines.append(
                f"{kind} run_{name}_cuda(CudaState& s) {{ if(s.auxiliary_batched_contractions && s.q>1) return run_auxiliary_batched(s); return s.{name}_contractions ? run_{name}_packed(s) : run_{name}_scalar(s); }}"
            )
        else:
            lines.append(
                f"{kind} run_{name}_cuda(CudaState& s) {{ return s.{name}_contractions ? run_{name}_packed(s) : run_{name}_scalar(s); }}"
            )
    lines += [
        "void prepare_contractions(CudaState& s,generativeqc::tensor::CudaContractionContext& context,std::size_t batch,std::size_t tail,std::size_t& calls,std::size_t& summands){",
        *(f"  bind_{name}_packed(s,context,1,calls,summands);" for name in programs()),
        "  if(batch>1) bind_auxiliary_batched(s,context,batch,calls,summands);",
        "  if(tail>1 && tail!=batch) bind_auxiliary_batched(s,context,tail,calls,summands);",
        "}",
        auxiliary_accumulation()[1],
    ]
    return "\n".join(
        [*lines, "}  // namespace generativeqc::cc::generated::dfhoist", ""]
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for name, producer in (
        ("cpu.hpp", cpu_header),
        ("cuda.cuh", cuda_header),
        ("cuda.cu", cuda_source),
    ):
        (args.output_dir / ("generated_df_ccsd_hoisted_" + name)).write_text(producer())


if __name__ == "__main__":
    main()
