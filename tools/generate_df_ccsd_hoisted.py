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
from generativeqc_compiler.tensor import Program

from tools.generate_df_ccsd_core import INPUTS as CORE_INPUTS
from tools.generate_rccsd_native import (
    REPRESENTATIVE,
    _cpu_function,
    _cuda_program,
    _label_dims,
    _packed_matrix_gemm,
    _required_function,
    _size,
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
            "#include <functional>",
            '#include "generated_df_ccsd_hoisted_cpu.hpp"',
            '#include "generated_df_ccsd_core_cuda.cuh"',
            "namespace generativeqc::cc::generated::dfhoist {",
            "struct CudaState : dfcore::CudaState {",
            *[f"  const double* {name}{{}};" for name in EXTRA_INPUTS],
            "  double *prepare_arena{}, *auxiliary_arena{};",
            "  // Optional row-major FP64 GEMM, beta=0; caller audits every result.",
            "  std::function<void(char,char,std::size_t,std::size_t,std::size_t,double,const double*,const double*,double*)> gemm;",
            "};",
            "PreparedOutputs run_prepare_cuda(CudaState& state);",
            "AuxiliaryOutputs run_auxiliary_cuda(CudaState& state);",
            "DeviceIterationOutputs run_iteration_cuda(CudaState& state);",
            "}  // namespace generativeqc::cc::generated::dfhoist",
            "",
        ]
    )


def cuda_source() -> str:
    """Emit an optional matrix schedule and retain the original scalar fallback."""
    lines = [
        '#include "generated_df_ccsd_hoisted_cuda.cuh"',
        "namespace generativeqc::cc::generated::dfhoist {",
    ]
    for name, program in programs().items():
        kind, fields = OUTPUTS[name]
        if name == "iteration":
            kind = "DeviceIterationOutputs"
        for suffix, p, callback in (
            ("scalar", program, None),
            ("packed", packed_programs()[name], "s.gemm"),
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
                    matrix_gemm=callback,
                )
            )
        lines.append(
            f"{kind} run_{name}_cuda(CudaState& s) {{ return s.gemm ? run_{name}_packed(s) : run_{name}_scalar(s); }}"
        )
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
