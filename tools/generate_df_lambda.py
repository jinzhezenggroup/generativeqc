"""Emit retained native DF Lambda actions from shared TensorIR AD."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import typing
from functools import cache
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if __package__ in (None, ""):
    sys.path[:0] = [str(ROOT), str(ROOT / "python")]

from generativeqc_compiler.cc.df_lambda import retained_response_programs
from generativeqc_compiler.cc.df_lambda_matrix import matrix_program
from generativeqc_compiler.cc.df_lambda_reduction import (
    build_df_lambda_reduction_programs,
)

from tools.generate_df_ccsd_core import programs as core_programs
from tools.generate_df_ccsd_hoisted import contraction_query
from tools.generate_df_ccsd_native import programs as virtual_programs
from tools.generate_rccsd_native import (
    REPRESENTATIVE,
    _cuda_program,
    _packed_batched_matrix_gemm,
    _packed_matrix_gemm,
    _required_function,
    _size,
    ordered_batch_accumulation,
)

if typing.TYPE_CHECKING:
    from generativeqc_compiler.tensor import Program


@cache
def staged_programs() -> dict[str, Program]:
    """Use the same prepare/reduce/core cuts for the primal and its adjoint."""
    p = build_df_lambda_reduction_programs(*REPRESENTATIVE)
    return {
        "staged_primal_prepare": p.primal.prepare,
        "staged_primal_auxiliary": p.primal.auxiliary,
        "staged_core": p.core,
        "staged_auxiliary": p.auxiliary,
        "staged_prepare": p.prepare,
        "staged_factors": p.factors,
        **{"staged_parameter_" + name: value for name, value in p.parameters.items()},
    }


def staged_type(name: str) -> str:
    return "DeviceParameterOutput" if "parameter_" in name else name + "_outputs"


BATCHED_STAGES = frozenset(
    ("staged_primal_auxiliary", "staged_auxiliary", "staged_factors")
)
ACCUMULATED_STAGES = ("staged_primal_auxiliary", "staged_auxiliary", "staged_prepare")


@cache
def matrix_programs() -> dict[str, Program]:
    """Q is a runtime batch extent; the symbolic representative is nonunit."""
    return {
        name: matrix_program(p, batch_size=3 if name in BATCHED_STAGES else None)
        for name, p in staged_programs().items()
    }


def accumulation_declaration(name: str) -> str:
    return _accumulation(name)[0]


def accumulation_source(name: str) -> str:
    return _accumulation(name)[1]


def _accumulation(name: str) -> tuple[str, str]:
    """Primal and adjoint consumers use the same strict-order batch lowering."""
    return ordered_batch_accumulation(
        matrix_programs()[name],
        name,
        "StagedCudaState",
        staged_type(name),
        "s.gemm?s.q:1" if name in BATCHED_STAGES else "1",
    )


def output_type(name: str) -> str:
    return (
        "DeviceParameterOutput"
        if name.startswith("parameter_")
        else "DeviceLambdaOutputs"
    )


def header() -> str:
    """Exact runtime-shape capacities and scientific identities; no GPU probe."""
    lines = [
        "// Generated DF Lambda capacities; do not edit.",
        "#pragma once",
        '#include "generated_df_ccsd_cpu.hpp"',
        "namespace generativeqc::cc::generated::dflambda {",
        "using df::checked_add; using df::checked_mul; using df::checked_product;",
    ]
    retained = retained_response_programs(*REPRESENTATIVE)
    virtual = virtual_programs("cuda")
    for prefix in ("", "independent_"):
        identity = hashlib.sha256(
            json.dumps(
                {
                    "core": retained[prefix + "transpose"].logical_hash,
                    "virtual": virtual["amplitude_vjp"].logical_hash,
                    "composition": "core plus every Q amplitude VJP; dense pair-projected Frobenius metric",
                },
                sort_keys=True,
                separators=(",", ":"),
            ).encode()
        ).hexdigest()
        lines.append(
            f'inline constexpr const char* {prefix}operator_hash="{identity}";'
        )
    for name, program in retained.items():
        lines += [
            f'inline constexpr const char* {name}_hash="{program.logical_hash}";',
            f"inline constexpr std::size_t {name}_operations={sum(n.op != 'input' for n in program.live_nodes)};",
            _required_function(program, name + "_arena_elements"),
            contraction_query(program, name + "_contraction_terms"),
        ]
    lines.append(
        contraction_query(core_programs()["replay"], "replay_contraction_terms")
    )
    for name, program in virtual.items():
        lines.append(
            contraction_query(program, "virtual_" + name + "_contraction_terms")
        )
    staged = staged_programs()
    identity = hashlib.sha256(
        json.dumps(
            {name: p.logical_hash for name, p in staged.items()}, sort_keys=True
        ).encode()
    ).hexdigest()
    lines.append(f'inline constexpr const char* staged_operator_hash="{identity}";')
    for name, program in staged.items():
        if "parameter_" not in name:
            fields = ",".join("*" + field for field in program.outputs)
            lines.append(f"struct {staged_type(name)} {{ const double {fields}; }};")
        lines += [
            f"inline constexpr std::size_t {name}_operations={sum(n.op != 'input' for n in program.live_nodes)};",
            _required_function(program, name + "_arena_elements"),
            contraction_query(program, name + "_contraction_terms"),
        ]
        packed = matrix_programs()[name]
        gemms = [
            g
            for n in packed.live_nodes
            if (g := _packed_matrix_gemm(n) or _packed_batched_matrix_gemm(n))
            is not None
        ]
        dimensions = sorted({dim for g in gemms for dim in g[2:]})
        lines += [
            f"inline constexpr std::size_t {name}_matrix_operations={sum(n.op != 'input' for n in packed.live_nodes)};",
            _required_function(packed, name + "_matrix_arena_elements", batch_dim=True),
            contraction_query(
                packed, name + "_matrix_contraction_terms", batch_dim=True
            ),
            f"inline bool {name}_matrix_dimensions_fit(std::size_t o,std::size_t v,std::size_t q) {{ return "
            + " && ".join(dim + "<=2147483647ULL" for dim in dimensions or ("0",))
            + "; }",
            f"inline std::size_t {name}_matrix_packing_elements(std::size_t o,std::size_t v,std::size_t q) {{ std::size_t total=0;",
            *(
                f"total=checked_add(total,{_size(n.spec)});"
                for n in packed.live_nodes
                if n.op in ("transpose", "broadcast")
            ),
            "return total; }",
        ]
    matrix_identity = hashlib.sha256(
        json.dumps(
            {name: p.logical_hash for name, p in matrix_programs().items()},
            sort_keys=True,
        ).encode()
    ).hexdigest()
    lines.append(
        f'inline constexpr const char* staged_matrix_operator_hash="{matrix_identity}";'
    )
    return "\n".join([*lines, "}", ""])


def cuda_header() -> str:
    return "\n".join(
        [
            "// Generated DF Lambda declarations; do not edit.",
            "#pragma once",
            '#include "generated_df_lambda.hpp"',
            '#include "generated_df_ccsd_core_cuda.cuh"',
            '#include "generated_df_ccsd_hoisted_cuda.cuh"',
            "namespace generativeqc::cc::generated::dflambda {",
            "using CudaState = dfcore::CudaState;",
            "struct StagedCudaState : dfhoist::CudaState {",
            "  const double *bar_df_tau{}, *bar_df_Lvv{}, *bar_df_Wvoov{},",
            "      *bar_df_Wvovo{}, *bar_df_Xv{}, *bar_df_D05_vv_ladder{}, *bar_df_singles_residual{};",
            "};",
            "// Caller clears the sticky flag at each complete core-plus-Q action boundary.",
            *(
                f"{output_type(name)} run_{name}_cuda(CudaState& state);"
                for name in retained_response_programs(*REPRESENTATIVE)
            ),
            *(
                f"{staged_type(name)} run_{name}_cuda(StagedCudaState& state);"
                for name in staged_programs()
            ),
            *(accumulation_declaration(name) + ";" for name in ACCUMULATED_STAGES),
            "}",
            "",
        ]
    )


def cuda_source() -> str:
    lines = [
        "#include <algorithm>",
        '#include "generated_df_lambda_cuda.cuh"',
        "namespace generativeqc::cc::generated::dflambda {",
    ]
    for name, program in retained_response_programs(*REPRESENTATIVE).items():
        inputs = {
            n.attrs["name"]: "s." + n.attrs["name"]
            for n in program.live_nodes
            if n.op == "input"
        }
        lines += [
            _cuda_program(
                program,
                name,
                output_type(name),
                input_overrides=inputs,
                reset_error=False,
            ),
            f"{output_type(name)} run_{name}_cuda(CudaState& state) {{ return run_{name}(state); }}",
        ]
    for name, program in staged_programs().items():
        inputs = {
            n.attrs["name"]: "s." + n.attrs["name"]
            for n in program.live_nodes
            if n.op == "input"
        }
        kind = staged_type(name)
        lines += [
            _cuda_program(
                program,
                name + "_scalar",
                kind,
                input_overrides=inputs,
                state_type="StagedCudaState",
                output_fields=tuple(program.outputs),
                reset_error=False,
            ),
            _cuda_program(
                matrix_programs()[name],
                name + "_matrix",
                kind,
                input_overrides=inputs,
                state_type="StagedCudaState",
                output_fields=tuple(program.outputs),
                reset_error=False,
                matrix_gemm="s.gemm",
                batched_matrix_gemm="s.batched_gemm",
                batch_dim=True,
            ),
            f"{kind} run_{name}_cuda(StagedCudaState& state) {{ return state.gemm ? run_{name}_matrix(state) : run_{name}_scalar(state); }}",
        ]
    lines.extend(accumulation_source(name) for name in ACCUMULATED_STAGES)
    return "\n".join([*lines, "}", ""])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for suffix, emit in (
        (".hpp", header),
        ("_cuda.cuh", cuda_header),
        ("_cuda.cu", cuda_source),
    ):
        (args.output_dir / ("generated_df_lambda" + suffix)).write_text(emit())


if __name__ == "__main__":
    main()
