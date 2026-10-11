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
from generativeqc_compiler.common.provenance import canonical_hash
from generativeqc_compiler.tensor.iteration_reuse import (
    IterationReusePlan,
    analyze_iteration_reuse,
    invariant_frontier,
)
from generativeqc_compiler.tensor.program import node_hashes

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


@cache
def audit_programs() -> dict[str, Program]:
    """Lower the original expanded audit, never the solver's staged cut graph."""
    return {
        "audit_core": retained_response_programs(*REPRESENTATIVE)[
            "independent_transpose"
        ],
        "audit_auxiliary": virtual_programs("cuda")["amplitude_vjp"],
        "primal_virtual": virtual_programs("cuda")["virtual"],
    }


def staged_type(name: str) -> str:
    return "DeviceParameterOutput" if "parameter_" in name else name + "_outputs"


BATCHED_STAGES = frozenset(
    (
        "staged_primal_auxiliary",
        "staged_auxiliary",
        "staged_factors",
        "audit_auxiliary",
        "primal_virtual",
    )
)
ACCUMULATED_STAGES = (
    "staged_primal_auxiliary",
    "staged_auxiliary",
    "staged_prepare",
    "audit_auxiliary",
    "primal_virtual",
)


@cache
def matrix_programs() -> dict[str, Program]:
    """Q is a runtime batch extent; the symbolic representative is nonunit."""
    return {
        name: matrix_program(p, batch_size=3 if name in BATCHED_STAGES else None)
        for name, p in {**staged_programs(), **audit_programs()}.items()
    }


@cache
def core_reuse_plan() -> IterationReusePlan:
    """Reuse only primal dependencies within one immutable native Lambda owner.

    All cotangent inputs remain dynamic, including energy and reduced-cut seeds.
    The generic purity/dependency proof and native retained-slot planner are the
    same ones used by the conventional CCSD iteration owner; no Lambda algebra
    or shape-based cache identity is introduced here.
    """
    program = matrix_programs()["staged_core"]
    return analyze_iteration_reuse(
        program,
        invariant_inputs=tuple(
            node.attrs["name"]
            for node in program.live_nodes
            if node.op == "input" and not node.attrs["name"].startswith("bar_")
        ),
    )


@cache
def core_reuse_frontier() -> tuple:
    """Persist only the exact immutable-to-dynamic boundary of the core graph."""
    return invariant_frontier(matrix_programs()["staged_core"], core_reuse_plan())


@cache
def core_reuse_identity() -> str:
    """Distinguish frontier/scratch storage from the former whole-graph arena."""
    hashes = node_hashes(matrix_programs()["staged_core"].live_nodes)
    return canonical_hash(
        {
            "dependency_proof": core_reuse_plan().identity,
            "frontier": [hashes[node] for node in core_reuse_frontier()],
            "storage": "disjoint persistent frontier and borrowed owner scratch v1",
        }
    )


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
        f"s.{name}_contractions?s.q:1" if name in BATCHED_STAGES else "1",
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
    for name, program in {**staged, **audit_programs()}.items():
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
            {name: matrix_programs()[name].logical_hash for name in staged_programs()},
            sort_keys=True,
        ).encode()
    ).hexdigest()
    lines.append(
        f'inline constexpr const char* staged_matrix_operator_hash="{matrix_identity}";'
    )
    audit_identity = hashlib.sha256(
        json.dumps(
            {name: matrix_programs()[name].logical_hash for name in audit_programs()},
            sort_keys=True,
        ).encode()
    ).hexdigest()
    lines.append(
        f'inline constexpr const char* audit_matrix_operator_hash="{audit_identity}";'
    )
    program = matrix_programs()["staged_core"]
    reuse = core_reuse_plan()
    frontier = core_reuse_frontier()
    lines += [
        f'inline constexpr const char* staged_core_reuse_hash="{core_reuse_identity()}";',
        f'inline constexpr const char* staged_core_reuse_proof_hash="{reuse.identity}";',
        _required_function(
            program,
            "staged_core_reuse_arena_elements",
            retained_nodes=frontier,
            storage_slice=slice(len(frontier)),
        ),
        _required_function(
            program,
            "staged_core_reuse_scratch_arena_elements",
            retained_nodes=frontier,
            storage_slice=slice(len(frontier), None),
        ),
    ]
    for phase, nodes in (
        ("prepare", reuse.invariant_nodes),
        ("dynamic", reuse.dynamic_nodes),
    ):
        prefix = "staged_core_reuse_" + phase
        lines += [
            f"inline constexpr std::size_t {prefix}_operations={len(nodes)};",
            contraction_query(program, prefix + "_contraction_terms", nodes=nodes),
            f"inline std::size_t {prefix}_packing_elements(std::size_t o,std::size_t v) {{ std::size_t total=0;",
            *(
                f"total=checked_add(total,{_size(node.spec)});"
                for node in nodes
                if node.op in ("transpose", "broadcast")
            ),
            "return total; }",
        ]
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
            "  double* core_reuse_arena{};",
            "  double* audit_arena{};",
            "  generativeqc::tensor::PreparedContractions core_reuse_prepare_contractions, core_reuse_dynamic_contractions;",
            *(
                f"  generativeqc::tensor::PreparedContractions {name}_contractions;"
                for name in {**staged_programs(), **audit_programs()}
            ),
            "  const double *bar_df_tau{}, *bar_df_Lvv{}, *bar_df_Wvoov{},",
            "      *bar_df_Wvovo{}, *bar_df_Xv{}, *bar_df_D05_vv_ladder{}, *bar_df_singles_residual{};",
            "};",
            "inline std::size_t contraction_host_bytes(std::size_t variants,bool parameters){",
            "  std::size_t bytes=0;",
            *(
                (
                    "  if(parameters) "
                    if name == "staged_factors" or "parameter_" in name
                    else "  "
                )
                + f"bytes+=generativeqc::tensor::PreparedContractions::storage_bytes({sum(_packed_matrix_gemm(n) is not None or _packed_batched_matrix_gemm(n) is not None for n in p.live_nodes)},"
                + ("variants" if name in BATCHED_STAGES else "1")
                + ");"
                for name in staged_programs()
                for p in (matrix_programs()[name],)
            ),
            "  return bytes; }",
            "inline std::size_t audit_contraction_host_bytes(std::size_t variants){ return "
            + "+".join(
                f"generativeqc::tensor::PreparedContractions::storage_bytes({sum(_packed_matrix_gemm(node) is not None or _packed_batched_matrix_gemm(node) is not None for node in matrix_programs()[name].live_nodes)},"
                + ("variants" if name in BATCHED_STAGES else "1")
                + ")"
                for name in audit_programs()
            )
            + "; }",
            "void prepare_audit_contractions(StagedCudaState&,generativeqc::tensor::CudaContractionContext&,std::size_t batch,std::size_t tail,bool primal,std::size_t& calls,std::size_t& summands);",
            "inline std::size_t core_reuse_contraction_host_bytes(){ return "
            + "+".join(
                f"generativeqc::tensor::PreparedContractions::storage_bytes({sum(_packed_matrix_gemm(node) is not None for node in nodes)})"
                for nodes in (
                    core_reuse_plan().invariant_nodes,
                    core_reuse_plan().dynamic_nodes,
                )
            )
            + "; }",
            "void prepare_core_reuse_contractions(StagedCudaState&,generativeqc::tensor::CudaContractionContext&,std::size_t& calls,std::size_t& summands);",
            "void run_staged_core_reuse_prepare_cuda(StagedCudaState&);",
            "staged_core_outputs run_staged_core_reuse_dynamic_cuda(StagedCudaState&);",
            "void prepare_contractions(StagedCudaState&,generativeqc::tensor::CudaContractionContext&,std::size_t batch,std::size_t tail,bool parameters,std::size_t& calls,std::size_t& summands);",
            "// Caller clears the sticky flag at each complete core-plus-Q action boundary.",
            *(
                f"{output_type(name)} run_{name}_cuda(CudaState& state);"
                for name in retained_response_programs(*REPRESENTATIVE)
            ),
            *(
                f"{staged_type(name)} run_{name}_cuda(StagedCudaState& state);"
                for name in {**staged_programs(), **audit_programs()}
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
    for name, program in {**staged_programs(), **audit_programs()}.items():
        inputs = {
            n.attrs["name"]: "s."
            + {
                "bar_df_virtual_singles": "bar_singles_residual",
                "bar_df_virtual_doubles": "bar_doubles_residual",
            }.get(n.attrs["name"], n.attrs["name"])
            for n in program.live_nodes
            if n.op == "input"
        }
        kind = staged_type(name)
        if name not in audit_programs():
            lines.append(
                _cuda_program(
                    program,
                    name + "_scalar",
                    kind,
                    input_overrides=inputs,
                    state_type="StagedCudaState",
                    output_fields=tuple(program.outputs),
                    reset_error=False,
                )
            )
        lines += [
            _cuda_program(
                matrix_programs()[name],
                name + "_matrix",
                kind,
                input_overrides=inputs,
                state_type="StagedCudaState",
                output_fields=tuple(program.outputs),
                reset_error=False,
                prepared_contractions=f"s.{name}_contractions",
                batch_dim=True,
                arena_field="audit_arena" if name in audit_programs() else None,
            ),
            f"{kind} run_{name}_cuda(StagedCudaState& state) {{ return "
            + (
                f"run_{name}_matrix(state)"
                if name in audit_programs()
                else f"state.{name}_contractions ? run_{name}_matrix(state) : run_{name}_scalar(state)"
            )
            + "; }",
        ]
    program = matrix_programs()["staged_core"]
    for phase in ("prepare", "dynamic"):
        kind = "void" if phase == "prepare" else staged_type("staged_core")
        prefix = "staged_core_reuse_" + phase
        lines += [
            _cuda_program(
                program,
                prefix,
                kind,
                input_overrides={
                    node.attrs["name"]: "s." + node.attrs["name"]
                    for node in program.live_nodes
                    if node.op == "input"
                },
                state_type="StagedCudaState",
                output_fields=tuple(program.outputs),
                reset_error=False,
                prepared_contractions=f"s.core_reuse_{phase}_contractions",
                arena_field="response_arena",
                batch_dim=True,
                kernel_prefix="staged_core_matrix",
                emit_kernels=False,
                reuse_plan=core_reuse_plan(),
                reuse_phase=phase,
                reuse_storage_nodes=core_reuse_frontier(),
                retained_arena_field="core_reuse_arena",
            ),
            f"{kind} run_{prefix}_cuda(StagedCudaState& state) {{ "
            + ("" if phase == "prepare" else "return ")
            + f"run_{prefix}(state); }}",
        ]
    lines += [
        "void prepare_core_reuse_contractions(StagedCudaState& s,generativeqc::tensor::CudaContractionContext& context,std::size_t& calls,std::size_t& summands){",
        "  bind_staged_core_reuse_prepare(s,context,1,calls,summands);",
        "  bind_staged_core_reuse_dynamic(s,context,1,calls,summands);",
        "}",
    ]
    lines.append(
        "void prepare_contractions(StagedCudaState& s,generativeqc::tensor::CudaContractionContext& context,std::size_t batch,std::size_t tail,bool parameters,std::size_t& calls,std::size_t& summands){"
    )
    for name in staged_programs():
        optional = name == "staged_factors" or "parameter_" in name
        if optional:
            lines.append("  if(parameters){")
        extent = "batch" if name in BATCHED_STAGES else "1"
        lines.append(f"  bind_{name}_matrix(s,context,{extent},calls,summands);")
        if name in BATCHED_STAGES:
            lines.append(
                f"  if(tail && tail != batch) bind_{name}_matrix(s,context,tail,calls,summands);"
            )
        if optional:
            lines.append("  }")
    lines.append("}")
    lines += [
        "void prepare_audit_contractions(StagedCudaState& s,generativeqc::tensor::CudaContractionContext& context,std::size_t batch,std::size_t tail,bool primal,std::size_t& calls,std::size_t& summands){",
        "  bind_audit_core_matrix(s,context,1,calls,summands);",
        "  bind_audit_auxiliary_matrix(s,context,batch,calls,summands);",
        "  if(tail && tail!=batch) bind_audit_auxiliary_matrix(s,context,tail,calls,summands);",
        "  if(primal){",
        "    bind_primal_virtual_matrix(s,context,batch,calls,summands);",
        "    if(tail && tail!=batch) bind_primal_virtual_matrix(s,context,tail,calls,summands);",
        "  }",
        "}",
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
