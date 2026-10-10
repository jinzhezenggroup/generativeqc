"""Emit optional occupied-spectator-pair DF ladder actions, not a solver policy.

The consumer owns symmetry/range admission, original tau retention, and bounded
fallback. These separately generated actions own no solver admission policy
and never replace the expanded physical replay.
"""

from __future__ import annotations

import argparse
import sys
import typing
from fractions import Fraction
from functools import cache
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if __package__ in (None, ""):
    sys.path[:0] = [str(ROOT), str(ROOT / "python")]

from generativeqc_compiler.cc.df_gemm import pack_df_contractions
from generativeqc_compiler.cc.df_lambda_matrix import matrix_program
from generativeqc_compiler.cc.df_spectator_pairs import (
    LADDER_OUTPUT,
    PAIRED_TAU_INPUT,
    build_ladder_pair_majorant,
    factor_ladder_dressing,
    fold_occupied_ladder_pairs,
)
from generativeqc_compiler.tensor.native_arena import analyze_native_copy_roundtrips

from tools import generate_df_ccsd_hoisted as hoisted
from tools.generate_rccsd_native import (
    _cpu_function,
    _cuda_program,
    _dim,
    _execution_nodes,
    _occupied_pair_declaration,
    _packed_batched_matrix_gemm,
    _packed_matrix_gemm,
    _required_function,
    _size,
    contraction_query,
    ordered_batch_accumulation,
)

if typing.TYPE_CHECKING:
    from generativeqc_compiler.tensor import Node, Program

FIELDS = hoisted.OUTPUTS["auxiliary"][1]
INPUTS = (*hoisted.INPUTS, PAIRED_TAU_INPUT)


def _power_upper(coefficient: Fraction) -> str:
    """Emit a rational's upper power without conversion through floating point."""
    if coefficient < 0:
        raise ValueError("positive-bound lowering requires nonnegative coefficients")
    if not coefficient:
        return "pair_bound::Power::zero()"
    numerator, denominator = coefficient.numerator, coefficient.denominator
    exponent = numerator.bit_length() - denominator.bit_length()
    exceeds = (
        numerator > denominator << exponent
        if exponent >= 0
        else numerator << -exponent > denominator
    )
    return f"pair_bound::Power::of({exponent + exceeds})"


def _positive_bound_function(program: Program, names: tuple[str, ...]) -> str:
    """Lower the same positive scalar DAG into integer-exponent upper intervals."""
    lines = [
        f"inline std::array<pair_bound::Power,2> ladder_coefficient_bounds(const std::array<pair_bound::Power,{len(names)}>& norms) {{"
    ]
    bound_names = {}
    for position, node in enumerate(program.dependency_order):
        if node.spec.indices:
            raise ValueError(
                "positive-bound lowering only supports scalar coefficients"
            )
        variable = f"bound_{position}"
        bound_names[node] = variable
        if node.op == "input":
            expression = f"norms[{names.index(node.attrs['name'])}]"
        elif node.op == "constant":
            expression = _power_upper(Fraction(*node.attrs["values"][0]))
        elif node.op == "einsum":
            coefficient = _power_upper(Fraction(*node.attrs["coefficient"]))
            expression = (
                "pair_bound::product({"
                + ",".join(
                    (coefficient, *(bound_names[child] for child in node.inputs))
                )
                + "})"
            )
        elif node.op == "add":
            terms = [
                "pair_bound::product({"
                + _power_upper(Fraction(*coefficient))
                + ","
                + bound_names[child]
                + "})"
                for child, coefficient in zip(
                    node.inputs, node.attrs["coefficients"], strict=True
                )
            ]
            expression = "pair_bound::sum({" + ",".join(terms) + "})"
        else:
            raise ValueError("unsupported positive-bound scalar primitive")
        lines.append(f"  const auto {variable}={expression};")
    lines += [
        "  return {"
        + ",".join(
            bound_names[program.outputs[name]]
            for name in ("coefficient_0", "coefficient_1")
        )
        + "};",
        "}",
    ]
    return "\n".join(lines)


def _ladder_residual_gain() -> Fraction:
    """Prove the retained ladder's max-norm gain to the physical doubles residual."""

    @cache
    def gain(node: Node) -> Fraction:
        if node.op == "input":
            return Fraction(node.attrs["name"] == LADDER_OUTPUT)
        children = [gain(child) for child in node.inputs]
        if not any(children):
            return Fraction(0)
        if node.op == "transpose":
            return children[0]
        if node.op == "add":
            return sum(
                (
                    abs(Fraction(*coefficient)) * child
                    for coefficient, child in zip(
                        node.attrs["coefficients"], children, strict=True
                    )
                ),
                Fraction(0),
            )
        if (
            node.op == "einsum"
            and len(children) == 1
            and set(node.attrs["labels"][0]) == set(node.attrs["output"])
        ):
            return abs(Fraction(*node.attrs["coefficient"])) * children[0]
        raise ValueError(
            "ladder residual gain no longer has a linear permutation-only path"
        )

    return gain(hoisted.programs()["iteration"].outputs["doubles_residual"])


@cache
def programs() -> dict[str, Program]:
    """Retain all other cuts and derive both Q layouts from the admitted fold."""
    folded = fold_occupied_ladder_pairs(
        factor_ladder_dressing(hoisted.programs()["auxiliary"])
    )
    return {
        "auxiliary": folded,
        "auxiliary_packed": pack_df_contractions(folded),
        "auxiliary_batched": matrix_program(folded, batch_size=3),
    }


def auxiliary_accumulation() -> tuple[str, str]:
    """Reflect only the folded ladder while retaining ascending individual Q adds."""
    return ordered_batch_accumulation(
        hoisted.batched_auxiliary_program(),
        "occupied_auxiliary",
        "CudaState",
        "AuxiliaryOutputs",
        "s.q",
        dict(
            zip(
                FIELDS,
                ("lvv", "wvoov", "wvovo", "xv", "ladder", "singles"),
                strict=True,
            )
        ),
        source_program=programs()["auxiliary_batched"],
        element_offsets={LADDER_OUTPUT: "occupied_ladder_offset(x,o,v)"},
    )


def cpu_header() -> str:
    """Expose exact work/storage queries and borrowed test/admission actions."""
    lines = [
        "// Generated occupied-spectator-pair DF actions; do not edit.",
        "#pragma once",
        "#include <array>",
        '#include "cc/df_pair_bound.hpp"',
        '#include "generated_df_ccsd_hoisted_cpu.hpp"',
        "namespace generativeqc::cc::generated::dfpairs {",
        "inline constexpr bool ladder_dressing_factored="
        + str(
            bool(
                programs()["auxiliary"].provenance.get(
                    "df_ladder_dressing_factorization"
                )
            )
        ).lower()
        + ";",
        "using df::checked_add; using df::checked_mul; using df::checked_product;",
        "using dfhoist::AuxiliaryOutputs;",
        "struct Inputs : dfhoist::Inputs {",
        f"  const double* {PAIRED_TAU_INPUT}{{}};",
        "};",
        "inline std::size_t occupied_tau_elements(std::size_t o,std::size_t v) {",
        _occupied_pair_declaration(checked=True),
        "  return checked_product({occupied_pairs,v,v}); }",
        "#if defined(__CUDACC__)",
        "__host__ __device__",
        "#endif",
        "inline std::size_t occupied_ladder_offset(std::size_t flat,std::size_t occupied,std::size_t virtuals) {",
        "  auto second_virtual=flat%virtuals;flat/=virtuals;auto first_virtual=flat%virtuals;flat/=virtuals;",
        "  auto second_occupied=flat%occupied;auto first_occupied=flat/occupied;",
        "  if(first_occupied>second_occupied) {",
        "    const auto saved_occupied=first_occupied;first_occupied=second_occupied;second_occupied=saved_occupied;",
        "    const auto saved_virtual=first_virtual;first_virtual=second_virtual;second_virtual=saved_virtual; }",
        "  const auto pair=first_occupied*(2*occupied-first_occupied+1)/2+second_occupied-first_occupied;",
        "  return (pair*virtuals+first_virtual)*virtuals+second_virtual; }",
    ]
    for name, program in programs().items():
        batched = name.endswith("batched")
        lines += [
            f'inline constexpr const char* {name}_equation_hash="{program.logical_hash}";',
            f"inline constexpr std::size_t {name}_operations={sum(node.op != 'input' for node in program.live_nodes)};",
            _required_function(program, name + "_arena_elements", batch_dim=batched),
            contraction_query(program, name + "_contraction_terms", batch_dim=batched),
            f"inline std::size_t {name}_packing_elements(std::size_t o,std::size_t v{',std::size_t q' if batched else ''}) {{",
            _occupied_pair_declaration(checked=True),
            "  std::size_t total=0;",
            *(
                f"  total=checked_add(total,{_size(node.spec)});"
                for node in program.live_nodes
                if node.op in ("transpose", "broadcast")
            ),
            "  return total; }",
        ]
        dimensions = sorted(
            {
                dimension
                for node in program.live_nodes
                if (
                    gemm := _packed_matrix_gemm(node)
                    or _packed_batched_matrix_gemm(node)
                )
                is not None
                for dimension in gemm[2:]
            }
        )
        lines += [
            f"inline bool {name}_dimensions_fit(std::size_t o,std::size_t v{',std::size_t q' if batched else ''}) {{",
            _occupied_pair_declaration(checked=True),
            "  return "
            + " && ".join(
                dimension + "<=2147483647ULL" for dimension in dimensions or ("0",)
            )
            + "; }",
        ]
        if name != "auxiliary_packed":
            lines.append(
                _cpu_function(
                    program,
                    "run_" + name + "_cpu",
                    "AuxiliaryOutputs",
                    input_overrides={key: "inputs." + key for key in INPUTS},
                    output_fields=FIELDS,
                    batch_dim=batched,
                )
            )
    majorant = build_ladder_pair_majorant(hoisted.programs()["auxiliary"])
    lines += [
        "struct NormInputs {",
        *(f"  const double* {name}{{}};" for name, _, _ in majorant.factor_norms),
        "};",
        "struct NormOutputs { const double *coefficient_0, *coefficient_1; };",
        "enum class NormSource { bov, bvv };",
        "struct NormBinding { NormSource source; unsigned summed_axes; };",
        f"inline constexpr std::array<NormBinding,{len(majorant.factor_norms)}> geometry_norms{{{{",
        *(
            f"  {{NormSource::{source},{sum(1 << axis for axis in axes)}}},"
            for _, source, axes in majorant.factor_norms
        ),
        "}};",
        f"inline constexpr unsigned amplitude_norm_axes={sum(1 << axis for axis in majorant.amplitude_norm[1])};",
        "inline NormInputs bind_ladder_norms(const double* values) { return {"
        + ",".join(
            "values+" + str(position) for position in range(len(majorant.factor_norms))
        )
        + "}; }",
        _required_function(majorant.coefficients, "ladder_norm_arena_elements"),
        _cpu_function(
            majorant.coefficients,
            "run_ladder_norm_cpu",
            "NormOutputs",
            signature="const NormInputs& inputs",
            input_overrides={
                name: "inputs." + name for name, _, _ in majorant.factor_norms
            },
            output_fields=("coefficient_0", "coefficient_1"),
        ),
        _positive_bound_function(
            majorant.coefficients, tuple(name for name, _, _ in majorant.factor_norms)
        ),
        f"inline constexpr auto ladder_residual_gain={_power_upper(_ladder_residual_gain())};",
    ]
    return "\n".join(
        [*lines, "}  // namespace generativeqc::cc::generated::dfpairs", ""]
    )


def cuda_header() -> str:
    """Borrow the original owner state; additional descriptors are independently optional."""
    single, batched = programs()["auxiliary_packed"], programs()["auxiliary_batched"]
    single_gemms = sum(
        _packed_matrix_gemm(node) is not None for node in single.live_nodes
    )
    batch_gemms = sum(
        _packed_matrix_gemm(node) is not None
        or _packed_batched_matrix_gemm(node) is not None
        for node in batched.live_nodes
    )
    copy_metadata = []
    for name in ("auxiliary_packed", "auxiliary_batched"):
        program = programs()[name]
        copies = analyze_native_copy_roundtrips(
            program, dimension_symbol=_dim, execution_nodes=_execution_nodes(program)
        )
        operations = sum(node.op != "input" for node in program.live_nodes)
        copy_metadata += [
            f"inline constexpr std::size_t {name}_cuda_operations={operations - len(copies.elided_nodes)};",
            f"inline constexpr std::size_t {name}_elided_copy_operations={len(copies.elided_nodes)};",
            f'inline constexpr const char* {name}_copy_plan_hash="{copies.identity}";',
        ]
    return "\n".join(
        [
            "// Generated occupied-spectator-pair DF CUDA declarations; do not edit.",
            "#pragma once",
            '#include "generated_df_ccsd_spectator_pairs_cpu.hpp"',
            '#include "generated_df_ccsd_hoisted_cuda.cuh"',
            "namespace generativeqc::cc::generated::dfpairs {",
            *copy_metadata,
            "struct CudaState : dfhoist::CudaState {",
            f"  const double* {PAIRED_TAU_INPUT}{{}};",
            "  tensor::PreparedContractions paired_contractions, paired_batched_contractions;",
            "};",
            (
                "inline constexpr std::size_t contraction_host_bytes(std::size_t variants) { return "
                f"tensor::PreparedContractions::storage_bytes({single_gemms})+"
                f"tensor::PreparedContractions::storage_bytes({batch_gemms},variants); }}"
            ),
            "void prepare_contractions(CudaState&,tensor::CudaContractionContext&,std::size_t batch,std::size_t tail,std::size_t& calls,std::size_t& summands);",
            "AuxiliaryOutputs run_auxiliary_cuda(CudaState&);",
            auxiliary_accumulation()[0] + ";",
            "}  // namespace generativeqc::cc::generated::dfpairs",
            "",
        ]
    )


def cuda_source() -> str:
    """Emit separately prepared GEMMs and the proven storage-only reflection."""
    lines = [
        "#include <algorithm>",
        '#include "tensor/cuda_reduction.cuh"',
        '#include "generated_df_ccsd_spectator_pairs_cuda.cuh"',
        "namespace generativeqc::cc::generated::dfpairs {",
    ]
    for name, program in programs().items():
        batched = name.endswith("batched")
        lines.append(
            _cuda_program(
                program,
                "occupied_" + name,
                "AuxiliaryOutputs",
                input_overrides={key: "s." + key for key in INPUTS},
                arena_field="auxiliary_arena",
                output_fields=FIELDS,
                reset_error=False,
                prepared_contractions=(
                    "s.paired_batched_contractions"
                    if batched
                    else "s.paired_contractions"
                    if name.endswith("packed")
                    else None
                ),
                batch_dim=batched,
                elide_native_copy_roundtrips=name != "auxiliary",
            )
        )
    lines += [
        "void prepare_contractions(CudaState& s,tensor::CudaContractionContext& context,std::size_t batch,std::size_t tail,std::size_t& calls,std::size_t& summands) {",
        "  bind_occupied_auxiliary_packed(s,context,1,calls,summands);",
        "  if(batch>1) bind_occupied_auxiliary_batched(s,context,batch,calls,summands);",
        "  if(tail>1 && tail!=batch) bind_occupied_auxiliary_batched(s,context,tail,calls,summands);",
        "}",
        "AuxiliaryOutputs run_auxiliary_cuda(CudaState& s) {",
        '  if(s.q>1) { if(!s.paired_batched_contractions) throw std::logic_error("occupied-pair Q batch was not admitted"); return run_occupied_auxiliary_batched(s); }',
        "  return s.paired_contractions ? run_occupied_auxiliary_packed(s) : run_occupied_auxiliary(s);",
        "}",
        auxiliary_accumulation()[1],
    ]
    return "\n".join(
        [*lines, "}  // namespace generativeqc::cc::generated::dfpairs", ""]
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for suffix, producer in (
        ("cpu.hpp", cpu_header),
        ("cuda.cuh", cuda_header),
        ("cuda.cu", cuda_source),
    ):
        (args.output_dir / ("generated_df_ccsd_spectator_pairs_" + suffix)).write_text(
            producer()
        )


if __name__ == "__main__":
    main()
