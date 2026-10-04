"""Lower bounded RHF frame AD to runtime-shaped native matrix actions."""

from __future__ import annotations

import argparse
import sys
import typing
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if __package__ in (None, ""):
    sys.path[:0] = [str(ROOT), str(ROOT / "python")]

from generativeqc_compiler.method.rhf_orbital_response import build_rhf_frame_response

from tools.generate_df_ccsd_hoisted import contraction_query
from tools.generate_rccsd_native import (
    REPRESENTATIVE,
    _cpu_function,
    _cuda_program,
    _packed_batched_matrix_gemm,
    _packed_matrix_gemm,
    _required_function,
)

if typing.TYPE_CHECKING:
    from generativeqc_compiler.tensor import Program

STAGES = ("primal", "potential_seed", "weights", "density_direction", "orbital_action")


def programs() -> dict[str, Program]:
    """The concrete witness supplies domains; generated dimensions remain runtime values."""
    maps = build_rhf_frame_response(*REPRESENTATIVE)
    return {name: getattr(maps, name) for name in STAGES}


def inputs(program: Program) -> tuple[str, ...]:
    return tuple(
        sorted({n.attrs["name"] for n in program.live_nodes if n.op == "input"})
    )


def output_type(name: str) -> str:
    return "".join(part.title() for part in name.split("_")) + "Outputs"


def prepared_recipes(program: Program) -> list[tuple[str, ...]]:
    """Use the same direct-view recognition as the canonical native emitter."""
    return [
        recipe
        for node in program.live_nodes
        if (recipe := (_packed_matrix_gemm(node) or _packed_batched_matrix_gemm(node)))
        is not None
    ]


def cpu_header() -> str:
    """Scalar oracle, capacity queries and semantic work from the same IR."""
    maps = programs()
    all_inputs = sorted(set().union(*(inputs(p) for p in maps.values())))
    lines = [
        "// Generated RHF matrix frame response; do not edit.",
        "#pragma once",
        "#include <algorithm>",
        "#include <cmath>",
        "#include <cstdint>",
        "#include <initializer_list>",
        '#include "posthf/capacity.hpp"',
        "namespace generativeqc::scf::generated::rhf_frame {",
        "using posthf::checked_add; using posthf::checked_mul;",
        "inline std::size_t checked_product(std::initializer_list<std::size_t> factors) {",
        "  std::size_t n=1; for(auto f:factors)n=checked_mul(n,f); return n; }",
        "struct Inputs {",
        *(f"  const double* {name}{{}};" for name in all_inputs),
        "};",
    ]
    for name, p in maps.items():
        kind = output_type(name)
        lines += [
            f"struct {kind} {{",
            *(f"  const double* {key}{{}};" for key in p.outputs),
            "};",
            f'inline constexpr const char* {name}_hash="{p.logical_hash}";',
            f"inline constexpr std::size_t {name}_operations={sum(n.op != 'input' for n in p.live_nodes)};",
            f"inline constexpr std::size_t {name}_gemms={sum(_packed_matrix_gemm(n) is not None for n in p.live_nodes)};",
            _required_function(p, name + "_arena_elements"),
            contraction_query(p, name + "_contraction_terms"),
            _cpu_function(
                p,
                "run_" + name + "_cpu",
                kind,
                input_overrides={key: "inputs." + key for key in inputs(p)},
                output_fields=tuple(p.outputs),
            ),
        ]
    return "\n".join([*lines, "}", ""])


def cuda_header() -> str:
    """Borrow prepared semantic tables, with the original scalar arena fallback.

    The provider owns every submission and sticky finite-result audit. Five
    stage tables are prepared once and outlive the generated traversal. Null
    bindings also select the independent scalar final-residual audit.
    """
    recipes = {name: prepared_recipes(program) for name, program in programs().items()}
    dimensions = sorted(
        {
            dimension
            for stage in recipes.values()
            for recipe in stage
            for dimension in recipe[2:]
        }
    )
    return "\n".join(
        [
            "// Generated RHF frame device interface; do not edit.",
            "#pragma once",
            "#include <cuda_runtime.h>",
            '#include "tensor/cuda_contraction.cuh"',
            '#include "generated_rhf_frame_response_cpu.hpp"',
            "namespace generativeqc::scf::generated::rhf_frame {",
            f"inline constexpr std::size_t prepared_stages={len(STAGES)};",
            "inline std::size_t prepared_host_bytes(){return "
            + "+".join(
                f"tensor::PreparedContractions::storage_bytes({len(stage)})"
                for stage in recipes.values()
            )
            + ";}",
            "inline bool prepared_dimensions_fit(std::size_t o,std::size_t v){try{const auto n=checked_add(o,v); return "
            + " && ".join(f"{d}<=2147483647ULL" for d in dimensions)
            + ";}catch(const std::length_error&){return false;}}",
            "struct CudaState : Inputs {",
            "  std::size_t o{},v{};",
            "  double* response_arena{};",
            "  int* error{};",
            "  cudaStream_t stream{};",
            "  // Borrowed contiguous stage tables; nullptr retains scalar execution.",
            "  tensor::PreparedContractions* contractions{};",
            "};",
            "void prepare_contractions(CudaState&,tensor::CudaContractionContext&,std::size_t& calls,std::size_t& summands);",
            *(
                f"{output_type(name)} run_{name}_cuda(CudaState& state);"
                for name in STAGES
            ),
            "}",
            "",
        ]
    )


def cuda_source() -> str:
    """Retain the strict scalar schedule as a bounded runtime fallback."""
    lines = [
        '#include "generated_rhf_frame_response_cuda.cuh"',
        '#include "tensor/cuda_runtime.cuh"',
        "namespace generativeqc::scf::generated::rhf_frame {",
    ]
    for stage, (name, p) in enumerate(programs().items()):
        for suffix, prepared in (("scalar", False), ("prepared", True)):
            lines.append(
                _cuda_program(
                    p,
                    name + "_" + suffix,
                    output_type(name),
                    input_overrides={key: "s." + key for key in inputs(p)},
                    output_fields=tuple(p.outputs),
                    prepared_contractions=f"s.contractions[{stage}]"
                    if prepared
                    else None,
                    kernel_prefix=name + "_scalar",
                    emit_kernels=not prepared,
                    reset_error=False,
                )
            )
        lines.append(
            f"{output_type(name)} run_{name}_cuda(CudaState& s) {{ "
            f"return s.contractions ? run_{name}_prepared(s) : run_{name}_scalar(s); }}"
        )
    lines += [
        "void prepare_contractions(CudaState& s,tensor::CudaContractionContext& context,std::size_t& calls,std::size_t& summands){",
        *(f"bind_{name}_prepared(s,context,1,calls,summands);" for name in STAGES),
        "}",
    ]
    return "\n".join([*lines, "}", ""])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    output = parser.parse_args().output_dir
    output.mkdir(parents=True, exist_ok=True)
    for suffix, emit in (
        ("cpu.hpp", cpu_header),
        ("cuda.cuh", cuda_header),
        ("cuda.cu", cuda_source),
    ):
        (output / ("generated_rhf_frame_response_" + suffix)).write_text(emit())


if __name__ == "__main__":
    main()
