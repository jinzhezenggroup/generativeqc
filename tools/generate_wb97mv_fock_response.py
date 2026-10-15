"""AOT physical Fock maps and explicit semantic-work admission bounds."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "python"), str(ROOT)]

from generativeqc_compiler.method.spec import resolve_method
from generativeqc_compiler.method.wb97mv_fock_response import fock_maps
from generativeqc_compiler.tensor.cuda_affine import emit_affine_cuda

from tools.generate_df_kernels import write_if_changed


def generate() -> str:
    """Emit maps, method constants and a checked count of scheduled scalar terms.

    The affine schedule executes primal validation dependencies too. Count all
    nodes, including those dependencies, rather than advertising only the size
    of its output or the contraction one might implement in a future schedule.
    These are scalar term visits, not measured FLOPs or an endpoint benchmark.
    """
    programs = fock_maps()
    text = [
        emit_affine_cuda(
            programs,
            namespace="generativeqc_wb97mv_fock_maps",
            extents={"orbital": "nao", "auxiliary": "naux", "spin": "spins"},
        ),
        "namespace generativeqc_wb97mv_fock_maps {",
    ]
    short, long, nonlocal_primitive = resolve_method("WB97M-V").primitives[1:]
    for name, value in (
        ("short_exchange", short.coefficient),
        ("range_correction", long.coefficient - short.coefficient),
        ("omega", short.omega),
        ("nonlocal_b", nonlocal_primitive.spec.b),
        ("nonlocal_c", nonlocal_primitive.spec.c),
        ("nonlocal_coefficient", nonlocal_primitive.coefficient),
    ):
        text.append(f"inline constexpr double {name} = {float(value):.17g};")
    names = {"orbital": "nao", "auxiliary": "naux", "spin": "spins"}
    text.extend(
        [
            "inline bool add_work(std::size_t& total, std::initializer_list<std::size_t> factors) {",
            " std::size_t term = 1;",
            " for (auto factor : factors) {",
            "  if (factor && term > std::numeric_limits<std::size_t>::max() / factor) return false;",
            "  term *= factor;",
            " }",
            " if (total > std::numeric_limits<std::size_t>::max() - term) return false;",
            " total += term; return true;",
            "}",
        ]
    )
    for name, program in programs.items():
        text.append(
            f"inline bool work_{name}(Extents shape, std::size_t& terms) {{ terms = 0;"
        )
        for node in program.nodes:
            if node.op == "einsum":
                spaces = {}
                for source, labels in zip(
                    node.inputs, node.attrs["labels"], strict=True
                ):
                    spaces.update(zip(labels, source.spec.indices, strict=True))
                indices = spaces.values()
            elif node.op == "reduce":
                indices = node.inputs[0].spec.indices
            else:
                indices = node.spec.indices
            factors = [f"shape.{names[index.space.name]}" for index in indices]
            text.append(
                " if (!add_work(terms, {" + ", ".join(factors) + "})) return false;"
            )
        text.append(" return true; }")
    text.append("}")
    return "\n".join(text) + "\n"


def main() -> None:
    """Generation runs during compilation, never inside the native endpoint."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    write_if_changed(arguments.output, generate())


if __name__ == "__main__":
    main()
