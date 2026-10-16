"""Emit linked native WB97M-V stationary source composition and work receipts."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "python"), str(ROOT)]

from generativeqc_compiler.dft._generated_native_semilocal import SEMILOCAL_FAMILIES
from generativeqc_compiler.method.ks_execution import compile_ks_execution_plan
from generativeqc_compiler.method.spec import resolve_method
from generativeqc_compiler.method.wb97mv_stationary_assembly import assembly_maps
from generativeqc_compiler.tensor.cuda_affine import emit_affine_cuda, emit_affine_work

from tools.generate_df_kernels import write_if_changed


def generate() -> str:
    """No public runtime, GPU, compiler subprocess or reference engine imports."""
    programs = assembly_maps()
    namespace = "generativeqc_wb97mv_stationary_maps"
    extents = {
        "assembly_ao": "nao",
        "assembly_spin": "spins",
        "assembly_coordinate": "coordinates",
        "assembly_pair": "pairs",
        "assembly_axis": "axes",
        "assembly_neighbor": "neighbors",
    }
    return (
        "\n".join(
            (
                emit_affine_cuda(programs, namespace=namespace, extents=extents),
                emit_affine_work(programs, namespace=namespace, extents=extents),
            )
        )
        + "\n"
    )


def generate_model() -> str:
    """Serialize the canonical complete MethodIR, not native coefficient policy."""
    domain = next(
        record["scf_domain"]
        for record in SEMILOCAL_FAMILIES
        if record["name"] == "WB97M-V"
    )
    lines = [
        "#pragma once",
        '#include "generativeqc/generativeqc.h"',
        "namespace generativeqc::dft::wb97mv_stationary_model {",
    ]
    for spin, count in (("unpolarized", 1), ("polarized", 2)):
        plan = compile_ks_execution_plan(resolve_method("WB97M-V", spin=spin))
        components = plan.semilocal.functional.components
        lines.append(
            f"inline constexpr generativeqc_ks_semilocal_component components_{count}[]{{"
        )
        lines.extend(
            f"  {{{json.dumps(name)}, {float(value):.17g}}},"
            for name, value in components
        )
        lines.append("};")
        lines.append(
            f"inline constexpr generativeqc_ks_exchange_term exchange_{count}[]{{"
        )
        for term in plan.exchange:
            operation = {
                "full-range": "FULL_RANGE",
                "short-range": "SHORT_RANGE",
                "long-range": "LONG_RANGE",
            }[term.operator]
            lines.append(
                f"  {{GENERATIVEQC_KS_EXCHANGE_{operation}, {float(term.coefficient):.17g}, {float(term.omega):.17g}, {float(term.fock_coefficient):.17g}}},"
            )
        lines.append("};")
        nonlocal_part = plan.nonlocal_correlation
        if nonlocal_part is None or nonlocal_part.spec.variant != "vv10":
            raise ValueError(
                "native stationary model requires the complete VV10 primitive"
            )
        lines.extend(
            (
                f"inline constexpr const char* identity_{count} = {json.dumps(plan.method.identity)};",
                f"inline void configure_{count}(generativeqc_ks_options& options) {{",
                f"  options.scf_domain = {json.dumps(domain)};",
                f"  options.spin_channels = {count};",
                f"  options.semilocal_components = components_{count};",
                f"  options.semilocal_component_count = {len(components)};",
                f"  options.semilocal_range_omega = {float(plan.semilocal.functional.range_omega):.17g};",
                f"  options.exchange_terms = exchange_{count};",
                f"  options.exchange_term_count = {len(plan.exchange)};",
                "  options.has_nonlocal_correlation = 1;",
                "  options.nonlocal_variant = GENERATIVEQC_NONLOCAL_VV10;",
                f"  options.nonlocal_b = {float(nonlocal_part.spec.b):.17g};",
                f"  options.nonlocal_c = {float(nonlocal_part.spec.c):.17g};",
                f"  options.nonlocal_coefficient = {float(nonlocal_part.coefficient):.17g};",
                "}",
            )
        )
    lines.append("}")
    return "\n".join(lines) + "\n"


def main() -> None:
    """Source generation is a finite build operation, never a runtime fallback."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model-output", type=Path)
    arguments = parser.parse_args()
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    write_if_changed(arguments.output, generate())
    if arguments.model_output:
        arguments.model_output.parent.mkdir(parents=True, exist_ok=True)
        write_if_changed(arguments.model_output, generate_model())


if __name__ == "__main__":
    main()
