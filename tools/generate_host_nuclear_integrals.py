"""Build-only host contractions for an explicitly selected hybrid inventory."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "python"), str(ROOT)]

from generativeqc_compiler.method.host_hessian_aot import parse_inventory, programs
from generativeqc_compiler.method.host_nuclear_integrals import (
    integral_density_response_program,
    integral_weight_program,
    integral_weight_response_program,
    quartet_direction_program,
)
from generativeqc_compiler.method.spec import resolve_method
from generativeqc_compiler.tensor.host_affine import emit_affine_host

from tools.generate_df_kernels import write_if_changed


def generate(payload: dict) -> str:
    """Retain exact method/leaf identity alongside checked CPU TensorIR maps."""
    inventory = parse_inventory(payload)
    methods = {
        spin: resolve_method(inventory.method, spin=spin) for spin in inventory.spins
    }
    projection = next(
        (
            entry.identity
            for entry in programs(inventory)
            if entry.name == "integral_direction"
        ),
        "",
    )
    if not projection:
        raise ValueError(
            "native integral maps require an explicit integral_direction leaf"
        )
    contractions = {
        f"quartet_{spin}": quartet_direction_program(method)
        for spin, method in methods.items()
    }
    contractions.update(
        {
            f"density_response_{spin}": integral_density_response_program(method)
            for spin, method in methods.items()
        }
    )
    contractions.update(
        {
            f"{source}_weight_{spin}": integral_weight_program(method, source)
            for spin, method in methods.items()
            for source in ("one_electron", "overlap_pulay", "quartet")
        }
    )
    contractions.update(
        {
            f"{source}_response_weight_{spin}": integral_weight_response_program(
                method, source
            )
            for spin, method in methods.items()
            for source in ("one_electron", "overlap_pulay", "quartet")
        }
    )
    source = emit_affine_host(
        contractions,
        namespace="generativeqc_host_integral_maps",
        extents={"ordered_terms": "terms", "density_spin": "spins"},
    )
    source += "namespace generativeqc_host_integral_maps {\n"
    source += (
        "inline constexpr std::string_view projection_identity = "
        + json.dumps(projection)
        + ";\n"
    )
    source += "inline std::string_view method_identity(std::size_t spins) noexcept {\n"
    for spin, method in methods.items():
        source += (
            f" if (spins == {2 if spin == 'polarized' else 1}) return "
            + json.dumps(method.identity)
            + ";\n"
        )
    source += ' return ""; }\n}\n'
    namespace = "generativeqc_host_integral_maps"
    artifact_namespace = namespace + "_" + hashlib.sha256(source.encode()).hexdigest()
    return source.replace(namespace, artifact_namespace) + (
        f"namespace {namespace} = {artifact_namespace};\n"
    )


def main() -> None:
    """Generate only declared linked method mathematics, with no runtime load."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inventory", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    arguments = parser.parse_args()
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    write_if_changed(
        arguments.output, generate(json.loads(arguments.inventory.read_text()))
    )


if __name__ == "__main__":
    main()
