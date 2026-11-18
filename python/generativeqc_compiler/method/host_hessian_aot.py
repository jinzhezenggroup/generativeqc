"""Build-time CPU leaves for a future native stationary Hessian owner.

The package reuses existing physical scalar Graph roots. It supplies neither
orbital response nor molecular assembly and must not enable endpoint admission.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from fractions import Fraction
from functools import cache
from typing import Any

from generativeqc_compiler.common.provenance import canonical_hash
from generativeqc_compiler.dft.native_semilocal import native_semilocal_record
from generativeqc_compiler.geometry.d3_column import D3_COLUMN_ARRAYS, d3_column_program
from generativeqc_compiler.integral.scalar_c import ScalarCEmitter
from generativeqc_compiler.xc.coefficients import coefficient_program
from generativeqc_compiler.xc.grid_geometry_response import (
    ao_first_motion_program,
    gga_energy_mixed_program,
    gga_field_response_program,
)
from generativeqc_compiler.xc.grid_response_ir import (
    grid_mixed_response_program,
    grid_response_graph_identity,
)
from generativeqc_compiler.xc.program import output_set
from generativeqc_compiler.xc.semilocal_codegen import build_roots

from .host_nuclear_integrals import integral_direction_program
from .nuclear_repulsion import nuclear_pair_program
from .spec import resolve_method

SCHEMA = "generativeqc.host-hessian-aot-inventory/v1"
ARRAYS = (
    "xc",
    "coefficients",
    "coefficient_response",
    "grid_mixed_norm",
    "grid_mixed_ratio",
    "grid_mixed_log",
    "grid_mixed_becke",
    "nuclear_pair_hvp",
    "nuclear_pair_column",
    "ao_first_motion",
    "gga_field_response",
    "gga_energy_mixed",
    "integral_direction",
    *D3_COLUMN_ARRAYS,
)
SPINS = ("unpolarized", "polarized")


@dataclass(frozen=True)
class HostInventory:
    """Explicit physical-method and scalar-leaf closure, not a capability."""

    method: str
    spins: tuple[str, ...]
    arrays: tuple[str, ...]


def parse_inventory(payload: Any) -> HostInventory:
    """Reject ignored metadata and unsupported physical-method domains."""
    if (
        not isinstance(payload, dict)
        or set(payload) != {"schema", "method", "spins", "arrays"}
        or payload["schema"] != SCHEMA
        or not isinstance(payload["method"], str)
        or not isinstance(payload["spins"], list)
        or not isinstance(payload["arrays"], list)
        or not payload["spins"]
        or not payload["arrays"]
        or any(spin not in SPINS for spin in payload["spins"])
        or any(name not in ARRAYS for name in payload["arrays"])
        or len(set(payload["spins"])) != len(payload["spins"])
        or len(set(payload["arrays"])) != len(payload["arrays"])
    ):
        raise ValueError("invalid host Hessian AOT inventory")
    inventory = HostInventory(
        payload["method"],
        tuple(sorted(payload["spins"])),
        tuple(sorted(payload["arrays"])),
    )
    for spin in inventory.spins:
        method = resolve_method(inventory.method, spin=spin)
        if {primitive.kind for primitive in method.primitives} != {
            "semilocal_xc",
            "exact_exchange",
        }:
            raise ValueError("host Hessian leaves require a full-range hybrid MethodIR")
        functional = next(
            primitive.functional
            for primitive in method.primitives
            if primitive.kind == "semilocal_xc"
        )
        if functional.ingredients != ("rho", "sigma") or not {
            name for name, coefficient in functional.components if coefficient
        } <= {"LDA_X", "GGA_X_B88", "LDA_C_VWN_RPA", "GGA_C_LYP"}:
            raise ValueError(
                "host Hessian leaves require the native B88/VWN-RPA/LYP family"
            )
    return inventory


@dataclass(frozen=True)
class HostArrayProgram:
    """One input-major/output-major FP64 lane ABI around canonical scalar roots."""

    name: str
    graph: Any
    roots: tuple
    identity: str

    @property
    def inputs(self) -> tuple[str, ...]:
        """Order reachable variables only, including exact spin labels."""
        return tuple(
            sorted(
                {
                    str(self.graph.nodes[index].payload)
                    for index in self.graph.topological_order(self.roots)
                    if self.graph.nodes[index].operation == "variable"
                }
            )
        )

    def emit(self) -> str:
        """Emit allocation-free checked host arithmetic with lazy AD branches.

        A nonzero return invalidates the entire output, including earlier lanes.
        Byte/element overflow and pointer admission precede every memory access.
        Inactive branches retain Graph semantics; no fast-math or floors enter.
        """
        if not self.roots:
            raise ValueError("host array program requires roots")
        names = self.inputs
        emitter = ScalarCEmitter(
            self.graph,
            {name: f"input[{index} * size + lane]" for index, name in enumerate(names)},
        )
        emitter.emit(self.roots)
        prefix = f"gqc_host_hessian_{self.name}"
        lines = [
            '#include "dft/host_hessian_aot.hpp"',
            "#include <array>",
            "#include <cmath>",
            "#include <limits>",
            f'extern "C" int {prefix}(const double* input, std::size_t input_count, double* output, std::size_t output_count, std::size_t size) noexcept {{',
            "  constexpr auto limit = std::numeric_limits<std::size_t>::max() / sizeof(double);",
            f"  if (size > limit / {max(1, len(names), len(self.roots))} || input_count != size * {len(names)} || output_count != size * {len(self.roots)}) return -1;",
            "  if (!size) return 0;",
            "  if (!output || (input_count && !input)) return -1;",
            "  for (std::size_t lane = 0; lane < size; ++lane) {",
            *emitter.lines,
        ]
        for index, root in enumerate(self.roots):
            target = f"output[{index} * size + lane]"
            lines += [
                f"    {target} = {emitter.reference(root)};",
                f"    if (!std::isfinite({target})) return {index + 1};",
            ]
        lines += ["  }", "  return 0;", "}"]
        quoted = ", ".join(json.dumps(name) for name in names)
        lines += [
            "namespace generativeqc::dft::host_hessian_aot {",
            f"const Program& {prefix}_program() noexcept {{",
            f"  static constexpr std::array<std::string_view, {len(names)}> inputs{{{quoted}}};",
            f'  static const Program value{{"{self.name}", "{self.identity}", inputs, {len(self.roots)}, &::{prefix}}};',
            "  return value;",
            "}",
            "}",
            "",
        ]
        return "\n".join(lines)


@cache
def programs(inventory: HostInventory) -> tuple[HostArrayProgram, ...]:
    """Select existing method, coefficient, grid and nuclear AD roots unchanged."""
    result = []
    for kind in inventory.arrays:
        spins = (
            inventory.spins
            if kind
            in (
                "xc",
                "coefficients",
                "coefficient_response",
                "gga_field_response",
                "gga_energy_mixed",
            )
            else (None,)
        )
        for spin in spins:
            method = resolve_method(inventory.method, spin=spin or "unpolarized")
            name = f"{kind}_{spin}" if spin else kind
            if kind == "xc":
                functional = next(
                    primitive.functional
                    for primitive in method.primitives
                    if primitive.kind == "semilocal_xc"
                )
                graph, roots, expression = build_roots(
                    functional, output_set(functional, 2), production=True
                )
            else:
                if kind in ("coefficients", "coefficient_response"):
                    program = coefficient_program(
                        spin,
                        "gga",
                        response=kind == "coefficient_response",
                        kinetic=False,
                    )
                    graph, roots = program.graph, tuple(program.roots)
                elif kind.startswith("grid_mixed_"):
                    program = grid_mixed_response_program(
                        kind.removeprefix("grid_mixed_"), 3
                    )
                    graph, roots = program.graph, tuple(program.roots)
                elif kind == "nuclear_pair_hvp":
                    graph, nuclear_roots = nuclear_pair_program()
                    roots = tuple(nuclear_roots[4:])
                elif kind == "nuclear_pair_column":
                    graph, roots = nuclear_pair_program()
                elif kind == "ao_first_motion":
                    graph, roots = ao_first_motion_program()
                elif kind == "integral_direction":
                    graph, roots = integral_direction_program()
                elif kind in D3_COLUMN_ARRAYS:
                    graph, roots = d3_column_program(kind)
                elif kind in ("gga_field_response", "gga_energy_mixed"):
                    functional = next(
                        primitive.functional
                        for primitive in method.primitives
                        if primitive.kind == "semilocal_xc"
                    )
                    graph, roots = (
                        gga_energy_mixed_program(functional)
                        if kind == "gga_energy_mixed"
                        else gga_field_response_program(functional)
                    )
                else:
                    raise ValueError("unknown host Hessian scalar leaf")
                expression = grid_response_graph_identity(
                    graph, roots, name, schema="generativeqc.host-hessian-scalar/v1"
                )
            identity = canonical_hash(
                {
                    "schema": "generativeqc.host-hessian-leaf/v1",
                    "name": name,
                    "method": method.to_payload(),
                    "expression": expression,
                }
            )
            result.append(HostArrayProgram(name, graph, tuple(roots), identity))
    return tuple(sorted(result, key=lambda program: program.name))


def emit_registry(inventory: HostInventory) -> str:
    """Publish explicit linked leaves; an absent identity remains a package miss."""
    selected = programs(inventory)
    bindings = []
    if {"xc", "coefficient_response"} <= set(inventory.arrays):
        entries = {program.name: program for program in selected}
        for spin in inventory.spins:
            method = resolve_method(inventory.method, spin=spin)
            functional = next(
                primitive.functional
                for primitive in method.primitives
                if primitive.kind == "semilocal_xc"
            )
            exchange = next(
                primitive.coefficient
                for primitive in method.primitives
                if primitive.kind == "exact_exchange"
            )
            try:
                record = native_semilocal_record(functional, spin=spin)
            except ValueError:
                continue
            if exchange != Fraction(record["cuda_global_hybrid_exact_exchange"]):
                continue
            bindings.append(
                "    {"
                + json.dumps(method.identity)
                + f", {record['code']}U, {record['domain_version']}U, "
                + f"{2 if spin == 'polarized' else 1}, {float(exchange)!r}, "
                + json.dumps(entries["xc_" + spin].identity)
                + ", "
                + json.dumps(entries["coefficient_response_" + spin].identity)
                + ", "
                + json.dumps(
                    entries["gga_field_response_" + spin].identity
                    if "gga_field_response" in inventory.arrays
                    else ""
                )
                + ", "
                + json.dumps(
                    entries["gga_energy_mixed_" + spin].identity
                    if "gga_energy_mixed" in inventory.arrays
                    else ""
                )
                + "},"
            )
    return "\n".join(
        [
            '#include "dft/host_hessian_aot.hpp"',
            "#include <array>",
            "namespace generativeqc::dft::host_hessian_aot {",
            *(
                f"const Program& gqc_host_hessian_{program.name}_program() noexcept;"
                for program in selected
            ),
            "std::span<const Program> programs() noexcept {",
            f"  static const std::array<Program, {len(selected)}> entries{{",
            *(
                f"    gqc_host_hessian_{program.name}_program(),"
                for program in selected
            ),
            "  };",
            "  return entries;",
            "}",
            "const Program* find(std::string_view name, std::string_view identity) noexcept {",
            "  for (const auto& program : programs())",
            "    if (program.name == name && program.identity == identity) return &program;",
            "  return nullptr;",
            "}",
            "std::span<const MethodBinding> method_bindings() noexcept {",
            f"  static const std::array<MethodBinding, {len(bindings)}> entries{{{{",
            *bindings,
            "  }};",
            "  return entries;",
            "}",
            "const MethodBinding* find_method(std::string_view identity) noexcept {",
            "  for (const auto& binding : method_bindings())",
            "    if (binding.identity == identity) return &binding;",
            "  return nullptr;",
            "}",
            "}",
            "",
        ]
    )
