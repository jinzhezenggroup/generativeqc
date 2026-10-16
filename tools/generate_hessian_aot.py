"""Package explicitly selected compiler-AD leaves as linked native CUDA code.

Python is a build dependency only. This development package is NOT an endpoint
capability: the complete native stationary/response consumer remains necessary.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "python"), str(ROOT)]

from generativeqc_compiler.common.provenance import canonical_hash
from generativeqc_compiler.dft.ao import jet_indices
from generativeqc_compiler.dft.ao_cuda import axis_expression
from generativeqc_compiler.dft.ao_resident_cuda import emit_resident_ao
from generativeqc_compiler.integral.hessian_aot import (
    IntegralRequest,
    emit_integral,
    integral_program_identity,
    integral_request,
)
from generativeqc_compiler.xc.grid_response_ir import grid_response_graph_identity

from tools.generate_df_kernels import write_if_changed
from tools.hessian_aot_inventory import SCALAR_PROGRAMS, emit_scalar

SCHEMA = "generativeqc.hessian-aot-inventory/v1"


@dataclass(frozen=True)
class Inventory:
    """Explicit package contents; absence never implies another execution route."""

    ao: bool
    arrays: tuple[str, ...]
    integrals: tuple[IntegralRequest, ...]
    integral_schedule: str = "expanded"

    @cached_property
    def names(self) -> tuple[str, ...]:
        return tuple(
            sorted(
                (
                    *(("ao",) if self.ao else ()),
                    *self.arrays,
                    *(item.name for item in self.integrals),
                )
            )
        )


def parse_inventory(payload: object) -> Inventory:
    """Validate scientific selection and reject ambiguous/ignored metadata."""
    if (
        not isinstance(payload, dict)
        or not {"schema", "ao", "arrays", "integrals"} <= payload.keys()
        or set(payload) - {"schema", "ao", "arrays", "integrals", "integral_schedule"}
        or payload["schema"] != SCHEMA
        or type(payload["ao"]) is not bool
        or not isinstance(payload["arrays"], list)
        or not isinstance(payload["integrals"], list)
        or payload.get("integral_schedule", "expanded")
        not in ("expanded", "factorized")
    ):
        raise ValueError("invalid Hessian AOT package inventory")
    arrays = payload["arrays"]
    if any(not isinstance(name, str) or name not in SCALAR_PROGRAMS for name in arrays):
        raise ValueError("unknown Hessian AOT scalar inventory entry")
    integrals = tuple(integral_request(value) for value in payload["integrals"])
    names = [*arrays, *(item.name for item in integrals)]
    if len(set(names)) != len(names) or (not names and not payload["ao"]):
        raise ValueError("Hessian AOT inventory must be nonempty and unique")
    return Inventory(
        payload["ao"],
        tuple(sorted(arrays)),
        tuple(sorted(integrals, key=lambda item: item.name)),
        payload.get("integral_schedule", "expanded"),
    )


def _factorized_requests(inventory: Inventory) -> tuple[IntegralRequest, ...]:
    if inventory.integral_schedule != "factorized":
        return ()
    return tuple(
        item
        for item in inventory.integrals
        if item.family in ("eri", "three_center_eri", "coulomb_metric")
    )


def compilation_units(inventory: Inventory) -> tuple[str, ...]:
    """Group only host metadata; compile the reusable CUDA algebra once."""
    requests = _factorized_requests(inventory)
    names = {item.name for item in requests}
    return (
        *(name + ".cu" for name in inventory.names if name not in names),
        *(("factorized_integrals.cu",) if requests else ()),
        *(
            f"integral_metadata_{index:04d}.cpp"
            for index in range((len(requests) + 255) // 256)
        ),
    )


def emit_compilation_unit(inventory: Inventory, unit: str) -> str:
    """Keep CUDA and host-only compilation membership explicit and finite."""
    if unit not in compilation_units(inventory):
        raise ValueError("unit is absent from the explicit Hessian AOT inventory")
    if unit == "factorized_integrals.cu":
        from generativeqc_compiler.integral.factorized_second import (
            emit_factorized_resident,
        )

        return emit_factorized_resident()
    if unit.startswith("integral_metadata_"):
        index = int(unit.removeprefix("integral_metadata_").removesuffix(".cpp"))
        return "\n".join(
            _emit_factorized_program(item)
            for item in _factorized_requests(inventory)[256 * index : 256 * (index + 1)]
        )
    return emit_program(inventory, unit.removesuffix(".cu"))


def _emit_factorized_program(request: IntegralRequest) -> str:
    """Preserve exact component selectors and the existing native leaf ABI."""
    family = ("eri", "three_center_eri", "coulomb_metric").index(request.family)
    radial = ("full_range", "long_range", "short_range").index(request.radial)
    angular = ",".join(map(str, request.angular))
    padded = ",".join(
        map(str, (*request.angular, *(0 for _ in range(4 - len(request.angular)))))
    )
    components = ",".join(map(str, request.components))
    return f'''
#include "dft/hessian_aot.hpp"
#include "dft/factorized_hessian_integral.hpp"
#include <array>
namespace generativeqc::dft::hessian_aot {{
const Program& program_{request.name}() noexcept {{
  static constexpr std::array<unsigned,{len(request.angular)}> angular{{{angular}}};
  static constexpr std::array<std::uint32_t,{len(request.components)}> components{{{components}}};
  static const Program program{{
    "{request.name}","{integral_program_identity(request)}",Kind::integral,{{}},{1 + 6 * request.centers},components,
    {request.centers},{request.record_stride},nullptr,nullptr,
    &factorized_integral_callback<{family},{radial},{padded},{components}>,
    {{"{request.family}",angular,"{request.radial}",{request.payload()["omega"]:.17g}}}
  }};
  return program;
}}
}}
'''


def _ao_identity() -> str:
    axes = []
    for power in range(4):
        for order in range(4):
            graph, root = axis_expression(power, order)
            axes.append(
                grid_response_graph_identity(graph, (root,), f"axis-{power}-{order}")
            )
    return canonical_hash(
        {
            "schema": "generativeqc.hessian-ao-aot/v1",
            "axes": axes,
            "jets": jet_indices(3),
            "basis": "normalized-packed-16-double-record",
        }
    )


def emit_program(inventory: Inventory, name: str) -> str:
    """Isolate symbols, preserving the original emitted mathematical bodies.

    Identifier aliases follow the existing stationary native packaging policy.
    They prevent ODR/device-link collisions between component tiles and other
    native AO consumers without textual rewriting of arithmetic or literals.
    """
    if name not in inventory.names:
        raise ValueError("program is absent from the explicit Hessian AOT inventory")
    prefix = f"gqc_hessian_aot_{name}"
    aliases = {}
    inputs, components = (), ()
    centers, stride = 0, 0
    angular, signature = (), "{}"
    callbacks = ["nullptr", "nullptr", "nullptr"]
    if name == "ao":
        source, identity = emit_resident_ao(), _ao_identity()
        kind, outputs = "ao", 20
        callbacks[1] = f"&{prefix}_launch"
        aliases = {
            "generativeqc_grid_policy": f"{prefix}_policy",
            "resident_ao": f"{prefix}_kernel",
            "generativeqc_resident_ao": f"{prefix}_launch",
        }
    elif name in inventory.arrays:
        source, identity, inputs, outputs = emit_scalar(name)
        kind = "array"
        callbacks[0] = f"&{prefix}_launch"
        aliases = {
            "array_graph": f"{prefix}_kernel",
            "generativeqc_array_graph": f"{prefix}_launch",
        }
    else:
        request = next(item for item in inventory.integrals if item.name == name)
        if inventory.integral_schedule == "factorized" and request.family in (
            "eri",
            "three_center_eri",
            "coulomb_metric",
        ):
            return _emit_factorized_program(request)
        source, identity = emit_integral(request)
        kind = "integral"
        centers, stride = request.centers, request.record_stride
        components, outputs = request.components, 1 + 6 * request.centers
        angular = request.angular
        signature = (
            f'{{"{request.family}", angular, "{request.radial}", '
            f"{request.payload()['omega']:.17g}}}"
        )
        callbacks[2] = f"&{prefix}_launch"
        aliases = {
            "resident_evaluate": f"{prefix}_evaluate",
            "resident_integral_kernel": f"{prefix}_kernel",
            "generativeqc_resident_integral": f"{prefix}_launch",
            **{
                f"{stem}_{index}": f"{prefix}_{stem}_{index}"
                for stem in ("resident_component", "component")
                for index in range(len(components))
            },
        }
    quoted_inputs = ",".join(json.dumps(value) for value in inputs)
    packed_components = ",".join(str(value) for value in components)
    return (
        "// Build-time compiler-AD package; no runtime Python or compiler.\n"
        '#include "dft/hessian_aot.hpp"\n#include <array>\n'
        + "\n".join(f"#define {old} {new}" for old, new in sorted(aliases.items()))
        + "\n"
        + source
        + "\n"
        + "\n".join(f"#undef {old}" for old in sorted(aliases))
        + f"""
namespace generativeqc::dft::hessian_aot {{
const Program& program_{name}() noexcept {{
  static constexpr std::array<std::string_view, {len(inputs)}> inputs{{{quoted_inputs}}};
  static constexpr std::array<std::uint32_t, {len(components)}> components{{{packed_components}}};
  [[maybe_unused]] static constexpr std::array<unsigned, {len(angular)}> angular{{{",".join(map(str, angular))}}};
  static const Program program{{
    "{name}", "{identity}", Kind::{kind}, inputs, {outputs}, components,
    {centers}, {stride}, {", ".join(callbacks)}, {signature}
  }};
  return program;
}}
}}
"""
    )


def emit_registry(inventory: Inventory) -> str:
    declarations = "\n".join(
        f"const Program& program_{name}() noexcept;" for name in inventory.names
    )
    entries = ",\n".join(f"    &program_{name}" for name in inventory.names)
    return f"""// Native AOT leaf inventory; this does not advertise a molecular endpoint.
#include "dft/hessian_aot.hpp"
#include <array>
namespace generativeqc::dft::hessian_aot {{
{declarations}
std::span<const Program> programs() noexcept {{
  using Getter = const Program& (*)() noexcept;
  static constexpr std::array<Getter, {len(inventory.names)}> getters{{{{
{entries}
  }}}};
  // Keep both compilation work and startup stack bounded for large closures.
  // Construct directly in static storage; do not return a multi-MiB stack array
  // from a lambda or expand tens of thousands of Program copies into one CFG.
  struct Inventory {{
    std::array<Program, {len(inventory.names)}> entries{{}};
    Inventory() noexcept {{
      for (std::size_t index=0; index<entries.size(); ++index)
        entries[index]=getters[index]();
    }}
  }};
  static const Inventory inventory;
  return inventory.entries;
}}
}}
"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inventory", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--output-dir", type=Path)
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument("--list-programs", action="store_true")
    selection.add_argument("--list-units", action="store_true")
    selection.add_argument("--unit")
    selection.add_argument("--program")
    selection.add_argument("--registry", action="store_true")
    selection.add_argument("--all", action="store_true")
    args = parser.parse_args()
    inventory = parse_inventory(json.loads(args.inventory.read_text()))
    if args.list_programs or args.list_units:
        if args.output or args.output_dir:
            parser.error("inventory discovery must not write output")
        print(
            ";".join(
                compilation_units(inventory) if args.list_units else inventory.names
            )
        )
    elif args.all:
        if args.output or not args.output_dir:
            parser.error("--all requires --output-dir and forbids --output")
        for unit in compilation_units(inventory):
            write_if_changed(
                args.output_dir / unit, emit_compilation_unit(inventory, unit)
            )
        write_if_changed(args.output_dir / "registry.cpp", emit_registry(inventory))
    else:
        if not args.output or args.output_dir:
            parser.error("program/registry generation requires --output only")
        write_if_changed(
            args.output,
            emit_registry(inventory)
            if args.registry
            else emit_compilation_unit(inventory, args.unit)
            if args.unit
            else emit_program(inventory, args.program),
        )


if __name__ == "__main__":
    main()
