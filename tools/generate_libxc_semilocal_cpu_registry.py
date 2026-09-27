"""Generate sharded CPU SemilocalPointProgram registry for automatic Libxc."""

from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "python"), str(ROOT)]

from vibeqc_compiler.common.provenance import canonical_hash
from vibeqc_compiler.integral.scalar_c import ScalarCEmitter
from vibeqc_compiler.xc.automatic_semilocal import (
    AUTOMATIC_SCF_DOMAIN,
    automatic_functional_code,
)
from vibeqc_compiler.xc.bulk_runtime import build_bulk_runtime_program
from vibeqc_compiler.xc.libxc_bulk_capabilities import functional_capability
from vibeqc_compiler.xc.libxc_work import (
    LIBXC_WORK_DOMAIN_VERSION,
    automatic_work_policy,
    polarized_work_setup,
)
from vibeqc_compiler.xc.spec import AUTO_BULK_COMPONENTS

SHARD_COUNT = 8
SUPPORTED_LAYOUTS = {
    ("rho_a", "rho_b"): 1,
    ("rho_a", "rho_b", "sigma_aa", "sigma_ab", "sigma_bb"): 7,
    (
        "rho_a",
        "rho_b",
        "sigma_aa",
        "sigma_ab",
        "sigma_bb",
        "tau_a",
        "tau_b",
    ): 15,
}


@dataclass(frozen=True)
class RegistryEntry:
    name: str
    libxc_id: int
    code: int

    @property
    def stem(self) -> str:
        return re.sub(r"[^a-z0-9]+", "_", self.name.lower()).strip("_")


def registry_entries() -> tuple[RegistryEntry, ...]:
    """Return every structurally supported automatic component."""
    result = []
    codes = set()
    for name in sorted(AUTO_BULK_COMPONENTS):
        capability = functional_capability(name)
        if set(capability.required_ingredients) - {"rho", "sigma", "tau"}:
            continue
        code = automatic_functional_code(name)
        if code in codes:
            raise ValueError(f"duplicate automatic Libxc functional code: {code:#x}")
        codes.add(code)
        result.append(RegistryEntry(name, capability.libxc_id, code))
    if not result:
        raise RuntimeError("automatic Libxc CPU registry is empty")
    return tuple(result)


def _point_program_source(entry: RegistryEntry) -> str:
    program = build_bulk_runtime_program(
        entry.name,
        spin="polarized",
        order=1,
    )
    features = program.spec.features
    try:
        ingredient_mask = SUPPORTED_LAYOUTS[features]
    except KeyError as exc:
        raise ValueError(
            f"unsupported automatic Libxc point layout for {entry.name}: {features!r}"
        ) from exc
    expected_outputs = ((), *((i,) for i in range(len(features))))
    if program.outputs != expected_outputs:
        raise ValueError(f"automatic Libxc E/vxc output layout changed: {entry.name}")

    policy = automatic_work_policy(entry.name)
    work_lines, work_arguments = polarized_work_setup(policy, features)
    variables = dict(zip(features, work_arguments, strict=True))
    emitter = ScalarCEmitter(program.graph, variables)
    emitter.emit(program.roots)
    refs = [emitter.reference(root) for root in program.roots]
    expression_identity = canonical_hash(
        {
            "schema": "vibeqc.automatic-libxc-work-point.v1",
            "interior_expression": program.expression_hash,
            "work_policy": policy.to_payload(),
            "features": features,
        }
    )
    stem = entry.stem
    lines = [
        f"SemilocalPointValue automatic_{stem}_point(",
        "    const double rho[2], const double (&gradient)[2][3], const double tau[2]) {",
        "  if (!std::isfinite(rho[0]) || !std::isfinite(rho[1]) ||",
        "      rho[0] < 0.0 || rho[1] < 0.0)",
        f'    throw std::domain_error("{entry.name} requires finite nonnegative density");',
        "  if (!std::isfinite(rho[0] + rho[1]))",
        f'    throw std::domain_error("{entry.name} total density is nonfinite");',
        "  const double rho_a = rho[0];",
        "  const double rho_b = rho[1];",
    ]
    if ingredient_mask != 1:
        lines.extend(
            [
                "  for (unsigned spin = 0; spin < 2; ++spin)",
                "    for (unsigned axis = 0; axis < 3; ++axis)",
                "      if (!std::isfinite(gradient[spin][axis]))",
                f'        throw std::domain_error("{entry.name} density gradient is nonfinite");',
                "  const double sigma_aa = gradient[0][0] * gradient[0][0] +",
                "      gradient[0][1] * gradient[0][1] + gradient[0][2] * gradient[0][2];",
                "  const double sigma_ab = gradient[0][0] * gradient[1][0] +",
                "      gradient[0][1] * gradient[1][1] + gradient[0][2] * gradient[1][2];",
                "  const double sigma_bb = gradient[1][0] * gradient[1][0] +",
                "      gradient[1][1] * gradient[1][1] + gradient[1][2] * gradient[1][2];",
            ]
        )
    else:
        lines.extend(["  (void)gradient;", "  (void)tau;"])
    if ingredient_mask == 15:
        lines.extend(
            [
                "  if (!std::isfinite(tau[0]) || !std::isfinite(tau[1]) ||",
                "      tau[0] < 0.0 || tau[1] < 0.0)",
                f'    throw std::domain_error("{entry.name} requires finite nonnegative tau");',
                "  const double tau_a = tau[0];",
                "  const double tau_b = tau[1];",
            ]
        )
    elif ingredient_mask == 7:
        lines.append("  (void)tau;")

    lines.extend(work_lines)

    lines.extend(emitter.lines)
    lines.extend(
        [
            "  SemilocalPointValue out{};",
            f"  out.energy = {refs[0]} * total_density / (work_rho_a + work_rho_b);",
            f"  out.rho[0] = {refs[1]};",
            f"  out.rho[1] = {refs[2]};",
        ]
    )
    if ingredient_mask != 1:
        lines.extend(
            [
                "  for (unsigned axis = 0; axis < 3; ++axis) {",
                f"    out.gradient[0][axis] = 2.0 * {refs[3]} * gradient[0][axis] +",
                f"        {refs[4]} * gradient[1][axis];",
                f"    out.gradient[1][axis] = {refs[4]} * gradient[0][axis] +",
                f"        2.0 * {refs[5]} * gradient[1][axis];",
                "  }",
            ]
        )
    if ingredient_mask == 15:
        lines.extend(
            [
                f"  out.kinetic[0] = 0.5 * {refs[6]};",
                f"  out.kinetic[1] = 0.5 * {refs[7]};",
            ]
        )
    lines.extend(
        [
            "  if (!std::isfinite(out.energy) || !std::isfinite(out.rho[0]) ||",
            "      !std::isfinite(out.rho[1]))",
            f'    throw std::domain_error("{entry.name} produced nonfinite E/vrho");',
            "  for (unsigned spin = 0; spin < 2; ++spin) {",
            "    if (!std::isfinite(out.kinetic[spin]))",
            f'      throw std::domain_error("{entry.name} produced nonfinite vtau");',
            "    for (double value : out.gradient[spin])",
            "      if (!std::isfinite(value))",
            f'        throw std::domain_error("{entry.name} produced nonfinite gradient derivative");',
            "  }",
            "  return out;",
            "}",
            f'inline constexpr const char* kAutomatic_{stem}_ExpressionIdentity = "{expression_identity}";',
            f"const SemilocalPointProgram kAutomatic_{stem}_Program{{",
            f'    "{entry.name}", kAutomatic_{stem}_ExpressionIdentity, {ingredient_mask}U,',
            f"    {LIBXC_WORK_DOMAIN_VERSION}U, automatic_{stem}_point}};",
            "",
        ]
    )
    return "\n".join(lines)


def emit_header() -> str:
    return f"""// Generated automatic Libxc CPU registry; do not edit.
#pragma once
#include <cstdint>
#include <string_view>
#include "dft/xc.hpp"

namespace vibeqc::dft::generated {{
inline constexpr const char* kAutomaticLibxcScfDomain =
    "{AUTOMATIC_SCF_DOMAIN}";
struct AutomaticLibxcEntry {{
  const SemilocalPointProgram* program{{}};
  std::uint32_t functional_code{{}};
  constexpr explicit operator bool() const noexcept {{ return program != nullptr; }}
}};
AutomaticLibxcEntry automatic_libxc_entry(std::string_view name) noexcept;
AutomaticLibxcEntry automatic_libxc_entry(std::uint32_t functional_code) noexcept;
}}  // namespace vibeqc::dft::generated
"""


def emit_shard(index: int, entries: tuple[RegistryEntry, ...]) -> str:
    selected = entries[index::SHARD_COUNT]
    declarations = "\n".join(_point_program_source(entry) for entry in selected)
    cases = "\n".join(
        [
            f'  if (name == "{entry.name}") return {{&kAutomatic_{entry.stem}_Program, 0x{entry.code:x}U}};'
            for entry in selected
        ]
    )
    return "\n".join(
        [
            "// Generated automatic Libxc CPU point programs; do not edit.",
            "#include <cmath>",
            "#include <stdexcept>",
            "#include <string_view>",
            '#include "generated_libxc_semilocal_registry.hpp"',
            "",
            "namespace vibeqc::dft::generated {",
            "namespace {",
            declarations,
            "}  // namespace",
            f"AutomaticLibxcEntry automatic_libxc_entry_shard_{index}(std::string_view name) noexcept {{",
            "  try {",
            cases,
            "  } catch (...) {",
            "    return {};",
            "  }",
            "  return {};",
            "}",
            "}  // namespace vibeqc::dft::generated",
            "",
        ]
    )


def emit_registry() -> str:
    code_cases = "\n".join(
        [
            f'    case 0x{entry.code:x}U: return automatic_libxc_entry("{entry.name}");'
            for entry in registry_entries()
        ]
    )
    declarations = "\n".join(
        f"AutomaticLibxcEntry automatic_libxc_entry_shard_{index}(std::string_view) noexcept;"
        for index in range(SHARD_COUNT)
    )
    calls = "\n".join(
        [
            f"  if (const auto entry = automatic_libxc_entry_shard_{index}(name); entry) return entry;"
            for index in range(SHARD_COUNT)
        ]
    )
    return "\n".join(  # noqa: FLY002
        [
            "// Generated automatic Libxc CPU registry dispatcher; do not edit.",
            '#include "generated_libxc_semilocal_registry.hpp"',
            "",
            "namespace vibeqc::dft::generated {",
            declarations,
            "",
            "AutomaticLibxcEntry automatic_libxc_entry(std::string_view name) noexcept {",
            calls,
            "  return {};",
            "}",
            "",
            "AutomaticLibxcEntry automatic_libxc_entry(std::uint32_t functional_code) noexcept {",
            "  switch (functional_code) {",
            code_cases,
            "    default: return {};",
            "  }",
            "}",
            "}  // namespace vibeqc::dft::generated",
            "",
        ]
    )


def write_if_changed(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists() or path.read_text(encoding="utf-8") != content:
        path.write_text(content, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-directory", required=True, type=Path)
    args = parser.parse_args()
    entries = registry_entries()
    output = args.output_directory
    write_if_changed(output / "generated_libxc_semilocal_registry.hpp", emit_header())
    write_if_changed(output / "generated_libxc_semilocal_registry.cpp", emit_registry())
    for index in range(SHARD_COUNT):
        write_if_changed(
            output / f"generated_libxc_semilocal_{index}.cpp",
            emit_shard(index, entries),
        )


if __name__ == "__main__":
    main()
