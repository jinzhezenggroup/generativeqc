"""Generate sharded CPU SemilocalPointProgram registry for automatic Libxc."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "python"), str(ROOT)]

from generativeqc_compiler.xc.automatic_point import emit_automatic_point
from generativeqc_compiler.xc.automatic_semilocal import (
    AUTOMATIC_SCF_DOMAIN,
)
from generativeqc_compiler.xc.automatic_semilocal import (
    AutomaticRegistryEntry as RegistryEntry,
)
from generativeqc_compiler.xc.automatic_semilocal import (
    automatic_registry_entries as registry_entries,
)
from generativeqc_compiler.xc.libxc_work import LIBXC_WORK_DOMAIN_VERSION

SHARD_COUNT = 8


def _point_program_source(entry: RegistryEntry) -> str:
    """Bind the common point lowering to the existing CPU registry descriptor."""
    point = emit_automatic_point(entry.name)
    stem = entry.stem
    return "\n".join(
        [
            point.source,
            f'inline constexpr const char* kAutomatic_{stem}_ExpressionIdentity = "{point.expression_identity}";',
            f"const SemilocalPointProgram kAutomatic_{stem}_Program{{",
            f'    "{entry.name}", kAutomatic_{stem}_ExpressionIdentity, {point.ingredient_mask}U,',
            f"    {LIBXC_WORK_DOMAIN_VERSION}U, {point.function_name}}};",
            "",
        ]
    )


def emit_header() -> str:
    return f"""// Generated automatic Libxc CPU registry; do not edit.
#pragma once
#include <cstdint>
#include <string_view>
#include "dft/xc.hpp"

namespace generativeqc::dft::generated {{
inline constexpr const char* kAutomaticLibxcScfDomain =
    "{AUTOMATIC_SCF_DOMAIN}";
struct AutomaticLibxcEntry {{
  const SemilocalPointProgram* program{{}};
  std::uint32_t functional_code{{}};
  constexpr explicit operator bool() const noexcept {{ return program != nullptr; }}
}};
AutomaticLibxcEntry automatic_libxc_entry(std::string_view name) noexcept;
AutomaticLibxcEntry automatic_libxc_entry(std::uint32_t functional_code) noexcept;
}}  // namespace generativeqc::dft::generated
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
            "namespace generativeqc::dft::generated {",
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
            "}  // namespace generativeqc::dft::generated",
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
            "namespace generativeqc::dft::generated {",
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
            "}  // namespace generativeqc::dft::generated",
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
