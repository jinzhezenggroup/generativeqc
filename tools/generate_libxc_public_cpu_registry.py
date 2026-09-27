"""Generate the build-time registry for evidence-admitted bulk Libxc CPU programs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from vibeqc_compiler.method._generated_libxc_public_evidence import PUBLIC_EVIDENCE
from vibeqc_compiler.method.bulk_ks import resolve_public_bulk_ks
from vibeqc_compiler.method.libxc_public_evidence import installed_public_evidence
from vibeqc_compiler.xc.bulk_point_program import (
    SemilocalPointBinding,
    bind_runtime_semilocal_point_program,
)
from vibeqc_compiler.xc.bulk_runtime import (
    PRODUCTION_DENSITY_CANDIDATE_DOMAIN,
    build_bulk_runtime_program,
)


def _binding(name: str) -> SemilocalPointBinding:
    evidence = installed_public_evidence(name)
    if evidence is None:
        raise ValueError(f"{name} installed public evidence is absent or stale")
    unpolarized = resolve_public_bulk_ks(
        name, spin="unpolarized", backend="cpu", evidence=evidence
    )
    polarized = resolve_public_bulk_ks(
        name, spin="polarized", backend="cpu", evidence=evidence
    )
    program = build_bulk_runtime_program(
        name,
        spin="polarized",
        order=1,
        domain=PRODUCTION_DENSITY_CANDIDATE_DOMAIN,
    )
    binding = bind_runtime_semilocal_point_program(program)
    expected = {
        unpolarized.compiled_cpu_binding_identity,
        polarized.compiled_cpu_binding_identity,
    }
    if expected != {binding.identity}:
        raise ValueError(f"{name} installed public evidence has stale compiled binding")
    return binding


def render_registry() -> str:
    """Render one deterministic C++ header over the installed public inventory."""
    names = sorted(PUBLIC_EVIDENCE)
    chunks = [
        "#pragma once",
        "",
        "#include <string_view>",
        '#include "dft/xc.hpp"',
        "",
    ]
    records: list[tuple[str, str]] = []
    for index, name in enumerate(names):
        binding = _binding(name)
        namespace_name = f"bulk_public_{index}"
        chunks.append(
            binding.emit_source(
                namespace_name=namespace_name,
                expose_accessor=True,
            )
        )
        records.append((name, namespace_name))

    chunks.extend(
        [
            "",
            "namespace vibeqc::dft::bulk_public {",
            "inline const SemilocalPointProgram* find(std::string_view name) noexcept {",
        ]
    )
    for name, namespace_name in records:
        chunks.extend(
            [
                f"  if (name == {json.dumps(name)})",
                "    return &::vibeqc::dft::"
                + namespace_name
                + "::point_program();",
            ]
        )
    chunks.extend(
        [
            "  return nullptr;",
            "}",
            "}  // namespace vibeqc::dft::bulk_public",
            "",
        ]
    )
    return "\n".join(chunks)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(render_registry(), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
