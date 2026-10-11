"""Generate structural automatic Libxc CUDA AOT entries, not public admission.

Both backends share one registration census and point science. Shards bound
compiler unit size; ordinary execution has no NVCC/Libxc dependency. This is
not a new artifact-cache owner or qualification of optional CUDA mechanisms.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "python"), str(ROOT)]

from generativeqc_compiler.xc.automatic_cuda import (
    emit_automatic_cuda_descriptor,
    emit_automatic_cuda_provider,
)
from generativeqc_compiler.xc.automatic_semilocal import (
    AutomaticRegistryEntry,
    automatic_registry_entries,
)

from tools.generate_libxc_semilocal_cpu_registry import write_if_changed

SHARD_COUNT = 32


def emit_header() -> str:
    """Declare a transport lookup with process-lifetime library ownership."""
    return """// Generated automatic Libxc CUDA registry; do not edit.
#pragma once
#include "dft/cuda_xc_point_program.hpp"
namespace generativeqc::dft::generated {
const CudaXcPointProgram* automatic_cuda_point_entry(std::uint32_t code) noexcept;
}
"""


def emit_registry(*, enabled: bool = True) -> str:
    """Dispatch structural codes without scientific arithmetic or name policy."""
    if not enabled:
        return """// Generated disabled automatic Libxc CUDA registry; do not edit.
#include "generated_libxc_semilocal_cuda_registry.hpp"
namespace generativeqc::dft::generated {
const CudaXcPointProgram* automatic_cuda_point_entry(std::uint32_t) noexcept {
  return nullptr;
}
}
"""
    declarations = "\n".join(
        f"const CudaXcPointProgram* automatic_cuda_point_shard_{index}(std::uint32_t) noexcept;"
        for index in range(SHARD_COUNT)
    )
    calls = "\n".join(
        f"  if (const auto* entry = automatic_cuda_point_shard_{index}(code)) return entry;"
        for index in range(SHARD_COUNT)
    )
    return f"""// Generated automatic Libxc CUDA registry; do not edit.
#include "generated_libxc_semilocal_cuda_registry.hpp"
namespace generativeqc::dft::generated {{
{declarations}
const CudaXcPointProgram* automatic_cuda_point_entry(std::uint32_t code) noexcept {{
{calls}
  return nullptr;
}}
}}
"""


def emit_shard(index: int, entries: tuple[AutomaticRegistryEntry, ...]) -> str:
    """Emit one bounded AOT unit from the common complete structural census."""
    if not 0 <= index < SHARD_COUNT:
        raise ValueError("automatic CUDA registry shard index is out of range")
    selected = entries[index::SHARD_COUNT]
    providers = [emit_automatic_cuda_provider(entry.name) for entry in selected]
    definitions = "\n".join(
        provider.source + emit_automatic_cuda_descriptor(provider)
        for provider in providers
    )
    cases = "\n".join(
        f"    case 0x{entry.code:x}U: return &{provider.namespace}::program;"
        for entry, provider in zip(selected, providers, strict=True)
    )
    return f"""// Generated automatic Libxc CUDA point programs; do not edit.
#include "generated_libxc_semilocal_cuda_registry.hpp"
{definitions}
namespace generativeqc::dft::generated {{
const CudaXcPointProgram* automatic_cuda_point_shard_{index}(std::uint32_t code) noexcept {{
  switch (code) {{
{cases}
    default: return nullptr;
  }}
}}
}}
"""


def main() -> None:
    """Preserve mtimes for equivalent generated files and compiler-cache reuse."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-directory", required=True, type=Path)
    parser.add_argument("--disabled", action="store_true")
    args = parser.parse_args()
    output = args.output_directory
    write_if_changed(
        output / "generated_libxc_semilocal_cuda_registry.hpp", emit_header()
    )
    write_if_changed(
        output / "generated_libxc_semilocal_cuda_registry.cpp",
        emit_registry(enabled=not args.disabled),
    )
    if args.disabled:
        return
    entries = automatic_registry_entries()
    for index in range(SHARD_COUNT):
        write_if_changed(
            output / f"generated_libxc_semilocal_cuda_{index}.cu",
            emit_shard(index, entries),
        )


if __name__ == "__main__":
    main()
