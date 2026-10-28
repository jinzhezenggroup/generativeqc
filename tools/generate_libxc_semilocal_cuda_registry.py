"""Generate structural automatic Libxc CUDA AOT entries, not public admission.

Both backends share one registration census and point science. Shards bound
compiler unit size; ordinary execution has no NVCC/Libxc dependency. This is
not a new artifact-cache owner or qualification of optional CUDA mechanisms.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Literal

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "python"), str(ROOT)]

from generativeqc_compiler.common.provenance import canonical_hash
from generativeqc_compiler.xc.automatic_cuda import (
    emit_automatic_cuda_descriptor,
    emit_automatic_cuda_provider,
)
from generativeqc_compiler.xc.automatic_semilocal import (
    AutomaticRegistryEntry,
    automatic_registry_entries,
)
from generativeqc_compiler.xc.parameter_cuda_package import (
    parameter_worker_groups,
    parameterized_cuda_package,
)

from tools.generate_libxc_semilocal_cpu_registry import write_if_changed

SHARD_COUNT = 32


def select_entries(
    functionals: list[str] | None = None,
) -> tuple[AutomaticRegistryEntry, ...]:
    """Select an explicit package, preserving structural order and provenance.

    Absence of a selection preserves the existing opt-in full-catalog census.
    Unsupported names fail before any output is written. Duplicate requests
    cannot create duplicate kernels, and request order cannot change shards.
    This is an artifact-content policy, not public or scientific admission.
    """
    entries = automatic_registry_entries()
    if functionals is None:
        return entries
    requested = set(functionals)
    known = {entry.name for entry in entries}
    if not requested or requested - known:
        unknown = ", ".join(sorted(requested - known)) or "<empty selection>"
        raise ValueError(f"unregistered automatic CUDA AOT functional: {unknown}")
    return tuple(entry for entry in entries if entry.name in requested)


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


def emit_shard(
    index: int,
    entries: tuple[AutomaticRegistryEntry, ...],
    *,
    share_parameters: bool = True,
    parameter_policy: Literal["qualified", "discover"] = "discover",
) -> str:
    """Emit one bounded AOT unit from an authenticated structural package."""
    if not 0 <= index < SHARD_COUNT:
        raise ValueError("automatic CUDA registry shard index is out of range")
    selected = entries[index::SHARD_COUNT]
    if share_parameters:
        workers, registrations = parameterized_cuda_package(
            tuple(entry.name for entry in entries), policy=parameter_policy
        )
        providers = [registrations[entry.name] for entry in selected]
    else:
        # Explicit specialization remains available for qualification and for
        # toolchains/devices outside the measured shared-worker population.
        workers = {}
        providers = [emit_automatic_cuda_provider(entry.name) for entry in selected]
    # Registration placement stays stable. Each private family worker has one
    # CUDA owner; other shards reference its host launcher through declarations.
    shared = "\n".join(
        source
        for identity, source in workers.items()
        if int(identity[:16], 16) % SHARD_COUNT == index
    )
    definitions = (
        shared
        + "\n"
        + "\n".join(
            provider.source + emit_automatic_cuda_descriptor(provider)
            for provider in providers
        )
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
    parser.add_argument(
        "--share-parameters",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="share compatible workers by default; --no-share-parameters retains literal specialization",
    )
    parser.add_argument(
        "--parameter-policy",
        choices=("qualified", "discover"),
        default=None,
        help="discover compatible catalog-wide workers by default; qualified retains legacy prepared-product groups",
    )
    parser.add_argument(
        "--sharing-report",
        type=Path,
        help="write selected groups, coefficient ABI sizes and retained specializations as JSON",
    )
    parser.add_argument(
        "--functional",
        action="append",
        help="package only requested structural registrations (repeatable)",
    )
    args = parser.parse_args()
    if args.disabled and args.functional:
        parser.error("--functional cannot be combined with --disabled")
    if args.disabled and args.share_parameters:
        parser.error("--share-parameters cannot be combined with --disabled")
    if args.disabled and args.sharing_report:
        parser.error("--sharing-report cannot be combined with --disabled")
    if args.disabled and args.parameter_policy not in (None, "qualified"):
        parser.error("--parameter-policy discover cannot be combined with --disabled")
    parameter_policy = args.parameter_policy or "discover"
    try:
        entries = () if args.disabled else select_entries(args.functional)
    except ValueError as error:
        parser.error(str(error))
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
    for index in range(SHARD_COUNT):
        write_if_changed(
            output / f"generated_libxc_semilocal_cuda_{index}.cu",
            emit_shard(
                index,
                entries,
                share_parameters=args.share_parameters is not False,
                parameter_policy=parameter_policy,
            ),
        )
    if args.sharing_report:
        names = tuple(entry.name for entry in entries)
        candidates = parameter_worker_groups(names)
        groups = (
            parameter_worker_groups(
                names, include_candidates=parameter_policy == "discover"
            )
            if args.share_parameters is not False
            else ()
        )
        shared = {name for group in groups for name in group.names}
        report = {
            "schema": "generativeqc.xc-parameter-sharing-plan.v1",
            "policy": parameter_policy
            if args.share_parameters is not False
            else "specialized",
            "registrations": len(names),
            "shared_registrations": len(shared),
            "candidate_groups": [
                {
                    "names": group.names,
                    "strategy": group.strategy,
                    "contract_identity": canonical_hash(group.contract),
                    "parameter_bytes": 8 * len(group.providers[0].point.parameters),
                    "selected": set(group.names) <= shared,
                    # Eligibility under the default execution policy does not
                    # qualify a new public functional or a different GPU model.
                    "default_qualified": True,
                }
                for group in candidates
            ],
            "specialized_registrations": sorted(set(names) - shared),
            "point_kernels": 4 * (len(names) - len(shared) + len(groups)),
            "qualification": "structural compatibility; no performance or public admission claim",
        }
        write_if_changed(args.sharing_report, json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
