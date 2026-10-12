"""Static native bindings to the existing compiler-owned stationary profiles.

Only identifier ownership and native transport metadata are added. Mathematical
bodies, source ordering, weight coefficients and the generated reduction remain
the same as the independently qualified packaged stationary wrappers.
"""

from __future__ import annotations

import json
import re

from generativeqc_compiler.common.paths import asset_path
from generativeqc_compiler.tensor.cuda_inline import exact_cuda_literal

from .stationary_cuda import (
    _qualified_aot_profile,
    emit_stationary_profile_aot_wrapper_cuda,
    qualified_sp_requests,
    stationary_runtime_sources,
)

NATIVE_STATIONARY_CUDA_PROFILES = ("pbe_rks", "pbe_uks", "pbe0_rks", "pbe0_uks")


def emit_native_stationary_profile_cuda(profile: str) -> str:
    """Isolate one unchanged scientific wrapper and emit its typed native table.

    Preprocessor token aliases cover the included runtime as well as generated
    code; unlike textual replacement they cannot change literals or provenance
    comments. The shared external ``first_derivative`` symbol is never aliased.
    """
    if profile not in NATIVE_STATIONARY_CUDA_PROFILES:
        raise ValueError(f"unsupported native stationary CUDA profile: {profile}")
    selected = _qualified_aot_profile(profile)
    plan = selected.plan
    sources = stationary_runtime_sources(plan)
    if sources != plan.source_names or len(sources) > 64:
        raise ValueError(
            "native stationary binding requires a complete bounded source inventory"
        )
    wrapper = emit_stationary_profile_aot_wrapper_cuda(profile)
    runtime = asset_path("src/dft/stationary_gradient_cuda.cuh").read_text()
    transport = asset_path("src/dft/native_stationary_runtime.cuh").read_text()
    identifiers = set(
        re.findall(r"\bstationary_[A-Za-z0-9_]+\b", wrapper + runtime + transport)
    )
    identifiers.update(re.findall(r"\bgenerativeqc_grid_[A-Za-z0-9_]+\b", wrapper))
    identifiers.update(
        ("generativeqc_stationary_cuda", "StationaryPointValue", "ao_pullback")
    )
    identifiers.update(
        (
            "local_ratio_geometry",
            "local_ratio_prepared",
            "local_norm",
            "local_ratio",
            "local_log",
            "local_becke",
        )
    )
    prefix = f"generativeqc_native_{profile}_"
    aliases = "\n".join(
        f"#define {name} {prefix}{name}" for name in sorted(identifiers)
    )
    slots = {name: sources.index(name) for name in sources}
    exchange_slot = slots.get("exact_exchange", -1)
    integral_mask = sum(
        1 << slots[name]
        for name in ("one_electron", "coulomb", "overlap_pulay", "exact_exchange")
        if name in slots
    )
    nuclear_kind = qualified_sp_requests().index(("nuclear", ()))
    exchange = exact_cuda_literal(plan.exchange.coefficient if plan.exchange else 0)
    metadata = f"""
namespace generativeqc_stationary_cuda {{
constexpr size_t native_one_electron_slot = {slots["one_electron"]};
constexpr size_t native_coulomb_slot = {slots["coulomb"]};
constexpr size_t native_overlap_slot = {slots["overlap_pulay"]};
constexpr int native_exchange_slot = {exchange_slot};
constexpr size_t native_exchange_storage_slot = {max(exchange_slot, 0)};
constexpr uint64_t native_exchange_mask = {0 if exchange_slot < 0 else 1 << exchange_slot};
constexpr uint64_t native_required_integral_mask = {integral_mask};
constexpr unsigned native_nuclear_kind = {nuclear_kind};
}}
#include "dft/native_stationary_runtime.cuh"
namespace generativeqc::dft {{
const CudaStationaryBinding& native_stationary_profile_{profile}() {{
  using namespace generativeqc_stationary_cuda;
  static const char* const sources[] = {{{", ".join(json.dumps(name) for name in sources)}}};
  static const CudaStationaryBinding binding{{
      {json.dumps(profile)}, {json.dumps(plan.identity)}, {json.dumps(plan.method.identity)},
      sources, {len(sources)}, {plan.spin_blocks}, {exchange}, {nuclear_kind},
      stationary_geometry_max_lanes, stationary_geometry_max_scratch_bytes,
      stationary_geometry_max_threads, stationary_ao_jets, stationary_jets,
      stationary_jets > 1 ? 11U : 1U,
      native_numeric_bytes, native_create, native_topology, native_reset_geometry, native_nuclear,
      native_consume_integrals, native_geometry, native_finish_gradient,
      native_source_span, native_metrics, native_destroy}};
  return binding;
}}
}}
"""
    return (
        '#include "dft/cuda_stationary_binding.hpp"\n'
        + aliases
        + "\n"
        + wrapper
        + metadata
    )


def emit_native_stationary_registry_cpp(profiles: tuple[str, ...]) -> str:
    """Emit a deterministic registry containing only the linked profile objects."""
    if len(set(profiles)) != len(profiles) or any(
        profile not in NATIVE_STATIONARY_CUDA_PROFILES for profile in profiles
    ):
        raise ValueError("invalid native stationary CUDA registry inventory")
    lines = [
        '#include "dft/cuda_stationary_binding.hpp"',
        "namespace generativeqc::dft {",
    ]
    for profile in sorted(profiles):
        lines.append(
            f"const CudaStationaryBinding& native_stationary_profile_{profile}();"
        )
    lines.append(
        "const CudaStationaryBinding* cuda_stationary_binding(std::string_view profile) noexcept {"
    )
    for profile in sorted(profiles):
        lines.append(
            f'  if (profile == "{profile}") return &native_stationary_profile_{profile}();'
        )
    lines.extend(("  return nullptr;", "}", "}", ""))
    return "\n".join(lines)
