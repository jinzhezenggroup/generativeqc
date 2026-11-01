"""Static native bindings to the existing compiler-owned stationary profiles.

Only identifier ownership and native transport metadata are added. Mathematical
bodies, source ordering, weight coefficients and the generated reduction remain
the same as the independently qualified packaged stationary wrappers.
"""

from __future__ import annotations

import json
import re
from fractions import Fraction
from functools import cache
from typing import TYPE_CHECKING

from generativeqc_compiler.common.paths import asset_path
from generativeqc_compiler.common.provenance import canonical_hash, file_hash
from generativeqc_compiler.dft._generated_native_semilocal import SEMILOCAL_FAMILIES
from generativeqc_compiler.tensor.cuda_inline import exact_cuda_literal
from generativeqc_compiler.xc._generated_split_hybrids import SPLIT_HYBRIDS
from generativeqc_compiler.xc.automatic_semilocal import (
    automatic_composition_name,
    automatic_functional_code,
    automatic_scf_domain,
)
from generativeqc_compiler.xc.density_work import (
    DENSITY_WORK_DOMAIN,
    DENSITY_WORK_DOMAIN_VERSION,
)
from generativeqc_compiler.xc.libxc_work import LIBXC_WORK_DOMAIN_VERSION

from .ks_execution import compile_ks_execution_plan
from .stationary_cuda import (
    AUTOMATIC_STATIONARY_AOT_PROFILE_NAMES,
    COMPOSITE_STATIONARY_CUDA_EXTERNAL_SOURCES,
    QUALIFIED_PARTITION_ITERATIONS,
    QUALIFIED_STATIONARY_AOT_PROFILE_NAMES,
    _qualified_aot_profile,
    emit_stationary_profile_aot_wrapper_cuda,
    emit_stationary_wrapper_cuda,
    qualified_sp_requests,
    stationary_aot_profile_contract_identity,
    stationary_runtime_sources,
)

if TYPE_CHECKING:
    from pathlib import Path


NATIVE_STATIONARY_CUDA_PROFILES = (
    tuple(
        profile
        for profile in QUALIFIED_STATIONARY_AOT_PROFILE_NAMES
        if set((plan := _qualified_aot_profile(profile).plan).external_source_names)
        <= set(COMPOSITE_STATIONARY_CUDA_EXTERNAL_SOURCES)
    )
    + AUTOMATIC_STATIONARY_AOT_PROFILE_NAMES
)


def native_stationary_model(profile: str) -> tuple[int, int, Fraction, Fraction]:
    """Derive native point identity/scales from the graph and generated metadata.

    Descriptive method/profile names never decide scientific admission. Fixed
    point programs require their exact component graph; scalable programs carry
    the actual outer X/C weights, including nonstandard hybrid compositions.
    """
    plan = compile_ks_execution_plan(_qualified_aot_profile(profile).plan.method)
    functional = plan.semilocal.functional
    actual = dict(functional.components)
    if profile.startswith("libxc_"):
        if (
            plan.nonlocal_correlation
            or plan.post_scf
            or functional.range_omega
            or len(plan.exchange) > 1
            or any(term.operator != "full-range" for term in plan.exchange)
        ):
            raise ValueError(
                "automatic stationary model requires a semilocal/global-hybrid recipe"
            )
        name = automatic_composition_name(
            functional.components,
            plan.exchange[0].coefficient if plan.exchange else Fraction(0),
        )
        if name is None:
            raise ValueError("unregistered automatic stationary composition")
        domain = (
            DENSITY_WORK_DOMAIN_VERSION
            if automatic_scf_domain(name) == DENSITY_WORK_DOMAIN
            else LIBXC_WORK_DOMAIN_VERSION
        )
        return (
            automatic_functional_code(name),
            domain,
            Fraction(1),
            Fraction(1),
        )
    for record in SPLIT_HYBRIDS.values():
        expected = {
            name: Fraction(coefficient) for name, coefficient in record["components"]
        }
        if actual != expected or functional.range_omega != 0:
            continue
        if (
            len(plan.exchange) != 1
            or plan.exchange[0].operator != "full-range"
            or plan.exchange[0].omega != 0
            or plan.exchange[0].coefficient != Fraction(record["exact_exchange"])
            or plan.nonlocal_correlation is not None
            or plan.post_scf
        ):
            raise ValueError(
                "native split-hybrid stationary model requires its exact graph"
            )
        # Native KS already publishes split-global-hybrid domain v4. The
        # point program owns its internal X/C coefficients; the stationary
        # wrapper carries the separately verified exact-exchange weight.
        return record["functional_code"], 4, Fraction(1), Fraction(1)
    for record in SEMILOCAL_FAMILIES:
        expected = {
            name: Fraction(coefficient) for name, coefficient in record["components"]
        }
        if functional.range_omega != Fraction(record["range_omega"]):
            continue
        if record["coefficient_policy"] == "native-scales":
            if not actual.keys() <= expected.keys():
                continue
            exchange, correlation = (
                actual.get(name, Fraction(0)) / coefficient
                for name, coefficient in expected.items()
            )
        elif actual == expected:
            exchange = correlation = Fraction(1)
        else:
            continue
        if exchange < 0 or correlation < 0:
            raise ValueError("native stationary point scales must be nonnegative")
        return record["code"], record["domain_version"], exchange, correlation
    raise ValueError("stationary graph has no native semilocal execution metadata")


def emit_native_stationary_profile_cuda(
    profile: str, *, staged_point: bool = False, source_cache: Path | None = None
) -> str:
    """Isolate one unchanged scientific wrapper and emit its typed native table.

    Preprocessor token aliases cover the included runtime as well as generated
    code; unlike textual replacement they cannot change literals or provenance
    comments. The shared external ``first_derivative`` symbol is never aliased.
    """
    if profile not in NATIVE_STATIONARY_CUDA_PROFILES:
        raise ValueError(f"unsupported native stationary CUDA profile: {profile}")
    selected = _qualified_aot_profile(profile)
    plan = selected.plan
    sources = stationary_runtime_sources(plan, include_external=True)
    if sources != plan.source_names or len(sources) > 64:
        raise ValueError(
            "native stationary binding requires a complete bounded source inventory"
        )
    # Native source reduction is a complete ordered unit sum in ProgramIR.
    # Its only CUDA kernel is linked once for all admitted method graphs.
    reduction_program = plan.reduction_program(atoms=1)
    reduction = reduction_program.outputs["gradient"]
    if (
        reduction.op != "add"
        or reduction.spec.shape != (1, 3)
        or reduction.spec.dtype != "float64"
        or any(
            node.op != "input"
            or node.spec.shape != (1, 3)
            or node.spec.dtype != "float64"
            for node in reduction.inputs
        )
        or tuple(node.attrs.get("name") for node in reduction.inputs) != sources
        or tuple(reduction.attrs.get("coefficients", ())) != ((1, 1),) * len(sources)
    ):
        raise ValueError(
            "native shared CUDA reduction requires the exact unit-source sum"
        )
    shared = native_stationary_shared_group(profile) if staged_point else None
    wrapper = (
        emit_stationary_wrapper_cuda(
            functional=selected.functional,
            plan=plan,
            include_reduction=False,
            staged_point=True,
            source_cache=source_cache,
            include_external=True,
        )
        if staged_point
        else emit_stationary_profile_aot_wrapper_cuda(
            profile,
            include_reduction=False,
            source_cache=source_cache,
            include_external=True,
        )
    )
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
        f"{name}{suffix}"
        for name in (
            "local_ratio_geometry",
            "local_ratio_prepared",
            "local_norm",
            "local_ratio",
            "local_log",
            "local_becke",
        )
        for suffix in ("", "_fp32")
    )
    prefix = f"generativeqc_native_{shared or profile}_"
    source_identity = canonical_hash(
        {
            "schema": "generativeqc.native-stationary-source.v1",
            "contract": (
                native_stationary_shared_contract(profile)
                if shared
                else stationary_aot_profile_contract_identity(profile)
            ),
            "wrapper": wrapper,
            "transport": transport,
            "runtime_header": file_hash(
                asset_path("src/dft/stationary_gradient_cuda.cuh")
            ),
            "point_abi": file_hash(asset_path("src/dft/cuda_xc_point_value.hpp")),
            "binding_header": file_hash(
                asset_path("src/dft/cuda_stationary_binding.hpp")
            ),
            "shared_kernels": file_hash(
                asset_path("src/dft/native_stationary_shared_kernels.cu")
            ),
        }
    )
    aliases = "\n".join(
        f"#define {name} {prefix}{name}" for name in sorted(identifiers)
    )
    slots = {name: sources.index(name) for name in sources}
    exchange_slot = slots.get("exact_exchange", -1)
    integral_mask = sum(
        1 << slots[name]
        for name in (
            "one_electron",
            "coulomb",
            "overlap_pulay",
            "exact_exchange",
            "exchange_short_range",
            "exchange_long_range",
        )
        if name in slots
    )
    nuclear_kind = qualified_sp_requests().index(("nuclear", ()))
    exchange = exact_cuda_literal(plan.exchange.coefficient if plan.exchange else 0)
    functional, domain, exchange_scale, correlation_scale = native_stationary_model(
        profile
    )
    if functional != selected.functional:
        raise ValueError("stationary profile and native point selector disagree")
    profile_name = shared or profile
    plan_identity = source_identity if shared else plan.identity
    method_identity = "" if shared else plan.method.identity
    if shared:
        functional = domain = 0
    execution = compile_ks_execution_plan(plan.method)
    ranged = {
        term.operator: term
        for term in execution.exchange
        if term.operator != "full-range"
    }
    sr = ranged.get("short-range")
    lr = ranged.get("long-range")
    if (sr is None) != (lr is None) or (
        sr is not None and lr is not None and sr.omega != lr.omega
    ):
        raise ValueError("native stationary range graph requires matched SR/LR")
    nlc = execution.nonlocal_correlation
    if bool(ranged) != bool(nlc):
        raise ValueError(
            "native stationary composite requires complete range/VV10 providers"
        )
    range_values = ", ".join(
        exact_cuda_literal(value)
        for value in (
            sr.coefficient if sr else 0,
            lr.coefficient if lr else 0,
            sr.omega if sr else 0,
            nlc.spec.b if nlc else 0,
            nlc.spec.c if nlc else 0,
            nlc.coefficient if nlc else 0,
        )
    )
    # Partition response has no XC or spin dependence. Every admitted native
    # profile shares the existing large-system crossover and resource guards.
    metadata = f"""
namespace generativeqc_stationary_cuda {{
constexpr double native_short_exchange = {exact_cuda_literal(sr.coefficient if sr else 0)};
constexpr double native_long_exchange = {exact_cuda_literal(lr.coefficient if lr else 0)};
constexpr int native_short_exchange_slot = {slots.get("exchange_short_range", -1)};
constexpr int native_long_exchange_slot = {slots.get("exchange_long_range", -1)};
constexpr int native_nonlocal_slot = {slots.get("nonlocal_ao", -1)};
constexpr size_t native_one_electron_slot = {slots["one_electron"]};
constexpr size_t native_coulomb_slot = {slots["coulomb"]};
constexpr size_t native_overlap_slot = {slots["overlap_pulay"]};
constexpr int native_exchange_slot = {exchange_slot};
constexpr size_t native_exchange_storage_slot = {max(exchange_slot, 0)};
constexpr uint64_t native_exchange_mask = {0 if exchange_slot < 0 else 1 << exchange_slot};
constexpr uint64_t native_required_integral_mask = {integral_mask};
constexpr unsigned native_nuclear_kind = {nuclear_kind};
constexpr size_t native_phased_becke_min_atoms = 48;
}}
#include "dft/native_stationary_runtime.cuh"
namespace generativeqc::dft {{
const CudaStationaryBinding& native_stationary_profile_{profile_name}() {{
  using namespace generativeqc_stationary_cuda;
  static const char* const sources[] = {{{", ".join(json.dumps(name) for name in sources)}}};
  static const CudaStationaryBinding binding{{
      {json.dumps(profile_name)}, {json.dumps(plan_identity)}, {json.dumps(method_identity)},
      sources, {len(sources)}, {plan.spin_blocks}, {exchange}, {nuclear_kind},
      stationary_geometry_max_lanes, stationary_geometry_max_scratch_bytes,
      stationary_geometry_max_threads, stationary_ao_jets, stationary_jets,
      stationary_jets > 1 ? 11U : 1U,
      native_numeric_bytes, native_create, native_topology, native_reset_geometry, native_nuclear,
      native_consume_integrals, native_geometry<false>, native_finish_gradient,
      native_source_span, native_metrics, native_destroy,
      {{{functional}U, {domain}U, {plan.spin_blocks}U, {QUALIFIED_PARTITION_ITERATIONS}U,
       {exact_cuda_literal(exchange_scale)}, {exact_cuda_literal(correlation_scale)},
       {exchange}, {range_values}}},
      stationary_fp32_point_capable ? native_geometry<true> : nullptr, native_grid_topology,
      native_nonlocal_slot >= 0 ? native_nonlocal_geometry : nullptr,
      {json.dumps(source_identity)}, nullptr,
      {"native_bind_point" if shared else "nullptr"},
      {json.dumps(source_identity) if shared else "nullptr"},
      {json.dumps(profile_name)}, {"native_point_metrics" if shared else "nullptr"}
  }};
  return binding;
}}
}}
"""
    return (
        '#include "dft/cuda_stationary_binding.hpp"\n'
        "#define GENERATIVEQC_NATIVE_SHARED_REDUCTION 1\n"
        + ("#define GENERATIVEQC_NATIVE_STAGED_POINT 1\n" if shared else "")
        + f"// stationary-reduction-program: {reduction_program.logical_hash}\n"
        + aliases
        + "\n"
        + wrapper
        + metadata
    )


@cache
def native_stationary_shared_contract(profile: str) -> dict:
    """Authenticate the actual force-consumer ABI independently of XC identity.

    Sharing requires the exact source inventory, weights, reduction, ingredient
    layout and spin convention. Registration names or a loose family label
    cannot establish equivalence. Backend/target/toolchain remain build inputs.
    """
    if profile not in AUTOMATIC_STATIONARY_AOT_PROFILE_NAMES:
        raise ValueError("shared automatic consumer requires an automatic profile")
    selected = _qualified_aot_profile(profile)
    plan = selected.plan
    execution = compile_ks_execution_plan(plan.method)
    # This also rejects unregistered recipes, range terms and corrections.
    native_stationary_model(profile)
    from .stationary_cuda import stationary_aot_profile_weight_programs

    return {
        "schema": "generativeqc.stationary-shared-consumer.v1",
        "ingredients": execution.semilocal.functional.ingredients,
        "spin_blocks": plan.spin_blocks,
        "sources": stationary_runtime_sources(plan),
        "weights": stationary_aot_profile_weight_programs(profile),
        "reduction": plan.reduction_program(atoms=1).logical_hash,
        "partition_iterations": QUALIFIED_PARTITION_ITERATIONS,
        "precision": "float64",
        "point_layout": "physical-spin-DevicePointValue-v1",
        "schedule": "bounded-or-cooperative-or-phased-v1",
    }


def native_stationary_shared_group(profile: str) -> str:
    """Name a consumer by its complete compiler-owned execution contract."""
    return "shared_" + canonical_hash(native_stationary_shared_contract(profile))[:24]


def emit_native_stationary_profile_cpp(profile: str) -> str:
    """Bind scientific provenance to a shared consumer without another CUDA TU."""
    selected = _qualified_aot_profile(profile)
    plan = selected.plan
    group = native_stationary_shared_group(profile)
    functional, domain, exchange, correlation = native_stationary_model(profile)
    # The shared consumer owns integral weights, while each registration must
    # still authenticate the same physical EXX fraction in its public model.
    full_range = exact_cuda_literal(plan.exchange.coefficient if plan.exchange else 0)
    identity = canonical_hash(
        {
            "schema": "generativeqc.stationary-staged-binding.v1",
            "contract": stationary_aot_profile_contract_identity(profile),
            "consumer": native_stationary_shared_contract(profile),
            "emitter": file_hash(
                asset_path(
                    "python/generativeqc_compiler/method/stationary_native_cuda.py"
                )
            ),
            "binding": file_hash(asset_path("src/dft/cuda_stationary_binding.hpp")),
        }
    )
    return f"""// Generated immutable point/scientific registration; no CUDA kernels.
#include "dft/cuda_stationary_binding.hpp"
#include "generated_libxc_semilocal_cuda_registry.hpp"
namespace generativeqc::dft {{
const CudaStationaryBinding& native_stationary_profile_{group}();
const CudaStationaryBinding& native_stationary_profile_{profile}() {{
  static const CudaStationaryBinding binding = [] {{
    auto value = native_stationary_profile_{group}();
    value.profile = {json.dumps(profile)};
    value.plan_identity = {json.dumps(plan.identity)};
    value.method_identity = {json.dumps(plan.method.identity)};
    value.source_identity = {json.dumps(identity)};
    value.model = {{{functional}U, {domain}U, {plan.spin_blocks}U,
                    {QUALIFIED_PARTITION_ITERATIONS}U,
                    {exact_cuda_literal(exchange)}, {exact_cuda_literal(correlation)}, {full_range}}};
    value.point_program = generated::automatic_cuda_point_entry({functional}U);
    return value;
  }}();
  return binding;
}}
}}
"""


def native_stationary_package_sources(
    profiles: tuple[str, ...], *, fused_profiles: tuple[str, ...] = ()
) -> dict[str, str]:
    """Map deterministic output names to their representative profile.

    A .cpp file groups provenance/point bindings for one shared consumer. A
    shared .cu file is emitted once per authenticated consumer ABI. Explicit fused specializations remain
    available for measured hot packages; they are never runtime compilations.
    """
    if len(set(profiles)) != len(profiles) or set(profiles) - set(
        NATIVE_STATIONARY_CUDA_PROFILES
    ):
        raise ValueError("invalid native stationary package profiles")
    if set(fused_profiles) - set(profiles):
        raise ValueError("fused profiles must belong to the selected force package")
    sources = {}
    for profile in sorted(profiles):
        if (
            profile in AUTOMATIC_STATIONARY_AOT_PROFILE_NAMES
            and profile not in fused_profiles
        ):
            group = native_stationary_shared_group(profile)
            sources.setdefault(f"native_{group}_bindings.cpp", profile)
            sources.setdefault(f"native_{group}.cu", profile)
        else:
            sources[f"native_{profile}.cu"] = profile
    return sources


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
        "std::span<const CudaStationaryBinding* const> cuda_stationary_bindings() noexcept {"
    )
    if profiles:
        lines.append("  static const CudaStationaryBinding* const bindings[] = {")
        lines.extend(
            f"    &native_stationary_profile_{profile}(),"
            for profile in sorted(profiles)
        )
        lines.extend(("  };", "  return bindings;"))
    else:
        lines.append("  return {};")
    lines.extend(
        (
            "}",
            "const CudaStationaryBinding* cuda_stationary_binding(std::string_view profile) noexcept {",
            "  for (const auto* binding : cuda_stationary_bindings())",
            "    if (profile == binding->profile && (!binding->bind_point || binding->point_program)) return binding;",
            "  return nullptr;",
            "}",
            "const CudaStationaryBinding* cuda_stationary_binding(const CudaStationaryModel& model) noexcept {",
            "  for (const auto* binding : cuda_stationary_bindings())",
            "    if (model == binding->model && (!binding->bind_point || binding->point_program)) return binding;",
            "  return nullptr;",
            "}",
            "}",
            "",
        )
    )
    return "\n".join(lines)
