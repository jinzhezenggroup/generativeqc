"""Share compatible point executables without merging scientific registrations.

Mathematics still comes from the pinned Maple importer and ordinary physical
point lowering. Only scalar coefficients become immutable device data. Family
membership proposes a prepared-coefficient group; the general discovery path
compares specialized arithmetic across the entire selected catalog, independent
of source-entry names. Exact bodies and both spin ABIs must agree in either case.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, replace
from functools import lru_cache
from typing import Literal

from generativeqc_compiler.common.provenance import canonical_hash
from generativeqc_compiler.dft.xc_point_cuda import (
    emit_automatic_point_kernel,
    emit_device_point_value,
    emit_equal_spin_point_selector,
    emit_point_traversal,
)
from generativeqc_compiler.integral.scalar_c import format_constant

from .automatic_cuda import (
    AutomaticCudaPointProvider,
    emit_automatic_cuda_provider,
    emit_automatic_launch_validation,
)
from .automatic_point import AutomaticPointSource, emit_automatic_point
from .automatic_semilocal import automatic_functional_code
from .libxc_bulk import read_catalog

# Preserve the measured post-AD product optimization, which can join workers
# with different literal CSE shapes. All other entries use name-independent
# discovery below; enrollment here does not limit the general planner's scope.
_PREPARED_FAMILY_ENTRIES = frozenset({"maple/mgga_exc/mgga_x_ktbm.mpl"})
# Leave room for the ordinary point-launch arguments under CUDA's portable
# 4 KiB kernel-parameter limit, even on toolchains lacking larger parameter space.
_MAX_PARAMETER_VALUES = 384
_PRIVATE_BEGIN = "#if defined(__GNUC__) || defined(__clang__)\n#pragma GCC visibility push(hidden)\n#endif\n"
_PRIVATE_END = (
    "#if defined(__GNUC__) || defined(__clang__)\n#pragma GCC visibility pop\n#endif\n"
)
_REGISTRATION_PRELUDE = """#include "dft/cuda_xc_point_program.hpp"
#include <cstdlib>
#include <cstring>
#include <limits>
#include <stdexcept>
"""


# Only the worker needs tensor execution helpers. Including cuda_runtime.cuh in
# every host registration would also emit its unrelated conversion kernels.
_PRELUDE = '#include "tensor/cuda_runtime.cuh"\n' + _REGISTRATION_PRELUDE


def _body(point: AutomaticPointSource, identifier: str) -> str:
    """Erase only the emitter-owned function token, keeping guards and constants."""
    return re.sub(rf"\b{re.escape(point.function_name)}\b", identifier, point.source)


def _contract(point: AutomaticPointSource) -> dict:
    return {
        "body": _body(point, "point"),
        "features": point.features,
        "outputs": point.outputs,
        "spin": point.spin,
        "domain": point.domain,
        "domain_version": point.domain_version,
        "ingredients": point.ingredient_mask,
        "parameters": tuple(name for name, _ in point.parameters),
    }


@dataclass(frozen=True)
class ParameterWorkerGroup:
    """One exact paired/general execution contract and its immutable row bindings."""

    strategy: str
    providers: tuple[AutomaticCudaPointProvider, ...]
    # Keep the established prepared-worker package identity stable. General
    # discovery binds its complete source/ABI contract directly instead.
    prepared_key: str | None = None

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(provider.point.name for provider in self.providers)

    @property
    def contract(self) -> dict:
        provider = self.providers[0]
        return {
            "point": _contract(provider.point),
            "restricted": _contract(provider.restricted),
        }


def _bind_constant_group(
    providers: list[AutomaticCudaPointProvider],
) -> ParameterWorkerGroup:
    """Keep common slots literal and deduplicate identical varying columns.

    General and density-only unpolarized Graphs need not have the same constant
    lists. Bind each independent list into a common launch row; never reinterpret
    the polarized point as an unpolarized density worker.
    """
    columns = tuple(
        zip(
            *(
                tuple(
                    value
                    for _, value in (*p.point.parameters, *p.restricted.parameters)
                )
                for p in providers
            ),
            strict=True,
        )
    )
    slots: dict[tuple[float, ...], int] = {}
    references = []
    for column in columns:
        if len(set(column)) == 1:
            references.append(format_constant(column[0]))
        else:
            index = slots.setdefault(column, len(slots))
            references.append(f"parameters[{index}]")
    general_count = len(providers[0].point.parameters)

    def bind(
        point: AutomaticPointSource,
        offset: int,
        values: tuple[tuple[str, float], ...],
    ) -> AutomaticPointSource:
        # These tokens belong to the point emitter's coefficient ABI, not
        # arbitrary C++ expressions. Substitution changes only leaf bindings.
        source = re.sub(
            r"\bparameters\[(\d+)\]",
            lambda match: references[offset + int(match[1])],
            point.source,
        )
        return replace(point, source=source, parameters=values)

    result = []
    for row, provider in enumerate(providers):
        values = tuple((f"coefficient_{i}", column[row]) for column, i in slots.items())

        result.append(
            replace(
                provider,
                point=bind(provider.point, 0, values),
                restricted=bind(provider.restricted, general_count, values),
            )
        )
    return ParameterWorkerGroup("specialized-constants", tuple(result))


def _constant_groups(
    providers: dict[str, AutomaticCudaPointProvider],
) -> tuple[ParameterWorkerGroup, ...]:
    """Discover compatible optimized bodies across entries, kinds and families."""
    candidates: dict[str, list[AutomaticCudaPointProvider]] = {}
    for name, original in providers.items():
        point = emit_automatic_point(name, backend="cuda", parameterize_constants=True)
        density_only = original.feature_terms == 1
        restricted = emit_automatic_point(
            name,
            backend="cuda",
            equal_spin=not density_only,
            spin="unpolarized" if density_only else "polarized",
            parameterize_constants=True,
        )
        candidate = replace(original, point=point, restricted=restricted)
        identity = canonical_hash(
            {"point": _contract(point), "restricted": _contract(restricted)}
        )
        candidates.setdefault(identity, []).append(candidate)
    groups = []
    for members in candidates.values():
        if len(members) < 2:
            continue
        group = _bind_constant_group(members)
        if len(group.providers[0].point.parameters) <= _MAX_PARAMETER_VALUES:
            groups.append(group)
    return tuple(groups)


def _worker(
    namespace: str,
    providers: list[AutomaticCudaPointProvider],
    parameters: tuple[str, ...],
    values: list[list[float]],
) -> str:
    """Emit one paired/general producer family with uniform immutable parameters."""
    provider = providers[0]
    table = ",\n".join(
        "{" + ",".join(float(value).hex() for value in row) + "}" for row in values
    )
    adapters = ""
    for label, point in (
        ("General", provider.point),
        ("Restricted", provider.restricted),
    ):
        function = "point_" + label.lower()
        arguments = ", row.values" if "const double* parameters" in point.source else ""
        adapters += (
            _body(point, function)
            + f"""
struct {label} {{
  __device__ static DevicePointValue physical_point(const ParameterRow& row, const double rho[2],
      const double (&gradient)[2][3], const double tau[2], double, double) {{
    return {function}(rho, gradient, tau{arguments});
  }}
}};
"""
        )
        if label == "Restricted":
            # Pair only the batched adapter: eliminating redundant input work
            # helps its tile-address/parameter-load schedule, while applying
            # the same rewrite to ordinary launches can regress their latency.
            # Both adapters call the same mathematical body; replacing the
            # existing batched instantiation adds no point kernel variants.
            adapters += f"""
struct RestrictedBatched {{
  __device__ static DevicePointValue physical_point(const ParameterRow& row, const double rho[2],
      const double (&gradient)[2][3], const double tau[2], double, double) {{
    // The host selector proves RKS, hence identical physical spin channels.
    // Expose that identity to native CSE across the parameterized point call;
    // the common traversal still owns runtime output and tile addressing.
    const double paired_rho[2]{{rho[0], rho[0]}};
    const double paired_gradient[2][3]{{
        {{gradient[0][0], gradient[0][1], gradient[0][2]}},
        {{gradient[0][0], gradient[0][1], gradient[0][2]}}}};
    const double paired_tau[2]{{tau[0], tau[0]}};
    return {function}(paired_rho, paired_gradient, paired_tau{arguments});
  }}
}};
"""
    launches = ""
    for batched in (False, True):
        name = "launch_batch" if batched else "launch"
        restricted_evaluator = "RestrictedBatched" if batched else "Restricted"
        # Grid-constant storage is qualified only for the ordinary general ABI.
        # Applying it to batched or paired kernels can worsen their scheduling;
        # this replaces one instantiation and retains four kernels per group.
        general_kernel = (
            "automatic_evaluate_points"
            if batched
            else "automatic_evaluate_ordinary_points"
        )
        launches += f"""
void {name}(cudaStream_t stream, const double* features, const double* weights,
    std::size_t count, std::size_t spins, double* coefficients, double* totals,
    int* error, std::size_t row, bool restricted, std::size_t tile_points) {{
  if (restricted)
    automatic_evaluate_points<{provider.feature_terms}, {str(batched).lower()}, {restricted_evaluator}>
        <<<generativeqc_tensor::blocks(count, 128), 128, 0, stream>>>(
            features, weights, count, spins, coefficients, totals, error, parameter_rows[row], 1.0, 1.0, tile_points);
  else
    {general_kernel}<{provider.feature_terms}, {str(batched).lower()}, General>
        <<<generativeqc_tensor::blocks(count, 128), 128, 0, stream>>>(
            features, weights, count, spins, coefficients, totals, error, parameter_rows[row], 1.0, 1.0, tile_points);
}}
"""
    return f"""{_PRELUDE}{_PRIVATE_BEGIN}
namespace {namespace} {{
using generativeqc_tensor::I;
using generativeqc_tensor::finite;
{emit_device_point_value()}
// The host authenticates one immutable row and CUDA copies it into launch
// parameter space. Fixed offsets avoid per-thread device-table address loads;
// concurrent launches never mutate shared device state.
struct ParameterRow {{ double values[{max(1, len(parameters))}]; }};
constexpr ParameterRow parameter_rows[{len(values)}] = {{
{table}
}};
{adapters}
{emit_point_traversal(parameter_type="const ParameterRow&")}
{emit_automatic_point_kernel(parameter_type="const ParameterRow")}
{emit_automatic_point_kernel(parameter_type="const __grid_constant__ ParameterRow", name="automatic_evaluate_ordinary_points")}
{launches}
}}
{_PRIVATE_END}
"""


def _registration(
    provider: AutomaticCudaPointProvider, namespace: str, row: int
) -> str:
    """Validate public identities before forwarding to a private family worker."""
    declarations = "\n".join(
        f"void {name}(cudaStream_t, const double*, const double*, std::size_t, std::size_t, "
        "double*, double*, int*, std::size_t, bool, std::size_t);"
        for name in ("launch", "launch_batch")
    )
    code = automatic_functional_code(provider.point.name)
    selector = (
        "spins == 1" if provider.feature_terms == 1 else "use_equal_spin_points(spins)"
    )
    validate_selector = (
        "  (void)use_equal_spin_points(spins);\n" if provider.feature_terms == 1 else ""
    )
    return f"""{_REGISTRATION_PRELUDE}{_PRIVATE_BEGIN}
namespace {namespace} {{
{declarations}
}}
namespace {provider.namespace} {{
using I = std::int64_t;
{emit_equal_spin_point_selector()}
{emit_automatic_launch_validation(code, provider.feature_terms)}
void launch(cudaStream_t stream, const double* features, const double* weights,
    std::size_t count, std::size_t spins, double* coefficients, double* totals,
    int* error, std::uint32_t functional, double exchange, double correlation, const double* delta) {{
  validate_launch(count, spins, functional, exchange, correlation);
  if (delta) throw std::invalid_argument("automatic CUDA XC response is unavailable");
  if (!count) return;
{validate_selector}\
  {namespace}::launch(stream, features, weights, count, spins, coefficients, totals,
                      error, {row}, {selector}, count);
}}
void launch_batch(cudaStream_t stream, const double* features, const double* weights,
    std::size_t count, std::size_t spins, double* coefficients, double* totals,
    int* error, std::uint32_t functional, double exchange, double correlation, std::size_t tile_points) {{
  validate_launch(count, spins, functional, exchange, correlation);
  if (!tile_points || tile_points > std::size_t{{std::numeric_limits<I>::max()}})
    throw std::invalid_argument("invalid automatic CUDA XC point tile size");
  if (!count) return;
{validate_selector}\
  {namespace}::launch_batch(stream, features, weights, count, spins, coefficients, totals,
                            error, {row}, {selector}, tile_points);
}}
}}
{_PRIVATE_END}
"""


def parameter_worker_groups(
    names: tuple[str, ...], *, include_candidates: bool = True
) -> tuple[ParameterWorkerGroup, ...]:
    """Plan exact worker sharing over the entire selected automatic XC catalog.

    Single-registration groups use the existing specialization. A group includes
    both polarized and equal-spin contracts and exact ordered parameter names;
    source-entry membership alone cannot make different executions share code.
    Structural discovery is the default execution policy. The legacy qualified
    policy can restrict selection to the established prepared-product workers.
    """
    return _parameter_worker_plan(tuple(sorted(set(names))), include_candidates)[1]


@lru_cache(maxsize=8)
def _parameter_worker_plan(
    names: tuple[str, ...], include_candidates: bool
) -> tuple[dict[str, AutomaticCudaPointProvider], tuple[ParameterWorkerGroup, ...]]:
    """Retain literal providers once for both discovery and package emission."""
    records = {r["name"]: r for r in read_catalog()["registrations"]}
    providers = {
        name: emit_automatic_cuda_provider(name) for name in sorted(set(names))
    }
    groups: dict[str, list[AutomaticCudaPointProvider]] = {}
    for name, original in providers.items():
        record = records[name]
        if record["entry"] not in _PREPARED_FAMILY_ENTRIES:
            continue
        parameters = tuple(
            sorted(key for key in record["bindings"] if key.startswith("params_a_"))
        )
        point = emit_automatic_point(
            name, backend="cuda", parameter_names=parameters, precompute_parameters=True
        )
        restricted = emit_automatic_point(
            name,
            backend="cuda",
            equal_spin=True,
            parameter_names=parameters,
            precompute_parameters=True,
        )
        if point.parameters != restricted.parameters:
            raise ValueError("shared XC point paths require the same parameter row")
        parameters = tuple(name for name, _ in point.parameters)
        provider = replace(original, point=point, restricted=restricted)
        key = canonical_hash(
            {
                "entry": record["entry"],
                "parameters": parameters,
                "point": _contract(point),
                "restricted": _contract(restricted),
            }
        )
        groups.setdefault(key, []).append(provider)
    planned = []
    for key, members in sorted(groups.items()):
        if len(members) < 2:
            continue
        planned.append(ParameterWorkerGroup("prepared-products", tuple(members), key))
    claimed = {name for group in planned for name in group.names}
    if include_candidates:
        planned.extend(
            _constant_groups(
                {name: p for name, p in providers.items() if name not in claimed}
            )
        )
    return providers, tuple(sorted(planned, key=lambda group: group.names))


@lru_cache(maxsize=8)
def parameterized_cuda_package(
    names: tuple[str, ...],
    *,
    policy: Literal["qualified", "discover"] = "discover",
) -> tuple[dict[str, str], dict[str, AutomaticCudaPointProvider]]:
    """Share exact compatible workers by default, with a legacy policy rollback.

    Discovery proves compatible mathematics/ABIs, not universal performance or
    public method admission. Incompatible, oversized and single-entry contracts
    keep literal specialization; ``qualified`` retains prepared-product groups.
    """
    if policy not in ("qualified", "discover"):
        raise ValueError("XC parameter sharing policy must be qualified or discover")
    originals, groups = _parameter_worker_plan(
        tuple(sorted(set(names))), policy == "discover"
    )
    # The planner's canonical providers are cached and must remain immutable.
    providers = dict(originals)
    workers = {}
    for group in groups:
        parameters = tuple(name for name, _ in group.providers[0].point.parameters)
        values = [
            [value for _, value in provider.point.parameters]
            for provider in group.providers
        ]
        # Table rows and their values are immutable module data and must bind
        # the worker symbol too; another package cannot interpose a different row.
        template = _worker(
            "generativeqc::dft::cuda_xc_detail::parameter_family",
            list(group.providers),
            parameters,
            values,
        )
        identity = canonical_hash(
            {
                "contract": group.prepared_key or group.contract,
                "rows": list(zip(group.names, values, strict=True)),
                "executable_source": template,
            }
        )
        namespace = f"generativeqc::dft::cuda_xc_detail::parameter_family_{identity}"
        workers[identity] = _worker(
            namespace, list(group.providers), parameters, values
        )
        for row, name in enumerate(group.names):
            # CPU/CUDA admission compares the canonical registered mathematics,
            # including its actual coefficients. Keep that identity; bind this
            # alternative execution lowering through the complete worker source
            # identity and the authenticated immutable parameter row instead.
            original = providers[name]
            providers[name] = replace(
                original, source=_registration(original, namespace, row)
            )
    return workers, providers
