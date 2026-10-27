"""Share audited parameter-family point executables without merging registrations.

Mathematics still comes from the pinned Maple importer and ordinary physical
point lowering. Only scalar coefficients become immutable device data. Family
membership proposes a group; exact parameterized bodies and complete point
layouts must agree before the group can share a CUDA worker.
"""

from __future__ import annotations

import re
from dataclasses import replace
from functools import lru_cache

from generativeqc_compiler.common.provenance import canonical_hash
from generativeqc_compiler.dft.xc_point_cuda import (
    emit_automatic_point_kernel,
    emit_device_point_value,
    emit_equal_spin_point_selector,
    emit_point_traversal,
)

from .automatic_cuda import (
    AutomaticCudaPointProvider,
    emit_automatic_cuda_provider,
    emit_automatic_launch_validation,
)
from .automatic_point import AutomaticPointSource, emit_automatic_point
from .automatic_semilocal import automatic_functional_code
from .libxc_bulk import read_catalog

# This bounds an executable optimization, not functional admission. Other
# registrations retain their existing specialized emitters and public coverage.
_FAMILY_ENTRIES = frozenset({"maple/mgga_exc/mgga_x_ktbm.mpl"})
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
    }


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
        adapters += (
            _body(point, function)
            + f"""
struct {label} {{
  __device__ static DevicePointValue physical_point(I row, const double rho[2],
      const double (&gradient)[2][3], const double tau[2], double, double) {{
    return {function}(rho, gradient, tau, coefficients[row]);
  }}
}};
"""
        )
    launches = ""
    for batched in (False, True):
        name = "launch_batch" if batched else "launch"
        launches += f"""
void {name}(cudaStream_t stream, const double* features, const double* weights,
    std::size_t count, std::size_t spins, double* coefficients, double* totals,
    int* error, std::size_t row, bool restricted, std::size_t tile_points) {{
  if (restricted)
    automatic_evaluate_points<{provider.feature_terms}, {str(batched).lower()}, Restricted>
        <<<generativeqc_tensor::blocks(count, 128), 128, 0, stream>>>(
            features, weights, count, spins, coefficients, totals, error, row, 1.0, 1.0, tile_points);
  else
    automatic_evaluate_points<{provider.feature_terms}, {str(batched).lower()}, General>
        <<<generativeqc_tensor::blocks(count, 128), 128, 0, stream>>>(
            features, weights, count, spins, coefficients, totals, error, row, 1.0, 1.0, tile_points);
}}
"""
    return f"""{_PRELUDE}{_PRIVATE_BEGIN}
namespace {namespace} {{
using generativeqc_tensor::I;
using generativeqc_tensor::finite;
{emit_device_point_value()}
// Each grid uses one authenticated parameter row. No per-point method switch.
__device__ __constant__ double coefficients[{len(values)}][{len(parameters)}] = {{
{table}
}};
{adapters}
{emit_point_traversal()}
{emit_automatic_point_kernel()}
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
  {namespace}::launch(stream, features, weights, count, spins, coefficients, totals,
                      error, {row}, use_equal_spin_points(spins), count);
}}
void launch_batch(cudaStream_t stream, const double* features, const double* weights,
    std::size_t count, std::size_t spins, double* coefficients, double* totals,
    int* error, std::uint32_t functional, double exchange, double correlation, std::size_t tile_points) {{
  validate_launch(count, spins, functional, exchange, correlation);
  if (!tile_points || tile_points > std::size_t{{std::numeric_limits<I>::max()}})
    throw std::invalid_argument("invalid automatic CUDA XC point tile size");
  if (!count) return;
  {namespace}::launch_batch(stream, features, weights, count, spins, coefficients, totals,
                            error, {row}, use_equal_spin_points(spins), tile_points);
}}
}}
{_PRIVATE_END}
"""


@lru_cache(maxsize=8)
def parameterized_cuda_package(
    names: tuple[str, ...],
) -> tuple[dict[str, str], dict[str, AutomaticCudaPointProvider]]:
    """Return shared units and per-registration providers for the selected package.

    Single-registration groups use the existing specialization. A group includes
    both polarized and equal-spin contracts and exact ordered parameter names;
    source-entry membership alone cannot make different executions share code.
    """
    records = {r["name"]: r for r in read_catalog()["registrations"]}
    providers = {
        name: emit_automatic_cuda_provider(name) for name in sorted(set(names))
    }
    groups: dict[
        str, list[tuple[str, AutomaticCudaPointProvider, tuple[str, ...], list[float]]]
    ] = {}
    for name, original in providers.items():
        record = records[name]
        if record["entry"] not in _FAMILY_ENTRIES:
            continue
        parameters = tuple(
            sorted(key for key in record["bindings"] if key.startswith("params_a_"))
        )
        point = emit_automatic_point(name, backend="cuda", parameter_names=parameters)
        restricted = emit_automatic_point(
            name, backend="cuda", equal_spin=True, parameter_names=parameters
        )
        provider = replace(original, point=point, restricted=restricted)
        key = canonical_hash(
            {
                "entry": record["entry"],
                "parameters": parameters,
                "point": _contract(point),
                "restricted": _contract(restricted),
            }
        )
        values = [float(record["bindings"][key]) for key in parameters]
        groups.setdefault(key, []).append((name, provider, parameters, values))
    workers = {}
    for key, members in sorted(groups.items()):
        if len(members) < 2:
            continue
        # Table rows and their values are immutable module data and must bind
        # the worker symbol too; another package cannot interpose a different row.
        template = _worker(
            "generativeqc::dft::cuda_xc_detail::parameter_family",
            [m[1] for m in members],
            members[0][2],
            [m[3] for m in members],
        )
        identity = canonical_hash(
            {
                "contract": key,
                "rows": [(m[0], m[3]) for m in members],
                "executable_source": template,
            }
        )
        namespace = f"generativeqc::dft::cuda_xc_detail::parameter_family_{identity}"
        workers[identity] = _worker(
            namespace, [m[1] for m in members], members[0][2], [m[3] for m in members]
        )
        for row, (name, _, _, _) in enumerate(members):
            # CPU/CUDA admission compares the canonical registered mathematics,
            # including its actual coefficients. Keep that identity; bind this
            # alternative execution lowering through the complete worker source
            # identity and the authenticated immutable parameter row instead.
            original = providers[name]
            providers[name] = replace(
                original, source=_registration(original, namespace, row)
            )
    return workers, providers
