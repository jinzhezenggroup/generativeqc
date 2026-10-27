"""Executable point sharing for bounded native CUDA AOT packages.

Registration/provenance identities remain separate from exact emitted worker
identity. Only token-normalized symbol names are erased for executable reuse;
floors, branches, domain versions, layouts and requested outputs remain inputs.
The existing automatic point compiler remains the sole mathematical owner.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from functools import lru_cache
from typing import TYPE_CHECKING

from generativeqc_compiler.common.paths import asset_path
from generativeqc_compiler.common.provenance import canonical_hash, file_hash
from generativeqc_compiler.dft.xc_point_cuda import (
    emit_automatic_point_kernel,
    emit_device_point_value,
    emit_point_traversal,
    emit_stationary_point_kernel,
)

from .automatic_cuda import AutomaticCudaPointProvider, emit_automatic_cuda_provider
from .automatic_semilocal import automatic_functional_code

if TYPE_CHECKING:
    from .automatic_point import AutomaticPointSource

# Hide generated callbacks and kernel stubs after including shared runtime
# headers. A target-wide hidden default also localizes their inline allocation
# ledger/TLS owners, splitting native clients from the core's lifetime records.
_PRIVATE_BEGIN = """
#if defined(__GNUC__) || defined(__clang__)
#pragma GCC visibility push(hidden)
#endif
"""
_PRIVATE_END = """
#if defined(__GNUC__) || defined(__clang__)
#pragma GCC visibility pop
#endif
"""


@dataclass(frozen=True)
class CudaPointExecutable:
    """One compiled point worker, shared only when its complete ABI agrees."""

    identity: str
    source: str
    stationary: bool

    @property
    def namespace(self) -> str:
        return f"generativeqc::dft::cuda_xc_detail::executable_{self.identity}"


def _executable(
    point: AutomaticPointSource, *, stationary: bool
) -> CudaPointExecutable:
    if point.backend != "cuda":
        raise ValueError("CUDA executable packaging requires a CUDA point source")
    # The emitter owns this identifier. Replace whole tokens only; mathematical
    # constants, branches and provenance remain otherwise untouched.
    body = re.sub(rf"\b{re.escape(point.function_name)}\b", "point", point.source)
    terms = {2: 1, 5: 4, 7: 5, 1: 1}[len(point.features)]
    contract = {
        "schema": "generativeqc.cuda-point-executable.v1",
        "backend": point.backend,
        "body": body,
        "features": point.features,
        "outputs": point.outputs,
        "spin": point.spin,
        "domain": point.domain,
        "domain_version": point.domain_version,
        "ingredient_mask": point.ingredient_mask,
        "stationary": stationary,
        "traversal": emit_point_traversal(),
        "energy_kernel": emit_automatic_point_kernel(),
        "stationary_kernel": emit_stationary_point_kernel() if stationary else "",
        "transport": file_hash(asset_path("src/dft/cuda_xc_point_value.hpp")),
        "emitter": file_hash(
            asset_path("python/generativeqc_compiler/xc/automatic_cuda_package.py")
        ),
    }
    identity = canonical_hash(contract)
    stationary_source = ""
    if stationary:
        stationary_source = (
            emit_stationary_point_kernel()
            + r"""
void launch_stationary(cudaStream_t stream, const double* features, std::size_t count,
                      std::size_t, DevicePointValue* output, int* error,
                      const int* producer_error) {
  stationary_evaluate_points<@TERMS@, Evaluator>
      <<<generativeqc_tensor::blocks(count, 128), 128, 0, stream>>>(
          features, count, output, error, producer_error);
}
""".replace("@TERMS@", str(terms))
        )
    source = f"""
#include "tensor/cuda_runtime.cuh"
#include "dft/cuda_xc_point_program.hpp"
{_PRIVATE_BEGIN}
namespace generativeqc::dft::cuda_xc_detail::executable_{identity} {{
using generativeqc_tensor::I;
using generativeqc_tensor::finite;
{emit_device_point_value(shared=True)}
{body}
struct Evaluator {{
  __device__ static DevicePointValue physical_point(I, const double rho[2],
      const double (&gradient)[2][3], const double tau[2], double, double) {{
    return point(rho, gradient, tau);
  }}
}};
{emit_point_traversal()}
{emit_automatic_point_kernel()}
void launch(cudaStream_t stream, const double* features, const double* weights,
            std::size_t count, std::size_t spins, double* coefficients,
            double* totals, int* error, std::uint32_t functional,
            double exchange, double correlation, const double*) {{
  automatic_evaluate_points<{terms}, false, Evaluator>
      <<<generativeqc_tensor::blocks(count, 128), 128, 0, stream>>>(
          features, weights, count, spins, coefficients, totals, error,
          functional, exchange, correlation, count);
}}
void launch_batch(cudaStream_t stream, const double* features, const double* weights,
                  std::size_t count, std::size_t spins, double* coefficients,
                  double* totals, int* error, std::uint32_t functional,
                  double exchange, double correlation, std::size_t tile_points) {{
  automatic_evaluate_points<{terms}, true, Evaluator>
      <<<generativeqc_tensor::blocks(count, 128), 128, 0, stream>>>(
          features, weights, count, spins, coefficients, totals, error,
          functional, exchange, correlation, tile_points);
}}
{stationary_source}
}}
{_PRIVATE_END}
"""
    return CudaPointExecutable(identity, source, stationary)


def _declarations(worker: CudaPointExecutable) -> str:
    return f"""
namespace {worker.namespace} {{
void launch(cudaStream_t, const double*, const double*, std::size_t, std::size_t,
            double*, double*, int*, std::uint32_t, double, double, const double*);
void launch_batch(cudaStream_t, const double*, const double*, std::size_t, std::size_t,
                  double*, double*, int*, std::uint32_t, double, double, std::size_t);
void launch_stationary(cudaStream_t, const double*, std::size_t, std::size_t,
                      DevicePointValue*, int*, const int*);
}}
"""


def _registration(
    provider: AutomaticCudaPointProvider,
    general: CudaPointExecutable,
    restricted: CudaPointExecutable,
    force_spins: tuple[int, ...],
) -> str:
    """Keep code/domain/provenance validation outside shared mathematical objects."""
    point = provider.point
    code = automatic_functional_code(point.name)
    density_only = provider.feature_terms == 1
    selector = "spins == 1" if density_only else "use_equal_spin_points(spins)"
    restricted_identity = (
        f'"{provider.restricted.expression_identity}"' if density_only else "nullptr"
    )
    stationary = ""
    if force_spins:
        spin_gate = " || ".join(f"spins == {spin}" for spin in force_spins)
        stationary = f"""
void launch_stationary(cudaStream_t stream, const double* features, std::size_t count,
                      std::size_t spins, DevicePointValue* output, int* error,
                      const int* producer_error) {{
  validate(count, spins, {code}U, 1.0, 1.0);
  if (!({spin_gate}) || !features || !output || !error || !count || count > SIZE_MAX / 10)
    throw std::invalid_argument("unpackaged or invalid stationary point layout");
"""
        for spin in force_spins:
            worker = restricted if spin == 1 and density_only else general
            stationary += f"""
  if (spins == {spin}) {worker.namespace}::launch_stationary(
      stream, features, count, spins, output, error, producer_error);
"""
        stationary += "}\n"
    wrapper = f"""
void validate(std::size_t count, std::size_t spins, std::uint32_t code,
              double exchange, double correlation) {{
  if (code != {code}U || (spins != 1 && spins != 2) ||
      count > std::size_t{{std::numeric_limits<std::int64_t>::max()}} / 10 ||
      exchange != 1.0 || correlation != 1.0)
    throw std::invalid_argument("incompatible automatic CUDA XC point binding");
}}
void launch(cudaStream_t stream, const double* features, const double* weights,
            std::size_t count, std::size_t spins, double* coefficients,
            double* totals, int* error, std::uint32_t code,
            double exchange, double correlation, const double* delta) {{
  validate(count, spins, code, exchange, correlation);
  if (delta) throw std::invalid_argument("automatic CUDA XC response is unavailable");
  if (!count) return;
  {"(void)use_equal_spin_points(spins);" if density_only else ""}
  const auto execute = {selector} ? &{restricted.namespace}::launch : &{general.namespace}::launch;
  execute(stream, features, weights, count, spins, coefficients, totals, error,
          code, exchange, correlation, nullptr);
}}
void launch_batch(cudaStream_t stream, const double* features, const double* weights,
                  std::size_t count, std::size_t spins, double* coefficients,
                  double* totals, int* error, std::uint32_t code,
                  double exchange, double correlation, std::size_t tile_points) {{
  validate(count, spins, code, exchange, correlation);
  if (!tile_points || tile_points > std::size_t{{std::numeric_limits<std::int64_t>::max()}})
    throw std::invalid_argument("invalid automatic CUDA XC point tile size");
  if (!count) return;
  {"(void)use_equal_spin_points(spins);" if density_only else ""}
  const auto execute = {selector} ? &{restricted.namespace}::launch_batch : &{general.namespace}::launch_batch;
  execute(stream, features, weights, count, spins, coefficients, totals, error,
          code, exchange, correlation, tile_points);
}}
{stationary}
"""
    identity = canonical_hash(
        {
            "schema": "generativeqc.packaged-cuda-point-source.v1",
            "point": point.source_identity,
            "restricted": provider.restricted.source_identity,
            "general_executable": general.identity,
            "restricted_executable": restricted.identity,
            "wrapper": wrapper,
        }
    )
    from generativeqc_compiler.dft.xc_point_cuda import emit_equal_spin_point_selector

    return f"""
#include <cstdlib>
#include <cstring>
#include <limits>
#include <stdexcept>
{_declarations(general)}
{_declarations(restricted)}
{_PRIVATE_BEGIN}
namespace {provider.namespace} {{
{emit_equal_spin_point_selector()}
{wrapper}
const CudaXcPointProgram program{{
    "{point.name}", "{provider.expression_identity}", "{identity}",
    {code}U, {point.ingredient_mask}U, {point.domain_version}U,
    {provider.feature_terms}, &launch, &launch_batch, {{}},
    "{point.expression_identity}", {restricted_identity},
    {"&launch_stationary" if force_spins else "nullptr"},
    "{general.identity}", "{restricted.identity}"}};
}}
{_PRIVATE_END}
"""


@lru_cache(maxsize=8)
def automatic_cuda_package(
    names: tuple[str, ...], force_profiles: tuple[str, ...] | None = None
) -> tuple[dict[str, CudaPointExecutable], dict[str, str]]:
    """Return unique device workers and separate immutable registrations.

    None includes both force spins; an empty tuple packages energy only.
    Explicit profiles prune force launches without changing energy derivatives.
    """
    workers: dict[str, CudaPointExecutable] = {}
    registrations = {}
    for name in sorted(set(names)):
        provider = emit_automatic_cuda_provider(name)
        code = automatic_functional_code(name)
        spins = tuple(
            spin
            for spin, label in ((1, "rks"), (2, "uks"))
            if force_profiles is None or f"libxc_{code}_{label}" in force_profiles
        )
        density_only = provider.feature_terms == 1
        general = _executable(
            provider.point, stationary=2 in spins or (1 in spins and not density_only)
        )
        restricted = _executable(
            provider.restricted, stationary=1 in spins and density_only
        )
        for worker in (general, restricted):
            workers.setdefault(worker.identity, worker)
        registrations[name] = _registration(provider, general, restricted, spins)
    return dict(sorted(workers.items())), registrations
