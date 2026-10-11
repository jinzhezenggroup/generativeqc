"""Bind automatic point mathematics to the existing CUDA tile traversal.

These are compiler products, not backend admission, installed AOT bundles or a
new cache owner. Native preparation must bind lifetime, target and resource facts
before accepting their launchers. Ordinary execution never invokes this emitter
or NVCC; qualification probes may compile its products explicitly.
"""

from dataclasses import dataclass

from generativeqc_compiler.common.provenance import canonical_hash
from generativeqc_compiler.dft.xc_point_cuda import (
    emit_automatic_point_kernel,
    emit_device_point_value,
    emit_point_traversal,
)

from .automatic_point import AutomaticPointSource, emit_automatic_point
from .automatic_semilocal import automatic_functional_code
from .libxc_work import LIBXC_WORK_DOMAIN_VERSION


@dataclass(frozen=True)
class AutomaticCudaPointProvider:
    """Static provider plus the shared physical-spin/channel-major tile ABI."""

    point: AutomaticPointSource
    provider_name: str
    feature_terms: int
    source: str

    @property
    def namespace(self) -> str:
        """Isolate programs in sharded AOT units without duplicate definitions."""
        return f"generativeqc::dft::cuda_xc_detail::{self.provider_name}"

    @property
    def source_identity(self) -> str:
        """Bind exact traversal, mathematical source and feature layout bytes."""
        return canonical_hash(
            {
                "schema": "generativeqc.automatic-cuda-point-provider.v1",
                "point": self.point.source_identity,
                "provider": self.provider_name,
                "feature_terms": self.feature_terms,
                "source": self.source,
            }
        )


def emit_automatic_cuda_provider(name: str) -> AutomaticCudaPointProvider:
    """Emit a static E/vxc provider using the incumbent CUDA point traversal.

    RKS/UKS remain runtime layouts, not separate mathematical/AOT variants. The
    ordinary provider has no response/fxc entry. Component scaling is rejected
    rather than silently reinterpreting an imported immutable point program.
    Host callers validate tile/spin dimensions before submitting either kernel.
    """
    point = emit_automatic_point(name, backend="cuda")
    feature_terms = {2: 1, 5: 4, 7: 5}[len(point.features)]
    provider_name = f"{point.function_name}_provider"
    code = automatic_functional_code(point.name)
    source = "\n".join(
        (
            '#include "tensor/cuda_runtime.cuh"',
            '#include "dft/cuda_xc_point_program.hpp"',
            "#include <limits>",
            "#include <stdexcept>",
            f"namespace generativeqc::dft::cuda_xc_detail::{provider_name} {{",
            "using generativeqc_tensor::I;",
            "using generativeqc_tensor::finite;",
            emit_device_point_value(),
            point.source,
            f"struct {provider_name} {{",
            "  __device__ static DevicePointValue physical_point(",
            "      I, const double rho[2], const double (&gradient)[2][3],",
            "      const double tau[2], double exchange_scale, double correlation_scale) {",
            "    if (exchange_scale != 1.0 || correlation_scale != 1.0) {",
            "      DevicePointValue invalid;",
            "      invalid.valid = false;",
            "      return invalid;",
            "    }",
            f"    return {point.function_name}(rho, gradient, tau);",
            "  }",
            "};",
            emit_point_traversal(),
            emit_automatic_point_kernel(),
            _emit_launchers(code, feature_terms, provider_name),
            "}  // namespace",
            "",
        )
    )
    return AutomaticCudaPointProvider(point, provider_name, feature_terms, source)


def _emit_launchers(code: int, feature_terms: int, provider_name: str) -> str:
    return f"""
void validate_launch(std::size_t count, std::size_t spins, std::uint32_t functional,
                     double exchange_scale, double correlation_scale) {{
  if (functional != 0x{code:x}U || (spins != 1 && spins != 2) ||
      count > std::size_t{{std::numeric_limits<I>::max()}} /
                  std::max(std::size_t{{3}}, spins * {feature_terms}) ||
      exchange_scale != 1.0 || correlation_scale != 1.0)
    throw std::invalid_argument("incompatible automatic CUDA XC point binding");
}}
void launch(cudaStream_t stream, const double* features, const double* weights,
            std::size_t count, std::size_t spins, double* coefficients,
            double* totals, int* error, std::uint32_t functional,
            double exchange_scale, double correlation_scale, const double* delta) {{
  validate_launch(count, spins, functional, exchange_scale, correlation_scale);
  if (delta) throw std::invalid_argument("automatic CUDA XC response is unavailable");
  if (!count) return;
  automatic_evaluate_points<{feature_terms}, false, {provider_name}>
      <<<generativeqc_tensor::blocks(count, 128), 128, 0, stream>>>(
          features, weights, count, spins, coefficients, totals, error,
          functional, exchange_scale, correlation_scale, count);
}}
void launch_batch(cudaStream_t stream, const double* features, const double* weights,
                  std::size_t count, std::size_t spins, double* coefficients,
                  double* totals, int* error, std::uint32_t functional,
                  double exchange_scale, double correlation_scale, std::size_t tile_points) {{
  validate_launch(count, spins, functional, exchange_scale, correlation_scale);
  if (!tile_points || tile_points > std::size_t{{std::numeric_limits<I>::max()}})
    throw std::invalid_argument("invalid automatic CUDA XC point tile size");
  if (!count) return;
  automatic_evaluate_points<{feature_terms}, true, {provider_name}>
      <<<generativeqc_tensor::blocks(count, 128), 128, 0, stream>>>(
          features, weights, count, spins, coefficients, totals, error,
          functional, exchange_scale, correlation_scale, tile_points);
}}
"""


def emit_automatic_cuda_descriptor(provider: AutomaticCudaPointProvider) -> str:
    """Emit immutable library-owned identities and executable, not public, facts."""
    point = provider.point
    code = automatic_functional_code(point.name)
    return f"""
namespace {provider.namespace} {{
const CudaXcPointProgram program{{
    "{point.name}", "{point.expression_identity}", "{provider.source_identity}",
    0x{code:x}U, {point.ingredient_mask}U, {LIBXC_WORK_DOMAIN_VERSION}U,
    {provider.feature_terms}, &launch, &launch_batch, {{}}}};
}}
"""
