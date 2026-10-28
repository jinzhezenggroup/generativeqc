"""Shared CUDA point ABI and bounded physical-feature traversal.

XC owns the point mathematics. This owner only embeds RKS into physical spins,
walks compact channel-major tiles, and publishes the existing coefficient/totals
layout. Callers supply the native CUDA context (``I``, ``finite`` and CUDA builtins)
and a statically compiled point provider; no runtime compiler or oracle is used.
"""

_POINT_VALUE = r"""struct DevicePointValue {
  double energy{}, rho[2]{}, gradient[2][3]{}, kinetic[2]{};
  bool valid{true};
};
"""

_POINT_TRAVERSAL = r"""
template <I feature_terms, bool response, bool batched, class PointEvaluator>
__device__ __forceinline__ void evaluate_point_tiles(
    const double* input_features, const double* input_weights,
    I domain_count, I spins, double* output_coefficients,
    double* output_totals, int* error, I functional,
    double exchange_scale, double correlation_scale,
    const double* delta, I tile_points) {
  static_assert(feature_terms == 1 || feature_terms == 4 || feature_terms == 5);
  static_assert(!response || feature_terms < 5);
  static_assert(!batched || !response);
  for (I point = I(blockIdx.x) * blockDim.x + threadIdx.x; point < domain_count;
       point += I(blockDim.x) * gridDim.x) {
    // Channel-major slots remain compact for the final partial grid tile.
    const I begin = batched ? (point / tile_points) * tile_points : 0;
    const I count = batched ? min(tile_points, domain_count - begin) : domain_count;
    const I local_point = point - begin;
    const double* features = input_features + begin * spins * feature_terms;
    const double* weights = input_weights + begin;
    double* coefficients = output_coefficients + begin * spins * feature_terms;
    double* point_totals = output_totals + 3 * begin;
    double rho[2]{}, gradient[2][3]{}, tau[2]{};
    for (I spin = 0; spin < 2; ++spin) {
      const I source = spins == 1 ? 0 : spin;
      const double scale = spins == 1 ? 0.5 : 1.0;
      rho[spin] = scale * features[source * feature_terms * count + local_point];
      if (feature_terms >= 4)
        for (I axis = 0; axis < 3; ++axis)
          gradient[spin][axis] = scale * features[
              (source * feature_terms + axis + 1) * count + local_point];
      if (feature_terms == 5)
        tau[spin] = scale * features[(source * feature_terms + 4) * count + local_point];
    }
    DevicePointValue xc;
    if constexpr (response)
      xc = PointEvaluator::response_point(
          features, delta, local_point, count, spins, feature_terms, functional,
          exchange_scale, correlation_scale);
    else
      xc = PointEvaluator::physical_point(
          functional, rho, gradient, tau, exchange_scale, correlation_scale);
    if (!xc.valid) atomicCAS(error, 0, 3);
    point_totals[local_point] = finite(weights[local_point] * xc.energy, error, 2);
    for (I spin = 0; spin < 2; ++spin)
      point_totals[(spin + 1) * count + local_point] =
          finite(weights[local_point] * rho[spin], error, 2);
    for (I spin = 0; spin < spins; ++spin) {
      coefficients[spin * feature_terms * count + local_point] =
          spins == 1 ? 0.5 * (xc.rho[0] + xc.rho[1]) : xc.rho[spin];
      if (feature_terms >= 4)
        for (I axis = 0; axis < 3; ++axis)
          coefficients[(spin * feature_terms + axis + 1) * count + local_point] =
              spins == 1 ? 0.5 * (xc.gradient[0][axis] + xc.gradient[1][axis])
                         : xc.gradient[spin][axis];
      if (feature_terms == 5)
        coefficients[(spin * feature_terms + 4) * count + local_point] =
            spins == 1 ? 0.5 * (xc.kinetic[0] + xc.kinetic[1]) : xc.kinetic[spin];
    }
  }
}
"""

_AUTOMATIC_KERNEL = r"""
template <I feature_terms, bool batched, class PointEvaluator>
__global__ void automatic_evaluate_points(
    const double* features, const double* weights, I count, I spins,
    double* coefficients, double* point_totals, int* error,
    I functional, double exchange_scale, double correlation_scale, I tile_points) {
  evaluate_point_tiles<feature_terms, false, batched, PointEvaluator>(
      features, weights, count, spins, coefficients, point_totals, error,
      functional, exchange_scale, correlation_scale, nullptr, tile_points);
}
"""

_EQUAL_SPIN_SELECTOR = r"""
// Only the RKS feature producer proves paired rho/gradient/tau equality.
// UKS and response must retain the incumbent polarized point program.
bool use_equal_spin_points(std::size_t spins) {
  if (spins != 1) return false;
  const char* selection = std::getenv("GENERATIVEQC_CUDA_XC_EQUAL_SPIN");
  if (!selection || std::strcmp(selection, "1") == 0) return true;
  if (std::strcmp(selection, "0") == 0) return false;
  throw std::invalid_argument("GENERATIVEQC_CUDA_XC_EQUAL_SPIN must be '0' or '1'");
}
"""


def emit_device_point_value(*, shared: bool = False) -> str:
    """Emit the existing physical E/vxc ABI, including the half-vtau coefficient."""
    if shared:
        return "using DevicePointValue = ::generativeqc::dft::DevicePointValue;\n"
    return _POINT_VALUE


def emit_stationary_point_kernel(*, parameter_type: str | None = None) -> str:
    """Publish the same point differential for a staged resident force tile.

    Geometry already owns paired physical spin features, including the RKS
    half-density embedding. Parameter workers pass the same immutable row used
    by energy evaluation; the traversal and physical derivative layout agree.
    """
    source = r"""
template <I feature_terms, class PointEvaluator>
__global__ void stationary_evaluate_points(const double* features, std::size_t count,
    DevicePointValue* output, int* error, const int* producer_error) {
  const auto point = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (point >= count) return;
  if ((producer_error && *producer_error) || *error) {
    atomicExch(error, 1);
    return;
  }
  double rho[2]{}, gradient[2][3]{}, tau[2]{};
  for (unsigned spin = 0; spin < 2; ++spin) {
    rho[spin] = features[5 * spin * count + point];
    if constexpr (feature_terms >= 4)
      for (unsigned axis = 0; axis < 3; ++axis)
        gradient[spin][axis] = features[(5 * spin + axis + 1) * count + point];
    if constexpr (feature_terms == 5)
      tau[spin] = features[(5 * spin + 4) * count + point];
  }
  const auto value = PointEvaluator::physical_point(0, rho, gradient, tau, 1.0, 1.0);
  output[point] = value;
  if (!value.valid) atomicCAS(error, 0, 3);
}
"""
    if parameter_type is not None:
        source = source.replace(
            "const int* producer_error)",
            f"const int* producer_error, {parameter_type} parameters)",
        ).replace("physical_point(0,", "physical_point(parameters,")
    return source


def emit_point_traversal(*, parameter_type: str = "I") -> str:
    """Emit the one shared single/batched tile traversal for static providers.

    Native preparation validates spin/count/tile dimensions. A response provider
    supplies ``response_point``; an ordinary provider supplies ``physical_point``.
    Discarded response branches impose no response/fxc capability on the latter.
    A private worker may carry a typed immutable parameter row instead of the
    ordinary functional code; the physical traversal and output ABI are shared.
    """
    return _POINT_TRAVERSAL.replace("I functional", f"{parameter_type} functional")


def emit_automatic_point_kernel(
    *, parameter_type: str = "I", name: str = "automatic_evaluate_points"
) -> str:
    """Emit a named entry with a functional code or private parameter row.

    Distinct parameter-storage ABIs may name separate entries while sharing the
    same physical traversal. Emitting a template does not instantiate a kernel;
    the package owner selects exactly the entries needed by its launchers.
    """
    return _AUTOMATIC_KERNEL.replace(
        "I functional", f"{parameter_type} functional"
    ).replace("automatic_evaluate_points", name)


def emit_equal_spin_point_selector() -> str:
    """Emit the default-on policy for curated, split and automatic RKS providers.

    The caller supplies the standard cstdlib/cstring/stdexcept includes. Source
    availability grants neither public method admission nor response support.
    """
    return _EQUAL_SPIN_SELECTOR
