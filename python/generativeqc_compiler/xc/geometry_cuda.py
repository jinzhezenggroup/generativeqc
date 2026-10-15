"""CUDA AO translation pullback and Becke local partials from shared graphs."""

import typing
from dataclasses import replace
from fractions import Fraction

from generativeqc_compiler.dft.ao import jet_indices
from generativeqc_compiler.integral.scalar_c import ScalarCEmitter

from ._generated_native_semilocal import (
    SEMILOCAL_FAMILY_BY_CODE,
    SEMILOCAL_FAMILY_CODES,
)
from ._generated_split_hybrids import SPLIT_HYBRIDS
from .coefficients import jet_pullback_program
from .grid_native import emit_grid_adjoint, emit_grid_partials
from .scalar_schedule import XcScalarConsumer, select_xc_scalar_schedule
from .semilocal_codegen import emit_polarized_semilocal, emit_r2scan_program
from .spec import FunctionalSpec
from .wb97mv_maple import DENSITY_THRESHOLD, SIGMA_THRESHOLD, TAU_THRESHOLD

_SPLIT_HYBRID_BY_CODE = {
    record["functional_code"]: (identifier, record)
    for identifier, record in SPLIT_HYBRIDS.items()
}
_REGISTERED_STATIONARY_CODES = SEMILOCAL_FAMILY_CODES | frozenset(_SPLIT_HYBRID_BY_CODE)


def _functional_code(functional: typing.Any, pbe: typing.Any) -> int:
    """Resolve the stationary semilocal selector without weakening old callers."""
    if functional is None:
        if type(pbe) is not bool:
            raise TypeError(
                "geometry lowering requires a registered functional code or a boolean PBE flag"
            )
        return int(pbe)
    if pbe is not None:
        raise ValueError("specify functional or pbe, not both")
    if type(functional) is not int or functional not in _REGISTERED_STATIONARY_CODES:
        raise ValueError(
            "geometry lowering requires a registered semilocal functional code"
        )
    return functional


def _emit_composed_point(
    code: int, semilocal: FunctionalSpec, *, scalar_type: str = "double"
) -> str:
    """Reuse the energy consumer's exact generated graph and boundary policy.

    The force consumer never differentiates clipping or substitutes an interior
    point formula. Imported programs use the same bounded FP64 force schedule
    for either spin and every functional, without altering any SCF point graph.
    Curated programs already lower the same bounded powers after AD.
    """
    polarized = replace(semilocal, spin="polarized")
    curated = SEMILOCAL_FAMILY_BY_CODE.get(code)
    if curated is not None:
        if curated["stationary_kernel"] != "composed":
            raise ValueError("curated stationary kernel is not composition-generated")
        raw_name = "stationary_semilocal_raw"
        body = emit_polarized_semilocal(
            polarized,
            value_type="StationarySemilocalRaw",
            function_name=raw_name,
            identity_constant="kStationarySemilocalIdentity",
            production=True,
            function_qualifier="__device__ inline",
            scalar_type=scalar_type,
        )
        features = 7 if curated["requires_tau"] else 5
    else:
        from .split_hybrid_codegen import emit_split_hybrid_device_body

        try:
            identifier, record = _SPLIT_HYBRID_BY_CODE[code]
        except KeyError as error:
            raise ValueError("unknown composed stationary functional code") from error
        if dict(polarized.components) != {
            name: Fraction(weight) for name, weight in record["components"]
        }:
            raise ValueError("stationary split-hybrid point composition mismatch")
        schedule = select_xc_scalar_schedule(
            consumer=XcScalarConsumer.STATIONARY_FORCE,
            scalar_type=scalar_type,
        )
        body, _, selected = emit_split_hybrid_device_body(
            identifier,
            scalar_type=scalar_type,
            lower_integer_powers=schedule.lower_integer_powers,
        )
        # The generated symbol follows the same canonical registry convention.
        import re

        raw_name = re.sub(r"[^a-z0-9]+", "_", identifier.lower()).strip("_") + "_device"
        features = len(selected)
    arguments = "rho[0], rho[1], sigma[0], sigma[1], sigma[2]"
    if features == 7:
        arguments += ", tau[0], tau[1]"
    return "\n".join(
        (
            body,
            "struct StationaryPointValue {",
            "  double energy{}, rho[2]{}, gradient[2][3]{}, kinetic[2]{};",
            "  bool valid{true};",
            "};",
            "template <bool equal_spin = false>",
            "__device__ inline StationaryPointValue stationary_evaluate_point(",
            "    const double rho[2], const double gradient[2][3], const double tau[2]) {",
            "  StationaryPointValue out;",
            "  double sigma[3]{};",
            "  for (unsigned k = 0; k < 3; ++k) {",
            "    sigma[0] += gradient[0][k] * gradient[0][k];",
            "    sigma[1] += gradient[0][k] * gradient[1][k];",
            "    sigma[2] += gradient[1][k] * gradient[1][k];",
            "  }",
            f"  const auto raw = equal_spin ? {raw_name}_equal_spin({arguments})",
            f"                              : {raw_name}({arguments});",
            "  out.valid = isfinite(raw.energy_density);",
            "  for (double value : raw.feature_derivative) out.valid = out.valid && isfinite(value);",
            "  out.energy = raw.energy_density;",
            "  for (unsigned s = 0; s < 2; ++s) out.rho[s] = raw.feature_derivative[s];",
            "  for (unsigned k = 0; k < 3; ++k) {",
            "    out.gradient[0][k] = 2.0 * raw.feature_derivative[2] * gradient[0][k] + raw.feature_derivative[3] * gradient[1][k];",
            "    out.gradient[1][k] = raw.feature_derivative[3] * gradient[0][k] + 2.0 * raw.feature_derivative[4] * gradient[1][k];",
            "  }",
            *(
                (
                    "  for (unsigned s = 0; s < 2; ++s) out.kinetic[s] = 0.5 * raw.feature_derivative[5 + s];",
                )
                if features == 7
                else ()
            ),
            "  return out;",
            "}",
            "",
        )
    )


def _emit_restricted_point_capability(
    functional: int, semilocal: FunctionalSpec | None
) -> str:
    """Describe mathematics, not runtime proof of an equal-spin producer.

    Every registered unpolarized consumer has an equal-spin lowering. Runtime
    generation/scratch gates must still prove that its producer supplies equal
    features. The old PBE0 composition check preserves established restricted
    mathematics and subnormal policy, not candidate eligibility.
    """
    record = SEMILOCAL_FAMILY_BY_CODE.get(functional)
    legacy_default = (
        record is not None
        and record["stationary_kernel"] == "pbe"
        and semilocal is not None
        and semilocal.spin == "unpolarized"
        and dict(semilocal.components)
        == {"GGA_X_PBE": Fraction(3, 4), "GGA_C_PBE": Fraction(1)}
    )
    capable = (
        semilocal is not None
        and semilocal.spin == "unpolarized"
        and (functional in _REGISTERED_STATIONARY_CODES)
    )
    declaration = [
        "",
        "#include <cstdlib>",
        "#include <cstring>",
        "#include <stdexcept>",
        f"constexpr bool stationary_restricted_point_capable = {str(capable).lower()};",
        "// Retained spelling for the v1 binding-metrics ABI, not a method whitelist.",
        f"constexpr bool stationary_pbe0_restricted_point_capable = {str(capable).lower()};",
        "inline bool stationary_restricted_point_selected(bool requested) {",
        "  if (!requested || !stationary_restricted_point_capable) return false;",
        '  const char* selection = std::getenv("GENERATIVEQC_CUDA_XC_EQUAL_SPIN");',
        '  if (!selection || std::strcmp(selection, "1") == 0) return true;',
        '  if (std::strcmp(selection, "0") == 0) return false;',
        "    throw std::invalid_argument(\"GENERATIVEQC_CUDA_XC_EQUAL_SPIN must be '0' or '1'\");",
        "}",
        "__device__ inline StationaryPointValue stationary_evaluate_restricted_point(",
        "    const double rho[2], const double gradient[2][3], const double tau[2]) {",
    ]
    if not capable:
        return "\n".join(
            [
                *declaration,
                "  return stationary_evaluate_point(rho, gradient, tau);",
                "}",
            ]
        )
    if not legacy_default:
        return "\n".join(
            [
                *declaration,
                "  return stationary_evaluate_point<true>(rho, gradient, tau);",
                "}",
            ]
        )
    return "\n".join(
        [
            *declaration,
            "  const auto raw = generativeqc::dft::point::evaluate_pbe0_restricted_bound(rho[0], gradient[0]);",
            "  StationaryPointValue out;",
            "  out.energy = raw.energy;",
            "  out.valid = raw.valid;",
            "  for (unsigned spin = 0; spin < 2; ++spin) {",
            "    out.rho[spin] = raw.rho[spin];",
            "    for (unsigned axis = 0; axis < 3; ++axis)",
            "      out.gradient[spin][axis] = raw.gradient[spin][axis];",
            "  }",
            "  return out;",
            "}",
        ]
    )


def _emit_stationary_point(
    functional: int,
    *,
    semilocal: FunctionalSpec | None = None,
    scalar_type: str = "double",
) -> str:
    """Emit a point differential with the SCF owner's work-domain semantics.

    Typed composed graphs retain FP64 conditioning and Cartesian conversion.
    Only the guarded caller may select their float graph; this emitter alone
    neither admits molecular Fast nor guarantees an observable error budget.
    The PBE/LDA owner has its own normalized mixed implementation.
    """
    if scalar_type not in ("double", "float"):
        raise ValueError("scalar type must be double or float")
    curated: typing.Mapping[str, typing.Any] | None = SEMILOCAL_FAMILY_BY_CODE.get(
        functional
    )
    if functional in _SPLIT_HYBRID_BY_CODE:
        if not isinstance(semilocal, FunctionalSpec):
            raise ValueError("composed stationary point requires its FunctionalSpec")
        return _emit_composed_point(functional, semilocal, scalar_type=scalar_type)
    if curated is None:
        raise ValueError("unknown stationary semilocal functional code")
    kernel = curated["stationary_kernel"]
    if kernel == "composed":
        if not isinstance(semilocal, FunctionalSpec):
            raise ValueError("composed stationary point requires its FunctionalSpec")
        return _emit_composed_point(functional, semilocal, scalar_type=scalar_type)
    if kernel in ("lda", "pbe"):
        if scalar_type != "double":
            raise ValueError("PBE/LDA float graphs require the normalized mixed owner")
        pbe = "true" if kernel == "pbe" else "false"
        scales = ""
        if semilocal is not None:
            components = dict(semilocal.components)
            component_names = tuple(name for name, _ in curated["components"])
            if len(component_names) != 2:
                raise ValueError("direct stationary point requires two components")
            x_name, c_name = component_names
            if not set(components) <= set(component_names):
                raise ValueError(
                    "stationary point selector disagrees with semilocal components"
                )
            x, c = float(components.get(x_name, 0)), float(components.get(c_name, 0))
            if (x, c) != (1.0, 1.0):
                scales = f", {x.hex()}, {c.hex()}"
        return "\n".join(
            [
                "struct StationaryPointValue {",
                "  double energy{}, rho[2]{}, gradient[2][3]{}, kinetic[2]{};",
                "  bool valid{true};",
                "};",
                "template <bool equal_spin = false>",
                "__device__ inline StationaryPointValue stationary_evaluate_point(",
                "    const double rho[2], const double gradient[2][3], const double tau[2]) {",
                "  const auto raw = equal_spin",
                f"      ? generativeqc::dft::point::evaluate_with_channels<4, true>({pbe}, rho, gradient{scales})",
                f"      : generativeqc::dft::point::evaluate({pbe}, rho, gradient{scales});",
                "  StationaryPointValue out;",
                "  out.energy = raw.energy;",
                "  out.valid = raw.valid;",
                "  for (unsigned s = 0; s < 2; ++s) {",
                "    out.rho[s] = raw.rho[s];",
                "    for (unsigned k = 0; k < 3; ++k) out.gradient[s][k] = raw.gradient[s][k];",
                "  }",
                "  return out;",
                "}",
            ]
        )

    if kernel == "wb97mv":
        if not isinstance(semilocal, FunctionalSpec):
            raise ValueError(
                "omegaB97M-V stationary geometry requires its FunctionalSpec"
            )
        active = {name for name, coefficient in semilocal.components if coefficient}
        expected_components = {name for name, _ in curated["components"]}
        if active != expected_components:
            raise ValueError(
                f"{curated['name']} stationary geometry requires its canonical semilocal components"
            )
        # GridTaskView always supplies alpha/beta features, splitting an RKS
        # density equally. Change only that ABI convention: the MethodIR owns
        # the exact component weights and range parameter for both spin modes.
        polarized_semilocal = (
            semilocal
            if semilocal.spin == "polarized"
            else replace(semilocal, spin="polarized")
        )
        raw = emit_polarized_semilocal(
            polarized_semilocal,
            value_type="StationaryWb97mvRaw",
            function_name="stationary_wb97mv_raw",
            identity_constant="kStationaryWb97mvExpressionIdentity",
            production=True,
            function_qualifier="__device__ inline",
            scalar_type=scalar_type,
        )
        return "\n".join(
            [
                raw.rstrip("\n"),
                "struct StationaryPointValue {",
                "  double energy{}, rho[2]{}, gradient[2][3]{}, kinetic[2]{};",
                "  bool valid{true};",
                "};",
                "template <bool equal_spin = false>",
                "__device__ inline StationaryPointValue stationary_evaluate_point(",
                "    const double rho[2], const double gradient[2][3], const double tau[2]) {",
                "  StationaryPointValue out;",
                "  for (unsigned s = 0; s < 2; ++s) {",
                "    if (!isfinite(rho[s]) || rho[s] < 0.0 || !isfinite(tau[s]) || tau[s] < 0.0) {",
                "      out.valid = false;",
                "      return out;",
                "    }",
                "    for (unsigned k = 0; k < 3; ++k)",
                "      if (!isfinite(gradient[s][k])) { out.valid = false; return out; }",
                "  }",
                "  const double total = rho[0] + rho[1];",
                f"  constexpr double density_threshold = {float(DENSITY_THRESHOLD).hex()};",
                f"  constexpr double sigma_threshold = {float(SIGMA_THRESHOLD).hex()};",
                f"  constexpr double tau_threshold = {float(TAU_THRESHOLD).hex()};",
                "  if (total < density_threshold) return out;",
                "  double sigma[3]{};",
                "  for (unsigned k = 0; k < 3; ++k) {",
                "    sigma[0] += gradient[0][k] * gradient[0][k];",
                "    sigma[1] += gradient[0][k] * gradient[1][k];",
                "    sigma[2] += gradient[1][k] * gradient[1][k];",
                "  }",
                "  const double sigma_floor = sigma_threshold * sigma_threshold;",
                "  double work_rho[2]{fmax(density_threshold, rho[0]), fmax(density_threshold, rho[1])};",
                "  double work_sigma[3]{fmax(sigma_floor, sigma[0]), sigma[1], fmax(sigma_floor, sigma[2])};",
                "  const double sigma_average = 0.5 * (work_sigma[0] + work_sigma[2]);",
                "  work_sigma[1] = fmax(-sigma_average, fmin(sigma_average, work_sigma[1]));",
                "  double work_tau[2]{fmax(tau_threshold, tau[0]), fmax(tau_threshold, tau[1])};",
                "  const auto raw = equal_spin ? stationary_wb97mv_raw_equal_spin(",
                "      work_rho[0], work_rho[1], work_sigma[0], work_sigma[1], work_sigma[2],",
                "      work_tau[0], work_tau[1]) : stationary_wb97mv_raw(",
                "      work_rho[0], work_rho[1], work_sigma[0], work_sigma[1], work_sigma[2],",
                "      work_tau[0], work_tau[1]);",
                "  out.valid = isfinite(raw.energy_density);",
                "  for (double value : raw.feature_derivative)",
                "    out.valid = out.valid && isfinite(value);",
                "  if (!out.valid) return out;",
                "  out.energy = raw.energy_density;",
                "  out.rho[0] = raw.feature_derivative[0];",
                "  out.rho[1] = raw.feature_derivative[1];",
                "  for (unsigned k = 0; k < 3; ++k) {",
                "    out.gradient[0][k] = 2.0 * raw.feature_derivative[2] * gradient[0][k] +",
                "                         raw.feature_derivative[3] * gradient[1][k];",
                "    out.gradient[1][k] = raw.feature_derivative[3] * gradient[0][k] +",
                "                         2.0 * raw.feature_derivative[4] * gradient[1][k];",
                "  }",
                "  out.kinetic[0] = 0.5 * raw.feature_derivative[5];",
                "  out.kinetic[1] = 0.5 * raw.feature_derivative[6];",
                "  return out;",
                "}",
            ]
        )

    if kernel != "r2scan":
        raise ValueError(f"unsupported stationary kernel {kernel!r}")
    spec = FunctionalSpec(
        "stationary-curated-semilocal",
        tuple(
            (name, Fraction(coefficient)) for name, coefficient in curated["components"]
        ),
        spin="polarized",
        range_omega=Fraction(curated["range_omega"]),
    )
    if semilocal is not None and (
        dict(semilocal.components) != dict(spec.components)
        or semilocal.range_omega != spec.range_omega
        or semilocal.exact_exchange != spec.exact_exchange
        or semilocal.long_range_exchange != spec.long_range_exchange
        or semilocal.version != spec.version
    ):
        raise ValueError(
            "R2SCAN stationary geometry requires its canonical semilocal components and parameters"
        )
    # Use the SCF owner's component work-domain policy, including empty spins;
    # differentiating the unfloored interior graph is singular at rho_s=tau_s=0.
    body = emit_r2scan_program(
        value_type="StationaryR2scanRaw",
        function_name="stationary_r2scan_raw",
        identity_constant="kStationaryR2scanIdentity",
        qualifier="__device__ inline",
        scalar_type=scalar_type,
    )
    return (
        body
        + """
struct StationaryPointValue {
  double energy{}, rho[2]{}, gradient[2][3]{}, kinetic[2]{};
  bool valid{true};
};
template <bool equal_spin = false>
__device__ inline StationaryPointValue stationary_evaluate_point(
    const double rho[2], const double gradient[2][3], const double tau[2]) {
  StationaryPointValue out;
  for (unsigned spin = 0; spin < 2; ++spin) {
    out.valid = out.valid && isfinite(rho[spin]) && rho[spin] >= 0.0 &&
                isfinite(tau[spin]) && tau[spin] >= 0.0;
    for (double value : gradient[spin]) out.valid = out.valid && isfinite(value);
  }
  if (!out.valid) return out;
  const double total = rho[0] + rho[1];
  constexpr double tail_low = 1.0e-56, tail_high = 1.0e-52;
  if (total <= tail_low) return out;
  double sigma[3]{};
  for (unsigned k = 0; k < 3; ++k) {
    sigma[0] += gradient[0][k] * gradient[0][k];
    sigma[1] += gradient[0][k] * gradient[1][k];
    sigma[2] += gradient[1][k] * gradient[1][k];
  }
  const auto raw = equal_spin ? stationary_r2scan_raw_equal_spin(
      rho[0], rho[1], sigma[0], sigma[1], sigma[2], tau[0], tau[1])
      : stationary_r2scan_raw(
      rho[0], rho[1], sigma[0], sigma[1], sigma[2], tau[0], tau[1]);
  double energy = raw.energy_density, derivative[7];
  for (unsigned index = 0; index < 7; ++index) derivative[index] = raw.feature_derivative[index];
  out.valid = isfinite(energy);
  for (double value : derivative) out.valid = out.valid && isfinite(value);
  if (!out.valid) return out;
  if (total < tail_high) {
    const double width = tail_high - tail_low;
    const double x = (total - tail_low) / width;
    const double x2 = x * x, x3 = x2 * x;
    const double scale = x3 * (10.0 + x * (-15.0 + 6.0 * x));
    const double dscale = 30.0 * x2 * (1.0 - x) * (1.0 - x) / width;
    const double unscaled = energy;
    energy *= scale;
    derivative[0] = scale * derivative[0] + dscale * unscaled;
    derivative[1] = scale * derivative[1] + dscale * unscaled;
    for (unsigned index = 2; index < 7; ++index) derivative[index] *= scale;
  }
  out.energy = energy;
  out.rho[0] = derivative[0];
  out.rho[1] = derivative[1];
  for (unsigned axis = 0; axis < 3; ++axis) {
    out.gradient[0][axis] = 2.0 * derivative[2] * gradient[0][axis] + derivative[3] * gradient[1][axis];
    out.gradient[1][axis] = derivative[3] * gradient[0][axis] + 2.0 * derivative[4] * gradient[1][axis];
  }
  out.kinetic[0] = 0.5 * derivative[5];
  out.kinetic[1] = 0.5 * derivative[6];
  return out;
}
"""
    )


def emit_geometry_cuda(
    *,
    functional: typing.Any = None,
    pbe: typing.Any = None,
    iterations: typing.Any = 3,
    semilocal: FunctionalSpec | None = None,
) -> typing.Any:
    """Lower AO bilinear AD; the caller supplies one exact semilocal selector."""
    code = _functional_code(functional, pbe)
    if semilocal is not None:
        family = (
            "mgga"
            if "tau" in semilocal.ingredients
            else "gga"
            if "sigma" in semilocal.ingredients
            else "lda"
        )
    else:
        record = SEMILOCAL_FAMILY_BY_CODE.get(code)
        if record is None:
            raise ValueError(
                "generated split-hybrid geometry lowering requires its FunctionalSpec"
            )
        family = (
            "mgga"
            if record["requires_tau"]
            else "gga"
            if record["requires_gradient"]
            else "lda"
        )
    program = jet_pullback_program(family)
    coefficient_count = {"lda": 1, "gga": 4, "mgga": 5}[family]
    variables = {
        **{f"c{j}": f"c[{j}]" for j in range(coefficient_count)},
        **{f"{leg}{j}": f"w[{j}]" for leg in "xy" for j in range(len(program.roots))},
    }
    emitter = ScalarCEmitter(program.graph, variables)
    emitter.emit(program.roots)
    narrowed = ScalarCEmitter(program.graph, variables, scalar_type="float")
    narrowed.emit(program.roots)
    derivative_order = 0 if family == "lda" else 1
    domain = jet_indices(derivative_order)
    lookup = jet_indices(derivative_order + 1)
    shifts = []
    for index in domain:
        row = []
        for k in range(3):
            shifted = list(index)
            shifted[k] += 1
            row.append(lookup.index(tuple(shifted)))
        shifts.append("{" + ",".join(map(str, row)) + "}")
    return "\n".join(
        [
            emit_grid_adjoint(),
            '#include "dft/xc_point.hpp"',
            '#include "dft/xc_point_mixed.hpp"',
            emit_grid_partials(iterations, device=True),
            emit_grid_partials(
                iterations, device=True, scalar_type="float", suffix="_fp32"
            ),
            f"constexpr unsigned stationary_functional = {code};",
            f"constexpr unsigned stationary_jets = {len(domain)};",
            f"constexpr unsigned stationary_ao_jets = {len(lookup)};",
            f"constexpr unsigned stationary_coefficients = {coefficient_count};",
            _emit_stationary_point(code, semilocal=semilocal),
            _emit_restricted_point_capability(code, semilocal),
            _emit_mixed_point(code, semilocal),
            f"__device__ __constant__ unsigned stationary_shift[{len(domain)}][3] = {{{','.join(shifts)}}};",
            "__device__ void ao_pullback(const double* c, const double* w, double* out) {",
            *emitter.lines,
            *(
                f"out[{j}] = {emitter.reference(r)};"
                for j, r in enumerate(program.roots)
            ),
            "}",
            "__device__ void ao_pullback(const float* c, const float* w, float* out) {",
            *narrowed.lines,
            *(
                f"out[{index}] = {narrowed.reference(root)};"
                for index, root in enumerate(program.roots)
            ),
            "}",
            "",
        ]
    )


def _emit_mixed_point(code: int, semilocal: FunctionalSpec | None) -> str:
    """Reuse typed graphs with strict tails and execution-owned work reporting.

    Capability means that a compiler candidate exists, not that a public method
    or budget has been qualified. The runtime still owns that admission. Keep
    input conditioning and Cartesian conversion with their original owner so
    an FP32 candidate cannot introduce a second functional or clipping policy.
    """
    curated = SEMILOCAL_FAMILY_BY_CODE.get(code)
    normalized = curated is not None and curated["stationary_kernel"] in ("lda", "pbe")
    capable = code in _REGISTERED_STATIONARY_CODES
    lines = [
        f"constexpr bool stationary_fp32_point_capable = {str(capable).lower()};",
        "__device__ inline StationaryPointValue stationary_evaluate_point_fp32(",
        "    const double rho[2], const double gradient[2][3], const double tau[2],",
        "    bool* fp32_used = nullptr) {",
        "  if (fp32_used) *fp32_used = false;",
    ]
    if not capable:
        return "\n".join(
            [*lines, "  return stationary_evaluate_point(rho, gradient, tau);", "}"]
        )
    if not normalized:
        if semilocal is not None:
            requires_tau = "tau" in semilocal.ingredients
        elif curated is not None:
            requires_tau = curated["requires_tau"]
        else:
            raise ValueError("split-hybrid mixed geometry requires its FunctionalSpec")
        return "\n".join(
            [
                "namespace stationary_narrowed_point {",
                _emit_stationary_point(code, semilocal=semilocal, scalar_type="float"),
                "}",
                *lines,
                # Near full polarization, cancellation in 1-zeta amplifies
                # float rounding in minority-spin meta-GGA derivatives.
                "  bool interior = rho[0] <= 16.0 * rho[1] && rho[1] <= 16.0 * rho[0];",
                "  for (unsigned spin = 0; spin < 2; ++spin) {",
                "    interior = interior && rho[spin] >= 1e-6 && rho[spin] <= 1e3;",
                "    for (unsigned axis = 0; axis < 3; ++axis)",
                "      interior = interior && isfinite(gradient[spin][axis]) &&",
                "                 fabs(gradient[spin][axis]) <= 100.0 * rho[spin];",
                *(
                    [
                        "    interior = interior && tau[spin] >= 1e-6 && tau[spin] <= 1e6;"
                    ]
                    if requires_tau
                    else []
                ),
                "  }",
                "  if (interior) {",
                "    bool equal_spin = rho[0] == rho[1];",
                "    for (unsigned axis = 0; axis < 3; ++axis)",
                "      equal_spin = equal_spin && gradient[0][axis] == gradient[1][axis];",
                *(
                    ["    equal_spin = equal_spin && tau[0] == tau[1];"]
                    if requires_tau
                    else []
                ),
                "    const auto candidate = equal_spin",
                "        ? stationary_narrowed_point::stationary_evaluate_point<true>(rho, gradient, tau)",
                "        : stationary_narrowed_point::stationary_evaluate_point(rho, gradient, tau);",
                "    if (candidate.valid) {",
                "      StationaryPointValue result;",
                "      result.energy = candidate.energy;",
                "      for (unsigned spin = 0; spin < 2; ++spin) {",
                "        result.rho[spin] = candidate.rho[spin];",
                "        result.kinetic[spin] = candidate.kinetic[spin];",
                "        for (unsigned axis = 0; axis < 3; ++axis)",
                "          result.gradient[spin][axis] = candidate.gradient[spin][axis];",
                "      }",
                "      if (fp32_used) *fp32_used = true;",
                "      return result;",
                "    }",
                "  }",
                "  return stationary_evaluate_point(rho, gradient, tau);",
                "}",
            ]
        )
    exchange = correlation = 1.0
    if semilocal is not None:
        components = dict(semilocal.components)
        exchange_name, correlation_name = (name for name, _ in curated["components"])
        exchange = float(components.get(exchange_name, 0))
        correlation = float(components.get(correlation_name, 0))
    pbe = str(curated["stationary_kernel"] == "pbe").lower()
    return "\n".join(
        [
            *lines,
            "  const auto mixed = generativeqc::dft::point::evaluate_pbe_full_mixed(",
            f"      {pbe}, rho, gradient, {exchange.hex()}, {correlation.hex()});",
            "  if (fp32_used) *fp32_used = mixed.fp32_correlation;",
            "  const auto& raw = mixed.value;",
            "  StationaryPointValue result;",
            "  result.energy = raw.energy;",
            "  result.valid = raw.valid;",
            "  for (unsigned spin = 0; spin < 2; ++spin) {",
            "    result.rho[spin] = raw.rho[spin];",
            "    for (unsigned axis = 0; axis < 3; ++axis)",
            "      result.gradient[spin][axis] = raw.gradient[spin][axis];",
            "  }",
            "  return result;",
            "}",
        ]
    )
