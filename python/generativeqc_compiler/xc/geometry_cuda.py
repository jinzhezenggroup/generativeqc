"""CUDA AO translation pullback and Becke local partials from shared graphs."""

import typing
from dataclasses import replace
from fractions import Fraction

from generativeqc_compiler.dft.ao import jet_indices
from generativeqc_compiler.integral.expr import AlgebraForm
from generativeqc_compiler.integral.scalar_c import ScalarCEmitter

from ._generated_native_semilocal import (
    SEMILOCAL_FAMILY_BY_CODE,
    SEMILOCAL_FAMILY_CODES,
)
from ._generated_split_hybrids import SPLIT_HYBRIDS
from .coefficients import jet_pullback_program
from .grid_native import emit_grid_adjoint, emit_grid_partials
from .semilocal_codegen import emit_polarized_semilocal
from .semilocal_family import energy_expression
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


def _emit_composed_point(code: int, semilocal: FunctionalSpec) -> str:
    """Reuse the energy consumer's exact generated graph and boundary policy.

    The force consumer changes only the rho/sigma/tau-to-Cartesian ABI; it
    never differentiates clipping or substitutes an interior point formula.
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
        body, _, selected = emit_split_hybrid_device_body(identifier)
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
            "__device__ inline StationaryPointValue stationary_evaluate_point(",
            "    const double rho[2], const double gradient[2][3], const double tau[2]) {",
            "  StationaryPointValue out;",
            "  double sigma[3]{};",
            "  for (unsigned k = 0; k < 3; ++k) {",
            "    sigma[0] += gradient[0][k] * gradient[0][k];",
            "    sigma[1] += gradient[0][k] * gradient[1][k];",
            "    sigma[2] += gradient[1][k] * gradient[1][k];",
            "  }",
            f"  const auto raw = {raw_name}({arguments});",
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


def _emit_stationary_point(
    functional: int, *, semilocal: FunctionalSpec | None = None
) -> str:
    """Emit the exact SCF-domain point differential consumed by geometry CUDA."""
    curated: typing.Mapping[str, typing.Any] | None = SEMILOCAL_FAMILY_BY_CODE.get(
        functional
    )
    if functional in _SPLIT_HYBRID_BY_CODE:
        if not isinstance(semilocal, FunctionalSpec):
            raise ValueError("composed stationary point requires its FunctionalSpec")
        return _emit_composed_point(functional, semilocal)
    if curated is None:
        raise ValueError("unknown stationary semilocal functional code")
    kernel = curated["stationary_kernel"]
    if kernel == "composed":
        if not isinstance(semilocal, FunctionalSpec):
            raise ValueError("composed stationary point requires its FunctionalSpec")
        return _emit_composed_point(functional, semilocal)
    if kernel in ("lda", "pbe"):
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
                "__device__ inline StationaryPointValue stationary_evaluate_point(",
                "    const double rho[2], const double gradient[2][3], const double tau[2]) {",
                f"  const auto raw = generativeqc::dft::point::evaluate({pbe}, rho, gradient{scales});",
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
        )
        return "\n".join(
            [
                raw.rstrip("\n"),
                "struct StationaryPointValue {",
                "  double energy{}, rho[2]{}, gradient[2][3]{}, kinetic[2]{};",
                "  bool valid{true};",
                "};",
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
                "  const auto raw = stationary_wb97mv_raw(",
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
    graph, energy, feature_variables = energy_expression(spec, production=True)
    roots = (
        energy,
        *(graph.differentiate(energy, value) for value in feature_variables),
    )
    graph, roots = graph.apply_algebra_form(roots, AlgebraForm.FACTORED_NARY)
    graph, roots = graph.lower_small_integer_powers(roots)
    variables = {
        "rho_a": "rho[0]",
        "rho_b": "rho[1]",
        "sigma_aa": "sigma[0]",
        "sigma_ab": "sigma[1]",
        "sigma_bb": "sigma[2]",
        "tau_a": "tau[0]",
        "tau_b": "tau[1]",
    }
    emitter = ScalarCEmitter(graph, variables)
    emitter.emit(roots)
    refs = [emitter.reference(root) for root in roots]
    return "\n".join(
        [
            "struct StationaryPointValue {",
            "  double energy{}, rho[2]{}, gradient[2][3]{}, kinetic[2]{};",
            "  bool valid{true};",
            "};",
            "__device__ inline StationaryPointValue stationary_evaluate_point(",
            "    const double rho[2], const double gradient[2][3], const double tau[2]) {",
            "  StationaryPointValue out;",
            "  const double total = rho[0] + rho[1];",
            "  constexpr double tail_low = 1.0e-56, tail_high = 1.0e-52;",
            "  if (total <= tail_low) return out;",
            "  double sigma[3]{};",
            "  for (unsigned k = 0; k < 3; ++k) {",
            "    sigma[0] += gradient[0][k] * gradient[0][k];",
            "    sigma[1] += gradient[0][k] * gradient[1][k];",
            "    sigma[2] += gradient[1][k] * gradient[1][k];",
            "  }",
            *emitter.lines,
            f"  double energy = {refs[0]};",
            "  double derivative[7]{" + ", ".join(refs[1:]) + "};",
            "  out.valid = isfinite(energy);",
            "  for (double value : derivative) out.valid = out.valid && isfinite(value);",
            "  if (!out.valid) return out;",
            "  if (total < tail_high) {",
            "    const double width = tail_high - tail_low;",
            "    const double x = (total - tail_low) / width;",
            "    const double x2 = x * x, x3 = x2 * x;",
            "    const double scale = x3 * (10.0 + x * (-15.0 + 6.0 * x));",
            "    const double dscale = 30.0 * x2 * (1.0 - x) * (1.0 - x) / width;",
            "    const double unscaled = energy;",
            "    energy *= scale;",
            "    derivative[0] = scale * derivative[0] + dscale * unscaled;",
            "    derivative[1] = scale * derivative[1] + dscale * unscaled;",
            "    for (unsigned i = 2; i < 7; ++i) derivative[i] *= scale;",
            "  }",
            "  out.energy = energy;",
            "  out.rho[0] = derivative[0];",
            "  out.rho[1] = derivative[1];",
            "  for (unsigned k = 0; k < 3; ++k) {",
            "    out.gradient[0][k] = 2.0 * derivative[2] * gradient[0][k] + derivative[3] * gradient[1][k];",
            "    out.gradient[1][k] = derivative[3] * gradient[0][k] + 2.0 * derivative[4] * gradient[1][k];",
            "  }",
            "  out.kinetic[0] = 0.5 * derivative[5];",
            "  out.kinetic[1] = 0.5 * derivative[6];",
            "  return out;",
            "}",
        ]
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
            emit_grid_partials(iterations, device=True),
            f"constexpr unsigned stationary_functional = {code};",
            f"constexpr unsigned stationary_jets = {len(domain)};",
            f"constexpr unsigned stationary_ao_jets = {len(lookup)};",
            f"constexpr unsigned stationary_coefficients = {coefficient_count};",
            _emit_stationary_point(code, semilocal=semilocal),
            f"__device__ __constant__ unsigned stationary_shift[{len(domain)}][3] = {{{','.join(shifts)}}};",
            "__device__ void ao_pullback(const double* c, const double* w, double* out) {",
            *emitter.lines,
            *(
                f"out[{j}] = {emitter.reference(r)};"
                for j, r in enumerate(program.roots)
            ),
            "}",
            "",
        ]
    )
