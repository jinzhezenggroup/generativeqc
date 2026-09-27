"""Lightweight XC family dispatch shared by runtime and code generation."""

from __future__ import annotations

import typing

from .bulk_runtime import build_bulk_runtime_program
from .rsh_expressions import energy_expression as rsh_energy_expression
from .semilocal_family import energy_expression as semilocal_energy_expression
from .spec import (
    AUTO_BULK_COMPONENTS,
    SPECIAL_EXPRESSION_COMPONENTS,
    WB97MV_COMPONENTS,
    UnsupportedXC,
)
from .wb97mv_maple import energy_expression as wb97mv_energy_expression


def build_pointwise_energy_expression(spec: typing.Any) -> typing.Any:
    """Bridge one automatic Libxc Graph into shared native pointwise lowering.

    This is deliberately the interior mathematical kernel. Physical zero-spin,
    zero-gradient, density-tail, and tau boundaries belong to libxc_work.
    Native/public execution must evaluate derivatives at work coordinates rather
    than differentiating through the work transformation.
    """
    active = tuple(
        (name, coefficient) for name, coefficient in spec.components if coefficient
    )
    if not active or any(name not in AUTO_BULK_COMPONENTS for name, _ in active):
        raise UnsupportedXC(
            "bulk pointwise native bridge requires only automatic Libxc components"
        )
    if len(active) != 1 or active[0][1] != 1:
        raise UnsupportedXC(
            "bulk pointwise native bridge currently requires one unit-weight component"
        )
    program = build_bulk_runtime_program(active[0][0], spin=spec.spin, order=0)
    expected_features = tuple(spec.features[: len(program.spec.features)])
    if program.spec.features != expected_features:
        raise UnsupportedXC(
            "bulk pointwise native bridge requires a rho/sigma/tau prefix feature layout"
        )
    # Keep the common FunctionalSpec ABI width even when the imported expression
    # does not consume every trailing feature. Graph.variable() interns missing
    # ABI variables, so differentiation produces exact zeros instead of indexing
    # past the compact Libxc runtime projection.
    variables = tuple(program.graph.variable(name) for name in spec.features)
    return program.graph, program.roots[0], variables


def build_energy_expression(
    spec: typing.Any, *, production: bool = False
) -> typing.Any:
    """Build the family-selected scalar energy DAG without runtime dependencies."""
    if type(production) is not bool:
        raise TypeError("production must be bool")
    active = {name for name, coefficient in spec.components if coefficient}
    automatic = active & set(AUTO_BULK_COMPONENTS)
    if automatic:
        if automatic != active:
            raise UnsupportedXC(
                "automatic Libxc components cannot mix with a separately-owned XC family"
            )
        if len(automatic) != 1:
            raise UnsupportedXC(
                "automatic Libxc lowering currently requires one functional component"
            )
        return build_pointwise_energy_expression(spec)
    if active & set(WB97MV_COMPONENTS):
        if not active <= set(WB97MV_COMPONENTS):
            raise UnsupportedXC(
                "omegaB97M-V semilocal components cannot be mixed with another XC family"
            )
        return wb97mv_energy_expression(spec)
    if active & set(SPECIAL_EXPRESSION_COMPONENTS):
        return rsh_energy_expression(spec, production=production)
    return semilocal_energy_expression(spec, production=production)
