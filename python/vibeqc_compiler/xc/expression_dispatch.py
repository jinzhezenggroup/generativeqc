# Copyright (C) 2017 M.A.L. Marques
# Copyright (C) 2026 VibeQC contributors
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. See upstream/libxc/7.0.0/COPYING or https://mozilla.org/MPL/2.0/.
"""Lightweight XC family dispatch shared by runtime and code generation."""

from __future__ import annotations

import math
import typing
from fractions import Fraction as F

from vibeqc_compiler.integral.expr import Expr, Graph

from . import ityh_maple
from .b3lyp_production_policy import b88_exchange as production_b88_exchange
from .b3lyp_production_policy import lyp_correlation as production_lyp_correlation
from .b88_vwn_maple import b88_exchange as maple_b88_exchange
from .b88_vwn_maple import vwn_correlation as maple_vwn_correlation
from .p86_pz_maple import p86_correlation, pz_correlation
from .pw91_maple import pw91_correlation as imported_pw91_correlation
from .pw91_maple import pw91_exchange as imported_pw91_exchange
from .pw_maple import pw_correlation
from .rsh_maple import lyp_correlation
from .semilocal_family import energy_expression as semilocal_energy_expression
from .spec import (
    AUTO_BULK_COMPONENTS,
    SPECIAL_EXPRESSION_COMPONENTS,
    WB97MV_COMPONENTS,
    UnsupportedXC,
)
from .wb97mv_maple import energy_expression as wb97mv_energy_expression


def _build_special_energy_expression(
    spec: typing.Any, *, production: bool = False
) -> typing.Any:
    """Compose retained pinned-Libxc adapters without a legacy dispatch module."""
    if type(production) is not bool:
        raise TypeError("production must be bool")
    graph = Graph()
    variables = tuple(graph.variable(name) for name in spec.features)
    if spec.spin == "polarized":
        ra, rb = variables[:2]
    else:
        rho = variables[0]
        ra = rb = rho / 2
    n = ra + rb
    cx = F(3, 8) * (3 / math.pi) ** (1 / 3) * 4 ** (2 / 3)

    def lda_exchange() -> Expr:
        return graph.sum(-cx * density.pow(4 / 3) for density in (ra, rb))

    builders = {
        "LDA_X": lda_exchange,
        "GGA_X_B88": lambda: (
            production_b88_exchange(graph, spec, variables)
            if production
            else maple_b88_exchange(graph, spec, variables)
        ),
        "GGA_X_ITYH": lambda: ityh_maple.ityh_exchange(graph, spec, variables),
        "GGA_X_PW91": lambda: imported_pw91_exchange(graph, spec, variables),
        "LDA_C_PW": lambda: pw_correlation(graph, spec, variables, modified=False),
        "GGA_C_PW91": lambda: imported_pw91_correlation(graph, spec, variables),
        "LDA_C_PZ": lambda: pz_correlation(graph, spec, variables),
        "GGA_C_P86": lambda: p86_correlation(graph, spec, variables),
        "LDA_C_VWN": lambda: maple_vwn_correlation(
            graph, spec, variables, "LDA_C_VWN"
        ),
        "LDA_C_VWN_RPA": lambda: maple_vwn_correlation(
            graph, spec, variables, "LDA_C_VWN_RPA"
        ),
        "GGA_C_LYP": lambda: (
            production_lyp_correlation(graph, spec, variables)
            if production
            else lyp_correlation(graph, spec, variables)
        ),
    }
    energy = graph.sum(
        coefficient * builders[name]()
        for name, coefficient in spec.components
        if coefficient
    )
    if production:
        supported = {"LDA_X", "GGA_X_B88", "LDA_C_VWN_RPA", "GGA_C_LYP"}
        active = {name for name, coefficient in spec.components if coefficient}
        if not active <= supported:
            raise ValueError(
                "production tail is qualified only for the canonical B3LYP semilocal family"
            )
        energy = graph.select_le(n, F(1, 10**18), 0, energy)
    return graph, energy, variables


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
    # Keep the lightweight compiler import graph free of NumPy/runtime dependencies.
    # The generic runtime projection is needed only when this automatic path is
    # actually lowered.
    from .bulk_runtime import build_bulk_runtime_program

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
        return _build_special_energy_expression(spec, production=production)
    return semilocal_energy_expression(spec, production=production)
