"""Independent fixed-model algebra for the explicitly admitted CBS fixtures.

The oracle combination never consumes coefficients returned by production.
Decimal exp and exact rational arithmetic derive this task's 2/3 cardinal model.
"""

from __future__ import annotations

import json
import math
from decimal import Decimal, localcontext
from typing import Any

ROLES = ("correction_ccsdt", "mp2_low", "hf_low", "mp2_high", "hf_high")


def number(value: Any) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
    ):
        raise ValueError("missing or nonfinite model coefficient")
    return float(value)


def coefficients() -> tuple[list[float], list[float], list[float]]:
    with localcontext() as context:
        context.prec = 80
        hf_low = -Decimal(1) / (Decimal("1.63").exp() - 1)
        hf_high = 1 - hf_low
        mp_low = -Decimal(8) / 19
        mp_high = Decimal(27) / 19
        return (
            [
                float(x)
                for x in (1, mp_low - 1, hf_low - mp_low, mp_high, hf_high - mp_high)
            ],
            [float(hf_low), float(hf_high)],
            [float(mp_low), float(mp_high)],
        )


def equal_coefficient(actual: Any, expected: float) -> bool:
    # Different libm implementations may round the unchanged model by an ULP.
    # This bound applies to dimensionless metadata, never to scientific E/F gates.
    return abs(number(actual) - expected) <= 4 * math.ulp(expected)


def components(call: dict, sample: dict, inputs: dict) -> list[dict]:
    values = sample.get("components")
    if not isinstance(values, list) or len(values) != 5:
        raise ValueError("five model constituents are required")
    by_role = {}
    for component in values:
        roles = component.get("roles")
        if not isinstance(roles, (tuple, list)) or len(roles) != len(set(roles)):
            raise ValueError("missing, duplicate or ambiguous model constituent role")
        canonical = set(roles).intersection(ROLES)
        if len(canonical) != 1:
            raise ValueError("missing or ambiguous canonical model role")
        role = next(iter(canonical))
        wanted = {role, "correction_mp2"} if role == "mp2_low" else {role}
        if set(roles) != wanted or role in by_role:
            raise ValueError("missing alias, unexpected alias or duplicate model role")
        by_role[role] = component
    ordered = [by_role[role] for role in ROLES]
    if call["fixture"] == "archive-water":
        identities = inputs["archive_water_expected_basis_identities"]
        methods = ("df-ccsd(t)", "mp2", "rhf", "mp2", "rhf")
    else:
        reference = json.loads(
            inputs["conventional"][f"tests/reference_data/cbs/{call['fixture']}.json"][
                "content_utf8"
            ]
        )
        low, high = reference["basis_identities"]
        identities = [low, low, low, high, high]
        methods = ("ccsd(t)", "mp2", "rhf", "mp2", "rhf")
    expected, hf, mp = coefficients()
    for component, identity, method, weight in zip(
        ordered, identities, methods, expected, strict=True
    ):
        if (
            component.get("basis_identity") != identity
            or component.get("method") != method
        ):
            raise ValueError("constituent method/basis differs from admitted model")
        if not equal_coefficient(component.get("coefficient"), weight):
            raise ValueError(
                "returned coefficient differs from independent fixed model"
            )
    for field, wanted in (("hf_weights", hf), ("mp2_weights", mp)):
        actual = sample.get(field)
        if (
            not isinstance(actual, (list, tuple))
            or len(actual) != 2
            or not all(
                equal_coefficient(a, b) for a, b in zip(actual, wanted, strict=True)
            )
        ):
            raise ValueError(
                "returned extrapolation weights differ from independent model"
            )
    if not equal_coefficient(
        sample.get("convergence", {}).get("coefficient_l1"), sum(map(abs, expected))
    ):
        raise ValueError("returned coefficient L1 differs from independent model")
    return ordered


def archive_expected(oracle: dict, forces: list) -> tuple[float, list]:
    weights, _, _ = coefficients()
    energy = sum(
        w * number(e)
        for w, e in zip(weights, oracle["component_energies"], strict=True)
    )
    if len(forces) != 5 or any(
        len(component) != 3 or any(len(row) != 3 for row in component)
        for component in forces
    ):
        raise ValueError("archive independent force tensor is incomplete")
    combined = [
        [sum(weights[k] * number(forces[k][a][d]) for k in range(5)) for d in range(3)]
        for a in range(3)
    ]
    return energy, combined
