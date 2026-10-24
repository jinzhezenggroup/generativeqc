# Decision: capture roll through bounded existing gather maps

Status: implemented
Date: 2026-10-16

## Decision

Generic Boolean and real arrays may capture static roll using existing reshape
and gather primitives. No new scientific algebra or runtime primitive is needed.
Normalize shifts modulo extents, accumulate repeated axes, and check the total
nonidentity gather positions against 65,536 before materializing any map.
Flattened rolls retain row-major order. Scientific index spaces remain explicit
and are rejected even for identity rolls.

## Alternatives and invariants

An unrestricted static map can allocate memory proportional to a symbolic axis
without allocating its input. Avoid that path, including for trillion-element
metadata. A new modular-index TensorIR operation could remove the map cost but
would need separate serialization, backend and derivative qualification. Whole
extent shifts and cancelling repeated shifts can remain metadata-only identities.
This feature makes no endpoint performance or wider dtype/GPU admission claim.

## Evidence and revisit conditions

`tests/python/test_array_api_roll.py` compares eager/reference values with NumPy,
checks invalid arguments, empty/scalar/large metadata and the budget boundary,
and validates native CPU values and JVP/VJP against independent NumPy and finite
difference oracles with reference fallback forbidden. Existing shape and capability
coverage remains applicable. Revisit the static bound only with a separately
qualified compact indexing primitive. Refs #2111.
