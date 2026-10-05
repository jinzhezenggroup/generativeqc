"""Canonical RCCSD denominator algebra with the original FP64 grouping.

The physical gap is checked before applying the level shift. In particular,
the doubles consumer must not sum already shifted singles: that changes the
rounding contract. Explicit supplied denominators never use this expression.
"""

from generativeqc_compiler.tensor.ir import TensorSpec, add, input_tensor
from generativeqc_compiler.tensor.program import Program


def canonical_denominator_program(*, doubles: bool) -> Program:
    """Return physical and shifted scalar gaps without reassociation."""
    scalar = TensorSpec((), role="input")
    ei, ea, shift = (input_tensor(name, scalar) for name in ("ei", "ea", "shift"))
    physical = add(ei, ea, coefficients=(1, -1))
    if doubles:
        ej, eb = (input_tensor(name, scalar) for name in ("ej", "eb"))
        physical = add(physical, add(ej, eb, coefficients=(1, -1)))
    return Program(
        {
            "physical": physical,
            "shifted": add(physical, shift, coefficients=(1, -2 if doubles else -1)),
        },
        provenance={"denominator": "canonical RCCSD ordered pair gaps, FP64"},
    )
