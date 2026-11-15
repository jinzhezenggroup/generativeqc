"""Retain bounded through-f kernels beside the complete through-g family.

Both families lower the existing scalar programs and contraction policy. Only
their namespace, finite angular bound and native binding differ; no recurrence,
normalization or derivative formula is duplicated here.
"""

from .one_electron_cuda import emit_one_electron_values_cuda
from .one_electron_derivatives_cuda import emit_one_electron_derivatives_cuda
from .one_electron_policy_cuda import emit_one_electron_policy_cuda


def one_electron_family_inventory() -> dict:
    """Describe the producer-validated admission and conservative fallback."""
    return {
        "schema": "generativeqc.one_electron_angular_families",
        "version": 1,
        "families": {"through_f": 3, "through_g": 4},
        "selection": "maximum angular momentum of every packed physical shell",
        "unknown_metadata": "through_g",
        "ao_expansion_stride_is_not_an_angular_bound": True,
    }


def _through_f(source: str) -> str:
    """Give canonical emitted declarations independent namespace/header guards."""
    return source.replace(
        "GENERATIVEQC_GENERATED_ONE_ELECTRON_",
        "GENERATIVEQC_GENERATED_ONE_ELECTRON_THROUGH_F_",
    ).replace("generated_one_electron", "generated_one_electron_through_f")


def emit_one_electron_value_family_cuda() -> str:
    """Emit the existing through-f scalar roots and shared output policy."""
    policy = emit_one_electron_policy_cuda().replace(
        '#include "generated_one_electron_values.cuh"\n', ""
    )
    return "\n".join(
        (
            "#ifndef GENERATIVEQC_ONE_ELECTRON_VALUE_FAMILY_CUH",
            "#define GENERATIVEQC_ONE_ELECTRON_VALUE_FAMILY_CUH",
            _through_f(emit_one_electron_values_cuda(3) + policy),
            "namespace generativeqc::scf::generated_one_electron_through_f {",
            "struct ValueFamily : ValuePolicy {",
            "  static constexpr unsigned maximum_angular_momentum = 3;",
            "};",
            "}",
            "#endif",
            "",
        )
    )


def emit_one_electron_derivative_family_cuda() -> str:
    """Expose both derivative families without widening the through-f program."""
    return "\n".join(
        (
            "#ifndef GENERATIVEQC_ONE_ELECTRON_DERIVATIVE_FAMILY_CUH",
            "#define GENERATIVEQC_ONE_ELECTRON_DERIVATIVE_FAMILY_CUH",
            _through_f(emit_one_electron_derivatives_cuda(3)),
            "#endif",
            "",
        )
    )
