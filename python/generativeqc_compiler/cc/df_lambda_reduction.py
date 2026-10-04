"""Stage the adjoint of the existing auxiliary-reduced RCCSD residual.

The primal amplitudes are immutable during Lambda. Prepare tau and all reduced
intermediates once, differentiate the retained core once per response seed,
accumulate the auxiliary adjoints, then reverse tau once. This removes repeated
T2 consumers from the Q loop without adding a second electronic equation.
"""

from __future__ import annotations

from dataclasses import dataclass

from generativeqc_compiler.tensor import Program, transpose_program

from .df_hoist import (
    DFAuxiliaryReductionPrograms,
    _bounded,
    build_df_auxiliary_reduction_programs,
)
from .df_lambda import RETAINED_PARAMETERS
from .lambda_equations import AMPLITUDES, RESIDUALS, build_parameter_vjp


@dataclass(frozen=True)
class DFLambdaReductionPrograms:
    """Primal staging and its exact chain rule in dense Frobenius coordinates.

    ``core`` returns direct amplitude cotangents and seeds for each reduced
    auxiliary output. Pass the latter unchanged to every ``auxiliary`` call,
    summing its t1, t2 and tau cotangents. Reverse ``prepare`` with the summed tau
    seed and add its amplitude outputs to the direct core/auxiliary sources.
    ``factors`` uses the same core seeds and publishes each Q row separately.
    Retained parameter cotangents belong solely to ``parameters``; upstream
    Gram-product pullbacks still supply their factor dependence separately.

    A caller must invalidate primal staging if amplitudes or factors change.
    The original expanded core-plus-Q actions remain independent audits and
    bounded fallbacks. No solver iteration or denominator is differentiated.
    """

    primal: DFAuxiliaryReductionPrograms
    core: Program
    auxiliary: Program
    prepare: Program
    factors: Program
    parameters: dict[str, Program]


def build_df_lambda_reduction_programs(
    nocc: int, nvir: int
) -> DFLambdaReductionPrograms:
    """Differentiate compiler-owned prepare/Q-reduction/core cuts exactly once."""
    primal = build_df_auxiliary_reduction_programs(nocc, nvir)
    cuts = tuple(primal.auxiliary.outputs)
    core = _bounded(
        transpose_program(primal.core, RESIDUALS, inputs=(*AMPLITUDES, *cuts)).program
    )
    auxiliary = _bounded(
        # Some auxiliary intermediates retain direct T2 dependence in addition
        # to the shared tau cut. Both paths belong to the chain rule.
        transpose_program(
            primal.auxiliary, cuts, inputs=(*AMPLITUDES, "df_tau")
        ).program
    )
    prepare = _bounded(
        transpose_program(primal.prepare, ("df_tau",), inputs=AMPLITUDES).program
    )
    factors = _bounded(
        transpose_program(primal.auxiliary, cuts, inputs=("bov", "bvv")).program
    )
    parameters = {
        name: _bounded(build_parameter_vjp(primal.core, name).program)
        for name in RETAINED_PARAMETERS
    }
    return DFLambdaReductionPrograms(
        primal, core, auxiliary, prepare, factors, parameters
    )
