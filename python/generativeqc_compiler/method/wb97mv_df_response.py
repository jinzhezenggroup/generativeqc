"""Build-time full-rank WB97M-V fitted-primary response from incumbent AD.

The original stationary DF programs remain the mathematical authority. Generic
compiler contraction planning specializes them for a runtime-sized native CUDA
consumer, including primal-validation dependencies and explicit work admission.
No occupied/positive-density approximation or frozen-metric response is used.
"""

from __future__ import annotations

from functools import cache

from generativeqc_compiler.method.spec import resolve_method
from generativeqc_compiler.method.stationary_df_second import df_second_programs
from generativeqc_compiler.method.wb97mv_fock_response import fock_maps
from generativeqc_compiler.tensor import Program, add
from generativeqc_compiler.tensor.optimize import prepare_for_backend


@cache
def df_response_maps() -> dict[str, Program]:
    """Energy/Fock/source weights, their nuclear JVPs and the metric solve JVP.

    Fock motion holds D fixed; source-weight motion retains right dD, dA and
    d(M^-1). A native owner supplies d(M^-1) from the separate implicit inverse
    map, and contracts the resulting weights with raw integral AD. Whitening
    uses its actual out-auxiliary/source-auxiliary convention, not a guessed
    symmetric orientation. Only the physical full-rank inverse is differentiated.
    """
    short = resolve_method("WB97M-V").primitives[1]
    programs = {"inverse": fock_maps()["inverse"]}
    for prefix, spins in (("rks", 1), ("uks", 2)):
        energy, sources, moving_fock, inverse_direction, fock = df_second_programs(
            2, 3, spins, short.coefficient
        )
        reference = {
            "energy": add(*energy.outputs.values()),
            "fock": fock.outputs["bar_density"],
        }
        response = {"fock_direction": moving_fock.outputs["d_bar_density"]}
        for field in ("three_center", "metric"):
            reference[f"weight_{field}"] = add(
                *(fixed.outputs[field] for fixed, _ in sources)
            )
            response[f"weight_direction_{field}"] = add(
                *(moving.outputs[f"d_{field}"] for _, moving in sources)
            )
        programs[f"{prefix}_reference"] = Program(reference)
        programs[f"{prefix}_response"] = Program(response)
        programs["inverse_direction"] = inverse_direction
    # Reassociation is an explicit compiler schedule decision, not new method
    # algebra. Retaining the optimized programs also makes work counts literal.
    return {
        name: prepare_for_backend(program, "cuda") for name, program in programs.items()
    }
