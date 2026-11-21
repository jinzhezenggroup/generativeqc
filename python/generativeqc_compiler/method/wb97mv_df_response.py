"""Build-time WB97M-V fitted-primary response from incumbent AD.

The original stationary DF programs remain the mathematical authority. Generic
compiler contraction planning specializes them for a runtime-sized native CUDA
consumer, including primal-validation dependencies and explicit work admission.
No occupied/positive-density approximation or frozen-metric response is used.
"""

from __future__ import annotations

from functools import cache

from generativeqc_compiler.method.stationary_df_second import (
    df_second_programs,
    fitted_coulomb_charge_maps,
    fitted_pair_map,
)
from generativeqc_compiler.tensor import (
    Index,
    IndexSpace,
    Node,
    Program,
    TensorSpec,
    add,
    einsum,
    input_tensor,
    transpose,
)
from generativeqc_compiler.tensor.optimize import prepare_for_backend


@cache
def df_response_maps() -> dict[str, Program]:
    """Energy/Fock/source weights, their nuclear JVPs and the metric solve JVP.

    Fock motion holds D fixed; source-weight motion retains right dD, dA and
    metric motion. Full rank uses the implicit tangent of the fitted integral
    legs directly. The native owner contracts weights with raw integral AD. Whitening
    uses its actual out-auxiliary/source-auxiliary convention, not a guessed
    symmetric orientation. Spectral leaves return the seed and its motion
    before the metric pullback; the native owner differentiates the accepted
    truncated inverse using the separated retained/discarded eigenspaces.
    """
    auxiliary = IndexSpace("auxiliary", "auxiliary", 3)
    rows, columns = Index("first", auxiliary), Index("second", auxiliary)
    whitening = input_tensor("whitening", TensorSpec((rows, columns), role="parameter"))
    programs = {
        "inverse": Program({"inverse": einsum("pa,pb->ab", whitening, whitening)})
    }
    # The inverse acts on the auxiliary leg of each AO pair independently.
    pack = Index("pair", IndexSpace("symmetric_pair", "ao", 3))
    packed_spec = TensorSpec((pack, rows), role="input")
    raw = input_tensor("raw", packed_spec)
    raw_motion = input_tensor("raw_motion", packed_spec)
    inverse = input_tensor("inverse", whitening.spec)
    inverse_motion = input_tensor("inverse_motion", whitening.spec)
    programs["packed_spectral_fit_motion"] = Program(
        {
            "motion": add(
                einsum("ip,pq->iq", raw_motion, inverse),
                einsum("ip,pq->iq", raw, inverse_motion),
            )
        }
    )
    # A column-major Q from the SCF eigensolver is a row-major Q^T view.
    frame = input_tensor("frame_transposed", whitening.spec)
    panel = input_tensor("panel", whitening.spec)

    def symmetrize(value: Node) -> Node:
        return add(value, transpose(value, (1, 0)), coefficients=("1/2", "1/2"))

    projected = einsum("ij,kj->ik", symmetrize(panel), frame)
    programs["spectral_enter"] = Program(
        {"rotated": symmetrize(einsum("ij,jk->ik", frame, projected))}
    )
    expanded = einsum("ji,jk->ik", frame, panel)
    programs["spectral_leave"] = Program(
        {"rotated": symmetrize(einsum("ij,jk->ik", expanded, frame))}
    )
    for prefix, spins in (("rks", 1), ("uks", 2)):
        programs[f"{prefix}_fit_exchange"] = fitted_pair_map(2, 3, spins)
        programs.update(
            {
                f"{prefix}_{name}": graph
                for name, graph in fitted_coulomb_charge_maps(2, 3, spins).items()
            }
        )
        for suffix, coulomb in (("", True), ("_exchange", False)):
            for spectral, metric_field in ((False, "metric"), (True, "inverse")):
                energy, sources, moving_fock, fock = df_second_programs(
                    2,
                    3,
                    spins,
                    None,
                    coulomb=coulomb,
                    pullback_metric=not spectral,
                    retained_fit=True,
                    external_fit_motion=spectral,
                )
                reference = {
                    "energy": add(*energy.outputs.values()),
                    "fock": fock.outputs["bar_density"],
                }
                response = {"fock_direction": moving_fock.outputs["d_bar_density"]}
                for field in ("three_center", metric_field):
                    reference[f"weight_{field}"] = add(
                        *(fixed.outputs[field] for fixed, _ in sources)
                    )
                    response[f"weight_direction_{field}"] = add(
                        *(moving.outputs[f"d_{field}"] for _, moving in sources)
                    )
                selector = suffix + ("_spectral" if spectral else "")
                programs[f"{prefix}_reference{selector}"] = Program(reference)
                programs[f"{prefix}_response{selector}"] = Program(response)
    # Reassociation is an explicit compiler schedule decision, not new method
    # algebra. Retaining the optimized programs also makes work counts literal.
    return {
        name: prepare_for_backend(program, "cuda") for name, program in programs.items()
    }
