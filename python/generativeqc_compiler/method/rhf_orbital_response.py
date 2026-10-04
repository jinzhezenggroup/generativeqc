"""Matrix-sized RHF frame/Pulay maps around an external exact J/K action.

The runtime supplies G(D)=J(D)-K(D)/2 and its self-adjoint density action. This
module owns only closed-shell density, orbital projection, metric transport and
AD composition. No four-index integral or occupied-virtual Hessian is an input.
"""

from dataclasses import dataclass
from fractions import Fraction

from generativeqc_compiler.tensor import (
    Index,
    IndexSpace,
    Program,
    TensorSpec,
    add,
    einsum,
    input_tensor,
    linearize,
    slice_tensor,
    transpose,
    transpose_program,
)


@dataclass(frozen=True)
class RHFFrameResponsePrograms:
    """Two reverse stages separated by the physical exact J/K provider.

    1. Evaluate potential_seed to obtain symmetric bar_Fao.
    2. Supply bar_density=G(bar_Fao) and bar_frame from the correlation source;
       evaluate weights. The same C/F/h and other output seeds must be reused.

    Nuclear two-electron response contracts bar_Fao with G'(D). Hcore and
    overlap outputs are full AO Frobenius weights. Orbital response uses
    K_ia=x, K_ai=-x and A=-dFov/dx, matching the existing dense CC oracle.
    """

    primal: Program
    potential_seed: Program
    weights: Program
    density_direction: Program
    orbital_action: Program
    nocc: int
    nvir: int


def build_rhf_frame_response(nocc: int, nvir: int) -> RHFFrameResponsePrograms:
    """Build bounded matrix maps; native J/K evaluation remains runtime-owned.

    The maps differentiate unrestricted C, not an orthogonalized surrogate.
    Consequently the symmetric frame cotangent retains the Pulay multiplier.
    Same-space degeneracy introduces no divisions or eigenvector gauge choice.
    Dimensions are concrete compiler witnesses, suitable for runtime lowering.
    """
    if any(type(value) is not int or value < 1 for value in (nocc, nvir)):
        raise ValueError("RHF frame response requires occupied and virtual spaces")
    n = nocc + nvir
    space = IndexSpace("rhf_complete_frame", "orbital", n)
    indices = tuple(Index(c, space) for c in "pq")
    spec = TensorSpec(
        indices,
        role="parameter",
        differentiable=True,
        representation="restricted_spatial",
    )
    c = input_tensor("coefficients", spec)
    h = input_tensor("hcore", spec)
    raw_f = input_tensor("fock_ao", spec)
    # The physical Fock and its allowed variations are symmetric. Making that
    # projection explicit gives the density provider a symmetric cotangent,
    # even if a downstream MO block uses independent nonsymmetric coordinates.
    f = add(
        raw_f, transpose(raw_f, (1, 0)), coefficients=(Fraction(1, 2), Fraction(1, 2))
    )
    occupied = slice_tensor(c, ((0, n), (0, nocc)))
    density = einsum("ui,vi->uv", occupied, occupied, coefficient=2)
    fmo = einsum("up,uq->pq", c, einsum("uv,vq->uq", f, c))
    energy = einsum("uv,uv->", density, add(h, f), coefficient=Fraction(1, 2))
    primal = Program(
        {
            "fock_mo": fmo,
            "reference_electronic_energy": energy,
            "density": density,
            "frame": c,
        },
        provenance={
            "model": "RHF frame around external exact G(D)",
            "occupation": "D=2 Cocc Cocc.T",
            "fock_contract": "F=h+G(D); G is self-adjoint on symmetric densities",
            "nocc": nocc,
        },
    )
    reverse = transpose_program(
        primal,
        ("fock_mo", "reference_electronic_energy", "density", "frame"),
        inputs=("coefficients", "hcore", "fock_ao"),
    )
    bar_f = reverse.program.outputs["bar_fock_ao"]
    potential_seed = Program(
        {"fock_ao_weights": bar_f},
        provenance={"primal": primal.logical_hash, "stage": "exact G cotangent input"},
    )
    bar_c = reverse.program.outputs["bar_coefficients"]
    rotation = einsum("up,uq->pq", c, bar_c)
    stationarity = add(rotation, transpose(rotation, (1, 0)), coefficients=(1, -1))
    overlap_mo = add(
        rotation,
        transpose(rotation, (1, 0)),
        coefficients=(Fraction(-1, 4), Fraction(-1, 4)),
    )
    overlap_ao = einsum("up,vp->uv", einsum("uq,qp->up", c, overlap_mo), c)
    weights = Program(
        {
            "hcore": add(reverse.program.outputs["bar_hcore"], bar_f),
            "fock_ao_weights": bar_f,
            "frame_cotangent": bar_c,
            "rotation_gradient": rotation,
            "overlap": overlap_ao,
            "stationarity": stationarity,
            "orbital_rhs": add(
                slice_tensor(stationarity, ((0, nocc), (nocc, n))), coefficients=(-1,)
            ),
        },
        provenance={
            "primal": primal.logical_hash,
            "pullback": reverse.derivative_hash,
            "composition": "bar_density=G(bar_Fao); bar_frame=correlation source",
            "overlap_rule": "symmetric metric transport dU=-dSmo/2",
            "two_electron_rule": "contract bar_Fao with G'(D)",
        },
    )
    u = input_tensor("rotation", spec)
    rotated = einsum("up,pq->uq", c, u)
    occ_rotated = slice_tensor(rotated, ((0, n), (0, nocc)))
    rotated_f = einsum("up,uq->pq", rotated, einsum("uv,vq->uq", f, rotated))
    tangent_primal = Program(
        {
            "density": einsum("ui,vi->uv", occ_rotated, occ_rotated, coefficient=2),
            "fov": slice_tensor(rotated_f, ((0, nocc), (nocc, n))),
        },
        provenance={
            "model": "RHF frame tangent around external exact G(dD)",
            "nocc": nocc,
        },
    )
    direction = linearize(
        tangent_primal, ("rotation", "fock_ao"), outputs=("density", "fov")
    )
    density_direction = Program(
        {"density_direction": direction.program.outputs["d_density"]},
        provenance={
            "primal": tangent_primal.logical_hash,
            "stage": "exact G tangent input",
        },
    )
    orbital_action = Program(
        {"orbital_action": add(direction.program.outputs["d_fov"], coefficients=(-1,))},
        provenance={
            "primal": tangent_primal.logical_hash,
            "composition": "d_fock_ao=G(d_density)",
            "orbital_rule": "K_ia=x; K_ai=-x; A=-dFov/dx",
        },
    )
    return RHFFrameResponsePrograms(
        primal, potential_seed, weights, density_direction, orbital_action, nocc, nvir
    )
