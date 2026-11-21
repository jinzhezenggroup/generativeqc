"""Compiler AD of fitted J/K stationary weights and Fock response.

Full rank uses the implicit-fit pullback and tangent. Spectral leaves stop at
the inverse and accept its tangent from the native fixed-rank metric owner.
"""

from fractions import Fraction
from functools import cache

from ..tensor import (
    Index,
    IndexSpace,
    Node,
    Program,
    TensorSpec,
    add,
    broadcast,
    einsum,
    input_tensor,
    linearize,
    reduce_sum,
    transpose,
    transpose_program,
)
from ..tensor.ad_program import _derivative_input, _rebuild


def fitted_coulomb_energy(charge: Node, fitted_charge: Node) -> Node:
    """Coulomb energy in either raw/solved or whitened auxiliary coordinates."""
    return einsum("p,p->", charge, fitted_charge, coefficient=Fraction(1, 2))


def solve_fitted_charge(whitening: Node, rhs: Node) -> Node:
    """Apply the accepted inverse (or pseudoinverse) to an auxiliary vector."""
    whitened = einsum("pq,q->p", whitening, rhs)
    return einsum("pq,p->q", whitening, whitened)


def fitted_exchange_energy(
    density: Node, left: Node, right: Node, scale: Fraction, fraction: Node | None
) -> Node:
    """Exchange quadratic form; callers supply the two physical fitting legs."""
    energy = einsum(
        "sik,sjl,ijp,klp->", density, density, left, right, coefficient=scale
    )
    return energy if fraction is None else einsum(",->", fraction, energy)


def symmetric_fitted_fock(energy: Node) -> Program:
    """Pull the energy back to symmetric density coordinates."""
    unconstrained = transpose_program(
        Program({"energy": energy}), ("energy",), inputs=("density",)
    )
    weight = unconstrained.program.outputs["bar_density"]
    return Program(
        {
            "bar_density": einsum(
                "sij->sij",
                add(weight, transpose(weight, (0, 2, 1))),
                coefficient=Fraction(1, 2),
            )
        }
    )


def _symmetric_pair(node: Node, axes: tuple[int, ...], labels: str) -> Node:
    return einsum(labels, add(node, transpose(node, axes)), coefficient=Fraction(1, 2))


def _fit_pair(values: Node, inverse: Node) -> Node:
    return einsum(
        "ijp,pq->ijq", _symmetric_pair(values, (1, 0, 2), "ijp->ijp"), inverse
    )


def _left_dressed(density: Node, leg: Node) -> Node:
    return einsum("sik,kjp->sijp", density, leg)


def fitted_pair_map(nao: int, naux: int, spins: int | None = None) -> Program:
    """Prepare reusable fitted legs and, for exchange, their density dressing."""
    orbital = IndexSpace("orbital", "ao", nao)
    auxiliary = IndexSpace("auxiliary", "auxiliary", naux)
    row, column = Index("row", orbital), Index("column", orbital)
    first, second = Index("first", auxiliary), Index("second", auxiliary)
    values = input_tensor(
        "three_center", TensorSpec((row, column, first), role="input")
    )
    inverse = input_tensor("inverse", TensorSpec((first, second), role="input"))
    fitted = _fit_pair(values, inverse)
    outputs = {"fitted": fitted}
    if spins is not None:
        spin = Index("spin", IndexSpace("spin", "spin", spins))
        density = input_tensor("density", TensorSpec((spin, row, column), role="input"))
        outputs["half_dressed"] = _left_dressed(density, fitted)
    return Program(outputs)


def fitted_coulomb_charge_maps(nao: int, naux: int, spins: int) -> dict[str, Program]:
    """J-only maps with no AO-pair solve or explicit inverse.

    The Coulomb stationary energy only needs the fitted auxiliary charge
    x = M⁻¹(A:D). Keep x across directions. The frozen-density tangent solves
    M dx = dA:D - dM x, while the raw-source weights also include A:dD.
    Apply the accepted whitening on each side of its Gram factorization,
    including for indefinite density directions. Spectral leaves accept
    (dV)b from the fixed-rank matrix-function rule and return db for its
    rank-one metric pullback, where b = A:D and V = W^T W.
    """
    orbital = IndexSpace("orbital", "ao", nao)
    auxiliary = IndexSpace("auxiliary", "auxiliary", naux)
    spin = Index("spin", IndexSpace("spin", "spin", spins))
    row, column = Index("row", orbital), Index("column", orbital)
    first, second = Index("first", auxiliary), Index("second", auxiliary)
    panel = (spin, row, column)
    pairs = (row, column, first)

    def field(name: str, indices: tuple[Index, ...] = ()) -> Node:
        return input_tensor(name, TensorSpec(indices, role="input"))

    def pair(name: str) -> Node:
        return _symmetric_pair(field(name, pairs), (1, 0, 2), "ijp->ijp")

    whitening = field("whitening", (first, second))

    def charge(density: Node, source: Node) -> Node:
        return einsum("ij,ijp->p", density, source)

    def fock(source: Node, fitted: Node) -> Node:
        return broadcast(einsum("ijp,p->ij", source, fitted), panel, (1, 2))

    values = pair("three_center")
    total = reduce_sum(field("density", panel), (0,))
    rhs = charge(total, values)
    fitted = field("accepted_charge", (second,))
    seed = field("bar_coulomb")
    fock_seed = field("bar_energy")
    reference = {
        "energy": fitted_coulomb_energy(rhs, fitted),
        "fock": einsum(",sij->sij", fock_seed, fock(values, fitted)),
        "weight_three_center": einsum(",ij,p->ijp", seed, total, fitted),
    }
    delta_values = pair("d_three_center")
    delta_total = reduce_sum(field("d_density", panel), (0,))
    delta_metric = _symmetric_pair(field("d_metric", (first, second)), (1, 0), "pq->pq")
    fixed_rhs = charge(total, delta_values)
    density_rhs = charge(delta_total, values)
    frozen_rhs = add(
        fixed_rhs,
        einsum("p,pq->q", fitted, delta_metric, coefficient=-1),
    )
    frozen = solve_fitted_charge(whitening, frozen_rhs)
    density_fit = solve_fitted_charge(whitening, density_rhs)
    moving = add(frozen, density_fit)
    metric_half = einsum(",p,q->pq", seed, fitted, moving, coefficient=-1)

    def motions(fixed: Node, relaxed: Node) -> dict[str, Node]:
        return {
            "fock_direction": einsum(
                ",sij->sij",
                fock_seed,
                add(fock(delta_values, fitted), fock(values, fixed)),
            ),
            "weight_direction_three_center": add(
                einsum(",ij,p->ijp", seed, total, relaxed),
                einsum(",ij,p->ijp", seed, delta_total, fitted),
            ),
        }

    spectral_frozen = add(
        solve_fitted_charge(whitening, fixed_rhs),
        field("inverse_motion_charge", (second,)),
    )
    return {
        "fit_charge": Program({"fitted_charge": solve_fitted_charge(whitening, rhs)}),
        "fit_charge_spectral": Program(
            {"fitted_charge": solve_fitted_charge(whitening, rhs), "raw_charge": rhs}
        ),
        "reference_coulomb": Program(
            {
                **reference,
                "weight_metric": einsum(
                    ",p,q->pq", seed, fitted, fitted, coefficient=Fraction(-1, 2)
                ),
            }
        ),
        "reference_coulomb_spectral": Program(reference),
        "response_coulomb": Program(
            {
                **motions(frozen, moving),
                "weight_direction_metric": _symmetric_pair(
                    metric_half, (1, 0), "pq->pq"
                ),
            }
        ),
        "response_coulomb_spectral": Program(
            {
                **motions(spectral_frozen, add(spectral_frozen, density_fit)),
                "raw_charge_motion": add(fixed_rhs, density_rhs),
            }
        ),
    }


@cache
def df_second_programs(
    nao: int,
    naux: int,
    spins: int,
    exchange: Fraction | None,
    *,
    coulomb: bool = True,
    pullback_metric: bool = True,
    retained_fit: bool = False,
    external_fit_motion: bool = False,
) -> tuple:
    """Return fitted energy and response graphs with one physical spin convention.

    Zero omits the exchange graph and all of its contractions; ``None`` binds
    a runtime exchange fraction which is constant under nuclear/density AD.
    An exchange-only fitted source must not reuse the full-range charge graph.
    With ``pullback_metric=False`` the source weights stop at the inverse,
    so a fixed-rank spectral owner can supply its actual metric pullback.
    The native owner sets ``retained_fit`` to bind the prepared fitted panel
    as an immutable input without dropping its nuclear differential.
    ``external_fit_motion`` binds the spectral tangent solved on packed AO
    pairs; the remaining maps still differentiate the full symmetric panels.
    """
    if not coulomb and exchange == 0:
        raise ValueError("fitted source must include Coulomb or exchange")
    orbital = IndexSpace("orbital", "ao", nao)
    auxiliary = IndexSpace("auxiliary", "auxiliary", naux)
    spin = Index("spin", IndexSpace("spin", "spin", spins))
    row, column = Index("row", orbital), Index("column", orbital)
    first, second = Index("first", auxiliary), Index("second", auxiliary)

    def field(name: str, indices: tuple) -> Node:
        return input_tensor(
            name, TensorSpec(indices, role="input", differentiable=True)
        )

    density = field("density", (spin, row, column))
    values = field("three_center", (row, column, first))
    inverse = field("inverse", (first, second))
    # The accepted raw panels can differ within roundoff across an AO-pair
    # orbit (e.g. separate spherical contractions of nuclear directions).
    # Restrict the fitted equation to the physical symmetric subspace *before*
    # applying the inverse; otherwise it magnifies that unphysical component.
    pair_values = _symmetric_pair(values, (1, 0, 2), "ijp->ijp")
    fitted_value = (
        field("accepted_fit", (row, column, second))
        if retained_fit
        else _fit_pair(values, inverse)
    )
    fitted = field("fitted_three_center", (row, column, second))
    left_fitted = _left_dressed(density, fitted)
    value_direction = input_tensor(
        "d_three_center", TensorSpec((row, column, first), role="input")
    )
    pair_direction = _symmetric_pair(value_direction, (1, 0, 2), "ijp->ijp")
    if external_fit_motion and pullback_metric:
        raise ValueError("the packed fit motion expects a spectral inverse tangent")
    if external_fit_motion:
        fitted_direction = field("packed_fit_motion", (row, column, second))
    elif pullback_metric:
        # B M = A  ⇒  dB = (dA − B dM) M⁻¹. Contract the metric motion
        # with the physical fitted leg before the inverse; forming the
        # dense d(M⁻¹) first loses digits along nearly null eigenvectors.
        metric_direction = _symmetric_pair(
            field("d_metric", (first, second)), (1, 0), "pq->pq"
        )
        residual = add(
            pair_direction,
            einsum("ijp,pq->ijq", fitted_value, metric_direction, coefficient=-1),
        )
        fitted_direction = einsum("ijp,pq->ijq", residual, inverse)
    else:
        inverse_direction = field("d_inverse", (first, second))
        fitted_direction = add(
            einsum("ijp,pq->ijq", pair_direction, inverse),
            einsum("ijp,pq->ijq", pair_values, inverse_direction),
        )

    def bind_fit(
        program: Program, pair_replacements: dict[Node, Node] | None = None
    ) -> Program:
        # Linearize with B as an independent input, then install the accepted
        # metric branch's chain rule in both the source and Fock programs.
        bindings = {
            "fitted_three_center": fitted_value,
            "d_fitted_three_center": fitted_direction,
        }
        replacements = {
            node: bindings[node.attrs["name"]]
            for node in program.nodes
            if node.op == "input" and node.attrs["name"] in bindings
        }
        if retained_fit and left_fitted in program.nodes:
            replacements[left_fitted] = field(
                "accepted_dressed", left_fitted.spec.indices
            )
        replacements.update(pair_replacements or {})
        return _rebuild(program, replacements)

    energies = {}
    fock_terms = {}
    fock_seed = input_tensor("bar_energy", TensorSpec((), role="input"))

    def left_dressed(leg: Node) -> Node:
        return left_fitted if leg is fitted else _left_dressed(density, leg)

    if coulomb:
        total = reduce_sum(density, (0,))
        charge = einsum("ij,ijp->p", total, pair_values)
        fitted_charge = einsum("ij,ijp->p", total, fitted)
        energies["coulomb"] = fitted_coulomb_energy(charge, fitted_charge)
        coulomb_fock = einsum("ijq,q->ij", pair_values, fitted_charge)
        fock_terms["coulomb"] = broadcast(coulomb_fock, (spin, row, column), (1, 2))
    if exchange != 0:
        spin_factor = Fraction(1, 2) if spins == 2 else Fraction(1, 4)
        exchange_scale = -(Fraction(1) if exchange is None else exchange) * spin_factor
        fraction = (
            input_tensor("exchange_fraction", TensorSpec((), role="parameter"))
            if exchange is None
            else None
        )
        # A D B is symmetric for the physical inverse and AO-pair domains.
        # Evaluate it in one orientation, share D B with the raw-source
        # pullback, and explicitly take the adjoint on symmetric AO matrices.
        exchange_fock = _symmetric_pair(
            einsum("ikp,skjp->sij", pair_values, left_dressed(fitted)),
            (0, 2, 1),
            "sij->sij",
        )
        energies["exchange"] = einsum(
            "sij,sij->", density, exchange_fock, coefficient=exchange_scale
        )
        if fraction is not None:
            energies["exchange"] = einsum(",->", fraction, energies["exchange"])
            exchange_fock = einsum(",sij->sij", fraction, exchange_fock)
        fock_terms["exchange"] = einsum(
            "sij->sij", exchange_fock, coefficient=2 * exchange_scale
        )
    energy = Program(energies)
    source_programs = []
    for name in energy.outputs:
        # The two A legs of each energy give the same cotangent on the
        # symmetric source domain. Form that leg once: moving M^-1 outside a
        # density contraction magnifies its rounding error at JKFIT condition
        # numbers. Apply the implicit metric pullback to the integral legs
        # before their outer product for the same reason. Linearize these
        # reassociated weights so the action uses the same ordering.
        seed = input_tensor(f"bar_{name}", TensorSpec((), role="input"))
        metric_field = "metric" if pullback_metric else "inverse"
        directions = (
            ("density", "fitted_three_center")
            if pullback_metric
            else ("density", "three_center", "fitted_three_center")
        )
        if name == "coulomb":
            raw_weight = einsum(",ij,p->ijp", seed, total, fitted_charge)
            metric_leg = fitted_charge if pullback_metric else charge
            metric_weight = einsum(
                ",p,q->pq",
                seed,
                metric_leg,
                metric_leg,
                coefficient=Fraction(-1 if pullback_metric else 1, 2),
            )
        else:
            if exchange is None:
                seed = einsum(",->", seed, fraction)
            metric_leg = fitted if pullback_metric else pair_values

            def dressed(leg: Node) -> Node:
                # Two AO matrix contractions, not an AO-quartet temporary.
                return einsum("silp,sjl->ijp", left_dressed(leg), density)

            raw_weight = einsum(
                ",ijp->ijp", seed, dressed(fitted), coefficient=2 * exchange_scale
            )
            # The auxiliary Gram uses tr((D B_p)(D B_q)), not a positive
            # density factor. Its two product-rule legs are transposes by
            # trace cyclicity, so contract just one and reflect it.
            left = left_dressed(metric_leg)
            metric_scale = -exchange_scale if pullback_metric else exchange_scale
            metric_weight = einsum(
                ",sijp,sjiq->pq", seed, left, left, coefficient=metric_scale
            )
        weights = Program({"three_center": raw_weight, metric_field: metric_weight})
        if name == "coulomb":
            moving = linearize(weights, directions).program
        else:
            left_inputs = (
                ("density", "fitted_three_center")
                if pullback_metric
                else ("density", "three_center")
            )
            left_motion = linearize(
                Program({"left": left}), left_inputs
            ).program.outputs["d_left"]
            if external_fit_motion:
                # D dB D is already symmetric. For the two moving-density
                # legs contract dD B D once and reflect the sum.
                half_dressed_motion = add(
                    _left_dressed(density, fitted_direction),
                    _left_dressed(_derivative_input(density, "d_density"), fitted),
                    coefficients=(1, 2),
                )
                raw_motion = _symmetric_pair(
                    einsum(
                        ",silp,sjl->ijp",
                        seed,
                        half_dressed_motion,
                        density,
                        coefficient=2 * exchange_scale,
                    ),
                    (1, 0, 2),
                    "ijp->ijp",
                )
            else:
                raw_motion = linearize(
                    Program({"three_center": raw_weight}),
                    ("density", "fitted_three_center"),
                ).program.outputs["d_three_center"]
            metric_motion = _symmetric_pair(
                einsum(
                    ",sijp,sjiq->pq",
                    seed,
                    left,
                    left_motion,
                    coefficient=2 * metric_scale,
                ),
                (1, 0),
                "pq->pq",
            )
            moving = Program(
                {"d_three_center": raw_motion, f"d_{metric_field}": metric_motion}
            )
        source_programs.append((bind_fit(weights), bind_fit(moving)))
    fock = Program(
        {"bar_density": einsum(",sij->sij", fock_seed, add(*fock_terms.values()))}
    )
    moving_terms = []
    for name, term in fock_terms.items():
        if name == "exchange" and pullback_metric:
            # For fixed D, A D dB = B D (dA - B dM) after moving the
            # symmetric inverse. Reflection swaps the two external AO
            # indices, so one exchange contraction suffices. Coulomb has
            # no such AO-index reflection and retains both product-rule legs.
            moving = bind_fit(
                Program({"d_term": term}), {pair_values: add(pair_direction, residual)}
            )
        else:
            moving = bind_fit(
                linearize(
                    Program({"term": term}), ("three_center", "fitted_three_center")
                ).program
            )
        moving_terms.append(moving.outputs["d_term"])
    fock_direction = Program(
        {"d_bar_density": einsum(",sij->sij", fock_seed, add(*moving_terms))}
    )
    return bind_fit(energy), tuple(source_programs), fock_direction, bind_fit(fock)
