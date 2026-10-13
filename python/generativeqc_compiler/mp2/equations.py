"""Restricted spatial MP2 tile equations expressed in the shared TensorIR."""

import typing

from generativeqc_compiler.tensor import (
    Index,
    IndexSpace,
    Program,
    TensorSpec,
    add,
    broadcast,
    divide,
    einsum,
    input_tensor,
    multiply,
    reduce_sum,
    transpose,
    transpose_program,
)


def energy_program(
    shape: typing.Any, *, differentiable: typing.Any = False
) -> typing.Any:
    """Consume g[i,j,a,b]=(ia|jb), x[i,j,a,b]=(ib|ja), no spin compression.

    Inputs are rectangular tiles, so x is a separately requested/reordered
    block, not a swap of a and b inside a possibly disjoint tile. All ordered
    ijab tuples participate: OS=g*g/D, SS=g*(g-x)/D, total=OS+SS.
    The denominator is eps_i+eps_j-eps_a-eps_b in Hartree. Only scalar
    energies are outputs; the division intermediate is bounded by this tile.
    """
    if len(shape) != 4 or any(type(n) is not int or n < 1 for n in shape):
        raise ValueError("MP2 tile must have four positive integer dimensions")
    axes = tuple(
        Index(name, IndexSpace(name + "_tile", kind, n))
        for name, kind, n in zip(
            "ijab",
            ("occupied", "occupied", "virtual", "virtual"),
            shape,
            strict=True,
        )
    )

    def tensor(name: typing.Any, indices: typing.Any) -> typing.Any:
        return input_tensor(
            name,
            TensorSpec(
                indices,
                representation="restricted_spatial",
                role="input",
                differentiable=differentiable,
            ),
        )

    g, x = tensor("g", axes), tensor("x", axes)
    eps = [tensor("e" + name, (axis,)) for name, axis in zip("ijab", axes, strict=True)]
    outputs = _energy_outputs(g, x, eps, axes)
    return Program(
        outputs,
        provenance={
            "issue": "193 A1 (part of A)",
            "reference": "real all-electron canonical closed-shell RHF",
            "integrals": "unscreened conventional chemists ERIs; all ordered ijab",
            "units": "Hartree",
            "amplitudes": "unantisymmetrized restricted spatial; tile temporary only",
        },
    )


def _energy_outputs(
    g: typing.Any,
    x: typing.Any,
    eps: typing.Any,
    axes: typing.Any,
    *,
    fock_response: bool = False,
) -> dict:
    """Shared ordered spatial-MP2 algebra, independent of the integral producer."""
    d = add(
        *(broadcast(e, axes, (k,)) for k, e in enumerate(eps)),
        coefficients=(1, 1, -1, -1),
    )
    t = divide(g, d)
    os = reduce_sum(multiply(t, g), (0, 1, 2, 3))
    ss = reduce_sum(multiply(t, add(g, x, coefficients=(1, -1))), (0, 1, 2, 3))
    outputs = {"opposite_spin": os, "same_spin": ss}
    if fock_response:
        # Differentiate the noncanonical MP2 amplitude equation L(F)T=G.
        # Its adjoint is (2G-X)/D at this canonical frame. Full same-space
        # Fock cotangents preserve rotational stationarity without dividing
        # by occupied/occupied or virtual/virtual gaps, including degeneracy.
        left = divide(add(g, x, coefficients=(2, -1)), d)
        outputs["bar_foo"] = add(
            einsum("ijab,kjab->ik", left, t),
            einsum("ijab,ikab->jk", left, t),
            coefficients=(-1, -1),
        )
        outputs["bar_fvv"] = add(
            einsum("ijab,ijcb->ac", left, t),
            einsum("ijab,ijac->bc", left, t),
        )
    return outputs


def df_energy_program(
    occupied: int, virtuals: int, auxiliaries: int, *, fock_response: bool = False
) -> Program:
    """DF correlation on supplied conventional RHF orbitals; never refit Fock.

    Bov is Q-major and already whitened with the molecular source's metric.
    Only occupied/virtual Gram integrals are formed, not a full NMO^4 tensor.
    The two occupied and two virtual epsilon inputs borrow the same respective
    physical spectra; their reverse outputs must be summed by the caller.
    """
    if any(
        type(size) is not int or size < 1 for size in (occupied, virtuals, auxiliaries)
    ):
        raise ValueError(
            "DF-MP2 requires positive integer occupied, virtual and auxiliary dimensions"
        )
    axes = (
        Index("Q", IndexSpace("mp2_auxiliary", "batch", auxiliaries)),
        Index("i", IndexSpace("mp2_occupied", "occupied", occupied)),
        Index("a", IndexSpace("mp2_virtual", "virtual", virtuals)),
    )
    bov = input_tensor("bov", TensorSpec(axes, role="parameter", differentiable=True))
    direct = einsum("Qia,Qjb->ijab", bov, bov)
    exchange = transpose(direct, (0, 1, 3, 2))
    axes = direct.spec.indices
    eps = [
        input_tensor("e" + name, TensorSpec((axis,), role="input", differentiable=True))
        for name, axis in zip("ijab", axes, strict=True)
    ]
    return Program(
        _energy_outputs(direct, exchange, eps, axes, fock_response=fock_response),
        provenance={"method": "conventional RHF plus correlation-only DF-MP2"},
    )


def df_energy_response_program(
    occupied: int, virtuals: int, auxiliaries: int
) -> Program:
    """Energy, unit-total full Fock response and exact factor/spectrum VJP.

    The factor/spectrum reverse exposes OS/SS seeds; the molecular consumer
    sets both to one, matching the full noncanonical Fock response outputs.
    """
    primal = df_energy_program(occupied, virtuals, auxiliaries, fock_response=True)
    reverse = transpose_program(primal, ("opposite_spin", "same_spin"))
    return Program(
        {**primal.outputs, **reverse.program.outputs},
        provenance={"method": "same-Hamiltonian DF-MP2 factor and spectrum response"},
    )


def unrestricted_energy_program(
    shape: typing.Any,
    *,
    channel: typing.Any,
    differentiable: typing.Any = False,
) -> typing.Any:
    """Build one spin-resolved canonical UMP2 tile energy.

    channel is alpha_alpha, beta_beta, or alpha_beta.
    Same-spin tiles use 1/4 * |g-x|^2 / D over ordered ijab tuples; the
    opposite-spin tile uses g^2 / D. Spin labels are part of every TensorIR
    index-space identity, so alpha/beta spaces cannot alias merely because
    their tile extents happen to match.
    """
    if len(shape) != 4 or any(type(n) is not int or n < 1 for n in shape):
        raise ValueError("UMP2 tile must have four positive integer dimensions")
    spins = {
        "alpha_alpha": ("alpha", "alpha", "alpha", "alpha"),
        "beta_beta": ("beta", "beta", "beta", "beta"),
        "alpha_beta": ("alpha", "beta", "alpha", "beta"),
    }
    if channel not in spins:
        raise ValueError("unknown UMP2 spin channel")
    axis_spins = spins[channel]
    axes = tuple(
        Index(
            name,
            IndexSpace(
                f"{name}_{spin}_tile",
                kind,
                n,
                spin=spin,
            ),
        )
        for name, kind, n, spin in zip(
            "ijab",
            ("occupied", "occupied", "virtual", "virtual"),
            shape,
            axis_spins,
            strict=True,
        )
    )

    def tensor(name: typing.Any, indices: typing.Any) -> typing.Any:
        return input_tensor(
            name,
            TensorSpec(
                indices,
                representation="spin_orbital",
                role="input",
                differentiable=differentiable,
            ),
        )

    g = tensor("g", axes)
    eps = [tensor("e" + name, (axis,)) for name, axis in zip("ijab", axes, strict=True)]
    denominator = add(
        *(broadcast(e, axes, (k,)) for k, e in enumerate(eps)),
        coefficients=(1, 1, -1, -1),
    )
    if channel == "alpha_beta":
        energy = reduce_sum(multiply(divide(g, denominator), g), (0, 1, 2, 3))
    else:
        x = tensor("x", axes)
        antisymmetrized = add(g, x, coefficients=(1, -1))
        raw = reduce_sum(
            multiply(divide(antisymmetrized, denominator), antisymmetrized),
            (0, 1, 2, 3),
        )
        energy = add(raw, coefficients=("1/4",))
    return Program(
        {"energy": energy},
        provenance={
            "issue": "1820",
            "reference": "real all-electron canonical UHF",
            "channel": channel,
            "integrals": "unscreened conventional chemists ERIs; all ordered ijab",
            "units": "Hartree",
            "amplitudes": "spin-orbital canonical MP2; tile temporary only",
        },
    )


def pair_energy_program() -> Program:
    """Return one ordered RHF-MP2 (i,j,a,b) contribution.

    This is the scalar scientific owner used by the streamed RI-MP2 CUDA
    reduction.  The nested denominator and product/division order deliberately
    match the qualified native kernel; block traversal and compensated sums
    remain runtime scheduling/reduction policy.
    """
    scalar = TensorSpec((), representation="restricted_spatial", role="input")
    g, x, ei, ej, ea, eb = (
        input_tensor(name, scalar) for name in ("g", "x", "ei", "ej", "ea", "eb")
    )
    denominator = add(
        add(add(ei, ej), ea, coefficients=(1, -1)),
        eb,
        coefficients=(1, -1),
    )
    opposite_spin = divide(multiply(g, g), denominator)
    same_spin = divide(multiply(g, add(g, x, coefficients=(1, -1))), denominator)
    return Program(
        {"opposite_spin": opposite_spin, "same_spin": same_spin},
        provenance={
            "kind": "ri-mp2-scalar-pair-energy",
            "reference": "restricted spatial MP2 ordered ijab",
            "arithmetic": "native-fp64-order-v1",
        },
    )


def cpu_capacity(program: typing.Any) -> typing.Any:
    """Conservative numeric capacity including interpreter temporaries.

    The shared interpreter retains logical nodes. Add two maximum-sized
    temporary arrays for primitive arithmetic/finiteness checks and scalar
    publication. Object headers and NumPy/BLAS allocator overhead are excluded.
    """
    sizes = [n.spec.size * n.spec.itemsize for n in program.live_nodes]
    return sum(sizes) + 2 * max(sizes) + 64
