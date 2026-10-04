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
    input_tensor,
    multiply,
    reduce_sum,
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
            "ijab", ("occupied", "occupied", "virtual", "virtual"), shape
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
    eps = [tensor("e" + name, (axis,)) for name, axis in zip("ijab", axes)]
    d = add(
        *(broadcast(e, axes, (k,)) for k, e in enumerate(eps)),
        coefficients=(1, 1, -1, -1),
    )
    t = divide(g, d)
    os = reduce_sum(multiply(t, g), (0, 1, 2, 3))
    ss = reduce_sum(multiply(t, add(g, x, coefficients=(1, -1))), (0, 1, 2, 3))
    return Program(
        {"opposite_spin": os, "same_spin": ss},
        provenance={
            "issue": "193 A1 (part of A)",
            "reference": "real all-electron canonical closed-shell RHF",
            "integrals": "unscreened conventional chemists ERIs; all ordered ijab",
            "units": "Hartree",
            "amplitudes": "unantisymmetrized restricted spatial; tile temporary only",
        },
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
    eps = [tensor("e" + name, (axis,)) for name, axis in zip("ijab", axes)]
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
