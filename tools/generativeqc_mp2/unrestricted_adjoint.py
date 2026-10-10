"""Bounded fixed-orbital UMP2 cotangents from the shared TensorIR VJP.

These internal MO weights are unrelaxed correlation-energy derivatives. They
do not include UHF response, AO derivatives, overlap weights or nuclear forces.
"""

from __future__ import annotations

import math
import typing
from dataclasses import dataclass
from itertools import product
from types import MappingProxyType

import numpy as np
from generativeqc_compiler.common.arrays import immutable
from generativeqc_compiler.mp2.equations import (
    cpu_capacity,
    unrestricted_energy_program,
)
from generativeqc_compiler.tensor import vjp
from generativeqc_compiler.tensor.types import checked_size

if typing.TYPE_CHECKING:
    from collections.abc import Mapping

CHANNEL_SPINS = MappingProxyType(
    {
        "alpha_alpha": ("alpha", "alpha"),
        "beta_beta": ("beta", "beta"),
        "alpha_beta": ("alpha", "beta"),
    }
)
ENERGY_FEEDS = ("ei", "ej", "ea", "eb")
# A flat slice may copy a non-contiguous FP64 input before its Boolean scan.
SCAN_BYTES = 4096 * 9


@dataclass(frozen=True)
class UMP2TileAdjoint:
    """Independent feed cotangents in ordered ijab layout, before scatter-add."""

    channel: str
    direct: np.ndarray
    exchange: np.ndarray | None
    occupied_i: np.ndarray
    occupied_j: np.ndarray
    virtual_a: np.ndarray
    virtual_b: np.ndarray
    equation_hash: str
    derivative_hash: str
    numeric_capacity_bytes: int


@dataclass(frozen=True)
class UMP2CanonicalAdjoint:
    """Physical ijab block and global spin epsilon cotangents; no spin factors left."""

    integrals_iajb: Mapping[str, np.ndarray]
    orbital_energies: Mapping[str, np.ndarray]
    reference_identity: str
    hamiltonian_id: str
    equation_hashes: tuple[str, ...]
    derivative_hashes: tuple[str, ...]
    minimum_absolute_denominator: float
    numeric_capacity_bytes: int
    tile_count: int
    backend: str = "numpy-cpu-autodiff"


def _positive(value: int, name: str) -> None:
    checked_size(value, name)
    if value < 1:
        raise ValueError(f"{name} must be positive")


def _array(value: np.ndarray, name: str) -> np.ndarray:
    # Do not coerce complex/FP32/list inputs or allocate before admission.
    if not isinstance(value, np.ndarray) or value.dtype != np.dtype("float64"):
        raise TypeError(f"{name} must be a real FP64 NumPy array")
    return value


def _finite(array: np.ndarray) -> None:
    for begin in range(0, array.size, 4096):
        if not np.isfinite(array.flat[begin : begin + 4096]).all():
            raise ValueError(
                "UMP2 adjoint inputs and accumulated weights must be finite"
            )


def _denominator(energies: tuple[np.ndarray, ...], threshold: float) -> float:
    if isinstance(threshold, bool) or not np.isfinite(threshold) or threshold <= 0:
        raise ValueError("denominator threshold must be finite and positive")
    if any(not e.size for e in energies):
        return 0.0
    with np.errstate(over="raise", invalid="raise"):
        try:
            largest = (
                max(energies[0])
                + max(energies[1])
                - min(energies[2])
                - min(energies[3])
            )
            smallest = (
                min(energies[0])
                + min(energies[1])
                - max(energies[2])
                - max(energies[3])
            )
        except FloatingPointError as exc:
            raise ValueError("nonfinite UMP2 denominator extrema") from exc
    if not np.isfinite(largest) or not np.isfinite(smallest):
        raise ValueError("nonfinite UMP2 denominator extrema")
    if largest >= -threshold:
        raise ValueError(
            "UMP2 denominator must be negative and separated from zero; no regularization"
        )
    return float(-largest)


def _tile_capacity(program: object, shape: tuple[int, ...], same: bool) -> int:
    # Four primal capacities cover primal+reverse live nodes, primitive work,
    # input-bar publication and immutable copies for this fixed DAG inventory.
    feeds = 8 * ((2 if same else 1) * math.prod(shape) + sum(shape))
    return checked_size(
        feeds + 4 * cpu_capacity(program) + SCAN_BYTES, "UMP2 tile capacity"
    )


def tile_energy_adjoint(
    feeds: Mapping[str, np.ndarray],
    *,
    channel: str,
    budget_bytes: int = 256 << 20,
    denominator_threshold: float = 1e-10,
) -> UMP2TileAdjoint:
    """VJP of one positive rectangular tile, retaining all four epsilon feeds.

    g[i,j,a,b]=(i_left a_left|j_right b_right). Same-spin x is an
    independently requested (i b|j a) feed in the *same* ijab layout, including
    disjoint rectangular tiles. Alpha-beta must not supply x. Inputs are
    borrowed synchronously and must not be mutated during the call. Returned
    arrays own immutable storage and cannot alias inputs or one another.
    """
    _positive(budget_bytes, "budget_bytes")
    if channel not in CHANNEL_SPINS:
        raise ValueError("unknown UMP2 spin channel")
    same = channel != "alpha_beta"
    expected = {"g", *ENERGY_FEEDS} | ({"x"} if same else set())
    if set(feeds) != expected:
        raise ValueError("UMP2 channel requires exact g/[x]/ei/ej/ea/eb feeds")
    arrays = {name: _array(value, name) for name, value in feeds.items()}
    shape = arrays["g"].shape
    if len(shape) != 4 or not all(shape):
        raise ValueError(
            "UMP2 tile requires four positive dimensions; skip empty channels"
        )
    if same and arrays["x"].shape != shape:
        raise ValueError("UMP2 exchange feed must match the direct tile shape")
    if any(arrays[name].shape != (shape[k],) for k, name in enumerate(ENERGY_FEEDS)):
        raise ValueError("UMP2 orbital-energy feed shape mismatch")
    program = unrestricted_energy_program(shape, channel=channel, differentiable=True)
    capacity = _tile_capacity(program, shape, same)
    if capacity > budget_bytes:
        raise MemoryError(
            f"UMP2 tile needs {capacity} numeric bytes, budget is {budget_bytes}"
        )
    for array in arrays.values():
        _finite(array)
    _denominator(tuple(arrays[name] for name in ENERGY_FEEDS), denominator_threshold)
    reverse = vjp(
        program, arrays, {"energy": np.array(1.0)}, max_bytes=4 * cpu_capacity(program)
    )
    bars = reverse.input_cotangents
    return UMP2TileAdjoint(
        channel,
        immutable(bars["g"]),
        immutable(bars["x"]) if same else None,
        *(immutable(bars[name]) for name in ENERGY_FEEDS),
        reverse.primal_logical_hash,
        reverse.derivative_hash,
        capacity,
    )


def canonical_energy_adjoint(
    integrals_iajb: Mapping[str, np.ndarray],
    orbital_energies: Mapping[str, np.ndarray],
    occupied: Mapping[str, int],
    *,
    reference_identity: str,
    hamiltonian_id: str,
    tile_shape: tuple[int, int, int, int] = (1, 1, 2, 2),
    budget_bytes: int = 256 << 20,
    denominator_threshold: float = 1e-10,
) -> UMP2CanonicalAdjoint:
    """Assemble fixed-orbital correlation cotangents for all ordered spin tuples.

    Canonical arrays have [no_left,no_right,nv_left,nv_right] layout, occupied
    orbitals precede virtuals in each global epsilon vector. Identity strings
    are caller attestations tying these MO buffers to a reference/Hamiltonian;
    this function does not validate UHF stationarity or recompute an SCF state.
    Numeric admission charges resident inputs, dense outputs, immutable
    publication copies/scratch plus the largest tile workspace. Python/allocator overhead
    is excluded. No result escapes on failure; inputs are never modified.
    """
    _positive(budget_bytes, "budget_bytes")
    for name, value in (
        ("reference_identity", reference_identity),
        ("hamiltonian_id", hamiltonian_id),
    ):
        if not isinstance(value, str) or not value:
            raise ValueError(f"{name} must be a nonempty identity")
    if (
        set(integrals_iajb) != set(CHANNEL_SPINS)
        or set(orbital_energies) != {"alpha", "beta"}
        or set(occupied) != {"alpha", "beta"}
    ):
        raise ValueError(
            "UMP2 requires exactly three spin blocks and alpha/beta energies/occupations"
        )
    if not isinstance(tile_shape, tuple) or len(tile_shape) != 4:
        raise ValueError("tile_shape must contain four positive integers")
    for extent in tile_shape:
        _positive(extent, "tile extent")
    eps = {spin: _array(value, spin) for spin, value in orbital_energies.items()}
    for spin, energy in eps.items():
        checked_size(occupied[spin], f"{spin} occupied")
        if energy.ndim != 1 or occupied[spin] > energy.size:
            raise ValueError("UMP2 occupation/energy dimensions are inconsistent")
    blocks = {name: _array(value, name) for name, value in integrals_iajb.items()}
    shapes = {}
    workspace = SCAN_BYTES
    for channel, (left, right) in CHANNEL_SPINS.items():
        nl, nr = occupied[left], occupied[right]
        shape = (nl, nr, eps[left].size - nl, eps[right].size - nr)
        if blocks[channel].shape != shape:
            raise ValueError(
                f"{channel} canonical integral shape mismatch: expected {shape}"
            )
        shapes[channel] = shape
        if not all(shape):
            continue
        # Account for tails as well as full tiles before *any* VJP execution.
        extents = [{min(n, t), n % t or min(n, t)} for n, t in zip(shape, tile_shape)]
        for tile in product(*extents):
            program = unrestricted_energy_program(
                tile, channel=channel, differentiable=True
            )
            workspace = max(workspace, _tile_capacity(program, tile, left == right))
    resident = sum(array.nbytes for array in (*blocks.values(), *eps.values()))
    # The fourth resident-sized allowance covers contiguous publication scratch
    # and finiteness masks even when callers supply Fortran/strided buffers.
    capacity = checked_size(4 * resident + workspace, "UMP2 canonical capacity")
    if capacity > budget_bytes:
        raise MemoryError(
            f"UMP2 canonical adjoint needs {capacity} numeric bytes, budget is {budget_bytes}"
        )
    for array in (*blocks.values(), *eps.values()):
        _finite(array)
    minima = []
    for channel, (left, right) in CHANNEL_SPINS.items():
        nl, nr = occupied[left], occupied[right]
        minimum = _denominator(
            (eps[left][:nl], eps[right][:nr], eps[left][nl:], eps[right][nr:]),
            denominator_threshold,
        )
        if all(shapes[channel]):
            minima.append(minimum)
    weights = {channel: np.zeros_like(array) for channel, array in blocks.items()}
    energy_weights = {spin: np.zeros_like(array) for spin, array in eps.items()}
    equations, derivatives = set(), set()
    count = 0
    with np.errstate(over="raise", invalid="raise"):
        for channel, (left, right) in CHANNEL_SPINS.items():
            shape = shapes[channel]
            nl, nr = occupied[left], occupied[right]
            for starts in product(*(range(0, n, t) for n, t in zip(shape, tile_shape))):
                i, j, a, b = slices = tuple(
                    slice(s, min(s + t, n))
                    for s, t, n in zip(starts, tile_shape, shape)
                )
                feeds = {
                    "g": blocks[channel][slices],
                    "ei": eps[left][:nl][i],
                    "ej": eps[right][:nr][j],
                    "ea": eps[left][nl:][a],
                    "eb": eps[right][nr:][b],
                }
                if left == right:
                    feeds["x"] = blocks[channel][i, j, b, a].swapaxes(2, 3)
                tile = tile_energy_adjoint(
                    feeds,
                    channel=channel,
                    budget_bytes=workspace,
                    denominator_threshold=denominator_threshold,
                )
                weights[channel][slices] += tile.direct
                if tile.exchange is not None:
                    weights[channel][i, j, b, a] += tile.exchange.swapaxes(2, 3)
                energy_weights[left][:nl][i] += tile.occupied_i
                energy_weights[right][:nr][j] += tile.occupied_j
                energy_weights[left][nl:][a] += tile.virtual_a
                energy_weights[right][nr:][b] += tile.virtual_b
                equations.add(tile.equation_hash)
                derivatives.add(tile.derivative_hash)
                count += 1
                del tile, feeds
    # Publication is all-or-nothing even if cumulative arithmetic overflows.
    for array in (*weights.values(), *energy_weights.values()):
        _finite(array)
    return UMP2CanonicalAdjoint(
        MappingProxyType({name: immutable(array) for name, array in weights.items()}),
        MappingProxyType(
            {name: immutable(array) for name, array in energy_weights.items()}
        ),
        reference_identity,
        hamiltonian_id,
        tuple(sorted(equations)),
        tuple(sorted(derivatives)),
        min(minima) if minima else 0.0,
        capacity,
        count,
    )
