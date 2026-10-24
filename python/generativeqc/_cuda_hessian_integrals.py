"""Bounded resident molecular sources built from compiler integral AD tiles."""

from __future__ import annotations

import ctypes as ct
import typing
from functools import partial
from itertools import islice, product
from math import prod, sqrt

import numpy as np
from generativeqc_compiler.integral.ir import four_center_eri_operator
from generativeqc_compiler.integral.range_separation import CoulombKernel
from generativeqc_compiler.integral.resident_second_cuda import (
    emit_resident_df_second,
    emit_resident_second,
)
from generativeqc_compiler.integral.second_derivatives import (
    build_eri_second_ir,
    build_one_electron_second_ir,
)
from generativeqc_compiler.integral.shell_spec import cartesian_components
from generativeqc_compiler.integral.weight_pullback import normalized_radial_primitives


class CudaIntegralSources:
    """Stream shell-local value/gradient/HVP tiles, never a nuclear Jacobian.

    A dense *primal* ERI resident option is explicitly budgeted by the endpoint.
    Derivative storage is always one component tile and <=64 radial products.
    Host loops enumerate fixed shell/layout metadata only; every integral,
    contraction, spherical transform and nuclear scatter runs on the device.
    """

    def __init__(self, topology: typing.Any, execution: typing.Any) -> None:
        self.topology, self.execution, self.cp = topology, execution, execution.cp
        self.shells = topology.shells
        self.offsets = np.cumsum((0, *topology.shell_sizes))
        self.centers = np.asarray([atom.position for atom in topology.atoms])
        self.primitives = self._primitives(self.shells)
        self.programs = {}
        self.statistics = {
            "shell_tiles": 0,
            "primitive_products": 0,
            "component_evaluations": 0,
        }

    @staticmethod
    def _primitives(shells: tuple) -> tuple:
        return tuple(
            normalized_radial_primitives(
                shell.angular_momentum,
                tuple(
                    (entry.exponent, entry.coefficient) for entry in shell.primitives
                ),
            )
            for shell in shells
        )

    def _evaluate(
        self,
        key: tuple,
        source: typing.Callable,
        indices: tuple,
        primitives: tuple,
        atoms: tuple,
        direction: typing.Any,
    ) -> typing.Any:
        if key not in self.programs:
            self.programs[key] = self.execution.function(
                source(),
                "generativeqc_resident_integral",
                [
                    ct.c_void_p,
                    ct.c_size_t,
                    ct.c_void_p,
                    ct.c_void_p,
                    ct.c_void_p,
                    ct.c_void_p,
                    ct.c_void_p,
                ],
                integral=True,
            )
        function = self.programs[key]
        output = self.cp.zeros((len(indices), 1 + 6 * len(atoms)))
        device_atoms = self.cp.asarray(atoms, dtype=self.cp.int64)
        radial_products = product(*primitives)
        while entries := tuple(islice(radial_products, 64)):
            records = self.cp.asarray(
                [
                    (
                        *[value[0] for value in entry],
                        *self.centers[list(atoms)].reshape(-1),
                        prod(value[1] for value in entry),
                    )
                    for entry in entries
                ],
                dtype=self.cp.float64,
            )
            tile = self.cp.empty_like(output)
            self.execution.launch(
                function,
                records.data.ptr,
                len(entries),
                device_atoms.data.ptr,
                direction.data.ptr,
                tile.data.ptr,
            )
            output += tile
            self.statistics["primitive_products"] += len(entries)
            self.statistics["component_evaluations"] += len(entries) * len(indices)
        self.statistics["shell_tiles"] += 1
        return output

    def stream(
        self,
        family: str,
        direction: typing.Any,
        *,
        radial: str = "full_range",
        omega: float = 0.0,
    ) -> typing.Iterator[tuple]:
        """Yield (Cartesian AO tuple, physical atoms, resident derivative rows)."""
        count = 4 if family == "eri" else 2
        for slots in product(range(len(self.shells)), repeat=count):
            shells = tuple(self.shells[slot] for slot in slots)
            angular = tuple(shell.angular_momentum for shell in shells)
            atoms = tuple(shell.atom_index for shell in shells)
            shape = tuple(len(cartesian_components(order)) for order in angular)
            nuclei = (
                range(len(self.centers)) if family == "nuclear_attraction" else (None,)
            )
            for nucleus in nuclei:
                charge = (
                    self.topology.atoms[nucleus].atomic_number
                    if nucleus is not None
                    else 1.0
                )
                physical = atoms if nucleus is None else (*atoms, nucleus)
                integral = (
                    build_eri_second_ir(
                        angular,
                        operator=four_center_eri_operator(CoulombKernel(radial, omega)),
                    )
                    if family == "eri"
                    else build_one_electron_second_ir(
                        family, angular, charge=float(charge), output="weighted_hvp"
                    )
                )
                for begin in range(0, prod(shape), 16):
                    indices = tuple(range(begin, min(begin + 16, prod(shape))))
                    key = (family, angular, radial, omega, charge, indices)
                    values = self._evaluate(
                        key,
                        partial(emit_resident_second, integral, indices),
                        indices,
                        tuple(self.primitives[slot] for slot in slots),
                        physical,
                        direction,
                    )
                    coordinate_rows, normalizations = [], []
                    for index in indices:
                        components = np.unravel_index(index, shape)
                        normalization = prod(
                            prod(range(1, 2 * label.count(axis), 2))
                            for order, entry in zip(angular, components, strict=True)
                            for label in (cartesian_components(order)[entry],)
                            for axis in "xyz"
                        )
                        coordinates = tuple(
                            int(self.offsets[slot] + component)
                            for slot, component in zip(slots, components, strict=True)
                        )
                        coordinate_rows.append(coordinates)
                        normalizations.append(sqrt(normalization))
                    coordinates = tuple(
                        self.cp.asarray(entries, dtype=self.cp.int64)
                        for entries in zip(*coordinate_rows, strict=True)
                    )
                    yield (
                        coordinates,
                        physical,
                        values / self.cp.asarray(normalizations)[:, None],
                    )

    def stream_df(
        self,
        family: str,
        direction: typing.Any,
        orbital: typing.Any,
        auxiliary: typing.Any,
    ) -> typing.Iterator[tuple]:
        bases = (
            (auxiliary, auxiliary)
            if family == "coulomb_metric"
            else (orbital, orbital, auxiliary)
        )
        offsets = tuple(
            np.cumsum(
                (
                    0,
                    *(
                        len(cartesian_components(shell.angular_momentum))
                        for shell in basis.shells
                    ),
                )
            )
            for basis in bases
        )
        primitives = tuple(self._primitives(basis.shells) for basis in bases)
        for slots in product(*(range(len(basis.shells)) for basis in bases)):
            shells = tuple(
                basis.shells[slot] for basis, slot in zip(bases, slots, strict=True)
            )
            angular = tuple(shell.angular_momentum for shell in shells)
            atoms = tuple(shell.atom_index for shell in shells)
            shape = tuple(len(cartesian_components(order)) for order in angular)
            for begin in range(0, prod(shape), 16):
                indices = tuple(range(begin, min(begin + 16, prod(shape))))
                key = (family, angular, indices)
                values = self._evaluate(
                    key,
                    partial(emit_resident_df_second, family, angular, indices),
                    indices,
                    tuple(
                        radials[slot]
                        for radials, slot in zip(primitives, slots, strict=True)
                    ),
                    atoms,
                    direction,
                )
                for row, index in enumerate(indices):
                    components = np.unravel_index(index, shape)
                    normalization = prod(
                        prod(
                            range(
                                1, 2 * cartesian_components(order)[entry].count(axis), 2
                            )
                        )
                        for order, entry in zip(angular, components, strict=True)
                        for axis in "xyz"
                    )
                    coordinates = tuple(
                        int(offset[slot] + component)
                        for offset, slot, component in zip(
                            offsets, slots, components, strict=True
                        )
                    )
                    yield coordinates, atoms, values[row] / sqrt(normalization)

    def tensor(
        self,
        family: str,
        direction: typing.Any,
        *,
        tangent: bool = False,
        **options: typing.Any,
    ) -> typing.Any:
        count = 4 if family == "eri" else 2
        output = self.cp.zeros((self.topology.nbf,) * count)
        for coordinates, atoms, values in self.stream(family, direction, **options):
            self.cp.add.at(
                output,
                coordinates,
                (
                    values[:, 1 : 1 + 3 * len(atoms)]
                    @ direction[list(atoms)].reshape(-1)
                    if tangent
                    else values[:, 0]
                ),
            )
        return output

    def jk(
        self,
        density: typing.Any,
        direction: typing.Any,
        *,
        tangent: bool = False,
        **options: typing.Any,
    ) -> tuple:
        """Bounded all-device fallback when a dense primal ERI is not resident."""
        coulomb = self.cp.zeros(density.shape[1:])
        exchange = self.cp.zeros_like(density)
        total = density.sum(axis=0)
        for coordinates, atoms, values in self.stream("eri", direction, **options):
            first, second, third, fourth = coordinates
            integrals = (
                values[:, 1 : 1 + 3 * len(atoms)] @ direction[list(atoms)].reshape(-1)
                if tangent
                else values[:, 0]
            )
            self.cp.add.at(coulomb, (first, second), integrals * total[third, fourth])
            for spin in range(len(density)):
                self.cp.add.at(
                    exchange[spin],
                    (first, third),
                    integrals * density[spin, second, fourth],
                )
        return coulomb, exchange

    def contract(
        self,
        family: str,
        direction: typing.Any,
        fixed: typing.Any,
        moving: typing.Any = None,
        **options: typing.Any,
    ) -> typing.Any:
        output = self.cp.zeros_like(direction)
        for coordinates, atoms, values in self.stream(family, direction, **options):
            count = len(atoms)
            fixed_tile, moving_tile = (
                fixed(coordinates)
                if callable(fixed)
                else (fixed[coordinates], moving[coordinates])
            )
            gradient = values[:, 1 : 1 + 3 * count]
            hvp = values[:, 1 + 3 * count :]
            self.cp.add.at(
                output,
                self.cp.asarray(atoms),
                (fixed_tile @ hvp + moving_tile @ gradient).reshape(count, 3),
            )
        return output
