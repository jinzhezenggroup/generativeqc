"""Bounded full-rank DF source adapter for composite stationary Hessians.

The prepared approximation is fitted full-range J/K plus exact LR exchange.
Raw three-center/metric derivatives and all stationary cotangent derivatives
are compiler generated. No direct-integral substitution or displaced SCF is
permitted. The native metric cutoff is checked before using inverse response.
"""

import typing
from fractions import Fraction
from itertools import product
from pathlib import Path

from generativeqc_compiler.common.array_execution import arrays as np
from generativeqc_compiler.common.arrays import immutable
from generativeqc_compiler.integral.df_second_execute import DFSecondEvaluator
from generativeqc_compiler.integral.shell_spec import cartesian_components
from generativeqc_compiler.integral.weight_pullback import normalized_radial_primitives
from generativeqc_compiler.method.stationary_df_second import df_second_programs
from generativeqc_compiler.tensor import execute

from .hessian_basis import HessianBasis


class DFSecondOrder:
    """Own one fixed fitting metric and stream molecular derivative contractions.

    The numeric budget includes dense primal/directional source tensors,
    cotangents, representation pullbacks and conservative TensorIR storage.
    Compiler processes, Python metadata, library mappings and the native SCF
    and Fock-plan owners have their existing separate resource contracts.
    """

    def __init__(
        self,
        state: typing.Any,
        orbital: HessianBasis,
        auxiliary: typing.Any,
        threshold: float,
        *,
        budget_bytes: int,
    ) -> None:
        self.state = state
        self.orbital = orbital
        self.auxiliary = auxiliary
        self.auxiliary_map = HessianBasis(auxiliary)
        self.spins = len(state.density)
        self.exchange = next(
            primitive.coefficient
            for primitive in state._source.method_ir.primitives
            if primitive.kind == "range_separated_exchange"
            and primitive.operator == "short-range"
        )
        self.threshold = threshold
        self.budget_bytes = budget_bytes
        self.evaluators = {}
        self.cache = None
        self.statistics = {"shell_tiles": 0, "primitive_products": 0}
        count, auxiliary_count = orbital.cartesian.nao, self.auxiliary_map.cartesian.nao
        self.reserved_bytes = 8 * (
            64 * count * count * auxiliary_count
            + 80 * auxiliary_count * auxiliary_count
            + 4096
        )
        if self.reserved_bytes > budget_bytes:
            self.close()
            raise ValueError("DF Hessian source storage exceeds integral_budget_bytes")
        self.external_bytes = 8 * (
            16 * count * count * auxiliary_count
            + 20 * auxiliary_count * auxiliary_count
            + 4096
        )
        self.ad_budget_bytes = budget_bytes - self.external_bytes
        try:
            self.programs = df_second_programs(
                orbital.public.nao,
                auxiliary.nao,
                self.spins,
                Fraction(str(self.exchange)),
            )
        except Exception:
            self.close()
            raise

    def close(self) -> None:
        """Release only the owned Cartesian auxiliary representation."""
        self.auxiliary_map.close()

    def check_budget(self, budget_bytes: int) -> None:
        """Recheck each HVP's allowance even when reusing prepared DF sources."""
        if self.reserved_bytes > budget_bytes:
            raise ValueError("retained DF Hessian sources exceed integral_budget_bytes")
        self.budget_bytes = budget_bytes
        self.ad_budget_bytes = budget_bytes - self.external_bytes

    def _stream(self, family: str, direction: np.ndarray) -> typing.Iterator[tuple]:
        bases = (
            (self.auxiliary_map.cartesian,) * 2
            if family == "coulomb_metric"
            else (self.orbital.cartesian,) * 2 + (self.auxiliary_map.cartesian,)
        )
        offsets = [
            np.cumsum(
                (
                    0,
                    *[
                        (shell.angular_momentum + 1) * (shell.angular_momentum + 2) // 2
                        for shell in basis.shells
                    ],
                )
            )
            for basis in bases
        ]
        centers = np.asarray([atom.position for atom in bases[0].atoms])
        for slots in product(*(range(len(basis.shells)) for basis in bases)):
            shells = tuple(
                basis.shells[slot] for basis, slot in zip(bases, slots, strict=True)
            )
            angular = tuple(shell.angular_momentum for shell in shells)
            labels = tuple(cartesian_components(order) for order in angular)
            components = tuple(product(*(range(len(entries)) for entries in labels)))
            atoms = tuple(shell.atom_index for shell in shells)
            primitives = tuple(
                normalized_radial_primitives(
                    shell.angular_momentum,
                    tuple(
                        (entry.exponent, entry.coefficient)
                        for entry in shell.primitives
                    ),
                )
                for shell in shells
            )
            for start in range(0, len(components), 16):
                indices = tuple(range(start, min(start + 16, len(components))))
                key = family, angular, indices
                if key not in self.evaluators:
                    self.evaluators[key] = DFSecondEvaluator(*key, self.cache)
                values = self.evaluators[key].contract(
                    primitives, centers[list(atoms)], direction[list(atoms)]
                )
                self.statistics["shell_tiles"] += 1
                self.statistics["primitive_products"] += np.prod(
                    [len(entries) for entries in primitives]
                )
                for packed, component in enumerate(
                    components[start : start + len(indices)]
                ):
                    normalization = 1.0
                    for position, entry in enumerate(component):
                        for axis in "xyz":
                            power = labels[position][entry].count(axis)
                            normalization *= np.prod(tuple(range(1, 2 * power, 2)))
                    coordinates = tuple(
                        int(offset[slot] + entry)
                        for offset, slot, entry in zip(
                            offsets, slots, component, strict=True
                        )
                    )
                    yield coordinates, atoms, values[packed] / np.sqrt(normalization)

    def _maps(self, family: str) -> tuple:
        orbital = self.orbital.transform
        auxiliary = self.auxiliary_map.transform
        return (
            (auxiliary,) * 2
            if family == "coulomb_metric"
            else (orbital, orbital, auxiliary)
        )

    def _public(self, value: np.ndarray, family: str) -> np.ndarray:
        for axis, transform in enumerate(self._maps(family)):
            if transform is not None:
                value = np.moveaxis(
                    np.tensordot(transform.T, value, axes=(1, axis)), 0, axis
                )
        return value

    def _pullback(self, value: np.ndarray, family: str) -> np.ndarray:
        for axis, transform in enumerate(self._maps(family)):
            if transform is not None:
                value = np.moveaxis(
                    np.tensordot(transform, value, axes=(1, axis)), 0, axis
                )
        return value

    def _values(self, direction: np.ndarray, *, tangent: bool) -> tuple:
        results = []
        for family in ("three_center_eri", "coulomb_metric"):
            shape = (
                (self.orbital.cartesian.nao,) * 2 + (self.auxiliary_map.cartesian.nao,)
                if family == "three_center_eri"
                else (self.auxiliary_map.cartesian.nao,) * 2
            )
            output = np.zeros(shape)
            for coordinates, atoms, values in self._stream(family, direction):
                output[coordinates] = (
                    np.dot(
                        values[1 : 1 + 3 * len(atoms)],
                        direction[list(atoms)].reshape(-1),
                    )
                    if tangent
                    else values[0]
                )
            results.append(self._public(output, family))
        return tuple(results)

    def prepare(self, cache: Path) -> None:
        """Bind the JIT cache and validate the native full-rank metric active set."""
        self.cache = Path(cache)
        self.three_center, self.metric = self._values(
            np.zeros((len(self.orbital.public.atoms), 3)), tangent=False
        )
        eigenvalues, eigenvectors = np.linalg.eigh(self.metric)
        cutoff = self.threshold * eigenvalues[-1]
        if eigenvalues[0] <= max(
            cutoff * 1.1, 64 * np.finfo(float).eps * eigenvalues[-1]
        ):
            raise NotImplementedError(
                "DF Hessian requires a full-rank metric separated from the native cutoff; rank-truncated metric response is not qualified"
            )
        self.inverse = immutable((eigenvectors / eigenvalues) @ eigenvectors.T)
        self.three_center = immutable(self.three_center)
        self.metric = immutable(self.metric)

    def _feeds(
        self, direction: np.ndarray, density_direction: np.ndarray | None = None
    ) -> dict:
        moving_values, moving_metric = self._values(direction, tangent=True)
        inverse_direction = execute(
            self.programs[3],
            {"inverse": self.inverse, "metric_direction": moving_metric},
            max_bytes=self.ad_budget_bytes,
        ).outputs["direction"]
        return {
            "density": self.state.density,
            "three_center": self.three_center,
            "inverse": self.inverse,
            "d_three_center": moving_values,
            "d_inverse": inverse_direction,
            "d_density": np.zeros_like(self.state.density)
            if density_direction is None
            else density_direction,
        }

    def frozen_fock(self, direction: np.ndarray) -> np.ndarray:
        """Differentiate fitted J/K at fixed physical-spin AO densities."""
        feeds = self._feeds(direction)
        feeds["bar_energy"] = np.asarray(1.0)
        value = execute(
            self.programs[2], feeds, max_bytes=self.ad_budget_bytes
        ).outputs["d_bar_density"]
        return value if self.spins == 2 else value[0]

    def fock(self) -> np.ndarray:
        """Reconstruct fitted J/K independently of the native Fock provider."""
        feeds = {
            "density": self.state.density,
            "three_center": self.three_center,
            "inverse": self.inverse,
            "bar_energy": np.asarray(1.0),
        }
        value = execute(
            self.programs[4], feeds, max_bytes=self.ad_budget_bytes
        ).outputs["bar_density"]
        return value if self.spins == 2 else value[0]

    def hvp(self, direction: np.ndarray, density_direction: np.ndarray) -> dict:
        """Contract fixed second integrals and moving first-integral cotangents."""
        if density_direction.ndim == 2:
            density_direction = density_direction[None]
        feeds = self._feeds(direction, density_direction)
        result = {}
        for name, (weights, moving_weights) in zip(
            ("coulomb", "exchange"), self.programs[1], strict=True
        ):
            feeds[f"bar_{name}"] = np.asarray(1.0)
            fixed = execute(weights, feeds, max_bytes=self.ad_budget_bytes).outputs
            moving = execute(
                moving_weights, feeds, max_bytes=self.ad_budget_bytes
            ).outputs
            output = np.zeros_like(direction)
            for field, family in (
                ("three_center", "three_center_eri"),
                ("metric", "coulomb_metric"),
            ):
                fixed_cart = self._pullback(fixed[field], family)
                moving_cart = self._pullback(moving[f"d_{field}"], family)
                for coordinates, atoms, values in self._stream(family, direction):
                    count = len(atoms)
                    gradient = values[1 : 1 + 3 * count].reshape(count, 3)
                    second = values[1 + 3 * count :].reshape(count, 3)
                    np.add.at(
                        output,
                        list(atoms),
                        fixed_cart[coordinates] * second
                        + moving_cart[coordinates] * gradient,
                    )
            result[name] = immutable(output)
        return result
