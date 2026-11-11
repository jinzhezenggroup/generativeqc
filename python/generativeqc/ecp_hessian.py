"""Bounded analytic or checked-grid ECP geometry matching the native operator."""

from __future__ import annotations

import typing
from functools import cache

from generativeqc_compiler.common.array_execution import arrays as np
from generativeqc_compiler.common.array_graph import evaluate_array_graph
from generativeqc_compiler.common.arrays import immutable
from generativeqc_compiler.dft.ao import directional_ao_jets
from generativeqc_compiler.integral.ecp_grid import harmonic_roots, radial_map_roots
from generativeqc_compiler.integral.ecp_policy import (
    COARSE_POLAR_POINTS,
    COARSE_RADIAL_POINTS,
    DERIVATIVE_ABS_TOLERANCE,
    REFINED_POLAR_POINTS,
    REFINED_RADIAL_POINTS,
)
from generativeqc_compiler.integral.ecp_projector import radial_roots
from generativeqc_compiler.method.stationary_grid_second import density_mixed

from ._dft_gradient import _native_ao_atoms

# A single denser validation grid can distinguish an under-resolved coarse
# check from an inaccurate native result. It never replaces the 224x44 native
# matrix, perturbation, or returned HVP, and is only evaluated after a finite
# mismatch in the original two-grid check.
HESSIAN_VALIDATION_RADIAL_POINTS = 320
HESSIAN_VALIDATION_POLAR_POINTS = 60


@cache
def _quadrature_nodes(count: int) -> tuple:
    """Host setup supplies fixed quadrature constants, never electronic response."""
    import numpy

    return numpy.polynomial.legendre.leggauss(count)


def _quadrature(radial: int, polar: int) -> tuple:
    graph, roots = radial_map_roots()
    nodes, weights = (np.asarray(value) for value in _quadrature_nodes(radial))
    radii, radial_weights = evaluate_array_graph(
        graph, roots, {"z": nodes, "weight": weights}
    )
    nodes, weights = (np.asarray(value) for value in _quadrature_nodes(polar))
    azimuths = np.arange(2 * polar) * np.pi / polar
    heights, azimuths = np.broadcast_arrays(nodes[:, None], azimuths[None])
    sphere = np.stack(
        (
            np.sqrt(1 - heights**2) * np.cos(azimuths),
            np.sqrt(1 - heights**2) * np.sin(azimuths),
            heights,
        ),
        axis=-1,
    ).reshape(-1, 3)
    angular_weights = np.broadcast_to(
        weights[:, None] * np.pi / polar, heights.shape
    ).reshape(-1)
    graph, roots = harmonic_roots()
    harmonics = np.asarray(
        [
            np.broadcast_to(value, (len(sphere),))
            for value in evaluate_array_graph(
                graph, roots, dict(zip("xyz", sphere.T, strict=True))
            )
        ]
    )
    return radii, radial_weights, sphere, angular_weights, harmonics * angular_weights


class ECPSecondOrder:
    """Match native residual channels with bounded validation and no displaced SCFs.

    CPU Gaussian residuals use the native analytic primal and derivative owner.
    Other operators retain a fixed ECP-centered radial/angular rule. Motion acts on AO
    jets relative to the ECP center; angular projection is a fixed linear map.
    TensorIR generates the mixed AO/density derivative of each contracted
    local or nonlocal energy. Only one radial layer is ever retained.
    """

    def __init__(
        self, state: typing.Any, basis: typing.Any, *, budget_bytes: int
    ) -> None:
        self.basis, self.budget_bytes = basis, budget_bytes
        self.terms = np.asarray(state._source.ecp_terms)
        self.centers = np.asarray([atom.position for atom in basis.atoms])
        self.ao_atoms = _native_ao_atoms(basis)
        self.ecp_centers = np.flatnonzero(np.asarray(state._source.ecp_cores))
        self.statistics = {
            "radial_layers": 0,
            "ao_points": 0,
            "hvp_validation_refinements": 0,
            "hvp_coarse_error": None,
            "hvp_validation_error": None,
        }
        self._analytic_source = (
            state._source
            if state._source.backend == "cpu"
            and all(
                term[2] == 2 and -1 <= term[1] <= 2 for term in state._source.ecp_terms
            )
            else None
        )
        if self._analytic_source is not None:
            first, first_work = self._analytic_source.ecp_gaussian(
                budget_bytes=budget_bytes
            )
            self.matrix = immutable(first[:, 0].sum(axis=0))
            self._analytic_first = first[:, 1:]
            self._analytic_density = immutable(state.density.sum(axis=0))
            retained = first.nbytes + self.matrix.nbytes + self._analytic_density.nbytes
            if retained >= budget_bytes:
                raise MemoryError("retained Gaussian ECP sources exceed budget_bytes")
            second, second_work = self._analytic_source.ecp_gaussian(
                budget_bytes=budget_bytes - retained, weights=self._analytic_density
            )
            self._analytic_second = second
            self._analytic_retained_bytes = retained + second.nbytes
            self.statistics.update(
                analytic_gaussian=True,
                resident_bytes=self._analytic_retained_bytes,
                **{"first_" + name: value for name, value in first_work.items()},
                **{"second_" + name: value for name, value in second_work.items()},
            )
            return
        self.matrix = immutable(self._integrate("matrix"))

    def _layers(self, radial: int, polar: int) -> typing.Iterator[tuple]:
        radii, radial_weights, sphere, angular_weights, projector = _quadrature(
            radial, polar
        )
        if 64 * len(sphere) * self.basis.nao * 8 > self.budget_bytes:
            raise MemoryError("ECP radial-layer jets exceed grid_budget_bytes")
        for center in self.ecp_centers:
            potentials = np.zeros((5, radial))
            for atom, channel, power, exponent, coefficient in self.terms:
                if int(atom) != center:
                    continue
                graph, roots = radial_roots(int(power))
                potentials[int(channel) + 1] += evaluate_array_graph(
                    graph,
                    roots,
                    {
                        "r": radii,
                        "weight": radial_weights,
                        "alpha": exponent,
                        "coefficient": coefficient,
                    },
                )[0]
            for index, radius in enumerate(radii):
                if not np.any(potentials[:, index]):
                    continue
                points = self.centers[center] + radius * sphere
                jets = self.basis.evaluate(points, 2, budget_bytes=self.budget_bytes)
                self.statistics["radial_layers"] += 1
                self.statistics["ao_points"] += len(points)
                yield (
                    center,
                    jets,
                    projector,
                    (
                        angular_weights * potentials[0, index],
                        np.repeat(potentials[1:, index], (1, 3, 5, 7)),
                    ),
                )

    def _motion(
        self, jets: np.ndarray, vector: np.ndarray, center: int, order: int
    ) -> np.ndarray:
        return directional_ao_jets(
            jets,
            order,
            ao_atoms=self.ao_atoms,
            point_motion=np.broadcast_to(vector[center], (jets.shape[1], 3)),
            center_motion=vector,
        )

    def _integrate(
        self,
        operation: str,
        *,
        vector: np.ndarray | None = None,
        density: np.ndarray | None = None,
        delta_density: np.ndarray | None = None,
        radial: int = REFINED_RADIAL_POINTS,
        polar: int = REFINED_POLAR_POINTS,
    ) -> np.ndarray:
        shape = (
            (2, len(self.centers), 3) if operation == "hvp" else (self.basis.nao,) * 2
        )
        result = np.zeros(shape)
        for center, jets, projector, weights in self._layers(radial, polar):
            motion = None if vector is None else self._motion(jets, vector, center, 0)
            for component, (values, measure) in enumerate(
                zip(
                    (jets, np.einsum("mp,jpa->jma", projector, jets)),
                    weights,
                    strict=True,
                )
            ):
                if not np.any(measure):
                    continue
                if operation == "matrix":
                    result += values[0].T @ (measure[:, None] * values[0])
                    continue
                right = (
                    motion
                    if component == 0
                    else np.einsum("mp,jpa->jma", projector, motion)
                )
                if operation == "frozen":
                    block = right[0].T @ (measure[:, None] * values[0])
                    result += block + block.T
                    continue
                for coordinate in range(vector.size):
                    left = np.zeros_like(vector)
                    left.reshape(-1)[coordinate] = 1
                    left_extended = self._motion(jets, left, center, 1)
                    mixed = self._motion(left_extended, vector, center, 0)
                    left_jets = left_extended[:1]
                    if component:
                        left_jets = np.einsum("mp,jpa->jma", projector, left_jets)
                        mixed = np.einsum("mp,jpa->jma", projector, mixed)
                    mixed_features = density_mixed(
                        values[0],
                        density,
                        left_jets[0],
                        right[0],
                        mixed[0],
                        delta_density,
                        max_bytes=self.budget_bytes,
                    )
                    result[component].reshape(-1)[coordinate] += np.dot(
                        measure, mixed_features
                    )
        return result

    def frozen_fock(self, vector: np.ndarray) -> np.ndarray:
        """Differentiate the actual residual local/projector AO operator."""
        if self._analytic_source is not None:
            self._analytic_source.check_current()
            direction = np.asarray(vector).reshape(-1)
            if vector.shape != self.centers.shape or not np.isfinite(direction).all():
                raise ValueError(
                    "ECP direction must match the finite nuclear coordinates"
                )
            result = np.einsum("pcab,c->ab", self._analytic_first, direction)
            self._analytic_source.check_current()
            return immutable(result)
        return immutable(self._integrate("frozen", vector=vector))

    def hvp(
        self, vector: np.ndarray, density: np.ndarray, delta_density: np.ndarray
    ) -> dict:
        """Use the native analytic operator or validate the fixed grid operator.

        Analytic sources reuse the fixed-density contracted coordinate Hessian,
        adding the density-response contraction with the same first derivatives.
        The fallback retains the original two-grid fast path. A mismatch permits one
        denser check of the native fine-grid result at the same absolute
        tolerance; nonfinite values and unresolved refinement still reject.
        """
        if self._analytic_source is not None:
            self._analytic_source.check_current()
            direction = np.asarray(vector).reshape(-1)
            if vector.shape != self.centers.shape or not np.isfinite(direction).all():
                raise ValueError(
                    "ECP direction must match the finite nuclear coordinates"
                )
            expected = self._analytic_density.shape
            if (
                density.shape != expected
                or delta_density.shape != expected
                or not (np.isfinite(density).all() and np.isfinite(delta_density).all())
            ):
                raise ValueError(
                    "ECP response densities require finite ordered AO matrices"
                )
            second = self._analytic_second
            if not np.array_equal(density, self._analytic_density):
                if self._analytic_retained_bytes >= self.budget_bytes:
                    raise MemoryError(
                        "retained Gaussian ECP sources exceed budget_bytes"
                    )
                second, work = self._analytic_source.ecp_gaussian(
                    budget_bytes=self.budget_bytes - self._analytic_retained_bytes,
                    weights=density,
                )
                self.statistics.update(
                    {"last_second_" + name: value for name, value in work.items()}
                )
            coordinates = direction.size
            curvature = second[:, 1 + coordinates :].reshape(
                2, coordinates, coordinates
            )
            values = np.einsum("pcd,d->pc", curvature, direction)
            values += np.einsum("pcab,ab->pc", self._analytic_first, delta_density)
            self._analytic_source.check_current()
            return {
                name: immutable(value.reshape(self.centers.shape))
                for name, value in zip(
                    ("ecp_local", "ecp_nonlocal"), values, strict=True
                )
            }
        self.statistics["hvp_coarse_error"] = None
        self.statistics["hvp_validation_error"] = None
        options = {"vector": vector, "density": density, "delta_density": delta_density}
        coarse = self._integrate(
            "hvp", radial=COARSE_RADIAL_POINTS, polar=COARSE_POLAR_POINTS, **options
        )
        fine = self._integrate("hvp", **options)
        if not np.isfinite(coarse).all() or not np.isfinite(fine).all():
            raise ArithmeticError("ECP Hessian quadrature convergence gate failed")
        error = np.max(abs(fine - coarse))
        if not np.isfinite(error):
            raise ArithmeticError("ECP Hessian quadrature convergence gate failed")
        self.statistics["hvp_coarse_error"] = float(error)
        if error > DERIVATIVE_ABS_TOLERANCE:
            self.statistics["hvp_validation_refinements"] += 1
            validation = self._integrate(
                "hvp",
                radial=HESSIAN_VALIDATION_RADIAL_POINTS,
                polar=HESSIAN_VALIDATION_POLAR_POINTS,
                **options,
            )
            if not np.isfinite(validation).all():
                raise ArithmeticError("ECP Hessian quadrature convergence gate failed")
            error = np.max(abs(validation - fine))
            if not np.isfinite(error):
                raise ArithmeticError("ECP Hessian quadrature convergence gate failed")
            self.statistics["hvp_validation_error"] = float(error)
            if error > DERIVATIVE_ABS_TOLERANCE:
                raise ArithmeticError("ECP Hessian quadrature convergence gate failed")
        return {
            name: immutable(value)
            for name, value in zip(("ecp_local", "ecp_nonlocal"), fine, strict=True)
        }
