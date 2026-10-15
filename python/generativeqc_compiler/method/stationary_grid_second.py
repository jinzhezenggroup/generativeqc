"""Method-level compiler AD of restricted meta-GGA/VV10 stationary sources.

AO features use TensorIR JVPs, and scalar functional/pair derivatives use the
shared Graph. These are fixed-branch partials, not an SCF differentiator. The
molecular consumer must supply the implicit density response and Becke motion.
"""

from __future__ import annotations

import math
import typing
from dataclasses import dataclass, replace
from fractions import Fraction
from functools import cache

from generativeqc_compiler.common.array_execution import arrays as np
from generativeqc_compiler.common.array_graph import evaluate_array_graph
from generativeqc_compiler.common.arrays import immutable
from generativeqc_compiler.common.provenance import canonical_hash
from generativeqc_compiler.dft.nonlocal_policy import MOLECULAR_VV10_DENSITY_THRESHOLD
from generativeqc_compiler.integral.expr import Graph
from generativeqc_compiler.method.nonlocal_pair import vv10_grid_pair_expression
from generativeqc_compiler.tensor import (
    Index,
    IndexSpace,
    Program,
    TensorSpec,
    add,
    einsum,
    execute,
    input_tensor,
    linearize,
)
from generativeqc_compiler.xc.coefficients import coefficient_program
from generativeqc_compiler.xc.program import build_program
from generativeqc_compiler.xc.wb97mv_maple import (
    DENSITY_THRESHOLD,
    SIGMA_THRESHOLD,
    TAU_THRESHOLD,
    WB97MV_COMPONENTS,
)

FIELD_COUNT = 9
FIELD_LAYOUT = ("rho", "gx", "gy", "gz", "tau", "weight", "x", "y", "z")
GRID_FIELD_WORKSPACE_PANELS = 32


@cache
def _nuclear_pair_program() -> tuple:
    graph = Graph()
    displacement = [graph.variable(f"r{axis}") for axis in range(3)]
    energy = graph.variable("charge_product") * graph.sum(
        value * value for value in displacement
    ).pow(-0.5)
    gradient = [graph.differentiate(energy, value) for value in displacement]
    leaves = {f"r{axis}": graph.variable(f"dr{axis}") for axis in range(3)}
    roots = tuple(
        graph.differentiate(value, graph.variable("direction"), leaves)
        for value in gradient
    )
    return graph, roots


def nuclear_repulsion_hvp(
    centers: np.ndarray, charges: np.ndarray, direction: np.ndarray
) -> np.ndarray:
    """Apply the compiler-differentiated 1/r pair Hessian with physical scatter."""
    first, second = np.triu_indices(len(centers), 1)
    result = np.zeros_like(centers)
    if not len(first):
        return result
    delta = centers[first] - centers[second]
    moving = direction[first] - direction[second]
    if np.any(np.linalg.norm(delta, axis=1) == 0):
        raise ValueError("nuclear Hessian requires distinct centers")
    graph, roots = _nuclear_pair_program()
    variables = {
        "charge_product": charges[first] * charges[second],
        **{f"r{axis}": delta[:, axis] for axis in range(3)},
        **{f"dr{axis}": moving[:, axis] for axis in range(3)},
    }
    values = np.asarray(evaluate_array_graph(graph, roots, variables)).T
    np.add.at(result, first, values)
    np.add.at(result, second, -values)
    return result


@cache
def feature_programs(points: int, nao: int) -> tuple:
    """Generate primal/first/mixed rho, Cartesian gradient and tau programs.

    The second JVP differentiates the first tangent too, so moving AO bases and
    a right-hand density response are both present in a mixed nuclear partial.
    No point-by-AO-pair panel is materialized.
    """
    point = Index("point", IndexSpace("point", "batch", points))
    orbital = Index("ao", IndexSpace("ao", "ao", nao))
    other = Index("other", orbital.space)
    density = input_tensor(
        "density", TensorSpec((orbital, other), role="input", differentiable=True)
    )
    aos = [
        input_tensor(
            f"ao{axis}", TensorSpec((point, orbital), role="input", differentiable=True)
        )
        for axis in range(4)
    ]

    def bilinear(left: typing.Any, right: typing.Any) -> typing.Any:
        return einsum("pi,ij,pj->p", left, density, right)

    primal = Program(
        {
            "rho": bilinear(aos[0], aos[0]),
            **{
                f"g{axis}": add(
                    bilinear(aos[axis + 1], aos[0]), bilinear(aos[0], aos[axis + 1])
                )
                for axis in range(3)
            },
            "tau": add(
                *(bilinear(value, value) for value in aos[1:]),
                coefficients=(Fraction(1, 2),) * 3,
            ),
        }
    )
    names = ("density", "ao0", "ao1", "ao2", "ao3")
    first = linearize(primal, names).program
    return primal, first


def features(
    jets: np.ndarray,
    density: np.ndarray,
    *,
    left: np.ndarray | None = None,
    right: np.ndarray | None = None,
    mixed: np.ndarray | None = None,
    delta_density: np.ndarray | None = None,
    max_bytes: int = 256 << 20,
) -> np.ndarray:
    """Evaluate a total-D AO partial within the TensorIR retained-byte budget."""
    primal, first = feature_programs(jets.shape[1], jets.shape[2])
    feeds = {"density": density, **{f"ao{axis}": jets[axis] for axis in range(4)}}
    labels = ("rho", "g0", "g1", "g2", "tau")
    if left is None:
        result = execute(primal, feeds, max_bytes=max_bytes).outputs
    else:
        feeds.update({f"d_ao{axis}": left[axis] for axis in range(4)})
        feeds["d_density"] = (
            np.zeros_like(density) if delta_density is None else delta_density
        )
        if right is None:
            result = execute(first, feeds, max_bytes=max_bytes).outputs
            labels = tuple(f"d_{label}" for label in labels)
        else:
            return _mixed_features(
                jets, density, left, right, mixed, feeds["d_density"], max_bytes
            )
    return np.stack([result[label] for label in labels])


@cache
def _mixed_program(points: int, nao: int) -> Program:
    """Separate left and right namespaces before nested TensorIR linearization."""
    _, first = feature_programs(points, nao)
    replacements = {}
    for node in first.nodes:
        attributes = node.attributes
        if node.op == "input" and node.attrs["name"].startswith("d_"):
            attributes = (("name", "left_" + node.attrs["name"][2:]),)
        replacements[node] = replace(
            node,
            inputs=tuple(replacements[value] for value in node.inputs),
            spec=replace(node.spec, differentiable=True),
            attributes=attributes,
        )
    renamed = Program(
        {name: replacements[value] for name, value in first.outputs.items()}
    )
    inputs = tuple(node.attrs["name"] for node in renamed.nodes if node.op == "input")
    return linearize(renamed, inputs).program


def _mixed_features(
    jets: np.ndarray,
    density: np.ndarray,
    left: np.ndarray,
    right: np.ndarray,
    mixed: np.ndarray,
    delta_density: np.ndarray,
    max_bytes: int,
) -> np.ndarray:
    program = _mixed_program(jets.shape[1], jets.shape[2])
    feeds = {
        "density": density,
        "left_density": np.zeros_like(density),
        "d_density": delta_density,
        "d_left_density": np.zeros_like(density),
    }
    for axis in range(4):
        feeds.update(
            {
                f"ao{axis}": jets[axis],
                f"left_ao{axis}": left[axis],
                f"d_ao{axis}": right[axis],
                f"d_left_ao{axis}": mixed[axis],
            }
        )
    result = execute(program, feeds, max_bytes=max_bytes).outputs
    return np.stack(
        [result[f"d_d_{label}"] for label in ("rho", "g0", "g1", "g2", "tau")]
    )


@cache
def _density_mixed_program(points: int, nao: int) -> Program:
    mixed = _mixed_program(points, nao)
    return Program({"value": mixed.outputs["d_d_rho"]})


def density_mixed(
    values: np.ndarray,
    density: np.ndarray,
    left: np.ndarray,
    right: np.ndarray,
    mixed: np.ndarray,
    delta_density: np.ndarray,
    *,
    max_bytes: int,
) -> np.ndarray:
    """Demand only the AO bilinear's mixed JVP, without gradient/tau panels.

    Projector energies depend on AO values, not on meta-GGA ingredients. Reuse
    the same compiler AD graph but prune unused outputs before execution.
    """
    feeds = {
        "density": density,
        "left_density": np.zeros_like(density),
        "d_density": delta_density,
        "d_left_density": np.zeros_like(density),
        "ao0": values,
        "left_ao0": left,
        "d_ao0": right,
        "d_left_ao0": mixed,
    }
    return execute(
        _density_mixed_program(*values.shape), feeds, max_bytes=max_bytes
    ).outputs["value"]


@dataclass(frozen=True)
class GridPartials:
    """Total-energy partials and their JVP in the declared Cartesian field ABI."""

    energy: float
    coefficients: np.ndarray
    directional_coefficients: np.ndarray
    identity: str
    pair_evaluations: int = 0
    semilocal: tuple = ()
    nonlocal_correlation: tuple = ()

    def __post_init__(self) -> None:
        """Detach cached partials so response actions cannot mutate one another."""
        for name in ("coefficients", "directional_coefficients"):
            object.__setattr__(self, name, immutable(getattr(self, name)))
        for name in ("semilocal", "nonlocal_correlation"):
            object.__setattr__(
                self, name, tuple(immutable(value) for value in getattr(self, name))
            )


@cache
def _pair_program(spec: typing.Any) -> tuple:
    """Differentiate the VV10 energy, including both legs' changing local scales."""
    if spec.variant != "vv10":
        raise NotImplementedError("second-order nonlocal grid currently requires VV10")
    graph = Graph()
    legs = [
        [graph.variable(f"{side}{index}") for index in range(FIELD_COUNT)]
        for side in ("a", "b")
    ]

    first, second = legs
    energy = vv10_grid_pair_expression(graph, first, second, spec)
    gradients = tuple(graph.differentiate(energy, variable) for variable in first)
    leaves = {
        f"{side}{index}": graph.variable(f"d{side}{index}")
        for side in ("a", "b")
        for index in range(FIELD_COUNT)
    }
    tangent = graph.variable("direction")
    directional = tuple(
        graph.differentiate(root, tangent, leaves) for root in gradients
    )
    return graph, (energy, *gradients, *directional)


class StationaryGridSecondOrder:
    """Bounded compiler-AD grid partials for a restricted meta-GGA/VV10 method.

    The Libxc work policy is applied outside functional differentiation, exactly
    as in SCF. The VV10 rho>=1e-8 mask is fixed on both legs. Work and temporary
    pair capacity have independent explicit limits; neither is a scaling claim.
    The full-field reserve covers cached reference partials, current response
    partials, geometry directions and immutable-publication copies together;
    scalar AD tiles use only the remaining budget.
    """

    def __init__(
        self,
        functional: typing.Any,
        nonlocal_primitive: typing.Any,
        *,
        budget_bytes: int = 128 << 20,
        max_pair_evaluations: int = 16_777_216,
    ) -> None:
        if functional.spin not in ("unpolarized", "polarized"):
            raise NotImplementedError("stationary grid second order requires RKS/UKS")
        self.spins = 2 if functional.spin == "polarized" else 1
        self.feature_count = 5 * self.spins
        self.field_count = self.feature_count + 4
        if {name for name, value in functional.components if value} != set(
            WB97MV_COMPONENTS
        ):
            raise NotImplementedError(
                "stationary grid work policy currently requires WB97M-V"
            )
        if type(budget_bytes) is not int or budget_bytes < 1:
            raise ValueError("grid_budget_bytes must be a positive integer")
        if type(max_pair_evaluations) is not int or max_pair_evaluations < 1:
            raise ValueError("max_nonlocal_pairs must be a positive integer")
        self.functional = functional
        self.nonlocal_primitive = nonlocal_primitive
        self.budget_bytes = budget_bytes
        self.max_pair_evaluations = max_pair_evaluations
        self.program = build_program(functional, order=2)
        pair_graph, pair_roots = _pair_program(nonlocal_primitive.spec)
        order = pair_graph.topological_order(pair_roots)
        positions = {identifier: index for index, identifier in enumerate(order)}
        pair_identity = canonical_hash(
            {
                "nodes": [
                    (
                        pair_graph.nodes[identifier].operation,
                        tuple(
                            positions[value]
                            for value in pair_graph.nodes[identifier].arguments
                        ),
                        str(pair_graph.nodes[identifier].payload),
                    )
                    for identifier in order
                ],
                "roots": tuple(positions[root.identifier] for root in pair_roots),
            }
        )
        self.identity = canonical_hash(
            {
                "schema": "generativeqc.stationary-grid-second-order/v1",
                "functional": functional.identity,
                "point_program": self.program.expression_hash,
                "pair_program": pair_identity,
                "nonlocal": nonlocal_primitive.to_payload(),
                "fields": FIELD_LAYOUT
                if self.spins == 1
                else tuple(
                    f"{spin}_{name}"
                    for spin in ("alpha", "beta")
                    for name in FIELD_LAYOUT[:5]
                )
                + FIELD_LAYOUT[5:],
            }
        )

    def evaluate(self, fields: np.ndarray, direction: np.ndarray) -> GridPartials:
        """Return exact fixed-branch energy partials and their AD JVP."""
        if self.spins == 2:
            return self._polarized_partials(fields, direction)
        if fields.shape != direction.shape or fields.shape[0] != FIELD_COUNT:
            raise ValueError("grid fields/directions require [9,point]")
        if not np.isfinite(fields).all() or not np.isfinite(direction).all():
            raise ValueError("grid fields/directions must be finite")
        if np.any(fields[0] < 0) or np.any(fields[4] < 0):
            raise ValueError("grid rho/tau must be nonnegative")
        retained_bytes = fields.nbytes * GRID_FIELD_WORKSPACE_PANELS
        if retained_bytes > self.budget_bytes:
            raise MemoryError("retained grid partials exceed grid_budget_bytes")
        base = np.zeros_like(fields)
        delta = np.zeros_like(fields)
        indices = np.flatnonzero(fields[0] >= float(MOLECULAR_VV10_DENSITY_THRESHOLD))
        count = len(indices)
        pairs = count * count
        if pairs > self.max_pair_evaluations:
            raise ValueError("VV10 second-order work exceeds max_nonlocal_pairs")
        active_indices = np.flatnonzero(fields[0] >= DENSITY_THRESHOLD)
        point_nodes = len(self.program.graph.topological_order(self.program.roots))
        remaining = self.budget_bytes - retained_bytes
        point_tile = min(256, remaining // (32 * point_nodes))
        if len(active_indices) and point_tile < 1:
            raise MemoryError("semilocal AD point tile exceeds grid_budget_bytes")
        energy = 0.0
        for begin in range(0, len(active_indices), max(1, point_tile)):
            active = active_indices[begin : begin + point_tile]
            values, moving = fields[:, active], direction[:, active]
            rho, gradient, tau = values[0], values[1:4].T, values[4]
            drho, dgradient, dtau = moving[0], moving[1:4].T, moving[4]
            sigma = np.sum(gradient * gradient, axis=1)
            dsigma = 2 * np.sum(gradient * dgradient, axis=1)
            sigma_floor = 4 * SIGMA_THRESHOLD**2
            work = np.stack(
                (
                    np.maximum(rho, 2 * DENSITY_THRESHOLD),
                    np.maximum(sigma, sigma_floor),
                    np.maximum(tau, 2 * TAU_THRESHOLD),
                )
            )
            dwork = np.stack(
                (
                    drho * (rho > 2 * DENSITY_THRESHOLD),
                    dsigma * (sigma > sigma_floor),
                    dtau * (tau > 2 * TAU_THRESHOLD),
                )
            )
            rows = np.asarray(
                evaluate_array_graph(
                    self.program.graph,
                    self.program.roots,
                    dict(zip(self.functional.features, work, strict=True)),
                )
            )
            point = self.program.unpack(rows)
            potential = point["gradient"]
            dpotential = np.einsum("ijp,jp->ip", point["hessian"], dwork)
            pullback = coefficient_program("unpolarized", "mgga")
            response = coefficient_program("unpolarized", "mgga", response=True)
            coeff = pullback.evaluate(gradient, potential)
            dcoeff = response.evaluate(
                gradient, potential, delta_gradient=dgradient, delta_v=dpotential
            )
            cartesian = np.vstack(
                (coeff["rho"][0], coeff["gradient"][0].T, 2 * coeff["tau"][0])
            )
            dcartesian = np.vstack(
                (dcoeff["rho"][0], dcoeff["gradient"][0].T, 2 * dcoeff["tau"][0])
            )
            base[:5, active] = cartesian * values[5]
            delta[:5, active] = dcartesian * values[5] + cartesian * moving[5]
            base[5, active] = point["energy_density"]
            delta[5, active] = np.sum(potential * dwork, axis=0)
            energy += np.dot(values[5], point["energy_density"])

        pair_energy, pair_base, pair_delta, pairs = self._nonlocal_partials(
            fields, direction, self.budget_bytes - retained_bytes
        )
        return GridPartials(
            energy + pair_energy,
            base + pair_base,
            delta + pair_delta,
            self.identity,
            pairs,
            (base, delta),
            (pair_base, pair_delta),
        )

    def _nonlocal_partials(
        self, fields: np.ndarray, direction: np.ndarray, remaining: int
    ) -> tuple:
        """Apply the one total-density VV10 graph for either spin convention."""
        indices = np.flatnonzero(fields[0] >= float(MOLECULAR_VV10_DENSITY_THRESHOLD))
        count = len(indices)
        pairs = count * count
        if pairs > self.max_pair_evaluations:
            raise ValueError("VV10 second-order work exceeds max_nonlocal_pairs")
        base, delta = np.zeros_like(fields), np.zeros_like(fields)
        energy = 0.0
        primitive = self.nonlocal_primitive
        graph, roots = _pair_program(primitive.spec)
        node_count = len(graph.topological_order(roots))
        tile = min(64, int(math.sqrt(max(0, remaining) / (32 * node_count))))
        if count and tile < 1:
            raise MemoryError("VV10 AD pair tile exceeds grid_budget_bytes")
        beta = (3 / float(primitive.spec.b) ** 2) ** 0.75 / 32
        factor = float(primitive.coefficient)
        for start in range(0, count, max(1, tile)):
            row = indices[start : start + tile]
            for begin in range(0, count, tile):
                column = indices[begin : begin + tile]
                variables = {
                    **{
                        f"a{index}": fields[index, row][:, None]
                        for index in range(FIELD_COUNT)
                    },
                    **{
                        f"b{index}": fields[index, column][None, :]
                        for index in range(FIELD_COUNT)
                    },
                    **{
                        f"da{index}": direction[index, row][:, None]
                        for index in range(FIELD_COUNT)
                    },
                    **{
                        f"db{index}": direction[index, column][None, :]
                        for index in range(FIELD_COUNT)
                    },
                }
                outputs = evaluate_array_graph(graph, roots, variables)
                energy += 0.5 * factor * np.sum(outputs[0])
                for index in range(FIELD_COUNT):
                    base[index, row] += factor * np.sum(
                        np.broadcast_to(outputs[1 + index], (len(row), len(column))),
                        axis=1,
                    )
                    delta[index, row] += factor * np.sum(
                        np.broadcast_to(
                            outputs[1 + FIELD_COUNT + index], (len(row), len(column))
                        ),
                        axis=1,
                    )
        base[0, indices] += factor * beta * fields[5, indices]
        base[5, indices] += factor * beta * fields[0, indices]
        delta[0, indices] += factor * beta * direction[5, indices]
        delta[5, indices] += factor * beta * direction[0, indices]
        energy += factor * beta * np.dot(fields[0, indices], fields[5, indices])
        if (
            not np.isfinite(energy)
            or not np.isfinite(base).all()
            or not np.isfinite(delta).all()
        ):
            raise FloatingPointError("nonfinite compiler-AD grid partials")
        return energy, base, delta, pairs

    def _polarized_partials(
        self, fields: np.ndarray, direction: np.ndarray
    ) -> GridPartials:
        """Keep cross-spin feature curvature and total-density VV10 coupling."""
        if fields.ndim != 2 or fields.shape != direction.shape or fields.shape[0] != 14:
            raise ValueError("polarized grid fields/directions require [14,point]")
        if not np.isfinite(fields).all() or not np.isfinite(direction).all():
            raise ValueError("grid fields/directions must be finite")
        if np.any(fields[[0, 4, 5, 9]] < 0):
            raise ValueError("grid rho/tau must be nonnegative")
        remaining = self.budget_bytes - fields.nbytes * GRID_FIELD_WORKSPACE_PANELS
        if remaining < 1:
            raise MemoryError("retained grid partials exceed grid_budget_bytes")
        base, delta = np.zeros_like(fields), np.zeros_like(fields)
        active_indices = np.flatnonzero(fields[0] + fields[5] >= DENSITY_THRESHOLD)
        nodes = len(self.program.graph.topological_order(self.program.roots))
        tile = min(256, remaining // (32 * nodes))
        if len(active_indices) and tile < 1:
            raise MemoryError("semilocal AD point tile exceeds grid_budget_bytes")
        energy = 0.0
        for begin in range(0, len(active_indices), max(1, tile)):
            active = active_indices[begin : begin + tile]
            values, moving = fields[:, active], direction[:, active]
            spin_values = values[:10].reshape(2, 5, -1)
            spin_moving = moving[:10].reshape(2, 5, -1)
            rho, tau = spin_values[:, 0], spin_values[:, 4]
            drho, dtau = spin_moving[:, 0], spin_moving[:, 4]
            gradient = spin_values[:, 1:4].transpose(0, 2, 1)
            dgradient = spin_moving[:, 1:4].transpose(0, 2, 1)
            sigma = np.stack(
                (
                    np.sum(gradient[0] ** 2, axis=1),
                    np.sum(gradient[0] * gradient[1], axis=1),
                    np.sum(gradient[1] ** 2, axis=1),
                )
            )
            dsigma = np.stack(
                (
                    2 * np.sum(gradient[0] * dgradient[0], axis=1),
                    np.sum(
                        gradient[0] * dgradient[1] + dgradient[0] * gradient[1], axis=1
                    ),
                    2 * np.sum(gradient[1] * dgradient[1], axis=1),
                )
            )
            work = np.vstack((rho, sigma, tau))
            dwork = np.vstack((drho, dsigma, dtau))
            floors = np.array(
                (
                    DENSITY_THRESHOLD,
                    DENSITY_THRESHOLD,
                    SIGMA_THRESHOLD**2,
                    -np.inf,
                    SIGMA_THRESHOLD**2,
                    TAU_THRESHOLD,
                    TAU_THRESHOLD,
                )
            )[:, None]
            dwork *= work > floors
            work = np.maximum(work, floors)
            bound = (work[2] + work[4]) / 2
            work[3] = np.clip(work[3], -bound, bound)
            rows = np.asarray(
                evaluate_array_graph(
                    self.program.graph,
                    self.program.roots,
                    dict(zip(self.functional.features, work, strict=True)),
                )
            )
            point = self.program.unpack(rows)
            potential = point["gradient"]
            dpotential = np.einsum("ijp,jp->ip", point["hessian"], dwork)
            coeff = coefficient_program("polarized", "mgga").evaluate(
                gradient, potential
            )
            dcoeff = coefficient_program("polarized", "mgga", response=True).evaluate(
                gradient,
                potential,
                delta_gradient=dgradient,
                delta_v=dpotential,
            )
            cartesian, dcartesian = (
                np.concatenate(
                    (
                        value["rho"][:, None],
                        value["gradient"].transpose(0, 2, 1),
                        2 * value["tau"][:, None],
                    ),
                    axis=1,
                ).reshape(10, -1)
                for value in (coeff, dcoeff)
            )
            base[:10, active] = cartesian * values[10]
            delta[:10, active] = dcartesian * values[10] + cartesian * moving[10]
            base[10, active] = point["energy_density"]
            delta[10, active] = np.sum(potential * dwork, axis=0)
            energy += np.dot(values[10], point["energy_density"])
        total = np.vstack((fields[:5] + fields[5:10], fields[10:]))
        moving_total = np.vstack((direction[:5] + direction[5:10], direction[10:]))
        pair_energy, pair_base, pair_delta, pairs = self._nonlocal_partials(
            total, moving_total, remaining
        )
        lifted_base = np.vstack((pair_base[:5], pair_base[:5], pair_base[5:]))
        lifted_delta = np.vstack((pair_delta[:5], pair_delta[:5], pair_delta[5:]))
        return GridPartials(
            energy + pair_energy,
            base + lifted_base,
            delta + lifted_delta,
            self.identity,
            pairs,
            (base, delta),
            (lifted_base, lifted_delta),
        )
