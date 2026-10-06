"""An authenticated active-prefix graph, not fixed-N extrapolation."""

from dataclasses import replace
from decimal import Decimal, localcontext

import numpy as np
import pytest
from generativeqc_compiler.integral.expr import Graph
from generativeqc_compiler.xc.becke_coefficients import (
    emit_becke_pair_coefficients,
    plan_becke_pair_coefficients,
)
from generativeqc_compiler.xc.becke_partition import (
    plan_becke_partition_derivative,
    recognize_becke_partition_domain_graph,
    validate_becke_partition_derivative,
)
from generativeqc_compiler.xc.grid_partition_ir import (
    grid_partition_domain_program,
    grid_partition_program,
)
from test_grid_response import decimal_partition


@pytest.mark.parametrize("atom_limit", [1, 2, 7])
@pytest.mark.parametrize("iterations", [1, 3, 5])
def test_dynamic_graph_binds_every_admitted_atom_count(
    atom_limit: int, iterations: int
) -> None:
    program = grid_partition_domain_program(atom_limit, iterations)
    operation = recognize_becke_partition_domain_graph(
        program, atom_limit=atom_limit, iterations=iterations
    )
    assert operation is not None
    assert operation.atoms is None
    assert operation.atom_limit == atom_limit
    assert operation.composition_identity == program.identity
    assert operation.identity in emit_becke_pair_coefficients(operation)
    for atoms in range(1, atom_limit + 2):
        assert operation.supports_atoms(atoms) == (atoms <= atom_limit)
        common = {
            "operation": operation,
            "atoms": atoms,
            "points": 17,
            "budget_bytes": 1 << 30,
        }
        for admitted in (
            plan_becke_pair_coefficients(**common, cached_geometry=True),
            plan_becke_partition_derivative(**common),
        ):
            assert (admitted is not None) == (atoms <= atom_limit)


@pytest.mark.parametrize("iterations", [1, 3, 5])
def test_padded_and_mixed_prefix_jvps_equal_actual_canonical_graphs(
    iterations: int,
) -> None:
    """Inactive zero separations and arbitrary tangents must remain unobserved.

    A single vectorized evaluation includes different active counts and owner
    selections. Exact-N graphs independently define each sample's objective;
    neither the bound nor a successful recognizer supplies its numeric answer.
    """
    atom_limit = 7
    counts = np.repeat(np.arange(1, atom_limit + 1), 3)
    rng = np.random.default_rng(1894128 + iterations)
    owners = np.array([rng.integers(atoms) for atoms in counts])
    centers = rng.normal(size=(atom_limit, 3))
    points = rng.normal(size=(len(counts), 3))
    bindings = {"atoms": counts, "owner": owners}
    for atom in range(atom_limit):
        bindings[f"distance_{atom}"] = np.where(
            counts > atom, np.linalg.norm(points - centers[atom], axis=1), 0
        )
        bindings[f"ddistance_{atom}"] = rng.normal(size=len(counts))
        for neighbor in range(atom):
            bindings[f"separation_{atom}_{neighbor}"] = np.where(
                counts > atom, np.linalg.norm(centers[atom] - centers[neighbor]), 0
            )
            bindings[f"dseparation_{atom}_{neighbor}"] = rng.normal(size=len(counts))
    with np.errstate(divide="raise", invalid="raise"):
        actual = grid_partition_domain_program(atom_limit, iterations).evaluate(
            **bindings
        )
    for sample, atoms in enumerate(counts):
        expected = grid_partition_program(int(atoms), iterations).evaluate(
            **{name: values[sample] for name, values in bindings.items()}
        )
        np.testing.assert_allclose(
            [root[sample] for root in actual], expected, atol=3e-14, rtol=3e-13
        )


def test_dynamic_recognition_rejects_fixed_witnesses_and_changed_roots() -> None:
    program = grid_partition_domain_program(4)
    operation = recognize_becke_partition_domain_graph(program, atom_limit=4)
    assert operation is not None
    assert (
        recognize_becke_partition_domain_graph(
            replace(program, identity="untrusted metadata"), atom_limit=4
        )
        == operation
    )
    for roots in (
        (program.roots[0] * 0.5, program.roots[1]),
        (program.roots[0], program.graph.constant(0)),
        (program.roots[1], program.roots[0]),
        (Graph().variable("weight"), program.roots[1]),
    ):
        assert (
            recognize_becke_partition_domain_graph(
                replace(program, roots=roots), atom_limit=4
            )
            is None
        )
    assert (
        recognize_becke_partition_domain_graph(grid_partition_program(4), atom_limit=4)
        is None
    )
    assert recognize_becke_partition_domain_graph(program, atom_limit=3) is None
    assert (
        recognize_becke_partition_domain_graph(program, atom_limit=4, iterations=1)
        is None
    )
    assert (
        recognize_becke_partition_domain_graph(
            program, atom_limit=4, partition="heteronuclear-adjusted"
        )
        is None
    )
    for changed in (
        replace(operation, atom_limit=5),
        replace(operation, atoms=4),
        replace(operation, composition_identity=None),
    ):
        with pytest.raises(ValueError):
            validate_becke_partition_derivative(changed)


@pytest.mark.parametrize("atom_limit", [0, 129, True, 1.5])
def test_dynamic_dimensions_remain_bounded(atom_limit: int) -> None:
    with pytest.raises(ValueError):
        grid_partition_domain_program(atom_limit)
    assert (
        recognize_becke_partition_domain_graph(
            grid_partition_domain_program(3), atom_limit=atom_limit
        )
        is None
    )


@pytest.mark.parametrize("atoms", [48, 96, 128])
def test_default_native_bound_matches_independent_large_geometry_difference(
    atoms: int,
) -> None:
    """Exercise the actual bound-128 witness at endpoint sizes and its cap.

    Centers on a sphere keep every product nonzero and avoid an uninformative
    saturated derivative. The selected point follows its owner. A 60-digit
    direct-product Decimal oracle is independent of both Graph and log/zero
    execution. Zero padding and arbitrary padding tangents remain explicit.
    """
    atom_limit, iterations = 128, 3
    rng = np.random.default_rng(18944896 + atoms)
    centers = rng.normal(size=(atoms, 3))
    centers /= np.linalg.norm(centers, axis=1)[:, None]
    motion = rng.normal(size=centers.shape)
    owner = atoms // 3
    bindings = {"atoms": atoms, "owner": owner}
    for atom in range(atom_limit):
        bindings[f"distance_{atom}"] = 0
        bindings[f"ddistance_{atom}"] = 17
        if atom < atoms:
            radius = np.linalg.norm(centers[atom])
            bindings[f"distance_{atom}"] = radius
            bindings[f"ddistance_{atom}"] = np.dot(
                -centers[atom] / radius, motion[owner] - motion[atom]
            )
        for neighbor in range(atom):
            name = f"separation_{atom}_{neighbor}"
            bindings[name] = 0
            bindings[f"d{name}"] = 23
            if atom < atoms:
                separation = centers[atom] - centers[neighbor]
                length = np.linalg.norm(separation)
                bindings[name] = length
                bindings[f"d{name}"] = np.dot(
                    separation / length, motion[atom] - motion[neighbor]
                )
    with np.errstate(divide="raise", invalid="raise"):
        primal, tangent = grid_partition_domain_program(
            atom_limit, iterations
        ).evaluate(**bindings)
    with localcontext() as context:
        context.prec = 60
        step = Decimal("1e-16")
        decimal_centers = [[Decimal(str(value)) for value in row] for row in centers]
        decimal_motion = [[Decimal(str(value)) for value in row] for row in motion]
        weights = []
        for sign in (-1, 0, 1):
            moved_centers = [
                [
                    value + sign * step * direction
                    for value, direction in zip(row, directions, strict=True)
                ]
                for row, directions in zip(decimal_centers, decimal_motion, strict=True)
            ]
            point = [[sign * step * direction for direction in decimal_motion[owner]]]
            weights.append(
                decimal_partition(point, moved_centers, iterations)[0][owner]
            )
        expected = float((weights[2] - weights[0]) / (2 * step))
    np.testing.assert_allclose(primal, float(weights[1]), atol=2e-14, rtol=2e-12)
    np.testing.assert_allclose(tangent, expected, atol=2e-12, rtol=2e-11)
