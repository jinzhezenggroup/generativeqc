"""Whole-objective recognition and canonical JVP versus independent force gates."""

from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING

import numpy as np
import pytest
import test_becke_cooperative as retained
from generativeqc_compiler.integral.expr import Graph
from generativeqc_compiler.xc.becke_coefficients import (
    emit_becke_pair_coefficients,
    plan_becke_pair_coefficients,
)
from generativeqc_compiler.xc.becke_partition import (
    recognize_becke_partition_graph,
)
from generativeqc_compiler.xc.grid_partition_ir import (
    grid_partition_domain_program,
    grid_partition_program,
)
from test_becke_pair_coefficients import helper

__all__ = ["helper"]

if TYPE_CHECKING:
    import ctypes as ct


@pytest.mark.parametrize("atoms", [1, 2, 3, 8])
@pytest.mark.parametrize("iterations", [1, 3, 5])
def test_whole_graph_recognition_emission_and_domain_admission(
    atoms: int, iterations: int
) -> None:
    program = grid_partition_program(atoms, iterations)
    operation = recognize_becke_partition_graph(
        program, atoms=atoms, iterations=iterations
    )
    assert operation is not None
    assert operation.composition_identity == program.identity
    assert operation.atoms == atoms
    assert operation.identity in emit_becke_pair_coefficients(operation)
    admitted = plan_becke_pair_coefficients(
        operation=operation,
        atoms=atoms,
        points=17,
        cached_geometry=True,
        budget_bytes=1 << 30,
    )
    assert admitted is not None
    assert (
        plan_becke_pair_coefficients(
            operation=operation,
            atoms=atoms + 1,
            points=17,
            cached_geometry=True,
            budget_bytes=1 << 30,
        )
        is None
    )
    assert (
        recognize_becke_partition_graph(
            program,
            atoms=atoms,
            iterations=iterations,
            partition="heteronuclear-adjusted",
        )
        is None
    )


def test_recognition_authenticates_composition_not_companion_hashes() -> None:
    program = grid_partition_program(3)
    expected = recognize_becke_partition_graph(program, atoms=3)
    assert expected is not None
    assert (
        recognize_becke_partition_graph(
            replace(program, identity="untrusted metadata"), atoms=3
        )
        == expected
    )
    mutations = [
        (program.roots[0] * 0.5, program.roots[1]),
        (program.roots[0], program.graph.constant(0)),
        (program.roots[1], program.roots[0]),
        (Graph().variable("weight"), program.roots[1]),
    ]
    for roots in mutations:
        assert (
            recognize_becke_partition_graph(replace(program, roots=roots), atoms=3)
            is None
        )
    assert recognize_becke_partition_graph(program, atoms=2) is None
    assert recognize_becke_partition_graph(program, atoms=3, iterations=1) is None
    assert recognize_becke_partition_graph(program, atoms=129) is None
    assert recognize_becke_partition_graph(program, atoms=True) is None


@pytest.mark.parametrize("atoms", [1, 2, 3, 8])
@pytest.mark.parametrize("dynamic_domain", [False, True])
def test_whole_generated_jvp_matches_generic_and_coefficient_force(
    helper: ct.CDLL, atoms: int, dynamic_domain: bool
) -> None:
    """Bind existing norm tangents; the entire product/ratio JVP comes from AD.

    Owner-attached point motion and every center-distance tangent participate.
    The generic compiled route remains separate from the coefficient route;
    neither reference is inferred from metadata or a recognition flag.
    """
    rng = np.random.default_rng(1894300 + atoms)
    centers = rng.normal(size=(atoms, 3)) * 2
    points = rng.normal(size=(17, 3)) * 3
    owners = rng.integers(atoms, size=len(points), dtype=np.int64)
    seeds = rng.normal(size=len(points))
    motion = rng.normal(size=centers.shape)
    bindings = {"owner": owners}
    for atom in range(atoms):
        delta = points - centers[atom]
        radius = np.linalg.norm(delta, axis=1)
        bindings[f"distance_{atom}"] = radius
        bindings[f"ddistance_{atom}"] = np.einsum(
            "pc,pc->p", delta / radius[:, None], motion[owners] - motion[atom]
        )
        for neighbor in range(atom):
            separation = centers[atom] - centers[neighbor]
            length = np.linalg.norm(separation)
            bindings[f"separation_{atom}_{neighbor}"] = length
            bindings[f"dseparation_{atom}_{neighbor}"] = np.dot(
                separation / length, motion[atom] - motion[neighbor]
            )
    if dynamic_domain:
        bindings["atoms"] = atoms
        for atom in range(atoms, 8):
            bindings[f"distance_{atom}"] = 0
            bindings[f"ddistance_{atom}"] = 17
            for neighbor in range(atom):
                bindings[f"separation_{atom}_{neighbor}"] = 0
                bindings[f"dseparation_{atom}_{neighbor}"] = 23
        program = grid_partition_domain_program(8, helper.iterations)
    else:
        program = grid_partition_program(atoms, helper.iterations)
    _, tangent = program.evaluate(**bindings)
    expected = np.sum(seeds * tangent)
    for route in (0, 2):
        status, gradient, _ = retained.run(
            helper, points, centers, owners, seeds, True, route
        )
        assert status == 0
        actual = np.einsum("ac,ac->", gradient, motion)
        np.testing.assert_allclose(actual, expected, atol=3e-11, rtol=3e-10)


def test_invalid_composition_dimensions_are_bounded() -> None:
    for atoms in (0, 129, True, 1.5):
        with pytest.raises(ValueError):
            grid_partition_program(atoms)
    with pytest.raises(ValueError):
        grid_partition_program(3, True)
