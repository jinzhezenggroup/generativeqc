"""Independent test inputs and public-AO oracle for MD-J engine qualification.

Nothing in this module is imported by production. Full ERI tensors are bounded
by the deliberately small fixtures; acceptance timing excludes their creation.
"""

from __future__ import annotations

import math
from itertools import product
from typing import Any

import numpy as np

WATER = [
    ("O", (0.0, 0.0, 0.0)),
    ("H", (0.0, -1.43233673, 1.10715266)),
    ("H", (0.0, 1.43233673, 1.10715266)),
]


def general_contraction_basis() -> tuple[list, tuple]:
    """Expand two columns over shared exponents, with signed contractions.

    The public Shell API represents general contractions as separate shells.
    Shared primitive exponents must not collapse distinct contraction columns.
    """
    from generativeqc import Primitive, Shell

    atoms = [("He", (0.0, 0.0, 0.0)), ("He", (0.3, -0.2, 1.6))]
    shells = tuple(
        Shell(
            atom,
            angular,
            tuple(Primitive(a, c) for a, c in zip(exponents, column, strict=True)),
        )
        for atom in range(2)
        for angular in range(3)
        for exponents, column in (
            ((1.7, 0.43), (0.71, -0.22)),
            ((1.7, 0.43), (-0.16, 0.83)),
        )
    )
    return atoms, shells


def public_ao_oracle(basis: Any) -> tuple[Any, np.ndarray, np.ndarray, np.ndarray]:
    """Use explicit native input shells, preserving phases and AO normalization."""
    from benchmarks._md_j_reference import reference_molecule

    molecule, scales = reference_molecule(basis)
    eri = molecule.intor("int2e") * np.einsum(
        "i,j,k,l->ijkl", scales, scales, scales, scales
    )
    count = len(scales)
    bounds = np.sqrt(
        np.abs(eri.reshape(count * count, count * count).diagonal())
    ).reshape(count, count)
    return molecule, scales, eri, bounds


def screened_j(
    eri: np.ndarray, bounds: np.ndarray, density: np.ndarray, cutoff: float
) -> np.ndarray:
    """Apply the exact public-AO comparison, including equality at the boundary."""
    total = density.sum(axis=0) if density.ndim == 3 else density
    mask = bounds[:, :, None, None] * bounds[None, None, :, :] >= cutoff
    return np.einsum("ijkl,kl->ij", np.where(mask, eri, 0.0), total, optimize=True)


def work_rows(stderr: str) -> list[dict[str, int]]:
    """Parse execution counters without substituting capacity for actual work."""
    return [
        {
            key: int(value)
            for key, value in (field.split("=") for field in line.split()[1:])
        }
        for line in stderr.splitlines()
        if line.startswith("MD_J_WORK ")
    ]


def cartesian_components(angular: int) -> list[tuple[int, int, int]]:
    """Libcint Cartesian ordering: xx,xy,xz,yy,yz,zz for d."""
    return [
        (x, y, angular - x - y)
        for x in range(angular, -1, -1)
        for y in range(angular - x, -1, -1)
    ]


def _odd_factorial(order: int) -> int:
    return math.prod(range(1, 2 * order, 2))


def primitive_normalization(exponent: float, powers: tuple[int, int, int]) -> float:
    """Unit Cartesian primitive norm from the Gaussian integral, not libcint."""
    return (2 * exponent / math.pi) ** 0.75 * math.sqrt(
        (4 * exponent) ** sum(powers) / math.prod(_odd_factorial(n) for n in powers)
    )


def contraction_weights(
    primitives: tuple[tuple[float, float], ...], powers: tuple[int, int, int]
) -> np.ndarray:
    """Normalize a signed contraction by its analytic same-center overlap."""
    weights = np.asarray(
        [
            coefficient * primitive_normalization(exponent, powers)
            for exponent, coefficient in primitives
        ]
    )
    overlap = sum(
        weights[i]
        * weights[j]
        * math.prod(
            _odd_factorial(order)
            * math.sqrt(math.pi)
            / (2**order * (a + b) ** (order + 0.5))
            for order in powers
        )
        for i, (a, _) in enumerate(primitives)
        for j, (b, _) in enumerate(primitives)
    )
    if not np.isfinite(overlap) or overlap <= 0:
        raise ValueError("singular contraction")
    return weights / math.sqrt(overlap)


def contracted_quartet(
    primitive_eri: Any,
    angular: tuple[int, int, int, int],
    centers: np.ndarray,
    primitives: tuple,
) -> np.ndarray:
    """Compose compiled E/R primitives, preserving every contraction column."""
    components = [cartesian_components(value) for value in angular]
    output = np.zeros(tuple(map(len, components)))
    centers = np.ascontiguousarray(centers, dtype=np.float64)
    for indices in product(*(range(len(values)) for values in components)):
        powers = tuple(components[shell][indices[shell]] for shell in range(4))
        weights = [
            contraction_weights(primitives[shell], powers[shell]) for shell in range(4)
        ]
        angular_data = np.ascontiguousarray(powers, dtype=np.uint32)
        value = 0.0
        for chosen in product(*(range(len(values)) for values in primitives)):
            exponents = np.asarray(
                [primitives[shell][chosen[shell]][0] for shell in range(4)],
                dtype=np.float64,
            )
            coefficient = math.prod(weights[shell][chosen[shell]] for shell in range(4))
            value += coefficient * primitive_eri(exponents, centers, angular_data)
        output[indices] = value
    return output


def packed_pipeline_fixture(
    representation: str, general: bool
) -> tuple[Any, np.ndarray, list[np.ndarray]]:
    """Independent normalized public s/p/d inputs for production kernel bodies.

    The spherical table is the explicit normalized real-solid-harmonic basis,
    independent of the production packing code and the libcint result tensor.
    Sharing primitive exponents across columns models the public Shell API's
    expanded general contractions without borrowing a native AO oracle.
    """
    from pyscf import gto

    positions = np.asarray([(0.0, 0.0, 0.0), (0.3, -0.4, 1.8)], dtype=np.float64)
    shell_atoms, shell_angular = [], []
    primitive_offsets, ao_offsets = [0], [0]
    exponents, coefficients, counts, angular_data, term_coefficients = (
        [],
        [],
        [],
        [],
        [],
    )
    reference_basis: dict[str, list] = {"He0": [], "He1": []}
    for atom in range(2):
        for order in range(3):
            columns = (
                [(0.71, -0.22), (-0.16, 0.83)]
                if general and order == 0
                else [(0.71, -0.22)]
            )
            for column in columns:
                primitives = tuple(
                    zip((1.7 + order * 0.3, 0.43 + order * 0.1), column, strict=True)
                )
                reference_basis[f"He{atom}"].append([order, *primitives])
                shell_atoms.append(atom)
                shell_angular.append(order)
                base_powers = (order, 0, 0)
                weights = contraction_weights(primitives, base_powers)
                exponents.extend(value[0] for value in primitives)
                coefficients.extend(weights)
                primitive_offsets.append(len(exponents))
                components = [[(powers, 1.0)] for powers in cartesian_components(order)]
                if representation == "spherical" and order == 2:
                    components = [
                        [((1, 1, 0), 1.0)],
                        [((0, 1, 1), 1.0)],
                        [((0, 0, 2), 1.0), ((2, 0, 0), -0.5), ((0, 2, 0), -0.5)],
                        [((1, 0, 1), 1.0)],
                        [((2, 0, 0), math.sqrt(3) / 2), ((0, 2, 0), -math.sqrt(3) / 2)],
                    ]
                for terms in components:
                    counts.append(len(terms))
                    for term in range(3):
                        powers, weight = (
                            terms[term] if term < len(terms) else ((0, 0, 0), 0.0)
                        )
                        angular_data.extend(powers)
                        term_coefficients.append(
                            weight
                            * primitive_normalization(1.0, powers)
                            / primitive_normalization(1.0, base_powers)
                        )
                ao_offsets.append(len(counts))
    molecule = gto.M(
        atom=[(f"He{i}", position) for i, position in enumerate(positions)],
        basis=reference_basis,
        unit="Bohr",
        cart=representation == "cartesian",
        verbose=0,
    )
    assert molecule.nao_nr() == len(counts)
    assert molecule.ao_loc_nr().tolist() == ao_offsets
    scale = 1 / np.sqrt(np.diag(molecule.intor("int1e_ovlp")))
    arrays = [
        positions,
        np.asarray(shell_atoms, dtype=np.int32),
        np.asarray(shell_angular, dtype=np.uint8),
        np.asarray(primitive_offsets, dtype=np.int64),
        np.asarray(ao_offsets, dtype=np.int64),
        np.asarray(exponents, dtype=np.float64),
        np.asarray(coefficients, dtype=np.float64),
        np.asarray(counts, dtype=np.uint8),
        np.asarray(angular_data, dtype=np.uint8),
        np.asarray(term_coefficients, dtype=np.float64),
    ]
    return molecule, scale, arrays
