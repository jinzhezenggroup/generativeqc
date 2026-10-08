"""Test-only public-AO alignment for independent MD-J/PBE0 qualification."""

from typing import Any

import numpy as np


def screened_product_count(bounds: Any, weights: Any, cutoff: float) -> int:
    """Count weighted accepted products in O(n log n) work and O(n) storage.

    Binary search tests the actual FP64 product rather than dividing the
    cutoff: division can change acceptance at an exactly equal boundary.
    This is semantic work counting only, never a production screening path.
    """
    values = np.asarray(bounds, dtype=np.float64)
    counts = np.asarray(weights, dtype=np.int64)
    if (
        values.ndim != 1
        or counts.shape != values.shape
        or not np.isfinite(values).all()
        or (values < 0).any()
        or (counts < 0).any()
        or not np.isfinite(cutoff)
        or cutoff < 0
    ):
        raise ValueError("invalid screened-product inventory")
    if sum(map(int, counts)) > 3_037_000_499:
        raise OverflowError("screened-product inventory exceeds int64 capacity")
    if not values.size:
        return 0
    order = np.argsort(values)
    sorted_values = values[order]
    suffix = np.append(np.cumsum(counts[order][::-1], dtype=np.int64)[::-1], 0)
    lower = np.zeros(values.size, dtype=np.int64)
    upper = np.full(values.size, values.size, dtype=np.int64)
    while np.any(lower < upper):
        active = lower < upper
        middle = (lower + upper) // 2
        accepted = sorted_values[np.minimum(middle, values.size - 1)] * values >= cutoff
        upper = np.where(active & accepted, middle, upper)
        lower = np.where(active & ~accepted, middle + 1, lower)
    return int(counts @ suffix[lower])


def schwarz_bounds(molecule: Any, scales: Any) -> np.ndarray:
    """Independent AO Schwarz bounds with only a single shell-quartet workspace."""
    offsets = molecule.ao_loc_nr()
    result = np.empty((molecule.nao_nr(), molecule.nao_nr()))
    for first in range(molecule.nbas):
        rows = slice(offsets[first], offsets[first + 1])
        for second in range(first + 1):
            columns = slice(offsets[second], offsets[second + 1])
            block = molecule.intor_by_shell("int2e", (first, second, first, second))
            width = (offsets[first + 1] - offsets[first]) * (
                offsets[second + 1] - offsets[second]
            )
            bound = np.sqrt(np.abs(block.reshape(width, width).diagonal())).reshape(
                offsets[first + 1] - offsets[first],
                offsets[second + 1] - offsets[second],
            )
            bound *= np.abs(scales[rows, None] * scales[None, columns])
            result[rows, columns] = bound
            result[columns, rows] = bound.T
    return result


def unordered_screened_product_count(bounds: Any, weights: Any, cutoff: float) -> int:
    """Unique shell-quartet source work, retaining full diagonal primitive products."""
    ordered = screened_product_count(bounds, weights, cutoff)
    diagonal = sum(
        int(weight) ** 2
        for bound, weight in zip(bounds, weights, strict=True)
        if float(bound) * float(bound) >= cutoff
    )
    return (ordered + diagonal) // 2


def reference_molecule(basis: Any) -> tuple[Any, np.ndarray]:
    """Preserve the actual input contraction phases and libcint normalization.

    A named PySCF basis need not choose the same contraction phases as the
    repository's canonical record. Reuse the existing explicit-shell converter;
    this is reference preparation only, never a production J or XC consumer.
    """
    from tools.generate_validation_references import pyscf_molecule

    inputs = {
        "atomic_numbers": [atom.atomic_number for atom in basis.atoms],
        "coordinates": [atom.position for atom in basis.atoms],
        "charge": basis.charge,
        "multiplicity": basis.multiplicity,
        "basis_representation": "spherical"
        if basis.representation == "real_spherical"
        else "cartesian",
        "shells": [
            {
                "atom_index": shell.atom_index,
                "angular_momentum": shell.angular_momentum,
                "primitives": [
                    [primitive.exponent, primitive.coefficient]
                    for primitive in shell.primitives
                ],
            }
            for shell in basis.shells
        ],
    }
    molecule, scale, _ = pyscf_molecule(inputs)
    return molecule, scale
