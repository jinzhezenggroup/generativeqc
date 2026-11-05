"""Exact real Coulomb-family permutation schedules and external-weight folding.

The same eight permutations apply to AO indices and shell indices. Distinct
shell orientations must be folded with their full component arrays: diagonal
shells do not justify multiplying an arbitrary ordered cotangent by eight.
"""

from __future__ import annotations

import typing

import numpy as np

_PERMUTATIONS = (
    (0, 1, 2, 3),
    (1, 0, 2, 3),
    (0, 1, 3, 2),
    (1, 0, 3, 2),
    (2, 3, 0, 1),
    (3, 2, 0, 1),
    (2, 3, 1, 0),
    (3, 2, 1, 0),
)


def canonical_eri_quartets(size: int) -> typing.Iterator[tuple[int, int, int, int]]:
    """Enumerate the unscreened eightfold domain once, including diagonals."""
    pairs = tuple(
        (first, second) for first in range(size) for second in range(first + 1)
    )
    for first_pair, left in enumerate(pairs):
        for second_pair in range(first_pair + 1):
            yield (*left, *pairs[second_pair])


def eri_quartet_orbits(indices: typing.Any) -> tuple[np.ndarray, np.ndarray]:
    """Bind eight AO permutations and undo duplicate diagonal multiplicities.

    Consumers still own every ordered density weight. Multiplicities correct
    duplicate AO tuples, not weights on distinct shell component arrays.
    """
    canonical = np.asarray(indices, dtype=np.int64)
    orbits = canonical[:, np.asarray(_PERMUTATIONS)]
    multiplicities = (
        np.where(canonical[:, 0] == canonical[:, 1], 1, 2)
        * np.where(canonical[:, 2] == canonical[:, 3], 1, 2)
        * np.where(np.all(canonical[:, :2] == canonical[:, 2:], axis=1), 1, 2)
        / 8
    )
    return orbits.reshape(-1, 4), multiplicities


def distinct_eri_shell_permutations(
    indices: tuple[int, int, int, int],
) -> tuple[tuple[tuple[int, ...], tuple[int, ...]], ...]:
    """Return each ordered shell tuple once and its canonical-axis permutation.

    A repeated shell still contains all its Cartesian components. Keeping one
    representative permutation avoids counting that same ordered block twice.
    """
    seen = set()
    result = []
    for permutation in _PERMUTATIONS:
        ordered = tuple(indices[axis] for axis in permutation)
        if ordered not in seen:
            seen.add(ordered)
            result.append((ordered, permutation))
    return tuple(result)


def fold_eri_shell_weights(
    indices: tuple[int, int, int, int],
    extents: tuple[int, ...],
    weights: typing.Callable[[tuple[int, ...]], typing.Any],
) -> np.ndarray:
    """Sum distinct ordered cotangents in canonical Cartesian component order.

    Every callback result is validated before use, including translation-zero
    quartets. At most the accumulator and one ordered shell block are live;
    the caller must admit that lease or retain its ordered bounded schedule.
    """
    result = np.zeros(extents)
    for ordered, permutation in distinct_eri_shell_permutations(indices):
        values = np.asarray(weights(ordered))
        if (
            values.shape != tuple(extents[axis] for axis in permutation)
            or values.dtype.kind not in "iuf"
            or not np.isfinite(values).all()
        ):
            raise ValueError(
                "four-center shell weights must be finite real with the ordered "
                "Cartesian shell shape"
            )
        inverse = tuple(permutation.index(axis) for axis in range(4))
        with np.errstate(over="raise", invalid="raise"):
            np.add(result, values.transpose(inverse), out=result)
        del values
    return result
