"""Independent screened-Coulomb force gates for actual weighted native tasks."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest
from test_direct_scalar_separable_host import _inputs, _run, native_pair_owner
from test_weighted_eri_separable import CLASSES, _libcint

LR_CLASSES = (*CLASSES, (2, 2, 0, 0))
__all__ = ["native_pair_owner"]


def _reference(
    inputs: dict[str, Any],
    angular: tuple[int, int, int, int],
    omega: float,
    exchange: float,
    slots: tuple[int, ...],
    threshold: float,
) -> np.ndarray:
    """Sum unique eightfold ERI orbits without using native force coefficients."""
    _, gradient = _libcint(angular, "distinct", slots, omega=omega)
    shape = tuple((degree + 1) * (degree + 2) // 2 for degree in angular)
    offsets = np.cumsum((0, *shape))
    dimension = int(offsets[-1])
    matrix = dimension**2
    spins = 2 if inputs["unrestricted"] else 1
    density = inputs["density"].reshape(-1, spins, matrix)[inputs["system"]]
    coefficients = inputs["coefficients"].reshape(-1, dimension)[inputs["system"]]
    bounds = inputs["bounds"].reshape(-1, matrix)[inputs["system"]]
    expected = np.linspace(0.01, 0.24, 24)
    seen = set()
    for index, component in enumerate(np.ndindex(*shape)):
        first, second, third, fourth = (
            int(offsets[slot]) + local
            for slot, local in zip(slots, component, strict=True)
        )
        if (
            bounds[first + dimension * second] * bounds[third + dimension * fourth]
            < threshold
        ):
            continue
        orbit = frozenset(
            {
                (first, second, third, fourth),
                (second, first, third, fourth),
                (first, second, fourth, third),
                (second, first, fourth, third),
                (third, fourth, first, second),
                (fourth, third, first, second),
                (third, fourth, second, first),
                (fourth, third, second, first),
            }
        )
        if orbit in seen:
            continue
        seen.add(orbit)
        weight = sum(
            0.5
            * exchange
            * sum(
                channel[left + dimension * ket_left]
                * channel[right + dimension * ket_right]
                for channel in density
            )
            for left, right, ket_left, ket_right in orbit
        ) * np.prod(coefficients[[first, second, third, fourth]])
        for center, shell in enumerate(slots):
            atom = inputs["atoms"][shell]
            expected[3 * atom : 3 * atom + 3] -= (
                weight * gradient[3 * center : 3 * center + 3, index]
            )
    return expected


@pytest.mark.parametrize("angular", LR_CLASSES)
@pytest.mark.parametrize("unrestricted", (False, True))
@pytest.mark.parametrize("omega", (0.03, 0.3, 1.7))
@pytest.mark.parametrize("exchange", (-0.63, 0.41))
def test_lr_weighted_native_matches_independent_libcint(
    native_pair_owner: Any,
    angular: tuple[int, int, int, int],
    unrestricted: bool,
    omega: float,
    exchange: float,
) -> None:
    """Qualify arbitrary exchange weights and relative/nuclear LR moment chains."""
    inputs = _inputs(
        unrestricted, system=1, angular=angular, class_index=LR_CLASSES.index(angular)
    )

    def execute(*args: Any) -> None:
        native_pair_owner.long_range(*args, omega)

    actual, counts = _run(
        execute, inputs, exchange=exchange, coulomb=0.0, orientation=7
    )
    nuclear, _ = _run(execute, inputs, exchange=exchange, coulomb=0.0, relative=False)
    reference = _reference(inputs, angular, omega, exchange, (0, 1, 2, 3), 1e-12)
    np.testing.assert_allclose(actual, reference, rtol=5e-11, atol=5e-12)
    np.testing.assert_allclose(nuclear, reference, rtol=5e-11, atol=5e-12)
    np.testing.assert_array_equal(counts[:4], 0)
    assert counts[8] == 1


@pytest.mark.parametrize("angular", LR_CLASSES)
@pytest.mark.parametrize("fault", (1, 2))
def test_lr_weighted_nonfinite_retries_before_publication(
    native_pair_owner: Any,
    angular: tuple[int, int, int, int],
    fault: int,
) -> None:
    """A fault in the second primitive cannot leave the first one's force behind."""
    inputs = _inputs(True, angular=angular, class_index=LR_CLASSES.index(angular))

    def execute(*args: Any) -> None:
        native_pair_owner.long_range(*args, 0.3)

    actual, counts = _run(execute, inputs, coulomb=0.0, exchange=-0.72, fault=fault)
    expected, _ = _run(
        execute, inputs, coulomb=0.0, exchange=-0.72, fault=fault, relative=False
    )
    np.testing.assert_allclose(actual, expected, rtol=5e-11, atol=5e-12)
    assert counts[8] == 2 and counts[9] == 2


TOPOLOGIES = tuple(
    (angular, topology)
    for angular in LR_CLASSES
    for topology in ("distinct", "same_ket", "same_pair")
    if (topology != "same_ket" or angular[2] == angular[3])
    and (topology != "same_pair" or angular[:2] == angular[2:])
)


@pytest.mark.parametrize("angular,topology", TOPOLOGIES)
@pytest.mark.parametrize("orientation", range(8))
@pytest.mark.parametrize("unrestricted", (False, True))
def test_lr_weighted_triangles_and_orientations(
    native_pair_owner: Any,
    angular: tuple[int, int, int, int],
    topology: str,
    orientation: int,
    unrestricted: bool,
) -> None:
    """Unique Libcint orbits cover every physical repeated-pair signature."""
    inputs = _inputs(
        unrestricted, angular=angular, class_index=LR_CLASSES.index(angular)
    )
    slots = (
        (0, 1, 0, 1)
        if topology == "same_pair"
        else (0, 1, 2, 2)
        if topology == "same_ket"
        else (0, 1, 2, 3)
    )

    def execute(*args: Any) -> None:
        native_pair_owner.long_range(*args, 0.3)

    actual, _ = _run(
        execute,
        inputs,
        coulomb=0.0,
        exchange=-0.72,
        orientation=orientation,
        same_pair=topology == "same_pair",
        same_ket=topology == "same_ket",
    )
    expected = _reference(inputs, angular, 0.3, -0.72, slots, 1e-12)
    np.testing.assert_allclose(actual, expected, rtol=5e-11, atol=5e-12)


@pytest.mark.parametrize("angular", LR_CLASSES)
@pytest.mark.parametrize("mask", ("sparse", "boundary", "empty", "zero_exchange"))
def test_lr_weighted_preserves_exact_ao_mask(
    native_pair_owner: Any, angular: tuple[int, int, int, int], mask: str
) -> None:
    """Boundary equality stays admitted and zero sources publish no atomics."""
    inputs = _inputs(True, angular=angular, class_index=LR_CLASSES.index(angular))
    exchange = 0.0 if mask == "zero_exchange" else -0.72
    if mask == "sparse":
        dimension = int(np.sqrt(inputs["bounds"].size))
        inputs["bounds"].reshape(dimension, dimension)[::2, :] = 1e-20
    elif mask == "boundary":
        inputs["bounds"][:] = 0.5
    elif mask == "empty":
        inputs["bounds"][:] = 0.0
    threshold = 0.25 if mask == "boundary" else 1e-12

    def execute(*args: Any) -> None:
        native_pair_owner.long_range(*args, 0.3)

    actual, _ = _run(
        execute, inputs, coulomb=0.0, exchange=exchange, threshold=threshold
    )
    expected = _reference(inputs, angular, 0.3, exchange, (0, 1, 2, 3), threshold)
    np.testing.assert_allclose(actual, expected, rtol=5e-11, atol=5e-12)


@pytest.mark.parametrize("angular", LR_CLASSES)
@pytest.mark.parametrize("atoms", ((0, 0, 1, 2), (0, 1, 1, 2), (0, 0, 0, 0)))
def test_lr_weighted_repeated_atoms_keep_nuclear_rollback(
    native_pair_owner: Any,
    angular: tuple[int, int, int, int],
    atoms: tuple[int, ...],
) -> None:
    """Coincident physical centers retain the qualified nuclear response."""
    inputs = _inputs(True, angular=angular, class_index=LR_CLASSES.index(angular))
    inputs["atoms"] = np.array(atoms, dtype=np.int32)

    def execute(*args: Any) -> None:
        native_pair_owner.long_range(*args, 0.3)

    actual, _ = _run(execute, inputs, coulomb=0.0, exchange=-0.72)
    expected, _ = _run(execute, inputs, coulomb=0.0, exchange=-0.72, relative=False)
    np.testing.assert_allclose(actual, expected, rtol=5e-11, atol=5e-12)
