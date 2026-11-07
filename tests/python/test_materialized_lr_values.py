"""Independent bounded canonical LR publication inventory and refusal gates."""

from __future__ import annotations

import itertools
import math

import numpy as np
import pytest

from tools.generativeqc_validation.materialized_lr_values import (
    REJECTED,
    SIZE_MAX,
    admit_request,
    canonical_layout,
    checked_extent,
    geometry_inputs,
    native_inputs,
    pair,
    publication_maps,
    quartet_key,
    raw_exchange,
    shell_pairs,
)

COMPOSITIONS = tuple(
    a for a in itertools.product(range(3), repeat=4) if 3 <= sum(a) <= 5
)


@pytest.mark.parametrize("angular", COMPOSITIONS)
def test_shell_components_inverse_canonical_slots(angular: tuple[int, ...]) -> None:
    """All SPD compositions, sorted ties, threshold equality, holes and empty rows."""
    native = native_inputs(geometry_inputs(angular, "separated"))
    offsets = native["offsets"]
    # Add repeated AO indices where equal angular slots can share a shell owner.
    owners_list = [[(1, 0), (3, 2)]]
    if angular[0] == angular[1]:
        owners_list.append([(0, 0), (3, 2)])
    if angular[2] == angular[3]:
        owners_list.append([(1, 0), (2, 2)])
    if angular == (1, 1, 1, 1):
        owners_list.append([(0, 0)])
    for owners in owners_list:
        same = len(owners) == 1 or sum(angular[:2]) == sum(angular[2:])
        first = shell_pairs(offsets, *owners[0])
        second = first if len(owners) == 1 else shell_pairs(offsets, *owners[1])
        if same:
            first = list(set(first + second))
            second = first
        bounds = {
            p: (0.0, 0.125, 0.25, 0.5, 1.0)[(p[0] + 2 * p[1]) % 5]
            for p in set(first + second)
        }
        for threshold in (0.0, 0.125, 2.0):
            inventories = []
            for compact in (False, True):
                layout = canonical_layout(
                    first, second, bounds, threshold, same=same, compact=compact
                )
                tasks, visits = publication_maps(angular, offsets, owners, layout)
                assert all(hit in (0, 1) for hit in visits)
                assert all(len(task["map"]) <= 192 for task in tasks)
                actual = {
                    quartet_key(layout.decode(slot))
                    for slot, hit in enumerate(visits)
                    if hit
                }
                # A brute-force pair product controls the inventory independently
                # of either prefix decode or component inverse-rank arithmetic.
                expected = {
                    quartet_key((*a, *b))
                    for a in first
                    for b in second
                    if bounds[a] * bounds[b] >= threshold
                }
                assert actual == expected
                for task in tasks:
                    for component, target in zip(
                        task["components"], task["map"], strict=False
                    ):
                        if target == REJECTED:
                            assert (
                                bounds[pair(*component[:2])]
                                * bounds[pair(*component[2:])]
                                < threshold
                            )
                            continue
                        # All eight AO/pair swaps bind the identical scalar slot.
                        i, j, k, l = component
                        for q in (
                            (i, j, k, l),
                            (j, i, k, l),
                            (i, j, l, k),
                            (j, i, l, k),
                            (k, l, i, j),
                            (l, k, i, j),
                            (k, l, j, i),
                            (l, k, j, i),
                        ):
                            assert layout.slot(q) == target
                inventories.append(actual)
            assert inventories[0] == inventories[1]


def test_sorted_triangle_ownership_and_exact_product_boundary() -> None:
    first = [(3, 0), (2, 1), (1, 0), (0, 0)]
    bounds = {first[0]: 0.25, first[1]: 1.0, first[2]: 0.5, first[3]: 0.0}
    dense = canonical_layout(first, first, bounds, 0.125, same=True, compact=False)
    compact = canonical_layout(first, first, bounds, 0.125, same=True, compact=True)
    # .25*.5 equals threshold and survives. The smaller AO pair ranks before
    # the larger AO pair, so numeric AO order cannot define triangle ownership.
    indices = (*first[0], *first[2])
    assert dense.slot(indices) == 4
    assert compact.slot(indices) is not None
    assert compact.prefix[-1] == compact.prefix[-2]  # an empty row
    assert dense.slot((*first[3], *first[1])) is None
    for slot in range(compact.count):
        assert compact.slot(compact.decode(slot)) == slot


@pytest.mark.parametrize(
    "kwargs",
    [
        {"precision": "fp32"},
        {"operator": "full_range"},
        {"operator": "short_range"},
        {"derivative_order": 1},
        {"omega": -0.01},
        {"omega": math.inf},
        {"omega": math.nan},
        {"capacity": 31},
    ],
)
def test_unsupported_requests_refuse(kwargs: dict) -> None:
    with pytest.raises(ValueError):
        admit_request((1, 1, 1, 0), **kwargs)


@pytest.mark.parametrize(
    "angular", [(0, 0, 0, 0), (2, 2, 2, 0), (3, 0, 0, 0), (1, 1, 1), (-1, 2, 1, 1)]
)
def test_order_and_shell_envelope_refuse(angular: tuple[int, ...]) -> None:
    with pytest.raises(ValueError):
        admit_request(angular)


def test_capacity_and_overflow_refuse_before_map() -> None:
    assert checked_extent(65536, 65536, False) == 1 << 32
    assert checked_extent(65536, 65536, True) == 65536 * 65537 // 2
    for args in (
        (SIZE_MAX, 2, False),
        (SIZE_MAX, SIZE_MAX, True),
        (5, 4, True),
        (-1, 1, False),
    ):
        with pytest.raises(ValueError):
            checked_extent(*args)
    with pytest.raises(ValueError):
        canonical_layout(
            [(0, 0)],
            [(1, 1)],
            {(0, 0): 1.0, (1, 1): 1.0},
            0.0,
            same=False,
            compact=True,
            capacity=0,
        )
    with pytest.raises(ValueError):
        canonical_layout(
            [(0, 0)],
            [(1, 1)],
            {(0, 0): math.nan, (1, 1): 1.0},
            0.0,
            same=False,
            compact=True,
        )


def test_signed_raw_k_orbit_matches_independent_dense_tensor() -> None:
    pairs = [(0, 0), (1, 0), (1, 1)]
    layout = canonical_layout(
        pairs, pairs, dict.fromkeys(pairs, 1.0), 0.0, same=True, compact=True
    )
    values = [(-1) ** i * (i + 0.3) for i in range(layout.count)]
    density = np.array([[0.2, -0.4], [0.7, -0.1]])
    eri = np.empty((2, 2, 2, 2))
    for q in itertools.product(range(2), repeat=4):
        eri[q] = values[layout.slot(q)]
    np.testing.assert_allclose(
        raw_exchange(values, layout, density),
        np.einsum("abcd,bd->ac", eri, density),
        atol=1e-15,
        rtol=1e-15,
    )


def test_libcint_zero_lr_semantics_and_native_normalization() -> None:
    """Bind the external normalization adapter without using a native evaluator."""
    pytest.importorskip("pyscf")
    from pyscf import lib

    from tools.generate_validation_references import pyscf_molecule

    lib.num_threads(1)
    inputs = geometry_inputs((2, 1, 1, 1), "coincident")
    native = native_inputs(inputs)
    mol, scale, _ = pyscf_molecule(inputs)
    assert np.isfinite(native["weights"]).all()
    for shell in range(4):
        # Independent overlap integrals verify normalized radial/component inputs.
        angular = inputs["shells"][shell]["angular_momentum"]
        labels = [
            tuple(p)
            for p in np.array(native["powers"]).reshape(-1, 3)[
                native["offsets"][shell] : native["offsets"][shell + 1]
            ]
        ]
        exponent_weights = list(
            zip(
                native["exponents"][
                    native["primitive_offsets"][shell] : native["primitive_offsets"][
                        shell + 1
                    ]
                ],
                native["weights"][
                    native["primitive_offsets"][shell] : native["primitive_offsets"][
                        shell + 1
                    ]
                ],
                strict=True,
            )
        )
        for local, powers in enumerate(labels):
            overlap = 0.0
            for a, ca in exponent_weights:
                for b, cb in exponent_weights:
                    term = ca * cb
                    for power in powers:
                        term *= (
                            math.prod(range(1, 2 * power, 2))
                            * math.sqrt(math.pi)
                            / (2**power * (a + b) ** (power + 0.5))
                        )
                    overlap += term
            ao = native["offsets"][shell] + local
            assert abs(overlap * native["coefficients"][ao] ** 2 - 1) < 2e-14
            assert angular == sum(powers)
    for omega in (0.03, 0.3, 3.0):
        with mol.with_range_coulomb(omega):
            value = mol.intor_by_shell("int2e", (0, 1, 2, 3))
        scales = [
            scale[native["offsets"][s] : native["offsets"][s + 1]] for s in range(4)
        ]
        normalized = value * np.einsum("i,j,k,l->ijkl", *scales)
        assert np.isfinite(normalized).all()


def test_reference_library_identity_matches_loaded_adapter() -> None:
    """A platform filename difference must fail before expensive reference work."""
    pytest.importorskip("pyscf")
    from pathlib import Path

    from tools.generativeqc_validation.materialized_lr_values import reference_identity

    identity = reference_identity()
    assert identity["pyscf"] == "2.14.0"
    assert Path(identity["libcint"]).is_file()
    assert Path(identity["cgto"]).is_file()
    assert len(identity["libcint_sha256"]) == len(identity["cgto_sha256"]) == 64
