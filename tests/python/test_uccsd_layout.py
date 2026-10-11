"""Independent fermionic action oracle for UCCSD layout signs and coefficients."""

from __future__ import annotations

import itertools
import typing

import numpy as np
import pytest
from generativeqc_compiler.cc.uccsd_layout import BLOCK_NAMES, UCCSDBlock, UCCSDLayout


def _fixture(
    layout: UCCSDLayout, seed: int = 37
) -> tuple[dict, np.ndarray, np.ndarray]:
    """Build full spin-orbital tensors without layout coordinate helpers."""
    rng = np.random.default_rng(seed)
    os = [0] * layout.oa + [1] * layout.ob
    vs = [0] * layout.va + [1] * layout.vb
    singles = rng.normal(size=(len(os), len(vs)))
    singles[np.array(os)[:, None] != np.array(vs)[None, :]] = 0
    doubles = np.zeros((len(os), len(os), len(vs), len(vs)))
    for i, j in itertools.combinations(range(len(os)), 2):
        for a, b in itertools.combinations(range(len(vs)), 2):
            if sorted((os[i], os[j])) == sorted((vs[a], vs[b])):
                value = rng.normal()
                doubles[i, j, a, b] = doubles[j, i, b, a] = value
                doubles[j, i, a, b] = doubles[i, j, b, a] = -value
    oa, va = layout.oa, layout.va
    return (
        {
            "t1a": singles[:oa, :va].copy(),
            "t1b": singles[oa:, va:].copy(),
            "t2aa": doubles[:oa, :oa, :va, :va].copy(),
            "t2ab": doubles[:oa, oa:, :va, va:].copy(),
            "t2bb": doubles[oa:, oa:, va:, va:].copy(),
        },
        singles,
        doubles,
    )


def _apply(state: int, operations: tuple[tuple[int, bool], ...]) -> tuple[int, int]:
    """Right-to-left CAR action on bit determinants, independent of amplitudes."""
    sign = 1
    for orbital, create in operations:
        occupied = bool(state & (1 << orbital))
        if occupied == create:
            return state, 0
        sign *= -1 if (state & ((1 << orbital) - 1)).bit_count() % 2 else 1
        state ^= 1 << orbital
    return state, sign


def _matrix(norb: int, nelec: int, terms: list) -> dict:
    """Sparse operator matrix on the entire fixed-particle determinant space."""
    result = {}
    for occupied in itertools.combinations(range(norb), nelec):
        source = sum(1 << p for p in occupied)
        for value, operations in terms:
            target, sign = _apply(source, operations)
            if sign and value:
                key = (target, source)
                result[key] = result.get(key, 0.0) + sign * value
    return {key: value for key, value in result.items() if value != 0}


def _orbitals(layout: UCCSDLayout) -> tuple[list[int], list[int]]:
    # Fock ordering: occupied alpha, virtual alpha, occupied beta, virtual beta.
    oa, ob, va, vb = layout.oa, layout.ob, layout.va, layout.vb
    return (
        list(range(oa)) + list(range(oa + va, oa + va + ob)),
        list(range(oa, oa + va)) + list(range(oa + va + ob, oa + va + ob + vb)),
    )


def _spin_terms(layout: UCCSDLayout, t1: np.ndarray, t2: np.ndarray) -> list:
    occ, vir = _orbitals(layout)
    terms = [
        (t1[i, a], ((occ[i], False), (vir[a], True))) for i, a in np.ndindex(t1.shape)
    ]
    terms += [
        (
            0.25 * t2[i, j, a, b],
            ((occ[i], False), (occ[j], False), (vir[b], True), (vir[a], True)),
        )
        for i, j, a, b in np.ndindex(t2.shape)
    ]
    return terms


def _packed_terms(
    layout: UCCSDLayout,
    packed: np.ndarray,
    *,
    wrong_sign: bool = False,
    wrong_factor: float = 1.0,
) -> list:
    occ, vir = _orbitals(layout)
    terms = []
    for block in layout.blocks:
        for slot, value in enumerate(packed[layout.block_slices[block.name]]):
            indices = block.representative(slot)
            beta = block.name in ("t1b", "t2bb")
            if len(indices) == 2:
                i, a = indices
                terms.append(
                    (
                        value,
                        (
                            (occ[i + beta * layout.oa], False),
                            (vir[a + beta * layout.va], True),
                        ),
                    )
                )
            else:
                i, j, a, b = indices
                i += beta * layout.oa
                a += beta * layout.va
                j += (beta or block.name == "t2ab") * layout.oa
                b += (beta or block.name == "t2ab") * layout.va
                terms.append(
                    (
                        value * wrong_factor * (-1 if wrong_sign else 1),
                        (
                            (occ[i], False),
                            (occ[j], False),
                            (vir[b], True),
                            (vir[a], True),
                        ),
                    )
                )
    return terms


def _assert_matrix_equal(left: dict, right: dict) -> None:
    assert left.keys() == right.keys()
    for key, value in left.items():
        assert value == pytest.approx(right[key], abs=1e-14, rel=1e-14)


@pytest.mark.parametrize(
    "dimensions",
    [
        (3, 2, 2, 3),
        (2, 1, 3, 2),
        (1, 3, 2, 1),
        (2, 0, 3, 0),
        (0, 2, 0, 3),
        (1, 1, 1, 1),
        (0, 0, 0, 0),
        (0, 1, 2, 0),
    ],
)
def test_independent_spin_orbital_determinant_action(
    dimensions: tuple[int, ...],
) -> None:
    layout = UCCSDLayout(*dimensions)
    full, t1, t2 = _fixture(layout)
    packed = layout.pack(full)
    replay = layout.unpack(packed)
    for name in BLOCK_NAMES:
        np.testing.assert_array_equal(replay[name], full[name])
    norb, nelec = sum(dimensions), layout.oa + layout.ob
    oracle = _matrix(norb, nelec, _spin_terms(layout, t1, t2))
    actual = _matrix(norb, nelec, _packed_terms(layout, packed))
    _assert_matrix_equal(actual, oracle)
    reference = sum(1 << p for p in _orbitals(layout)[0])
    norm = sum(value**2 for (_, source), value in actual.items() if source == reference)
    assert norm == pytest.approx(np.sum(packed**2))
    if np.any(t2):
        with pytest.raises(AssertionError):
            _assert_matrix_equal(
                _matrix(norb, nelec, _packed_terms(layout, packed, wrong_sign=True)),
                oracle,
            )
        with pytest.raises(AssertionError):
            _assert_matrix_equal(
                _matrix(norb, nelec, _packed_terms(layout, packed, wrong_factor=0.25)),
                oracle,
            )


def test_coordinates_signs_counts_metric_and_spin_descriptors() -> None:
    layout = UCCSDLayout(3, 2, 4, 3)
    full, _, _ = _fixture(layout)
    assert layout.packed_size == 12 + 6 + 18 + 72 + 3
    assert list(layout.block_slices) == list(BLOCK_NAMES)
    for block in layout.blocks:
        shape = block.spec.shape
        assert block.spec.dtype == "float64"
        assert block.spec.representation == "spin_orbital"
        assert tuple(i.name for i in block.spec.indices) == tuple(
            "ia" if len(shape) == 2 else "ijab"
        )
        expected_spins = {
            "t1a": ("alpha", "alpha"),
            "t1b": ("beta", "beta"),
            "t2aa": ("alpha",) * 4,
            "t2bb": ("beta",) * 4,
            "t2ab": ("alpha", "beta", "alpha", "beta"),
        }[block.name]
        assert tuple(i.space.spin for i in block.spec.indices) == expected_spins
        assert len(block.spec.symmetries) == (2 if block.same_spin else 0)
        packed = block.pack(full[block.name])
        representatives = (
            [
                (i, j, a, b)
                for i, j in itertools.combinations(range(shape[0]), 2)
                for a, b in itertools.combinations(range(shape[2]), 2)
            ]
            if block.same_spin
            else list(np.ndindex(shape))
        )
        assert [
            block.representative(slot) for slot in range(len(packed))
        ] == representatives
        np.testing.assert_array_equal(
            packed, np.array([full[block.name][indices] for indices in representatives])
        )
        assert np.sum(full[block.name] ** 2) == pytest.approx(
            block.full_metric_weight * np.sum(packed**2)
        )
        assert float(block.operator_prefactor) == (0.25 if block.same_spin else 1.0)
        for indices in np.ndindex(shape):
            slot, sign = block.coordinate(indices)
            if slot is None:
                assert sign == 0 and full[block.name][indices] == 0
            else:
                assert block.coordinate(block.representative(slot)) == (slot, 1)
                assert full[block.name][indices] == sign * packed[slot]
        if block.same_spin:
            assert block.coordinate((0, 1, 0, 1)) == (0, 1)
            assert block.coordinate((1, 0, 0, 1)) == (0, -1)
            assert block.coordinate((0, 1, 1, 0)) == (0, -1)
            assert block.coordinate((1, 0, 1, 0)) == (0, 1)
            assert block.coordinate((0, 0, 0, 1)) == (None, 0)
    mixed = layout.blocks[3]
    assert mixed.coordinate((2, 0, 3, 1)) == (((2 * 2 + 0) * 4 + 3) * 3 + 1, 1)
    with pytest.raises(ValueError):
        mixed.coordinate((0, 2, 1, 3))  # Sorting alpha/beta axes is illegal here.
    equal = UCCSDLayout(2, 2, 2, 2).blocks[3]
    assert equal.coordinate((1, 0, 1, 0))[0] != equal.coordinate((0, 1, 0, 1))[0]
    assert layout.block_slices == {
        "t1a": slice(0, 12),
        "t1b": slice(12, 18),
        "t2aa": slice(18, 36),
        "t2ab": slice(36, 108),
        "t2bb": slice(108, 111),
    }


def test_restricted_embedding_against_spin_free_operator() -> None:
    """Compare 1/2 sum R_ijab E_ai E_bj to packed unrestricted T2."""
    o, v = 2, 2
    rng = np.random.default_rng(81)
    r1 = rng.normal(size=(o, v))
    raw = rng.normal(size=(o, o, v, v))
    r2 = (raw + raw.transpose(1, 0, 3, 2)) * 0.5
    same = r2 - r2.swapaxes(2, 3)
    layout = UCCSDLayout(o, o, v, v)
    packed = layout.pack({"t1a": r1, "t1b": r1, "t2aa": same, "t2ab": r2, "t2bb": same})
    occ, vir = _orbitals(layout)
    terms = []
    for spin in (0, 1):
        for i, a in np.ndindex(r1.shape):
            terms.append(
                (r1[i, a], ((occ[i + spin * o], False), (vir[a + spin * v], True)))
            )
    for sigma, tau in itertools.product((0, 1), repeat=2):
        for i, j, a, b in np.ndindex(r2.shape):
            # E_ai E_bj: apply E_bj first. CAR supplies the fermionic signs.
            terms.append(
                (
                    0.5 * r2[i, j, a, b],
                    (
                        (occ[j + tau * o], False),
                        (vir[b + tau * v], True),
                        (occ[i + sigma * o], False),
                        (vir[a + sigma * v], True),
                    ),
                )
            )
    _assert_matrix_equal(
        _matrix(8, 4, terms), _matrix(8, 4, _packed_terms(layout, packed))
    )


@pytest.mark.parametrize("bad", [True, -1, 1.5, "2", np.int64(2), 2**63])
def test_dimension_type_and_range_fail_closed(bad: typing.Any) -> None:
    for position in range(4):
        dimensions = [2] * 4
        dimensions[position] = bad
        with pytest.raises(ValueError):
            UCCSDLayout(*dimensions)


def test_overflow_and_invalid_descriptor() -> None:
    with pytest.raises(ValueError):
        UCCSDLayout(100_000, 2, 100_000, 2)
    with pytest.raises(ValueError):
        UCCSDLayout(1, 1, 2**60, 1)  # FP64 byte overflow before allocation.
    with pytest.raises(ValueError):
        UCCSDBlock("t2ba", (2, 2, 2, 2))
    with pytest.raises(ValueError):
        UCCSDBlock("t1a", (2, 2, 2))


@pytest.mark.parametrize("bad", [True, -1, 1.2, "0", np.int64(0), 2**63])
def test_coordinate_and_slot_admission(bad: typing.Any) -> None:
    for block in UCCSDLayout(2, 2, 2, 2).blocks:
        with pytest.raises(ValueError):
            block.representative(bad)
        with pytest.raises(ValueError):
            block.coordinate((bad,) + (0,) * (len(block.spec.shape) - 1))


def test_exact_antisymmetry_no_projection_and_no_mutation() -> None:
    layout = UCCSDLayout(2, 2, 2, 2)
    full, _, _ = _fixture(layout)
    before = {name: value.copy() for name, value in full.items()}
    packed = layout.pack(full)
    assert not any(np.shares_memory(packed, value) for value in full.values())
    for name in ("t2aa", "t2bb"):
        for indices in (
            (0, 0, 0, 1),
            (0, 1, 0, 0),
            (0, 1, 0, 1),
            (1, 0, 0, 1),
            (0, 1, 1, 0),
        ):
            invalid = {key: value.copy() for key, value in before.items()}
            invalid[name][indices] = np.nextafter(invalid[name][indices], np.inf)
            with pytest.raises(ValueError):
                layout.pack(invalid)
    for name in BLOCK_NAMES:
        np.testing.assert_array_equal(full[name], before[name])
    for value in (np.nextafter(0.0, 1.0), np.finfo(np.float64).max):
        p = np.full(layout.packed_size, value)
        np.testing.assert_array_equal(layout.pack(layout.unpack(p)), p)


def test_shapes_nonfinite_dtype_and_empty_admission() -> None:
    layout = UCCSDLayout(2, 1, 2, 3)
    full, _, _ = _fixture(layout)
    for block in layout.blocks:
        for dtype in (np.float32, np.complex128, np.int64, bool, object):
            invalid = dict(full)
            invalid[block.name] = full[block.name].astype(dtype)
            with pytest.raises(ValueError):
                layout.pack(invalid)
        for value in (np.inf, -np.inf, np.nan):
            invalid = {name: array.copy() for name, array in full.items()}
            invalid[block.name].flat[0] = value
            with pytest.raises(ValueError):
                layout.pack(invalid)
        with pytest.raises(ValueError):
            block.pack(np.zeros((full[block.name].size,)))
        for index in ((), (0,) * (len(block.spec.shape) + 1)):
            with pytest.raises(ValueError):
                block.coordinate(index)
        with pytest.raises(ValueError):
            block.representative(block.packed_layout.storage_elements)
    for invalid in ({}, {**full, "t2ba": full["t2ab"]}):
        with pytest.raises(ValueError):
            layout.pack(invalid)
    for invalid in (
        [],
        np.zeros((layout.packed_size, 1)),
        np.zeros(layout.packed_size - 1),
        np.full(layout.packed_size, np.nan),
        np.zeros(layout.packed_size, dtype=np.complex128),
    ):
        with pytest.raises(ValueError):
            layout.unpack(invalid)
    empty = UCCSDLayout(0, 0, 0, 0)
    for dtype in (np.float32, np.complex128, object):
        with pytest.raises(ValueError):
            empty.unpack(np.array([], dtype=dtype))
    assert empty.pack(empty.unpack(np.array([], dtype=np.float64))).size == 0
    for block in empty.blocks:
        with pytest.raises(ValueError):
            block.representative(0)
        with pytest.raises(ValueError):
            block.coordinate((0,) * len(block.spec.shape))


def test_pair_inverse_large_exact_integer_coordinates() -> None:
    # Full byte extent is admitted but pair slots already exceed FP64's exact
    # integer range; no tensor allocation or floating square-root inverse.
    block = UCCSDBlock("t2aa", (200_000_000, 0, 2, 0))
    for i, j in ((0, 1), (100_000_001, 100_000_002), (199_999_998, 199_999_999)):
        slot, sign = block.coordinate((i, j, 0, 1))
        assert sign == 1
        assert block.representative(slot) == (i, j, 0, 1)


def test_array_subclasses_rejected_and_strided_arrays_preserved() -> None:
    layout = UCCSDLayout(2, 2, 2, 2)
    full, _, _ = _fixture(layout)
    for block in layout.blocks:
        invalid = np.ma.array(full[block.name], mask=False, copy=True)
        invalid.flat[0] = np.nan
        invalid.mask.flat[0] = True
        with pytest.raises(ValueError):
            block.pack(invalid)
    with pytest.raises(ValueError):
        layout.unpack(np.ma.array(np.ones(layout.packed_size), mask=False))
    strided = {name: np.asfortranarray(array) for name, array in full.items()}
    np.testing.assert_array_equal(layout.pack(strided), layout.pack(full))
    padded = np.zeros(2 * layout.packed_size)
    padded[::2] = layout.pack(full)
    for name, array in layout.unpack(padded[::2]).items():
        np.testing.assert_array_equal(array, full[name])
        assert not np.shares_memory(array, padded)
