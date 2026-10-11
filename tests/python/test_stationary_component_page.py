"""Device-free equivalence gates for bounded spherical derivative task packing."""

import typing
from itertools import product

import numpy as np
import pytest


def _owner(capacity: int, work_budget: int) -> typing.Any:
    """Build an owner with signed ragged components and non-local artifact kinds."""
    from generativeqc._stationary_cuda import _component_domain, _CudaSources
    from generativeqc_compiler.integral.first_derivative_schedule import (
        derivative_requests,
    )

    owner = object.__new__(_CudaSources)
    owner.integral_derivatives = True
    owner.component_mode = True
    owner.expansions = (
        (("", 1.0),),
        (("x", 0.321234567890123),),
        (("xx", 1.23456789012345), ("yy", -0.731234567890123), ("zz", 0.33)),
        (("xy", -1.0), ("xz", 0.75)),
        (("z", 0.5),),
    )
    owner.nao = len(owner.expansions)
    owner.aos = np.zeros((owner.nao, 16))
    owner.aos[:, 2] = [1, 2, 3, 1, 2]
    owner.kinds = {
        request: 7 + 3 * kind
        for kind, request in enumerate(
            derivative_requests(_component_domain(owner.expansions))
        )
    }
    owner.kind_tables = {}
    owner.component_page_arrays = None
    owner.tasks = np.empty((capacity, 9), dtype=np.int64)
    owner.charges = np.empty(capacity)
    owner.used = 0
    owner.pending_primitive_records = 0
    owner.page_work_budget = work_budget
    owner.scalar_packed_descriptors = 0
    owner.bulk_packed_descriptors = 0
    owner.bulk_pack_chunks = 0
    owner.pages = []

    def flush() -> None:
        if owner.used:
            owner.pages.append(
                (
                    owner.tasks[: owner.used].copy(),
                    owner.charges[: owner.used].copy(),
                    owner.pending_primitive_records,
                )
            )
            owner.used = 0
            owner.pending_primitive_records = 0

    owner.flush = flush
    return owner


@pytest.mark.parametrize(
    ("operator", "rank", "nucleus"),
    (
        ("overlap", 2, None),
        ("kinetic", 2, None),
        ("nuclear_attraction", 2, 2),
        ("four_center_eri", 4, None),
    ),
)
@pytest.mark.parametrize("capacity", [1, 7, 64])
@pytest.mark.parametrize("work_budget", [81, 1024])
@pytest.mark.parametrize("charge", [1.0, -0.71345678901234])
def test_component_page_matches_scalar_descriptors_and_flush_boundaries(
    monkeypatch: pytest.MonkeyPatch,
    operator: str,
    rank: int,
    nucleus: int | None,
    capacity: int,
    work_budget: int,
    charge: float,
) -> None:
    """Check exact FP64 products, encoded permutations, AO indices and page work."""
    scalar = _owner(capacity, work_budget)
    bulk = _owner(capacity, work_budget)
    coordinates = tuple(product(range(scalar.nao), repeat=rank))
    for indices in coordinates:
        scalar.integral(3, operator, indices, nucleus=nucleus, charge=charge)
    scalar.flush()
    monkeypatch.setenv("GENERATIVEQC_STATIONARY_COMPONENT_PACKING", "bulk")
    bulk.integral_page(3, operator, iter(coordinates), nucleus=nucleus, charge=charge)
    bulk.flush()

    assert len(bulk.pages) == len(scalar.pages)
    assert bulk.bulk_packed_descriptors == scalar.scalar_packed_descriptors
    assert bulk.scalar_packed_descriptors == 0
    assert bulk.bulk_packed_descriptors == sum(
        np.prod([len(bulk.expansions[index]) for index in row]) for row in coordinates
    )
    for actual, expected in zip(bulk.pages, scalar.pages, strict=True):
        np.testing.assert_array_equal(actual[0], expected[0])
        np.testing.assert_array_equal(
            actual[1].view(np.uint64), expected[1].view(np.uint64)
        )
        assert actual[2] == expected[2] <= work_budget
        assert len(actual[0]) <= capacity
    assert sum(array.nbytes for array in bulk.component_page_arrays) == 7 * 8 * bulk.nao
    assert all(table.size <= 10**4 for table in bulk.kind_tables.values())


def test_component_page_expanded_windows_stay_capacity_bounded(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A three-term d quartet spans windows rather than allocating page*81 rows."""
    owner = _owner(7, 1024)
    windows = []
    pack = owner._pack_integral_descriptors

    def record(
        source: int,
        indices: np.ndarray,
        kinds: np.ndarray,
        primitive_work: np.ndarray,
        nucleus: int | None,
        charges: np.ndarray,
    ) -> None:
        windows.append(len(indices))
        pack(source, indices, kinds, primitive_work, nucleus, charges)

    monkeypatch.setattr(owner, "_pack_integral_descriptors", record)
    monkeypatch.setenv("GENERATIVEQC_STATIONARY_COMPONENT_PACKING", "bulk")
    owner.integral_page(0, "four_center_eri", ((2, 2, 2, 2),) * 37)
    assert sum(windows) == 37 * 81
    assert max(windows) == 7


def test_component_pages_preserve_partially_filled_cross_operator_buffers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A new logical/operator page must not force an extra native flush."""
    scalar = _owner(7, 15)
    bulk = _owner(7, 15)
    coordinates = ((2, 0), (4, 3), (1, 2), (0, 0))
    for owner in (scalar, bulk):
        owner.integral(2, "overlap", (1, 4), charge=0.5)
    monkeypatch.setenv("GENERATIVEQC_STATIONARY_COMPONENT_PACKING", "bulk")
    for operator, nucleus in (("kinetic", None), ("nuclear_attraction", 3)):
        for indices in coordinates:
            scalar.integral(4, operator, indices, nucleus=nucleus, charge=-1.75)
        bulk.integral_page(4, operator, coordinates, nucleus=nucleus, charge=-1.75)
    scalar.flush()
    bulk.flush()
    assert len(scalar.pages) == len(bulk.pages)
    for actual, expected in zip(bulk.pages, scalar.pages, strict=True):
        np.testing.assert_array_equal(actual[0], expected[0])
        np.testing.assert_array_equal(
            actual[1].view(np.uint64), expected[1].view(np.uint64)
        )
        assert actual[2] == expected[2]


@pytest.mark.parametrize("indices", [(-1, 0), (0, 5)])
def test_component_page_rejects_invalid_public_ao_indices(
    monkeypatch: pytest.MonkeyPatch, indices: tuple[int, ...]
) -> None:
    owner = _owner(7, 1024)
    monkeypatch.setenv("GENERATIVEQC_STATIONARY_COMPONENT_PACKING", "bulk")
    with pytest.raises(ValueError, match="invalid AO indices"):
        owner.integral_page(0, "overlap", (indices,))
    assert owner.used == 0


def test_component_page_keeps_primitive_budget_gate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    owner = _owner(7, 8)
    monkeypatch.setenv("GENERATIVEQC_STATIONARY_COMPONENT_PACKING", "bulk")
    with pytest.raises(ValueError, match="primitive page work budget"):
        owner.integral_page(0, "overlap", ((2, 2),))
    assert owner.used == 0


def test_component_page_missing_artifact_kind_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    owner = _owner(7, 1024)
    owner.kinds.clear()
    monkeypatch.setenv("GENERATIVEQC_STATIONARY_COMPONENT_PACKING", "bulk")
    with pytest.raises(ValueError, match="unavailable derivative kind"):
        owner.integral_page(0, "overlap", ((0, 0),))
    assert owner.used == 0


@pytest.mark.parametrize("mode", [None, "bulk", "scalar", "invalid"])
def test_component_page_defaults_bulk_and_retains_scalar_fallback(
    monkeypatch: pytest.MonkeyPatch, mode: str | None
) -> None:
    owner = _owner(7, 1024)
    if mode is None:
        monkeypatch.delenv("GENERATIVEQC_STATIONARY_COMPONENT_PACKING", raising=False)
    else:
        monkeypatch.setenv("GENERATIVEQC_STATIONARY_COMPONENT_PACKING", mode)
    if mode == "invalid":
        with pytest.raises(ValueError, match="must be scalar or bulk"):
            owner.integral_page(0, "overlap", ((2, 0),))
        return
    owner.integral_page(0, "overlap", ((2, 0),))
    if mode == "scalar":
        assert owner.scalar_packed_descriptors == 3
        assert owner.component_page_arrays is None
        assert owner.bulk_packed_descriptors == 0
    else:
        assert owner.scalar_packed_descriptors == 0
        assert owner.component_page_arrays is not None
        assert owner.bulk_packed_descriptors == 3
