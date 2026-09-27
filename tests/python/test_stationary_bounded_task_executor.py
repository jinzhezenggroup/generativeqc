"""Host-only tests for the bounded stationary derivative task executor."""

from __future__ import annotations

import pytest
from vibeqc._stationary_cuda import _BoundedStationaryTaskExecutor
from vibeqc_compiler.common.runtime_domain import RuntimeTaskDomain


@pytest.mark.parametrize(
    ("extents", "page_capacity", "resident_capacity", "mode"),
    (
        ((2, 2), 4, 8, "fixed"),
        ((3, 3), 4, 16, "resident"),
        ((3, 4), 4, 8, "paged"),
    ),
)
def test_executor_classifies_and_covers_runtime_domain(
    extents: tuple[int, ...],
    page_capacity: int,
    resident_capacity: int,
    mode: str,
) -> None:
    domain = RuntimeTaskDomain.rectangular(extents)
    executor = _BoundedStationaryTaskExecutor(
        fixed_capacity=page_capacity,
        resident_capacity=resident_capacity,
        page_capacity=resident_capacity,
    )
    seen: list[tuple[int, ...]] = []
    page_finishes = 0

    def finish_page() -> None:
        nonlocal page_finishes
        page_finishes += 1

    execution = executor.execute(domain, seen.append, finish_page=finish_page)

    assert execution.mode == mode
    assert execution.domain_identity == domain.identity
    assert execution.logical_tasks == domain.logical_size
    assert execution.fixed_capacity == page_capacity
    assert execution.page_capacity == resident_capacity
    assert execution.resident_capacity == resident_capacity
    assert execution.producer_pages == domain.page_count(resident_capacity)
    assert page_finishes == execution.producer_pages
    assert seen == list(domain)


@pytest.mark.parametrize(
    ("page_capacity", "resident_capacity"),
    ((0, 1), (1, 0), (4, 3), (True, 4)),
)
def test_executor_rejects_invalid_capacities(
    page_capacity: object, resident_capacity: object
) -> None:
    with pytest.raises(ValueError):
        _BoundedStationaryTaskExecutor(
            fixed_capacity=page_capacity,  # type: ignore[arg-type]
            resident_capacity=resident_capacity,  # type: ignore[arg-type]
            page_capacity=resident_capacity,  # type: ignore[arg-type]
        )


def test_executor_rejects_non_domain_or_non_callback() -> None:
    executor = _BoundedStationaryTaskExecutor(
        fixed_capacity=2, resident_capacity=4, page_capacity=4
    )
    domain = RuntimeTaskDomain.rectangular((2,))

    with pytest.raises(TypeError, match="RuntimeTaskDomain"):
        executor.execute(object(), lambda _coordinate: None)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="submit callback"):
        executor.execute(domain, object())  # type: ignore[arg-type]


def test_execution_evidence_does_not_retain_coordinate_pages() -> None:
    domain = RuntimeTaskDomain.rectangular((17, 19))
    executor = _BoundedStationaryTaskExecutor(
        fixed_capacity=7, resident_capacity=32, page_capacity=32
    )
    seen = 0

    def count(_coordinate: tuple[int, ...]) -> None:
        nonlocal seen
        seen += 1

    execution = executor.execute(domain, count)

    assert execution.mode == "paged"
    assert execution.producer_pages == domain.page_count(32)
    assert seen == domain.logical_size
    assert not hasattr(execution, "coordinates")


def test_fixed_and_paged_schedules_preserve_domain_result_and_identity() -> None:
    domain = RuntimeTaskDomain.rectangular((5, 4, 3))

    def run(executor: _BoundedStationaryTaskExecutor) -> tuple[object, int, list[tuple[int, ...]]]:
        coordinates: list[tuple[int, ...]] = []
        total = 0

        def submit(coordinate: tuple[int, ...]) -> None:
            nonlocal total
            coordinates.append(coordinate)
            total += 1 + sum((axis + 1) * value for axis, value in enumerate(coordinate))

        execution = executor.execute(domain, submit)
        return execution, total, coordinates

    fixed, fixed_total, fixed_coordinates = run(
        _BoundedStationaryTaskExecutor(
            fixed_capacity=domain.logical_size,
            resident_capacity=domain.logical_size,
            page_capacity=domain.logical_size,
        )
    )
    paged, paged_total, paged_coordinates = run(
        _BoundedStationaryTaskExecutor(
            fixed_capacity=2,
            resident_capacity=7,
            page_capacity=7,
        )
    )

    assert fixed.mode == "fixed"
    assert paged.mode == "paged"
    assert fixed.domain_identity == paged.domain_identity == domain.identity
    assert fixed.logical_tasks == paged.logical_tasks == domain.logical_size
    assert fixed.producer_pages == 1
    assert paged.producer_pages == domain.page_count(7)
    assert fixed_coordinates == paged_coordinates == list(domain)
    assert fixed_total == paged_total
