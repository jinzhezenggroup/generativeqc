"""Host-only tests for the bounded stationary derivative task executor."""

from __future__ import annotations

import typing

import pytest
from vibeqc._stationary_cuda import _BoundedStationaryTaskExecutor
from vibeqc_compiler.common.provenance import canonical_hash
from vibeqc_compiler.common.runtime_domain import RuntimeTaskDomain, RuntimeTaskPage


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
    assert execution.source_schema == domain.to_payload()["schema"]
    assert execution.domain_identity == domain.identity
    assert execution.logical_tasks == domain.logical_size
    assert execution.fixed_capacity == page_capacity
    assert execution.page_capacity == resident_capacity
    assert execution.resident_capacity == resident_capacity
    assert execution.producer_pages == domain.page_count(resident_capacity)
    assert page_finishes == execution.producer_pages
    assert seen == list(domain)


def test_executor_preserves_complete_paging_at_effective_fixed_threshold() -> None:
    domain = RuntimeTaskDomain.rectangular((5, 4))
    executor = _BoundedStationaryTaskExecutor(
        fixed_capacity=16,
        resident_capacity=16,
        page_capacity=16,
    )
    seen: list[tuple[int, ...]] = []

    execution = executor.execute(domain, seen.append)

    assert execution.mode == "paged"
    assert execution.logical_tasks == 20
    assert execution.fixed_capacity == 16
    assert execution.resident_capacity == 16
    assert execution.page_capacity == 16
    assert execution.producer_pages == 2
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


def test_executor_rejects_invalid_task_source_or_callback() -> None:
    executor = _BoundedStationaryTaskExecutor(
        fixed_capacity=2, resident_capacity=4, page_capacity=4
    )
    domain = RuntimeTaskDomain.rectangular((2,))

    with pytest.raises(TypeError, match="versioned identity-bearing task source"):
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

    def run(
        executor: _BoundedStationaryTaskExecutor,
    ) -> tuple[object, int, list[tuple[int, ...]]]:
        coordinates: list[tuple[int, ...]] = []
        total = 0

        def submit(coordinate: tuple[int, ...]) -> None:
            nonlocal total
            coordinates.append(coordinate)
            total += 1 + sum(
                (axis + 1) * value for axis, value in enumerate(coordinate)
            )

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


def test_executor_accepts_structural_task_source_and_checks_page_identity() -> None:
    domain = RuntimeTaskDomain.rectangular((3, 2))

    class Source:
        payload: typing.ClassVar[dict[str, str]] = {
            "schema": "vibeqc.synthetic_stationary_task_source.v1",
            "logical_source": "fixture",
        }
        identity = canonical_hash(payload)
        logical_size = domain.logical_size

        @classmethod
        def pages(cls, capacity: int) -> typing.Iterator[RuntimeTaskPage]:
            for page in domain.pages(capacity):
                yield RuntimeTaskPage(
                    cls.identity,
                    page.rank,
                    page.ordinal,
                    page.offset,
                    page.capacity,
                    page.coordinates,
                )

        @classmethod
        def to_payload(cls) -> dict[str, object]:
            return dict(cls.payload)

    executor = _BoundedStationaryTaskExecutor(
        fixed_capacity=2, resident_capacity=4, page_capacity=4
    )
    seen: list[tuple[int, ...]] = []
    execution = executor.execute(Source(), seen.append)
    assert execution.domain_identity == Source.identity
    assert execution.source_schema == Source.payload["schema"]
    assert seen == list(domain)

    class WrongIdentity(Source):
        @classmethod
        def pages(cls, capacity: int) -> typing.Iterator[RuntimeTaskPage]:
            first = next(domain.pages(capacity))
            yield RuntimeTaskPage(
                "0" * 64,
                first.rank,
                first.ordinal,
                first.offset,
                first.capacity,
                first.coordinates,
            )

    with pytest.raises(RuntimeError, match="identity/order mismatch"):
        executor.execute(WrongIdentity(), lambda _coordinate: None)

    class WrongPayload(Source):
        @classmethod
        def to_payload(cls) -> dict[str, object]:
            return {
                "schema": "vibeqc.synthetic_stationary_task_source.v1",
                "logical_source": "changed",
            }

    with pytest.raises(ValueError, match="identity/payload mismatch"):
        executor.execute(WrongPayload(), lambda _coordinate: None)


def test_executor_admits_empty_screened_task_source_without_pages() -> None:
    class EmptySource:
        payload: typing.ClassVar[dict[str, str]] = {
            "schema": "vibeqc.synthetic_stationary_task_source.v1",
            "logical_source": "screened-empty",
        }
        identity = canonical_hash(payload)
        logical_size = 0

        @staticmethod
        def pages(_capacity: int) -> typing.Iterator[RuntimeTaskPage]:
            return iter(())

        @classmethod
        def to_payload(cls) -> dict[str, object]:
            return dict(cls.payload)

    executor = _BoundedStationaryTaskExecutor(
        fixed_capacity=2, resident_capacity=4, page_capacity=4
    )
    seen: list[tuple[int, ...]] = []
    execution = executor.execute(EmptySource(), seen.append)

    assert execution.mode == "empty"
    assert execution.logical_tasks == 0
    assert execution.producer_pages == 0
    assert seen == []
