"""Execution, exception and lifetime contracts of the selected HF controller."""

from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
from typing import Any

import pytest
from generativeqc import _native
from generativeqc._progressive_native import HFEvent, run_hf_schedule


@pytest.fixture
def library() -> Any:
    selected = _native.load_library()
    assert hasattr(selected, "generativeqc_progressive_hf_run_v1")
    return selected


def operation_log(
    source: int = 1,
    target: int = 3,
    retained: int = 1,
    failures: dict[HFEvent, BaseException] | None = None,
) -> tuple[Any, list[tuple[HFEvent, BaseException | None]], set[str]]:
    """Independent observable owner model; never execute scientific work."""
    events = []
    owners = set()
    failures = failures or {}

    def operation(event: HFEvent, exception_info: tuple) -> int:
        events.append((event, exception_info[1]))
        if event in (HFEvent.CLOSE_TARGET, HFEvent.CLOSE_SOURCE):
            owner = "target" if event == HFEvent.CLOSE_TARGET else "source"
            assert owner in owners
            owners.remove(owner)
        if event in failures:
            raise failures[event]
        if event == HFEvent.PREPARE_SOURCE:
            owners.add("source")
        elif event == HFEvent.PREPARE_TARGET:
            assert "source" in owners
            owners.add("target")
        elif event == HFEvent.PROJECT:
            assert owners == {"source", "target"}
        elif event == HFEvent.AUDIT_TARGET:
            assert not owners
        return {
            HFEvent.EXECUTE_SOURCE: source,
            HFEvent.EXECUTE_TARGET: target,
            HFEvent.RETAIN_TARGET: retained,
        }.get(event, 0)

    return operation, events, owners


@pytest.mark.parametrize("native", [True, False])
@pytest.mark.parametrize("source", [0, 1])
@pytest.mark.parametrize("target", [0, 1, 3])
@pytest.mark.parametrize("retained", [0, 1])
def test_success_does_not_substitute_source_or_skip_cleanup(
    library: Any, native: bool, source: int, target: int, retained: int
) -> None:
    operation, events, owners = operation_log(source, target, retained)
    owner = run_hf_schedule(library if native else SimpleNamespace(), operation)
    assert owner == ("native-v1" if native else "portable-compatibility")
    seen = [event for event, _ in events]
    assert HFEvent.EXECUTE_TARGET in seen
    assert (HFEvent.PROJECT in seen) == bool(source)
    assert (HFEvent.RETAIN_TARGET in seen) == (target == 3)
    assert (HFEvent.AUDIT_TARGET in seen) == (target == 3 and bool(retained))
    assert not owners


@pytest.mark.parametrize("native", [True, False])
@pytest.mark.parametrize("failure", list(HFEvent))
def test_each_callback_exception_cleans_prepared_owners_and_recovers(
    library: Any, native: bool, failure: HFEvent
) -> None:
    original = MemoryError(f"injected {failure.name}")
    operation, events, owners = operation_log(failures={failure: original})
    selected = library if native else SimpleNamespace()
    with pytest.raises(MemoryError) as raised:
        run_hf_schedule(selected, operation)
    assert raised.value is original
    assert not owners
    failed = False
    for event, prior in events:
        if failed and event in (HFEvent.CLOSE_TARGET, HFEvent.CLOSE_SOURCE):
            assert prior is original
        failed |= event == failure
    fresh, _, owners = operation_log()
    run_hf_schedule(selected, fresh)
    assert not owners


@pytest.mark.parametrize("native", [True, False])
def test_cleanup_failure_does_not_hide_first_exception(
    library: Any, native: bool
) -> None:
    original = KeyboardInterrupt("original cancellation")
    operation, _, owners = operation_log(
        failures={
            HFEvent.EXECUTE_TARGET: original,
            HFEvent.CLOSE_TARGET: RuntimeError("cleanup failure"),
        }
    )
    with pytest.raises(KeyboardInterrupt) as raised:
        run_hf_schedule(library if native else SimpleNamespace(), operation)
    assert raised.value is original
    assert not owners


def test_native_refusal_does_not_run_portable_schedule() -> None:
    class Refusal:
        def __call__(self, *_args: Any) -> int:
            return _native.STATUS_INVALID_ARGUMENT

    library = SimpleNamespace(
        generativeqc_progressive_hf_run_v1=Refusal(),
        generativeqc_status_message=lambda _status: b"injected refusal",
    )
    operation, events, owners = operation_log()
    with pytest.raises(RuntimeError, match="injected refusal"):
        run_hf_schedule(library, operation)
    assert events == [] and not owners


def test_native_empty_receipt_is_not_success() -> None:
    class InvalidReceipt:
        def __call__(self, *_args: Any) -> int:
            return 0

    operation, events, _ = operation_log()
    with pytest.raises(RuntimeError, match="inconsistent receipt"):
        run_hf_schedule(
            SimpleNamespace(generativeqc_progressive_hf_run_v1=InvalidReceipt()),
            operation,
        )
    assert not events


def test_reentrant_and_parallel_calls_have_independent_lifetimes(library: Any) -> None:
    def run(_index: int) -> None:
        operation, _, owners = operation_log()

        def reentrant(event: HFEvent, failure_info: tuple) -> int:
            if event == HFEvent.EXECUTE_TARGET:
                nested, _, nested_owners = operation_log(source=0)
                run_hf_schedule(library, nested)
                assert not nested_owners
            return operation(event, failure_info)

        run_hf_schedule(library, reentrant)
        assert not owners

    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(run, range(40)))
