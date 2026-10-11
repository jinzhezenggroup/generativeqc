"""Reference provenance reads runtime flags, not installation/package guesses."""

from types import SimpleNamespace

import pytest

from benchmarks.readme_omol25 import reference_endpoint_seconds, reference_xc_backend


@pytest.mark.parametrize(
    "flags,expected",
    [
        ([True], "cuda-libxc"),
        ([True, True], "cuda-libxc"),
        ([True, False], "pyscf-cpu-libxc"),
        ([False], "pyscf-cpu-libxc"),
        ([None], "unknown"),
        ([True, None], "unknown"),
        ([], "no-semilocal-xc"),
    ],
)
def test_reference_runtime_xcfun_flags_determine_the_whole_expression_route(
    flags: list[bool | None],
    expected: str,
) -> None:
    calls = []

    def initialize(code: str, spin: int) -> list:
        calls.append((code, spin))
        return [
            (SimpleNamespace(func_id=400 + index, on_gpu=flag), 0.5)
            for index, flag in enumerate(flags)
        ]

    engine = SimpleNamespace(
        xc="PBE0", _numint=SimpleNamespace(_init_xcfuns=initialize)
    )
    result = reference_xc_backend(engine, spin=1)
    assert calls == [("PBE0", 1)]
    assert result["backend"] == expected
    assert [value["on_gpu"] for value in result["components"]] == flags
    assert all(value["coefficient"] == 0.5 for value in result["components"])


def test_missing_reference_interface_does_not_claim_cpu_fallback() -> None:
    result = reference_xc_backend(SimpleNamespace(xc="PBE0", _numint=object()))
    assert result["backend"] == "unknown"
    assert result["unavailable_reason"].startswith("AttributeError:")


def test_reference_clock_waits_for_device_before_stopping() -> None:
    events: list[str] = []

    def synchronize() -> None:
        events.append("synchronized")

    def clock() -> float:
        events.append("clock-read")
        return 13.5

    assert (
        reference_endpoint_seconds(
            started=4.0,
            synchronize=synchronize,
            clock=clock,
        )
        == 9.5
    )
    assert events == ["synchronized", "clock-read"]


def test_reference_clock_does_not_report_failed_synchronization() -> None:
    def synchronize() -> None:
        raise RuntimeError("device synchronization failed")

    with pytest.raises(RuntimeError, match="device synchronization failed"):
        reference_endpoint_seconds(
            started=4.0,
            synchronize=synchronize,
            clock=lambda: pytest.fail("clock read before completed work"),
        )
