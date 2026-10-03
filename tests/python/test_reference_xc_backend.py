"""Reference provenance reads runtime flags, not installation/package guesses."""

import inspect
from types import SimpleNamespace

import pytest

from benchmarks.readme_omol25 import main, reference_xc_backend


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


def test_xc_probe_runs_after_the_complete_reference_endpoint_timer() -> None:
    source = inspect.getsource(main)
    begin = source.index("def execute_reference(")
    end = source.index("baseline = execute_reference(", begin)
    endpoint = source[begin:end]
    assert endpoint.index("seconds = perf_counter() - started") < endpoint.index(
        "reference_xc_backend(engine)"
    )
