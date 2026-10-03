"""The CC endpoint journal separates preparation and execution costs."""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any, Self

import pytest

from benchmarks import ccsdt_prepared_endpoint as benchmark


@dataclass
class _Diagnostics:
    calls: int = 1


def test_preparation_is_counted_once_and_outputs_are_retained(
    tmp_path: Any, monkeypatch: Any
) -> None:
    output = tmp_path / "run.json"
    library = tmp_path / "library.so"
    library.write_bytes(b"test-library-identity")
    monkeypatch.setenv("GENERATIVEQC_LIBRARY", str(library))
    monkeypatch.setattr(sys, "argv", ["benchmark", "--output", str(output)])
    monkeypatch.setattr(
        benchmark,
        "scaling_cases",
        lambda: {"water-3": SimpleNamespace(atoms=[(1, [0.0, 0.0, 0.0])])},
    )
    ticks = iter([0.0, 2.0, 10.0, 15.0, 20.0, 23.0, 30.0, 34.0, 40.0, 46.0])
    monkeypatch.setattr(benchmark, "perf_counter", lambda: next(ticks))

    class Prepared:
        def __enter__(self) -> Self:
            return self

        def __exit__(self, *_: object) -> None:
            pass

        def execute(self, **_: Any) -> Any:
            assert json.loads(output.read_text())["status"] == "running"
            return SimpleNamespace(
                items=[
                    SimpleNamespace(
                        energy=-1.0,
                        converged=True,
                        forces=None,
                        warm_start_used=False,
                        warm_start_fallback=False,
                        correlation=_Diagnostics(),
                        cc_performance=_Diagnostics(),
                    )
                ]
            )

    monkeypatch.setattr(
        benchmark,
        "Calculator",
        lambda **_: SimpleNamespace(prepare_batch=lambda _: Prepared()),
    )
    benchmark.main()
    result = json.loads(output.read_text())
    assert result["status"] == "measured"
    assert result["prepare_seconds"] == 2.0
    assert [row["seconds"] for row in result["rows"]] == [5.0, 3.0, 4.0, 6.0]
    assert [row["endpoint_seconds"] for row in result["rows"]] == [
        7.0,
        3.0,
        4.0,
        6.0,
    ]


def test_retained_output_is_rejected_before_runtime(
    monkeypatch: Any,
) -> None:
    monkeypatch.setattr(
        sys, "argv", ["benchmark", "--output", "benchmarks/results/forbidden.json"]
    )
    monkeypatch.setattr(
        benchmark,
        "scaling_cases",
        lambda: pytest.fail("output admission must precede runtime work"),
    )
    with pytest.raises(SystemExit) as error:
        benchmark.main()
    assert error.value.code == 2
