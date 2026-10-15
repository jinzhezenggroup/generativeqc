"""CPKS evidence must survive scientific failure without replacing another receipt."""

from __future__ import annotations

import builtins
import json
import sys
import types
from pathlib import Path
from typing import Any

import pytest

from benchmarks import cpu_cpks_endpoint as runner


@pytest.fixture
def invocation(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    """Keep scientific owners out of these receipt-only execution controls."""
    package = types.ModuleType("generativeqc")
    package.__file__ = __file__
    package.response_solver = package
    monkeypatch.setitem(sys.modules, "generativeqc", package)
    library = tmp_path / "library.so"
    library.write_bytes(b"scientific double")
    monkeypatch.setenv("GENERATIVEQC_LIBRARY", str(library))
    output = tmp_path / "output.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "cpu_cpks_endpoint",
            "--output",
            str(output),
            "--source-sha",
            "a" * 40,
            "--methods",
            "lda-rks",
            "--repeats",
            "3",
            "--warmups",
            "1",
        ],
    )
    calls: list[Any] = []

    def endpoint(*args: Any, **kwargs: Any) -> dict[str, Any]:
        calls.append((args, kwargs))
        return {"seconds": 1.0, "solution": [float(len(calls))]}

    monkeypatch.setattr(runner, "endpoint", endpoint)
    return {"output": output, "calls": calls}


@pytest.mark.parametrize("kind", ["file", "directory", "symlink", "dangling"])
def test_existing_output_refuses_before_imports(
    invocation: dict[str, Any], monkeypatch: pytest.MonkeyPatch, kind: str
) -> None:
    output = invocation["output"]
    target = output.with_name("target.json")
    if kind == "file":
        output.write_text("preserved")
    elif kind == "directory":
        output.mkdir()
    else:
        if kind == "symlink":
            target.write_text("preserved")
        output.symlink_to(target)
    original_import = builtins.__import__

    def guard(name: str, *args: Any, **kwargs: Any) -> Any:
        if name == "generativeqc":
            pytest.fail("refused receipt reached science")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guard)
    with pytest.raises(FileExistsError):
        runner.main()
    assert not invocation["calls"]
    if kind == "file":
        assert output.read_text() == "preserved"
    elif kind == "symlink":
        assert target.read_text() == "preserved"
    elif kind == "dangling":
        assert not target.exists() and output.is_symlink()


def test_competing_creator_cannot_be_overwritten(
    invocation: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    output = invocation["output"]
    original_open = Path.open

    def race(path: Path, mode: str = "r", *args: Any, **kwargs: Any) -> Any:
        if path == output and mode == "x":
            with original_open(path, "w") as stream:
                stream.write("competitor")
        return original_open(path, mode, *args, **kwargs)

    monkeypatch.setattr(Path, "open", race)
    with pytest.raises(FileExistsError):
        runner.main()
    assert output.read_text() == "competitor"
    assert not invocation["calls"]


@pytest.mark.parametrize("failure_index", [0, 1, 3, 5])
def test_partial_results_and_failure_position_survive(
    invocation: dict[str, Any], monkeypatch: pytest.MonkeyPatch, failure_index: int
) -> None:
    original_endpoint = runner.endpoint
    failure = RuntimeError("operator failure")

    def fail(*args: Any, **kwargs: Any) -> dict[str, Any]:
        if len(invocation["calls"]) == failure_index:
            raise failure
        return original_endpoint(*args, **kwargs)

    monkeypatch.setattr(runner, "endpoint", fail)
    with pytest.raises(RuntimeError) as caught:
        runner.main()
    assert caught.value is failure
    receipt = json.loads(invocation["output"].read_text())
    assert receipt["status"] == "failed"
    assert receipt["error"]["message"] == "operator failure"
    assert receipt["active"]["case"] == (
        "lda-rks-moved" if failure_index == 5 else "lda-rks-original"
    )
    phases = {0: "first_use", 1: "warmup", 3: "sample", 5: "first_use"}
    assert receipt["active"]["phase"] == phases[failure_index]
    if failure_index == 3:
        assert receipt["active"]["index"] == 1
        assert len(receipt["cases"]["lda-rks-original"]["samples"]) == 1
    elif failure_index == 5:
        assert len(receipt["cases"]["lda-rks-original"]["samples"]) == 3


def test_replaced_path_is_not_reopened(
    invocation: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    output = invocation["output"]
    reserved = output.with_name("reserved.json")
    target = output.with_name("competitor.json")
    target.write_text("competitor")
    original_endpoint = runner.endpoint

    def replace(*args: Any, **kwargs: Any) -> dict[str, Any]:
        if not reserved.exists():
            output.rename(reserved)
            output.symlink_to(target)
        return original_endpoint(*args, **kwargs)

    monkeypatch.setattr(runner, "endpoint", replace)
    runner.main()
    assert target.read_text() == "competitor"
    receipt = json.loads(reserved.read_text())
    assert receipt["status"] == "complete" and "active" not in receipt
    assert len(receipt["cases"]) == 2
    assert all(len(row["samples"]) == 3 for row in receipt["cases"].values())


def test_journal_failure_preserves_original_exception(
    invocation: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    failure = RuntimeError("original")
    original_write = runner._write_receipt

    def write(stream: Any, receipt: dict[str, Any]) -> None:
        if receipt["status"] == "failed":
            raise OSError("unavailable journal")
        original_write(stream, receipt)

    def fail(*args: Any, **kwargs: Any) -> Any:
        raise failure

    monkeypatch.setattr(runner, "_write_receipt", write)
    monkeypatch.setattr(runner, "endpoint", fail)
    with pytest.raises(RuntimeError) as caught:
        runner.main()
    assert caught.value is failure


def test_nonfinite_result_leaves_a_valid_failure_receipt(
    invocation: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        runner, "endpoint", lambda *args, **kwargs: {"seconds": float("nan")}
    )
    with pytest.raises(ValueError, match="Out of range float"):
        runner.main()
    receipt = json.loads(invocation["output"].read_text())
    assert receipt["status"] == "failed"
    assert receipt["cases"]["lda-rks-original"]["samples"] == []
