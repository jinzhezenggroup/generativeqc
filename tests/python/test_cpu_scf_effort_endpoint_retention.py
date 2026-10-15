"""Receipt ownership and failure evidence must not depend on scientific success."""

from __future__ import annotations

import builtins
import dataclasses
import json
import sys
import types
from pathlib import Path
from typing import Any

import pytest

from benchmarks import cpu_scf_effort_endpoint as runner


@pytest.fixture
def invocation(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    """Exercise the CLI with scientific doubles and an independently owned receipt."""
    source = tmp_path / "science.py"
    source.write_text("scientific double\n", encoding="utf-8")
    library = tmp_path / "library.so"
    library.write_bytes(b"native library double")

    @dataclasses.dataclass
    class Estimator:
        scale: float = 1.0

        @property
        def identity(self) -> str:
            return "calibration-v1"

        @classmethod
        def fit(cls, *args: Any, **kwargs: Any) -> Estimator:
            return cls()

    @dataclasses.dataclass
    class Case:
        atoms: tuple[Any, ...] = (("H", (0.0, 0.0, 0.0)),)

    package = types.ModuleType("generativeqc")
    package.__file__ = str(source)
    package.ScfForceErrorEstimator = Estimator
    package.force_aware_scf = package
    harness = types.ModuleType("scf_effort_geomopt_benchmark")
    harness.__file__ = str(source)
    harness.cases = lambda: {"h2": Case(), "lih": Case()}
    harness.calibration_samples = lambda *args: ([], [{"calibration": True}])
    numerics = types.ModuleType("tools.generativeqc_numerics")
    numerics.scf_effort_geomopt_benchmark = harness
    monkeypatch.setitem(sys.modules, "generativeqc", package)
    monkeypatch.setitem(sys.modules, "tools.generativeqc_numerics", numerics)
    monkeypatch.setenv("GENERATIVEQC_LIBRARY", str(library))
    calibration = tmp_path / "calibration.json"
    calibration.write_text(
        json.dumps(
            {
                "kind": "measured-scf-effort-calibration-v1",
                "estimator": {"scale": 1.0},
                "calibration_id": "calibration-v1",
            }
        ),
        encoding="utf-8",
    )
    output = tmp_path / "output.json"
    arguments = [
        "cpu_scf_effort_endpoint",
        "--output",
        str(output),
        "--source-sha",
        "a" * 40,
        "--calibration",
        str(calibration),
        "--cases",
        "h2",
        "--repeats",
        "3",
        "--warmups",
        "1",
    ]
    monkeypatch.setattr(sys, "argv", arguments)
    calls: list[Any] = []

    def endpoint(*args: Any) -> dict[str, Any]:
        calls.append(args)
        return {"seconds": 1.0, "result": len(calls)}

    monkeypatch.setattr(runner, "endpoint", endpoint)
    return {"output": output, "arguments": arguments, "calls": calls}


@pytest.mark.parametrize("kind", ["file", "directory", "symlink", "dangling"])
def test_existing_destination_refuses_before_scientific_imports(
    invocation: dict[str, Any], monkeypatch: pytest.MonkeyPatch, kind: str
) -> None:
    output = invocation["output"]
    target = output.with_name("target.json")
    if kind == "file":
        output.write_text("original evidence")
    elif kind == "directory":
        output.mkdir()
    else:
        if kind == "symlink":
            target.write_text("original evidence")
        output.symlink_to(target)
    original_import = builtins.__import__

    def guarded_import(name: str, *args: Any, **kwargs: Any) -> Any:
        if name == "generativeqc":
            pytest.fail("refused output reached scientific imports")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)
    with pytest.raises(FileExistsError):
        runner.main()
    assert not invocation["calls"]
    if kind == "file":
        assert output.read_text() == "original evidence"
    elif kind == "symlink":
        assert target.read_text() == "original evidence"
    elif kind == "dangling":
        assert output.is_symlink() and not target.exists()


def test_competing_creator_wins_without_overwrite(
    invocation: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    original_open = Path.open
    output = invocation["output"]

    def race(path: Path, mode: str = "r", *args: Any, **kwargs: Any) -> Any:
        if path == output and mode == "x":
            with original_open(path, "w") as stream:
                stream.write("competing evidence")
        return original_open(path, mode, *args, **kwargs)

    monkeypatch.setattr(Path, "open", race)
    with pytest.raises(FileExistsError):
        runner.main()
    assert output.read_text() == "competing evidence"
    assert not invocation["calls"]


def test_replaced_path_never_redirects_publication(
    invocation: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    output = invocation["output"]
    retained = output.with_name("reserved.json")
    target = output.with_name("competing.json")
    target.write_text("competing evidence")
    original_endpoint = runner.endpoint

    def replace_once(*args: Any) -> dict[str, Any]:
        if not retained.exists():
            output.rename(retained)
            output.symlink_to(target)
        return original_endpoint(*args)

    monkeypatch.setattr(runner, "endpoint", replace_once)
    runner.main()
    assert target.read_text() == "competing evidence"
    assert json.loads(retained.read_text())["status"] == "complete"


@pytest.mark.parametrize("failure_index", [0, 1, 3, 5])
def test_failure_retains_prior_work_and_original_exception(
    invocation: dict[str, Any], monkeypatch: pytest.MonkeyPatch, failure_index: int
) -> None:
    failure = RuntimeError("scientific failure")
    original_endpoint = runner.endpoint

    def fail(*args: Any) -> dict[str, Any]:
        if len(invocation["calls"]) == failure_index:
            raise failure
        return original_endpoint(*args)

    monkeypatch.setattr(runner, "endpoint", fail)
    with pytest.raises(RuntimeError) as caught:
        runner.main()
    assert caught.value is failure
    receipt = json.loads(invocation["output"].read_text())
    assert receipt["status"] == "failed"
    assert receipt["error"] == {"type": "RuntimeError", "message": "scientific failure"}
    if failure_index == 0:
        assert receipt["active"] == {"case": "h2-original", "phase": "first_use"}
    elif failure_index == 1:
        assert receipt["active"] == {
            "case": "h2-original",
            "phase": "warmup",
            "index": 0,
        }
    elif failure_index == 3:
        assert receipt["active"] == {
            "case": "h2-original",
            "phase": "sample",
            "index": 1,
        }
        assert len(receipt["cases"]["h2-original"]["samples"]) == 1
    else:
        assert len(receipt["cases"]["h2-original"]["samples"]) == 3
        assert receipt["active"] == {"case": "h2-moved", "phase": "first_use"}


def test_journal_failure_does_not_mask_scientific_exception(
    invocation: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    failure = RuntimeError("original failure")
    original_write = runner._write_receipt

    def write(stream: Any, receipt: dict[str, Any]) -> None:
        if receipt["status"] == "failed":
            raise OSError("journal unavailable")
        original_write(stream, receipt)

    def fail(*args: Any) -> Any:
        raise failure

    monkeypatch.setattr(runner, "endpoint", fail)
    monkeypatch.setattr(runner, "_write_receipt", write)
    with pytest.raises(RuntimeError) as caught:
        runner.main()
    assert caught.value is failure


def test_success_retains_all_samples(invocation: dict[str, Any]) -> None:
    runner.main()
    receipt = json.loads(invocation["output"].read_text())
    assert receipt["status"] == "complete" and "active" not in receipt
    assert set(receipt["cases"]) == {"h2-original", "h2-moved"}
    for row in receipt["cases"].values():
        assert len(row["samples"]) == 3
        assert row["median_seconds"] == 1.0
        assert row["first_use"]["seconds"] == 1.0


def test_calibration_can_be_published(invocation: dict[str, Any]) -> None:
    arguments = invocation["arguments"]
    index = arguments.index("--calibration")
    del arguments[index : index + 2]
    runner.main()
    receipt = json.loads(invocation["output"].read_text())
    assert receipt["status"] == "complete"
    assert receipt["kind"] == "measured-scf-effort-calibration-v1"
    assert receipt["calibration_id"] == "calibration-v1"
    assert not invocation["calls"]


def test_retained_tree_is_refused_before_work(
    invocation: dict[str, Any], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        sys.modules[runner.raw_output_path.__module__], "_REPOSITORY_ROOT", tmp_path
    )
    invocation["arguments"][2] = str(tmp_path / "benchmarks/results/evidence.json")
    with pytest.raises(SystemExit) as caught:
        runner.main()
    assert caught.value.code == 2
    assert not (tmp_path / "benchmarks").exists()
    assert not invocation["calls"]


@pytest.mark.parametrize(
    ("option", "value"), [("--repeats", "0"), ("--warmups", "-1"), ("--steps", "0")]
)
def test_invalid_controls_do_not_reserve_output(
    invocation: dict[str, Any], option: str, value: str
) -> None:
    invocation["arguments"].extend([option, value])
    with pytest.raises(SystemExit) as caught:
        runner.main()
    assert caught.value.code == 2
    assert not invocation["output"].exists()
    assert not invocation["calls"]


def test_import_failure_is_recorded_without_endpoint_work(
    invocation: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    original_import = builtins.__import__
    failure = ImportError("scientific dependency unavailable")

    def fail_import(name: str, *args: Any, **kwargs: Any) -> Any:
        if name == "generativeqc":
            raise failure
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fail_import)
    with pytest.raises(ImportError) as caught:
        runner.main()
    assert caught.value is failure
    receipt = json.loads(invocation["output"].read_text())
    assert receipt["status"] == "failed"
    assert receipt["active"] == {"phase": "initialization"}
    assert not invocation["calls"]


def test_nonfinite_result_does_not_poison_failure_receipt(
    invocation: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(runner, "endpoint", lambda *args: {"seconds": float("nan")})
    with pytest.raises(ValueError, match="Out of range float"):
        runner.main()
    receipt = json.loads(invocation["output"].read_text())
    assert receipt["status"] == "failed"
    assert receipt["cases"]["h2-original"]["samples"] == []
    assert "first_use" not in receipt["cases"]["h2-original"]
