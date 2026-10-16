"""Host-only lifecycle controls for the optional reference diagnostic journal."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from typing import TYPE_CHECKING

import pytest
from benchmarks._retention import raw_output_path

if TYPE_CHECKING:
    from collections.abc import Callable
    from types import ModuleType
    from typing import Any, TextIO


@pytest.fixture
def diagnostic(tmp_path: Path) -> ModuleType:
    path = (
        Path(__file__).resolve().parents[2]
        / "benchmarks/results/wb97mv-hessian-native-full-20261016/reference-selfcheck.py"
    )
    spec = importlib.util.spec_from_file_location("reference_selfcheck", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.raw_output_path = lambda path: raw_output_path(
        path, repository_root=tmp_path
    )
    return module


@pytest.mark.parametrize("existing_parent", [False, True])
def test_new_journal_precedes_work_and_completes(
    diagnostic: ModuleType, tmp_path: Path, existing_parent: bool
) -> None:
    output = tmp_path / ".artifacts" / "selfcheck.json"
    if existing_parent:
        output.parent.mkdir()
    report = {"cases": {}}

    def work(record: dict, save: Callable[[], None]) -> None:
        assert json.loads(output.read_text())["status"] == "running"
        record["cases"]["control"] = {"value": 1.0}
        save()
        assert json.loads(output.read_text())["cases"] == record["cases"]

    diagnostic.run(output, report, work)
    assert json.loads(output.read_text()) == {
        "cases": {"control": {"value": 1.0}},
        "status": "completed",
    }


@pytest.mark.parametrize(
    "kind", ["file", "directory", "hardlink", "symlink", "dangling_symlink"]
)
def test_existing_destinations_are_preserved_before_work(
    diagnostic: ModuleType, tmp_path: Path, kind: str
) -> None:
    output = tmp_path / "selfcheck.json"
    target = tmp_path / "old-evidence.json"
    target.write_text("preserve old evidence")
    if kind == "file":
        output.write_text("preserve output")
    elif kind == "directory":
        output.mkdir()
    elif kind == "hardlink":
        output.hardlink_to(target)
    elif kind == "symlink":
        output.symlink_to(target)
    else:
        output.symlink_to(tmp_path / "missing-target")

    def forbidden_work(*args: Any) -> None:
        pytest.fail("numerical work must not start with an occupied destination")

    with pytest.raises(OSError):
        diagnostic.run(output, {}, forbidden_work)
    assert target.read_text() == "preserve old evidence"
    if kind == "file":
        assert output.read_text() == "preserve output"
    if "symlink" in kind:
        assert output.is_symlink()
    assert not (tmp_path / "missing-target").exists()


@pytest.mark.parametrize("kind", ["file", "symlink", "dangling_symlink"])
def test_parent_blockers_reject_before_work(
    diagnostic: ModuleType, tmp_path: Path, kind: str
) -> None:
    parent = tmp_path / ".artifacts"
    target = tmp_path / "elsewhere"
    if kind == "file":
        parent.write_text("keep parent blocker")
    else:
        if kind == "symlink":
            target.mkdir()
        parent.symlink_to(target, target_is_directory=True)

    def forbidden_work(*args: Any) -> None:
        pytest.fail("numerical work must not start with an invalid parent")

    with pytest.raises(OSError):
        diagnostic.run(parent / "selfcheck.json", {}, forbidden_work)
    if kind == "file":
        assert parent.read_text() == "keep parent blocker"
    else:
        assert parent.is_symlink()
        assert not (target / "selfcheck.json").exists()


@pytest.mark.parametrize("alias", [False, True])
def test_retained_results_and_aliases_are_rejected(
    diagnostic: ModuleType, tmp_path: Path, alias: bool
) -> None:
    retained = tmp_path / "benchmarks" / "results"
    retained.mkdir(parents=True)
    parent = retained
    if alias:
        parent = tmp_path / ".artifacts"
        parent.symlink_to(retained, target_is_directory=True)

    def forbidden_work(*args: Any) -> None:
        pytest.fail("raw output must not enter reviewed results")

    with pytest.raises(ValueError, match="raw benchmark output"):
        diagnostic.run(parent / "selfcheck.json", {}, forbidden_work)
    assert not (retained / "selfcheck.json").exists()


@pytest.mark.parametrize("replacement_kind", ["file", "symlink"])
def test_path_replacement_does_not_redirect_owned_updates(
    diagnostic: ModuleType, tmp_path: Path, replacement_kind: str
) -> None:
    output = tmp_path / "selfcheck.json"
    owned = tmp_path / "reserved-inode.json"

    def work(report: dict, save: Callable[[], None]) -> None:
        output.rename(owned)
        replacement = tmp_path / "unrelated.json"
        replacement.write_text("unrelated replacement")
        if replacement_kind == "symlink":
            output.symlink_to(replacement)
        else:
            output.write_text("unrelated replacement")
        report["result"] = 7
        save()

    diagnostic.run(output, {}, work)
    assert output.read_text() == "unrelated replacement"
    assert json.loads(owned.read_text()) == {"status": "completed", "result": 7}


def test_serialization_failure_preserves_previous_journal(
    diagnostic: ModuleType, tmp_path: Path
) -> None:
    output = tmp_path / "selfcheck.json"

    def work(report: dict, save: Callable[[], None]) -> None:
        report["nonfinite"] = float("nan")
        save()

    with pytest.raises(ValueError, match="JSON compliant"):
        diagnostic.run(output, {}, work)
    assert json.loads(output.read_text()) == {"status": "running"}


class FailingStream:
    def __init__(
        self, stream: TextIO, close_error: BaseException | None = None
    ) -> None:
        self.stream = stream
        self.write_error = None
        self.close_error = close_error

    def __getattr__(self, name: str) -> Any:
        return getattr(self.stream, name)

    def write(self, value: str) -> int:
        if self.write_error is not None:
            raise self.write_error
        return self.stream.write(value)

    def close(self) -> None:
        self.stream.close()
        if self.close_error is not None:
            raise self.close_error


def intercept_stream(
    monkeypatch: pytest.MonkeyPatch,
    output: Path,
    close_error: BaseException | None = None,
    initial_write_error: BaseException | None = None,
) -> list[FailingStream]:
    original_open = Path.open
    streams = []

    def open_path(path: Path, *args: Any, **kwargs: Any) -> Any:
        stream = original_open(path, *args, **kwargs)
        if path == output and args and args[0] == "x+":
            wrapped = FailingStream(stream, close_error)
            wrapped.write_error = initial_write_error
            streams.append(wrapped)
            return wrapped
        return stream

    monkeypatch.setattr(Path, "open", open_path)
    return streams


@pytest.mark.parametrize("error_type", [RuntimeError, KeyboardInterrupt, SystemExit])
def test_primary_failure_survives_journal_and_close_errors(
    diagnostic: ModuleType,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    error_type: type[BaseException],
) -> None:
    output = tmp_path / "selfcheck.json"
    primary = error_type("original failure")
    streams = intercept_stream(monkeypatch, output, OSError("close failed"))

    def work(report: dict, save: Callable[[], None]) -> None:
        streams[-1].write_error = OSError("journal failed")
        raise primary

    with pytest.raises(error_type) as raised:
        diagnostic.run(output, {}, work)
    assert raised.value is primary
    assert streams[-1].stream.closed


def test_initial_journal_failure_prevents_work_and_retains_primary(
    diagnostic: ModuleType, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = tmp_path / "selfcheck.json"
    primary = OSError("initial journal failed")
    streams = intercept_stream(monkeypatch, output, OSError("close failed"), primary)

    def forbidden_work(*args: Any) -> None:
        pytest.fail("work must wait for the initial receipt")

    with pytest.raises(OSError) as raised:
        diagnostic.run(output, {}, forbidden_work)
    assert raised.value is primary
    assert streams[-1].stream.closed


def test_successful_work_does_not_hide_close_failure(
    diagnostic: ModuleType, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = tmp_path / "selfcheck.json"
    close_error = OSError("close failed")
    streams = intercept_stream(monkeypatch, output, close_error)
    with pytest.raises(OSError) as raised:
        diagnostic.run(output, {}, lambda report, save: None)
    assert raised.value is close_error
    assert streams[-1].stream.closed
    assert json.loads(output.read_text())["status"] == "completed"


def test_scientific_failure_is_retained_and_reraised(
    diagnostic: ModuleType, tmp_path: Path
) -> None:
    output = tmp_path / "selfcheck.json"
    primary = RuntimeError("SCF did not converge")

    def work(report: dict, save: Callable[[], None]) -> None:
        raise primary

    with pytest.raises(RuntimeError) as raised:
        diagnostic.run(output, {}, work)
    assert raised.value is primary
    assert json.loads(output.read_text()) == {
        "status": "failed",
        "failure": "RuntimeError: SCF did not converge",
    }


def test_symlink_created_during_reservation_cannot_be_followed(
    diagnostic: ModuleType, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = tmp_path / "selfcheck.json"
    target = tmp_path / "old-evidence.json"
    target.write_text("do not truncate")
    original_open = Path.open

    def raced_open(path: Path, *args: Any, **kwargs: Any) -> Any:
        if path == output and args and args[0] == "x+":
            output.symlink_to(target)
        return original_open(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", raced_open)
    with pytest.raises(FileExistsError):
        diagnostic.run(output, {}, lambda *args: pytest.fail("work started"))
    assert target.read_text() == "do not truncate"
    assert output.is_symlink()
