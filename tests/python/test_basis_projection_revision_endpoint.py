"""Receipt admission and failure evidence without scientific endpoint execution."""

import errno
import importlib.util
import io
import json
import subprocess
import sys
import textwrap
import typing
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def runner(monkeypatch: typing.Any, tmp_path: typing.Any) -> typing.Any:
    spec = importlib.util.spec_from_file_location(
        "projection_endpoint_review",
        ROOT / "benchmarks/basis_projection_revision_endpoint.py",
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    binary = tmp_path / "library.so"
    binary.write_bytes(b"bounded test library double")
    monkeypatch.setenv("GENERATIVEQC_LIBRARY", str(binary))
    monkeypatch.setattr(module, "source_identity", lambda root: "matched-source")
    monkeypatch.setattr(
        module.ctypes,
        "CDLL",
        lambda path: SimpleNamespace(
            generativeqc_get_source_identity=lambda: b"matched-source"
        ),
    )
    monkeypatch.setattr(module, "cases", lambda molecules: iter([{"name": "bounded"}]))
    module.calls = []

    def endpoint(inputs: typing.Any) -> typing.Any:
        module.calls.append(inputs["name"])
        return {"seconds": 0.01, "scientific_double": True}

    monkeypatch.setattr(module, "endpoint", endpoint)
    return module


def invoke(
    monkeypatch: typing.Any, runner: typing.Any, output: typing.Any
) -> typing.Any:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "runner",
            "--source-sha",
            "3" * 40,
            "--repeats",
            "1",
            "--warmups",
            "0",
            "--output",
            str(output),
        ],
    )
    runner.main()


@pytest.mark.parametrize("kind", ["existing", "dangling_symlink", "file_parent"])
def test_invalid_output_precedes_scientific_work(
    monkeypatch: typing.Any, runner: typing.Any, tmp_path: typing.Any, kind: typing.Any
) -> None:
    output = tmp_path / "receipt.json"
    if kind == "existing":
        output.write_text("preserved")
    elif kind == "dangling_symlink":
        output.symlink_to(tmp_path / "missing-target")
    else:
        parent = tmp_path / "regular-file"
        parent.write_text("preserved")
        output = parent / "receipt.json"
    with pytest.raises((SystemExit, OSError)):
        invoke(monkeypatch, runner, output)
    assert runner.calls == []
    if kind == "existing":
        assert output.read_text() == "preserved"
    elif kind == "dangling_symlink":
        assert output.is_symlink() and not (tmp_path / "missing-target").exists()
    else:
        assert parent.read_text() == "preserved"


def test_failure_receipt_retains_completed_work(
    monkeypatch: typing.Any, runner: typing.Any, tmp_path: typing.Any
) -> None:
    output = tmp_path / "failed.json"

    def endpoint(inputs: typing.Any) -> typing.Any:
        runner.calls.append(inputs["name"])
        if len(runner.calls) == 2:
            raise RuntimeError("controlled endpoint failure")
        return {"seconds": 0.01, "scientific_double": True}

    monkeypatch.setattr(runner, "endpoint", endpoint)
    with pytest.raises(RuntimeError, match="controlled endpoint failure"):
        invoke(monkeypatch, runner, output)
    assert output.exists()
    receipt = json.loads(output.read_text())
    assert receipt["status"] == "failed"
    assert receipt["cases"][0]["first_use"]["scientific_double"]
    assert "controlled endpoint failure" in receipt["error"]["message"]


def test_complete_receipt_keeps_samples_and_provenance(
    monkeypatch: typing.Any, runner: typing.Any, tmp_path: typing.Any
) -> None:
    output = tmp_path / "complete.json"
    invoke(monkeypatch, runner, output)
    receipt = json.loads(output.read_text())
    assert receipt["status"] == "complete"
    assert receipt["source_identity"] == "matched-source"
    assert receipt["binary_sha256"] == runner.file_hash(tmp_path / "library.so")
    assert receipt["runner_sha256"] == runner.file_hash(Path(runner.__file__))
    assert receipt["cases"][0]["first_use"]["seconds"] == 0.01
    assert receipt["cases"][0]["samples"] == [
        {"seconds": 0.01, "scientific_double": True}
    ]
    assert receipt["cases"][0]["median_seconds"] == 0.01
    assert runner.calls == ["bounded", "bounded"]


def test_source_mismatch_is_journaled_before_work(
    monkeypatch: typing.Any, runner: typing.Any, tmp_path: typing.Any
) -> None:
    monkeypatch.setattr(runner, "source_identity", lambda root: "wrong-source")
    output = tmp_path / "mismatch.json"
    with pytest.raises(RuntimeError, match="does not match"):
        invoke(monkeypatch, runner, output)
    assert runner.calls == []
    receipt = json.loads(output.read_text())
    assert receipt["status"] == "failed"
    assert receipt["cases"] == []
    assert "does not match" in receipt["error"]["message"]


def test_output_path_replacement_cannot_redirect_writes(
    monkeypatch: typing.Any, runner: typing.Any, tmp_path: typing.Any
) -> None:
    output, owned = tmp_path / "receipt.json", tmp_path / "owned.json"
    foreign = tmp_path / "foreign.json"
    foreign.write_text("preserved foreign receipt")
    original = runner.endpoint

    def endpoint(inputs: typing.Any) -> typing.Any:
        if not runner.calls:
            assert output.is_file()
            output.rename(owned)
            output.symlink_to(foreign)
        return original(inputs)

    monkeypatch.setattr(runner, "endpoint", endpoint)
    invoke(monkeypatch, runner, output)
    assert foreign.read_text() == "preserved foreign receipt"
    assert json.loads(owned.read_text())["status"] == "complete"


def test_raced_output_creation_stops_before_work(
    monkeypatch: typing.Any, runner: typing.Any, tmp_path: typing.Any
) -> None:
    output = tmp_path / "raced.json"
    original_open = Path.open

    def open_path(
        path: typing.Any,
        mode: typing.Any = "r",
        *args: typing.Any,
        **kwargs: typing.Any,
    ) -> typing.Any:
        if path == output and mode == "x":
            with original_open(path, "w") as stream:
                stream.write("racing owner")
        return original_open(path, mode, *args, **kwargs)

    monkeypatch.setattr(Path, "open", open_path)
    with pytest.raises(FileExistsError):
        invoke(monkeypatch, runner, output)
    assert runner.calls == []
    assert output.read_text() == "racing owner"


def test_nonfinite_record_keeps_failure_journal(
    monkeypatch: typing.Any, runner: typing.Any, tmp_path: typing.Any
) -> None:
    output = tmp_path / "nonfinite.json"
    original = runner.endpoint

    def endpoint(inputs: typing.Any) -> typing.Any:
        publication = original(inputs)
        if len(runner.calls) == 2:
            publication["energy"] = float("nan")
        return publication

    monkeypatch.setattr(runner, "endpoint", endpoint)
    with pytest.raises(ValueError, match="JSON"):
        invoke(monkeypatch, runner, output)
    receipt = json.loads(output.read_text())
    assert receipt["status"] == "failed"
    assert receipt["cases"][0]["first_use"]["scientific_double"]
    assert receipt["cases"][0]["samples"] == []


@pytest.mark.parametrize("error_type", [RuntimeError, KeyboardInterrupt])
def test_buffered_journal_close_preserves_original_exception(
    monkeypatch: typing.Any,
    runner: typing.Any,
    tmp_path: typing.Any,
    error_type: type[BaseException],
) -> None:
    """Real buffered writes can fail again when the owned stream closes."""
    output = tmp_path / "failed.json"
    failure = error_type("original scientific failure")
    original_open = Path.open
    sink = io.BytesIO()

    class FailingRaw(io.RawIOBase):
        fail = False
        failed_writes = 0

        def writable(self) -> bool:
            return True

        def seekable(self) -> bool:
            return True

        def seek(self, offset: int, whence: int = 0) -> int:
            return sink.seek(offset, whence)

        def tell(self) -> int:
            return sink.tell()

        def truncate(self, size: int | None = None) -> int:
            return sink.truncate(size)

        def write(self, data: bytes) -> int:
            if self.fail:
                self.failed_writes += 1
                raise OSError(errno.ENOSPC, "controlled journal device full")
            return sink.write(data)

    raw = FailingRaw()
    stream = io.TextIOWrapper(io.BufferedWriter(raw), encoding="utf-8")

    def open_receipt(
        path: Path, mode: str = "r", *args: typing.Any, **kwargs: typing.Any
    ) -> typing.Any:
        if path == output and mode == "x":
            return stream
        return original_open(path, mode, *args, **kwargs)

    def fail(inputs: typing.Any) -> typing.Any:
        assert json.loads(sink.getvalue())["status"] == "running"
        raw.fail = True
        raise failure

    monkeypatch.setattr(Path, "open", open_receipt)
    monkeypatch.setattr(runner, "endpoint", fail)
    with pytest.raises(error_type) as caught:
        invoke(monkeypatch, runner, output)
    assert caught.value is failure
    assert isinstance(caught.value.__cause__, OSError)
    assert stream.closed


def test_successful_science_surfaces_receipt_close_failure(
    monkeypatch: typing.Any, runner: typing.Any, tmp_path: typing.Any
) -> None:
    """A close failure must propagate when no earlier scientific error is active."""
    output = tmp_path / "complete.json"
    failure = OSError(errno.EIO, "receipt close failed")
    original_open = Path.open

    class CloseFailure(io.TextIOWrapper):
        def close(self) -> None:
            if not self.closed:
                super().close()
                raise failure

    def open_receipt(
        path: Path, mode: str = "r", *args: typing.Any, **kwargs: typing.Any
    ) -> typing.Any:
        if path == output and mode == "x":
            return CloseFailure(
                original_open(path, "xb", buffering=0), encoding="utf-8"
            )
        return original_open(path, mode, *args, **kwargs)

    monkeypatch.setattr(Path, "open", open_receipt)
    with pytest.raises(OSError) as caught:
        invoke(monkeypatch, runner, output)
    assert caught.value is failure
    assert json.loads(output.read_text())["status"] == "complete"
    assert runner.calls == ["bounded", "bounded"]


def test_os_write_limit_preserves_original_exception(
    runner: typing.Any, tmp_path: Path
) -> None:
    """Constrain only a child process to exercise an actual filesystem failure."""
    resource = pytest.importorskip("resource")
    if not hasattr(resource, "RLIMIT_FSIZE"):
        pytest.skip("requires a per-process file-size limit")
    script = textwrap.dedent(
        """
        import importlib.util
        import json
        import os
        from pathlib import Path
        import resource
        import signal
        import sys
        from types import SimpleNamespace

        source, destination = sys.argv[1:]
        spec = importlib.util.spec_from_file_location("receipt_runner", source)
        runner = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(runner)
        output = Path(destination)
        failure = RuntimeError("original scientific failure")
        runner.ctypes.CDLL = lambda path: SimpleNamespace(
            generativeqc_get_source_identity=lambda: b"matched-source")
        runner.source_identity = lambda root: "matched-source"
        runner.file_hash = lambda path: "bounded-test-hash"
        runner.cases = lambda molecules: iter([{"name": "bounded"}])
        os.environ["GENERATIVEQC_LIBRARY"] = source
        sys.argv = ["runner", "--source-sha", "f" * 40,
                    "--output", destination, "--repeats", "1", "--warmups", "0"]

        def fail(inputs):
            initial_size = output.stat().st_size
            signal.signal(signal.SIGXFSZ, signal.SIG_IGN)
            resource.setrlimit(resource.RLIMIT_FSIZE, (initial_size, initial_size))
            raise failure

        runner.endpoint = fail
        try:
            runner.main()
        except BaseException as caught:
            print(json.dumps({"preserved": caught is failure,
                              "type": type(caught).__name__}))
            sys.exit(0 if caught is failure else 1)
        raise AssertionError("expected scientific failure")
        """
    )
    result = subprocess.run(
        [sys.executable, "-c", script, runner.__file__, str(tmp_path / "receipt.json")],
        capture_output=True,
        text=True,
        timeout=20,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert json.loads(result.stdout) == {"preserved": True, "type": "RuntimeError"}
