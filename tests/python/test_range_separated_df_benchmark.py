"""Live endpoint journals retain old evidence and identify the actual binary."""

from __future__ import annotations

import hashlib
import json
import os
import typing
from types import SimpleNamespace

import pytest

from benchmarks import range_separated_df as runner

if typing.TYPE_CHECKING:
    from pathlib import Path


def arguments(output: Path, *extra: str) -> list[str]:
    return [
        "--device",
        "cpu",
        "--selector",
        "df",
        "--endpoint",
        "energy",
        "--build-boundary",
        "controlled test",
        "--source-head",
        "test-source",
        "--output",
        str(output),
        *extra,
    ]


@pytest.mark.parametrize("kind", ["file", "directory", "symlink", "dangling-symlink"])
def test_existing_outputs_are_rejected_before_preparation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, kind: str
) -> None:
    output = tmp_path / "output"
    target = tmp_path / "target"
    if kind == "file":
        output.write_bytes(b"prior negative receipt")
    elif kind == "directory":
        output.mkdir()
        (output / "receipt").write_bytes(b"prior negative receipt")
    else:
        if kind == "symlink":
            target.write_bytes(b"prior negative receipt")
        output.symlink_to(target)

    def forbidden(*args: typing.Any) -> None:
        pytest.fail("preparation reached before output reservation")

    monkeypatch.setattr(runner, "_measure", forbidden)
    with pytest.raises(FileExistsError):
        runner.main(arguments(output))
    if kind == "file":
        assert output.read_bytes() == b"prior negative receipt"
    elif kind == "directory":
        assert (output / "receipt").read_bytes() == b"prior negative receipt"
    elif kind == "symlink":
        assert output.is_symlink() and target.read_bytes() == b"prior negative receipt"
    else:
        assert output.is_symlink() and (not target.exists())


@pytest.mark.parametrize("error", [ImportError, RuntimeError, KeyboardInterrupt])
@pytest.mark.parametrize("previous_selector", [None, "inherited"])
def test_initialization_failures_keep_owned_journal(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    error: typing.Any,
    previous_selector: str | None,
) -> None:
    set_previous_selector(monkeypatch, previous_selector)
    output = tmp_path / "new.json"

    def failed(
        arguments: typing.Any, payload: typing.Any, save: typing.Any
    ) -> typing.Any:
        assert json.loads(output.read_text())["stage"] == "initialization"
        assert os.environ["GENERATIVEQC_DF_RANGE_EXCHANGE"] == "df"
        raise error("initialization failed")

    monkeypatch.setattr(runner, "_measure", failed)
    with pytest.raises(error, match="initialization failed"):
        runner.main(arguments(output))
    assert os.environ.get("GENERATIVEQC_DF_RANGE_EXCHANGE") == previous_selector
    receipt = json.loads(output.read_text())
    assert receipt["stage"] == "failed"
    assert receipt["failure"] == {
        "stage": "initialization",
        "type": error.__name__,
        "message": "initialization failed",
    }
    assert receipt["samples"] == []


def set_previous_selector(monkeypatch: pytest.MonkeyPatch, value: str | None) -> None:
    """Exercise both an absent override and exact restoration of caller state."""
    if value is None:
        monkeypatch.delenv("GENERATIVEQC_DF_RANGE_EXCHANGE", raising=False)
    else:
        monkeypatch.setenv("GENERATIVEQC_DF_RANGE_EXCHANGE", value)


def install_endpoint(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    *,
    preparation_error: bool = False,
    runtime_error: bool = False,
) -> typing.Any:
    import generativeqc
    import generativeqc._ks_snapshot
    import numpy as np

    loaded = tmp_path / "selected-profile.so"
    loaded.write_bytes(b"actual loaded library")
    requested = tmp_path / "environment-library.so"
    requested.write_bytes(b"different environment library")
    monkeypatch.setenv("GENERATIVEQC_LIBRARY", str(requested))
    state = {"clock": 0.0, "executions": [], "calculator": None}
    monkeypatch.setattr(runner.time, "perf_counter", lambda: state["clock"])
    identity = runner.loaded_library_identity

    def expensive_hash(calculator: typing.Any) -> typing.Any:
        state["clock"] += 100.0
        return identity(calculator)

    monkeypatch.setattr(runner, "loaded_library_identity", expensive_hash)

    class Batch:
        def __enter__(self) -> typing.Any:
            state["clock"] += 3.0
            if preparation_error:
                raise RuntimeError("preparation failed")
            return self

        def __exit__(self, *args: object) -> typing.Any:
            return False

        def _public_dft_cpu_force(
            self, index: typing.Any, geometry: typing.Any
        ) -> typing.Any:
            return (None, {"nonlocal_pair_evaluations": 26873856})

        def execute(self, **kwargs: typing.Any) -> typing.Any:
            if runtime_error and state["executions"]:
                raise RuntimeError("runtime failed")
            state["executions"].append(kwargs)
            state["clock"] += 4.0
            forces = None
            if "forces" in kwargs["properties"]:
                self._public_dft_cpu_force(0, None)
                forces = np.zeros((3, 3))
            return SimpleNamespace(
                items=[
                    SimpleNamespace(
                        converged=True,
                        energy=-1.0,
                        executed_backend="cpu_reference",
                        iterations=2,
                        density_rms=0.0,
                        energy_change=0.0,
                        basis_metadata={},
                        forces=forces,
                    )
                ]
            )

    class Calculator:
        def __init__(self, **kwargs: typing.Any) -> None:
            state["clock"] += 2.0
            self._library = SimpleNamespace(_name=str(loaded))
            state["calculator"] = kwargs

        def prepare_batch(self, *args: typing.Any, **kwargs: typing.Any) -> typing.Any:
            return Batch()

    class Snapshot:
        def __init__(self, *args: typing.Any) -> None:
            self.density_fitted = True
            self.range_density_fitted = (
                os.environ["GENERATIVEQC_DF_RANGE_EXCHANGE"] == "df"
            )
            self.metadata = [0, 24, 0, 0, 0, 5184]

        def close(self) -> None:
            pass

    monkeypatch.setattr(generativeqc, "Calculator", Calculator)
    monkeypatch.setattr(generativeqc._ks_snapshot, "NativeKsSnapshot", Snapshot)
    return (state, loaded)


@pytest.mark.parametrize("selector", ["direct", "df"])
@pytest.mark.parametrize("endpoint", ["energy", "forces"])
@pytest.mark.parametrize("previous_selector", [None, "direct", "df", "inherited"])
def test_complete_workload_and_loaded_identity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    selector: str,
    endpoint: str,
    previous_selector: str | None,
) -> None:
    set_previous_selector(monkeypatch, previous_selector)
    state, loaded = install_endpoint(monkeypatch, tmp_path)
    output = tmp_path / "new.json"
    runner.main(arguments(output, "--selector", selector, "--endpoint", endpoint))
    assert os.environ.get("GENERATIVEQC_DF_RANGE_EXCHANGE") == previous_selector
    receipt = json.loads(output.read_text())
    assert receipt["stage"] == "complete" and "failure" not in receipt
    assert receipt["library"] == str(loaded)
    assert receipt["library_sha256"] == hashlib.sha256(loaded.read_bytes()).hexdigest()
    assert receipt["prepare_seconds"] == 5.0
    assert [s["label"] for s in receipt["samples"]] == [
        "cold",
        "warm-0",
        "warm-1",
        "warm-2",
        "geometry-0",
        "geometry-1",
        "geometry-2",
    ]
    assert all(s["seconds"] == 4.0 for s in receipt["samples"])
    assert all(
        s["range_density_fitted"] == (selector == "df") for s in receipt["samples"]
    )
    assert all(
        (s["forces"] is not None) == (endpoint == "forces") for s in receipt["samples"]
    )
    assert all(call["strict"] for call in state["executions"])
    import numpy as np

    original = np.array([p for _, p in runner.WATER])
    moved = original.copy()
    moved[1] += (0.03, -0.02, 0.04)
    for index, expected in zip((4, 5, 6), (moved, original, moved), strict=True):
        np.testing.assert_array_equal(
            state["executions"][index]["coordinates"][0], expected
        )
    settings = state["calculator"]
    assert settings["energy_tolerance"] == 1e-10
    assert settings["density_tolerance"] == 1e-08
    assert settings["density_fitting_relative_threshold"] == 1e-10
    assert settings["screening_tolerance"] == 1e-14
    assert settings["max_iterations"] == 200


@pytest.mark.parametrize("phase", ["preparation", "runtime"])
@pytest.mark.parametrize("selector", ["direct", "df"])
@pytest.mark.parametrize("previous_selector", [None, "direct", "df", "inherited"])
def test_failures_retain_actual_stage_binary_and_prior_samples(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    phase: str,
    selector: str,
    previous_selector: str | None,
) -> None:
    set_previous_selector(monkeypatch, previous_selector)
    _state, loaded = install_endpoint(
        monkeypatch,
        tmp_path,
        preparation_error=phase == "preparation",
        runtime_error=phase == "runtime",
    )
    output = tmp_path / "new.json"
    with pytest.raises(RuntimeError, match=phase + " failed"):
        runner.main(arguments(output, "--selector", selector))
    assert os.environ.get("GENERATIVEQC_DF_RANGE_EXCHANGE") == previous_selector
    receipt = json.loads(output.read_text())
    assert receipt["stage"] == "failed"
    assert receipt["library"] == str(loaded)
    assert receipt["failure"]["stage"] == (
        "preparation" if phase == "preparation" else "warm-0"
    )
    assert [s["label"] for s in receipt["samples"]] == (
        [] if phase == "preparation" else ["cold"]
    )


def test_cuda_requires_scheduler_before_creation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("SLURM_JOB_ID", raising=False)
    output = tmp_path / "new.json"
    with pytest.raises(SystemExit):
        runner.main(arguments(output, "--device", "cuda"))
    assert not output.exists()


def campaign_shell_fixture(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, body: str, *, with_venv: bool
) -> tuple[Path, Path, Path, Path]:
    """Isolate the shell runner from every real checkout/interpreter/campaign."""
    import os
    import shlex
    from pathlib import Path

    root = Path(__file__).resolve().parents[2]
    checkout = tmp_path / "isolated-checkout"
    checkout.mkdir()
    commands = tmp_path / "commands"
    commands.mkdir()
    git = commands / "git"
    git.write_text(
        '#!/bin/sh\ncase "$*" in\n'
        f"'rev-parse --show-toplevel') printf '%s\\n' {shlex.quote(str(checkout))};;\n"
        "'rev-parse HEAD') printf '%040d\\n' 0;;\n"
        "*) exit 97;;\nesac\n"
    )
    git.chmod(0o755)
    trace = tmp_path / "interpreter-trace"
    interpreter = commands / "python"
    if with_venv:
        # The real runner prepends .venv/bin. A PATH-only stub is unsafe when
        # running the test inside an actual development/CI virtual environment.
        interpreter.write_text("#!/bin/sh\necho unexpected-PATH-python >&2\nexit 97\n")
        interpreter.chmod(0o755)
        interpreter = checkout / ".venv/bin/python"
        interpreter.parent.mkdir(parents=True)
    interpreter.write_text(
        f'#!/bin/sh\nprintf "%s\\n" "$0" >> {shlex.quote(str(trace))}\n' + body
    )
    interpreter.chmod(0o755)
    monkeypatch.setenv("PATH", str(commands) + os.pathsep + os.environ["PATH"])
    return (
        root / "benchmarks/run_range_separated_df_cpu.sh",
        checkout,
        trace,
        interpreter,
    )


@pytest.mark.parametrize("with_venv", (False, True))
@pytest.mark.parametrize("kind", ["file", "directory", "symlink", "dangling-symlink"])
def test_campaign_rejects_existing_output_before_start(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, kind: str, with_venv: bool
) -> None:
    import subprocess

    output = tmp_path / "campaign"
    target = tmp_path / "target"
    if kind == "file":
        output.write_bytes(b"prior receipt")
    elif kind == "directory":
        output.mkdir()
        (output / "receipt").write_bytes(b"prior receipt")
    else:
        if kind == "symlink":
            target.mkdir()
            (target / "receipt").write_bytes(b"prior receipt")
        output.symlink_to(target)
    marker = tmp_path / "launched"
    script, checkout, trace, _ = campaign_shell_fixture(
        tmp_path,
        monkeypatch,
        f'touch "{marker}"\nexit 42\n',
        with_venv=with_venv,
    )
    monkeypatch.setenv("RANGE_DF_EVIDENCE_DIRECTORY", str(output))
    monkeypatch.setenv("RANGE_DF_HARNESS_DIRECTORY", str(tmp_path))
    result = subprocess.run(
        ["bash", str(script)],
        cwd=checkout,
        capture_output=True,
        check=False,
        timeout=10,
    )
    assert result.returncode != 0 and not marker.exists() and not trace.exists()
    if kind == "file":
        assert output.read_bytes() == b"prior receipt"
    elif kind in {"directory", "symlink"}:
        assert (output / "receipt").read_bytes() == b"prior receipt"
    else:
        assert output.is_symlink() and not target.exists()


@pytest.mark.parametrize("with_venv", (False, True))
def test_campaign_keeps_failed_oracle_receipt_log_and_stage(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, with_venv: bool
) -> None:
    import subprocess

    script, checkout, trace, interpreter = campaign_shell_fixture(
        tmp_path,
        monkeypatch,
        'case "$*" in\n*oracle-all.py*)\n'
        '  echo "independent oracle rejected endpoint"\n'
        '  while [ "$#" -gt 0 ]; do\n'
        '    if [ "$1" = --output ]; then printf "{\\"accepted\\":false}\\n" > "$2"; break; fi\n'
        "    shift\n  done\n  exit 42;;\n*) exit 0;;\nesac\n",
        with_venv=with_venv,
    )
    output = tmp_path / "new-campaign"
    monkeypatch.setenv("RANGE_DF_EVIDENCE_DIRECTORY", str(output))
    monkeypatch.setenv("RANGE_DF_HARNESS_DIRECTORY", str(tmp_path))
    result = subprocess.run(
        ["bash", str(script)],
        cwd=checkout,
        capture_output=True,
        check=False,
        timeout=10,
    )
    assert result.returncode == 42
    assert set(trace.read_text().splitlines()) == {str(interpreter)}
    assert json.loads((output / "oracle-all-cpu.json").read_text()) == {
        "accepted": False
    }
    assert (
        output / "oracle-all-cpu.log"
    ).read_text().strip() == "independent oracle rejected endpoint"
    progress = [
        json.loads(line)
        for line in (output / "campaign-progress.jsonl").read_text().splitlines()
    ]
    assert progress[-1] == {"stage": "oracle", "status": "failed", "exit_code": 42}
    assert not any(row["status"] == "complete" for row in progress)
