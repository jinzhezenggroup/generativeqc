"""Execution and deliberate evidence publication have different output paths."""

from __future__ import annotations

import argparse
import ast
import sys
import typing
from pathlib import Path

import pytest

from benchmarks import _support

ROOT = Path(__file__).resolve().parents[2]


def active_runner_output_arguments() -> typing.Any:
    """Yield every live benchmark CLI output path, excluding frozen evidence scripts."""

    benchmark_root = ROOT / "benchmarks"
    for path in sorted(benchmark_root.rglob("*.py")):
        relative = path.relative_to(benchmark_root)
        if "results" in relative.parts or path.name == "_support.py":
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if (
                not isinstance(node, ast.Call)
                or not isinstance(node.func, ast.Attribute)
                or node.func.attr != "add_argument"
            ):
                continue
            options = [
                arg.value
                for arg in node.args
                if isinstance(arg, ast.Constant) and isinstance(arg.value, str)
            ]
            if not any(
                option == "--output"
                or option.endswith(
                    (
                        "-output",
                        "-output-dir",
                        "-output-directory",
                        "-output-file",
                        "-output-path",
                    )
                )
                for option in options
            ):
                continue
            yield path, node, options


@pytest.mark.parametrize("alias", [False, True])
def test_raw_writer_rejects_retained_tree_and_symlink_alias(
    tmp_path: typing.Any, monkeypatch: typing.Any, alias: typing.Any
) -> None:
    monkeypatch.setattr(_support, "_REPOSITORY_ROOT", tmp_path)
    retained = tmp_path / "benchmarks/results"
    retained.mkdir(parents=True)
    old = retained / "existing.json"
    old.write_text("original evidence")
    destination = retained
    if alias:
        destination = tmp_path / "alias"
        destination.symlink_to(retained, target_is_directory=True)
    with pytest.raises(ValueError, match="raw benchmark output"):
        _support.write_result(destination / "existing.json", {"overwrite": True})
    assert old.read_text() == "original evidence"
    with pytest.raises(ValueError, match="raw benchmark output"):
        _support.write_result(destination / "new/result.json", {"value": 1})
    assert not (retained / "new").exists()


def test_guard_rejects_relative_traversal_before_creating_anything(
    tmp_path: typing.Any, monkeypatch: typing.Any
) -> None:
    monkeypatch.setattr(_support, "_REPOSITORY_ROOT", tmp_path)
    monkeypatch.chdir(tmp_path)
    with pytest.raises(ValueError, match="raw benchmark output"):
        _support.write_result(".artifacts/../benchmarks/results/new.json", {})
    assert not (tmp_path / "benchmarks").exists()


def test_scratch_paths_and_neighbour_names_remain_supported(
    tmp_path: typing.Any, monkeypatch: typing.Any
) -> None:
    monkeypatch.setattr(_support, "_REPOSITORY_ROOT", tmp_path)
    for name in (
        ".artifacts/benchmarks/run.json",
        "benchmarks/results-copy/a.json",
        "scratch/run.json",
    ):
        path = tmp_path / name
        assert _support.write_result(path, {"samples": [1, 2]}) == path
        assert path.is_file()


def test_argparse_reports_bad_output_without_starting_work(
    tmp_path: typing.Any, monkeypatch: typing.Any
) -> None:
    monkeypatch.setattr(_support, "_REPOSITORY_ROOT", tmp_path)
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=_support.raw_output_path)
    with pytest.raises(SystemExit) as error:
        parser.parse_args(["--output", str(tmp_path / "benchmarks/results/x.json")])
    assert error.value.code == 2
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("destination_kind", ["retained", "alias", "traversal"])
def test_array_api_rhf_rejects_retained_output_before_work(
    tmp_path: typing.Any, monkeypatch: typing.Any, destination_kind: str
) -> None:
    from benchmarks import _retention, array_api_rhf_endpoint

    monkeypatch.setattr(_retention, "_REPOSITORY_ROOT", tmp_path)
    retained = tmp_path / "benchmarks/results"
    retained.mkdir(parents=True)
    existing = retained / "existing.json"
    existing.write_text("original evidence", encoding="utf-8")
    destination = retained
    if destination_kind == "alias":
        destination = tmp_path / "alias"
        destination.symlink_to(retained, target_is_directory=True)
    elif destination_kind == "traversal":
        destination = tmp_path / ".artifacts/../benchmarks/results"

    def forbidden(*args: typing.Any) -> typing.NoReturn:
        raise AssertionError("invalid output reached benchmark work")

    monkeypatch.setattr(array_api_rhf_endpoint, "_scheduler_guard", forbidden)
    monkeypatch.setattr(
        sys,
        "argv",
        ["array_api_rhf_endpoint", "--output", str(destination / existing.name)],
    )
    with pytest.raises(SystemExit) as error:
        array_api_rhf_endpoint.main()
    assert error.value.code == 2
    assert existing.read_text(encoding="utf-8") == "original evidence"
    assert not (tmp_path / ".artifacts").exists()


@pytest.mark.parametrize("destination_kind", ["retained", "alias", "traversal"])
def test_native_cpu_ks_rejects_retained_output_before_work(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, destination_kind: str
) -> None:
    from benchmarks import _retention, native_cpu_ks_force_endpoint

    monkeypatch.setattr(_retention, "_REPOSITORY_ROOT", tmp_path)
    retained = tmp_path / "benchmarks/results"
    retained.mkdir(parents=True)
    existing = retained / "existing.json"
    existing.write_text("original evidence", encoding="utf-8")
    destination = retained
    if destination_kind == "alias":
        destination = tmp_path / "alias"
        destination.symlink_to(retained, target_is_directory=True)
    elif destination_kind == "traversal":
        destination = tmp_path / ".artifacts/../benchmarks/results"

    def forbidden(*args: typing.Any, **kwargs: typing.Any) -> typing.NoReturn:
        raise AssertionError("invalid output reached benchmark work")

    monkeypatch.delenv("GENERATIVEQC_LIBRARY", raising=False)
    monkeypatch.setattr(native_cpu_ks_force_endpoint, "endpoint", forbidden)
    monkeypatch.setattr(native_cpu_ks_force_endpoint._native, "load_library", forbidden)
    monkeypatch.setattr(
        sys,
        "argv",
        ["native_cpu_ks_force_endpoint", "--output", str(destination / existing.name)],
    )
    with pytest.raises(SystemExit) as error:
        native_cpu_ks_force_endpoint.main()
    assert error.value.code == 2
    assert existing.read_text(encoding="utf-8") == "original evidence"
    assert not (tmp_path / ".artifacts").exists()


def test_prepared_lifecycle_cli_rejects_retained_output_before_qualification(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The lifecycle runner must reject retained output before numerical work."""
    from benchmarks import _retention, prepared_lifecycle

    monkeypatch.setattr(_retention, "_REPOSITORY_ROOT", tmp_path)
    monkeypatch.setattr(
        prepared_lifecycle,
        "qualify",
        lambda *arguments: pytest.fail("qualification started before output admission"),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "prepared_lifecycle.py",
            "--device",
            "cpu",
            "--method",
            "rhf",
            "--output",
            str(tmp_path / "benchmarks/results/receipt.json"),
        ],
    )
    with pytest.raises(SystemExit) as error:
        prepared_lifecycle.main()
    assert error.value.code == 2
    assert not (tmp_path / "benchmarks").exists()


@pytest.mark.parametrize("destination_kind", ["retained", "alias", "traversal"])
def test_xc_trajectory_rejects_retained_output_before_gpu_or_input_work(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, destination_kind: str
) -> None:
    """No GPU import, Slurm check or input read may precede output protection."""
    from benchmarks import pbe0_xc_trajectories

    monkeypatch.setattr(_support, "_REPOSITORY_ROOT", tmp_path)
    retained = tmp_path / "benchmarks/results"
    retained.mkdir(parents=True)
    existing = retained / "existing.json"
    existing.write_text("original evidence", encoding="utf-8")
    destination = retained
    if destination_kind == "alias":
        destination = tmp_path / "alias"
        destination.symlink_to(retained, target_is_directory=True)
    elif destination_kind == "traversal":
        destination = tmp_path / ".artifacts/../benchmarks/results"
    monkeypatch.delenv("SLURM_JOB_ID", raising=False)
    monkeypatch.setitem(sys.modules, "cupy", None)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "pbe0_xc_trajectories",
            "--atoms",
            "48",
            "--basis-file",
            str(tmp_path / "missing-basis.json"),
            "--reference",
            str(tmp_path / "missing-reference.json"),
            "--output",
            str(destination / existing.name),
        ],
    )
    with pytest.raises(SystemExit) as error:
        pbe0_xc_trajectories.main()
    assert error.value.code == 2
    assert existing.read_text(encoding="utf-8") == "original evidence"
    assert not (tmp_path / ".artifacts").exists()


def test_all_active_runner_outputs_are_guarded_during_argument_parsing() -> None:
    seen = []
    for path, node, options in active_runner_output_arguments():
        seen.append((path.relative_to(ROOT), tuple(options)))
        keywords = {kw.arg: kw.value for kw in node.keywords}
        assert isinstance(keywords.get("type"), ast.Name), (path, options)
        assert keywords["type"].id == "raw_output_path", (path, options)
        default = keywords.get("default")
        if isinstance(default, ast.Constant) and isinstance(default.value, str):
            assert default.value.startswith(".artifacts/"), (
                path,
                options,
                default.value,
            )
    assert seen
    assert (Path("benchmarks/h2_latency.py"), ("--output",)) in seen
    assert (
        Path("benchmarks/compare_gpu4pyscf_batch.py"),
        ("--progress-output",),
    ) in seen
    assert (
        Path("benchmarks/experiments/issue409-packed-values/run_endpoints.py"),
        ("--output",),
    ) in seen


@pytest.mark.parametrize("destination_kind", ["retained", "alias", "traversal"])
@pytest.mark.parametrize("worker", [False, True])
def test_rccsd_cpu_rejects_retained_output_before_work(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    destination_kind: str,
    worker: bool,
    capsys: pytest.CaptureFixture[str],
) -> None:
    from benchmarks import _retention, rccsd_cpu_compiled_endpoint

    monkeypatch.setattr(_retention, "_REPOSITORY_ROOT", tmp_path)
    retained = tmp_path / "benchmarks/results"
    retained.mkdir(parents=True)
    existing = retained / "existing.json"
    existing.write_text("original evidence", encoding="utf-8")
    destination = retained
    if destination_kind == "alias":
        destination = tmp_path / "alias"
        destination.symlink_to(retained, target_is_directory=True)
    elif destination_kind == "traversal":
        destination = tmp_path / ".artifacts/../benchmarks/results"

    def forbidden(*args: typing.Any, **kwargs: typing.Any) -> typing.NoReturn:
        raise AssertionError("invalid output reached benchmark work")

    monkeypatch.setattr(rccsd_cpu_compiled_endpoint, "_worker", forbidden)
    monkeypatch.setattr(
        rccsd_cpu_compiled_endpoint.tempfile, "TemporaryDirectory", forbidden
    )
    arguments = [
        "rccsd_cpu_compiled_endpoint",
        "--output",
        str(destination / existing.name),
    ]
    if worker:
        arguments += ["--worker", "--cache", str(tmp_path / "cache")]
    monkeypatch.setattr(sys, "argv", arguments)
    with pytest.raises(SystemExit) as error:
        rccsd_cpu_compiled_endpoint.main()
    assert error.value.code == 2
    assert "argument --output:" in capsys.readouterr().err
    assert existing.read_text(encoding="utf-8") == "original evidence"
    assert not (tmp_path / ".artifacts").exists()
    assert not (tmp_path / "cache").exists()


def test_rccsd_cpu_accepts_scratch_output_without_changing_stages(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    import json
    from types import SimpleNamespace

    from benchmarks import _retention, rccsd_cpu_compiled_endpoint

    monkeypatch.setattr(_retention, "_REPOSITORY_ROOT", tmp_path)

    def fake_run(*args: typing.Any, **kwargs: typing.Any) -> SimpleNamespace:
        return SimpleNamespace(returncode=0, stdout='{"energy": -1.0}', stderr="")

    monkeypatch.setattr(rccsd_cpu_compiled_endpoint.subprocess, "run", fake_run)
    destination = tmp_path / ".artifacts/benchmarks/rccsd.json"
    monkeypatch.setattr(
        sys,
        "argv",
        ["rccsd_cpu_compiled_endpoint", "--threads", "1", "--output", str(destination)],
    )
    rccsd_cpu_compiled_endpoint.main()
    payload = destination.read_text(encoding="utf-8")
    assert capsys.readouterr().out == payload
    record = json.loads(payload)
    assert record["schema"] == "generativeqc.rccsd.native-cpu-endpoint.v1"
    assert [row["stage"] for row in record["observations"]] == [
        "reference",
        "native_cold",
        "native_warm",
    ]
    assert all(row["blas_threads"] == 1 for row in record["observations"])
