"""Host-only acceptance checks for the opt-in GPU Hessian benchmark."""

from __future__ import annotations

import json
import lzma
import subprocess
import sys
import time
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable
    from typing import TextIO

import pytest

from benchmarks.compare_wb97mv_hessian_gpu4pyscf import _run_child, _summarize


def _sample(engine: str, seconds: float, matrix_value: float = 1.0) -> dict:
    return {
        "engine": engine,
        "wall_seconds": seconds,
        "result": {"matrix": [[matrix_value]]},
    }


def test_primed_medians_exclude_first_observed_sample() -> None:
    rows = [
        _sample("generativeqc", 100.0),
        _sample("gpu4pyscf", 200.0),
        _sample("gpu4pyscf", 2.0),
        _sample("generativeqc", 6.0),
        _sample("generativeqc", 4.0),
        _sample("gpu4pyscf", 4.0),
    ]
    summary = _summarize(rows, 1e-6)
    assert summary["accepted"]
    assert summary["artifact_primed_fresh_process_median_seconds"] == {
        "generativeqc": 5.0,
        "gpu4pyscf": 3.0,
    }
    assert summary["generativeqc_over_gpu4pyscf_wall_ratio"] == pytest.approx(5 / 3)


def test_inaccurate_repeat_prevents_a_qualified_speed_ratio() -> None:
    rows = [
        _sample("generativeqc", 3.0, 1.001),
        _sample("gpu4pyscf", 2.0),
        _sample("generativeqc", 1.0),
        _sample("gpu4pyscf", 1.0),
    ]
    summary = _summarize(rows, 1e-6)
    assert not summary["accepted"]
    assert summary["maximum_hessian_error_hartree_per_bohr2"] == pytest.approx(0.001)
    assert summary["generativeqc_over_gpu4pyscf_wall_ratio"] is None


def test_single_engine_preflight_is_not_a_speed_comparison() -> None:
    summary = _summarize([_sample("gpu4pyscf", 2.0)], 1e-6)
    assert not summary["accepted"]
    assert summary["maximum_hessian_error_hartree_per_bohr2"] is None
    assert summary["generativeqc_over_gpu4pyscf_wall_ratio"] is None


def test_nonfinite_repeat_cannot_be_hidden_by_a_finite_maximum_error() -> None:
    rows = [
        _sample("generativeqc", 2.0),
        _sample("gpu4pyscf", 1.0),
        _sample("generativeqc", 1.0, float("nan")),
    ]
    with pytest.raises(ValueError, match="finite real"):
        _summarize(rows, 1e-6)


def test_accuracy_comparison_does_not_broadcast_different_matrix_shapes() -> None:
    reference = _sample("gpu4pyscf", 1.0)
    reference["result"]["matrix"] = [[1.0, 1.0], [1.0, 1.0]]
    with pytest.raises(ValueError, match="broadcasting"):
        _summarize([_sample("generativeqc", 2.0), reference], 1e-6)


def test_retained_h2_baseline_reproduces_all_repeat_accuracy_and_timing_gates() -> None:
    path = (
        Path(__file__).resolve().parents[2]
        / "benchmarks/results/wb97mv-hessian-bridge-h2-20261015.json.xz"
    )
    case = json.loads(lzma.decompress(path.read_bytes()))["cases"]["h2"]
    samples = [
        {
            **row,
            "result": {
                **row["result"],
                "matrix": case["matrices"][row["result"]["matrix_index"]],
            },
        }
        for row in case["samples"]
    ]
    assert len(samples) == 6
    summary = _summarize(samples, case["summary"]["hessian_absolute_tolerance"])
    assert summary == case["summary"]
    assert not summary["accepted"]
    assert summary["generativeqc_over_gpu4pyscf_wall_ratio"] is None


def test_retained_water_timeout_does_not_invent_native_latency() -> None:
    root = Path(__file__).resolve().parents[2] / "benchmarks/results"
    receipt = json.loads(
        (root / "wb97mv-hessian-bridge-water-timeout-20261015.json").read_text()
    )
    assert (root / receipt["workload_reference"]).is_file()
    assert not receipt["accepted"]
    assert receipt["qualified_wall_ratio"] is None
    assert "timed out" in receipt["failure"]
    assert {row["engine"] for row in receipt["samples"]} == {"gpu4pyscf"}


def test_timeout_cleans_a_separately_grouped_endpoint_worker(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    original_read = Path.read_text

    def without_children(path: Path, *args: object, **kwargs: object) -> str:
        if path.name == "children":
            raise FileNotFoundError("kernel omits task/children")
        return original_read(path, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", without_children)
    prefix = tmp_path / "nested-worker"
    command = [
        sys.executable,
        "-c",
        (
            "import subprocess,sys; "
            "child=subprocess.Popen([sys.executable,'-c','import time; time.sleep(60)'],process_group=0); "
            "print(child.pid,flush=True); child.wait()"
        ),
    ]
    unrelated = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(30)"], start_new_session=True
    )
    started = time.monotonic()
    try:
        with pytest.raises(RuntimeError, match="timed out"):
            _run_child(command, prefix, 1)
        assert time.monotonic() - started < 10
        assert unrelated.poll() is None
    finally:
        unrelated.kill()
        unrelated.wait(timeout=5)
    child_id = int(prefix.with_suffix(".stdout").read_text())
    status = Path(f"/proc/{child_id}/status")
    for _attempt in range(100):
        try:
            state = status.read_text()
        except FileNotFoundError:
            return
        if "\nState:\tZ" in state:
            return
        time.sleep(0.01)
    pytest.fail("timed-out endpoint worker is still running")


def _runner_arguments(
    monkeypatch: pytest.MonkeyPatch, output: Path, *extra: str
) -> None:
    monkeypatch.setenv("SLURM_JOB_ID", "host-only-fault-injection")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "benchmark",
            "--output",
            str(output),
            "--cli",
            sys.executable,
            "--compiler-python",
            sys.executable,
            "--cases",
            "h2",
            "--repeats",
            "0",
            *extra,
        ],
    )


@pytest.mark.parametrize("alias", [False, True])
def test_retained_output_rejected_before_gpu_import(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture,
    alias: bool,
) -> None:
    from benchmarks import compare_wb97mv_hessian_gpu4pyscf as runner

    retained = Path(runner.__file__).resolve().parent / "results"
    if alias:
        link = tmp_path / "retained-alias"
        link.symlink_to(retained, target_is_directory=True)
        retained = link
    _runner_arguments(monkeypatch, retained / "never-written.json")
    monkeypatch.setitem(sys.modules, "cupy", None)
    with pytest.raises(SystemExit) as error:
        runner.main()
    assert error.value.code == 2
    assert "raw_output_path" in capsys.readouterr().err
    assert not (retained / "never-written.json").exists()


@pytest.mark.parametrize(
    "kind", ["existing", "symlink", "dangling-symlink", "concurrent-writer"]
)
def test_existing_output_preserved_before_gpu_import(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, kind: str
) -> None:
    from benchmarks import compare_wb97mv_hessian_gpu4pyscf as runner

    output, target = tmp_path / "output.json", tmp_path / "unrelated.json"
    if kind == "existing":
        output.write_text("existing evidence")
    elif "symlink" in kind:
        if kind == "symlink":
            target.write_text("unrelated evidence")
        output.symlink_to(target)
    else:
        original_open = Path.open

        def compete(
            path: Path, mode: str = "r", *args: object, **kwargs: object
        ) -> TextIO:
            if path == output and mode == "x+":
                with original_open(path, "x") as other:
                    other.write("competing evidence")
            return original_open(path, mode, *args, **kwargs)

        monkeypatch.setattr(Path, "open", compete)
    _runner_arguments(monkeypatch, output)
    monkeypatch.setitem(sys.modules, "cupy", None)
    with pytest.raises(SystemExit) as error:
        runner.main()
    assert error.value.code == 2
    if "symlink" in kind:
        assert output.is_symlink()
        assert (
            target.read_text() == "unrelated evidence"
            if kind == "symlink"
            else not target.exists()
        )
    else:
        assert output.read_text() == (
            "existing evidence" if kind == "existing" else "competing evidence"
        )


@pytest.mark.parametrize("symlink", [False, True])
def test_existing_sample_directory_preserved(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, symlink: bool
) -> None:
    from benchmarks import compare_wb97mv_hessian_gpu4pyscf as runner

    output, logs = tmp_path / "output.json", tmp_path / "output-samples"
    target = tmp_path / "prior-logs" if symlink else logs
    target.mkdir()
    (target / "h2.xyz").write_text("existing coordinates")
    if symlink:
        logs.symlink_to(target, target_is_directory=True)
    _runner_arguments(monkeypatch, output)
    monkeypatch.setitem(sys.modules, "cupy", None)
    with pytest.raises(SystemExit) as error:
        runner.main()
    assert error.value.code == 2
    assert (target / "h2.xyz").read_text() == "existing coordinates"


@pytest.mark.parametrize("replacement", ["file", "symlink"])
def test_updates_keep_the_reserved_output_descriptor(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, replacement: str
) -> None:
    from benchmarks import compare_wb97mv_hessian_gpu4pyscf as runner

    output, owned, target = (
        tmp_path / name for name in ("output.json", "owned.json", "other.json")
    )
    target.write_text("unrelated evidence")
    _runner_arguments(monkeypatch, output)

    def measure(arguments: object, payload: dict, logs: Path, save: object) -> None:
        assert json.loads(output.read_text())["status"] == "running"
        output.rename(owned)
        if replacement == "file":
            output.write_text("replacement evidence")
        else:
            output.symlink_to(target)
        raise RuntimeError("injected endpoint failure")

    monkeypatch.setattr(runner, "_measure", measure)
    with pytest.raises(RuntimeError, match="injected endpoint"):
        runner.main()
    assert json.loads(owned.read_text())["status"] == "failed"
    assert target.read_text() == "unrelated evidence"
    assert output.read_text() == (
        "replacement evidence" if replacement == "file" else "unrelated evidence"
    )


@pytest.mark.parametrize("suffix", [".stdout", ".stderr"])
@pytest.mark.parametrize("symlink", [False, True])
def test_child_logs_are_reserved_before_launch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, suffix: str, symlink: bool
) -> None:
    from benchmarks import compare_wb97mv_hessian_gpu4pyscf as runner

    prefix = tmp_path / "sample"
    target = tmp_path / "unrelated.log" if symlink else prefix.with_suffix(suffix)
    target.write_text("prior evidence")
    if symlink:
        prefix.with_suffix(suffix).symlink_to(target)

    def forbidden(*args: object, **kwargs: object) -> None:
        pytest.fail("endpoint launched before both log reservations")

    monkeypatch.setattr(runner.subprocess, "Popen", forbidden)
    with pytest.raises(FileExistsError):
        _run_child([sys.executable], prefix, 1)
    assert target.read_text() == "prior evidence"


@pytest.mark.parametrize(
    "extra",
    [
        ("--hessian-atol", "inf"),
        ("--hessian-atol", "nan"),
        ("--hessian-atol", "0"),
        ("--grid", "0", "4", "8"),
        ("--engines", "generativeqc", "generativeqc"),
        ("--cases", "h2", "h2"),
    ],
)
def test_invalid_scientific_controls_rejected_before_gpu_import(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, extra: tuple[str, ...]
) -> None:
    from benchmarks import compare_wb97mv_hessian_gpu4pyscf as runner

    output = tmp_path / "output.json"
    _runner_arguments(monkeypatch, output, *extra)
    monkeypatch.setitem(sys.modules, "cupy", None)
    with pytest.raises(SystemExit) as error:
        runner.main()
    assert error.value.code == 2
    assert not output.exists()


@pytest.mark.parametrize("value", [float("inf"), float("nan"), 0, -1])
def test_unbounded_accuracy_tolerance_is_not_a_qualified_comparison(
    value: float,
) -> None:
    with pytest.raises(ValueError, match="finite and positive"):
        _summarize([_sample("generativeqc", 1), _sample("gpu4pyscf", 2)], value)


@pytest.mark.parametrize("wrong_shape", [False, True])
def test_complete_host_injected_run_retains_samples_and_shape_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, wrong_shape: bool
) -> None:
    from types import SimpleNamespace

    from benchmarks import compare_wb97mv_hessian_gpu4pyscf as runner

    output = tmp_path / "output.json"
    _runner_arguments(monkeypatch, output, "--repeats", "1")
    monkeypatch.setenv("GENERATIVEQC_LIBRARY", sys.executable)
    cp = SimpleNamespace(
        cuda=SimpleNamespace(
            Stream=SimpleNamespace(null=SimpleNamespace(synchronize=lambda: None))
        )
    )
    monkeypatch.setitem(sys.modules, "cupy", cp)
    monkeypatch.setattr(runner, "cuda_accelerator_metadata", lambda _: {})
    monkeypatch.setattr(runner, "environment_metadata", lambda **kwargs: {})
    monkeypatch.setattr(runner, "_grid", lambda _: None)
    monkeypatch.setattr(runner, "asdict", lambda _: {})

    def sample(command: list[str], prefix: Path, timeout: float) -> dict:
        retained = json.loads(output.read_text())
        assert retained["status"] == "running"
        return {
            "wall_seconds": 1.0,
            "result": {
                "matrix": [[1.0]] if wrong_shape else [[1.0] * 6 for _ in range(6)],
                "diagnostics": {"numerical_host_fallbacks": 0},
            },
        }

    monkeypatch.setattr(runner, "_run_child", sample)
    if wrong_shape:
        with pytest.raises(RuntimeError, match="wrong molecular shape"):
            runner.main()
    else:
        runner.main()
    retained = json.loads(output.read_text())
    assert retained["status"] == ("failed" if wrong_shape else "completed")
    case = retained["cases"]["h2"]
    assert len(case["samples"]) == (1 if wrong_shape else 4)
    if wrong_shape:
        assert "summary" not in case
    else:
        assert case["summary"]["accepted"]
        assert case["summary"]["generativeqc_over_gpu4pyscf_wall_ratio"] == 1


def test_missing_gpu_dependency_retains_a_failed_attempt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from benchmarks import compare_wb97mv_hessian_gpu4pyscf as runner

    output = tmp_path / "output.json"
    _runner_arguments(monkeypatch, output)
    monkeypatch.setitem(sys.modules, "cupy", None)
    with pytest.raises(ModuleNotFoundError):
        runner.main()
    record = json.loads(output.read_text())
    assert record["status"] == "failed"
    assert "cupy" in record["failure"]
    assert record["cases"] == {}


def test_child_log_updates_do_not_follow_replaced_pathnames(tmp_path: Path) -> None:
    prefix = tmp_path / "sample"
    original = prefix.with_suffix(".stdout")
    moved = tmp_path / "original.stdout"
    payload = {"matrix": [[1.0]], "units": "hartree/bohr^2"}
    code = (
        "from pathlib import Path; import json; "
        f"p=Path({str(original)!r}); p.rename({str(moved)!r}); "
        "p.write_text('competing evidence'); "
        f"print({json.dumps(payload)!r})"
    )
    result = _run_child([sys.executable, "-c", code], prefix, 5)
    assert result["result"] == payload
    assert original.read_text() == "competing evidence"
    assert json.loads(moved.read_text()) == payload


@pytest.mark.parametrize("late", [False, True])
@pytest.mark.parametrize("field", ["matrix", "diagnostics"])
@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
def test_nonfinite_child_result_retains_failed_journal_and_exact_raw_log(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    field: str,
    value: float,
    late: bool,
) -> None:
    from types import SimpleNamespace

    from benchmarks import compare_wb97mv_hessian_gpu4pyscf as runner

    output = tmp_path / "output.json"
    _runner_arguments(
        monkeypatch,
        output,
        *(("--repeats", "2") if late else ("--engines", "gpu4pyscf")),
    )
    monkeypatch.setenv("GENERATIVEQC_LIBRARY", sys.executable)
    cp = SimpleNamespace(
        cuda=SimpleNamespace(
            Stream=SimpleNamespace(null=SimpleNamespace(synchronize=lambda: None))
        )
    )
    monkeypatch.setitem(sys.modules, "cupy", cp)
    monkeypatch.setattr(runner, "cuda_accelerator_metadata", lambda _: {})
    monkeypatch.setattr(runner, "environment_metadata", lambda **kwargs: {})
    monkeypatch.setattr(runner, "_grid", lambda _: None)
    monkeypatch.setattr(runner, "asdict", lambda _: {})
    payload = {
        "matrix": [[1.0] * 6 for _ in range(6)],
        "units": "hartree/bohr^2",
        "diagnostics": {"residual": 0.0, "numerical_host_fallbacks": 0},
    }
    finite_raw = json.dumps(payload) + "\n"
    if field == "matrix":
        payload["matrix"][0][0] = value
    else:
        payload["diagnostics"]["residual"] = value
    raw = json.dumps(payload) + "\n"
    commands = 0

    def command(*args: object) -> list[str]:
        nonlocal commands
        emitted = finite_raw if late and commands < 4 else raw
        commands += 1
        return [sys.executable, "-c", f"import sys; sys.stdout.write({emitted!r})"]

    monkeypatch.setattr(runner, "_command", command)
    with pytest.raises(
        RuntimeError, match="invalid Hessian output|nonfinite endpoint payload"
    ):
        runner.main()

    def forbidden_constant(token: str) -> None:
        pytest.fail(f"journal contains non-portable JSON constant {token}")

    retained = json.loads(output.read_text(), parse_constant=forbidden_constant)
    assert retained["status"] == "failed"
    case = retained["cases"]["h2"]
    assert "summary" not in case
    label = "h2-2-generativeqc" if late else "h2-0-gpu4pyscf"
    assert (tmp_path / f"output-samples/{label}.stdout").read_text() == raw
    if field == "diagnostics":
        assert case["samples"][-1]["result"]["diagnostics"]["residual"] is None
        assert retained["nonfinite_values"] == {
            f"$report.cases.h2.samples[{4 if late else 0}].result.diagnostics.residual": repr(
                value
            )
        }


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
def test_nonfinite_row_after_child_boundary_still_records_failed_state(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, value: float
) -> None:
    from benchmarks import compare_wb97mv_hessian_gpu4pyscf as runner

    output = tmp_path / "output.json"
    _runner_arguments(monkeypatch, output)

    def injected_measure(
        arguments: object, payload: dict, logs: Path, save: Callable[[], None]
    ) -> None:
        payload["cases"]["h2"] = {"samples": [{"result": {"matrix": [[value]]}}]}
        save()
        raise RuntimeError("nonfinite injected endpoint")

    monkeypatch.setattr(runner, "_measure", injected_measure)
    with pytest.raises(RuntimeError, match="nonfinite injected endpoint"):
        runner.main()
    retained = json.loads(output.read_text())
    assert retained["status"] == "failed"
    assert retained["cases"]["h2"]["samples"][0]["result"]["matrix"] == [[None]]
    assert retained["nonfinite_values"] == {
        "$report.cases.h2.samples[0].result.matrix[0][0]": repr(value)
    }
