"""Budget-Fock timing claims require numerical and executed-arithmetic gates."""

from __future__ import annotations

import copy
import json
import math
import sys
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

from benchmarks.pbe0_fast_fock import check_sample, portable_report

if TYPE_CHECKING:
    from typing import TextIO


@pytest.mark.parametrize("destination_kind", ["retained", "alias", "traversal"])
def test_reject_retained_output_before_scientific_work(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    destination_kind: str,
) -> None:
    from benchmarks import _retention, pbe0_fast_fock

    monkeypatch.setattr(_retention, "_REPOSITORY_ROOT", tmp_path)
    monkeypatch.chdir(tmp_path)
    retained = tmp_path / "benchmarks/results"
    retained.mkdir(parents=True)
    original = retained / "existing.json"
    original.write_text("original evidence", encoding="utf-8")
    destination = retained / "new/run.json"
    if destination_kind == "alias":
        alias = tmp_path / "alias"
        alias.symlink_to(retained, target_is_directory=True)
        destination = alias / "new/run.json"
    elif destination_kind == "traversal":
        destination = Path(".artifacts/../benchmarks/results/new/run.json")

    def forbidden(*args: object, **kwargs: object) -> None:
        raise AssertionError("scientific work started before output validation")

    for name in (
        "Calculator",
        "GridSpec",
        "protocol",
        "reference",
        "native_sample",
        "native_build_metadata",
    ):
        monkeypatch.setattr(pbe0_fast_fock, name, forbidden)
    monkeypatch.setenv("SLURM_JOB_ID", "host-output-guard-test")
    monkeypatch.setattr(
        sys,
        "argv",
        ["pbe0_fast_fock", "--engine", "reference", "--output", str(destination)],
    )
    with pytest.raises(SystemExit) as error:
        pbe0_fast_fock.main()
    assert error.value.code == 2
    assert "invalid raw_output_path value" in capsys.readouterr().err
    assert original.read_text(encoding="utf-8") == "original evidence"
    assert not (retained / "new").exists()
    assert not (tmp_path / ".artifacts").exists()


def test_rejected_nonfinite_diagnostic_remains_portable_and_observable() -> None:
    report = {"records": [{"energy": -1.0, "history": [{"hartree": float("nan")}]}]}
    portable = portable_report(report)
    assert portable["records"][0]["history"][0]["hartree"] is None
    assert portable["nonfinite_values"] == {
        "$report.records[0].history[0].hartree": "nan"
    }
    assert math.isnan(report["records"][0]["history"][0]["hartree"])
    json.dumps(portable, allow_nan=False)
    assert portable_report({"energy": -1.0}) == {"energy": -1.0}


def test_fast_claim_requires_fp32_exchange_and_one_endpoint() -> None:
    oracle = {"energy": -1.0, "forces": [[0.0, 0.0, 0.0]]}
    row = {
        **oracle,
        "precision_validation": {"status": "qualified_budget_fock", "solves": [{}]},
        "precision": {
            "operators": [
                {"name": name, "compute": "fp32", "accumulation": "fp64", "count": 3}
                for name in ("coulomb_j", "exchange_k", "coulomb_recurrence")
            ]
        },
    }
    assert check_sample(row, oracle, "fast")["energy_Eh"] == 0.0
    for defect in (
        "strict_k",
        "empty_k",
        "extra_solve",
        "force_error",
        "energy_error",
        "nan",
    ):
        invalid = copy.deepcopy(row)
        if defect == "strict_k":
            invalid["precision"]["operators"][1]["compute"] = "fp64"
        elif defect == "empty_k":
            invalid["precision"]["operators"][1]["count"] = 0
        elif defect == "extra_solve":
            invalid["precision_validation"]["solves"].append({})
        elif defect == "force_error":
            invalid["forces"][0][0] = 1e-3
        elif defect == "energy_error":
            invalid["energy"] += 1e-3
        else:
            invalid["energy"] = float("nan")
        with pytest.raises((RuntimeError, ValueError)):
            check_sample(invalid, oracle, "fast")


def mocked_runner(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple:
    """Exercise the runner's retained evidence without a compiler or device."""
    from benchmarks import pbe0_fast_fock as runner

    output = tmp_path / "run.json"
    oracle = tmp_path / "oracle.json"
    oracle.write_text(
        json.dumps(
            {
                "status": "reference-completed",
                "protocol": {"fixture": "independent"},
                "reference": {
                    "converged": True,
                    "reference_backend": "gpu4pyscf",
                    "energy": -1.0,
                    "forces": [[0.0, 0.0, 0.0]],
                },
            }
        )
    )
    monkeypatch.setenv("SLURM_JOB_ID", "host-retention-test")
    monkeypatch.delenv("GENERATIVEQC_LIBRARY", raising=False)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "pbe0_fast_fock",
            "--engine",
            "native",
            "--output",
            str(output),
            "--oracle",
            str(oracle),
            "--repeats",
            "1",
        ],
    )
    monkeypatch.setattr(runner, "GridSpec", lambda *_: object())
    monkeypatch.setattr(runner, "protocol", lambda *_: {"fixture": "independent"})
    settings = {
        "precision": "fp64",
        "energy_tolerance": 1e-10,
        "density_tolerance": 1e-8,
        "screening_tolerance": 1e-12,
    }
    monkeypatch.setattr(runner, "options", lambda *_: settings.copy())
    cleared = []

    class Calculator:
        def __init__(self, **kwargs: object) -> None:
            assert kwargs == settings

        def clear_cache(self) -> None:
            cleared.append(True)

    monkeypatch.setattr(runner, "Calculator", Calculator)
    build = {"library_path": "/actual/loaded/library", "library_sha256": "a" * 64}
    monkeypatch.setattr(runner, "native_build_metadata", lambda _: build.copy())

    def sample(mode: str, settings: dict, case: str) -> dict:
        retained = json.loads(output.read_text())
        assert retained["records"], "attempt must be saved before endpoint execution"
        attempt = retained["records"][-1]
        assert attempt["mode"] == mode and attempt["attempt_status"] == "running"
        return {
            "mode": mode,
            "seconds": 1.0,
            "energy": -1.0,
            "forces": [[0.0, 0.0, 0.0]],
            "precision_validation": {"status": "qualified_budget_fock", "solves": [{}]},
            "precision": {
                "operators": [
                    {
                        "name": name,
                        "compute": "fp32",
                        "accumulation": "fp64",
                        "count": 3,
                    }
                    for name in ("coulomb_j", "exchange_k", "coulomb_recurrence")
                ]
            },
        }

    monkeypatch.setattr(runner, "native_sample", sample)
    return runner, output, cleared


@pytest.mark.parametrize("kind", ["existing", "dangling-symlink", "concurrent-writer"])
def test_runner_reserves_new_output_before_scientific_work(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, kind: str
) -> None:
    runner, output, _ = mocked_runner(tmp_path, monkeypatch)
    target = tmp_path / "unrelated.json"
    original_open = Path.open
    if kind == "existing":
        output.write_text("existing evidence")
    elif kind == "dangling-symlink":
        output.symlink_to(target)
    else:

        def competing_open(
            path: Path, mode: str = "r", *args: object, **kwargs: object
        ) -> TextIO:
            if path == output and mode == "x+":
                with original_open(output, "x") as competitor:
                    competitor.write("competing evidence")
            return original_open(path, mode, *args, **kwargs)

        monkeypatch.setattr(Path, "open", competing_open)

    def forbidden(*args: object, **kwargs: object) -> None:
        pytest.fail("scientific work started before exclusive reservation")

    monkeypatch.setattr(runner, "GridSpec", forbidden)
    with pytest.raises(SystemExit) as error:
        runner.main()
    assert error.value.code == 2
    if kind == "dangling-symlink":
        assert output.is_symlink() and not target.exists()
    else:
        assert output.read_text() == (
            "existing evidence" if kind == "existing" else "competing evidence"
        )


@pytest.mark.parametrize("replacement", ["file", "symlink"])
def test_runner_updates_only_the_reserved_file_descriptor(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, replacement: str
) -> None:
    runner, output, _ = mocked_runner(tmp_path, monkeypatch)
    reserved = tmp_path / "reserved.json"
    unrelated = tmp_path / "unrelated.json"
    unrelated.write_text("unrelated evidence")

    def replace_name(*args: object) -> dict:
        assert json.loads(output.read_text())["status"] == "running"
        output.rename(reserved)
        if replacement == "file":
            output.write_text("replacement evidence")
        else:
            output.symlink_to(unrelated)
        return {"fixture": "independent"}

    monkeypatch.setattr(runner, "protocol", replace_name)
    # Reference mode keeps this test about the writer's descriptor ownership.
    monkeypatch.setattr(
        sys, "argv", ["bench", "--engine", "reference", "--output", str(output)]
    )
    monkeypatch.setattr(runner, "reference", lambda *args, **kwargs: {"complete": True})
    runner.main()
    assert json.loads(reserved.read_text())["status"] == "reference-completed"
    assert unrelated.read_text() == "unrelated evidence"
    assert output.read_text() == (
        "replacement evidence" if replacement == "file" else "unrelated evidence"
    )


@pytest.mark.parametrize("environment", [None, "different-unloaded-library"])
def test_runner_records_actual_loaded_library_and_all_attempts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, environment: str | None
) -> None:
    runner, output, cleared = mocked_runner(tmp_path, monkeypatch)
    if environment is not None:
        wrong = tmp_path / environment
        wrong.write_bytes(b"not the loaded binary")
        monkeypatch.setenv("GENERATIVEQC_LIBRARY", str(wrong))
    runner.main()
    report = json.loads(output.read_text())
    assert report["status"] == "measured" and report["performance_claim"]
    assert (
        report["native_library_sha256"] == report["build"]["library_sha256"] == "a" * 64
    )
    assert cleared == [True]
    assert [
        (row["mode"], row["repeat"], row["phase"]) for row in report["records"]
    ] == [
        ("fp64", -1, "first-call"),
        ("fast", -1, "first-call"),
        ("fast", 0, "measured"),
        ("fp64", 0, "measured"),
    ]
    assert all(
        row["attempt_status"] == "qualified" and row["oracle_gate"]
        for row in report["records"]
    )


@pytest.mark.parametrize("failure_index", [1, 3])
def test_runner_retains_failed_attempt_context_and_prior_records(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failure_index: int
) -> None:
    runner, output, _ = mocked_runner(tmp_path, monkeypatch)
    original = runner.native_sample
    calls = []

    def failing_sample(mode: str, settings: dict, case: str) -> dict:
        calls.append(mode)
        retained = json.loads(output.read_text())
        assert len(retained["records"]) == len(calls), (
            "attempt must be saved before endpoint execution"
        )
        attempt = retained["records"][-1]
        assert attempt["attempt_status"] == "running" and attempt["mode"] == mode
        if len(calls) == failure_index:
            raise RuntimeError("manufactured endpoint failure")
        return original(mode, settings, case)

    monkeypatch.setattr(runner, "native_sample", failing_sample)
    with pytest.raises(RuntimeError, match="manufactured endpoint failure"):
        runner.main()
    report = json.loads(output.read_text())
    assert report["status"] == "failed" and not report["performance_claim"]
    assert len(report["records"]) == failure_index
    failed = report["records"][-1]
    assert (failed["mode"], failed["repeat"], failed["phase"]) == (
        ("fp64", -1, "first-call") if failure_index == 1 else ("fast", 0, "measured")
    )
    assert failed["attempt_status"] == "failed"
    assert "manufactured endpoint failure" in failed["error"]
    assert "manufactured endpoint failure" in failed["traceback"]
    assert all(row["attempt_status"] == "qualified" for row in report["records"][:-1])


def test_runner_retains_failed_oracle_sample_before_rejection(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runner, output, _ = mocked_runner(tmp_path, monkeypatch)
    original = runner.native_sample

    def invalid_sample(*args: object) -> dict:
        row = original(*args)
        row["energy"] += 1.0
        return row

    monkeypatch.setattr(runner, "native_sample", invalid_sample)
    with pytest.raises(RuntimeError, match="independent energy/force budget failed"):
        runner.main()
    report = json.loads(output.read_text())
    failed = report["records"][-1]
    assert failed["energy"] == 0.0 and failed["attempt_status"] == "failed"
    assert not failed.get("oracle_gate") and not report["performance_claim"]


def test_runner_retains_protocol_failure_after_reservation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runner, output, _ = mocked_runner(tmp_path, monkeypatch)

    def failed_protocol(*args: object) -> None:
        raise RuntimeError("manufactured protocol failure")

    monkeypatch.setattr(runner, "protocol", failed_protocol)
    with pytest.raises(RuntimeError, match="manufactured protocol failure"):
        runner.main()
    report = json.loads(output.read_text())
    assert report["status"] == "failed" and report["protocol"] is None
    assert report["records"] == [] and not report["performance_claim"]
