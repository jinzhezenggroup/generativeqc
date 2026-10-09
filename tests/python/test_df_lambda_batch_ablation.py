"""Protocol invariants for strict Lambda Q-batch experiments."""

import argparse
import json
from collections.abc import Callable
from pathlib import Path

import pytest

from benchmarks import df_lambda_batch_ablation as sweep
from benchmarks.df_lambda_batch_ablation import cold_command
from benchmarks.df_lambda_core_reuse_ablation import endpoint_controls

MockSweep = tuple[argparse.Namespace, list[list[str]], list[dict], Callable[..., None]]


@pytest.mark.parametrize("batch", [1, 8, 16, 32, 488])
def test_only_lambda_batch_slot_changes(batch: int) -> None:
    baseline = cold_command(Path("endpoint"), Path("input"), Path("output"), 8)
    actual = cold_command(Path("endpoint"), Path("input"), Path("output"), batch)
    assert len(actual) == 28
    assert actual[7] == str(batch)
    assert actual[9] == "8"
    assert actual[23:] == ["0", "30", "1", "1", "0"]
    assert actual[:7] + actual[8:] == baseline[:7] + baseline[8:]


@pytest.mark.parametrize("batch", [0, -1])
def test_nonpositive_batch_limits_are_rejected(batch: int) -> None:
    with pytest.raises(ValueError, match="positive"):
        endpoint_controls(batch)


def test_default_comparison_is_explicit_and_reversed(mock_sweep: MockSweep) -> None:
    args, commands, _, _ = mock_sweep
    args.batch_limits = [8, 32]
    args.default_candidate = True
    result = sweep.run_sweep(args)
    assert [command[7] for command in commands] == ["8", "32", "32", "8"]
    assert [command[27] for command in commands] == ["0", "1", "1", "0"]
    assert [row["configuration"] for row in result["observations"]] == [
        "b8",
        "b32_p1",
        "b32_p1",
        "b8",
    ]


def test_replay_selector_is_trailing_and_disabled_by_default() -> None:
    baseline = cold_command(Path("endpoint"), Path("input"), Path("output"), 32)
    candidate = cold_command(Path("endpoint"), Path("input"), Path("output"), 32, True)
    assert baseline[-1] == "0" and candidate == [*baseline[:-1], "1"]


@pytest.fixture
def mock_sweep(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> MockSweep:
    """Mock every external process: these protocol tests never access a GPU."""
    endpoint, library = tmp_path / "endpoint", tmp_path / "library"
    endpoint.write_bytes(b"mock executable")
    library.write_bytes(b"mock library")
    args = argparse.Namespace(
        endpoint=endpoint,
        library=library,
        input=Path(sweep.__file__).parent
        / "results/rccsd-diis-ring-1900/ethane230.input",
        oracle=tmp_path / "oracle.json",
        finite_differences=tmp_path / "fd.json",
        output=tmp_path / "observations",
        batch_limits=[8, 16],
        repetitions=2,
        screen=False,
        instrumented=False,
        timeout=10,
    )
    monkeypatch.setenv("SLURM_JOB_ID", "mock-no-gpu")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "mock-no-gpu")
    monkeypatch.setattr(
        sweep.subprocess, "check_output", lambda *args, **kwargs: "mock GPU"
    )
    monkeypatch.setattr(sweep, "scientific_gates", lambda *args: {"mock": True})
    monkeypatch.setattr(sweep, "summarize", lambda rows: {"count": len(rows)})
    commands = []
    records = []

    def process(command: list[str], **kwargs: object) -> None:
        commands.append(command)
        record = {
            "lambda_matrix_gemm": True,
            "lambda_core_reuse": True,
            "lambda_audit_matrix": True,
            "lambda_batch_size": int(command[7]),
            "lambda_iterations": 21,
            "lambda_actions": 22,
            "forces": [0.0] * 24,
            "native_seconds": 500.0,
            "lambda_seconds": 110.0,
            "lambda_primal_matrix": command[27] == "1",
        }
        records.append(record)
        # Files do not exist before the first endpoint: premature oracle reads fail.
        args.oracle.write_text("{}")
        args.finite_differences.write_text("{}")
        Path(command[2]).write_text(json.dumps(record))

    monkeypatch.setattr(sweep.subprocess, "run", process)
    return args, commands, records, process


def test_matched_sweep_reverses_order_and_reads_oracles_after_production(
    mock_sweep: MockSweep,
) -> None:
    args, commands, _, _ = mock_sweep
    result = sweep.run_sweep(args)
    assert [int(command[7]) for command in commands] == [8, 16, 16, 8]
    assert not result["performance_promotion"]
    assert not result["screen_only"]
    assert (args.output / "matched-summary.json").exists()
    assert not (args.output / "screen-summary.json").exists()


def test_screen_cannot_publish_matched_qualification(mock_sweep: MockSweep) -> None:
    args, _, _, _ = mock_sweep
    args.screen, args.repetitions = True, 1
    result = sweep.run_sweep(args)
    assert result["screen_only"]
    assert (args.output / "screen-summary.json").exists()
    assert not (args.output / "matched-summary.json").exists()


def test_replay_cells_are_separate_and_reversed_with_batch_cells(
    mock_sweep: MockSweep,
) -> None:
    args, _, _, _ = mock_sweep
    args.replay_candidates = True
    result = sweep.run_sweep(args)
    assert [row["configuration"] for row in result["observations"]] == [
        "b8",
        "b16",
        "b8_p1",
        "b16_p1",
        "b16_p1",
        "b8_p1",
        "b16",
        "b8",
    ]


@pytest.mark.parametrize("limits", [[16], [8, 8], [8, 0], [8, -1]])
def test_invalid_cells_fail_before_any_external_process(
    mock_sweep: MockSweep, limits: list[int]
) -> None:
    args, commands, _, _ = mock_sweep
    args.batch_limits = limits
    with pytest.raises(ValueError):
        sweep.run_sweep(args)
    assert not commands
    assert not args.output.exists()


@pytest.mark.parametrize("failure", ["solver_work", "admission", "binary_change"])
def test_failed_samples_retain_raw_observations_without_completed_summary(
    mock_sweep: MockSweep, monkeypatch: pytest.MonkeyPatch, failure: str
) -> None:
    args, _, records, process = mock_sweep

    def failing_process(command: list[str], **kwargs: object) -> None:
        process(command, **kwargs)
        if len(records) != 2:
            return
        if failure == "binary_change":
            args.library.write_bytes(b"changed library")
        elif failure == "solver_work":
            records[-1]["lambda_iterations"] = 22
        else:
            records[-1]["lambda_audit_matrix"] = False
        Path(command[2]).write_text(json.dumps(records[-1]))

    monkeypatch.setattr(sweep.subprocess, "run", failing_process)
    with pytest.raises((ValueError, RuntimeError)):
        sweep.run_sweep(args)
    assert len((args.output / "observations.jsonl").read_text().splitlines()) == 1
    assert not (args.output / "matched-summary.json").exists()
    assert not (args.output / "screen-summary.json").exists()
