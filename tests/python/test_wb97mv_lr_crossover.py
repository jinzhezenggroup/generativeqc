"""Crossover receipts preserve failures and the caller's execution controls."""

from __future__ import annotations

import json
import os
from typing import TYPE_CHECKING, Any

import pytest

from benchmarks import wb97mv_lr_crossover as runner

if TYPE_CHECKING:
    from pathlib import Path


def arguments(output: Path, response: str = "auto", selector: str = "df") -> list[str]:
    """Supply a scheduled campaign without importing a GPU runtime."""
    return [
        "--selector",
        selector,
        "--response",
        response,
        "--auxiliary",
        "auxiliary.json",
        "--source-revision",
        "test-source",
        "--output",
        str(output),
    ]


@pytest.mark.parametrize("response", ["auto", "generic", "shell"])
@pytest.mark.parametrize("failed", [False, True])
@pytest.mark.parametrize("selector", ["default", "auto", "df", "direct"])
def test_campaign_restores_controls_and_retains_terminal_state(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    response: str,
    failed: bool,
    selector: str,
) -> None:
    """Automatic selection must unset inherited overrides only for this campaign."""
    monkeypatch.setenv("SLURM_JOB_ID", "test-allocation")
    monkeypatch.setenv("GENERATIVEQC_DF_RANGE_EXCHANGE", "direct")
    monkeypatch.setenv("GENERATIVEQC_DF_WEIGHTED_EXECUTION", "inherited")
    output = tmp_path / "campaign.json"

    def measure(options: Any, record: dict, retain: Any) -> None:
        assert os.environ.get("GENERATIVEQC_DF_RANGE_EXCHANGE") == (
            None if selector == "default" else selector
        )
        assert os.environ.get("GENERATIVEQC_DF_WEIGHTED_EXECUTION") == (
            None if response == "auto" else response
        )
        record["samples"].append({"phase": "cold", "seconds": 1.0})
        retain("warm-1")
        if failed:
            raise RuntimeError("independent endpoint failed")

    monkeypatch.setattr(runner, "_measure", measure)
    if failed:
        with pytest.raises(RuntimeError, match="independent endpoint failed"):
            runner.main(arguments(output, response, selector))
    else:
        runner.main(arguments(output, response, selector))
    assert os.environ["GENERATIVEQC_DF_RANGE_EXCHANGE"] == "direct"
    assert os.environ["GENERATIVEQC_DF_WEIGHTED_EXECUTION"] == "inherited"
    record = json.loads(output.read_text())
    assert record["status"] == ("failed" if failed else "complete")
    assert record["samples"] == [{"phase": "cold", "seconds": 1.0}]
    if failed:
        assert record["failure"]["stage"] == "warm-1"


def test_existing_receipt_is_never_overwritten(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("SLURM_JOB_ID", "test-allocation")
    output = tmp_path / "campaign.json"
    output.write_text("retained negative evidence")
    with pytest.raises(FileExistsError):
        runner.main(arguments(output))
    assert output.read_text() == "retained negative evidence"


def test_gpu_campaign_requires_scheduler(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("SLURM_JOB_ID", raising=False)
    output = tmp_path / "campaign.json"
    with pytest.raises(SystemExit):
        runner.main(arguments(output))
    assert not output.exists()
