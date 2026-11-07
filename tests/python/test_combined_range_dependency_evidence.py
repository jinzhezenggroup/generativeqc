"""Late source integration preserves its own default-only acceptance gates."""

from __future__ import annotations

import json
import lzma
import subprocess
import sys
from pathlib import Path

import pytest

DIRECTORY = (
    Path(__file__).resolve().parents[2]
    / "benchmarks/results/combined-range-build-20261107"
)


@pytest.mark.parametrize(
    "corruption",
    ["valid", "force", "work", "image", "selector", "timing", "source", "oracle"],
)
def test_integrated_default_gates_remain_active_under_optimized_python(
    tmp_path: Path, corruption: str
) -> None:
    """Do not relabel the earlier pair or trade scientific work for a new default."""
    evidence = json.loads(
        lzma.decompress((DIRECTORY / "dependency-evidence.json.xz").read_bytes())
    )
    population = evidence["devices"]["A100"]
    record = population["record"]
    sample = record["samples"][0]
    if corruption == "force":
        sample["forces"][0][0] += 1e-3
    elif corruption == "work":
        sample["ks_diagnostic"]["coulomb_exchange_work"]["exchange_applications"] *= 2
    elif corruption == "image":
        record["source_revision"] = "invalid"
    elif corruption == "selector":
        record["source_schedule_experiment"]["combined_range_build"] = "1"
    elif corruption == "timing":
        sample["seconds"] *= 2
    elif corruption == "source":
        next(
            entry
            for entry in sample["precision"]["operators"]
            if entry["name"] == "coulomb_recurrence"
        )["count"] *= 2
    elif corruption == "oracle":
        population["oracle_log"] = "missing independent oracle"
    (tmp_path / "dependency-evidence.json.xz").write_bytes(
        lzma.compress(json.dumps(evidence).encode())
    )
    command = (
        "import runpy; from pathlib import Path; "
        f"reader=runpy.run_path({str(DIRECTORY / 'replay_dependency.py')!r}); "
        f"reader['replay'](Path({str(tmp_path)!r}))"
    )
    result = subprocess.run(
        [sys.executable, "-O", "-c", command],
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    if corruption == "valid":
        assert result.returncode == 0, result.stderr
    else:
        assert result.returncode != 0 and "ValueError" in result.stderr
