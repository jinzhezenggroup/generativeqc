"""Historical DF endpoint compression preserves full scientific records."""

import gzip
import json
import lzma
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from tools.generativeqc_validation.record import load_json, load_record
from tools.generativeqc_validation.retention import digest

ROOT = Path(__file__).resolve().parents[2]
DIRECTORY = ROOT / "benchmarks/results/issue404-407-df"


def test_combined_rys_summary_archive_preserves_original_bytes() -> None:
    """Make evidence room without dropping or reserializing historical data."""
    path = ROOT / "benchmarks/results/issue404-combined-rys/summary.json.xz"
    expanded = lzma.decompress(path.read_bytes())
    assert len(expanded) == 131267
    assert digest(expanded) == (
        "c584f057cea4a1b5e4ac898ed3c9af5d03ee90131ad73551f5d1bb49948a99a8"
    )


@pytest.mark.parametrize("backend", ["serial", "blas"])
def test_metric_gemv_96ao_archives_preserve_original_bytes(backend: str) -> None:
    """Keep historical manifest identities valid after storage-only compaction."""
    directory = ROOT / "benchmarks/results/issue206-metric-gemv/measurements"
    metadata = load_json(directory / f"96ao-b4-forces-{backend}.storage.json")
    stored = (ROOT / metadata["stored_path"]).read_bytes()
    assert len(stored) == metadata["stored_bytes"]
    assert digest(stored) == metadata["stored_sha256"]
    expanded = gzip.decompress(stored)
    assert len(expanded) == metadata["original_bytes"]
    assert digest(expanded) == metadata["original_sha256"]
    expected = {
        "serial": "65e477fb993ac17e3326861ec217c3cbad98bf320e9d31c56d21baaa95ce379d",
        "blas": "e2dcdac6b650e8c96cae94c11d70b8f5b0314e80864de142f5df6d7cccb809d7",
    }
    assert digest(expanded) == expected[backend]
    assert load_record(ROOT / metadata["stored_path"]) == json.loads(expanded)


@pytest.mark.parametrize("reader", [load_json, load_record])
def test_df_policy_endpoint_archives_preserve_complete_records(
    reader: Callable[[Path], Any],
) -> None:
    manifest = load_json(DIRECTORY / "storage.json")
    assert manifest["schema"] == "generativeqc.lossless-evidence-storage.v1"
    assert {entry["logical_path"] for entry in manifest["files"]} == {
        "384-404.json",
        "768-404.json",
    }
    observations = diagnostics = 0
    for entry in manifest["files"]:
        assert entry["path"] == entry["logical_path"] + ".gz"
        assert entry["encoding"] == "gzip"
        path = DIRECTORY / entry["path"]
        stored = path.read_bytes()
        assert len(stored) == entry["bytes"]
        assert digest(stored) == entry["sha256"]
        assert stored[4:8] == bytes(4)
        raw = gzip.decompress(stored)
        assert len(raw) == entry["decoded_bytes"]
        assert digest(raw) == entry["decoded_sha256"]
        data = reader(path)
        assert data == json.loads(raw)
        assert data["aos"] == int(entry["logical_path"].split("-")[0])
        assert len(data["samples"]) == 10
        assert len(data["diagnostics"]) == 2
        for row in [*data["samples"], *data["diagnostics"]]:
            assert row["energies_hartree"]
            assert row["forces_hartree_per_bohr"]
            assert row["convergence"]
        observations += len(data["samples"])
        diagnostics += len(data["diagnostics"])
    assert (observations, diagnostics) == (20, 4)
