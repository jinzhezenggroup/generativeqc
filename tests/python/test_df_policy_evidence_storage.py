"""Historical DF endpoint compression preserves full scientific records."""

import gzip
import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from tools.generativeqc_validation.record import load_json, load_record
from tools.generativeqc_validation.retention import digest

ROOT = Path(__file__).resolve().parents[2]
DIRECTORY = ROOT / "benchmarks/results/issue404-407-df"


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
