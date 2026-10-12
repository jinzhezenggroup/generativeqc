"""Host-only controls for immutable CUDA HF evidence postprocessing."""

from __future__ import annotations

import gzip
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def runner(tmp_path_factory: pytest.TempPathFactory) -> object:
    """Load the reader without importing CUDA or creating a device context."""
    retained = ROOT / "benchmarks/results/cuda-hf-huckel-20261012/runner.py.gz"
    path = tmp_path_factory.mktemp("hf-guess-reader") / "runner.py"
    path.write_bytes(gzip.decompress(retained.read_bytes()))
    spec = importlib.util.spec_from_file_location("hf_guess_benchmark", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def evidence(tmp_path: Path, runner: object) -> Path:
    """Generate complete paired metadata; these are not numerical measurements."""
    reference = tmp_path / "reference-3.json"
    reference.write_text(
        json.dumps(
            {
                "schema": "generativeqc.cuda-hf-huckel.v1",
                "atoms": 3,
                "arm": "reference",
                "status": "measured",
            }
        )
    )
    for arm in ("core", "huckel"):
        for repeat in range(3):
            path = tmp_path / f"{arm}-3-{repeat}.json"
            density = path.with_suffix(".npy")
            np.save(density, np.eye(2))
            record = {
                "schema": "generativeqc.cuda-hf-huckel.v1",
                "atoms": 3,
                "arm": arm,
                "repeat": repeat,
                "status": "measured",
                "acceptance": {"gate": True},
                "reference_sha256": runner.digest(reference),
                "library_sha256": "fixed-library",
                "source_audit_sha256": "fixed-source",
                "basis_sha256": "fixed-basis",
                "harness_sha256": "fixed-harness",
                "gpu": {"pci_bus_id": "fixed-device"},
                "initial_guess": {
                    "outcome": "used",
                    "target_attempts": 1,
                    "work_counters_complete": True,
                    "preliminary_fock_builds": 7,
                    "preparation_seconds": 0.1,
                },
                "complete_seconds": 10.0 if arm == "core" else 9.0,
                "target_fock_builds": 20 if arm == "core" else 18,
                "final_density": {
                    "path": density.name,
                    "sha256": runner.digest(density),
                },
            }
            path.write_text(json.dumps(record))
    return tmp_path


def test_summary_does_not_parse_its_empty_redirected_output(
    evidence: Path, runner: object
) -> None:
    (evidence / "summary.json").write_text("")
    result = runner.summarize(evidence, (3,))
    assert result["cases"]["3"]["strict_five_percent_gate"]
    assert result["default_promoted"] is False


def test_positive_median_does_not_hide_an_unprofitable_pair(
    evidence: Path, runner: object
) -> None:
    path = evidence / "huckel-3-1.json"
    record = json.loads(path.read_text())
    record["complete_seconds"] = 9.6
    path.write_text(json.dumps(record))
    case = runner.summarize(evidence, (3,))["cases"]["3"]
    assert case["median_reduction"] > 0.05
    assert not case["strict_five_percent_gate"]


@pytest.mark.parametrize(
    "field,value",
    [
        ("status", "failed"),
        ("library_sha256", "other-library"),
        ("reference_sha256", "other-reference"),
        ("repeat", 3),
        ("acceptance", {"gate": False}),
    ],
)
def test_reader_keeps_identity_repeat_and_numerical_guards(
    evidence: Path, runner: object, field: str, value: object
) -> None:
    path = evidence / "huckel-3-0.json"
    record = json.loads(path.read_text())
    record[field] = value
    path.write_text(json.dumps(record))
    with pytest.raises(ValueError):
        runner.summarize(evidence, (3,))
