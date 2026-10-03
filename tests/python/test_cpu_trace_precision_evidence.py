"""Preserve the scoped trace evidence and fail-closed local decoder contract."""

import hashlib
import importlib.util
import json
import shutil
from pathlib import Path
from types import ModuleType

import pytest

from tools.generativeqc_validation.publication import validate_publication

ROOT = Path(__file__).resolve().parents[2]
CAPSULE = ROOT / "benchmarks/results/cpu-trace-precision-20261003"


def decoder() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "trace_evidence_decoder", CAPSULE / "decode_trace_evidence.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_trace_publication_is_scoped_and_hash_bound() -> None:
    publication = json.loads((CAPSULE / "publication.json").read_text())
    files = {
        row["path"]: (CAPSULE / row["path"]).read_bytes()
        for row in publication["files"]
    }
    validate_publication(publication, files)
    assert publication["decision"]["scope"] == "numerical"
    assert hashlib.sha256(files["scientific-records.json.gz"]).hexdigest() == (
        "e498bcb8535d0357670a1988b2be1d55cf0cb0e5d862583dda9455b6633e69a0"
    )
    manifest, expanded = decoder().load(CAPSULE)
    assert "NOT lossless" in manifest["contract"]
    assert len([path for path in expanded if path.startswith("harnesses/")]) == 6
    summary = json.loads(expanded["summary.json"])
    assert summary["retained_attempts"]
    assert summary["original_pbe96"]["settings"]["max_iterations"] == 150
    assert summary["original_pbe96"]["settings"]["energy_tolerance"] == 1e-12
    assert summary["original_pbe96"]["settings"]["density_tolerance"] == 1e-10


def test_trace_decoder_rejects_modified_member(tmp_path: Path) -> None:
    target = tmp_path / "capsule"
    shutil.copytree(CAPSULE, target)
    (target / "README.txt").write_text("modified")
    with pytest.raises(ValueError, match="capsule member changed"):
        decoder().load(target)


def test_trace_decoder_rejects_unsafe_member(tmp_path: Path) -> None:
    target = tmp_path / "capsule"
    shutil.copytree(CAPSULE, target)
    path = target / "capsule.json"
    manifest = json.loads(path.read_text())
    manifest["files"]["../escape"] = {"bytes": 0, "sha256": "0" * 64}
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="unsafe capsule path"):
        decoder().load(target)
