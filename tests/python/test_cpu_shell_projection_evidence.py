"""Keep selected CPU scalar evidence hash-bound without claiming raw recovery."""

import hashlib
import importlib.util
import json
import shutil
from pathlib import Path
from types import ModuleType

import pytest

from tools.generativeqc_validation.publication import validate_publication

ROOT = Path(__file__).resolve().parents[2]
CAPSULE = ROOT / "benchmarks/results/cpu-shell-projection-20261003"


def decoder() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "shell_projection_decoder", CAPSULE / "decode.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_shell_projection_publication_retains_scope_and_counts() -> None:
    publication = json.loads((CAPSULE / "publication.json").read_text())
    members = {
        r["path"]: (CAPSULE / r["path"]).read_bytes() for r in publication["files"]
    }
    validate_publication(publication, members)
    assert publication["decision"]["status"] == "inconclusive"
    expected = {
        "evidence.json.xz": "35c5b80c8b46fae24fa9cf806cfe4ed17351f2600dd3124bf1118dcbad92f7d8",
        "reproducer.json.xz": "66022d3f8a81bedcd6c0dd7e342812beb47143c79bf88c9f964f5cc08732c9d0",
        "independent-review.md.xz": "3418a18c94a93d1a524c686f0635ceac8f0d16f7cf4cf7ef52c6065719d9d743",
    }
    for name, digest in expected.items():
        assert hashlib.sha256(members[name]).hexdigest() == digest
    expanded, counts = decoder().load(CAPSULE)
    assert counts == {
        "primary_processes": 54,
        "primary_endpoints": 216,
        "historical_balanced_endpoints": 432,
        "intrusive_endpoints": 108,
        "historical_failed_dft_attempts": 13,
    }
    assert json.loads(expanded["integration.json"])["independent_prepared"]["passed"]


def test_shell_projection_decoder_rejects_changed_member(tmp_path: Path) -> None:
    target = tmp_path / "capsule"
    shutil.copytree(CAPSULE, target)
    (target / "README.md").write_text("changed")
    with pytest.raises(ValueError, match="size/checksum mismatch"):
        decoder().load(target)


def test_shell_projection_decoder_rejects_duplicate_member(tmp_path: Path) -> None:
    target = tmp_path / "capsule"
    shutil.copytree(CAPSULE, target)
    path = target / "publication.json"
    manifest = json.loads(path.read_text())
    manifest["files"].append(manifest["files"][0])
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="duplicate or self-referential"):
        decoder().load(target)
