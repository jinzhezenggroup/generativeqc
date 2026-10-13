"""Negative outcomes still require complete, campaign-bound provenance."""

from __future__ import annotations

import gzip
import hashlib
import importlib.util
import json
import shutil
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from types import ModuleType

ROOT = Path(__file__).resolve().parents[2]
CAMPAIGN = ROOT / "benchmarks/results/hybrid-provider-crossover-20261013"


def verifier() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "crossover_provenance_control", CAMPAIGN / "verify_evidence.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_authentic_negative_outcomes_remain_incomplete() -> None:
    result = verifier().verify()
    assert result["members"] == 56
    assert list(result["outcomes"].values()).count("INCOMPLETE") == 4
    assert list(result["outcomes"].values()).count("PILOT_ACCEPTED") == 2
    assert result["crossover_claim_eligible"] is False


@pytest.mark.parametrize(
    "name",
    [
        "native/campaign.json.gz",
        "build/build-identity.json.gz",
        "platform/slurm.txt",
        "traces/water-96-direct-trace.jsonl.gz",
    ],
)
def test_removed_provenance_cannot_be_reauthenticated(
    tmp_path: Path, name: str
) -> None:
    module = verifier()
    shutil.copytree(CAMPAIGN / "evidence", tmp_path / "evidence")
    module.HERE = tmp_path / "evidence"
    manifest_path = module.HERE / "manifest.json"
    manifest = json.loads(manifest_path.read_bytes())
    manifest["files"] = [row for row in manifest["files"] if row["path"] != name]
    manifest_path.write_text(json.dumps(manifest))
    (module.HERE / name).unlink()
    with pytest.raises(ValueError, match="missing, duplicate or unauthenticated"):
        module.verify()


@pytest.mark.parametrize("identity", ["revision", "dirty", "library"])
def test_failed_profile_identity_cannot_hide_in_incomplete_pair(
    tmp_path: Path, identity: str
) -> None:
    module = verifier()
    shutil.copytree(CAMPAIGN / "evidence", tmp_path / "evidence")
    module.HERE = tmp_path / "evidence"
    name = "profiles/water-96-direct-profile.json.gz"
    path = module.HERE / name
    profile = json.loads(gzip.decompress(path.read_bytes()))
    assert profile["status"] == "FAILED"
    if identity == "revision":
        profile["source"]["revision"] = "0" * 40
    elif identity == "dirty":
        profile["source"]["dirty"] = True
    else:
        profile["library"]["sha256"] = "0" * 64
    content = json.dumps(profile).encode()
    encoded = gzip.compress(content, mtime=0)
    path.write_bytes(encoded)
    manifest_path = module.HERE / "manifest.json"
    manifest = json.loads(manifest_path.read_bytes())
    entry = next(row for row in manifest["files"] if row["path"] == name)
    entry.update(
        bytes=len(encoded),
        sha256=hashlib.sha256(encoded).hexdigest(),
        content_bytes=len(content),
        content_sha256=hashlib.sha256(content).hexdigest(),
    )
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="diagnostic profile source/library"):
        module.verify()
