"""Negative outcomes still require complete, campaign-bound provenance."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import lzma
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

from benchmarks.hybrid_crossover_receipts import ReceiptBundle

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


def repack(root: Path, original: ReceiptBundle, members: dict[str, bytes]) -> None:
    """Reauthenticate all storage bindings so scientific checks must reject damage."""
    parts, rows, offset = [], [], 0
    for name, data in members.items():
        rows.append(
            {
                "path": name,
                "offset": offset,
                "bytes": len(data),
                "sha256": hashlib.sha256(data).hexdigest(),
            }
        )
        parts.append(data)
        offset += len(data)
    payload = b"".join(parts)
    encoded = lzma.compress(payload)
    storage = {
        **original.manifest["storage"],
        "bytes": len(encoded),
        "sha256": hashlib.sha256(encoded).hexdigest(),
        "content_bytes": len(payload),
        "content_sha256": hashlib.sha256(payload).hexdigest(),
    }
    manifest = {**original.manifest, "storage": storage, "files": rows}
    root.mkdir(parents=True)
    (root / storage["path"]).write_bytes(encoded)
    (root / "manifest.json").write_text(json.dumps(manifest))


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
    original = ReceiptBundle(module.HERE, module.required_paths())
    members = dict(original.members)
    members.pop(name)
    module.HERE = tmp_path / "evidence"
    repack(module.HERE, original, members)
    with pytest.raises(ValueError, match="missing or unexpected receipt member"):
        module.verify()


@pytest.mark.parametrize("identity", ["revision", "dirty", "library"])
def test_failed_profile_identity_cannot_hide_in_incomplete_pair(
    tmp_path: Path, identity: str
) -> None:
    module = verifier()
    original = ReceiptBundle(module.HERE, module.required_paths())
    members = dict(original.members)
    name = "profiles/water-96-direct-profile.json.gz"
    profile = json.loads(members[name])
    assert profile["status"] == "FAILED"
    if identity == "revision":
        profile["source"]["revision"] = "0" * 40
    elif identity == "dirty":
        profile["source"]["dirty"] = True
    else:
        profile["library"]["sha256"] = "0" * 64
    members[name] = json.dumps(profile).encode()
    module.HERE = tmp_path / "evidence"
    repack(module.HERE, original, members)
    with pytest.raises(ValueError, match="diagnostic profile source/library"):
        module.verify()
