"""Packed archival receipts retain exact bytes and fail closed on corruption."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import lzma
from pathlib import Path

import pytest

from benchmarks.hybrid_crossover_receipts import (
    FORMAT,
    MAX_PAYLOAD_BYTES,
    ReceiptBundle,
)

ROOT = Path(__file__).resolve().parents[2]


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def pack(root: Path, members: dict[str, bytes]) -> dict:
    """Create synthetic data, with no dependency on production source hashes."""
    parts, rows, offset = [], [], 0
    for name, data in members.items():
        rows.append(
            {
                "path": name,
                "offset": offset,
                "bytes": len(data),
                "sha256": digest(data),
            }
        )
        parts.append(data)
        offset += len(data)
    payload = b"".join(parts)
    encoded = lzma.compress(payload)
    manifest = {
        "storage": {
            "format": FORMAT,
            "path": "receipts.txt.xz",
            "bytes": len(encoded),
            "sha256": digest(encoded),
            "content_bytes": len(payload),
            "content_sha256": digest(payload),
        },
        "files": rows,
    }
    root.mkdir(parents=True, exist_ok=True)
    (root / "receipts.txt.xz").write_bytes(encoded)
    (root / "manifest.json").write_text(json.dumps(manifest))
    return manifest


def test_exact_named_bytes_need_no_extraction(tmp_path: Path) -> None:
    members = {"native/a.json.gz": b'{"x": 1.2300}\n', "trace/b": b"\x00\r\n"}
    pack(tmp_path, members)
    bundle = ReceiptBundle(tmp_path, set(members))
    assert bundle.members == members
    assert bundle.read("native/a.json.gz") == {"x": 1.23}
    assert {p.name for p in tmp_path.iterdir()} == {
        "manifest.json",
        "receipts.txt.xz",
    }


@pytest.mark.parametrize(
    "damage",
    [
        "corrupt-container",
        "truncated-container",
        "trailing-stream",
        "trailing-bytes",
        "duplicate-name",
        "missing-member",
        "unexpected-member",
        "overlap",
        "gap",
        "negative-length",
        "boolean-length",
        "fractional-length",
        "boolean-offset",
        "fractional-offset",
        "wrong-member-hash",
        "unused-payload",
        "declared-too-small",
        "declared-too-large",
        "excessive-payload",
        "boolean-container-size",
        "negative-container-size",
        "absolute-member",
        "traversal-member",
        "traversal-container",
        "link-member",
        "extra-file",
        "missing-container",
        "duplicate-json-key",
    ],
)
def test_invalid_storage_is_rejected(tmp_path: Path, damage: str) -> None:
    members = {"native/a.json.gz": b'{"x":1}\n', "trace/b": b"trace\n"}
    manifest = pack(tmp_path, members)
    storage, rows = manifest["storage"], manifest["files"]
    archive = tmp_path / storage["path"]
    if damage == "corrupt-container":
        archive.write_bytes(archive.read_bytes() + b"changed")
    elif damage in {"truncated-container", "trailing-stream", "trailing-bytes"}:
        encoded = archive.read_bytes()
        encoded = (
            encoded[:-1]
            if damage == "truncated-container"
            else encoded
            + (lzma.compress(b"foreign") if damage == "trailing-stream" else b"x")
        )
        archive.write_bytes(encoded)
        storage.update(bytes=len(encoded), sha256=digest(encoded))
    elif damage == "duplicate-name":
        rows[1]["path"] = rows[0]["path"]
    elif damage == "missing-member":
        rows.pop()
    elif damage == "unexpected-member":
        rows[1]["path"] = "foreign"
    elif damage in {"overlap", "gap"}:
        rows[1]["offset"] += -1 if damage == "overlap" else 1
    elif damage in {"negative-length", "boolean-length", "fractional-length"}:
        rows[0]["bytes"] = {
            "negative-length": -1,
            "boolean-length": True,
            "fractional-length": 1.5,
        }[damage]
    elif damage in {"boolean-offset", "fractional-offset"}:
        rows[0]["offset"] = False if damage == "boolean-offset" else 0.0
    elif damage == "wrong-member-hash":
        rows[0]["sha256"] = "0" * 64
    elif damage == "unused-payload":
        rows[1]["bytes"] -= 1
        rows[1]["sha256"] = digest(members["trace/b"][:-1])
    elif damage in {"declared-too-small", "declared-too-large"}:
        storage["content_bytes"] += -1 if damage == "declared-too-small" else 1
    elif damage == "excessive-payload":
        storage["content_bytes"] = MAX_PAYLOAD_BYTES + 1
    elif damage in {"boolean-container-size", "negative-container-size"}:
        storage["bytes"] = True if damage == "boolean-container-size" else -1
    elif damage in {"absolute-member", "traversal-member"}:
        rows[0]["path"] = "/outside" if damage == "absolute-member" else "../outside"
    elif damage == "traversal-container":
        storage["path"] = "../receipts.txt.xz"
    elif damage == "link-member":
        rows[0]["linkname"] = "../outside"
    elif damage == "extra-file":
        (tmp_path / "unexpected").write_bytes(b"x")
    elif damage == "missing-container":
        archive.unlink()
    text = json.dumps(manifest)
    if damage == "duplicate-json-key":
        text = text.replace('"offset": 0', '"offset": 0, "offset": 0', 1)
    (tmp_path / "manifest.json").write_text(text)
    with pytest.raises((ValueError, lzma.LZMAError)):
        ReceiptBundle(tmp_path, set(members))


@pytest.mark.parametrize("target", ["manifest.json", "receipts.txt.xz"])
def test_storage_links_are_rejected(tmp_path: Path, target: str) -> None:
    root = tmp_path / "evidence"
    members = {"a": b"receipt"}
    pack(root, members)
    source = root / target
    destination = tmp_path / target
    source.rename(destination)
    try:
        source.symlink_to(destination)
    except OSError:
        pytest.skip("creating symlinks is not available on this host")
    with pytest.raises(ValueError, match="link"):
        ReceiptBundle(root, set(members))


@pytest.mark.parametrize(
    "campaign, members, pairs",
    [
        ("hybrid-provider-crossover-20261013", 56, 6),
        ("hybrid-provider-crossover-2054", 25, 4),
    ],
)
def test_retained_scientific_gates(campaign: str, members: int, pairs: int) -> None:
    path = ROOT / "benchmarks/results" / campaign / "verify_evidence.py"
    spec = importlib.util.spec_from_file_location(campaign, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    result = module.verify()
    assert result["members"] == members
    assert len(result["outcomes"]) == pairs
    assert result["crossover_claim_eligible"] is False


@pytest.mark.parametrize(
    "damage",
    [
        "foreign-oracle",
        "lost-scheduler-receipt",
        "foreign-failed-profile-source",
        "dirty-failed-profile-source",
        "foreign-failed-profile-library",
    ],
)
def test_container_cannot_replace_scientific_identity(
    tmp_path: Path, damage: str
) -> None:
    path = (
        ROOT
        / "benchmarks/results/hybrid-provider-crossover-20261013/verify_evidence.py"
    )
    spec = importlib.util.spec_from_file_location("packed_scientific_control", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    original = ReceiptBundle(module.HERE, module.required_paths())
    members = dict(original.members)
    if damage == "foreign-oracle":
        name = "oracles/formaldehyde-direct-0.json.gz"
        oracle = json.loads(members[name])
        oracle["native_record_sha256"] = "0" * 64
        members[name] = json.dumps(oracle).encode()
    elif damage == "lost-scheduler-receipt":
        members.pop("platform/slurm.txt")
    else:
        name = "profiles/water-96-direct-profile.json.gz"
        profile = json.loads(members[name])
        assert profile["status"] == "FAILED"
        if damage == "foreign-failed-profile-source":
            profile["source"]["revision"] = "0" * 40
        elif damage == "dirty-failed-profile-source":
            profile["source"]["dirty"] = True
        else:
            profile["library"]["sha256"] = "0" * 64
        members[name] = json.dumps(profile).encode()
    packed = pack(tmp_path, members)
    manifest = {**original.manifest, **packed}
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    module.HERE = tmp_path
    with pytest.raises(ValueError):
        module.verify()
