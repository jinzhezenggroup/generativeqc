"""Compaction edits only authenticated storage leaves and fails closed."""

from __future__ import annotations

import gzip
import json
from pathlib import Path

import pytest
from test_compact_evidence_safety import publication, snapshot

from tools import compact_evidence_publications as compact
from tools.generativeqc_validation.storage_json import (
    decode_storage_json,
    replace_metadata,
)

__all__ = ["publication"]


def update(publication: Path, name: str, raw: bytes) -> None:
    directory = publication / "campaign"
    (directory / name).write_bytes(raw)
    path = directory / "publication.json"
    record = json.loads(path.read_bytes())
    entry = next(entry for entry in record["files"] if entry["path"] == name)
    entry.update(bytes=len(raw), sha256=compact.digest(raw))
    path.write_bytes(compact.json_bytes(record))


def test_scientific_number_tokens_and_strings_are_byte_preserved(
    publication: Path,
) -> None:
    evidence = publication / "campaign/evidence.json"
    original = evidence.read_bytes()
    tokens = b'"precise":1.23456789012345678901234567890,"tiny":1e-9999,"big":1e9999,"signed":-0.000,"escaped":"\\u03b1",'
    original = b"{" + tokens + original[1:]
    update(publication, "evidence.json", original)
    compact.main([])
    packed = gzip.decompress(evidence.with_suffix(".json.gz").read_bytes())
    assert tokens in packed
    old = decode_storage_json(original)
    new = decode_storage_json(packed)
    del old["attachments"], new["attachments"]
    assert old == new


@pytest.mark.parametrize(
    "bad", [b'{"x":1,"x":2}', b'{"x":NaN}', b"[Infinity]", b'{"x":']
)
def test_invalid_json_rejected_before_any_mutation(
    publication: Path, bad: bytes
) -> None:
    update(publication, "samples.json", bad)
    before = snapshot(publication)
    with pytest.raises(ValueError):
        compact.main([])
    assert snapshot(publication) == before


@pytest.mark.parametrize(
    "damage",
    [
        "schema",
        "role",
        "size-type",
        "mixed",
        "duplicate-key",
        "reference-hash",
        "reference-size",
        "reference-outside",
        "mixed-parts",
    ],
)
def test_malformed_storage_contract_is_failure_closed(
    publication: Path, damage: str
) -> None:
    path = publication / "campaign/publication.json"
    manifest = json.loads(path.read_bytes())
    if damage == "schema":
        manifest["schema"] = "unreviewed.storage.v9"
    elif damage == "role":
        manifest["files"][0]["role"] = "unknown"
    elif damage == "size-type":
        manifest["files"][0]["bytes"] = True
    elif damage == "mixed":
        entry = manifest["files"][0].copy()
        entry["path"] += ".gz"
        raw = gzip.compress(
            (publication / "campaign/samples.json").read_bytes(), mtime=0
        )
        (publication / "campaign" / entry["path"]).write_bytes(raw)
        entry.update(bytes=len(raw), sha256=compact.digest(raw))
        manifest["files"].append(entry)
    elif damage == "duplicate-key":
        path.write_bytes(b'{"schema":"ignored",' + compact.json_bytes(manifest)[1:])
    else:
        evidence = json.loads((publication / "campaign/evidence.json").read_bytes())
        if damage == "reference-hash":
            evidence["attachments"][0]["sha256"] = "0" * 64
        elif damage == "reference-size":
            evidence["attachments"][0]["bytes"] += 1
        elif damage == "reference-outside":
            evidence["attachments"][0]["path"] = "../outside.json"
        else:
            evidence["record_parts"] = {"energy": [evidence["attachments"][0]]}
        update(publication, "evidence.json", compact.json_bytes(evidence))
    if damage in {"schema", "role", "size-type", "mixed"}:
        path.write_bytes(compact.json_bytes(manifest))
    before = snapshot(publication)
    with pytest.raises(ValueError):
        compact.main([])
    assert snapshot(publication) == before


def test_truncated_packed_member_rejected_even_without_new_candidates(
    publication: Path,
) -> None:
    compact.main([])
    path = publication / "campaign/samples.json.gz"
    update(publication, "samples.json.gz", path.read_bytes()[:-4])
    before = snapshot(publication)
    with pytest.raises(ValueError, match="truncated"):
        compact.main([])
    assert snapshot(publication) == before


def test_review_failure_rolls_back_every_publication_member(
    publication: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    review = compact.REVIEW
    record = json.loads(review.read_bytes())
    # Force a real review-row retirement while preserving the original identity.
    samples = publication / "campaign/samples.json"
    record["files"] = [
        {
            "path": "campaign/samples.json",
            "bytes": samples.stat().st_size,
            "sha256": compact.digest(samples.read_bytes()),
        }
    ]
    review.write_bytes(compact.json_bytes(record))
    before = snapshot(publication)
    replace = Path.replace
    injected = False

    def fail_review(path: Path, target: Path) -> Path:
        nonlocal injected
        if Path(target) == review and not injected:
            injected = True
            raise OSError("injected review commit failure")
        return replace(path, target)

    monkeypatch.setattr(Path, "replace", fail_review)
    with pytest.raises(OSError, match="review commit"):
        compact.main([])
    assert injected
    assert snapshot(publication) == before


def test_short_staged_output_never_retires_original(
    publication: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    before = snapshot(publication)
    write = Path.write_bytes

    def short_write(path: Path, data: bytes) -> int:
        return write(path, data[:2] if path.name.startswith("new-") else data)

    monkeypatch.setattr(Path, "write_bytes", short_write)
    with pytest.raises(OSError, match="incomplete staged output"):
        compact.main([])
    assert snapshot(publication) == before


def test_unfinished_transaction_requires_manual_inspection(publication: Path) -> None:
    recovery = publication / ".evidence-stage-interrupted"
    recovery.mkdir()
    (recovery / "old-0").write_bytes(b"original measurement")
    before = snapshot(publication)
    with pytest.raises(ValueError, match="recovery directory"):
        compact.main([])
    assert snapshot(publication) == before


def test_leaf_patches_handle_nested_arrays_and_reject_missing_targets() -> None:
    raw = b'{"a":[{},[],{"path":"old","raw":1.000000000000000000001}],"z":true}'
    assert replace_metadata(raw, {("a", 2, "path"): "new"}) == raw.replace(
        b'"old"', b'"new"'
    )
    with pytest.raises(ValueError, match="scalar leaf"):
        replace_metadata(raw, {("absent",): 4})


@pytest.mark.parametrize("operation", ["unlink", "replace"])
def test_interrupt_after_filesystem_side_effect_restores_all_bytes(
    publication: Path, monkeypatch: pytest.MonkeyPatch, operation: str
) -> None:
    before = snapshot(publication)
    unlink, replace = Path.unlink, Path.replace
    injected = False

    def remove(path: Path, missing_ok: bool = False) -> None:
        nonlocal injected
        unlink(path, missing_ok=missing_ok)
        if operation == "unlink" and path.name == "samples.json" and not injected:
            injected = True
            raise KeyboardInterrupt("after unlink")

    def move(path: Path, target: Path) -> Path:
        nonlocal injected
        result = replace(path, target)
        if (
            operation == "replace"
            and Path(target).name == "publication.json"
            and not injected
        ):
            injected = True
            raise KeyboardInterrupt("after replace")
        return result

    monkeypatch.setattr(Path, "unlink", remove)
    monkeypatch.setattr(Path, "replace", move)
    with pytest.raises(KeyboardInterrupt, match="after"):
        compact.main([])
    assert injected
    assert snapshot(publication) == before


@pytest.mark.parametrize("codec", ["gzip", "xz"])
@pytest.mark.parametrize("damage", ["bomb", "trailing", "concatenated", "truncated"])
def test_bounded_single_stream_decompression(
    monkeypatch: pytest.MonkeyPatch, codec: str, damage: str
) -> None:
    import lzma

    encode = gzip.compress if codec == "gzip" else lzma.compress
    suffix = ".gz" if codec == "gzip" else ".xz"
    raw = b"x" * (4096 if damage == "bomb" else 20)
    packed = encode(raw)
    monkeypatch.setattr(compact, "MAX_EXPANDED_BYTES", 128)
    if damage == "trailing":
        packed += b"untrusted trailer"
    elif damage == "concatenated":
        packed += encode(b"another stream")
    elif damage == "truncated":
        packed = packed[:-5]
    with pytest.raises(ValueError, match="expanded-byte|truncated|trailing"):
        compact._plain(packed, "samples.json" + suffix)


@pytest.mark.parametrize(
    "limit",
    [
        "MAX_STORED_BYTES",
        "MAX_TOTAL_STORED",
        "MAX_TOTAL_EXPANDED",
        "MAX_MEMBERS",
        "MAX_BUNDLE_MEMBERS",
    ],
)
def test_resource_limit_failure_cannot_mutate(
    publication: Path, monkeypatch: pytest.MonkeyPatch, limit: str
) -> None:
    before = snapshot(publication)
    monkeypatch.setattr(compact, limit, 1)
    with pytest.raises(ValueError):
        compact.main([])
    assert snapshot(publication) == before


def test_json_depth_and_token_limits_before_decoding(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tools.generativeqc_validation import storage_json

    with pytest.raises(ValueError, match="nesting/token"):
        decode_storage_json(b"[" * 65 + b"0" + b"]" * 65)
    monkeypatch.setattr(storage_json, "MAX_JSON_TOKENS", 8)
    with pytest.raises(ValueError, match="nesting/token"):
        decode_storage_json(b"[" + b"1," * 10 + b"2]")


def test_exact_existing_companion_is_reused_without_overwrite(
    publication: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = publication / "campaign/samples.json"
    companion = source.with_suffix(".json.gz")
    raw = source.read_bytes()
    companion.write_bytes(compact.compressed(raw))
    original_stat = companion.stat()
    compact.main([])
    assert companion.stat().st_ino == original_stat.st_ino
    assert companion.stat().st_mtime_ns == original_stat.st_mtime_ns
    assert gzip.decompress(companion.read_bytes()) == raw
    assert not source.exists()


def test_reused_companion_survives_other_write_failure(
    publication: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = publication / "campaign/samples.json"
    companion = source.with_suffix(".json.gz")
    companion.write_bytes(compact.compressed(source.read_bytes()))
    before = snapshot(publication)
    replace = Path.replace

    def fail(path: Path, target: Path) -> Path:
        if Path(target).name == "publication.json":
            raise OSError("manifest failure")
        return replace(path, target)

    monkeypatch.setattr(Path, "replace", fail)
    with pytest.raises(OSError, match="manifest failure"):
        compact.main([])
    assert snapshot(publication) == before


@pytest.mark.parametrize(
    "field,limit",
    [
        ("stored", "MAX_TOTAL_STORED"),
        ("expanded", "MAX_TOTAL_EXPANDED"),
        ("members", "MAX_MEMBERS"),
    ],
)
def test_later_bundle_budget_failure_cannot_commit_earlier_bundle(
    publication: Path, monkeypatch: pytest.MonkeyPatch, field: str, limit: str
) -> None:
    import shutil

    first = "campaign/publication.json"
    budget = compact.Budget()
    compact.plan_publication(first, budget=budget)
    shutil.copytree(publication / "campaign", publication / "second")
    monkeypatch.setattr(
        compact, "publication_paths", lambda: (first, "second/publication.json")
    )
    monkeypatch.setattr(compact, limit, getattr(budget, field) + 1)
    before = snapshot(publication)
    with pytest.raises(ValueError, match="batch exceeds"):
        compact.main([])
    assert snapshot(publication) == before


def test_gzip_inflation_cannot_mutate_a_publication(
    publication: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    compact.main([])
    update(
        publication,
        "samples.json.gz",
        gzip.compress(b"[" + b"1," * 5000 + b"2]", mtime=0),
    )
    monkeypatch.setattr(compact, "MAX_EXPANDED_BYTES", 512)
    before = snapshot(publication)
    with pytest.raises(ValueError, match="expanded-byte"):
        compact.main([])
    assert snapshot(publication) == before
