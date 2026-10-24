"""Read-only publication checks bound live payloads without weakening mutations."""

from __future__ import annotations

import hashlib
import json
import weakref
from typing import TYPE_CHECKING

import pytest

from tools import compact_evidence_publications as compact

if TYPE_CHECKING:
    from pathlib import Path


def bundle(
    root: Path, name: str, *, payload: bytes = b'{"value": 1}\n', role: str = "summary"
) -> tuple[str, int]:
    folder = root / name
    folder.mkdir()
    (folder / "data.json").write_bytes(payload)
    manifest = json.dumps(
        {
            "schema": "generativeqc.benchmark-publication.v1",
            "files": [
                {
                    "path": "data.json",
                    "role": role,
                    "bytes": len(payload),
                    "sha256": hashlib.sha256(payload).hexdigest(),
                }
            ],
        }
    ).encode()
    (folder / "publication.json").write_bytes(manifest)
    return f"{name}/publication.json", len(manifest) + len(payload)


def snapshot(root: Path) -> dict[str, bytes]:
    return {str(p): p.read_bytes() for p in root.rglob("*") if p.is_file()}


@pytest.mark.parametrize(
    "limit", ["MAX_TOTAL_STORED", "MAX_TOTAL_EXPANDED", "MAX_MEMBERS"]
)
def test_check_streams_individually_bounded_publications(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, limit: str
) -> None:
    monkeypatch.setattr(compact, "ROOT", tmp_path)
    targets = [bundle(tmp_path, name) for name in ("one", "two", "three")]
    monkeypatch.setattr(
        compact, limit, 2 if limit == "MAX_MEMBERS" else max(n for _, n in targets)
    )
    before = snapshot(tmp_path)
    original = compact.plan_publication
    previous = []

    def observed(target: str, *, strict: bool, budget: compact.Budget) -> compact.Plan:
        assert all(ref() is None for ref in previous)
        plan = original(target, strict=strict, budget=budget)
        previous.append(weakref.ref(plan))
        return plan

    monkeypatch.setattr(compact, "plan_publication", observed)
    compact.main(
        ["--check", *[arg for path, _ in targets for arg in ("--publication", path)]]
    )
    assert len(previous) == 3
    assert all(ref() is None for ref in previous)
    assert snapshot(tmp_path) == before


@pytest.mark.parametrize(
    "limit", ["MAX_TOTAL_STORED", "MAX_TOTAL_EXPANDED", "MAX_MEMBERS"]
)
def test_check_still_rejects_one_oversized_publication(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, limit: str
) -> None:
    monkeypatch.setattr(compact, "ROOT", tmp_path)
    path, size = bundle(tmp_path, "one")
    monkeypatch.setattr(compact, limit, 1 if limit == "MAX_MEMBERS" else size - 1)
    before = snapshot(tmp_path)
    with pytest.raises(ValueError, match="resource limits"):
        compact.main(["--check", "--publication", path])
    assert snapshot(tmp_path) == before


@pytest.mark.parametrize("failure", ["identity", "uncompacted", "duplicate"])
def test_check_rejects_late_bad_or_overlapping_publication(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failure: str
) -> None:
    monkeypatch.setattr(compact, "ROOT", tmp_path)
    first, _ = bundle(tmp_path, "one")
    second, _ = bundle(
        tmp_path,
        "two",
        payload=json.dumps({"padding": "x" * (compact.THRESHOLD + 1)}).encode()
        if failure == "uncompacted"
        else b'{"value": 2}\n',
        role="samples" if failure == "uncompacted" else "summary",
    )
    if failure == "identity":
        (tmp_path / "two/data.json").write_bytes(b"changed")
    if failure == "duplicate":
        second = first
    before = snapshot(tmp_path)
    with pytest.raises(
        (ValueError, SystemExit),
        match={
            "identity": "identity mismatch",
            "uncompacted": "not compacted",
            "duplicate": "overlapping publication inventories",
        }[failure],
    ):
        compact.main(["--check", "--publication", first, "--publication", second])
    assert snapshot(tmp_path) == before


def test_manual_mutation_keeps_shared_batch_budget(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(compact, "ROOT", tmp_path)
    one, size = bundle(tmp_path, "one")
    two, _ = bundle(tmp_path, "two")
    monkeypatch.setattr(compact, "MAX_TOTAL_STORED", size)
    before = snapshot(tmp_path)
    with pytest.raises(ValueError, match="resource limits"):
        compact.main(["--publication", one, "--publication", two])
    assert snapshot(tmp_path) == before
