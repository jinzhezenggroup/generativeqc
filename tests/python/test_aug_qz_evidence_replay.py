"""Exercise authenticated campaign restoration and unchanged scientific gates."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from types import ModuleType

from benchmarks._exact_fp64_evidence import load

FOLDER = (
    Path(__file__).resolve().parents[2]
    / "benchmarks/results/full-df-aug-qz-3055-20261109"
)


def module(path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location("aug_qz_replay", path)
    assert spec is not None and spec.loader is not None
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def test_campaign_restores_every_authenticated_text_and_replays(tmp_path: Path) -> None:
    replay = module(FOLDER / "replay.py")
    _, original = load(FOLDER)
    replay.replay(tmp_path)
    assert {p.name: p.read_bytes() for p in tmp_path.iterdir()} == {
        name: text.encode() for name, text in original.items()
    }
    with pytest.raises(ValueError, match="empty"):
        replay.restore(tmp_path)


def test_independent_gate_rejects_perturbed_energy(tmp_path: Path) -> None:
    module(FOLDER / "replay.py").restore(tmp_path)
    path = tmp_path / "aug-cc-pvqz-mp2--1.json"
    receipt = json.loads(path.read_text())
    component = next(iter(receipt["components"].values()))
    component["energy"] += 0.001
    path.write_text(json.dumps(receipt))
    with pytest.raises(AssertionError):
        module(tmp_path / "summarize.py").summarize("aug-cc-pvqz-mp2--1")


@pytest.mark.parametrize("name", ["receipts.json.xz", "fp64.xz"])
def test_corrupted_stream_is_rejected(tmp_path: Path, name: str) -> None:
    for source in ("receipts.json.xz", "fp64.xz", "sha256.txt"):
        data = (FOLDER / source).read_bytes()
        (tmp_path / source).write_bytes(data + b"corrupt" if source == name else data)
    with pytest.raises(ValueError, match=name):
        load(tmp_path)
