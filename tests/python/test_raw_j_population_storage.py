"""Behavioral integrity controls for lossless raw-J evidence storage."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import lzma
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

from benchmarks._exact_fp64_evidence import load

if TYPE_CHECKING:
    from types import ModuleType

REPO = Path(__file__).resolve().parents[2]
SCRIPT = REPO / "benchmarks/results/df-raw-coulomb-20261023/reproduce.py"


@pytest.fixture
def replay() -> ModuleType:
    spec = importlib.util.spec_from_file_location("raw_j_replay", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize(
    "args,optimize", [(["-O"], ""), (["-OO"], ""), ([], "1"), ([], "2")]
)
def test_replay_rejects_disabled_acceptance_assertions(
    args: list[str], optimize: str
) -> None:
    env = dict(
        os.environ,
        PYTHONOPTIMIZE=optimize,
        PYTHONPATH=os.pathsep.join((str(REPO), str(REPO / "python"))),
    )
    result = subprocess.run(
        [sys.executable, *args, str(SCRIPT)],
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode != 0
    assert "Evidence validation requires assertions" in result.stderr
    assert not result.stdout


def test_population_bytes_match_recorded_integrity(replay: ModuleType) -> None:
    manifest = json.loads((replay.ROOT / "receipt-populations.json").read_text())
    for name, expected in manifest["members"].items():
        raw = replay.restored_population(name.removesuffix(".json"))
        assert len(raw) == expected["bytes"]
        assert hashlib.sha256(raw).hexdigest() == expected["sha256"]
        assert replay.read(name.removesuffix(".json")) == json.loads(raw)


@pytest.mark.parametrize("corruption", ["bundle", "missing", "extra", "text", "size"])
def test_corrupt_population_is_rejected(
    replay: ModuleType, tmp_path: Path, corruption: str
) -> None:
    manifest = json.loads((replay.ROOT / "receipt-populations.json").read_text())
    folder = tmp_path / manifest["storage_directory"]
    shutil.copytree(replay.ROOT / manifest["storage_directory"], folder)
    _, files = load(folder)
    members = json.loads(files["original-container.json"])
    name = next(iter(members))
    if corruption == "bundle":
        (folder / "receipts.json.xz").write_bytes(b"broken")
    else:
        if corruption == "missing":
            del members[name]
        elif corruption == "extra":
            members["unexpected.json"] = "{}"
        elif corruption == "text":
            members[name] += " "
        else:
            manifest["members"][name]["bytes"] += 1
        container = json.dumps(members)
        manifest["original_decoded_sha256"] = hashlib.sha256(
            container.encode()
        ).hexdigest()
        texts = dict(members, **{"original-container.json": container})
        bundle = {
            "schema": "generativeqc.exact-fp64-evidence.v3",
            "float_count": 0,
            "files": [
                {
                    "path": key,
                    "text": text,
                    "sha256": hashlib.sha256(text.encode()).hexdigest(),
                }
                for key, text in texts.items()
            ],
        }
        streams = {
            "receipts.json.xz": lzma.compress(json.dumps(bundle).encode()),
            "fp64.xz": lzma.compress(b""),
        }
        for key, payload in streams.items():
            (folder / key).write_bytes(payload)
        (folder / "sha256.txt").write_text(
            "".join(
                hashlib.sha256(payload).hexdigest() + "  " + key + "\n"
                for key, payload in streams.items()
            )
        )
    (tmp_path / "receipt-populations.json").write_text(json.dumps(manifest))
    replay.ROOT = tmp_path
    with pytest.raises(ValueError, match="mismatch"):
        replay.restored_population(name.removesuffix(".json"))
