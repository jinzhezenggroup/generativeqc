"""Exercise supplement ownership and archived-input integrity without a GPU."""

from __future__ import annotations

import hashlib
import json
import lzma
import os
import runpy
import subprocess
from pathlib import Path

import pytest

PUBLICATION = (
    Path(__file__).resolve().parents[2] / "benchmarks/results/readme-refresh-20261015"
)
RUNNER = PUBLICATION / "reproduce-df96.sh"
REDUCER = PUBLICATION / "reduce.py"


@pytest.fixture
def replay(tmp_path: Path) -> dict:
    """Run the real shell workflow with bounded fake scheduler/GPU endpoints."""
    publication = tmp_path / "publication"
    publication.mkdir()
    runner = publication / "reproduce-df96.sh"
    runner.write_bytes(RUNNER.read_bytes())
    consumer = b"# frozen executed benchmark consumer\n"
    archive = publication / "df96-benchmark-consumer.py.xz"
    payload = lzma.compress(consumer)
    archive.write_bytes(payload)
    manifest = {
        "files": {
            archive.name: {
                "bytes": len(payload),
                "sha256": hashlib.sha256(payload).hexdigest(),
                "uncompressed": {
                    "bytes": len(consumer),
                    "sha256": hashlib.sha256(consumer).hexdigest(),
                },
            }
        }
    }
    (publication / "manifest.json").write_text(json.dumps(manifest))
    root = tmp_path / "frozen"
    (root / "source/benchmarks").mkdir(parents=True)
    (root / "source/benchmarks/readme_omol25.py").write_text("# original consumer\n")
    (root / "runtime").mkdir()
    (root / "runtime/libgenerativeqc.so").write_bytes(b"test library")
    fake_tools = tmp_path / "fake-tools.sh"
    fake_tools.write_text(
        """scontrol() { printf 'fake scheduler receipt\\n'; }
nvidia-smi() { printf 'fake assigned GPU receipt\\n'; }
sccache() { printf 'sccache test\\n'; }
ccache() { printf 'ccache test\\n'; }
timeout() {
    printf '%s\\n' "$PWD" > "$README_TEST_CALL"
    printf '%s\\n' "$@" >> "$README_TEST_CALL"
    return 124
}
"""
    )
    env = os.environ | {
        "BASH_ENV": str(fake_tools),
        "SLURM_JOB_ID": "123",
        "CUDA_VISIBLE_DEVICES": "0",
        "README_REFRESH_ROOT": str(root),
        "README_DF96_OUTPUT": str(tmp_path / "output"),
        "README_TEST_CALL": str(tmp_path / "measurement-call.txt"),
    }
    return {
        "tmp": tmp_path,
        "runner": runner,
        "root": root,
        "env": env,
        "consumer": consumer,
    }


def invoke(replay: dict) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["bash", str(replay["runner"])],
        cwd=replay["tmp"],
        env=replay["env"],
        capture_output=True,
        text=True,
        timeout=15,
        check=False,
    )


@pytest.mark.parametrize("kind", ["directory", "symlink", "dangling_symlink"])
def test_replay_refuses_existing_output_without_touching_evidence(
    replay: dict, kind: str
) -> None:
    output = Path(replay["env"]["README_DF96_OUTPUT"])
    target = replay["tmp"] / "previous"
    if kind == "directory":
        target = output
    if kind != "dangling_symlink":
        (target / "receipts").mkdir(parents=True)
        (target / "receipts/started.txt").write_bytes(b"prior receipt\n")
        directory = target / "measurements/cuda/wb97m-v/96"
        directory.mkdir(parents=True)
        (directory / "native.log").write_bytes(b"prior log\n")
        (directory / "native.outcome").write_bytes(b"prior outcome\n")
    if kind != "directory":
        output.symlink_to(target, target_is_directory=True)
    before = {
        p.relative_to(target): p.read_bytes() for p in target.rglob("*") if p.is_file()
    }
    result = invoke(replay)
    assert result.returncode != 0
    assert not Path(replay["env"]["README_TEST_CALL"]).exists()
    assert {
        p.relative_to(target): p.read_bytes() for p in target.rglob("*") if p.is_file()
    } == before
    assert not (target / "source").exists()


def test_relative_paths_keep_all_receipts_in_the_reserved_output(replay: dict) -> None:
    replay["env"].update(
        README_REFRESH_ROOT="frozen", README_DF96_OUTPUT="nested/output"
    )
    result = invoke(replay)
    assert result.returncode == 124, result.stderr
    output = replay["tmp"] / "nested/output"
    assert (output / "source/benchmarks/readme_omol25.py").read_bytes() == replay[
        "consumer"
    ]
    assert not (output / "source/nested").exists()
    directory = output / "measurements/cuda/wb97m-v/96"
    assert json.loads((directory / "native.outcome").read_bytes()) == {
        "exit_code": 124,
        "time_limit_seconds": 14400,
    }
    assert (output / "receipts/exit-code.txt").read_text() == "124\n"
    call = Path(replay["env"]["README_TEST_CALL"]).read_text().splitlines()
    assert call[0] == str(output / "source")
    assert str(replay["root"] / "measurements/cuda/inputs/def2-svp.json") in call
    assert str(directory / "native.json") in call
    before = {
        p.relative_to(output): p.read_bytes() for p in output.rglob("*") if p.is_file()
    }
    assert invoke(replay).returncode != 124
    assert {
        p.relative_to(output): p.read_bytes() for p in output.rglob("*") if p.is_file()
    } == before


@pytest.mark.parametrize("identity", ["archive", "expanded"])
def test_replay_rejects_corrupt_consumer_before_measurement(
    replay: dict, identity: str
) -> None:
    manifest_path = replay["runner"].parent / "manifest.json"
    manifest = json.loads(manifest_path.read_bytes())
    receipt = manifest["files"]["df96-benchmark-consumer.py.xz"]
    if identity == "expanded":
        receipt = receipt["uncompressed"]
    receipt["sha256"] = "0" * 64
    manifest_path.write_text(json.dumps(manifest))
    result = invoke(replay)
    assert result.returncode != 0
    assert "consumer checksum mismatch" in result.stderr
    assert not Path(replay["env"]["README_TEST_CALL"]).exists()


def test_reducer_retains_expanded_archive_identity(tmp_path: Path) -> None:
    receipt = runpy.run_path(str(REDUCER))["receipt"]
    payload = b"preserve exact historical consumer and evidence bytes\n"
    archive = tmp_path / "consumer.py.xz"
    archive.write_bytes(lzma.compress(payload))
    identity = receipt(archive)
    assert identity["uncompressed"] == {
        "bytes": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
    }
