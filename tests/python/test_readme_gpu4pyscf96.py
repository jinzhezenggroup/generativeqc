"""A reference timing supplement cannot replace unrelated or native evidence."""

from __future__ import annotations

import copy
import json
import os
import signal
import subprocess
from pathlib import Path
from types import ModuleType
from typing import IO, Any

import pytest

from tools.render_readme_gpu4pyscf96 import PHASE_COUNTS, overlay_reference


@pytest.fixture
def observations() -> tuple[dict, dict]:
    """Separate native resource hints from the shared scientific protocol."""
    protocol = {"atoms": 96, "grid": [48, 16, 32], "basis": "retained-basis"}
    point = {
        "atoms": 96,
        "protocol": {**protocol, "density_fitting_memory_budget_bytes": 12 << 30},
        "engines": {
            "native": {"status": "completed_unchecked", "records": ["historical"]},
            "reference": {"status": "not_run"},
        },
        "independent_references": ["historical native gate owner"],
    }
    summary = {
        "density_fitting": True,
        "methods": {
            "wb97m-v": [{"atoms": 48, "historical": True}, point],
            "pbe": [{"historical": True}],
        },
    }
    update = {
        "atoms": 96,
        "protocol": {**protocol, "density_fitting_memory_budget_bytes": 0},
        "engines": {
            "reference": {
                "status": "measured",
                "outcome": {"exit_code": 0, "time_limit_seconds": None},
                "records": [
                    {"phase": phase, "complete_seconds": repeat + 1.0}
                    for phase, count in PHASE_COUNTS.items()
                    for repeat in range(count)
                ],
            }
        },
    }
    return summary, update


def test_reference_overlay_preserves_native_and_other_populations(
    observations: tuple[dict, dict],
) -> None:
    """Updating the reference does not reinterpret a historically unchecked run."""
    summary, update = observations
    original = copy.deepcopy(summary)
    result = overlay_reference(summary, update)
    assert summary == original
    assert result["methods"]["pbe"] == original["methods"]["pbe"]
    assert result["methods"]["wb97m-v"][0] == original["methods"]["wb97m-v"][0]
    point = result["methods"]["wb97m-v"][1]
    assert point["engines"]["reference"] == update["engines"]["reference"]
    point["engines"]["reference"] = original["methods"]["wb97m-v"][1]["engines"][
        "reference"
    ]
    assert point == original["methods"]["wb97m-v"][1]


@pytest.mark.parametrize(
    "failure", ["protocol", "case", "status", "phase", "exit", "timeout"]
)
def test_reference_overlay_rejects_incomplete_or_different_runs(
    observations: tuple[dict, dict], failure: str
) -> None:
    """Partial timing populations and different inputs cannot fill the missing point."""
    summary, update = observations
    reference = update["engines"]["reference"]
    if failure == "protocol":
        update["protocol"]["grid"] = [24, 8, 16]
    elif failure == "case":
        update["atoms"] = 48
    elif failure == "status":
        reference["status"] = "failed"
    elif failure == "phase":
        reference["records"].pop()
    elif failure == "exit":
        reference["outcome"]["exit_code"] = 1
    else:
        reference["outcome"]["time_limit_seconds"] = 1800
    with pytest.raises(ValueError):
        overlay_reference(summary, update)


Campaign = tuple[ModuleType, Path, Path, dict[str, dict]]


@pytest.fixture
def campaign(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, observations: tuple[dict, dict]
) -> Campaign:
    """Exercise retention without GPU execution or historical source blobs."""
    from tools import render_readme_gpu4pyscf96 as renderer

    root = tmp_path / "campaign"
    root.mkdir()
    (root / "inputs").mkdir()
    (root / "inputs/def2-svp.json").write_text("basis")
    (root / "reproduce.sh").write_text("launcher")
    (root / "receipts").mkdir()
    publication = tmp_path / "publication"
    publication.mkdir()
    summary, update = observations
    updates = {}
    for mode, label in (("none", "direct"), ("cuda", "df")):
        directory = root / "measurements" / mode / "wb97m-v/96"
        directory.mkdir(parents=True)
        (directory / "reference.json").write_text(f"raw-{mode}")
        (root / "receipts" / f"{mode}.job.txt").write_text(f"receipt-{mode}")
        (root / f"{label}.log").write_text(f"log-{mode}")
        updates[mode] = copy.deepcopy(update)
        updates[mode]["engines"]["reference"]["scheduler"] = {"SLURM_JOB_ID": mode}
    monkeypatch.setattr(renderer, "PUBLICATION", publication)
    monkeypatch.setattr(
        renderer,
        "historical_summaries",
        lambda: dict.fromkeys(updates, summary),
        raising=False,
    )
    monkeypatch.setattr(
        renderer,
        "collect_point",
        lambda directory, size: updates[directory.parent.name],
    )
    return renderer, root, publication, updates


@pytest.mark.parametrize(
    "failure", ["second_run", "missing_receipt", "protocol", "repeat"]
)
def test_retention_validates_every_input_before_writing(
    campaign: Campaign, failure: str
) -> None:
    renderer, root, publication, updates = campaign
    sentinel = publication / "direct-reference.json.xz"
    sentinel.write_bytes(b"unique historical evidence")
    if failure == "second_run":
        updates["cuda"]["engines"]["reference"]["status"] = "failed"
    elif failure == "missing_receipt":
        (root / "receipts/cuda.job.txt").unlink()
    elif failure == "protocol":
        updates["cuda"]["protocol"]["grid"] = [1, 2, 3]
    else:
        updates["cuda"]["engines"]["reference"]["records"].pop()
    with pytest.raises((ValueError, FileNotFoundError)):
        renderer.retain_observations(root)
    assert sentinel.read_bytes() == b"unique historical evidence"
    assert list(publication.iterdir()) == [sentinel]


@pytest.mark.parametrize(
    "name", ["direct-reference.json.xz", "df-reference.json.xz", "manifest.json"]
)
@pytest.mark.parametrize("kind", ["regular", "dangling", "device", "fifo"])
def test_retention_never_replaces_existing_evidence(
    campaign: Campaign, name: str, kind: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    renderer, root, publication, _ = campaign
    target = publication / name
    if kind in ("dangling", "device"):
        target.symlink_to(
            publication.parent / "missing-target"
            if kind == "dangling"
            else Path("/dev/null")
        )
    elif kind == "fifo":
        os.mkfifo(target)
        original = Path.open

        def no_fifo_open(path: Path, *args: Any, **kwargs: Any) -> IO[Any]:
            assert path != target, "refuse a FIFO before attempting blocking I/O"
            return original(path, *args, **kwargs)

        monkeypatch.setattr(Path, "open", no_fifo_open)
    else:
        target.write_bytes(b"unique evidence")
    with pytest.raises(ValueError):
        renderer.retain_observations(root)
    assert list(publication.iterdir()) == [target]
    if kind in ("dangling", "device"):
        assert target.is_symlink()
    elif kind == "fifo":
        assert target.is_fifo()
    else:
        assert target.read_bytes() == b"unique evidence"


def test_retention_identical_replay_is_idempotent(campaign: Campaign) -> None:
    renderer, root, publication, _ = campaign
    renderer.retain_observations(root)
    before = {
        p.name: (p.read_bytes(), p.stat().st_mtime_ns) for p in publication.iterdir()
    }
    renderer.retain_observations(root)
    assert {
        p.name: (p.read_bytes(), p.stat().st_mtime_ns) for p in publication.iterdir()
    } == before


PUBLICATION = (
    Path(__file__).resolve().parents[2]
    / "benchmarks/results/readme-gpu4pyscf96-20261018"
)
SHELL_STUBS = r"""
scontrol() { printf 'synthetic allocation\n'; }
nvidia-smi() { printf 'synthetic GPU\n'; }
python() {
    printf 'called\n' >> "$README_GPU4PYSCF96_ROOT/calls"
    while [[ "$1" != --output ]]; do shift; done
    printf '{"synthetic":true}\n' > "$2"
    printf 'synthetic benchmark log\n'
    return "${FAKE_EXIT:-0}"
}
export -f date scontrol nvidia-smi python
exec bash "$1" none
"""


def _launcher(root: Path, exit_code: int = 0) -> subprocess.Popen[str]:
    env = dict(
        os.environ,
        SLURM_JOB_ID="123",
        CUDA_VISIBLE_DEVICES="0",
        README_GPU4PYSCF96_ROOT=str(root),
        FAKE_EXIT=str(exit_code),
    )
    return subprocess.Popen(
        ["bash", "-c", SHELL_STUBS, "test", str(PUBLICATION / "reproduce.sh")],
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        start_new_session=True,
    )


def _finish(process: subprocess.Popen[str]) -> None:
    """Do not leave a blocked synthetic subprocess behind on a failing control."""
    try:
        process.communicate(timeout=10)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        process.communicate(timeout=5)
        pytest.fail("synthetic launcher blocked instead of refusing the output")


def _campaign_root(tmp_path: Path) -> Path:
    root = tmp_path / "campaign"
    (root / "source").mkdir(parents=True)
    return root


@pytest.mark.parametrize(
    "entry",
    ["empty", "reference.json", "reference.log", "reference.outcome", "symlink"],
)
def test_launcher_refuses_existing_population_even_without_journal(
    tmp_path: Path, entry: str
) -> None:
    root = _campaign_root(tmp_path)
    directory = root / "measurements/none/wb97m-v/96"
    directory.parent.mkdir(parents=True)
    if entry == "symlink":
        directory.symlink_to(root / "missing")
    else:
        directory.mkdir()
        if entry != "empty":
            (directory / entry).write_text("unique prior evidence")
    process = _launcher(root)
    _finish(process)
    assert process.returncode != 0
    assert not (root / "calls").exists()
    if entry == "symlink":
        assert directory.is_symlink() and not (root / "missing").exists()
    elif entry != "empty":
        assert (directory / entry).read_text() == "unique prior evidence"


@pytest.mark.parametrize("link", ["regular", "dangling", "device", "fifo"])
def test_launcher_preserves_existing_scheduler_receipt(
    tmp_path: Path, link: str
) -> None:
    root = _campaign_root(tmp_path)
    (root / "receipts").mkdir()
    receipt = root / "receipts/123.job.txt"
    if link in ("dangling", "device"):
        receipt.symlink_to(
            root / "missing" if link == "dangling" else Path("/dev/null")
        )
    elif link == "fifo":
        os.mkfifo(receipt)
    else:
        receipt.write_text("unique prior allocation")
    process = _launcher(root)
    _finish(process)
    assert process.returncode != 0
    assert not (root / "calls").exists()
    if link in ("dangling", "device"):
        assert receipt.is_symlink()
    elif link == "fifo":
        assert receipt.is_fifo()
    else:
        assert receipt.read_text() == "unique prior allocation"
    assert not (root / "missing").exists()


@pytest.mark.parametrize("exit_code", [0, 42])
def test_launcher_records_exact_benchmark_exit(tmp_path: Path, exit_code: int) -> None:
    root = _campaign_root(tmp_path)
    process = _launcher(root, exit_code)
    _finish(process)
    assert process.returncode == exit_code
    directory = root / "measurements/none/wb97m-v/96"
    assert json.loads((directory / "reference.outcome").read_text()) == {
        "exit_code": exit_code,
        "time_limit_seconds": None,
    }
    assert (directory / "reference.log").read_text() == "synthetic benchmark log\n"
    assert (directory / "reference.json").is_file()


def test_concurrent_launchers_have_one_population_owner(tmp_path: Path) -> None:
    root = _campaign_root(tmp_path)
    processes = [_launcher(root), _launcher(root)]
    for process in processes:
        _finish(process)
    assert sorted(process.returncode for process in processes) == [0, 1]
    assert (root / "calls").read_text().splitlines() == ["called"]


@pytest.mark.parametrize(
    "fault",
    [
        "none",
        "direct-fails",
        "df-fails",
        "log",
        "launcher",
        "log-device",
        "launcher-device",
    ],
)
def test_documented_launch_preserves_logs_and_reports_both_jobs(
    tmp_path: Path, fault: str
) -> None:
    root = _campaign_root(tmp_path)
    stub = tmp_path / "srun"
    stub.write_text(
        '#!/bin/bash\nmode="${!#}"\nprintf "%s\\n" "$mode" >> "$README_GPU4PYSCF96_ROOT/jobs"\n[[ "$mode" != "${FAIL_MODE:-}" ]]\n'
    )
    stub.chmod(0o755)
    sentinel = root / ("reproduce.sh" if fault.startswith("launcher") else "direct.log")
    if fault.endswith("-device"):
        sentinel.symlink_to("/dev/null")
    elif fault in ("log", "launcher"):
        sentinel.write_text("unique prior evidence")
    readme = (PUBLICATION / "README.md").read_text()
    snippet = readme.split("```bash\n")[1].split("```")[0]
    snippet = snippet.replace("/path/to/campaign", str(root))
    result = subprocess.run(
        ["bash", "-c", snippet],
        cwd=PUBLICATION.parents[2],
        env={
            **os.environ,
            "PATH": f"{tmp_path}:{os.environ['PATH']}",
            "FAIL_MODE": {"direct-fails": "none", "df-fails": "cuda"}.get(fault, ""),
        },
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
    )
    assert (result.returncode == 0) == (fault == "none"), result.stderr
    if fault.endswith("-device"):
        assert sentinel.is_symlink()
    elif fault in ("log", "launcher"):
        assert sentinel.read_text() == "unique prior evidence"
    if fault in ("none", "direct-fails", "df-fails"):
        assert sorted((root / "jobs").read_text().splitlines()) == ["cuda", "none"]


def test_retention_rejects_concurrent_different_file(
    campaign: Campaign, monkeypatch: pytest.MonkeyPatch
) -> None:
    renderer, root, publication, _ = campaign
    target = publication / "direct-reference.json.xz"
    original = Path.open

    def race(path: Path, mode: str = "r", *args: Any, **kwargs: Any) -> IO[Any]:
        if path == target and mode == "xb":
            with original(path, "wb") as stream:
                stream.write(b"concurrent different evidence")
        return original(path, mode, *args, **kwargs)

    monkeypatch.setattr(Path, "open", race)
    with pytest.raises(ValueError):
        renderer.retain_observations(root)
    assert target.read_bytes() == b"concurrent different evidence"
    assert not (publication / "manifest.json").exists()
