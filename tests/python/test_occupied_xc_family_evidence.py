"""Keep current-source family measurements replayable without a GPU stack."""

from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import lzma
import os
import runpy
import shutil
import subprocess
import sys
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from types import ModuleType

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "benchmarks/results/occupied-xc-families-20261105/replay.py"


@pytest.fixture(scope="module")
def retained() -> tuple[ModuleType, dict]:
    """Load only the standard-library reader and immutable retained capsule."""
    spec = importlib.util.spec_from_file_location("occupied_family_replay", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module, module.load_capsules()["endpoints"]


def test_retained_family_population_and_admission(
    retained: tuple[ModuleType, dict],
) -> None:
    module, capsule = retained
    summary = module.replay(capsule)
    assert len(summary["current"]) == 26
    for name in ("large-m06-2x-rks", "confirmation-large-m06-2x-rks"):
        assert summary["current"][name]["phases"]["cold"]["profitable"]
    assert summary["original"]["cold"]["profitable"]
    assert not summary["original"]["warm"]["regresses"]
    assert summary["current"]["confirmation-small-pbe0-rks"]["phases"]["warm"][
        "regresses"
    ]
    assert set(capsule["diagnostics"]) == {
        f"{method}-{spin}"
        for method in ("lda", "pbe", "pbe0", "b3lyp", "r2scan", "m06-2x", "mn15")
        for spin in ("rks", "uks")
    }
    for population in capsule["populations"].values():
        assert len(population) == 6
    for diagnostic in capsule["diagnostics"].values():
        assert any(diagnostic["frames"][0])


@pytest.mark.parametrize(
    "mutation",
    [
        "image",
        "host",
        "force",
        "inventory",
        "precision",
        "reference",
        "tail",
        "work",
        "original",
        "source_work",
        "jk",
        "history",
    ],
)
def test_retained_family_corruption_is_rejected(
    retained: tuple[ModuleType, dict], mutation: str
) -> None:
    module, original = retained
    capsule = copy.deepcopy(original)
    record = next(iter(capsule["populations"].values()))[-1]
    if mutation == "image":
        record["library_sha256"] = "0" * 64
    elif mutation == "host":
        record["host"] = "another-host"
    elif mutation == "force":
        record["observations"][0]["forces"][0][0] += 1
    elif mutation == "inventory":
        record["observations"][0]["precision"]["operator_inventory_complete"] = False
    elif mutation == "precision":
        record["observations"][0]["precision"]["operators"][0]["compute"] = "fp32"
    elif mutation == "reference":
        capsule["second_references"].clear()
    elif mutation == "tail":
        next(iter(capsule["diagnostics"].values()))["frames"][0][-1] = 1
    elif mutation == "work":
        capsule["source_handoff"][1]["receipt"]["observations"][0]["fock_builds"] += 1
    elif mutation == "original":
        capsule["original_r2scan3c"]["r0-unset"]["source_revision"] = "0" * 40
    elif mutation == "source_work":
        capsule["source_handoff"][1]["receipt"]["observations"][0]["precision"][
            "operators"
        ][0]["count"] += 1
    elif mutation == "jk":
        capsule["incremental_handoff"]["after"]["unset"]["trace_work"][0][
            "structured_basis"
        ] += 1
    else:
        capsule["history_diagnosis"]["records"][0]["source_identity"] = "0" * 64
    with pytest.raises(ValueError):
        module.replay(capsule)


def test_integrated_image_is_qualified_separately(
    retained: tuple[ModuleType, dict],
) -> None:
    module, capsule = retained
    supplement = module.load_capsules()["integration"]
    assert supplement["source_identity"] != capsule["source_identity"]
    summary = module.replay_integration(supplement, capsule)
    assert summary["m06"]["phases"]["cold"]["profitable"]
    assert summary["original"]["cold"]["profitable"]
    for phase in summary["m06"]["phases"].values():
        assert not phase["regresses"]
        assert (
            phase["iterations_fock_builds"]["matrix"]
            == phase["iterations_fock_builds"]["default"]
        )
    for phase in summary["original"].values():
        assert not phase["regresses"]
        assert (
            phase["iterations_fock_builds"]["0"]
            == phase["iterations_fock_builds"]["unset"]
        )


def test_integrated_image_corruption_is_rejected(
    retained: tuple[ModuleType, dict],
) -> None:
    module, capsule = retained
    module.integration_corruption_probes(module.load_capsules()["integration"], capsule)


def test_original_transports_recover_without_repository_history(
    retained: tuple[ModuleType, dict], tmp_path: Path
) -> None:
    module, capsule = retained
    for name in ("fp64.xz", "receipts.json.xz", "sha256.txt", "storage-receipt.json"):
        shutil.copyfile(SCRIPT.parent / name, tmp_path / name)
    originals = module.recover_originals(tmp_path)
    receipt = json.loads((tmp_path / "storage-receipt.json").read_text())
    for row in receipt["originals"]:
        packed = originals[row["path"]]
        assert len(packed) == row["bytes"]
        assert hashlib.sha256(packed).hexdigest() == row["sha256"]
        raw = lzma.decompress(packed)
        assert len(raw) == row["text_bytes"]
        assert hashlib.sha256(raw).hexdigest() == row["text_sha256"]
    assert json.loads(lzma.decompress(originals["endpoints.json.xz"])) == capsule


def test_original_export_refuses_an_existing_destination(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    existing = tmp_path / "endpoints.json.xz"
    existing.write_bytes(b"existing evidence must remain untouched")
    monkeypatch.setattr(
        sys, "argv", [str(SCRIPT), "--restore-originals", str(tmp_path)]
    )
    with pytest.raises(FileExistsError):
        runpy.run_path(str(SCRIPT), run_name="__main__")
    assert existing.read_bytes() == b"existing evidence must remain untouched"
    assert not (tmp_path / "integration.json.xz").exists()


@pytest.mark.parametrize("field", ["bytes", "sha256", "git_blob_sha1", "text_sha256"])
def test_original_transport_identity_corruption_is_rejected(
    retained: tuple[ModuleType, dict], tmp_path: Path, field: str
) -> None:
    module, _ = retained
    for name in ("fp64.xz", "receipts.json.xz", "sha256.txt", "storage-receipt.json"):
        shutil.copyfile(SCRIPT.parent / name, tmp_path / name)
    path = tmp_path / "storage-receipt.json"
    receipt = json.loads(path.read_text())
    receipt["originals"][0][field] = 0 if field == "bytes" else "0" * 64
    path.write_text(json.dumps(receipt))
    with pytest.raises(ValueError, match="original (compressed transport|JSON bytes)"):
        module.recover_originals(tmp_path)


def test_integration_binds_the_complete_reconstructed_baseline(
    retained: tuple[ModuleType, dict],
) -> None:
    module, capsule = retained
    supplement = module.load_capsules()["integration"]
    altered = dict(capsule, unexpected_field="must not escape the transport binding")
    with pytest.raises(ValueError, match="integration reference provenance"):
        module.replay_integration(supplement, altered)


@pytest.mark.parametrize("optimize", [False, True])
def test_rehashed_fp64_corruption_cannot_change_original_text(
    tmp_path: Path, optimize: bool
) -> None:
    for name in ("fp64.xz", "receipts.json.xz", "sha256.txt", "storage-receipt.json"):
        shutil.copyfile(SCRIPT.parent / name, tmp_path / name)
    path = tmp_path / "fp64.xz"
    planes = bytearray(lzma.decompress(path.read_bytes()))
    planes[0] ^= 1
    path.write_bytes(lzma.compress(planes))
    (tmp_path / "sha256.txt").write_text(
        "".join(
            hashlib.sha256((tmp_path / name).read_bytes()).hexdigest()
            + "  "
            + name
            + "\n"
            for name in ("fp64.xz", "receipts.json.xz")
        )
    )
    code = (
        "import importlib.util, pathlib; "
        f"spec = importlib.util.spec_from_file_location('family_replay', {str(SCRIPT)!r}); "
        "module = importlib.util.module_from_spec(spec); "
        "spec.loader.exec_module(module); "
        f"module.recover_originals(pathlib.Path({str(tmp_path)!r}))"
    )
    process = subprocess.run(
        [sys.executable, *(["-O"] if optimize else []), "-c", code],
        env=dict(
            os.environ,
            PYTHONPATH=os.pathsep.join((str(ROOT), str(ROOT / "python"))),
        ),
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert process.returncode != 0
    assert "ValueError: endpoints.json" in process.stderr


@pytest.mark.parametrize("optimize", [False, True])
def test_retained_family_gates_survive_optimized_python(optimize: bool) -> None:
    command = [sys.executable, *(["-O"] if optimize else []), str(SCRIPT)]
    process = subprocess.run(
        command,
        env=dict(
            os.environ,
            PYTHONPATH=os.pathsep.join((str(ROOT), str(ROOT / "python"))),
            GENERATIVEQC_LIBRARY=str(SCRIPT.parent / "unavailable-offline-library.so"),
        ),
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert process.returncode == 0, process.stderr
