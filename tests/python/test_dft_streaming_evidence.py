"""Offline scientific and corruption gates for native DFT default evidence."""

from __future__ import annotations

import hashlib
import json
import runpy
import shutil
from pathlib import Path

import pytest

from benchmarks._exact_fp64_evidence import load
from benchmarks.dft_streaming_qualification import qualify_native

ROOT = Path(__file__).resolve().parents[2]
FAMILY = ROOT / "benchmarks/results/dft-shared-rys-streaming-20261108"
MASTER = ROOT / "benchmarks/results/dft-streaming-master-20261109"


def test_native_default_qualification_is_scoped_and_preserves_raw_k_oracles() -> None:
    replay = runpy.run_path(str(FAMILY / "replay.py"))["replay"]
    result = replay()
    assert result["default_promotion_qualified"]
    assert not result["all_readme_leadership_established"]
    assert result["raw_k_matrix_predicates"] == {"a100": 144, "5090": 144}


@pytest.mark.parametrize("stream", ("receipts.json.xz", "fp64.xz"))
@pytest.mark.parametrize("family", (FAMILY, MASTER), ids=("historical", "master"))
def test_evidence_rejects_changed_compressed_stream(
    tmp_path: Path, stream: str, family: Path
) -> None:
    for name in ("receipts.json.xz", "fp64.xz", "sha256.txt"):
        shutil.copyfile(family / name, tmp_path / name)
    path = tmp_path / stream
    payload = path.read_bytes()
    path.write_bytes(bytes((payload[0] ^ 1,)) + payload[1:])
    with pytest.raises(ValueError):
        load(tmp_path)


def test_changed_initial_guess_work_does_not_qualify(tmp_path: Path) -> None:
    _, files = load(FAMILY)
    for name, text in files.items():
        if name.startswith("native-a100/"):
            (tmp_path / name.removeprefix("native-a100/")).write_text(text)
    path = tmp_path / "pbe0-2-candidate.json"
    record = json.loads(path.read_text())
    record["samples"][0]["initial_guess"]["preliminary_iterations"] += 1
    path.write_text(json.dumps(record))
    cohort = json.loads((FAMILY / "cohort.json").read_text())
    with pytest.raises(ValueError, match="initial-guess"):
        qualify_native(tmp_path, "a100", images=cohort["native_images"])


def test_storage_migration_authenticates_every_original_seed_receipt_byte() -> None:
    bundle, files = load(ROOT / "benchmarks/results/cuda-ks-seed-products-20261101")
    original = files["original-receipts.json"]
    assert (
        hashlib.sha256(original.encode()).hexdigest()
        == bundle["experiment"]["original_raw_sha256"]
    )
    assert len(json.loads(original)["records"]) == 139


def test_master_qualification_keeps_reference_gaps_and_df_scope_visible() -> None:
    result = runpy.run_path(str(MASTER / "replay.py"))["replay"]()
    assert result["default_promotion_qualified"]
    assert not result["all_readme_leadership_established"]
    assert result["historical_raw_k_matrix_predicates"] == {"a100": 144, "5090": 144}
    reference = result["known_remaining_gaps"]["b3lyp_direct96_reference_seconds"]
    native = result["cards"]["a100"]["methods"]["b3lyp"]["cold_means"]["candidate"]
    assert native > reference
    for diagnostic in result["cards"]["a100"]["df_single_pair_diagnostics"].values():
        assert diagnostic["single_pair_not_abba"]


@pytest.mark.parametrize(
    ("change", "message"),
    (
        ("missing-df", "Incomplete DF diagnostics"),
        ("df-image", "Wrong normal DF ELF"),
        ("df-forces", "DF paired E/F failed"),
        ("cold-census", "B3LYP automatic dispatch was not proved"),
    ),
)
def test_master_diagnostics_reject_incomplete_or_changed_evidence(
    tmp_path: Path, change: str, message: str
) -> None:
    _, files = load(MASTER)
    for name, text in files.items():
        if name.startswith("native-a100/"):
            path = tmp_path / name.removeprefix("native-a100/")
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(text)
    if change == "missing-df":
        (tmp_path / "pbe0-df-reference.json").unlink()
    else:
        name = {
            "df-image": "identity-pbe0-df-candidate.json",
            "df-forces": "pbe0-df-candidate.json",
            "cold-census": "b3lyp-default-execution-proof.json",
        }[change]
        path = tmp_path / name
        record = json.loads(path.read_text())
        if change == "df-image":
            record["library_sha256"] = "0" * 64
        elif change == "df-forces":
            record["samples"][0]["forces"][0][0] += 0.1
        else:
            record["samples"][0]["fock_builds"] += 1
        path.write_text(json.dumps(record))
    cohort = json.loads((MASTER / "cohort.json").read_text())
    with pytest.raises(ValueError, match=message):
        qualify_native(tmp_path, "a100", images=cohort["native_images"])


@pytest.mark.parametrize("family", (FAMILY, MASTER), ids=("historical", "master"))
@pytest.mark.parametrize("arm", ("control", "candidate", "both"))
@pytest.mark.parametrize("corruption", ("empty", "partial", "failed"))
def test_small_population_must_be_complete(
    tmp_path: Path, arm: str, corruption: str, family: Path
) -> None:
    _, files = load(family)
    for name, text in files.items():
        if name.startswith("native-a100/"):
            path = tmp_path / name.removeprefix("native-a100/")
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(text)
    arms = ("control", "candidate") if arm == "both" else (arm,)
    for selected in arms:
        path = tmp_path / f"pbe0-48-{selected}.json"
        record = json.loads(path.read_text())
        if corruption == "failed":
            record["status"] = "failed"
        else:
            record["samples"] = [] if corruption == "empty" else record["samples"][:1]
        path.write_text(json.dumps(record))
    with pytest.raises(ValueError, match="Incomplete 48-atom population"):
        qualify_native(
            tmp_path,
            "a100",
            images=json.loads((family / "cohort.json").read_text())["native_images"],
        )
