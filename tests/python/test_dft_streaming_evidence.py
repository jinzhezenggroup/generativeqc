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


@pytest.fixture
def native_a100(tmp_path: Path) -> Path:
    """Materialize authenticated receipts for isolated semantic mutations."""
    _, files = load(FAMILY)
    for name, text in files.items():
        if name.startswith("native-a100/"):
            (tmp_path / name.removeprefix("native-a100/")).write_text(text)
    return tmp_path


def test_native_default_qualification_is_scoped_and_preserves_raw_k_oracles() -> None:
    replay = runpy.run_path(str(FAMILY / "replay.py"))["replay"]
    result = replay()
    assert result["default_promotion_qualified"]
    assert not result["all_readme_leadership_established"]
    assert result["raw_k_matrix_predicates"] == {"a100": 144, "5090": 144}


@pytest.mark.parametrize("stream", ("receipts.json.xz", "fp64.xz"))
def test_evidence_rejects_changed_compressed_stream(
    tmp_path: Path, stream: str
) -> None:
    for name in ("receipts.json.xz", "fp64.xz", "sha256.txt"):
        shutil.copyfile(FAMILY / name, tmp_path / name)
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
    with pytest.raises(ValueError, match="initial-guess"):
        qualify_native(tmp_path, "a100")


def test_storage_migration_authenticates_every_original_seed_receipt_byte() -> None:
    bundle, files = load(ROOT / "benchmarks/results/cuda-ks-seed-products-20261101")
    original = files["original-receipts.json"]
    assert (
        hashlib.sha256(original.encode()).hexdigest()
        == bundle["experiment"]["original_raw_sha256"]
    )
    assert len(json.loads(original)["records"]) == 139


@pytest.mark.parametrize("arm", ("control", "candidate", "both"))
@pytest.mark.parametrize("corruption", ("empty", "partial", "failed"))
def test_small_population_must_be_complete(
    tmp_path: Path, arm: str, corruption: str
) -> None:
    _, files = load(FAMILY)
    for name, text in files.items():
        if name.startswith("native-a100/"):
            (tmp_path / name.removeprefix("native-a100/")).write_text(text)
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
        qualify_native(tmp_path, "a100")


@pytest.mark.parametrize("arm,ordinal", (("control", 1), ("candidate", 2)))
@pytest.mark.parametrize("atoms", (48, 96))
@pytest.mark.parametrize("field", ("source_identity", "library_sha256", "missing"))
def test_endpoint_requires_its_own_native_identity(
    native_a100: Path, arm: str, ordinal: int, atoms: int, field: str
) -> None:
    """A valid sidecar cannot qualify an endpoint from a different image."""
    tag = "48" if atoms == 48 else str(ordinal)
    path = native_a100 / f"pbe0-{tag}-{arm}.json"
    record = json.loads(path.read_text())
    if field == "missing":
        record.pop("native_build")
    else:
        record["native_build"][field] = "0" * 64
    path.write_text(json.dumps(record))
    with pytest.raises(ValueError, match="endpoint native build"):
        qualify_native(native_a100, "a100")


def test_small_workload_cannot_be_replaced_by_passing_96_atoms(
    native_a100: Path,
) -> None:
    """Paired Direct96 receipts must not become an independent Direct48 pass."""
    for arm, ordinal in (("control", 1), ("candidate", 2)):
        (native_a100 / f"pbe0-48-{arm}.json").write_text(
            (native_a100 / f"pbe0-{ordinal}-{arm}.json").read_text()
        )
    with pytest.raises(ValueError, match="48-atom workload"):
        qualify_native(native_a100, "a100")


@pytest.mark.parametrize("field,value", (("method", "b3lyp"), ("mode", "df")))
def test_small_workload_requires_pbe0_direct(
    native_a100: Path, field: str, value: str
) -> None:
    """The small endpoint's method and mode belong to its declared contract."""
    path = native_a100 / "pbe0-48-candidate.json"
    record = json.loads(path.read_text())
    record[field] = value
    path.write_text(json.dumps(record))
    with pytest.raises(ValueError, match="48-atom workload"):
        qualify_native(native_a100, "a100")


@pytest.mark.parametrize("corruption", ("duplicate", "unexpected", "count"))
def test_census_requires_each_expected_class_once(
    native_a100: Path, corruption: str
) -> None:
    """Six rows do not prove six classes when a symbol is duplicated."""
    path = native_a100 / "default-cold-kernels.tsv"
    rows = path.read_text().splitlines()
    indices = [i for i, row in enumerate(rows) if "shared_rys_xy_streaming" in row]
    if corruption == "duplicate":
        rows[indices[1]] = rows[indices[0]]
    elif corruption == "unexpected":
        rows[indices[0]] = rows[indices[0]].replace("_ddpp_", "_ssss_")
    else:
        rows[indices[0]] = "8\t" + rows[indices[0]].split("\t", 1)[1]
    path.write_text("\n".join(rows) + "\n")
    with pytest.raises(ValueError, match="shared-XY execution"):
        qualify_native(native_a100, "a100")


@pytest.mark.parametrize("corruption", ("duplicate", "unexpected", "missing"))
def test_raw_k_requires_each_predicate_once(corruption: str) -> None:
    """Authenticated log bytes still need a complete scientific predicate set."""
    _, files = load(FAMILY)
    cohort = json.loads((FAMILY / "cohort.json").read_text())
    lines = files[cohort["raw_k_logs"]["a100"]].splitlines()
    indices = [i for i, line in enumerate(lines) if line.startswith("PASS ")]
    if corruption == "duplicate":
        lines[indices[1]] = lines[indices[0]]
    elif corruption == "unexpected":
        lines[indices[0]] = lines[indices[0]].replace("dppp_", "ssss_")
    else:
        lines.pop(indices[0])
    reader = runpy.run_path(str(FAMILY / "replay.py"))
    with pytest.raises(ValueError, match="raw-K oracle"):
        reader["raw_k_predicates"]("\n".join(lines))


def test_replay_rejects_duplicate_raw_k_predicate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Replay rejects a malformed original log after the storage layer loads."""
    bundle, files = load(FAMILY)
    cohort = json.loads((FAMILY / "cohort.json").read_text())
    name = cohort["raw_k_logs"]["a100"]
    lines = files[name].splitlines()
    indices = [i for i, line in enumerate(lines) if line.startswith("PASS ")]
    lines[indices[1]] = lines[indices[0]]
    files[name] = "\n".join(lines)
    replay = runpy.run_path(str(FAMILY / "replay.py"))["replay"]
    monkeypatch.setitem(replay.__globals__, "load", lambda _: (bundle, files))
    with pytest.raises(ValueError, match="raw-K oracle"):
        replay()
