"""Offline integrity gates for retained scientific density-workload arrays."""

import io
import json
import zipfile
from hashlib import sha256
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from tools.density_workload_matrix import load_workloads

DIRECTORY = (
    Path(__file__).resolve().parents[2] / "benchmarks/results/density-candidates/inputs"
)


def _archive(members: list[tuple[str, bytes]]) -> bytes:
    output = io.BytesIO()
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_LZMA) as archive:
        for name, data in members:
            archive.writestr(name, data)
    return output.getvalue()


@pytest.fixture
def small_workload(
    tmp_path: Path,
) -> tuple[Path, dict[str, Any], list[tuple[str, bytes]]]:
    """Reuse a small independent fixture without invoking its producer."""
    name = "water_svp"
    manifest = json.loads((DIRECTORY / "manifest.json").read_text())
    entry = next(row for row in manifest if row["name"] == name)
    (tmp_path / f"{name}.json").write_bytes((DIRECTORY / f"{name}.json").read_bytes())
    with zipfile.ZipFile(DIRECTORY / f"{name}.npz") as archive:
        members = [(member, archive.read(member)) for member in archive.namelist()]
    packed = _archive(members)
    entry["archive_sha256"] = sha256(packed).hexdigest()
    (tmp_path / "manifest.json").write_text(json.dumps([entry]))
    (tmp_path / f"{name}.npz").write_bytes(packed)
    return tmp_path, entry, members


def test_retained_workloads_keep_scientific_identities_without_extraction(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def no_extract(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("reference loading must not extract archive members")

    monkeypatch.setattr(zipfile.ZipFile, "extract", no_extract)
    monkeypatch.setattr(zipfile.ZipFile, "extractall", no_extract)
    entries = json.loads((DIRECTORY / "manifest.json").read_text())
    loaded = list(load_workloads(DIRECTORY))
    assert [name for name, *_ in loaded] == [entry["name"] for entry in entries]
    for entry, (name, meta, arrays, _) in zip(entries, loaded, strict=True):
        assert meta == json.loads((DIRECTORY / f"{name}.json").read_text())
        assert meta["identity"] == entry["identity"]
        if "array_files" not in entry:
            with zipfile.ZipFile(DIRECTORY / f"{name}.npz") as archive:
                names = archive.namelist()
                assert len(names) == len(set(names))
                assert set(names) == {key + ".npy" for key in arrays}
                assert archive.testzip() is None


@pytest.mark.parametrize("change", ["corrupt", "extra", "duplicate"])
def test_archive_hash_rejects_corruption_and_inventory_tampering(
    small_workload: Any, change: str
) -> None:
    directory, entry, members = small_workload
    path = directory / f"{entry['name']}.npz"
    if change == "corrupt":
        data = bytearray(path.read_bytes())
        data[len(data) // 2] ^= 1
        path.write_bytes(data)
    elif change == "extra":
        path.write_bytes(_archive([*members, ("extra.npy", members[0][1])]))
    else:
        with pytest.warns(UserWarning, match="Duplicate name"):
            path.write_bytes(_archive([*members, members[0]]))
    # Storage authentication precedes decoding. Even duplicate identical arrays
    # are a different archive and cannot reuse the approved storage identity.
    with pytest.raises(ValueError, match="archive identity mismatch"):
        list(load_workloads(directory))


@pytest.mark.parametrize("change", ["extra", "missing", "dtype", "shape", "value"])
def test_rebound_storage_still_requires_original_numeric_contract(
    small_workload: Any, change: str
) -> None:
    directory, entry, members = small_workload
    if change == "extra":
        members.append(("extra.npy", members[0][1]))
    elif change == "missing":
        members.pop()
    else:
        index = next(i for i, (name, _) in enumerate(members) if name == "points.npy")
        name, raw = members[index]
        array = np.load(io.BytesIO(raw), allow_pickle=False)
        if change == "dtype":
            array = array.astype(np.float32)
        elif change == "shape":
            array = array.reshape(-1)
        else:
            array = array.copy()
            array.flat[0] += 1
        output = io.BytesIO()
        np.save(output, array, allow_pickle=False)
        members[index] = name, output.getvalue()
    packed = _archive(members)
    (directory / f"{entry['name']}.npz").write_bytes(packed)
    entry["archive_sha256"] = sha256(packed).hexdigest()
    (directory / "manifest.json").write_text(json.dumps([entry]))
    error = (
        "block mismatch" if change in {"extra", "missing"} else "numeric hash mismatch"
    )
    with pytest.raises(ValueError, match=error):
        list(load_workloads(directory))
