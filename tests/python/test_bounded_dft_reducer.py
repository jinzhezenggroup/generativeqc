"""Source admission for bounded DFT publication, independent of numerical gates."""

from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import lzma
import sys
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

from benchmarks.readme_dft import SCHEMA

if TYPE_CHECKING:
    from types import ModuleType

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def reducer() -> ModuleType:
    """Load the campaign's actual standalone reducer without running its CLI."""
    path = ROOT / "benchmarks/results/bounded-dft-schedules-2475-2479/reduce.py"
    spec = importlib.util.spec_from_file_location("bounded_dft_reducer", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def frozen_source(
    reducer: ModuleType, tmp_path_factory: pytest.TempPathFactory
) -> dict:
    """Use the real checked-out source inventory and a hash-bound test image."""
    campaign = tmp_path_factory.mktemp("bounded-dft-source")
    (campaign / "ao-source").symlink_to(ROOT, target_is_directory=True)
    library = campaign / "ao/lib/libgenerativeqc.so"
    library.parent.mkdir(parents=True)
    library.write_bytes(b"source-binding test image, not a scientific measurement")
    return reducer.freeze_source(campaign, "ao")


def receipt(source: dict) -> dict:
    """Return all twelve independently checkable records with complete bindings."""
    rows = []
    for phase, geometry, count in (
        ("cold", 0, 1),
        ("warm", 0, 5),
        ("moved", 1, 1),
        ("moved-warm", 1, 5),
    ):
        rows.extend(
            {
                "phase": phase,
                "geometry": geometry,
                "repeat": repeat,
                "energy": -76.0,
                "forces": [[0.0, 0.0, 0.0]] * 3,
                "converged": True,
                "status": 0,
                "iterations": 1,
                "fock_builds": 1,
                "complete_seconds": 1.0,
                "gate": True,
            }
            for repeat in range(count)
        )
    return {
        "schema": SCHEMA,
        "status": "measured",
        "stage": "complete",
        "protocol": {
            "atoms": 3,
            "repeats": 5,
            "force_return": "host_array_in_timed_endpoint",
        },
        "records": rows,
        "native_build": {
            "probe": {"source_identity": source["native_source_identity"]},
            "library_sha256": source["installed_library_sha256"],
        },
        "source_file_sha256": copy.deepcopy(source["consumer_source_file_sha256"]),
    }


def reduce_receipt(
    reducer: ModuleType,
    source: dict,
    raw: dict,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[dict, str]:
    """Exercise the real collection/acceptance/table path and retain input bytes."""
    directory = tmp_path / "endpoints/ao-host/direct/pbe/3"
    directory.mkdir(parents=True)
    reference = directory / "reference.json"
    reference.write_text(json.dumps(receipt(source)))
    raw["reference_sha256"] = hashlib.sha256(reference.read_bytes()).hexdigest()
    native = directory / "native.json"
    native.write_text(json.dumps(raw))
    before = native.read_bytes()
    for engine in ("reference", "native"):
        (directory / f"{engine}.outcome").write_text('{"exit_code":0}')
    output, table = tmp_path / "result.json.xz", tmp_path / "result.md"
    monkeypatch.setattr(reducer, "freeze_source", lambda *_: copy.deepcopy(source))
    monkeypatch.setattr(
        reducer, "planned_points", lambda: {("ao-host", "direct", "pbe", 3)}
    )
    monkeypatch.setattr(
        sys,
        "argv",
        ["reduce", str(tmp_path), "--output", str(output), "--table", str(table)],
    )
    reducer.main()
    result = next(
        point
        for point in json.loads(lzma.decompress(output.read_bytes()))["points"]
        if point["method"] == "pbe" and point["mode"] == "direct"
    )
    assert native.read_bytes() == before
    assert (
        result["engines"]["native"]["raw_sha256"] == hashlib.sha256(before).hexdigest()
    )
    assert result["engines"]["native"]["records"] == [
        {key: value for key, value in row.items() if key != "forces"}
        for row in raw["records"]
    ]
    return result, table.read_text()


@pytest.mark.parametrize("fock_builds", (None, 1))
def test_source_matched_complete_population_keeps_every_sample(
    fock_builds: int | None,
    reducer: ModuleType,
    frozen_source: dict,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    raw = receipt(frozen_source)
    for row in raw["records"]:
        row["fock_builds"] = fock_builds
    point, table = reduce_receipt(reducer, frozen_source, raw, tmp_path, monkeypatch)
    assert point["source_binding"] == {"status": "verified", "reasons": []}
    assert point["accepted_phase_seconds"] == {
        "cold": [1.0],
        "warm": [1.0] * 5,
        "moved": [1.0],
        "moved-warm": [1.0] * 5,
    }
    count = "—" if fock_builds is None else "1"
    assert f"1.000 [1.000, 1.000] (1/{count})" in table


@pytest.mark.parametrize(
    "defect",
    (
        "missing_build",
        "malformed_build",
        "empty_build",
        "missing_probe",
        "malformed_probe",
        "wrong_source",
        "missing_library",
        "wrong_library",
        "missing_consumers",
        "malformed_consumers",
        "empty_consumers",
        "partial_consumers",
        "wrong_consumer",
        "extra_consumer",
    ),
)
def test_unbound_receipts_keep_science_but_publish_no_accepted_timing(
    defect: str,
    reducer: ModuleType,
    frozen_source: dict,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    raw = receipt(frozen_source)
    build, consumers = raw["native_build"], raw["source_file_sha256"]
    if defect == "missing_build":
        raw.pop("native_build")
    elif defect == "malformed_build":
        raw["native_build"] = [build]
    elif defect == "empty_build":
        raw["native_build"] = {}
    elif defect == "missing_probe":
        build.pop("probe")
    elif defect == "malformed_probe":
        build["probe"] = []
    elif defect == "wrong_source":
        build["probe"]["source_identity"] = "0" * 64
    elif defect == "missing_library":
        build.pop("library_sha256")
    elif defect == "wrong_library":
        build["library_sha256"] = "0" * 64
    elif defect == "missing_consumers":
        raw.pop("source_file_sha256")
    elif defect == "malformed_consumers":
        raw["source_file_sha256"] = list(consumers)
    elif defect == "empty_consumers":
        raw["source_file_sha256"] = {}
    elif defect == "partial_consumers":
        consumers.pop(next(iter(consumers)))
    elif defect == "wrong_consumer":
        consumers[next(iter(consumers))] = "0" * 64
    else:
        consumers["../unrelated.py"] = "0" * 64
    point, table = reduce_receipt(reducer, frozen_source, raw, tmp_path, monkeypatch)
    assert point["source_binding"]["status"] == "unqualified"
    assert point["source_binding"]["reasons"]
    assert set(point["engines"]["native"]["phase_status"].values()) == {"measured"}
    assert all(not values for values in point["accepted_phase_seconds"].values())
    assert table.count("source_unqualified") == 4
    assert "1.000" not in table


def test_partial_population_does_not_fabricate_missing_measurements(
    reducer: ModuleType,
    frozen_source: dict,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    raw = receipt(frozen_source)
    raw.update(status="running", stage="warm/0")
    raw["records"] = raw["records"][:2]
    raw.pop("native_build")
    point, table = reduce_receipt(reducer, frozen_source, raw, tmp_path, monkeypatch)
    assert point["source_binding"]["status"] == "unqualified"
    assert point["engines"]["native"]["phase_status"]["warm"] == "incomplete"
    assert all(not values for values in point["accepted_phase_seconds"].values())
    assert "1.000" not in table
