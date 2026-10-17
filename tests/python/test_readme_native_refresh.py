"""Protect native-only publication overlays and independent acceptance gates."""

from __future__ import annotations

import copy
import hashlib
import json
import lzma
import runpy
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from pathlib import Path


@pytest.fixture
def publication(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict:
    """Build synthetic endpoint evidence without depending on measured numbers."""
    module = runpy.run_path("benchmarks/results/readme-refresh-20261015/refresh.py")
    state = module["verify"].__globals__
    monkeypatch.setitem(state, "ROOT", tmp_path)
    baselines = {}
    populations = {}
    for mode, name in (("none", "summary.json.xz"), ("cuda", "df-summary.json.xz")):
        points = []
        updates = []
        for atoms in (3, 12, 48, 96):
            protocol = {"atoms": atoms, "basis": "retained-basis", "grid": [48, 16, 32]}
            points.append(
                {
                    "atoms": atoms,
                    "protocol": protocol,
                    "engines": {
                        "native": {"historical": True},
                        "reference": {"timing": atoms},
                    },
                    "independent_references": [{"energy": -atoms}],
                }
            )
            rows = [
                {
                    "phase": phase,
                    "energy": -atoms,
                    "complete_seconds": 0.5,
                    "converged": True,
                    "status": 0,
                    "gate": True,
                    "energy_error": 0.0,
                    "force_error": 0.0,
                }
                for phase, count in module["PHASE_COUNTS"].items()
                for _ in range(count)
            ]
            native = {
                "status": "measured",
                "native_build": {
                    "library_sha256": "binary",
                    "probe": {"source_identity": "source"},
                },
                "native_experiment": {
                    "precision_policy": "fp64",
                    "preliminary_provider": "none",
                    "public_initialization_policy": True,
                },
                "records": rows,
                "phase_status": dict.fromkeys(module["PHASE_COUNTS"], "measured"),
            }
            updates.append(
                {
                    "atoms": atoms,
                    "protocol": copy.deepcopy(protocol),
                    "native": native,
                    "cache_priming": {
                        "outcome": {"exit_code": 0},
                        "phase_counts": module["PHASE_COUNTS"],
                        "checked_calls": 12,
                        "fresh_measurement_process": True,
                        "native_build": copy.deepcopy(native["native_build"]),
                    },
                }
            )
        baselines[name] = {"density_fitting": mode == "cuda", "methods": {"hf": points}}
        populations[mode] = {"hf": updates}
    monkeypatch.setitem(state, "baseline_summary", lambda name: baselines[name])
    return {
        "module": module,
        "root": tmp_path,
        "baselines": baselines,
        "campaign": {"source_commit": "snapshot", "populations": populations},
    }


def retain(publication: dict) -> None:
    """Bind each mutated fixture to its actual bytes so semantic gates are tested."""
    root = publication["root"]
    payload = lzma.compress(json.dumps(publication["campaign"]).encode())
    (root / "native-summary.json.xz").write_bytes(payload)
    manifest = {
        "source_commit": "snapshot",
        "native_library_sha256": "binary",
        "native_source_identity": "source",
        "baseline_files": {},
        "files": {
            "native-summary.json.xz": {
                "bytes": len(payload),
                "sha256": hashlib.sha256(payload).hexdigest(),
            }
        },
    }
    settings = {
        "source_commit": "snapshot",
        "native_build": {
            "library_sha256": "binary",
            "probe": {"source_identity": "source"},
        },
    }
    settings_payload = json.dumps(settings).encode() + b"\n"
    (root / "build-settings.json").write_bytes(settings_payload)
    manifest["files"]["build-settings.json"] = {
        "bytes": len(settings_payload),
        "sha256": hashlib.sha256(settings_payload).hexdigest(),
    }
    (root / "manifest.json").write_text(json.dumps(manifest))


@pytest.mark.parametrize(
    "mutation",
    [
        None,
        "force_budget",
        "accuracy",
        "lr_cache",
        "source",
        "phase_status",
        "receipt",
        "missing_cold",
    ],
)
@pytest.mark.parametrize("retained_calls", [1, 4, 6, 7, 11, 12])
def test_budgeted_df96_remains_unchecked_and_resource_bound(
    publication: dict,
    monkeypatch: pytest.MonkeyPatch,
    mutation: str | None,
    retained_calls: int,
) -> None:
    """An explicit-budget extension cannot inherit accuracy or alter production inputs."""
    module = publication["module"]
    baseline = publication["baselines"]["df-summary.json.xz"]
    baseline["methods"]["wb97m-v"] = copy.deepcopy(baseline["methods"]["hf"][:-1])
    updates = copy.deepcopy(publication["campaign"]["populations"]["cuda"]["hf"])
    publication["campaign"]["populations"]["cuda"]["wb97m-v"] = updates
    for point in updates:
        point["native"]["source_file_sha256"] = {
            "production": "retained-production",
            "benchmarks/readme_omol25.py": "historical-consumer",
        }
        point["native"]["native_schedule_settings"] = {
            "GENERATIVEQC_STATIONARY_CACHE": "retained-cache"
        }
    added = updates[-1]
    added["protocol"]["density_fitting_memory_budget_bytes"] = 12 << 30
    monkeypatch.setitem(
        module["verify"].__globals__, "df96_protocol", lambda: added["protocol"]
    )
    native = added["native"]
    native["status"] = "completed_unchecked" if retained_calls == 12 else "timeout"
    native["outcome"] = {"exit_code": 0 if retained_calls == 12 else 124}
    native["records"] = native["records"][:retained_calls]
    native["phase_status"] = {
        phase: "completed_unchecked"
        if sum(row["phase"] == phase for row in native["records"]) == expected
        else "incomplete"
        for phase, expected in module["PHASE_COUNTS"].items()
    }
    native["native_experiment"].update(
        force_max_device_bytes=4 << 30, force_max_host_bytes=4 << 30
    )
    native["native_schedule_settings"]["GENERATIVEQC_KS_RESIDENT_MAX_BYTES"] = str(
        2 << 30
    )
    for row in native["records"]:
        row.update(gate=None, execution_gate=True, energy_error=None, force_error=None)
    consumer = b"synthetic benchmark resource controls\n"
    payload = lzma.compress(consumer)
    identity = hashlib.sha256(consumer).hexdigest()
    native["source_file_sha256"]["benchmarks/readme_omol25.py"] = identity
    if mutation == "force_budget":
        native["native_experiment"]["force_max_host_bytes"] = None
    elif mutation == "accuracy":
        native["records"][0].update(gate=True, energy_error=0.0, force_error=0.0)
    elif mutation == "lr_cache":
        native["native_schedule_settings"].pop("GENERATIVEQC_KS_RESIDENT_MAX_BYTES")
    elif mutation == "source":
        native["source_file_sha256"]["production"] = "different-production"
    elif mutation == "phase_status":
        native["phase_status"]["cold"] = "incomplete"
    elif mutation == "receipt":
        native["outcome"]["exit_code"] = 1
    elif mutation == "missing_cold":
        native["records"].pop(0)
    retain(publication)
    archive = publication["root"] / "df96-benchmark-consumer.py.xz"
    archive.write_bytes(payload)
    path = publication["root"] / "manifest.json"
    manifest = json.loads(path.read_bytes())
    manifest["files"][archive.name] = {
        "bytes": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
        "uncompressed": {"bytes": len(consumer), "sha256": identity},
    }
    path.write_text(json.dumps(manifest))
    if mutation is None:
        expected_counts = {
            "complete_populations": 12 if retained_calls == 12 else 11,
            "checked_calls": 132,
            "unchecked_calls": retained_calls,
        }
        if retained_calls != 12:
            expected_counts["incomplete_populations"] = 1
        assert module["verify"]() == expected_counts
        combined = module["combined_summary"]("df-summary.json.xz")
        engines = combined["methods"]["wb97m-v"][-1]["engines"]
        assert engines["reference"]["status"] == "not_run"
        from tools.render_readme_dft import phase_rows

        assert len(phase_rows(engines["native"], "cold")) == 1
        assert len(phase_rows(engines["native"], "warm")) == (
            5 if retained_calls >= 6 else 0
        )
    else:
        with pytest.raises(ValueError):
            module["verify"]()


def test_refresh_preserves_reference_and_original_baseline(publication: dict) -> None:
    """An overlay changes only native observations, never reference timing/arrays."""
    before = copy.deepcopy(publication["baselines"])
    retain(publication)
    counts = publication["module"]["verify"]()
    assert counts == {"complete_populations": 8, "checked_calls": 96}
    combined = publication["module"]["combined_summary"]("summary.json.xz")
    original = before["summary.json.xz"]["methods"]["hf"][0]
    current = combined["methods"]["hf"][0]
    assert current["engines"]["reference"] == original["engines"]["reference"]
    assert current["independent_references"] == original["independent_references"]
    assert "historical" not in current["engines"]["native"]
    assert publication["baselines"] == before


@pytest.mark.parametrize(
    "failure",
    [
        "protocol",
        "accuracy",
        "sample",
        "primer",
        "precision",
        "missing-build",
        "missing-policy",
        "primer-image",
        "missing-mode",
    ],
)
def test_refresh_rejects_invalid_qualification(publication: dict, failure: str) -> None:
    """A new checksum cannot turn scientific or lifecycle failures into passes."""
    point = publication["campaign"]["populations"]["none"]["hf"][0]
    if failure == "protocol":
        point["protocol"]["grid"] = [16, 16, 16]
    elif failure == "accuracy":
        point["native"]["records"][0]["force_error"] = 1e-5
    elif failure == "sample":
        point["native"]["records"].pop()
    elif failure == "primer":
        point["cache_priming"]["fresh_measurement_process"] = False
    elif failure == "precision":
        point["native"]["native_experiment"]["precision_policy"] = "fast"
    elif failure == "missing-build":
        point["native"].pop("native_build")
    elif failure == "missing-policy":
        point["native"].pop("native_experiment")
    elif failure == "primer-image":
        point["cache_priming"]["native_build"]["library_sha256"] = "stale"
    else:
        publication["campaign"]["populations"].pop("cuda")
    retain(publication)
    with pytest.raises(ValueError):
        publication["module"]["verify"]()


def test_refresh_does_not_invent_missing_accuracy(publication: dict) -> None:
    """Completed native work can remain visible without becoming an accuracy pass."""
    point = publication["campaign"]["populations"]["cuda"]["hf"][-1]
    point["native"]["status"] = "completed_unchecked"
    point["native"]["phase_status"] = dict.fromkeys(
        point["native"]["phase_status"], "completed_unchecked"
    )
    for row in point["native"]["records"]:
        row.update(gate=None, energy_error=None, force_error=None)
    retain(publication)
    counts = publication["module"]["verify"]()
    assert counts["checked_calls"] == 84
    assert counts["unchecked_calls"] == 12


@pytest.mark.parametrize(
    "field", ["source_commit", "native_source_identity", "native_library_sha256"]
)
def test_refresh_rejects_relabeling_the_frozen_build(
    publication: dict, field: str
) -> None:
    """Self-consistent scalar records still need the independently retained build."""
    retain(publication)
    path = publication["root"] / "manifest.json"
    manifest = json.loads(path.read_bytes())
    manifest[field] = "another-build"
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="frozen build"):
        publication["module"]["verify"]()


@pytest.mark.parametrize("corrupt_expanded_identity", [False, True])
def test_refresh_checks_losslessly_archived_figure_identity(
    publication: dict, corrupt_expanded_identity: bool
) -> None:
    """A valid archive checksum cannot conceal changes to historical SVG bytes."""
    retain(publication)
    root = publication["root"]
    original = b"<svg>historical figure</svg>"
    compressed = lzma.compress(original)
    (root / "figure.svg.xz").write_bytes(compressed)
    path = root / "manifest.json"
    manifest = json.loads(path.read_bytes())
    manifest["files"]["figure.svg.xz"] = {
        "bytes": len(compressed),
        "sha256": hashlib.sha256(compressed).hexdigest(),
        "uncompressed": {
            "bytes": len(original),
            "sha256": "wrong"
            if corrupt_expanded_identity
            else hashlib.sha256(original).hexdigest(),
        },
    }
    path.write_text(json.dumps(manifest))
    if corrupt_expanded_identity:
        with pytest.raises(ValueError, match="historical figure identity"):
            publication["module"]["verify"]()
    else:
        publication["module"]["verify"]()
