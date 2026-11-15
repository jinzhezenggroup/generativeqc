"""Replay real positive and negative smoke receipts without promoting defaults."""

from __future__ import annotations

import copy
import hashlib
import json
import lzma
from typing import TYPE_CHECKING

import pytest

from tools.generativeqc_validation.cuda_aot_evidence import (
    apply_card_comparison_evidence,
    apply_endpoint_evidence,
)
from tools.generativeqc_validation.cuda_aot_policy import ROOT

if TYPE_CHECKING:
    from pathlib import Path


@pytest.fixture
def endpoint_bundle() -> dict:
    """Use the retained raw oracle/run records, not precomputed passed flags."""
    declaration = json.loads(
        (ROOT / "manifests/maintenance/cuda_aot_capability_evidence.json").read_text()
    )["endpoint_smoke"]
    bundle = json.loads(lzma.decompress((ROOT / declaration["path"]).read_bytes()))
    bundle["control"] = {
        name: bundle["control"][name] for name in ("scontrol.txt", "library-sha256.txt")
    }
    bundle.pop("build_receipts")
    bundle.pop("nondefault_diagnostic_not_acceptance")
    return bundle


def replay(bundle: dict, root: Path, source_identity: str | None = None) -> dict:
    """Keep raw-record digests independent of the container's storage digest."""
    path = root / "endpoint.json.xz"
    path.write_bytes(lzma.compress(json.dumps(bundle).encode()))
    declaration = {
        "path": path.name,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }
    report = {"rows": [{"full_endpoint": "not-run-for-this-audit"}]}
    apply_endpoint_evidence(
        report, root, declaration, source_identity or bundle["native_source_identity"]
    )
    return report


def test_stock_positive_and_failed_force_gates_remain_distinct(
    endpoint_bundle: dict, tmp_path: Path
) -> None:
    report = replay(endpoint_bundle, tmp_path)
    evidence = report["endpoint_smoke_evidence"]
    assert evidence["source_status"] == "current-source"
    assert report["rows"] == [{"full_endpoint": "not-run-for-this-audit"}]
    for arm in ("auto", "portable"):
        assert evidence["results"]["def2-svp"][arm]["processes"] == 4
        phases = evidence["results"]["def2-tzvp"][arm]["phases"]
        assert phases["cold"]["status"] == "failed-scientific-gate"
        assert phases["warm"]["status"] == "failed-scientific-gate"
        assert phases["moved"]["status"] == "passed"
        assert all(
            phase["status"] == "passed"
            for phase in evidence["results"]["def2-svp"][arm]["phases"].values()
        )


def test_endpoint_source_changes_leave_only_historical_observations(
    endpoint_bundle: dict, tmp_path: Path
) -> None:
    report = replay(endpoint_bundle, tmp_path, "0" * 64)
    assert (
        report["endpoint_smoke_evidence"]["source_status"] == "historical-source-only"
    )


def test_translation_replay_preserves_small_residuals_across_python_versions(
    endpoint_bundle: dict, tmp_path: Path
) -> None:
    """Synthetic exact cancellation exposes Python 3.11's naive sum loss."""
    bundle = copy.deepcopy(endpoint_bundle)
    forces = [[1.0, 0.0, 0.0], [1e-16, 0.0, 0.0], [-1.0, 0.0, 0.0]]
    for files in bundle["campaigns"].values():
        index = json.loads(files["index.json"])
        reference = json.loads(files["reference.json"])
        for record in reference["records"].values():
            record["forces"] = forces
        files["reference.json"] = json.dumps(reference)
        index["reference_sha256"] = hashlib.sha256(
            files["reference.json"].encode()
        ).hexdigest()
        for row in index["rows"]:
            sample = json.loads(files[row["file"]])
            for phase, values in sample["phases"].items():
                values["forces"] = forces
                row["validation"][phase]["passed"] = True
            files[row["file"]] = json.dumps(sample)
            row["artifact_sha256"] = hashlib.sha256(
                files[row["file"]].encode()
            ).hexdigest()
        files["index.json"] = json.dumps(index)
    evidence = replay(bundle, tmp_path)["endpoint_smoke_evidence"]
    assert all(
        phase["maximum_translation_error"] == 1e-16
        for arms in evidence["results"].values()
        for arm in arms.values()
        for phase in arm["phases"].values()
    )


@pytest.mark.parametrize(
    "mutation",
    (
        "raw-tamper",
        "process-reuse",
        "pid-reuse",
        "device",
        "source",
        "commit",
        "device-description",
        "profile",
        "partial",
        "cpu",
        "controls",
        "nan",
        "relabeled-failure",
        "library",
        "unlimited",
        "missing-process",
    ),
)
def test_endpoint_integrity_and_negative_qualification_boundaries(
    endpoint_bundle: dict, tmp_path: Path, mutation: str
) -> None:
    bundle = copy.deepcopy(endpoint_bundle)
    files = bundle["campaigns"]["def2-svp"]
    index = json.loads(files["index.json"])
    row = index["rows"][0]
    sample = json.loads(files[row["file"]])
    if mutation == "raw-tamper":
        files[row["file"]] += " "
    elif mutation == "process-reuse":
        index["rows"][1] = copy.deepcopy(row)
    elif mutation == "relabeled-failure":
        failed_files = bundle["campaigns"]["def2-tzvp"]
        failed_index = json.loads(failed_files["index.json"])
        failed_index["rows"][0]["validation"]["cold"]["passed"] = True
        failed_files["index.json"] = json.dumps(failed_index)
    elif mutation == "library":
        row["library_sha256"] = "0" * 64
    elif mutation == "unlimited":
        bundle["control"]["scontrol.txt"] = "JobId=17464 TimeLimit=UNLIMITED"
    elif mutation == "missing-process":
        index["rows"].pop()
    elif mutation == "commit":
        index["source_commit"] = "1" * 40
    else:
        if mutation == "pid-reuse":
            sample["pid"] = json.loads(files[index["rows"][1]["file"]])["pid"]
        elif mutation == "device":
            sample["device_uuid"] = "other-physical-device"
        elif mutation == "source":
            sample["probe"]["source_identity"] = "0" * 64
        elif mutation == "device-description":
            sample["probe"]["device"]["major"] = 9
        elif mutation == "profile":
            sample["probe"]["device"]["official_profile"] = "portable_cuda"
        elif mutation == "partial":
            sample["phases"].pop("warm")
        elif mutation == "cpu":
            sample["phases"]["cold"]["executed_backend"] = "cpu"
        elif mutation == "controls":
            sample["nondefault_controls"] = {"density_tolerance": 1e-10}
        else:
            sample["phases"]["cold"]["forces"][0][2] = float("nan")
        files[row["file"]] = json.dumps(sample)
        row["artifact_sha256"] = hashlib.sha256(files[row["file"]].encode()).hexdigest()
    files["index.json"] = json.dumps(index)
    with pytest.raises(ValueError):
        replay(bundle, tmp_path)


@pytest.mark.parametrize("historical", (False, True))
@pytest.mark.parametrize(
    "mutation", (None, "same-card", "binary", "source", "build-origin", "build-receipt")
)
def test_same_binary_card_comparison_preserves_qualification_boundaries(
    endpoint_bundle: dict, tmp_path: Path, historical: bool, mutation: str | None
) -> None:
    identity = "0" * 64 if historical else endpoint_bundle["native_source_identity"]
    primary = copy.deepcopy(endpoint_bundle)
    primary["build_receipts"] = {
        "fixture.txt": "Synthetic shared-receipt checker input"
    }
    report = replay(primary, tmp_path, identity)
    secondary = copy.deepcopy(endpoint_bundle)
    secondary["build_receipts"] = {
        "retained_bundle": {
            "path": "endpoint.json.xz",
            "sha256": hashlib.sha256(
                (tmp_path / "endpoint.json.xz").read_bytes()
            ).hexdigest(),
        },
        "section": "build_receipts",
        "file_sha256": {
            name: hashlib.sha256(text.encode()).hexdigest()
            for name, text in primary["build_receipts"].items()
        },
    }
    if mutation == "build-origin":
        secondary["build_receipts"]["retained_bundle"]["sha256"] = "0" * 64
    elif mutation == "build-receipt":
        secondary["build_receipts"]["file_sha256"]["fixture.txt"] = "0" * 64
    if mutation == "source":
        secondary["native_source_identity"] = "1" * 64
    for files in secondary["campaigns"].values():
        index = json.loads(files["index.json"])
        index["native_source_identity"] = secondary["native_source_identity"]
        for row in index["rows"]:
            sample = json.loads(files[row["file"]])
            sample["probe"]["source_identity"] = secondary["native_source_identity"]
            if mutation != "same-card":
                sample["device_uuid"] = "distinct-physical-card"
                sample["probe"]["device"]["name"] = "NVIDIA RTX PRO 6000"
            if mutation == "binary" and row["arm"] == "auto":
                old_digest = row["library_sha256"]
                secondary["control"]["library-sha256.txt"] = secondary["control"][
                    "library-sha256.txt"
                ].replace(old_digest, "2" * 64)
                row["library_sha256"] = "2" * 64
            files[row["file"]] = json.dumps(sample)
            row["artifact_sha256"] = hashlib.sha256(
                files[row["file"]].encode()
            ).hexdigest()
        files["index.json"] = json.dumps(index)
    path = tmp_path / "secondary.json.xz"
    path.write_bytes(lzma.compress(json.dumps(secondary).encode()))
    declaration = {
        "path": path.name,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }
    if mutation:
        with pytest.raises(ValueError, match="card comparison"):
            apply_card_comparison_evidence(report, tmp_path, declaration, identity)
        return
    apply_card_comparison_evidence(report, tmp_path, declaration, identity)
    comparison = report["same_binary_card_comparison"]
    assert comparison["primary_device_uuid"] != comparison["secondary"]["device_uuid"]
    assert comparison["secondary"]["source_status"] == (
        "historical-source-only" if historical else "current-source"
    )
    assert (
        comparison["secondary"]["results"]["def2-tzvp"]["auto"]["phases"]["cold"][
            "status"
        ]
        == "failed-scientific-gate"
    )
    assert report["rows"] == [{"full_endpoint": "not-run-for-this-audit"}]
