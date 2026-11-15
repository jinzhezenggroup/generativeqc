"""Replay real positive and negative smoke receipts without promoting defaults."""

from __future__ import annotations

import copy
import hashlib
import json
import lzma
from typing import TYPE_CHECKING

import pytest

from tools.generativeqc_validation.cuda_aot_evidence import apply_endpoint_evidence
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


@pytest.mark.parametrize(
    "mutation",
    (
        "raw-tamper",
        "process-reuse",
        "pid-reuse",
        "device",
        "source",
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
    else:
        if mutation == "pid-reuse":
            sample["pid"] = json.loads(files[index["rows"][1]["file"]])["pid"]
        elif mutation == "device":
            sample["device_uuid"] = "other-physical-device"
        elif mutation == "source":
            sample["probe"]["source_identity"] = "0" * 64
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
