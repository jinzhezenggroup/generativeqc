"""A small publication checks summaries; missing raw never earns raw qualification."""

from __future__ import annotations

import copy
import gzip
import hashlib
import json
import sys
from pathlib import Path

import pytest

from benchmarks.cc_cupti_capsule import ORACLE_SHA, compressed_json, verify_summary
from benchmarks.cc_trace_reader import InvalidTrace

BUNDLE = (
    Path(__file__).resolve().parents[2]
    / "benchmarks/results/cc-cupti-attribution-230ao-20261019"
)


def published() -> tuple[dict, dict]:
    capsule = json.loads(gzip.decompress((BUNDLE / "campaign.json.gz").read_bytes()))
    oracle_bytes = gzip.decompress((BUNDLE / "oracle.json.gz").read_bytes())
    assert hashlib.sha256(oracle_bytes).hexdigest() == ORACLE_SHA
    return capsule, json.loads(oracle_bytes)


def test_summary_recomputes_scalar_contract_but_cannot_promote_missing_raw() -> None:
    capsule, oracle = published()
    result = verify_summary(capsule, oracle)
    assert result["derived_scalar_bits_work_summary"].startswith("PASS")
    assert result["full_raw_qualification"].startswith("UNAVAILABLE")
    assert result["whole_program_fine_attribution"] == "INCOMPLETE"
    assert result["implicit_wait_causality_profitability"] == "UNAVAILABLE"
    assert result["reported_exclusive_ledger_ns"]["unattributed"] > 0
    assert "late" in result["execution_supervision_limitation"].lower()


@pytest.mark.parametrize(
    "corruption",
    [
        "argv",
        "workspace",
        "epoch",
        "work",
        "bits",
        "ledger",
        "anchor",
        "source",
        "witness",
        "locale",
    ],
)
def test_derived_summary_corruption_refuses(corruption: str) -> None:
    capsule, oracle = copy.deepcopy(published())
    child = capsule["children"][0]
    if corruption == "argv":
        child["plan"]["normalized_argv"].append("--unadmitted")
    elif corruption == "workspace":
        child["plan"]["profile"]["variables"]["CUBLAS_WORKSPACE_CONFIG"] = {
            "present": False,
            "value": None,
        }
    elif corruption == "epoch":
        child["witness"]["process"]["starttime"] = "999999"
    elif corruption == "work":
        child["endpoint"]["df_transform_fmas"] += 1
    elif corruption == "bits":
        child["endpoint"]["total_energy"] += 1e-12
    elif corruption == "ledger":
        capsule["reported_audit"]["selected_phase_qualification"][
            "exclusive_ledger_ns"
        ]["unattributed"] += 1
    elif corruption == "anchor":
        capsule["original_documents"]["spec.json"]["sha256"] = "0" * 64
    elif corruption == "source":
        capsule["identities"]["production_source"] = "0" * 40
    elif corruption == "witness":
        child["witness"]["status"] = "COMPLETE"
    else:
        child["plan"]["locale"]["LC_ALL"]["value"] = "unadmitted"
    with pytest.raises(InvalidTrace):
        verify_summary(capsule, oracle)


@pytest.mark.parametrize("rebinding", ["library", "scalar"])
def test_full_replay_rejects_self_consistent_capsule_rebinding(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, rebinding: str
) -> None:
    from benchmarks import cc_cupti_capsule as module

    original, oracle = published()
    changed = copy.deepcopy(original)
    if rebinding == "library":
        changed["identities"]["artifact_sha256"]["library"] = "0" * 64
        changed["reported_audit"]["frozen_packet"]["library_sha256"] = "0" * 64
    else:
        for child in changed["children"]:
            child["endpoint"]["total_energy"] += 1e-12
    assert verify_summary(changed, oracle)["full_raw_qualification"].startswith(
        "UNAVAILABLE"
    )
    oracle_bytes = gzip.decompress((BUNDLE / "oracle.json.gz").read_bytes())
    monkeypatch.setattr(
        module,
        "compressed_json",
        lambda p: (
            (changed, b"derived") if p.name == "capsule" else (oracle, oracle_bytes)
        ),
    )
    monkeypatch.setattr(
        module, "audit_campaign", lambda *args: original["reported_audit"]
    )
    monkeypatch.setattr(module, "derive_capsule", lambda *args: original)
    output = tmp_path / "must-not-certify.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "capsule",
            "--capsule",
            "capsule",
            "--oracle",
            "oracle",
            "--campaign",
            str(tmp_path / "original"),
            "--output",
            str(output),
        ],
    )
    with pytest.raises(InvalidTrace, match="projection"):
        module.main()
    assert not output.exists()


@pytest.mark.parametrize("source_available", [False, True])
def test_full_evidence_requires_both_original_raw_and_tooling_recovery(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, source_available: bool
) -> None:
    from benchmarks import cc_cupti_capsule as module

    capsule, oracle = published()
    oracle_bytes = gzip.decompress((BUNDLE / "oracle.json.gz").read_bytes())
    monkeypatch.setattr(
        module,
        "compressed_json",
        lambda p: (
            (capsule, b"derived") if p.name == "capsule" else (oracle, oracle_bytes)
        ),
    )
    monkeypatch.setattr(
        module, "audit_campaign", lambda *args: capsule["reported_audit"]
    )
    monkeypatch.setattr(module, "derive_capsule", lambda *args: capsule)
    monkeypatch.setattr(
        module,
        "recover_tooling_sources",
        lambda *args: {"status": "PASS_RECONSTRUCTED_17_MEASURED_TOOLING_INPUTS"},
    )
    output = tmp_path / "new" / "nested" / "result.json"
    argv = [
        "capsule",
        "--capsule",
        "capsule",
        "--oracle",
        "oracle",
        "--campaign",
        str(tmp_path / "original"),
        "--output",
        str(output),
    ]
    if source_available:
        argv += ["--tooling-output", str(tmp_path / "recovered")]
    monkeypatch.setattr(sys, "argv", argv)
    assert module.main() == 0
    result = json.loads(output.read_bytes())
    assert result["full_raw_qualification"].startswith("PASS")
    assert result["full_evidence_reproduction"].startswith(
        "PASS" if source_available else "UNAVAILABLE"
    )


def test_unavailable_source_stops_full_replay_without_claiming_pass(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from benchmarks import cc_cupti_capsule as module
    from benchmarks.cc_cupti_source_recovery import SourceUnavailable

    capsule, oracle = published()
    oracle_bytes = gzip.decompress((BUNDLE / "oracle.json.gz").read_bytes())
    monkeypatch.setattr(
        module,
        "compressed_json",
        lambda p: (
            (capsule, b"derived") if p.name == "capsule" else (oracle, oracle_bytes)
        ),
    )

    def missing(*args: object) -> None:
        raise SourceUnavailable("controlled missing local publication tree")

    def forbidden(*args: object) -> None:
        pytest.fail("missing source entered the expensive original raw replay")

    monkeypatch.setattr(module, "recover_tooling_sources", missing)
    monkeypatch.setattr(module, "audit_campaign", forbidden)
    output = tmp_path / "new" / "nested" / "unavailable.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "capsule",
            "--capsule",
            "capsule",
            "--oracle",
            "oracle",
            "--campaign",
            str(tmp_path / "original"),
            "--tooling-output",
            str(tmp_path / "recovered"),
            "--output",
            str(output),
        ],
    )
    assert module.main() == 1
    result = json.loads(output.read_bytes())
    assert result["tooling_source_recovery"]["status"] == "UNAVAILABLE"
    assert result["full_raw_qualification"].startswith("UNAVAILABLE")
    assert result["full_evidence_reproduction"].startswith("UNAVAILABLE")


@pytest.mark.skipif(
    sys.platform != "linux", reason="actual opened regular nofollow/nonblock input"
)
@pytest.mark.parametrize(
    "corruption", ["trailing_member", "expansion_bomb", "symlink", "fifo"]
)
def test_compressed_input_remains_bounded_and_regular(
    tmp_path: Path, corruption: str
) -> None:
    import os

    path = tmp_path / "capsule.json.gz"
    if corruption == "trailing_member":
        path.write_bytes(gzip.compress(b"{}") + gzip.compress(b"{}"))
    elif corruption == "expansion_bomb":
        path.write_bytes(
            gzip.compress(json.dumps({"blob": "x" * (2 * 1024 * 1024)}).encode())
        )
    elif corruption == "symlink":
        path.symlink_to(BUNDLE / "campaign.json.gz")
    else:
        os.mkfifo(path)
    with pytest.raises(InvalidTrace):
        compressed_json(path)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
def test_compressed_input_rejects_nonfinite_nested_metadata(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, value: float
) -> None:
    from benchmarks import cc_cupti_capsule as module

    raw = gzip.compress(json.dumps({"metadata": {"value": value}}).encode())
    monkeypatch.setattr(module, "closed_bytes", lambda _: raw)
    with pytest.raises(ValueError):
        module.compressed_json(tmp_path / "capsule.json.gz")
