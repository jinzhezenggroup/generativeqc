"""Independent arithmetic and adversarial contracts for the residency consumer."""

from __future__ import annotations

import copy
import hashlib
import json
import subprocess
import sys
from pathlib import Path

from tools.audit_residency_receipts import audit, audit_df_trace

ROOT = Path(__file__).resolve().parents[2]
EVIDENCE = ROOT / "benchmarks/results/residency-receipts-1629"
OLD = ROOT / "benchmarks/results/issue206-practical-auxiliary"


def contract() -> dict:
    value = json.loads((EVIDENCE / "historical-oh-df-contract.json").read_text())
    value["identity"]["source_commit"] = "1" * 40
    value["identity"]["native_source"] = "2" * 64
    value["identity"]["build_sha256"] = "a" * 64
    value["identity"]["device"] = "test CUDA"
    value["identity"]["resource"] = "test resource"
    return value


def event(
    seq: int,
    region: str,
    direction: str,
    amount: int,
    payload: str,
    *,
    derived_from: int | None = None,
    dependency_domain: str = "density",
) -> dict:
    role = "replay" if region == "force_response" else "publication"
    return {
        "seq": seq,
        "region": region,
        "role": role,
        "owner": "DF force component trace",
        "domain": "OH prepared DF force",
        "kind": "transfer",
        "direction": direction,
        "bytes": amount,
        "payload": payload,
        "dependency_domain": dependency_domain,
        "derived_from": derived_from,
    }


def sync(seq: int, region: str, sync_type: str = "stream") -> dict:
    return {
        "seq": seq,
        "region": region,
        "role": "replay" if region == "force_response" else "publication",
        "owner": "DF force component trace",
        "domain": "OH prepared DF force",
        "kind": "sync",
        "sync_type": sync_type,
    }


def receipt(expected: dict, events: list[dict]) -> dict:
    return {
        "schema": "generativeqc.residency_receipt",
        "version": 1,
        "identity": copy.deepcopy(expected["identity"]),
        "completed": True,
        "execution": "stream",
        "coverage": {
            "regions": ["force_response", "final_state_validation"],
            "transfers": True,
            "synchronizations": True,
            "payload_links": True,
        },
        "event_count": len(events),
        "events": events,
    }


def test_oracle_totals_roles_payload_and_replay_sync_ratchet() -> None:
    expected = contract()
    events = [
        event(0, "force_response", "d2h", 32, "density-A"),
        event(1, "force_response", "h2d", 24, "density-A", derived_from=0),
        event(2, "force_response", "d2d", 64, "scratch"),
        sync(3, "force_response"),
        sync(4, "force_response", "event_wait"),
        event(5, "final_state_validation", "d2h", 128, "final-F"),
        sync(6, "final_state_validation"),
    ]
    observed = audit(expected, receipt(expected, events))
    assert observed["status"] == "FAIL"
    assert observed["totals"]["by_role"]["replay"] == {
        "h2d_bytes": 24,
        "d2h_bytes": 32,
        "d2d_bytes": 64,
        "syncs": 1,
        "event_waits": 1,
    }
    assert observed["totals"]["by_role"]["publication"]["d2h_bytes"] == 128
    assert observed["totals"]["round_trips"] == [
        {
            "d2h_seq": 0,
            "h2d_seq": 1,
            "payload": "density-A",
            "dependency_domain": "density",
        }
    ]
    assert observed["candidate_replay_syncs"] == 2
    assert observed["publication"]["syncs"] == 1
    assert any("round_trips" in reason for reason in observed["ratchet_failures"])
    expected["regions"][0]["ratchet"].update(
        {
            "h2d_bytes": 24,
            "d2h_bytes": 32,
            "d2d_bytes": 64,
            "syncs": 1,
            "event_waits": 1,
            "round_trips": 1,
        }
    )
    assert audit(expected, receipt(expected, events))["status"] == "PASS"


def test_unrelated_directions_do_not_make_a_round_trip() -> None:
    expected = contract()
    events = [
        event(0, "force_response", "d2h", 8, "energy"),
        event(1, "force_response", "h2d", 8, "density"),
    ]
    result = audit(expected, receipt(expected, events))
    assert result["totals"]["round_trips"] == []
    events[1]["derived_from"] = 0
    assert audit(expected, receipt(expected, events))["status"] == "INCOMPLETE"
    events[1]["payload"] = "energy"
    events[1]["dependency_domain"] = "unrelated"
    assert audit(expected, receipt(expected, events))["status"] == "INCOMPLETE"


def test_repeated_payload_round_trips_pair_by_exact_sequence() -> None:
    expected = contract()
    expected["regions"][0]["ratchet"].update(
        {"h2d_bytes": 16, "d2h_bytes": 16, "round_trips": 2}
    )
    events = [
        event(0, "force_response", "d2h", 8, "density-A"),
        event(1, "force_response", "h2d", 8, "density-A", derived_from=0),
        event(2, "force_response", "d2h", 8, "density-A"),
        event(3, "force_response", "h2d", 8, "density-A", derived_from=2),
    ]
    result = audit(expected, receipt(expected, events))
    assert result["status"] == "PASS"
    assert [
        (row["d2h_seq"], row["h2d_seq"]) for row in result["totals"]["round_trips"]
    ] == [
        (0, 1),
        (2, 3),
    ]
    events[-1]["derived_from"] = 0
    assert audit(expected, receipt(expected, events))["status"] == "PASS"
    events[-1]["derived_from"] = 1
    assert audit(expected, receipt(expected, events))["status"] == "INCOMPLETE"


def test_identity_role_owner_and_event_stream_tampering_is_incomplete() -> None:
    expected = contract()
    base = receipt(
        expected, [sync(0, "force_response"), sync(1, "final_state_validation")]
    )
    variants = []
    for key, value in (
        ("source_commit", "wrong"),
        ("build_sha256", "b" * 64),
        ("backend", "cpu"),
        ("device", "different"),
        ("endpoint", "different"),
        ("owner", "different"),
    ):
        candidate = copy.deepcopy(base)
        candidate["identity"][key] = value
        variants.append(candidate)
    for key in ("role", "owner", "domain"):
        candidate = copy.deepcopy(base)
        candidate["events"][0][key] = "wrong"
        variants.append(candidate)
    for sequence in ((0, 0), (1, 0), (0, 2)):
        candidate = copy.deepcopy(base)
        for row, number in zip(candidate["events"], sequence):
            row["seq"] = number
        variants.append(candidate)
    unknown = copy.deepcopy(base)
    unknown["events"][0]["kind"] = "unknown"
    variants.append(unknown)
    for candidate in variants:
        assert audit(expected, candidate)["status"] == "INCOMPLETE"


def test_overflow_setup_hot_capture_and_missing_coverage() -> None:
    expected = contract()
    publication = receipt(
        expected, [event(0, "final_state_validation", "d2h", 16, "published")]
    )
    assert audit(expected, publication)["status"] == "PASS"
    expected["regions"].append(
        {
            "name": "initial_upload",
            "role": "prepare",
            "owner": "DF force component trace",
            "domain": "OH prepared DF force",
            "hot": False,
            "ratchet": dict.fromkeys(
                (
                    "h2d_bytes",
                    "d2h_bytes",
                    "d2d_bytes",
                    "syncs",
                    "event_waits",
                    "round_trips",
                ),
                0,
            ),
        }
    )
    setup = receipt(expected, [event(0, "initial_upload", "h2d", 32, "seed")])
    setup["events"][0]["role"] = "prepare"
    setup["coverage"]["regions"].append("initial_upload")
    assert audit(expected, setup)["status"] == "PASS"
    hot = copy.deepcopy(setup)
    hot["events"][0].update({"region": "force_response", "role": "replay"})
    assert audit(expected, hot)["status"] == "FAIL"
    expected["regions"].pop()
    empty_capture = receipt(expected, [])
    empty_capture["execution"] = "graph_capture"
    assert audit(expected, empty_capture)["status"] == "INCOMPLETE"
    expected["regions"][0]["ratchet"]["h2d_bytes"] = (1 << 64) - 1
    oversized = receipt(
        expected,
        [
            event(0, "force_response", "h2d", (1 << 64) - 1, "a"),
            event(1, "force_response", "h2d", 1, "b"),
        ],
    )
    assert audit(expected, oversized)["status"] == "INCOMPLETE"
    for mutation in (
        "capture",
        "unfinished",
        "regions",
        "transfers",
        "syncs",
        "links",
        "count",
    ):
        candidate = copy.deepcopy(publication)
        if mutation == "capture":
            candidate["execution"] = "graph_capture"
        elif mutation == "unfinished":
            candidate["completed"] = False
        elif mutation == "regions":
            candidate["coverage"]["regions"] = ["final_state_validation"]
        elif mutation == "transfers":
            candidate["coverage"]["transfers"] = False
        elif mutation == "syncs":
            candidate["coverage"]["synchronizations"] = False
        elif mutation == "links":
            candidate["coverage"]["payload_links"] = False
        else:
            candidate["event_count"] += 1
        assert audit(expected, candidate)["status"] == "INCOMPLETE"


def test_oracle_and_compatibility_are_visible_without_hot_exemption() -> None:
    expected = contract()
    for role in ("oracle", "compatibility"):
        expected["regions"].append(
            {
                "name": role,
                "role": role,
                "owner": "DF force component trace",
                "domain": "OH prepared DF force",
                "hot": False,
                "ratchet": dict.fromkeys(
                    (
                        "h2d_bytes",
                        "d2h_bytes",
                        "d2d_bytes",
                        "syncs",
                        "event_waits",
                        "round_trips",
                    ),
                    0,
                ),
            }
        )
    observed = receipt(
        expected,
        [
            event(0, "oracle", "d2h", 17, "reference"),
            event(1, "compatibility", "h2d", 23, "fallback"),
        ],
    )
    observed["events"][0]["role"] = "oracle"
    observed["events"][1]["role"] = "compatibility"
    observed["coverage"]["regions"].extend(("oracle", "compatibility"))
    result = audit(expected, observed)
    assert result["status"] == "PASS"
    assert result["totals"]["by_role"]["oracle"]["d2h_bytes"] == 17
    assert result["totals"]["by_role"]["compatibility"]["h2d_bytes"] == 23


def test_historical_production_trace_is_source_matched_but_incomplete() -> None:
    historical = json.loads((EVIDENCE / "historical-oh-df-contract.json").read_text())
    manifest = json.loads((OLD / "manifest.json").read_text())
    trace = OLD / "diagnosis/control-diagnosis-v1/oh-def2-svp-spherical-uhf-auto.jsonl"
    result = audit_df_trace(historical, trace, manifest)
    assert result["status"] == "INCOMPLETE"
    assert (
        result["trace_sha256"]
        == "4a57178922c890bf16830087c26a3971f4970ea6d267d1ab62508dd84e35c6c0"
    )
    assert result["records"] == 8
    assert result["observed_counters"] == {
        "h2d_bytes": 300360,
        "d2h_bytes": 6048,
        "d2d_bytes": 0,
        "syncs": 6,
        "event_waits": 0,
    }
    assert result["diagnostic_profiler_events"] == 1562
    assert result["diagnostic_final_waits"] == 8
    historical["identity"]["native_source"] = "3" * 64
    assert audit_df_trace(historical, trace, manifest)["status"] == "INCOMPLETE"


def test_historical_trace_sha_tampering_and_empty_stream_are_incomplete(
    tmp_path: Path,
) -> None:
    historical = json.loads((EVIDENCE / "historical-oh-df-contract.json").read_text())
    manifest = json.loads((OLD / "manifest.json").read_text())
    trace = OLD / "diagnosis/control-diagnosis-v1/oh-def2-svp-spherical-uhf-auto.jsonl"
    altered = copy.deepcopy(manifest)
    matching = next(
        row
        for row in altered["records"]
        if row["retained"].endswith("oh-def2-svp-spherical-uhf-auto.jsonl")
    )
    matching["stored_sha256"] = "0" * 64
    assert "SHA-256 mismatch" in audit_df_trace(historical, trace, altered)["reason"]
    altered = copy.deepcopy(manifest)
    altered["schema"] = "wrong"
    assert audit_df_trace(historical, trace, altered) == {
        "status": "INCOMPLETE",
        "reason": "unexpected historical retention manifest schema",
    }

    altered = copy.deepcopy(manifest)
    empty = tmp_path / "empty.jsonl"
    empty.write_bytes(b"")
    blank_hash = hashlib.sha256(b"").hexdigest()
    altered["records"] = [
        {
            "retained": empty.as_posix(),
            "stored_sha256": blank_hash,
            "original_sha256": blank_hash,
        }
    ]
    assert audit_df_trace(historical, empty, altered) == {
        "status": "INCOMPLETE",
        "reason": "empty DF trace",
    }


def test_cli_bad_json_reports_incomplete(tmp_path: Path) -> None:
    expected = tmp_path / "contract.json"
    observed = tmp_path / "receipt.json"
    expected.write_text(json.dumps(contract()), encoding="utf-8")
    observed.write_text("{bad", encoding="utf-8")
    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "tools/audit_residency_receipts.py"),
            "--contract",
            str(expected),
            "--receipt",
            str(observed),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 2
    assert json.loads(result.stdout)["status"] == "INCOMPLETE"
