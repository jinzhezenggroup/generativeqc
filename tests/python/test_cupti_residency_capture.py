"""Synthetic activity protocol tests, not CUDA coverage/qualification evidence."""

from __future__ import annotations

import copy
from typing import Any

import pytest

from tools.audit_replay_allocations import InvalidReceipt
from tools.cupti_residency_capture import FIELDS, METADATA, summarize


def record(kind: int, **values: Any) -> dict[str, Any]:
    row = dict.fromkeys(FIELDS, 0)
    row.update(kind=kind, api_name=None)
    row.update(values)
    return row


def fixture() -> dict[str, Any]:
    records = [
        record(
            1,
            subtype=1,
            bytes=128,
            start=100,
            end=110,
            correlation_id=1,
            runtime_correlation_id=2,
            graph_node=99,
            graph=11,
            copy_count=1,
        ),
        record(38, subtype=4, start=120, end=130, correlation_id=3),
        record(39, subtype=3, correlation_id=1, external_id=1),
        record(39, subtype=3, correlation_id=2, external_id=1),
        record(39, subtype=3, correlation_id=3, external_id=8),
        record(5, correlation_id=2, start=90, end=115, api_name="cudaMemcpyAsync"),
        record(4, correlation_id=1, start=95, end=112, api_name="cuMemcpyHtoDAsync"),
    ]
    metadata = dict.fromkeys(METADATA, 0)
    metadata.update(record_count=len(records), capacity=32, cupti_version=28, stopped=1)
    return {
        "schema": "generativeqc.cupti-activity.v1",
        "metadata": metadata,
        "records": records,
    }


REGIONS = {
    1: {"name": "warm", "role": "replay"},
    8: {"name": "fence", "role": "observer"},
}


def test_real_activity_not_nested_api_calls_is_counted_and_fence_is_separate() -> None:
    result = summarize(fixture(), REGIONS)
    assert result["activity_stream_intact"]
    assert result["regions"][0]["h2d_bytes"] == 128
    assert result["regions"][0]["transfer_records"] == 1
    assert result["regions"][0]["graph_transfer_records"] == 1
    assert result["regions"][0]["syncs"] == 0
    assert result["regions"][1]["syncs"] == 1
    assert result["status"] == "INCOMPLETE"
    assert result["payload_links_complete"] is False
    assert result["residency_coverage_complete"] is False
    assert result["zero_round_trip_assertion"] is None
    assert all(row["payload_identity"] is None for row in result["events"])


def test_buffer_delivery_order_does_not_guess_phases() -> None:
    raw = fixture()
    raw["records"].reverse()
    assert (
        summarize(raw, REGIONS)["regions"] == summarize(fixture(), REGIONS)["regions"]
    )


@pytest.mark.parametrize(
    "field",
    [
        "dropped_records",
        "buffer_starvations",
        "parse_errors",
        "unknown_records",
        "buffers_pending",
        "finish_errors",
    ],
)
def test_native_loss_and_incomplete_stop_prevent_intact_claim(field: str) -> None:
    raw = fixture()
    raw["metadata"][field] = 1
    result = summarize(raw, REGIONS)
    assert not result["activity_stream_intact"]
    assert result["status"] == "INCOMPLETE"


@pytest.mark.parametrize(
    "mutation,match",
    [
        ("count", "count mismatch"),
        ("version", "unqualified"),
        ("duplicate", "conflicting"),
        ("phase", "disagreement"),
        ("bool", "uint64"),
        ("unknown", "kind"),
        ("fields", "fields"),
    ],
)
def test_malformed_or_ambiguous_activity_is_rejected(mutation: str, match: str) -> None:
    raw = fixture()
    if mutation == "count":
        raw["metadata"]["record_count"] -= 1
    elif mutation == "version":
        raw["metadata"]["cupti_version"] = 27
    elif mutation == "duplicate":
        raw["records"].append(record(39, subtype=3, correlation_id=1, external_id=8))
        raw["metadata"]["record_count"] += 1
    elif mutation == "phase":
        raw["records"][3]["external_id"] = 8
    elif mutation == "bool":
        raw["records"][0]["bytes"] = True
    elif mutation == "unknown":
        raw["records"][0]["kind"] = 7
    else:
        raw["records"][0]["invented"] = 1
    with pytest.raises(InvalidReceipt, match=match):
        summarize(raw, REGIONS)


def test_unassigned_activity_is_retained_not_hidden_as_zero() -> None:
    raw = fixture()
    raw["records"][0]["correlation_id"] = 10
    raw["records"][0]["runtime_correlation_id"] = 11
    result = summarize(raw, REGIONS)
    assert not result["activity_stream_intact"]
    assert result["unassigned"]["h2d_bytes"] == 128
    assert result["regions"][0]["h2d_bytes"] == 0
    assert result["zero_round_trip_assertion"] is None


def test_batch_multiplicity_unknown_timestamps_and_failed_waits_are_explicit() -> None:
    raw = fixture()
    raw["records"][0]["copy_count"] = 3
    assert summarize(raw, REGIONS)["regions"][0]["copy_operations"] == 3
    raw["records"][0]["start"] = 0
    raw["records"][1]["return_value"] = 4
    result = summarize(raw, REGIONS)
    assert not result["activity_stream_intact"]
    assert result["regions"][1]["syncs"] == 0
    assert result["events"][1]["return_value"] == 4


def test_source_snapshots_are_not_mutated_by_summarization() -> None:
    raw = fixture()
    before = copy.deepcopy(raw)
    result = summarize(raw, REGIONS)
    result["regions"][0]["region"]["name"] = "changed"
    assert raw == before
    assert REGIONS[1]["name"] == "warm"


@pytest.mark.parametrize(
    "api,subtype,counter",
    [
        ("cudaEventQuery_v3020", 1, "event_queries"),
        ("cuEventQuery", 1, "event_queries"),
        ("cudaStreamQuery_ptsz_v7000", 3, "stream_queries"),
        ("cuStreamQuery", 3, "stream_queries"),
    ],
)
@pytest.mark.parametrize("status", [0, 600])
def test_nonblocking_ready_and_not_ready_queries_are_not_waits(
    api: str,
    subtype: int,
    counter: str,
    status: int,
) -> None:
    raw = fixture()
    raw["records"][1].update(subtype=subtype, return_value=status)
    raw["records"].append(
        record(5, correlation_id=3, api_name=api, return_value=status)
    )
    raw["metadata"]["record_count"] += 1
    result = summarize(raw, REGIONS)
    observer = result["regions"][1]
    assert result["activity_stream_intact"]
    assert observer[counter] == 1
    assert observer["syncs"] == observer["event_waits"] == 0
    assert result["events"][1]["kind"] == "query"
    assert result["events"][1]["return_value"] == status


def test_ambiguous_event_query_or_synchronization_without_api_never_counts_wait() -> (
    None
):
    raw = fixture()
    raw["records"][1]["subtype"] = 1
    result = summarize(raw, REGIONS)
    assert not result["activity_stream_intact"]
    assert result["regions"][1]["event_waits"] == 0
    assert result["zero_round_trip_assertion"] is None


def test_zero_observed_graph_waits_or_empty_activity_cannot_certify_residency() -> None:
    raw = fixture()
    raw["records"] = []
    raw["metadata"]["record_count"] = 0
    result = summarize(raw, REGIONS)
    assert result["status"] == "INCOMPLETE"
    assert not result["residency_coverage_complete"]
    assert result["zero_round_trip_assertion"] is None
