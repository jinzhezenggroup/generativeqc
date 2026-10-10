"""Protocol tests: source graphs complement, never magically complete, CUPTI."""

from __future__ import annotations

import copy
from typing import Any

import pytest

from tools.audit_replay_allocations import InvalidReceipt
from tools.cupti_graph_inventory import FIELDS, METADATA, summarize_graphs
from tools.cupti_residency_capture import FIELDS as ACTIVITY_FIELDS
from tools.cupti_residency_capture import METADATA as ACTIVITY_METADATA

REGIONS = {1: {"name": "warm", "role": "replay"}}


def source_record(kind: int, **values: Any) -> dict[str, Any]:
    """Construct explicit scalar wire fields, not production execution evidence."""
    result = dict.fromkeys(FIELDS, 0)
    result.update(kind=kind, generation=1, role=1, graph=10, executable=20)
    result.update(values)
    return result


def activity_record(kind: int, **values: Any) -> dict[str, Any]:
    result = dict.fromkeys(ACTIVITY_FIELDS, 0)
    result.update(kind=kind, api_name=None)
    result.update(values)
    return result


def fixture() -> tuple[dict[str, Any], dict[str, Any]]:
    """One host submission with event/semaphore wait nodes but no sync activity."""
    source = [
        source_record(1),
        source_record(2, node=30, node_type=6),
        source_record(2, node=31, node_type=4),
        source_record(2, node=32, node_type=9, graph=11, parent_node=31),
        source_record(3, launch_id=1),
        source_record(4, launch_id=1),
        source_record(5),
    ]
    metadata = dict.fromkeys(METADATA, 0)
    metadata.update(
        record_count=len(source), capacity=32, node_limit=8, depth_limit=4, stopped=1
    )
    raw = {
        "schema": "generativeqc.cupti-graph-inventory.v1",
        "metadata": metadata,
        "source_dispatch_errors": 0,
        "records": source,
    }
    activities = [
        activity_record(
            5, correlation_id=100, api_name="cudaGraphLaunch_v10000", start=100, end=200
        ),
        activity_record(
            4, correlation_id=101, api_name="cuGraphLaunch_ptsz", start=110, end=190
        ),
        activity_record(39, subtype=3, correlation_id=100, external_id=1),
        activity_record(39, subtype=3, correlation_id=101, external_id=1),
        activity_record(39, subtype=4, correlation_id=100, external_id=1),
        activity_record(39, subtype=4, correlation_id=101, external_id=1),
    ]
    activity_metadata = dict.fromkeys(ACTIVITY_METADATA, 0)
    activity_metadata.update(
        record_count=len(activities), capacity=32, cupti_version=28, stopped=1
    )
    activity = {
        "schema": "generativeqc.cupti-activity.v1",
        "metadata": activity_metadata,
        "records": activities,
    }
    return raw, activity


def test_zero_cupti_waits_does_not_hide_real_graph_wait_nodes() -> None:
    raw, activity = fixture()
    result = summarize_graphs(raw, activity, REGIONS)
    assert result["source_inventory_intact"]
    graph = result["graphs"][0]
    assert graph["declared_event_wait_nodes"] == 1
    assert graph["declared_semaphore_wait_nodes"] == 1
    assert graph["accepted_host_submissions"] == 1
    assert result["host_submissions"][0]["region_id"] == 1
    assert result["graph_wait_execution_counts"] is None
    assert result["graph_wait_coverage_complete"] is False
    assert result["implicit_wait_coverage_complete"] is False
    assert result["zero_round_trip_assertion"] is None
    assert result["status"] == "INCOMPLETE"


@pytest.mark.parametrize("dynamic", ["device", "conditional", "host"])
def test_dynamic_execution_never_infers_node_multiplicity(dynamic: str) -> None:
    raw, activity = fixture()
    if dynamic == "device":
        for row in raw["records"]:
            if row["kind"] != 5:
                row["flags"] = 4
    else:
        raw["records"][3]["node_type"] = 13 if dynamic == "conditional" else 3
    result = summarize_graphs(raw, activity, REGIONS)
    assert result["source_inventory_intact"]
    assert result["graphs"][0]["dynamic_execution_unqualified"]
    assert result["graph_wait_execution_counts"] is None


@pytest.mark.parametrize("field", ["dropped_records", "errors", "outstanding_launches"])
def test_native_failure_prevents_intact_inventory(field: str) -> None:
    raw, activity = fixture()
    raw["metadata"][field] = 1
    assert not summarize_graphs(raw, activity, REGIONS)["source_inventory_intact"]


@pytest.mark.parametrize(
    "mutation",
    [
        "source-error",
        "activity-loss",
        "unknown-node",
        "missing-definition",
        "missing-api",
        "unfinished",
        "open-lifetime",
        "ancestry",
        "missing-phase",
        "extra-correlation",
        "status-disagreement",
    ],
)
def test_coverage_gaps_stay_visible(mutation: str) -> None:
    raw, activity = fixture()
    if mutation == "source-error":
        raw["source_dispatch_errors"] = 1
    elif mutation == "activity-loss":
        activity["metadata"]["dropped_records"] = 1
    elif mutation == "unknown-node":
        raw["records"][1]["node_type"] = 99
    elif mutation == "missing-definition":
        raw["records"].pop(0)
    elif mutation == "missing-api":
        activity["records"] = [row for row in activity["records"] if row["kind"] == 39]
    elif mutation == "unfinished":
        raw["records"].pop(5)
    elif mutation == "open-lifetime":
        raw["records"].pop()
    elif mutation == "ancestry":
        raw["records"][3]["parent_node"] = 99
    elif mutation == "missing-phase":
        activity["records"] = [
            row
            for row in activity["records"]
            if row["kind"] != 39 or row["subtype"] != 3
        ]
    elif mutation == "extra-correlation":
        activity["records"].append(
            activity_record(39, subtype=4, correlation_id=999, external_id=99)
        )
        activity["records"].append(
            activity_record(
                4, correlation_id=999, api_name="cuGraphLaunch", start=210, end=220
            )
        )
    else:
        activity["records"][0]["return_value"] = 1
    raw["metadata"]["record_count"] = len(raw["records"])
    activity["metadata"]["record_count"] = len(activity["records"])
    result = summarize_graphs(raw, activity, REGIONS)
    assert not result["source_inventory_intact"]
    assert result["status"] == "INCOMPLETE"


@pytest.mark.parametrize(
    "mutation,match",
    [
        ("generation", "duplicate graph generation"),
        ("launch", "duplicate/empty"),
        ("node", "duplicate graph node"),
        ("identity", "identity changed"),
        ("late", "after destruction"),
        ("correlation", "conflicting graph correlation"),
        ("fields", "fields"),
        ("bool", "uint64"),
        ("role", "role"),
        ("count", "count mismatch"),
        ("flags", "flags changed"),
    ],
)
def test_ambiguous_or_malformed_records_are_rejected(mutation: str, match: str) -> None:
    raw, activity = fixture()
    if mutation == "generation":
        raw["records"].insert(1, copy.deepcopy(raw["records"][0]))
    elif mutation == "launch":
        raw["records"].insert(5, copy.deepcopy(raw["records"][4]))
    elif mutation == "node":
        raw["records"].insert(2, copy.deepcopy(raw["records"][1]))
    elif mutation == "identity":
        raw["records"][4]["executable"] = 99
    elif mutation == "late":
        raw["records"].append(copy.deepcopy(raw["records"][1]))
    elif mutation == "correlation":
        activity["records"].append(
            activity_record(39, subtype=4, correlation_id=100, external_id=2)
        )
    elif mutation == "fields":
        raw["records"][0]["made_up"] = 1
    elif mutation == "bool":
        raw["records"][0]["flags"] = True
    elif mutation == "role":
        raw["records"][0]["role"] = 99
    elif mutation == "flags":
        raw["records"][4]["flags"] = 4
    raw["metadata"]["record_count"] = len(raw["records"]) - int(mutation == "count")
    activity["metadata"]["record_count"] = len(activity["records"])
    with pytest.raises(InvalidReceipt, match=match):
        summarize_graphs(raw, activity, REGIONS)


def test_source_generations_not_reused_handle_identities_own_lifetimes() -> None:
    raw, activity = fixture()
    second = copy.deepcopy(raw["records"])
    for row in second:
        row["generation"] = 2
        if row["launch_id"]:
            row["launch_id"] = 2
    raw["records"].extend(second)
    second_activity = copy.deepcopy(activity["records"])
    for row in second_activity:
        row["correlation_id"] += 10
        if row["kind"] == 39 and row["subtype"] == 4:
            row["external_id"] = 2
    activity["records"].extend(second_activity)
    raw["metadata"]["record_count"] = len(raw["records"])
    activity["metadata"]["record_count"] = len(activity["records"])
    activity["records"].reverse()
    result = summarize_graphs(raw, activity, REGIONS)
    assert result["source_inventory_intact"]
    assert [graph["generation"] for graph in result["graphs"]] == [1, 2]
    assert [submission["launch_id"] for submission in result["host_submissions"]] == [
        1,
        2,
    ]


def test_failed_submission_is_not_counted_as_accepted_work() -> None:
    raw, activity = fixture()
    raw["records"][5]["status"] = 1
    for row in activity["records"]:
        if row["kind"] in (4, 5):
            row["return_value"] = 1
    result = summarize_graphs(raw, activity, REGIONS)
    assert result["source_inventory_intact"]
    assert result["graphs"][0]["accepted_host_submissions"] == 0
