"""Synthetic source/activity joins, never real-GPU residency qualification."""

from __future__ import annotations

import copy
from typing import Any

import pytest

from tools.audit_replay_allocations import InvalidReceipt
from tools.cupti_residency_capture import FIELDS as ACTIVITY_FIELDS
from tools.cupti_residency_capture import METADATA as ACTIVITY_METADATA
from tools.cupti_source_capture import FIELDS, METADATA, summarize_sources

REGIONS = {
    1: {"name": "energy-warm", "role": "replay"},
    8: {"name": "fence", "role": "observer"},
}


def source_record(kind: int, **values: Any) -> dict[str, Any]:
    row = dict.fromkeys(FIELDS, 0)
    row.update(version=2, kind=kind, execution_id=1, owner=1)
    if kind in (1, 2):
        row.update(
            operation_id=2,
            role=4,
            site=1,
            payload=4,
            payload_instance=2,
            operation_kind=1,
            direction=2,
            bytes=128,
        )
    row.update(values)
    return row


def activity_record(kind: int, **values: Any) -> dict[str, Any]:
    row = dict.fromkeys(ACTIVITY_FIELDS, 0)
    row.update(kind=kind, api_name=None)
    row.update(values)
    return row


def fixture() -> tuple[dict[str, Any], dict[str, Any]]:
    """A named source publication plus one existing stream fence."""
    raw = {
        "schema": "generativeqc.cupti-source-boundaries.v1",
        "metadata": dict.fromkeys(METADATA, 0),
        "source_dispatch_errors": 0,
        "catalog": {
            "owner": {"1": "hf_bucket"},
            "role": {
                "4": "publication",
                "2": "iteration",
                "3": "tile",
                "5": "oracle",
                "6": "compatibility",
                "7": "lifetime",
            },
            "site": {"1": "hf.results", "3": "hf.results-fence"},
            "payload": {"0": "none", "4": "density"},
        },
        "records": [
            source_record(3),
            source_record(1),
            source_record(2),
            source_record(
                1,
                operation_id=3,
                site=3,
                payload=0,
                payload_instance=0,
                operation_kind=2,
                direction=0,
                bytes=0,
            ),
            source_record(
                2,
                operation_id=3,
                site=3,
                payload=0,
                payload_instance=0,
                operation_kind=2,
                direction=0,
                bytes=0,
            ),
            source_record(4),
        ],
    }
    raw["metadata"].update(
        record_count=len(raw["records"]),
        capacity=64,
        stopped=1,
        cupti_version=28,
        source_version=2,
    )
    records = [
        activity_record(
            5, correlation_id=100, api_name="cudaMemcpyAsync_v3020", start=90, end=100
        ),
        activity_record(
            4, correlation_id=101, api_name="cuMemcpyDtoHAsync_v2", start=91, end=99
        ),
        activity_record(
            1,
            subtype=2,
            bytes=128,
            correlation_id=101,
            runtime_correlation_id=100,
            start=100,
            end=120,
            copy_count=1,
        ),
        activity_record(
            5,
            correlation_id=102,
            api_name="cudaStreamSynchronize_v3020",
            start=200,
            end=210,
        ),
        activity_record(38, subtype=3, correlation_id=102, start=200, end=205),
        *[
            activity_record(39, subtype=3, correlation_id=identity, external_id=1)
            for identity in (100, 101, 102)
        ],
        *[
            activity_record(
                39,
                subtype=5,
                correlation_id=identity,
                external_id=2 if identity < 102 else 3,
            )
            for identity in (100, 101, 102)
        ],
    ]
    metadata = dict.fromkeys(ACTIVITY_METADATA, 0)
    metadata.update(record_count=len(records), capacity=64, stopped=1, cupti_version=28)
    activity = {
        "schema": "generativeqc.cupti-activity.v1",
        "metadata": metadata,
        "records": records,
    }
    return raw, activity


def recount(raw: dict[str, Any], activity: dict[str, Any]) -> None:
    raw["metadata"]["record_count"] = len(raw["records"])
    activity["metadata"]["record_count"] = len(activity["records"])


def event_fixture() -> tuple[dict[str, Any], dict[str, Any]]:
    """Replace only the fixture's existing stream API with an event API."""
    raw, activity = fixture()
    for row in raw["records"]:
        if row["operation_id"] == 3:
            row["operation_kind"] = 3
    for row in activity["records"]:
        if row["correlation_id"] == 102 and row["kind"] == 5:
            row["api_name"] = "cudaEventSynchronize_v3020"
        if row["correlation_id"] == 102 and row["kind"] == 38:
            row["subtype"] = 1
    return raw, activity


def test_event_fences_are_not_stream_fences_or_scientific_links() -> None:
    raw, activity = event_fixture()
    raw["catalog"]["owner"]["6"] = "posthf_df_source"
    raw["catalog"]["site"]["31"] = "posthf.df.publication-fence"
    for row in raw["records"]:
        row["owner"] = 6
        if row["operation_id"]:
            row["role"] = 6
        if row["operation_id"] == 3:
            row["site"] = 31
    result = summarize_sources(raw, activity, REGIONS)
    assert result["source_annotations_intact"], result["issues"]
    counters = result["by_owner"]["posthf_df_source"]
    assert counters["source_event_fences"] == 1
    assert counters["source_stream_fences"] == 0
    assert counters["activity_event_wait_records"] == 1
    assert counters["d2h_bytes"] == 128
    assert result["operations"][1]["kind"] == "event_sync"
    assert result["operations"][1]["events"][0]["sync_subtype"] == 1
    assert result["status"] == "INCOMPLETE"
    assert all(row["dependency_identity"] is None for row in result["operations"])


@pytest.mark.parametrize("subtype", [2, 3, 4])
def test_event_fence_rejects_device_side_wait_or_other_sync_activity(
    subtype: int,
) -> None:
    raw, activity = event_fixture()
    for row in activity["records"]:
        if row["kind"] == 38:
            row["subtype"] = subtype
    result = summarize_sources(raw, activity, REGIONS)
    assert not result["source_annotations_intact"]
    assert any("synchronization disagreement" in issue for issue in result["issues"])


@pytest.mark.parametrize(
    "api", ["cudaStreamSynchronize", "cudaDeviceSynchronize", "cudaEventQuery"]
)
def test_event_fence_requires_its_actual_blocking_runtime_api(api: str) -> None:
    raw, activity = event_fixture()
    for row in activity["records"]:
        if row["kind"] == 5 and row["correlation_id"] == 102:
            row["api_name"] = api + "_v3020"
    result = summarize_sources(raw, activity, REGIONS)
    assert not result["source_annotations_intact"]
    assert result["by_role"]["publication"]["source_event_fences"] == 0


def test_real_work_is_counted_once_and_named_from_source_not_public_phase() -> None:
    raw, activity = fixture()
    result = summarize_sources(raw, activity, REGIONS)
    assert result["source_annotations_intact"]
    assert result["observed_production_activity_fully_annotated"]
    assert result["by_role"]["publication"]["d2h_bytes"] == 128
    assert result["by_role"]["publication"]["source_transfer_calls"] == 1
    assert result["by_role"]["publication"]["source_stream_fences"] == 1
    assert result["by_role"]["publication"]["activity_sync_records"] == 1
    assert result["by_role"]["iteration"]["d2h_bytes"] == 0
    operation = result["operations"][0]
    assert operation["payload"] == "density" and operation["payload_instance"] == 2
    assert operation["owner"] == "hf_bucket" and operation["execution_id"] == 1
    assert operation["role"] == "publication" and operation["region_id"] == 1
    assert operation["source_call_verified"]
    assert result["status"] == "INCOMPLETE"
    assert result["zero_round_trip_assertion"] is None
    assert result["payload_dependency_links_complete"] is False
    assert result["graph_wait_coverage_complete"] is False
    assert result["implicit_wait_coverage_complete"] is False


@pytest.mark.parametrize("role", [2, 3, 7])
def test_iteration_tile_and_lifetime_boundaries_are_higher_risk_than_publication(
    role: int,
) -> None:
    raw, activity = fixture()
    baseline = summarize_sources(raw, activity, REGIONS)
    for row in raw["records"]:
        if row["kind"] in (1, 2):
            row["role"] = role
    result = summarize_sources(raw, activity, REGIONS)
    assert result["source_annotations_intact"]
    assert all(row["risk"] == "higher" for row in result["operations"])
    assert all(row["risk"] == "explicit_nonhot_role" for row in baseline["operations"])
    selected = {2: "iteration", 3: "tile", 7: "lifetime"}[role]
    assert result["by_role"][selected]["d2h_bytes"] == 128
    assert result["by_role"][selected]["source_stream_fences"] == 1


@pytest.mark.parametrize("role", [5, 6])
def test_explicit_oracle_and_compatibility_roles_are_legal_not_inferred(
    role: int,
) -> None:
    raw, activity = fixture()
    for row in raw["records"]:
        if row["kind"] in (1, 2):
            row["role"] = role
    result = summarize_sources(raw, activity, REGIONS)
    assert result["source_annotations_intact"]
    assert (
        result["by_role"]["oracle" if role == 5 else "compatibility"]["d2h_bytes"]
        == 128
    )


def test_activity_delivery_order_is_not_source_attribution() -> None:
    raw, activity = fixture()
    before = summarize_sources(raw, activity, REGIONS)
    activity["records"].reverse()
    after = summarize_sources(raw, activity, REGIONS)
    assert before["by_role"] == after["by_role"]
    assert after["source_annotations_intact"]


def test_zero_byte_calls_are_preserved_without_inventing_a_transfer() -> None:
    raw, activity = fixture()
    raw["records"][1]["bytes"] = raw["records"][2]["bytes"] = 0
    activity["records"] = [row for row in activity["records"] if row["kind"] != 1]
    recount(raw, activity)
    result = summarize_sources(raw, activity, REGIONS)
    assert result["source_annotations_intact"]
    assert result["by_role"]["publication"]["d2h_bytes"] == 0
    assert result["by_role"]["publication"]["source_zero_byte_calls"] == 1
    assert result["by_role"]["publication"]["source_transfer_calls"] == 1


def test_queries_never_become_verified_source_stream_fences() -> None:
    raw, activity = fixture()
    activity["records"][3]["api_name"] = "cudaStreamQuery_v3020"
    result = summarize_sources(raw, activity, REGIONS)
    assert not result["source_annotations_intact"]
    assert result["by_role"]["publication"]["source_stream_fences"] == 0
    assert result["by_role"]["publication"]["activity_sync_records"] == 0


def test_fence_api_calls_and_activity_records_are_separate_metrics() -> None:
    raw, activity = fixture()
    activity["records"] = [row for row in activity["records"] if row["kind"] != 38]
    recount(raw, activity)
    result = summarize_sources(raw, activity, REGIONS)
    assert result["source_annotations_intact"]
    assert result["by_role"]["publication"]["source_stream_fences"] == 1
    assert result["by_role"]["publication"]["activity_sync_records"] == 0
    assert result["implicit_wait_coverage_complete"] is False


@pytest.mark.parametrize(
    "mutation",
    [
        "source-loss",
        "source-error",
        "dispatch-error",
        "pending",
        "activity-loss",
        "missing-runtime",
        "missing-copy",
        "byte-mismatch",
        "copy-count",
        "status-mismatch",
        "open-execution",
        "unknown-role",
        "unknown-owner",
        "unknown-payload",
        "dependency",
        "source-error-record",
        "extra-correlation",
        "orphan-correlation",
        "unfinished-api",
    ],
)
def test_loss_unknowns_or_incomplete_joins_never_promote_to_pass(mutation: str) -> None:
    raw, activity = fixture()
    if mutation == "source-loss":
        raw["metadata"]["dropped_records"] = 1
    elif mutation == "source-error":
        raw["metadata"]["errors"] = 1
    elif mutation == "dispatch-error":
        raw["source_dispatch_errors"] = 1
    elif mutation == "pending":
        raw["metadata"]["outstanding_operations"] = 1
    elif mutation == "activity-loss":
        activity["metadata"]["dropped_records"] = 1
    elif mutation == "missing-runtime":
        activity["records"][0]["api_name"] = None
    elif mutation == "missing-copy":
        activity["records"] = [row for row in activity["records"] if row["kind"] != 1]
    elif mutation == "byte-mismatch":
        activity["records"][2]["bytes"] = 64
    elif mutation == "copy-count":
        activity["records"][2]["copy_count"] = 2
    elif mutation == "status-mismatch":
        activity["records"][0]["return_value"] = 1
    elif mutation == "open-execution":
        raw["records"].pop()
    elif mutation == "unknown-role":
        raw["catalog"]["role"]["4"] = None
    elif mutation == "unknown-owner":
        raw["catalog"]["owner"]["1"] = None
    elif mutation == "unknown-payload":
        raw["catalog"]["payload"]["4"] = None
    elif mutation == "dependency":
        raw["records"][1]["dependency"] = raw["records"][2]["dependency"] = 99
    elif mutation == "source-error-record":
        raw["records"].append(source_record(7, status=4))
        raw["metadata"]["errors"] = 1
    elif mutation == "extra-correlation":
        activity["records"].append(
            activity_record(39, subtype=5, correlation_id=999, external_id=99)
        )
        activity["records"].append(
            activity_record(
                5, correlation_id=999, api_name="cudaMemcpyAsync", start=300, end=310
            )
        )
    elif mutation == "orphan-correlation":
        activity["records"].append(
            activity_record(39, subtype=5, correlation_id=999, external_id=99)
        )
    else:
        activity["records"][0]["start"] = 0
    recount(raw, activity)
    result = summarize_sources(raw, activity, REGIONS)
    assert not result["source_annotations_intact"]
    assert not result["observed_production_activity_fully_annotated"]
    assert (
        result["status"] == "INCOMPLETE" and result["zero_round_trip_assertion"] is None
    )
    if mutation == "unknown-role":
        assert result["operations"][0]["risk"] == "unknown"


@pytest.mark.parametrize(
    "mutation,match",
    [
        ("fields", "fields"),
        ("bool", "uint64"),
        ("root-duplicate", "duplicate source execution"),
        ("op-duplicate", "duplicate source operation"),
        ("payload-mutation", "identity changed"),
        ("owner-mutation", "owner changed"),
        ("closed", "missing/closed"),
        ("end", "unmatched"),
        ("correlation", "conflicting source correlation"),
        ("catalog", "tag ID"),
        ("version", "version"),
        ("direction", "payload/direction"),
    ],
)
def test_malformed_or_conflicting_source_identity_is_rejected(
    mutation: str, match: str
) -> None:
    raw, activity = fixture()
    if mutation == "fields":
        raw["records"][0]["pointer"] = 123
    elif mutation == "bool":
        raw["records"][0]["execution_id"] = True
    elif mutation == "root-duplicate":
        raw["records"].insert(1, copy.deepcopy(raw["records"][0]))
    elif mutation == "op-duplicate":
        raw["records"].insert(2, copy.deepcopy(raw["records"][1]))
    elif mutation == "payload-mutation":
        raw["records"][2]["payload_instance"] = 99
    elif mutation == "owner-mutation":
        raw["records"][1]["owner"] = 99
    elif mutation == "closed":
        raw["records"].insert(1, copy.deepcopy(raw["records"][-1]))
    elif mutation == "end":
        raw["records"].pop(1)
    elif mutation == "correlation":
        activity["records"].append(
            activity_record(39, subtype=5, correlation_id=100, external_id=99)
        )
    elif mutation == "catalog":
        raw["catalog"]["role"]["04"] = "publication"
    elif mutation == "version":
        raw["records"][0]["version"] = 3
    else:
        raw["records"][1]["direction"] = 0
    recount(raw, activity)
    with pytest.raises(InvalidReceipt, match=match):
        summarize_sources(raw, activity, REGIONS)


def test_unannotated_work_stays_visible_and_observer_fences_remain_separate() -> None:
    raw, activity = fixture()
    activity["records"].extend(
        [
            activity_record(
                1,
                subtype=1,
                bytes=16,
                correlation_id=105,
                start=300,
                end=310,
                copy_count=1,
            ),
            activity_record(39, subtype=3, correlation_id=105, external_id=1),
            activity_record(38, subtype=4, correlation_id=106, start=320, end=330),
            activity_record(39, subtype=3, correlation_id=106, external_id=8),
        ]
    )
    recount(raw, activity)
    result = summarize_sources(raw, activity, REGIONS)
    assert result["source_annotations_intact"]
    assert not result["observed_production_activity_fully_annotated"]
    assert len(result["unannotated_production_activity"]) == 1
    assert result["unannotated_production_activity"][0]["bytes"] == 16
    assert result["by_role"]["publication"]["activity_sync_records"] == 1


def test_same_size_upload_and_download_do_not_create_a_dependency_link() -> None:
    raw, activity = fixture()
    upload_begin = source_record(1, operation_id=4, payload_instance=4, direction=1)
    upload_end = source_record(2, operation_id=4, payload_instance=4, direction=1)
    raw["records"][-1:-1] = [upload_begin, upload_end]
    activity["records"].extend(
        [
            activity_record(
                5, correlation_id=104, api_name="cudaMemcpyAsync", start=250, end=260
            ),
            activity_record(
                1,
                subtype=1,
                bytes=128,
                correlation_id=104,
                start=260,
                end=270,
                copy_count=1,
            ),
            activity_record(39, subtype=3, correlation_id=104, external_id=1),
            activity_record(39, subtype=5, correlation_id=104, external_id=4),
        ]
    )
    recount(raw, activity)
    result = summarize_sources(raw, activity, REGIONS)
    assert result["source_annotations_intact"]
    assert result["by_role"]["publication"]["h2d_bytes"] == 128
    assert result["by_role"]["publication"]["d2h_bytes"] == 128
    assert result["payload_dependency_links_complete"] is False
    assert all(row["dependency_identity"] is None for row in result["operations"])


def test_preparation_input_role_is_preserved_inside_a_public_warm_execute() -> None:
    """Source declarations, not enclosing replay names, identify input uploads."""
    raw, activity = fixture()
    raw["catalog"]["role"]["1"] = "prepare"
    raw["catalog"]["site"]["19"] = "hf.dynamic-inputs"
    raw["catalog"]["payload"]["75"] = "input.warm_density"
    for row in raw["records"]:
        if row["operation_id"] == 2:
            row.update(role=1, site=19, payload=75, direction=1)
    for row in activity["records"]:
        if row["kind"] == 4:
            row["api_name"] = "cuMemcpyHtoDAsync_v2"
        elif row["kind"] == 1:
            row["subtype"] = 1
    result = summarize_sources(raw, activity, REGIONS)
    assert result["source_annotations_intact"]
    assert result["by_role"]["prepare"]["source_transfer_calls"] == 1
    assert result["by_role"]["prepare"]["h2d_bytes"] == 128
    assert result["by_role"]["publication"]["d2h_bytes"] == 0
    operation = result["operations"][0]
    assert operation["region_id"] == 1
    assert operation["role"] == "prepare"
    assert operation["payload"] == "input.warm_density"
    assert operation["dependency_identity"] is None
    assert result["payload_dependency_links_complete"] is False
    assert result["zero_round_trip_assertion"] is None


@pytest.mark.parametrize(
    ("owner", "role", "site", "owner_name", "role_name", "site_name", "risk"),
    [
        (
            2,
            1,
            24,
            "hf_graph_setup",
            "prepare",
            "hf.graph-upload-fence",
            "explicit_nonhot_role",
        ),
        (
            4,
            7,
            26,
            "hf_eigensolver_resources",
            "lifetime",
            "hf.eigensolver-release-fence",
            "higher",
        ),
        (
            5,
            7,
            27,
            "device_resource_ledger",
            "lifetime",
            "resource-ledger.release-fence",
            "higher",
        ),
        (
            5,
            7,
            28,
            "device_resource_ledger",
            "lifetime",
            "resource-ledger.rollback-fence",
            "higher",
        ),
    ],
)
def test_nested_source_owners_do_not_inherit_bucket_publication_role(
    owner: int,
    role: int,
    site: int,
    owner_name: str,
    role_name: str,
    site_name: str,
    risk: str,
) -> None:
    """Owner and role projections are alternate views, not additive work totals."""
    raw, activity = fixture()
    raw["catalog"]["owner"][str(owner)] = owner_name
    raw["catalog"]["role"][str(role)] = role_name
    raw["catalog"]["site"][str(site)] = site_name
    for row in raw["records"]:
        if row["operation_id"] == 3:
            row.update(execution_id=4, owner=owner, role=role, site=site)
    raw["records"].insert(3, source_record(3, execution_id=4, owner=owner))
    raw["records"].insert(-1, source_record(4, execution_id=4, owner=owner))
    recount(raw, activity)
    result = summarize_sources(raw, activity, REGIONS)
    assert result["source_annotations_intact"]
    assert all(row["closed"] for row in result["executions"])
    assert result["by_owner"]["hf_bucket"]["d2h_bytes"] == 128
    assert result["by_owner"]["hf_bucket"]["source_stream_fences"] == 0
    assert result["by_owner"][owner_name]["source_stream_fences"] == 1
    assert result["by_owner"][owner_name]["d2h_bytes"] == 0
    assert result["by_role"][role_name]["source_stream_fences"] == 1
    assert result["by_role"]["publication"]["source_stream_fences"] == 0
    assert result["operations"][1]["risk"] == risk
    assert result["payload_dependency_links_complete"] is False
    assert result["zero_round_trip_assertion"] is None


@pytest.mark.parametrize("api_status", [1, 2])
def test_source_status_matches_the_exact_runtime_error_not_just_failure_boolean(
    api_status: int,
) -> None:
    raw, activity = fixture()
    raw["records"][4]["status"] = 1
    next(
        row
        for row in activity["records"]
        if row["kind"] == 5 and row["correlation_id"] == 102
    )["return_value"] = api_status
    result = summarize_sources(raw, activity, REGIONS)
    assert result["source_annotations_intact"] == (api_status == 1)
    assert result["operations"][1]["source_call_verified"] == (api_status == 1)
    assert result["by_role"]["publication"]["source_stream_fences"] == 0
    assert result["status"] == "INCOMPLETE"
    assert result["zero_round_trip_assertion"] is None
