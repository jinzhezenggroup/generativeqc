"""Synthetic review-gate tests; fixture histories are not real-GPU evidence."""

from __future__ import annotations

import copy
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from test_cupti_source_capture import (
    REGIONS,
    activity_record,
    event_fixture,
    fixture,
    recount,
    source_record,
)

from tools import capture_prepared_residency
from tools.audit_replay_allocations import InvalidReceipt
from tools.capture_prepared_allocations import workload
from tools.cupti_source_capture import WORK_COUNTERS, summarize_sources
from tools.cupti_source_ratchet import (
    LEGACY_SCHEMA,
    SCHEMA,
    SCOPE,
    check_work_ratchet,
    load_work_ratchet,
    workload_digest,
)

CASE = {"method": "synthetic", "basis": "fixture", "precision": "fp64"}
ROOT = Path(__file__).resolve().parents[2]
QUALIFIED = ROOT / "manifests/residency_work_ratchets/hf_prepared_direct_fp64.v1.json"


def policy() -> dict[str, Any]:
    """Independent literal limits, not maxima learned from the checked history."""
    maximum = dict.fromkeys(WORK_COUNTERS, 0)
    maximum.update(
        source_transfer_calls=1,
        source_stream_fences=1,
        d2h_bytes=128,
        activity_sync_records=1,
    )
    return {
        "schema": SCHEMA,
        "scope": SCOPE,
        "profiles": [
            {
                "name": "synthetic-work-gate",
                "workload_sha256": workload_digest(CASE),
                "baseline": {
                    "source_tree": "1" * 40,
                    "library_sha256": "2" * 64,
                    "collector_sha256": "3" * 64,
                    "slurm_job_id": "synthetic-not-GPU-evidence",
                },
                "regions": [
                    {
                        "name": "energy-warm",
                        "role": "replay",
                        "owner_roles": [
                            {
                                "owner": "hf_bucket",
                                "role": "publication",
                                "max": maximum,
                            }
                        ],
                    }
                ],
                "unannotated_activity_limit": 0,
            }
        ],
    }


def selected(tmp_path: Path, value: dict[str, Any] | None = None) -> dict[str, Any]:
    path = tmp_path / "policy.json"
    path.write_text(json.dumps(value if value is not None else policy()))
    return load_work_ratchet(path, CASE)


def test_independent_observed_budget_is_never_a_residency_pass(tmp_path: Path) -> None:
    raw, activity = fixture()
    limits = selected(tmp_path)
    before = copy.deepcopy((raw, activity, limits, REGIONS))
    result = check_work_ratchet(limits, raw, activity, REGIONS, CASE)
    assert result["status"] == "WITHIN_OBSERVED_RATCHET"
    assert not result["issues"] and not result["violations"]
    assert result["residency_status"] == "INCOMPLETE"
    assert result["zero_round_trip_assertion"] is None
    assert not result["payload_dependency_links_complete"]
    assert not result["implicit_wait_coverage_complete"]
    assert not result["graph_wait_coverage_complete"]
    assert result["counts"][0]["counters"]["d2h_bytes"] == 128
    assert result["counts"][0]["counters"]["source_stream_fences"] == 1
    result["counts"][0]["max"]["d2h_bytes"] = 0
    assert (raw, activity, limits, REGIONS) == before


@pytest.mark.parametrize("legacy", [False, True])
def test_event_fences_require_separately_reviewed_limits(
    tmp_path: Path, legacy: bool
) -> None:
    raw, activity = event_fixture()
    contract = policy()
    maximum = contract["profiles"][0]["regions"][0]["owner_roles"][0]["max"]
    maximum["activity_event_wait_records"] = 1
    if legacy:
        contract["schema"] = LEGACY_SCHEMA
        maximum.pop("source_event_fences")
    path = tmp_path / "policy.json"
    path.write_text(json.dumps(contract))
    selected = load_work_ratchet(path, CASE)
    result = check_work_ratchet(selected, raw, activity, REGIONS, CASE)
    assert result["status"] == "VIOLATION"
    assert result["violations"] == [
        {
            "region": "energy-warm",
            "owner": "hf_bucket",
            "role": "publication",
            "counter": "source_event_fences",
            "observed": 1,
            "maximum": 0,
        }
    ]
    if not legacy:
        maximum["source_event_fences"] = 1
        path.write_text(json.dumps(contract))
        result = check_work_ratchet(
            load_work_ratchet(path, CASE), raw, activity, REGIONS, CASE
        )
        assert result["status"] == "WITHIN_OBSERVED_RATCHET"
        assert result["residency_status"] == "INCOMPLETE"


def test_legacy_policies_cannot_acquire_an_unreviewed_event_allowance(
    tmp_path: Path,
) -> None:
    contract = policy()
    contract["schema"] = LEGACY_SCHEMA
    path = tmp_path / "policy.json"
    path.write_text(json.dumps(contract))
    with pytest.raises(InvalidReceipt, match="work limits"):
        load_work_ratchet(path, CASE)


@pytest.mark.parametrize(
    "counter",
    [
        "source_transfer_calls",
        "source_stream_fences",
        "d2h_bytes",
        "activity_sync_records",
    ],
)
def test_counter_amplification_requires_review_without_learning_new_limits(
    tmp_path: Path, counter: str
) -> None:
    value = policy()
    value["profiles"][0]["regions"][0]["owner_roles"][0]["max"][counter] -= 1
    raw, activity = fixture()
    result = check_work_ratchet(selected(tmp_path, value), raw, activity, REGIONS, CASE)
    assert result["status"] == "VIOLATION"
    assert {row["counter"] for row in result["violations"]} == {counter}
    assert result["residency_status"] == "INCOMPLETE"
    assert result["zero_round_trip_assertion"] is None


@pytest.mark.parametrize(
    ("owner", "role", "label", "site"),
    [
        (1, 2, "iteration", 3),
        (1, 3, "tile", 3),
        (1, 5, "oracle", 3),
        (1, 6, "compatibility", 3),
        (2, 1, "prepare", 24),
        (5, 7, "lifetime", 27),
    ],
)
def test_unlisted_owner_role_cannot_borrow_publication_allowance(
    tmp_path: Path, owner: int, role: int, label: str, site: int
) -> None:
    raw, activity = fixture()
    raw["catalog"]["role"][str(role)] = label
    if owner != 1:
        raw["catalog"]["owner"][str(owner)] = (
            "hf_graph_setup" if owner == 2 else "device_resource_ledger"
        )
        raw["catalog"]["site"][str(site)] = (
            "hf.graph-upload-fence" if owner == 2 else "resource-ledger.release-fence"
        )
        raw["records"].insert(3, source_record(3, execution_id=4, owner=owner))
        raw["records"].insert(-1, source_record(4, execution_id=4, owner=owner))
    for row in raw["records"]:
        if row["operation_id"] == 3:
            row.update(
                owner=owner, role=role, site=site, execution_id=1 if owner == 1 else 4
            )
    recount(raw, activity)
    result = check_work_ratchet(selected(tmp_path), raw, activity, REGIONS, CASE)
    assert result["status"] == "VIOLATION"
    assert {row["counter"] for row in result["violations"]} == {
        "source_stream_fences",
        "activity_sync_records",
    }
    assert all(
        row["role"] == label and row["maximum"] == 0 for row in result["violations"]
    )


def test_zero_byte_calls_still_have_a_semantic_call_budget(tmp_path: Path) -> None:
    raw, activity = fixture()
    raw["records"][-1:-1] = [
        source_record(kind, operation_id=4, payload_instance=4, bytes=0)
        for kind in (1, 2)
    ]
    activity["records"].extend(
        [
            activity_record(
                5, correlation_id=104, api_name="cudaMemcpyAsync", start=300, end=310
            ),
            activity_record(39, subtype=3, correlation_id=104, external_id=1),
            activity_record(39, subtype=5, correlation_id=104, external_id=4),
        ]
    )
    recount(raw, activity)
    result = check_work_ratchet(selected(tmp_path), raw, activity, REGIONS, CASE)
    assert result["status"] == "VIOLATION"
    assert {row["counter"] for row in result["violations"]} == {
        "source_transfer_calls",
        "source_zero_byte_calls",
    }
    assert result["counts"][0]["counters"]["d2h_bytes"] == 128


@pytest.mark.parametrize("phase", [1, 8])
def test_unknown_blocking_work_is_rejected_only_in_selected_production_scope(
    tmp_path: Path, phase: int
) -> None:
    raw, activity = fixture()
    activity["records"].extend(
        [
            activity_record(38, subtype=4, correlation_id=104, start=300, end=310),
            activity_record(39, subtype=3, correlation_id=104, external_id=phase),
        ]
    )
    recount(raw, activity)
    result = check_work_ratchet(selected(tmp_path), raw, activity, REGIONS, CASE)
    assert result["status"] == (
        "VIOLATION" if phase == 1 else "WITHIN_OBSERVED_RATCHET"
    )
    assert len(result["unannotated_selected_blocking_activity"]) == int(phase == 1)


def test_preparation_work_outside_selected_replay_stays_legal(tmp_path: Path) -> None:
    raw, activity = fixture()
    activity["records"].extend(
        [
            activity_record(38, subtype=4, correlation_id=104, start=300, end=310),
            activity_record(39, subtype=3, correlation_id=104, external_id=2),
        ]
    )
    recount(raw, activity)
    regions = {**REGIONS, 2: {"name": "prepare", "role": "prepare"}}
    result = check_work_ratchet(selected(tmp_path), raw, activity, regions, CASE)
    assert result["status"] == "WITHIN_OBSERVED_RATCHET"
    assert result["residency_status"] == "INCOMPLETE"


@pytest.mark.parametrize("role", [5, 6])
def test_explicit_oracle_compatibility_limits_remain_legal_and_labeled(
    tmp_path: Path, role: int
) -> None:
    raw, activity = fixture()
    label = raw["catalog"]["role"][str(role)]
    for row in raw["records"]:
        if row["operation_id"] == 3:
            row["role"] = role
    value = policy()
    maximum = dict.fromkeys(WORK_COUNTERS, 0)
    maximum.update(source_stream_fences=1, activity_sync_records=1)
    value["profiles"][0]["regions"][0]["owner_roles"].append(
        {"owner": "hf_bucket", "role": label, "max": maximum}
    )
    result = check_work_ratchet(selected(tmp_path, value), raw, activity, REGIONS, CASE)
    assert result["status"] == "WITHIN_OBSERVED_RATCHET"
    assert {row["role"] for row in result["counts"]} == {"publication", label}


@pytest.mark.parametrize("status", [0, 600])
def test_nonblocking_queries_remain_separate_not_invented_fences(
    tmp_path: Path, status: int
) -> None:
    raw, activity = fixture()
    activity["records"].extend(
        [
            activity_record(
                5,
                correlation_id=104,
                api_name="cudaStreamQuery",
                return_value=status,
                start=300,
                end=310,
            ),
            activity_record(
                38,
                subtype=3,
                correlation_id=104,
                return_value=status,
                start=300,
                end=305,
            ),
            activity_record(39, subtype=3, correlation_id=104, external_id=1),
        ]
    )
    recount(raw, activity)
    result = check_work_ratchet(selected(tmp_path), raw, activity, REGIONS, CASE)
    assert result["status"] == "WITHIN_OBSERVED_RATCHET"
    assert len(result["unannotated_selected_nonblocking_queries"]) == 1
    assert result["counts"][0]["counters"]["activity_sync_records"] == 1


@pytest.mark.parametrize(
    "field", ["dropped_records", "errors", "outstanding_operations"]
)
def test_source_loss_cannot_authorize_within_budget(tmp_path: Path, field: str) -> None:
    raw, activity = fixture()
    raw["metadata"][field] = 1
    result = check_work_ratchet(selected(tmp_path), raw, activity, REGIONS, CASE)
    assert result["status"] == "INCOMPLETE"
    assert result["issues"]


def test_missing_observed_replay_is_not_a_zero_work_assertion(tmp_path: Path) -> None:
    raw, activity = fixture()
    for row in activity["records"]:
        if row["kind"] == 39 and row["subtype"] == 3:
            row["external_id"] = 2
    regions = {**REGIONS, 2: {"name": "prepare", "role": "prepare"}}
    result = check_work_ratchet(selected(tmp_path), raw, activity, regions, CASE)
    assert result["status"] == "INCOMPLETE"
    assert result["issues"] == [
        "no observed API provenance for selected region energy-warm"
    ]


@pytest.mark.parametrize(
    "mutation",
    [
        "selector",
        "duplicate-region",
        "duplicate-owner",
        "missing-counter",
        "boolean",
        "overflow",
        "negative",
        "unknown-exemption",
        "nonhot",
        "unknown-role",
        "scope",
        "baseline",
    ],
)
def test_invalid_policies_fail_closed(tmp_path: Path, mutation: str) -> None:
    value = policy()
    profile = value["profiles"][0]
    region = profile["regions"][0]
    limits = region["owner_roles"][0]["max"]
    if mutation == "selector":
        value["profiles"].append(copy.deepcopy(profile))
    elif mutation == "duplicate-region":
        profile["regions"].append(copy.deepcopy(region))
    elif mutation == "duplicate-owner":
        region["owner_roles"].append(copy.deepcopy(region["owner_roles"][0]))
    elif mutation == "missing-counter":
        limits.pop("h2d_bytes")
    elif mutation in {"boolean", "overflow", "negative"}:
        limits["h2d_bytes"] = {"boolean": True, "overflow": 1 << 64, "negative": -1}[
            mutation
        ]
    elif mutation == "unknown-exemption":
        profile["unannotated_activity_limit"] = 1
    elif mutation == "nonhot":
        region["role"] = "publication"
    elif mutation == "unknown-role":
        region["owner_roles"][0]["role"] = "file-path-exemption"
    elif mutation == "scope":
        value["scope"] = "complete CUDA residency"
    else:
        profile["baseline"]["source_tree"] = "stale-unknown-source"
    with pytest.raises(InvalidReceipt):
        selected(tmp_path, value)


@pytest.mark.parametrize(
    "contents",
    [
        '{"schema":1,"schema":2}',
        '{"value":NaN}',
        '{"value":1e999}',
        "[" * 2000 + "]" * 2000,
        " " * ((1 << 20) + 1),
    ],
)
def test_policy_decode_rejects_duplicate_nonfinite_nested_or_oversized_json(
    tmp_path: Path, contents: str
) -> None:
    path = tmp_path / "policy.json"
    path.write_text(contents)
    with pytest.raises(InvalidReceipt):
        load_work_ratchet(path, CASE)


def test_changed_workload_and_changed_policy_do_not_reuse_an_independent_pin(
    tmp_path: Path,
) -> None:
    pinned = selected(tmp_path)
    with pytest.raises(InvalidReceipt, match="no ratchet profile"):
        load_work_ratchet(tmp_path / "policy.json", {**CASE, "precision": "fp32"})
    raw, activity = fixture()
    with pytest.raises(InvalidReceipt, match="workload mismatch"):
        check_work_ratchet(
            pinned, raw, activity, REGIONS, {**CASE, "precision": "fp32"}
        )
    value = policy()
    value["profiles"][0]["regions"][0]["owner_roles"][0]["max"]["d2h_bytes"] += 1
    assert selected(tmp_path, value)["sha256"] != pinned["sha256"]


@pytest.mark.parametrize("method", ["rhf", "uhf"])
def test_retained_profiles_match_only_their_exact_qualified_workload(
    method: str,
) -> None:
    profile = load_work_ratchet(QUALIFIED, workload(method))["profile"]
    assert profile["name"] == f"prepared-{method}-direct-fp64-h2-water-h2-v1"
    assert [row["name"] for row in profile["regions"]] == ["energy-warm", "force-warm"]
    assert profile["baseline"]["slurm_job_id"] == "6933"


def test_phase_owner_role_is_an_alternate_projection_not_extra_work() -> None:
    raw, activity = fixture()
    summary = summarize_sources(raw, activity, REGIONS)
    assert summary["by_phase_owner_role"] == [
        {
            "region_id": 1,
            "owner": "hf_bucket",
            "role": "publication",
            "counters": summary["by_owner"]["hf_bucket"],
        }
    ]


def test_capture_contract_pins_independent_profile_and_policy_changes_without_gpu(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Only metadata/assignment doubles run here; no CUDA/CUPTI library is loaded."""
    monkeypatch.setenv("SLURM_JOB_ID", "cpu-metadata-double")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    monkeypatch.setattr(
        capture_prepared_residency.subprocess,
        "check_output",
        lambda *args, **kwargs: "GPU-cpu-metadata-double\n",
    )
    monkeypatch.setattr(capture_prepared_residency, "verify_source", lambda *args: None)
    required = [
        "tools/cupti_residency_capture.cpp",
        "tools/cupti_residency_capture.py",
        "tools/capture_prepared_residency.py",
        "tools/cupti_graph_inventory.cpp",
        "tools/cupti_graph_inventory.py",
        "tools/cupti_source_capture.cpp",
        "tools/cupti_source_capture.py",
        "tools/cupti_source_ratchet.py",
        "src/runtime/residency_observer.hpp",
        "src/runtime/residency_boundaries.hpp",
        "src/runtime/residency_cuda.cuh",
        "src/runtime/resource_cuda.cuh",
        "src/scf/cuda_rhf.cpp",
        "src/scf/cuda/rhf_graph.cpp",
        "src/scf/cuda/rhf_graph.hpp",
        "src/scf/cuda/resources.cpp",
        "src/scf/cuda/eigensolver.cpp",
        "src/api/c_api_resources.cpp",
    ]
    manifest = tmp_path / "source.json"
    manifest.write_text(
        json.dumps(
            {
                "source_commit": "4" * 40,
                "source_tree": "5" * 40,
                "entries": [{"path": path} for path in required],
            }
        )
    )
    binary = tmp_path / "not-a-library"
    binary.write_bytes(b"metadata-double-not-executable")
    limits = tmp_path / "independent-policy.json"
    limits.write_bytes(QUALIFIED.read_bytes())
    arguments = SimpleNamespace(
        source_manifest=manifest,
        library=binary,
        collector=binary,
        cupti_library=binary,
        work_ratchet=limits,
        method="rhf",
        toolchain="CPU metadata test, not GPU proof",
        capacity=64,
        graph_node_limit=64,
        graph_depth_limit=4,
    )
    pinned = capture_prepared_residency.contract(arguments)
    assert pinned["observed_work_ratchet"] == load_work_ratchet(limits, workload("rhf"))
    value = json.loads(limits.read_text())
    value["profiles"][0]["regions"][0]["owner_roles"][0]["max"]["h2d_bytes"] += 1
    limits.write_text(json.dumps(value))
    assert capture_prepared_residency.contract(arguments) != pinned
