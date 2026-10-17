"""Independent event oracles for the RHF provider-retention diagnostic."""

import hashlib
import json
from copy import deepcopy
from pathlib import Path

import pytest

import benchmarks.rhf_provider_retention_report as report_module
from benchmarks.rhf_provider_retention_report import summarize

IDENTITY = {
    "source_commit": "a" * 40,
    "library_sha256": "b" * 64,
    "endpoint_sha256": "c" * 64,
    "shim_sha256": "d" * 64,
    "event_log_sha256": "e" * 64,
    "device_uuid": "GPU-test",
    "job_id": "probe-1",
}
POLICY = {
    "source_commit": "a" * 40,
    "providers": {
        "cublas": {
            "allowance_bytes": 100,
            "required_events": [
                "cublas_create",
                "cublas_set_stream",
                "cublas_set_pointer_mode",
            ],
        },
        "cusolver": {
            "allowance_bytes": 150,
            "required_events": [
                "cusolver_create",
                "cusolver_set_stream",
                "cusolver_create_params",
            ],
        },
    },
}


def event(sequence: int, name: str, **values: object) -> dict:
    return {
        "schema": "generativeqc.rhf-provider-probe.v1",
        "sequence": sequence,
        "pid": 10,
        "tid": 20,
        "event": name,
        "phase": "end",
        "status": 0,
        **values,
    }


def ordered_records(*, before: int = 1000, after: int = 700) -> list[dict]:
    records = [
        event(
            0,
            "cuda_mem_get_info",
            origin="application",
            observed=True,
            free_bytes=before,
            total_bytes=2000,
        ),
        event(1, "cublas_create", phase="begin"),
        event(
            2,
            "cuda_mem_get_info",
            origin="provider_internal",
            observed=True,
            free_bytes=990,
            total_bytes=2000,
        ),
        event(3, "cublas_create"),
        event(4, "cublas_set_stream", phase="begin"),
        event(5, "cublas_set_stream"),
        event(6, "cublas_set_pointer_mode", phase="begin"),
        event(7, "cublas_set_pointer_mode"),
        event(8, "cusolver_create", phase="begin"),
        event(9, "cusolver_create"),
        event(10, "cusolver_set_stream", phase="begin"),
        event(11, "cusolver_set_stream"),
        event(12, "cusolver_create_params", phase="begin"),
        event(13, "cusolver_create_params"),
        event(
            14,
            "cuda_mem_get_info",
            origin="application",
            observed=True,
            free_bytes=after,
            total_bytes=2000,
        ),
    ]
    records.append(
        event(
            15,
            "terminal",
            events_written=15,
            dropped_events=0,
            resolution_errors=0,
        )
    )
    return records


def resequence(records: list[dict]) -> list[dict]:
    result = deepcopy(records)
    terminal = next(row for row in result if row["event"] == "terminal")
    ordinary = [row for row in result if row["event"] != "terminal"]
    for sequence, row in enumerate(ordinary):
        row["sequence"] = sequence
    terminal["sequence"] = len(ordinary)
    terminal["events_written"] = len(ordinary)
    return [*ordinary, terminal]


def test_ordered_manual_oracle_separates_vendor_queries_and_allowance() -> None:
    result = summarize(ordered_records(), POLICY, IDENTITY)
    assert result["status"] == "COMPLETE"
    assert result["measurement"] == {
        "pid": 10,
        "tid": 20,
        "before_free_bytes": 1000,
        "after_free_bytes": 700,
        "total_bytes": 2000,
        "retained_delta_bytes": 300,
        "effective_allowance_bytes": 250,
        "exceeds_allowance": True,
        "observed_provider_flags": {"cublas": True, "cusolver": True},
        "provider_internal_mem_queries": 1,
    }


def test_after_above_before_yields_zero_without_underflow() -> None:
    result = summarize(ordered_records(before=1000, after=1100), POLICY, IDENTITY)
    assert result["status"] == "COMPLETE"
    assert result["measurement"]["retained_delta_bytes"] == 0
    assert result["measurement"]["exceeds_allowance"] is False


@pytest.mark.parametrize("query_index", [0, 2, 14])
@pytest.mark.parametrize(
    "invalid_values",
    [
        {"free_bytes": 2001},
        {"free_bytes": 0, "total_bytes": 0},
        {"total_bytes": 2**64},
        {"status": 1},
    ],
)
def test_successful_observation_requires_valid_memory_domain(
    query_index: int, invalid_values: dict
) -> None:
    records = ordered_records()
    records[query_index].update(invalid_values)
    with pytest.raises(ValueError, match="memory query"):
        summarize(records, POLICY, IDENTITY)


def test_unobserved_failed_vendor_query_does_not_invent_memory_values() -> None:
    records = ordered_records()
    records[2].update(status=1, observed=False, free_bytes=0, total_bytes=0)
    result = summarize(records, POLICY, IDENTITY)
    assert result["status"] == "COMPLETE"
    assert result["measurement"]["retained_delta_bytes"] == 300


def test_uint64_maximum_memory_value_is_in_domain() -> None:
    maximum = 2**64 - 1
    records = ordered_records(before=maximum, after=maximum)
    for row in (records[0], records[14]):
        row["total_bytes"] = maximum
    result = summarize(records, POLICY, IDENTITY)
    assert result["status"] == "COMPLETE"
    assert result["measurement"]["total_bytes"] == maximum
    assert result["measurement"]["retained_delta_bytes"] == 0


def test_missing_terminal_is_incomplete() -> None:
    result = summarize(ordered_records()[:-1], POLICY, IDENTITY)
    assert result["status"] == "INCOMPLETE"
    assert "missing terminal record" in result["issues"]
    assert result["measurement"] is None


@pytest.mark.parametrize(
    ("field", "value", "issue"),
    (
        ("dropped_events", 2, "capture dropped 2 event(s)"),
        ("resolution_errors", 1, "interposer resolution failed 1 time(s)"),
        ("write_errors", 1, "interposer write failed 1 time(s)"),
        ("events_written", 8, "terminal event count does not match records"),
    ),
)
def test_loss_overflow_or_short_terminal_is_incomplete(
    field: str, value: int, issue: str
) -> None:
    records = ordered_records()
    records[-1][field] = value
    result = summarize(records, POLICY, IDENTITY)
    assert result["status"] == "INCOMPLETE"
    assert issue in result["issues"]
    assert result["measurement"] is None


def test_dropped_sequence_gap_is_retained_as_incomplete() -> None:
    records = ordered_records()
    del records[3]
    records[-1]["dropped_events"] = 1
    records[-1]["events_written"] = len(records) - 1
    result = summarize(records, POLICY, IDENTITY)
    assert result["status"] == "INCOMPLETE"
    assert "event sequence contains gaps" in result["issues"]
    assert "capture dropped 1 event(s)" in result["issues"]


def test_missing_required_provider_is_unknown_not_false_or_zero() -> None:
    records = [
        row for row in ordered_records() if not row["event"].startswith("cusolver_")
    ]
    result = summarize(resequence(records), POLICY, IDENTITY)
    assert result["status"] == "INCOMPLETE"
    assert "missing required provider event: cusolver" in result["issues"]
    assert result["observed_provider_flags"] == {"cublas": True, "cusolver": None}
    assert result["measurement"] is None


def test_provider_error_is_preserved_and_blocks_derived_allowance() -> None:
    records = ordered_records()
    next(row for row in records if row["event"] == "cusolver_create")["status"] = 7
    result = summarize(records, POLICY, IDENTITY)
    assert result["status"] == "INCOMPLETE"
    assert result["provider_errors"] == [{"event": "cusolver_create", "status": 7}]
    assert result["measurement"] is None


def test_missing_provider_configuration_event_is_incomplete() -> None:
    records = [
        row for row in ordered_records() if row["event"] != "cublas_set_pointer_mode"
    ]
    result = summarize(resequence(records), POLICY, IDENTITY)
    assert result["status"] == "INCOMPLETE"
    assert (
        "missing required provider event: cublas_set_pointer_mode" in result["issues"]
    )
    assert result["observed_provider_flags"]["cublas"] is None


def test_source_policy_provider_event_order_is_required() -> None:
    records = ordered_records()
    terminal = records[-1]
    ordinary = records[:-1]
    stream_pair = [row for row in ordinary if row["event"] == "cublas_set_stream"]
    ordinary = [row for row in ordinary if row["event"] != "cublas_set_stream"]
    ordinary[1:1] = stream_pair
    result = summarize(resequence([*ordinary, terminal]), POLICY, IDENTITY)
    assert result["status"] == "INCOMPLETE"
    assert "provider event order mismatch: cublas" in result["issues"]
    assert result["observed_provider_flags"]["cublas"] is None


def test_required_provider_begin_end_pairs_are_complete() -> None:
    records = [row for row in ordered_records() if row["phase"] != "begin"]
    result = summarize(resequence(records), POLICY, IDENTITY)
    assert result["status"] == "INCOMPLETE"
    assert "missing provider begin: cublas_create" in result["issues"]
    assert "missing provider begin: cusolver_create" in result["issues"]
    assert result["observed_provider_flags"] == {"cublas": None, "cusolver": None}


def test_nested_provider_pairs_are_ambiguous() -> None:
    records = ordered_records()
    ordinary, terminal = records[:-1], records[-1]
    create_begin = next(
        index
        for index, row in enumerate(ordinary)
        if row["event"] == "cublas_create" and row["phase"] == "begin"
    )
    stream_begin = next(
        row
        for row in ordinary
        if row["event"] == "cublas_set_stream" and row["phase"] == "begin"
    )
    ordinary.remove(stream_begin)
    ordinary.insert(create_begin + 1, stream_begin)
    result = summarize(resequence([*ordinary, terminal]), POLICY, IDENTITY)
    assert result["status"] == "INCOMPLETE"
    assert "provider event pairs overlap: cublas" in result["issues"]
    assert result["observed_provider_flags"]["cublas"] is None


def nested_vendor_records() -> list[dict]:
    records = ordered_records()
    index = next(
        index
        for index, row in enumerate(records)
        if row["event"] == "cusolver_create" and row["phase"] == "begin"
    )
    # Real provider wrappers can call one another. The legacy shim labels these
    # application too, so only balanced begin/end nesting establishes the census.
    records[index + 1 : index + 1] = [
        event(0, "cublas_create", phase="begin", origin="application"),
        event(0, "cublas_create", phase="end", origin="application"),
        event(0, "cublas_set_stream", phase="begin", origin="application"),
        event(0, "cublas_set_stream", phase="end", origin="application"),
    ]
    return resequence(records)


def test_balanced_vendor_nesting_keeps_external_provider_census() -> None:
    result = summarize(nested_vendor_records(), POLICY, IDENTITY)
    assert result["status"] == "COMPLETE"
    assert result["observed_provider_flags"] == {"cublas": True, "cusolver": True}
    candidate = result["candidate_intervals"][0]
    assert candidate["provider_nesting_valid"] is True
    assert [call["event"] for call in candidate["top_level_provider_calls"]] == [
        "cublas_create",
        "cublas_set_stream",
        "cublas_set_pointer_mode",
        "cusolver_create",
        "cusolver_set_stream",
        "cusolver_create_params",
    ]
    assert [call["event"] for call in candidate["nested_provider_calls"]] == [
        "cublas_create",
        "cublas_set_stream",
    ]
    assert candidate["raw_delta_bytes"] == 300


def test_equal_repeated_vendor_intervals_do_not_select_a_primary() -> None:
    first = nested_vendor_records()
    second = nested_vendor_records()
    result = summarize(resequence([*first[:-1], *second]), POLICY, IDENTITY)
    assert result["status"] == "INCOMPLETE"
    assert result["measurement"] is None
    assert result["observed_provider_flags"] == {"cublas": None, "cusolver": None}
    assert "ambiguous provider measurement pairs: 2" in result["issues"]
    assert [
        candidate["raw_delta_bytes"] for candidate in result["candidate_intervals"]
    ] == [300, 300]
    assert all(
        candidate["provider_nesting_valid"]
        for candidate in result["candidate_intervals"]
    )


def test_observed_provider_outside_source_policy_is_rejected() -> None:
    policy = deepcopy(POLICY)
    del policy["providers"]["cusolver"]
    result = summarize(ordered_records(), policy, IDENTITY)
    assert result["status"] == "INCOMPLETE"
    assert "observed provider absent from source policy: cusolver" in result["issues"]
    assert result["observed_provider_flags"] == {"cublas": True, "cusolver": True}
    assert result["unexpected_observed_providers"] == ["cusolver"]
    assert result["measurement"] is None


def test_two_candidate_pairs_are_ambiguous() -> None:
    first = ordered_records()[:-1]
    second = ordered_records()[:-1]
    records = resequence([*first, *second, ordered_records()[-1]])
    result = summarize(records, POLICY, IDENTITY)
    assert result["status"] == "INCOMPLETE"
    assert "ambiguous provider measurement pairs: 2" in result["issues"]


def test_provider_events_outside_measurement_do_not_manufacture_flags() -> None:
    records = ordered_records()
    before = records.pop(0)
    records.insert(7, before)
    result = summarize(resequence(records), POLICY, IDENTITY)
    assert result["status"] == "INCOMPLETE"
    assert "missing required provider event: cublas" in result["issues"]
    assert result["observed_provider_flags"]["cublas"] is None


def test_missing_after_query_is_incomplete() -> None:
    records = ordered_records()
    records = [
        row
        for row in records
        if not (row["event"] == "cuda_mem_get_info" and row.get("free_bytes") == 700)
    ]
    result = summarize(resequence(records), POLICY, IDENTITY)
    assert result["status"] == "INCOMPLETE"
    assert "no complete provider measurement pair" in result["issues"]


def test_identity_mismatch_is_rejected() -> None:
    policy = deepcopy(POLICY)
    policy["source_commit"] = "f" * 40
    with pytest.raises(ValueError, match="source identity"):
        summarize(ordered_records(), policy, IDENTITY)


def test_shim_resolution_failure_retains_incomplete_report() -> None:
    records = ordered_records()
    records.insert(1, event(0, "resolution_error", status=-1, origin="interposer"))
    records[-1]["resolution_errors"] = 1
    result = summarize(resequence(records), POLICY, IDENTITY)
    assert result["status"] == "INCOMPLETE"
    assert result["measurement"] is None
    assert "interposer resolution failed 1 time(s)" in result["issues"]


def test_observed_resolution_error_cannot_be_hidden_by_terminal_counter() -> None:
    records = ordered_records()
    records.insert(1, event(0, "resolution_error", status=-1, origin="interposer"))
    result = summarize(resequence(records), POLICY, IDENTITY)
    assert result["status"] == "INCOMPLETE"
    assert result["measurement"] is None


def test_duplicate_sequence_is_rejected() -> None:
    records = ordered_records()
    records[1]["sequence"] = records[0]["sequence"]
    with pytest.raises(ValueError, match="sequence"):
        summarize(records, POLICY, IDENTITY)


def test_missing_phase_is_rejected() -> None:
    records = ordered_records()
    del records[2]["phase"]
    with pytest.raises(ValueError, match="phase"):
        summarize(records, POLICY, IDENTITY)


def test_unknown_memory_query_origin_is_rejected() -> None:
    records = ordered_records()
    records[0]["origin"] = "guessed"
    with pytest.raises(ValueError, match="origin"):
        summarize(records, POLICY, IDENTITY)


def test_jsonl_loader_and_cli_write_bounded_report(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    events_path = tmp_path / "events.jsonl"
    events_path.write_text(
        "".join(json.dumps(record) + "\n" for record in ordered_records()),
        encoding="utf-8",
    )
    policy_path, identity_path = tmp_path / "policy.json", tmp_path / "identity.json"
    policy_path.write_text(json.dumps(POLICY), encoding="utf-8")
    identity = deepcopy(IDENTITY)
    identity["event_log_sha256"] = hashlib.sha256(events_path.read_bytes()).hexdigest()
    identity_path.write_text(json.dumps(identity), encoding="utf-8")
    output = tmp_path / "report.json"
    assert report_module.load_events(events_path) == ordered_records()
    monkeypatch.setattr(
        "sys.argv",
        [
            "rhf_provider_retention_report.py",
            "--events",
            str(events_path),
            "--policy",
            str(policy_path),
            "--identity",
            str(identity_path),
            "--output",
            str(output),
        ],
    )
    assert report_module.main() == 0
    assert json.loads(output.read_text())["status"] == "COMPLETE"


@pytest.mark.parametrize("contents", ("\n", "[]\n", "{bad json}\n"))
def test_jsonl_loader_rejects_empty_nonobject_or_invalid_rows(
    tmp_path: Path, contents: str
) -> None:
    path = tmp_path / "bad.jsonl"
    path.write_text(contents, encoding="utf-8")
    with pytest.raises(ValueError):
        report_module.load_events(path)


def test_cli_rejects_event_log_identity_mismatch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    events = tmp_path / "events.jsonl"
    events.write_text(
        "".join(json.dumps(record) + "\n" for record in ordered_records()),
        encoding="utf-8",
    )
    policy, identity = tmp_path / "policy.json", tmp_path / "identity.json"
    policy.write_text(json.dumps(POLICY), encoding="utf-8")
    identity.write_text(json.dumps(IDENTITY), encoding="utf-8")
    monkeypatch.setattr(
        "sys.argv",
        [
            "rhf_provider_retention_report.py",
            "--events",
            str(events),
            "--policy",
            str(policy),
            "--identity",
            str(identity),
            "--output",
            str(tmp_path / "report.json"),
        ],
    )
    with pytest.raises(ValueError, match="event log identity"):
        report_module.main()
