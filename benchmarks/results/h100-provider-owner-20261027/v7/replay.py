"""Replay retained engineering outcomes; no model/library/GPU execution."""

import hashlib
import json
import lzma
from pathlib import Path


def main() -> None:
    root = Path(__file__).resolve().parent
    summary = json.loads((root / "summary.json").read_bytes())
    loaded = {}
    for name, row in summary["archives"].items():
        data = (root / name).read_bytes()
        assert (
            len(data) == row["bytes"]
            and hashlib.sha256(data).hexdigest() == row["sha256"]
        )
        loaded[name] = json.loads(lzma.decompress(data))
    source = loaded["source-closure.json.xz"]
    assert source["schema"] == "generativeqc.h100-owner-v7-source-delta.v1"
    ancestor = source["base_archive"]
    assert ancestor["path"] == "../v6/source-closure.json.xz"
    base_bytes = (root.parent / "v6/source-closure.json.xz").read_bytes()
    assert len(base_bytes) == ancestor["bytes"]
    assert hashlib.sha256(base_bytes).hexdigest() == ancestor["sha256"]
    base_sources = json.loads(lzma.decompress(base_bytes))["sources"]
    assert source["changed_sources"].keys() <= source["manifest"]["files"].keys()
    source["sources"] = {
        name: (
            source["changed_sources"][name]
            if name in source["changed_sources"]
            else base_sources[name]
        )
        for name in source["manifest"]["files"]
    }
    assert source["manifest"]["base_commit"] == summary["measured_source_base_commit"]
    for name, row in source["manifest"]["files"].items():
        data = source["sources"][name].encode()
        assert (
            len(data) == row["bytes"]
            and hashlib.sha256(data).hexdigest() == row["sha256"]
        )
    manifest_bytes = (json.dumps(source["manifest"], indent=2) + "\n").encode()
    assert (
        hashlib.sha256(manifest_bytes).hexdigest() == summary["source_manifest_sha256"]
    )
    payload = loaded["receipt.json.xz"]
    assert (
        payload["remote_collection_metadata"]["source_manifest_sha256"]
        == summary["source_manifest_sha256"]
    )
    build = json.loads(payload["exported_generated_text"]["build/receipt.json"])
    assert build["source_manifest_sha256"] == summary["source_manifest_sha256"]
    assert build["binary"] == summary["binary"]
    assert payload["capabilities_enabled"] == 0
    assert (
        payload["original_persistence"]["status"]
        == "ORIGINAL_BYTES_PERSISTED_AND_REHASHED"
    )
    assert payload["host_red_regression"]["returncode"] != 0
    assert (
        "failed attachment kept a published bucket"
        in payload["host_red_regression"]["stdout"]
    )
    assert payload["formatted_host_regression"]["returncode"] == 0
    compacted = payload["compacted_generated_members"]["build/link.map"]
    witness = payload["remote_collection_metadata"]["inventory"]["build/link.map"]
    assert compacted["original_bytes"] == witness["bytes"]
    assert compacted["original_sha256"] == witness["sha256"]
    excerpt = compacted["excerpt"].encode()
    assert len(excerpt) == compacted["excerpt_bytes"]
    assert hashlib.sha256(excerpt).hexdigest() == compacted["excerpt_sha256"]
    assert compacted["full_text_in_this_capsule"] is False
    frames, receipt = payload["frames"], payload["receipt"]
    statuses = [row for row in frames if row["event"] == "raw_api_return"]
    assert len(statuses) == summary["api_success_records"]
    assert all(row["raw_status"] == 0 for row in statuses)
    schema = [row for row in frames if row["event"] == "actual_inventory_schema"]
    inventory = [row for row in frames if row["event"] == "actual_owned_inventory"]
    assert [(r["slot"], r["kind"]) for r in schema] == [
        (r["slot"], r["kind"]) for r in inventory
    ]
    assert len(inventory) == summary["inventory_slots"]
    close = [row for row in frames if row["event"] == "actual_close_report"]
    assert len(close) == summary["close_operations"]
    assert all(row["raw_status"] in (None, 0) for row in close)
    outcome = next(row for row in frames if row["event"] == "owner_close_outcome")
    assert outcome["confirmed"] and outcome["closed"] and outcome["bucket_absent"]
    assert not outcome["source_lease_active"] and outcome["owner_pin_count"] == 0
    identity = next(
        row for row in frames if row["event"] == "PRE_VENDOR_IDENTITY_WAIT_ROOT"
    )
    post = next(row for row in frames if row["event"] == "post_close_current_identity")
    for key in (
        "device",
        "device_uuid",
        "context_id",
        "driver_version",
        "runtime_version",
    ):
        assert post[key] == identity[key]
    assert post["build_ids"] == [row["build_id"] for row in identity["libraries"]]
    assert receipt["engineering_close_confirmed"]
    assert receipt["transport"]["returncode"] == 0 and receipt["transport"]["reaped"]
    assert receipt["transport"]["owned_group_absent"] is True
    assert (
        receipt["actual_final_clock"]["observed_mono_ns"]
        < receipt["original_native_D_mono_ns"]
    )
    reservation = next(
        row for row in frames if row["event"] == "reservation_and_observation"
    )
    capacity = next(
        row for row in frames if row["event"] == "existing_capacity_interfaces"
    )
    assert reservation["A"] == summary["A"]
    assert (
        reservation["observed_synchronized_free_delta"]
        == summary["observed_synchronized_free_delta_bytes"]
    )
    assert (
        capacity["B_endpoint"] - capacity["retained_A_once"]
        == capacity["endpoint_remaining"]
    )
    assert (
        reservation["observed_synchronized_free_delta"]
        <= reservation["A"]
        <= capacity["retained_A_once"]
    )
    assert payload["science"] == "NOT_RUN" and payload["installed_sdk"] == "INCOMPLETE"
    assert payload["privacy"]["redacted_export_is_remote_original"] is False
    assert payload["remote_collection_metadata"]["complete"] is True
    assert payload["host_contract_transport"]["returncode"] == 0
    assert (
        f"{summary['host_contract_tests']} passed"
        in payload["host_contract_transport"]["stdout"]
    )
    policy = payload["root_conditional_policy"]
    grant = payload["root_grant_receipt"]
    assert (
        hashlib.sha256((json.dumps(policy, indent=2) + "\n").encode()).hexdigest()
        == grant["conditional_policy_sha256"]
    )
    assert grant["deadline_extended"] is False
    assert grant["actual_grant_mono_ns"] < grant["original_grant_D_mono_ns"] - 5 * 10**9
    current = json.loads(payload["exported_generated_text"]["identity-qualified.json"])
    assert current["source_sha256"] == policy["source_manifest_sha256"]
    assert current["device_uuid"] == policy["device_uuid"]
    for actual, expected in zip(current["libraries"], policy["libraries"], strict=True):
        for key in ("index", "path", "bytes", "build_id"):
            assert actual[key] == expected[key]
        assert actual["file_witness"]["sha256"] == expected["sha256"]
    print("PASS_RETAINED_MODEL_FREE_ENGINEERING_ONLY; SDK_INCOMPLETE; SCIENCE_NOT_RUN")


if __name__ == "__main__":
    main()
