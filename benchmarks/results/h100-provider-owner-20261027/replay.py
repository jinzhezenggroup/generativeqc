"""Replay retained engineering outcomes; no model/library/GPU execution."""

import hashlib
import io
import json
import lzma
import tarfile
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
    shared = summary["receipt_storage"]
    assert shared["path"] in {"v4-v6-receipts.tar.xz", "../v4-v6-receipts.tar.xz"}
    pool_root = root.parent if shared["path"].startswith("../") else root
    pool_bytes = (pool_root / "v4-v6-receipts.tar.xz").read_bytes()
    assert (
        len(pool_bytes) == shared["bytes"]
        and hashlib.sha256(pool_bytes).hexdigest() == shared["sha256"]
    )
    with tarfile.open(fileobj=io.BytesIO(pool_bytes), mode="r:xz") as archive:
        assert set(archive.getnames()) == {"v4/receipt.json", "v6/receipt.json"}
        entry = archive.getmember(shared["member"])
        assert entry.isfile()
        receipt_bytes = archive.extractfile(entry).read()
    assert hashlib.sha256(receipt_bytes).hexdigest() == shared["uncompressed_sha256"]
    recovered_receipt = lzma.compress(receipt_bytes, preset=9)
    original_receipt = summary["original_archives"]["receipt.json.xz"]
    assert len(recovered_receipt) == original_receipt["bytes"]
    assert hashlib.sha256(recovered_receipt).hexdigest() == original_receipt["sha256"]
    loaded["receipt.json.xz"] = json.loads(receipt_bytes)
    delta = loaded["source-closure.json.xz"]
    assert delta["schema"] == "generativeqc.h100-owner-v4-source-delta-proposal.v1"
    anchor = delta["base_archive"]
    assert anchor["path"] == "v6/source-closure.json.xz"
    base_bytes = (root / anchor["path"]).read_bytes()
    assert (
        len(base_bytes) == anchor["bytes"]
        and hashlib.sha256(base_bytes).hexdigest() == anchor["sha256"]
    )
    base_sources = json.loads(lzma.decompress(base_bytes))["sources"]
    source = {
        "manifest": delta["manifest"],
        "sources": {
            name: delta["changed_sources"][name]
            if name in delta["changed_sources"]
            else base_sources[name]
            for name in delta["manifest"]["files"]
        },
    }
    recovered_source_bytes = (
        json.dumps(source, ensure_ascii=True, separators=(",", ":")) + "\n"
    ).encode()
    assert (
        hashlib.sha256(recovered_source_bytes).hexdigest()
        == delta["original_uncompressed_sha256"]
    )
    recovered_source = lzma.compress(recovered_source_bytes, preset=9)
    original_source = summary["original_archives"]["source-closure.json.xz"]
    assert len(recovered_source) == original_source["bytes"]
    assert (
        hashlib.sha256(recovered_source).hexdigest()
        == original_source["sha256"]
        == delta["original_archive_sha256_witness"]
    )
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
    assert "36 passed" in payload["host_contract_transport"]["stdout"]
    print("PASS_RETAINED_MODEL_FREE_ENGINEERING_ONLY; SDK_INCOMPLETE; SCIENCE_NOT_RUN")


if __name__ == "__main__":
    main()
