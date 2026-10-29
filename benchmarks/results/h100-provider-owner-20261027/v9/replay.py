"""Offline full-text/model-free replay; no native library, GPU or network use."""

import hashlib
import json
import lzma
from pathlib import Path


def digest(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def source_closure(
    path: Path, family: Path, seen: set[Path]
) -> tuple[dict[str, str], dict]:
    path = path.resolve()
    assert path.is_relative_to(family) and path not in seen and not path.is_symlink()
    seen.add(path)
    value = json.loads(lzma.decompress(path.read_bytes()))
    manifest = value["manifest"]
    if "sources" in value:
        sources = value["sources"]
    else:
        row = value["base_archive"]
        assert row["path"] in (
            "../v7/source-closure.json.xz",
            "../v6/source-closure.json.xz",
        )
        ancestor = (path.parent / row["path"]).resolve()
        assert ancestor.is_relative_to(family) and not ancestor.is_symlink()
        raw = ancestor.read_bytes()
        assert len(raw) == row["bytes"] and digest(raw) == row["sha256"]
        base, _ = source_closure(ancestor, family, seen)
        assert value["changed_sources"].keys() <= manifest["files"].keys()
        sources = {
            n: value["changed_sources"].get(n, base.get(n)) for n in manifest["files"]
        }
    assert sources.keys() == manifest["files"].keys()
    for name, row in manifest["files"].items():
        raw = sources[name].encode()
        assert len(raw) == row["bytes"] and digest(raw) == row["sha256"]
    return sources, manifest


def main() -> None:
    root = Path(__file__).resolve().parent
    summary = json.loads((root / "summary.json").read_bytes())
    loaded = {}
    for name, row in summary["archives"].items():
        assert Path(name).name == name
        raw = (root / name).read_bytes()
        assert len(raw) == row["bytes"] and digest(raw) == row["sha256"]
        loaded[name] = json.loads(lzma.decompress(raw))
    sources, manifest = source_closure(
        root / "source-closure.json.xz", root.parent.resolve(), set()
    )
    assert manifest["base_commit"] == summary["measured_source_base_commit"]
    assert len(sources) == summary["source_inputs"] == 105
    assert (
        digest((json.dumps(manifest, indent=2) + "\n").encode())
        == summary["source_manifest_sha256"]
    )
    payload = loaded["receipt.json.xz"]
    assert payload["scope"] == "MODEL_FREE_ENGINEERING_ONLY"
    texts = payload["exported_generated_text"]
    assert (
        texts.keys()
        == payload["public_files"].keys()
        == payload["original_byte_witnesses"].keys()
    )
    assert len(texts) == summary["full_generated_text_members"]
    for name, text in texts.items():
        raw = text.encode()
        row = payload["public_files"][name]
        assert len(raw) == row["bytes"] and digest(raw) == row["sha256"]
        original = payload["original_byte_witnesses"][name]
        if row["original_bytes_available_in_public"]:
            assert len(raw) == original["bytes"] and digest(raw) == original["sha256"]
    original = payload["original_byte_witnesses"]
    assert (
        sum(row["bytes"] for row in original.values())
        == summary["full_original_generated_text_bytes"]
    )
    assert "build/link.map" in texts and original["build/link.map"]["bytes"] > 4 << 20
    parsed = {n: json.loads(raw) for n, raw in texts.items() if n.endswith(".json")}
    build = parsed["build/receipt.json"]
    assert build["binary"] == summary["binary"]
    assert build["source_manifest_sha256"] == summary["source_manifest_sha256"]
    closure = parsed["build/closure.json"]
    assert (
        closure["status"] == "COMPLETE" and closure["foreground_group_absent"] is True
    )
    assert (
        closure["supervisor_cleanup"]["reaped"]
        and closure["supervisor_cleanup"]["owned_group_absent"]
    )
    assert closure["owned_ipc_cleanup"]["status"] == "CLOSED_OWNED_IPC"
    binding = parsed["run-binding.json"]
    assert binding["source_manifest_sha256"] == summary["source_manifest_sha256"]
    for phase in ("build", "native"):
        terminal = parsed[phase + "-main-terminal.json"]
        seal = parsed[phase + "-main-terminal.seal.json"]
        assert (
            terminal["status"] == "COMPLETE"
            and terminal["phase"] == phase
            and terminal["normal_returncode"] == 0
        )
        assert terminal["source_manifest_sha256"] == summary["source_manifest_sha256"]
        assert (
            terminal["actual_final_clock"]["observed_mono_ns"]
            <= terminal["effective_D_mono_ns"]
            <= binding["hard_D_mono_ns"]
        )
        assert (
            seal["actual_post_fsync_clock"]["observed_mono_ns"]
            <= terminal["effective_D_mono_ns"]
        )
        assert (
            seal["terminal_original_sha256"]
            == original[phase + "-main-terminal.json"]["sha256"]
        )
        assert (
            terminal["dependent_original_sha256"]
            == original[terminal["dependent_member"]]["sha256"]
        )
        assert (
            terminal["dependent_original_bytes"]
            == original[terminal["dependent_member"]]["bytes"]
        )
        assert (
            terminal["dependent_original_mode"]
            == original[terminal["dependent_member"]]["mode"]
        )
    frames = [json.loads(line) for line in texts["native.raw"].splitlines()]
    statuses = [row for row in frames if row["event"] == "raw_api_return"]
    assert len(statuses) == summary["api_success_records"] == 53
    assert all(row["raw_status"] == 0 for row in statuses)
    schema = [row for row in frames if row["event"] == "actual_inventory_schema"]
    inventory = [row for row in frames if row["event"] == "actual_owned_inventory"]
    assert [(r["slot"], r["kind"]) for r in schema] == [
        (r["slot"], r["kind"]) for r in inventory
    ]
    assert len(inventory) == summary["inventory_slots"] == 21
    close = [row for row in frames if row["event"] == "actual_close_report"]
    assert len(close) == summary["close_operations"] == 24
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
    receipt = parsed["native-receipt.json"]
    assert (
        receipt["engineering_close_confirmed"] and not receipt["remaining_known_actors"]
    )
    assert (
        receipt["verification"]["parsed_close_inventory_complete"]
        and receipt["verification"]["post_close_identity_matches"]
    )
    assert (
        receipt["transport"]["returncode"] == 0
        and receipt["transport"]["reaped"]
        and receipt["transport"]["owned_group_absent"]
    )
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
    policy = payload["root_conditional_policy"]
    grant = payload["root_grant_receipt"]
    assert (
        digest((json.dumps(policy, indent=2) + "\n").encode())
        == payload["public_policy_sha256"]
    )
    assert (
        original["root-conditional-policy.json"]["sha256"]
        == grant["conditional_policy_sha256"]
    )
    assert grant["deadline_extended"] is False
    assert grant["actual_grant_mono_ns"] < grant["original_grant_D_mono_ns"] - 5 * 10**9
    current = parsed["identity-qualified.json"]
    assert current["source_sha256"] == policy["source_manifest_sha256"]
    assert current["device_uuid"] == policy["device_uuid"]
    for actual, expected in zip(current["libraries"], policy["libraries"], strict=True):
        for key in ("index", "path", "bytes", "build_id"):
            assert actual[key] == expected[key]
        assert actual["file_witness"]["sha256"] == expected["sha256"]
    assert payload["root_observation"]["authentication"][
        "normal_terminal_authenticated"
    ]
    assert not payload["root_observation"]["deadline_extended"]
    assert not payload["control_failures"]["v8"]["cache_server_created"]
    assert not payload["control_failures"]["v8"]["native_attempt_claim"]
    assert payload["control_failures"]["v9_original_collector"][
        "source_collector_unchanged"
    ]
    assert (
        payload["host_validation"]["tests"] == 226
        and payload["host_validation"]["subtests"] == 25
    )
    assert payload["science"] == "NOT_RUN" and payload["installed_sdk"] == "INCOMPLETE"
    assert (
        payload["capabilities_enabled"] == 0
        and not summary["default_allowance_changed"]
    )
    assert payload["privacy"]["redacted_export_is_remote_original"] is False
    print(
        "PASS_CURRENT_INPUT_MODEL_FREE_ENGINEERING_ONLY; SDK_INCOMPLETE; SCIENCE_NOT_RUN"
    )


if __name__ == "__main__":
    main()
