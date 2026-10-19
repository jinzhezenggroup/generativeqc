"""Executable scheduler negative controls; none of these tests qualify a GPU."""

from __future__ import annotations

import json
from copy import deepcopy
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from pathlib import Path

import pytest

from benchmarks.qualify_residency_qz import snapshot
from tools import residency_allocation
from tools.audit_replay_allocations import InvalidReceipt
from tools.residency_allocation import SCHEMA, validate_qz


@pytest.fixture
def receipt() -> dict[str, Any]:
    allocation = {
        "project_id": "project-test",
        "workspace_id": "ws-test",
        "workload_id": "job-test",
        "workload_name": "qualification",
        "instance_id": "pod-test",
        "pod": "pod-test",
        "node": "node-test",
        "compute_group": "H200-test",
        "kind": "job",
        "status": "RUNNING",
        "gpu_count": 1,
        "gpu_uuid": "GPU-test",
    }
    return {
        "schema": SCHEMA,
        "observed_unix_seconds": 1000,
        "allocation": allocation,
        "api": {
            "job": {
                "project_id": "project-test",
                "workspace_id": "ws-test",
                "job_id": "job-test",
                "name": "qualification",
                "logic_compute_group_name": "H200-test",
                "gpu_count": 1,
                "status": "job_running",
                "node_count": 1,
            },
            "instance": {
                "name": "pod-test",
                "node": "node-test",
                "instance_status": "instance_running",
            },
        },
    }


def check(receipt: dict[str, Any], **kwargs: Any) -> dict[str, Any]:
    options = {
        "uuid": "GPU-test",
        "hostname": "pod-test",
        "now": 1010,
        "environment": {
            "CUDA_VISIBLE_DEVICES": "0",
            "NVIDIA_VISIBLE_DEVICES": "GPU-test",
        },
    }
    options.update(kwargs)
    return validate_qz(receipt, **options)


def test_qz_binds_api_container_and_native_uuid(receipt: dict[str, Any]) -> None:
    result = check(receipt)
    assert result["allocation"]["workload_id"] == "job-test"
    assert result["platform"] == "qz"
    assert result["cuda_visible_devices"] == "0"


@pytest.mark.parametrize(
    "field",
    [
        "project_id",
        "workspace_id",
        "workload_id",
        "workload_name",
        "instance_id",
        "pod",
        "node",
        "compute_group",
        "gpu_count",
        "gpu_uuid",
        "status",
    ],
)
def test_forged_or_inconsistent_allocation_rejected(
    receipt: dict[str, Any], field: str
) -> None:
    receipt["allocation"][field] = "counterfeit"
    with pytest.raises(InvalidReceipt):
        check(receipt)


@pytest.mark.parametrize(
    "field",
    [
        "project_id",
        "workspace_id",
        "workload_id",
        "instance_id",
        "pod",
        "node",
        "compute_group",
    ],
)
def test_missing_identity_rejected(receipt: dict[str, Any], field: str) -> None:
    del receipt["allocation"][field]
    with pytest.raises(InvalidReceipt):
        check(receipt)


@pytest.mark.parametrize(
    "change",
    [
        {"schema": "old"},
        {"observed_unix_seconds": 1011},
        {"observed_unix_seconds": -10000},
    ],
)
def test_old_schema_stale_and_future_observations_rejected(
    receipt: dict[str, Any], change: dict[str, Any]
) -> None:
    receipt.update(change)
    with pytest.raises(InvalidReceipt):
        check(receipt)


@pytest.mark.parametrize(
    "environment",
    [
        {},
        {"CUDA_VISIBLE_DEVICES": "0"},
        {"CUDA_VISIBLE_DEVICES": "1", "NVIDIA_VISIBLE_DEVICES": "GPU-test"},
        {"CUDA_VISIBLE_DEVICES": "0", "NVIDIA_VISIBLE_DEVICES": "GPU-other"},
        {
            "CUDA_VISIBLE_DEVICES": "0",
            "NVIDIA_VISIBLE_DEVICES": "GPU-test",
            "SLURM_JOB_ID": "fake",
        },
    ],
)
def test_visibility_drift_or_fake_slurm_rejected(
    receipt: dict[str, Any], environment: dict[str, str]
) -> None:
    with pytest.raises(InvalidReceipt):
        check(receipt, environment=environment)


@pytest.mark.parametrize(
    "section,field,value",
    [
        ("job", "status", "job_succeeded"),
        ("job", "node_count", 2),
        ("instance", "instance_status", "instance_success"),
        ("instance", "name", "other"),
        ("instance", "node", "other"),
        ("job", "gpu_count", 2),
    ],
)
def test_allocation_api_mismatch_rejected(
    receipt: dict[str, Any], section: str, field: str, value: Any
) -> None:
    receipt["api"][section][field] = value
    with pytest.raises(InvalidReceipt):
        check(receipt)


def test_host_or_runtime_drift_rejected(receipt: dict[str, Any]) -> None:
    original = deepcopy(receipt)
    for change in ({"hostname": "other"}, {"uuid": "GPU-other"}):
        with pytest.raises(InvalidReceipt):
            check(receipt, **change)
    assert receipt == original


def test_wrong_receipt_digest_fails_before_loading_native(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "receipt.json"
    path.write_text("{}")
    monkeypatch.setattr(
        residency_allocation, "native_uuid", lambda _: pytest.fail("loaded CUDA")
    )
    with pytest.raises(InvalidReceipt, match="hash mismatch"):
        residency_allocation.allocation_identity(tmp_path / "library", path, "0" * 64)


def test_live_snapshot_rejects_instance_move_during_probe(
    receipt: dict[str, Any],
) -> None:
    instance = SimpleNamespace(raw=receipt["api"]["instance"], label="rank=0")
    moved = SimpleNamespace(raw={**instance.raw, "node": "other"})
    calls = iter(((instance,), (moved,)))
    job = SimpleNamespace(status="RUNNING", ref="test-ref", raw=receipt["api"]["job"])
    probe = {
        "hostname": "pod-test",
        "environment": {
            "CUDA_VISIBLE_DEVICES": "0",
            "NVIDIA_VISIBLE_DEVICES": "GPU-test",
        },
        "nvidia_smi": "GPU-test, Disabled",
    }
    client = SimpleNamespace(
        jobs=SimpleNamespace(
            get=lambda *_args, **_kwargs: job,
            instances=lambda *_: next(calls),
            exec=lambda *_args, **_kwargs: SimpleNamespace(
                completed=True,
                returncode=0,
                truncated=False,
                stdout=json.dumps(probe),
                output="",
            ),
        )
    )
    with pytest.raises(InvalidReceipt, match="assignment changed"):
        snapshot(client, "qualification", "workspace", "/usr/bin/python")
