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
from tools.capture_prepared_allocations import digest
from tools.cupti_graph_inventory import GraphInventory
from tools.cupti_residency_capture import CuptiCapture
from tools.cupti_source_capture import SourceInventory
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


@pytest.fixture
def constructor_environment(
    receipt: dict[str, Any], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> Any:
    """ABI doubles exercise real Python admission; they never load CUDA/CUPTI."""
    from tools import cupti_graph_inventory

    native_path, collector_path, proof_path = (
        tmp_path / name for name in ("native-double", "collector-double", "proof.json")
    )
    native_path.write_bytes(b"not a native library")
    collector_path.write_bytes(b"not a CUPTI collector")
    state = {
        "uuid": "GPU-test",
        "hostname": "pod-test",
        "now": 1010,
        "runtime": 12090,
        "driver": 13020,
        "enabled": False,
    }

    class LibraryDouble:
        def __init__(self, name: str) -> None:
            self._name = name
            self._handle = id(self)

        def __getattr__(self, name: str) -> Any:
            def function(*args: Any) -> int:
                if name == "cudaRuntimeGetVersion":
                    args[0]._obj.value = state["runtime"]
                elif name == "cudaDriverGetVersion":
                    args[0]._obj.value = state["driver"]
                elif name.startswith("generativeqc_cupti_") and "begin" in name:
                    state["enabled"] = True
                return 0

            setattr(self, name, function)
            return function

    native, collector = (
        LibraryDouble(str(native_path)),
        LibraryDouble(str(collector_path)),
    )
    monkeypatch.delenv("SLURM_JOB_ID", raising=False)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    monkeypatch.setenv("NVIDIA_VISIBLE_DEVICES", "GPU-test")
    monkeypatch.setattr(residency_allocation, "native_uuid", lambda _: state["uuid"])
    monkeypatch.setattr(
        residency_allocation.socket, "gethostname", lambda: state["hostname"]
    )
    monkeypatch.setattr(residency_allocation.time, "time", lambda: state["now"])
    monkeypatch.setattr(
        residency_allocation.ctypes,
        "CDLL",
        lambda path: native if str(path) == str(native_path) else collector,
    )
    monkeypatch.setattr(cupti_graph_inventory, "OBSERVER", lambda *_: object())

    def options() -> dict[str, Any]:
        proof_path.write_text(json.dumps(receipt))
        return {
            "native_library": native_path,
            "qz_receipt": proof_path,
            "qz_receipt_sha256": digest(proof_path),
        }

    def construct(kind: str, **kwargs: Any) -> Any:
        if kind == "activity":
            return CuptiCapture(collector_path, capacity=64, **kwargs)
        cls = GraphInventory if kind == "graph" else SourceInventory
        return cls(collector_path, native, capacity=64, **kwargs)

    return SimpleNamespace(
        state=state,
        options=options,
        construct=construct,
        collector_path=collector_path,
        native=native,
    )


@pytest.mark.parametrize("kind", ["activity", "graph", "source"])
def test_real_constructor_accepts_verified_qz_without_slurm(
    constructor_environment: Any, kind: str
) -> None:
    env = constructor_environment
    assert env.construct(kind, **env.options()) is not None
    assert env.state["enabled"]


@pytest.mark.parametrize("kind", ["activity", "graph", "source"])
@pytest.mark.parametrize(
    "fault",
    [
        "missing",
        "hash",
        "gpu",
        "container",
        "stale",
        "mixed",
        "driver",
        "runtime",
        "collector_as_native",
        "bool_receipt",
        "bool_native",
        "bool_hash",
        "schema",
        "native_handle",
    ],
)
def test_real_constructor_rejects_bad_proof_before_enabling_cupti(
    constructor_environment: Any,
    receipt: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
    kind: str,
    fault: str,
) -> None:
    env = constructor_environment
    if fault == "schema":
        receipt["schema"] = "old"
    options = env.options()
    if fault == "missing":
        options["qz_receipt"] = None
    elif fault == "hash":
        options["qz_receipt_sha256"] = "0" * 64
    elif fault == "gpu":
        env.state["uuid"] = "GPU-other"
    elif fault == "container":
        env.state["hostname"] = "other"
    elif fault == "stale":
        env.state["now"] = 9000
    elif fault == "mixed":
        monkeypatch.setenv("SLURM_JOB_ID", "fabricated")
    elif fault in ("driver", "runtime"):
        env.state[fault] = 12080
    elif fault == "collector_as_native":
        options["native_library"] = env.collector_path
    elif fault.startswith("bool_"):
        options[
            {
                "bool_receipt": "qz_receipt",
                "bool_native": "native_library",
                "bool_hash": "qz_receipt_sha256",
            }[fault]
        ] = True
    elif fault == "native_handle":
        env.native._name = "different-native-library"
        if kind == "activity":
            # Activity admission has no existing handle; its distinct-library
            # contract is checked by the collector_as_native case instead.
            options["native_library"] = env.collector_path
    with pytest.raises(InvalidReceipt):
        env.construct(kind, **options)
    assert not env.state["enabled"]


@pytest.mark.parametrize("kind", ["activity", "graph", "source"])
def test_existing_direct_slurm_constructors_remain_supported(
    constructor_environment: Any, monkeypatch: pytest.MonkeyPatch, kind: str
) -> None:
    monkeypatch.setenv("SLURM_JOB_ID", "host-ABI-double")
    assert constructor_environment.construct(kind) is not None


def test_self_reported_platform_is_not_a_constructor_authorization(
    constructor_environment: Any,
) -> None:
    env = constructor_environment
    with pytest.raises(TypeError):
        env.construct("activity", platform="qz")
    assert not env.state["enabled"]


def test_receipt_hash_and_identity_use_one_byte_snapshot(
    constructor_environment: Any,
    receipt: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    env = constructor_environment
    options = env.options()
    replacement = deepcopy(receipt)
    replacement["allocation"]["project_id"] = "unreviewed-project"
    replacement["api"]["job"]["project_id"] = "unreviewed-project"

    def replace_during_native_probe(_: Any) -> str:
        options["qz_receipt"].write_text(json.dumps(replacement))
        return env.state["uuid"]

    monkeypatch.setattr(
        residency_allocation, "native_uuid", replace_during_native_probe
    )
    _, proof = residency_allocation.allocation_identity(
        options["native_library"],
        options["qz_receipt"],
        options["qz_receipt_sha256"],
    )
    assert proof["allocation"] == receipt["allocation"]
    assert proof["receipt_sha256"] == options["qz_receipt_sha256"]
    assert digest(options["qz_receipt"]) != proof["receipt_sha256"]


@pytest.mark.parametrize("kind", ["activity", "graph", "source"])
def test_constructor_rejects_hardlinked_collector_as_native(
    constructor_environment: Any, kind: str
) -> None:
    env = constructor_environment
    options = env.options()
    alias = env.collector_path.with_name("collector-hardlink")
    alias.hardlink_to(env.collector_path)
    options["native_library"] = alias
    env.native._name = str(alias)
    with pytest.raises(InvalidReceipt, match="collector cannot substitute"):
        env.construct(kind, **options)
    assert not env.state["enabled"]


@pytest.mark.parametrize("kind", ["graph", "source"])
def test_constructor_rejects_same_name_with_different_loaded_handle(
    constructor_environment: Any, kind: str
) -> None:
    env = constructor_environment
    other_native = type(env.native)(env.native._name)
    assert other_native._handle != env.native._handle
    constructor = GraphInventory if kind == "graph" else SourceInventory
    with pytest.raises(InvalidReceipt, match="handle/library mismatch"):
        constructor(env.collector_path, other_native, capacity=64, **env.options())
    assert not env.state["enabled"]


@pytest.mark.parametrize("kind", ["graph", "source"])
def test_constructor_accepts_another_wrapper_for_the_same_loaded_handle(
    constructor_environment: Any, kind: str
) -> None:
    env = constructor_environment
    same_native = type(env.native)(env.native._name)
    same_native._handle = env.native._handle
    constructor = GraphInventory if kind == "graph" else SourceInventory
    assert constructor(env.collector_path, same_native, capacity=64, **env.options())
    assert env.state["enabled"]


def test_one_snapshot_parser_still_rejects_duplicate_keys(
    constructor_environment: Any,
) -> None:
    env = constructor_environment
    options = env.options()
    proof_path = options["qz_receipt"]
    proof_path.write_text('{"schema":"first","schema":"second"}')
    with pytest.raises(InvalidReceipt, match="duplicate JSON field"):
        residency_allocation.allocation_identity(
            options["native_library"], proof_path, digest(proof_path)
        )
    assert not env.state["enabled"]
