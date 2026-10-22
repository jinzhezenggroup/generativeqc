"""Qualification-only allocation evidence; never grant residency acceptance.

qz receipts are operator-retained API observations, not signed attestations.
Their independently supplied SHA-256 binds the reviewed observation, while the
container and native runtime checks reject reuse on another allocation. Keep the
original API response alongside the receipt for external provenance review.
"""

from __future__ import annotations

import ctypes
import hashlib
import json
import os
import socket
import subprocess
import time
from pathlib import Path
from typing import Any

from tools.audit_replay_allocations import _require, _unique_object
from tools.capture_prepared_allocations import visible_device_uuid

SCHEMA = "generativeqc.qz-allocation.v1"


def require_cupti_allocation(
    collector: Path,
    *,
    native_library: Path | None = None,
    qz_receipt: Path | None = None,
    qz_receipt_sha256: str | None = None,
    native: Any = None,
) -> None:
    """Reject invalid allocation before enabling any activity/source observer."""
    if all(value is None for value in (native_library, qz_receipt, qz_receipt_sha256)):
        _require(bool(os.environ.get("SLURM_JOB_ID")), "CUPTI capture requires Slurm")
        return
    _require(
        isinstance(native_library, Path)
        and isinstance(qz_receipt, Path)
        and type(qz_receipt_sha256) is str
        and len(qz_receipt_sha256) == 64
        and all(c in "0123456789abcdef" for c in qz_receipt_sha256),
        "incomplete or invalid qz CUPTI allocation inputs",
    )
    native_library = native_library.resolve(strict=True)
    collector = collector.resolve(strict=True)
    _require(
        not native_library.samefile(collector),
        "collector cannot substitute for the qualified native library",
    )
    if native is not None:
        _require(
            isinstance(getattr(native, "_name", None), str)
            and Path(native._name).resolve() == native_library,
            "observer native handle/library mismatch",
        )
    allocation_identity(
        native_library, qz_receipt, qz_receipt_sha256, observer_native=native
    )


def validate_qz(
    receipt: dict[str, Any],
    *,
    uuid: str,
    hostname: str,
    environment: dict[str, str],
    now: float,
) -> dict[str, Any]:
    """Join a bounded, single-instance API observation to this exact container."""
    _require(type(receipt) is dict, "invalid qz receipt")
    _require(receipt.get("schema") == SCHEMA, "unsupported qz allocation schema")
    _require(not environment.get("SLURM_JOB_ID"), "ambiguous qz/Slurm allocation")
    observed = receipt.get("observed_unix_seconds")
    _require(
        type(observed) in (int, float) and 0 <= now - observed <= 7200,
        "stale or future qz observation",
    )
    allocation = receipt.get("allocation", {})
    _require(type(allocation) is dict, "invalid qz allocation")
    for name in (
        "project_id",
        "workspace_id",
        "workload_id",
        "workload_name",
        "instance_id",
        "pod",
        "node",
        "compute_group",
    ):
        value = allocation.get(name)
        _require(type(value) is str and bool(value.strip()), f"missing qz {name}")
    _require(allocation.get("kind") == "job", "unsupported qz workload")
    _require(allocation.get("status") == "RUNNING", "qz allocation is not running")
    _require(
        allocation.get("gpu_count") == 1 and type(allocation["gpu_count"]) is int,
        "qz requires exactly one assigned GPU",
    )
    _require(allocation["pod"] == hostname, "qz allocation/container mismatch")
    _require(allocation.get("gpu_uuid") == uuid, "qz allocation/GPU mismatch")
    _require(
        environment.get("NVIDIA_VISIBLE_DEVICES") == uuid,
        "qz NVIDIA visibility must identify the assigned full GPU",
    )
    visible = environment.get("CUDA_VISIBLE_DEVICES", "")
    _require(visible in ("0", uuid), "qz CUDA visibility mismatch")
    api = receipt.get("api", {})
    _require(type(api) is dict, "invalid qz API observation")
    job, instance = api.get("job", {}), api.get("instance", {})
    _require(type(job) is dict and type(instance) is dict, "invalid qz API identities")
    for field, name in (
        ("project_id", "project_id"),
        ("workspace_id", "workspace_id"),
        ("job_id", "workload_id"),
        ("name", "workload_name"),
        ("logic_compute_group_name", "compute_group"),
        ("gpu_count", "gpu_count"),
    ):
        _require(
            job.get(field) == allocation[name], f"qz API/allocation {name} mismatch"
        )
    _require(
        job.get("status") == "job_running"
        and type(job.get("node_count")) is int
        and job["node_count"] == 1,
        "qz API job must be running on one node",
    )
    _require(
        instance.get("instance_status") == "instance_running",
        "qz instance is not running",
    )
    _require(
        instance.get("name") == allocation["instance_id"] == allocation["pod"],
        "qz instance/pod mismatch",
    )
    _require(instance.get("node") == allocation["node"], "qz instance node mismatch")
    return {
        "platform": "qz",
        "allocation": allocation,
        "observed_unix_seconds": observed,
        "cuda_visible_devices": visible,
        "nvidia_visible_devices": uuid,
    }


def native_uuid(library: Path) -> str:
    """Resolve CUDA ordinal zero through the pinned library, rejecting MIG."""
    native = ctypes.CDLL(str(library.resolve()))
    count = ctypes.c_int()
    native.cudaGetDeviceCount.argtypes = [ctypes.POINTER(ctypes.c_int)]
    native.cudaGetDeviceCount.restype = ctypes.c_int
    _require(
        native.cudaGetDeviceCount(ctypes.byref(count)) == 0 and count.value == 1,
        "expected exactly one visible CUDA device",
    )
    bus = ctypes.create_string_buffer(32)
    native.cudaDeviceGetPCIBusId.argtypes = [
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.c_int,
    ]
    native.cudaDeviceGetPCIBusId.restype = ctypes.c_int
    _require(
        native.cudaDeviceGetPCIBusId(bus, len(bus), 0) == 0 and bool(bus.value),
        "cannot identify visible CUDA device 0",
    )
    fields = (
        subprocess.check_output(
            [
                "nvidia-smi",
                f"--id={bus.value.decode('ascii')}",
                "--query-gpu=uuid,mig.mode.current",
                "--format=csv,noheader",
            ],
            text=True,
            timeout=30,
        )
        .strip()
        .split(",")
    )
    _require(len(fields) == 2, "ambiguous assigned GPU UUID")
    uuid, mig = (field.strip() for field in fields)
    _require(uuid.startswith("GPU-") and "\n" not in uuid, "invalid full GPU UUID")
    _require(mig in ("Disabled", "[N/A]"), "MIG mode is unsupported")
    return uuid


def allocation_identity(
    library: Path,
    receipt_path: Path | None = None,
    receipt_sha256: str | None = None,
    *,
    observer_native: Any = None,
) -> tuple[str, dict[str, Any]]:
    """Preserve the original Slurm contract unless an explicit qz proof is given."""
    if receipt_path is None:
        _require(receipt_sha256 is None, "qz hash without receipt")
        uuid = visible_device_uuid(library)
        return uuid, {
            "platform": "slurm",
            "job_id": os.environ["SLURM_JOB_ID"],
            "cuda_visible_devices": os.environ["CUDA_VISIBLE_DEVICES"],
        }
    receipt_bytes = receipt_path.read_bytes()
    _require(
        receipt_sha256 is not None
        and hashlib.sha256(receipt_bytes).hexdigest() == receipt_sha256,
        "qz receipt hash mismatch",
    )
    uuid = native_uuid(library)
    proof = validate_qz(
        json.loads(receipt_bytes, object_pairs_hook=_unique_object),
        uuid=uuid,
        hostname=socket.gethostname(),
        environment=dict(os.environ),
        now=time.time(),
    )
    proof["receipt_sha256"] = receipt_sha256
    native = ctypes.CDLL(str(library.resolve()))
    if observer_native is not None:
        # Reuse the existing qualification load, rather than loading another
        # DSO solely for comparison. _handle is the dlopen/LoadLibrary handle;
        # _name alone may resolve to a different DSO after a symlink changes.
        _require(
            type(getattr(observer_native, "_handle", None)) is int
            and observer_native._handle == native._handle,
            "observer native handle/library mismatch",
        )
    versions = {}
    for symbol in ("cudaRuntimeGetVersion", "cudaDriverGetVersion"):
        function = getattr(native, symbol)
        function.argtypes = [ctypes.POINTER(ctypes.c_int)]
        function.restype = ctypes.c_int
        value = ctypes.c_int()
        _require(function(ctypes.byref(value)) == 0, f"cannot read {symbol}")
        versions[symbol] = value.value
    _require(
        versions["cudaRuntimeGetVersion"] == 12090
        and versions["cudaDriverGetVersion"] >= 12090,
        "qz qualification requires CUDA 12.9 runtime and compatible driver",
    )
    proof["cuda_versions"] = versions
    proof["execution_environment"] = {
        key: {"present": key in os.environ, "value": os.environ.get(key)}
        for key in ("CUDA_DEVICE_ORDER", "CUBLAS_WORKSPACE_CONFIG")
    }
    return uuid, proof
