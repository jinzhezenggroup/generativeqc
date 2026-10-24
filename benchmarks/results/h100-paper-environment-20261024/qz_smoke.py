"""Run one source-bound setup endpoint under a coordinator-submitted qz Job.

This transport adapter never calls the historical Slurm-only CLI and never
changes its guards or scientific functions. It is not a timing-matrix runner.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.metadata
import json
import math
import numbers
import os
import platform
import socket
import stat
import struct
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

EXPECTED_WORKSPACE = "分布式训练空间"
EXPECTED_PROJECT = "原子级化学反应基座模型2.0"
EXPECTED_GROUP = "开发区-H100-cuda13.2版本-183核"


def digest(path: Path) -> str:
    checksum = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            checksum.update(block)
    return checksum.hexdigest()


def load_json(path: Path) -> Any:
    def reject(value: str) -> None:
        raise ValueError(f"nonfinite JSON constant: {value}")

    return json.loads(path.read_text(encoding="utf-8"), parse_constant=reject)


def checked_file(path: Path, expected: str) -> Path:
    if not isinstance(expected, str) or len(expected) != 64:
        raise ValueError("missing SHA-256 binding")
    path = path.resolve(strict=True)
    if digest(path) != expected:
        raise ValueError(f"artifact hash mismatch: {path.name}")
    return path


def checked_source_tree(root: Path, inventory: Path, expected: str) -> dict:
    inventory = checked_file(inventory, expected)
    files = load_json(inventory)["source_tree"]
    actual = {p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file()}
    if actual != set(files):
        raise ValueError("frozen source tree contains missing or extra files")
    for relative, identity in files.items():
        path = root / relative
        if path.name == ".env" or not path.resolve(strict=True).is_relative_to(root):
            raise ValueError("invalid frozen source inventory path")
        checked_file(path, identity["sha256"])
        if (
            path.stat().st_size != identity["bytes"]
            or oct(stat.S_IMODE(path.stat().st_mode)) != identity["mode"]
        ):
            raise ValueError(
                "frozen source mode/size differs from authenticated inventory"
            )
    return {"inventory_sha256": expected, "verified_files": len(files)}


def persist(path: Path, record: dict) -> None:
    def scalar(value: Any) -> Any:
        if isinstance(value, dict):
            return {k: scalar(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [scalar(v) for v in value]
        if isinstance(value, float) and not math.isfinite(value):
            return {"raw_nonfinite": str(value)}
        if hasattr(value, "tolist"):
            return scalar(value.tolist())
        return value

    temporary = path.with_suffix(".tmp")
    temporary.write_text(
        json.dumps(scalar(record), indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def finite_endpoint(energy: Any, forces: Any, atoms: int) -> None:
    if isinstance(energy, bool) or not isinstance(energy, numbers.Real):
        raise TypeError("endpoint energy is not a scalar number")
    if not math.isfinite(energy):
        raise ValueError("nonfinite endpoint energy")
    if hasattr(forces, "tolist"):
        forces = forces.tolist()
    if not isinstance(forces, (list, tuple)) or len(forces) != atoms:
        raise ValueError("missing complete host force vector")
    for row in forces:
        if not isinstance(row, (list, tuple)) or len(row) != 3:
            raise ValueError("invalid host force shape")
        if any(
            isinstance(x, bool)
            or not isinstance(x, numbers.Real)
            or not math.isfinite(x)
            for x in row
        ):
            raise ValueError("nonfinite host force")


def retain_numeric_bytes(output: Path, sample: dict, index: int) -> dict:
    """Preserve returned IEEE values before JSON tagging or numerical gates.

    Python floats are stored as little-endian binary64. NumPy arrays/scalars
    retain their dtype, shape and C-order bytes, including NaN payloads and -0.
    The index records field paths and offsets; no pickle or object dtype is used.
    """
    path = output / f"values-{index:03d}.bin"
    entries = []
    with path.open("xb") as stream:

        def walk(value: Any, field: list) -> None:
            if hasattr(value, "dtype") and hasattr(value, "tobytes"):
                if value.dtype.hasobject:
                    raise TypeError("object array cannot be retained as numeric bytes")
                payload = value.tobytes(order="C")
                dtype, shape = value.dtype.str, list(value.shape)
            elif isinstance(value, float):
                payload, dtype, shape = struct.pack("<d", value), "<f8", []
            elif isinstance(value, dict):
                for key, item in value.items():
                    walk(item, [*field, key])
                return
            elif isinstance(value, (list, tuple)):
                for key, item in enumerate(value):
                    walk(item, [*field, key])
                return
            else:
                return
            entries.append(
                {
                    "field": field,
                    "offset": stream.tell(),
                    "bytes": len(payload),
                    "dtype": dtype,
                    "shape": shape,
                }
            )
            stream.write(payload)

        walk(sample, [])
    index_path = path.with_suffix(".index.json")
    with index_path.open("x", encoding="utf-8") as stream:
        json.dump(entries, stream, indent=2, allow_nan=False)
        stream.write("\n")
    return {
        "values_file": path.name,
        "values_sha256": digest(path),
        "index_file": index_path.name,
        "index_sha256": digest(index_path),
    }


def device_observation() -> dict:
    import cupy as cp

    count = cp.cuda.runtime.getDeviceCount()
    if count != 1:
        raise RuntimeError(f"setup requires exactly one visible GPU; observed {count}")
    props = cp.cuda.runtime.getDeviceProperties(0)
    name = props["name"]
    name = name.decode() if isinstance(name, bytes) else name
    capability = [int(props["major"]), int(props["minor"])]
    if "H100" not in name or capability != [9, 0]:
        raise RuntimeError(f"unexpected allocated target: {name}, {capability}")
    # Keep scheduler-assigned visibility unchanged. No Slurm fields are invented.
    query = subprocess.run(
        [
            "nvidia-smi",
            "--id=" + cp.cuda.Device(0).pci_bus_id,
            "--query-gpu=name,uuid,pci.bus_id,memory.total,driver_version,power.limit",
            "--format=csv,noheader",
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=15,
    )
    return {
        "hostname": socket.gethostname(),
        "visible_device_count": count,
        "name": name,
        "compute_capability": capability,
        "memory_bytes": int(props["totalGlobalMem"]),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "nvidia_smi": query.stdout.strip(),
        "cgroup": {
            n: Path("/sys/fs/cgroup", n).read_text().strip()
            for n in ("cpu.max", "memory.max")
        },
    }


def environment_observation() -> dict:
    versions = {}
    for name in (
        "numpy",
        "scipy",
        "pyscf",
        "gpu4pyscf-cuda12x",
        "cupy-cuda12x",
        "gpu4pyscf-libxc-cuda12x",
        "pyscf-dispersion",
        "cutensor-cu12",
        "sccache",
    ):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return {
        "python": sys.version,
        "executable": sys.executable,
        "platform": platform.platform(),
        "packages": versions,
        "thread_controls": {
            name: os.environ.get(name)
            for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")
        },
    }


def reference_backend_observation() -> dict:
    """Read the already imported 1.8.1 selector without setting any backend.

    In this pinned version, a None selector with an available cutensor module
    means the native cuTENSOR implementation. Missing module/fields remain
    unavailable; importing GPU4PySCF here would itself alter failure evidence.
    """
    module = sys.modules.get("gpu4pyscf.lib.cutensor")
    if module is None:
        return {"status": "UNAVAILABLE", "engine": None, "reason": "module not loaded"}
    missing = object()
    selector = getattr(module, "contract_engine", missing)
    if selector is None and getattr(module, "cutensor", None) is not None:
        engine = "cutensor"
    elif isinstance(selector, str) and selector in ("cupy", "opt_einsum", "cuquantum"):
        engine = selector
    else:
        return {
            "status": "UNAVAILABLE",
            "engine": None,
            "reason": "selector not observed",
        }
    path = Path(module.__file__) if getattr(module, "__file__", None) else None
    return {
        "status": "OBSERVED",
        "engine": engine,
        "selector": selector,
        "module_path": str(path) if path else None,
        "module_sha256": digest(path)
        if path and path.is_file() and path.suffix in {".py", ".so"}
        else None,
        "backend_override_applied_by_adapter": False,
    }


def call_hf(
    driver: Any, arm: str, atoms: list, inputs: Path, record: dict, retain: Any
) -> None:
    args = SimpleNamespace(
        arm="standard" if arm == "native-hf" else "gpu4pyscf",
        method="hf",
        mode="direct",
        warm_repeats=0,
        inputs=inputs,
    )
    function = driver.run_native if arm == "native-hf" else driver.run_reference
    function(args, atoms, record, retain)


def bind_allocated_instance(allocation: dict, job_name: str, device: dict) -> dict:
    """Bind actual CLI rank/node to the unique hash-bound runtime preflight.

    The supported CLI exposes no opaque instance ID or hostname. Rank is a
    locator within this Job, never an invented platform ID. Hostname comes only
    from the actual preflight and must match the current process container.
    """
    if allocation.get("schema") != "generativeqc.h100-environment-allocation.v2":
        raise ValueError("explicit missing-platform-ID proof contract v2 required")
    status = allocation["platform_status"]
    instances = allocation["platform_instances"]
    raw_status = allocation["raw_status_response"]
    raw_instances = allocation["raw_instances_response"]
    if (
        raw_status.get("success") is not True
        or raw_instances.get("success") is not True
        or raw_status.get("data") != status
        or raw_instances.get("data") != instances
    ):
        raise ValueError(
            "both untouched platform envelopes must report success and match extracted data"
        )
    context = allocation["query_context"]
    if (
        context.get("workspace") != EXPECTED_WORKSPACE
        or context.get("account_alias") != "qz"
        or context.get("job_name") != job_name
        or context.get("scope_source") != "root_cli_query_arguments"
        or context.get("status_sha256") != allocation.get("original_status_sha256")
        or context.get("instances_sha256")
        != allocation.get("original_instances_sha256")
    ):
        raise ValueError(
            "workspace/account/Job query context does not bind the original response hashes"
        )
    if (
        status.get("project") != EXPECTED_PROJECT
        or status.get("compute_group") != EXPECTED_GROUP
    ):
        raise ValueError(
            "actual project or compute group differs from root's selected resource"
        )
    if (
        allocation.get("platform_job_uid") is not None
        or allocation.get("platform_job_uid_status") != "UNAVAILABLE"
    ):
        raise ValueError("unexposed platform Job UID must remain UNAVAILABLE")
    if (
        status.get("name") != job_name
        or status.get("status") != "RUNNING"
        or instances.get("name") != job_name
    ):
        raise ValueError("platform status/instances belong to a different Job")
    resource = status.get("resource", {})
    if (
        any(
            type(resource.get(k)) is not int or resource[k] != v
            for k, v in {
                "gpu": 1,
                "nodes": 1,
                "cpu": 20,
                "memory_gib": 200,
                "shared_memory_gib": 32,
            }.items()
        )
        or "H100" not in resource.get("gpu_type", "")
        or "80GB" not in resource.get("gpu_type", "").replace(" ", "")
    ):
        raise ValueError(
            "actual platform allocation must be one GPU/one node with CPU and memory recorded"
        )
    rows = instances.get("items", [])
    normalized = allocation.get("instances", [])
    if len(rows) != 1 or len(normalized) != 1:
        raise ValueError("rank/node/preflight association must be unique")
    raw, row = rows[0], normalized[0]
    if (
        type(raw.get("rank")) is not int
        or raw["rank"] != 0
        or raw.get("status") != "instance_running"
        or raw.get("type") != "instance_type_worker"
    ):
        raise ValueError("actual CLI row is not the single running rank-0 worker")
    node = raw.get("node")
    if (
        not isinstance(node, str)
        or not node
        or status.get("nodes") != [node]
        or row.get("node") != node
        or row.get("rank") != 0
    ):
        raise ValueError("actual Job/node/rank association differs")
    if any(
        raw.get(k) is not None for k in ("id", "instance_id", "hostname", "pod_uid")
    ):
        raise ValueError(
            "platform schema changed; explicit proof-contract review required"
        )
    if (
        row.get("id") is not None
        or row.get("platform_instance_id") is not None
        or row.get("platform_instance_id_status") != "UNAVAILABLE"
    ):
        raise ValueError(
            "opaque platform instance ID must remain explicitly UNAVAILABLE"
        )
    observation = allocation["runtime_preflight"]
    path = checked_file(Path(observation["path"]), observation["sha256"])
    preflight = load_json(path)
    if preflight.get("schema") not in {
        "generativeqc.h100-environment-gate-preflight.v1",
        "generativeqc.h100-environment-gate-preflight.v2",
    }:
        raise ValueError("wrong runtime preflight schema")
    if preflight.get("job_name") != job_name:
        raise ValueError("runtime preflight identifies another Job")
    boot = preflight.get("device", {})
    if (
        preflight.get("endpoints_started") != 0
        or preflight.get("wait_cap_seconds") != 300
        or boot.get("visible_device_count") != 1
        or boot.get("compute_capability") != [9, 0]
        or "H100" not in boot.get("name", "")
        or "80GB" not in boot.get("name", "").replace(" ", "")
        or not (75 * (1 << 30) <= boot.get("memory_bytes", 0) <= 85 * (1 << 30))
    ):
        raise ValueError(
            "preflight must be zero-endpoint, one H10080/SM90 with the fixed 300s gate"
        )
    hostname = preflight.get("device", {}).get("hostname")
    if (
        not isinstance(hostname, str)
        or not hostname
        or row.get("hostname_source") != "runtime_preflight"
        or row.get("hostname") != hostname
        or device.get("hostname") != hostname
    ):
        raise RuntimeError(
            "runtime hostname does not bind to the unique preflight and rank/node"
        )
    return {
        "job_name": job_name,
        "rank": 0,
        "node": node,
        "platform_type": raw["type"],
        "platform_status": raw["status"],
        "platform_instance_id": None,
        "platform_instance_id_status": "UNAVAILABLE",
        "runtime_hostname": hostname,
        "runtime_hostname_source": "hash_bound_gate_preflight",
        "runtime_preflight_sha256": observation["sha256"],
    }


def execute(plan: dict, arm: str, output: Path) -> dict:
    """Validate bound inputs before importing a frozen source or loading a DSO."""
    if plan.get("schema") != "generativeqc.h100-environment-smoke-plan.v1":
        raise ValueError("wrong setup-plan schema")
    if arm not in {"native-hf", "gpu4pyscf-hf", "cbs-c2"}:
        raise ValueError("unknown setup arm")
    if (
        not plan.get("expected_job_name")
        or plan.get("job_name") != plan["expected_job_name"]
    ):
        raise ValueError("actual Job name differs from frozen expected Job name")
    source = plan["sources"]["cbs" if arm == "cbs-c2" else "hf_dft"]
    root = Path(source["root"]).resolve(strict=True)
    source_driver = checked_file(root / source["driver"], source["driver_sha256"])
    source_tree = checked_source_tree(
        root, Path(source["inventory"]), source["inventory_sha256"]
    )
    checked_file(Path(source["library"]), source["library_sha256"])
    checked_file(Path(plan["protocol_manifest"]), plan["protocol_manifest_sha256"])
    proof = checked_file(
        Path(plan["allocation_proof"]), plan["allocation_proof_sha256"]
    )
    allocation = load_json(proof)
    if allocation.get("name") != plan["job_name"]:
        raise ValueError("qz allocation identity differs from submitted Job")
    if allocation.get("gpu_count") != 1 or allocation.get("provider") != "qz":
        raise ValueError("missing authoritative qz single-GPU allocation proof")
    if (
        allocation.get("status") != "RUNNING"
        or not allocation.get("observed_utc")
        or not allocation.get("platform_status")
        or not allocation.get("instances")
        or not allocation.get("resources")
    ):
        raise ValueError("incomplete live qz status/instances/allocation evidence")
    environment = environment_observation()
    expected_packages = plan["expected_packages"]
    if not expected_packages or any(
        name not in environment["packages"] or environment["packages"][name] != version
        for name, version in expected_packages.items()
    ):
        raise ValueError("runtime packages differ from the frozen environment plan")
    # Exclusive directory claims preserve all earlier attempts and partial output.
    output.mkdir(parents=True, exist_ok=False)
    receipt = output / "raw.json"
    record = {
        "schema": "generativeqc.h100-environment-smoke.v1",
        "arm": arm,
        "status": "preflight",
        "scientific_acceptance": "INCOMPLETE",
        "timing_use": "setup qualification only; excluded from performance cohorts",
        "source": source,
        "source_tree": source_tree,
        "source_driver_sha256": digest(source_driver),
        "adapter_sha256": digest(Path(__file__)),
        "allocation_proof_sha256": digest(proof),
        "allocation": allocation,
        "environment": environment,
        "samples": [],
        "numeric_originals": [],
    }
    persist(receipt, record)
    try:
        record["device"] = device_observation()
        record["allocated_instance"] = bind_allocated_instance(
            allocation, plan["job_name"], record["device"]
        )
        os.environ["GENERATIVEQC_LIBRARY"] = source["library"]
        sys.path[:0] = [str(root), str(root / "python")]
        module = (
            "benchmarks.readme_cbs" if arm == "cbs-c2" else "benchmarks.readme_defaults"
        )
        driver = importlib.import_module(module)
        if Path(driver.__file__).resolve() != source_driver:
            raise RuntimeError("imported benchmark driver is outside the frozen source")
        record["status"] = "running"
        persist(receipt, record)

        def retain(sample: dict) -> None:
            record["numeric_originals"].append(
                retain_numeric_bytes(output, sample, len(record["samples"]))
            )
            record["samples"].append(sample)
            # Preserve returned data before replay/acceptance checks.
            persist(receipt, record)

        if arm == "cbs-c2":
            dataset = checked_file(Path(plan["qm7"]), plan["qm7_sha256"])
            atoms = driver.geometry(dataset, 2)
            budget = 32768 << 20
            protocol, orbital, auxiliary = driver.setup(atoms, 2, budget, "df")
            record["protocol"] = protocol
            persist(receipt, record)
            sample = driver.native_call(
                atoms, orbital, auxiliary, budget, "df", "standard"
            )
            retain(sample)
            finite_endpoint(sample["energy"], sample["forces"], len(atoms))
            if not sample.get("converged") or sample.get("executed_backend") != "cuda":
                raise RuntimeError("CBS did not return a converged CUDA endpoint")
        else:
            from benchmarks.readme_hf_scaling import scaling_cases

            atoms = scaling_cases()["water-3"].atoms
            record["geometry_bohr"] = atoms
            call_hf(driver, arm, atoms, Path(plan["hf_inputs"]), record, retain)
            if len(record["samples"]) != 1:
                raise RuntimeError("tiny HF setup returned an unexpected sample count")
            sample = record["samples"][0]
            finite_endpoint(sample["energy"], sample["forces"], len(atoms))
        record["status"] = "FINITE_ENDPOINT_RETURNED"
    except BaseException as error:
        record.update(status="FAILED", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        if arm == "gpu4pyscf-hf":
            record["reference_contraction_backend"] = reference_backend_observation()
        persist(receipt, record)
    return record


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--plan-sha256", required=True)
    parser.add_argument(
        "--arm", choices=("native-hf", "gpu4pyscf-hf", "cbs-c2"), required=True
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    checked_file(args.plan, args.plan_sha256)
    execute(load_json(args.plan), args.arm, args.output)


if __name__ == "__main__":
    main()
