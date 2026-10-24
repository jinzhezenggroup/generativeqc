"""Fresh-process Standard E+F cache-state experiment for issue #2680.

The controller never imports the production runtime. Each worker has one private
working directory and a whitelisted environment. Observation forwards compiler
and cache calls to their original functions without changing their results.
NVCC driver invocation counts are not internal compiler-pass counts or CPU time.
"""

from __future__ import annotations

import argparse
import dataclasses
import functools
import hashlib
import json
import math
import os
import signal
import subprocess
import sys
import time
from pathlib import Path, PurePosixPath
from types import SimpleNamespace
from typing import Any

from benchmarks._retention import raw_output_path

SCHEMA = "generativeqc.runtime-cache-state.v1"
CASES = (
    "water-12-pbe0",
    "water-12-wb97m-v",
    "water-48-pbe0",
    "water-48-wb97m-v",
    "cbs-c2",
)
ROOT = Path(__file__).resolve().parents[1]
SAFE_ENV = (
    "PATH",
    "LD_LIBRARY_PATH",
    "CUDA_HOME",
    "CUDA_PATH",
    "CUDA_VISIBLE_DEVICES",
    "CUDA_DEVICE_ORDER",
    "SystemRoot",
    "WINDIR",
    "COMSPEC",
    "PATHEXT",
    "SLURM_JOB_ID",
)
THREAD_ENV = {
    "OMP_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
}


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            value.update(block)
    return value.hexdigest()


def identity(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
    ).hexdigest()


def serial(value: Any) -> Any:
    if dataclasses.is_dataclass(value):
        return dataclasses.asdict(value)
    if hasattr(value, "tolist"):
        return value.tolist()
    raise TypeError(type(value).__name__)


def save(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, allow_nan=False, default=serial) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    temporary.replace(path)


def census(root: Path) -> list[dict]:
    """Bind exact private cache contents, rejecting links escaping their owner."""
    result = []
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise ValueError("cache census refuses symbolic links")
        if path.is_file():
            result.append(
                {
                    "path": path.relative_to(root).as_posix(),
                    "bytes": path.stat().st_size,
                    "sha256": digest(path),
                    "mode": f"{path.stat().st_mode & 0o777:04o}",
                }
            )
    return result


def cache_environment(
    root: Path, config: dict, inherited: dict[str, str]
) -> dict[str, str]:
    """Do not inherit compiler remote backends, flags, credentials or overrides."""
    environment = {key: inherited[key] for key in SAFE_ENV if key in inherited}
    environment.update(THREAD_ENV)
    environment.update(
        {
            "HOME": str(root / "home"),
            "XDG_CACHE_HOME": str(root / "xdg-cache"),
            "XDG_CONFIG_HOME": str(root / "xdg-config"),
            "TMPDIR": str(root / "tmp"),
            "TMP": str(root / "tmp"),
            "TEMP": str(root / "tmp"),
            "GENERATIVEQC_STATIONARY_CACHE": str(root / "artifact"),
            "GENERATIVEQC_TENSOR_CACHE": str(root / "tensor"),
            "GENERATIVEQC_PROFILE_CACHE": str(root / "profiles"),
            "GENERATIVEQC_JIT_SCCACHE_ROOT": str(root / "jit-slots"),
            "SCCACHE_DIR": str(root / "compiler"),
            "SCCACHE_CONF": str(root / "sccache.toml"),
            "SCCACHE_CACHED_CONF": str(root / "sccache-cached.toml"),
            "CCACHE_DIR": str(root / "ccache"),
            "CCACHE_CONFIGPATH": str(root / "ccache.conf"),
            "CUDA_CACHE_PATH": str(root / "cuda-driver"),
            "CUDA_CACHE_MAXSIZE": str(1 << 30),
            "CUPY_CACHE_DIR": str(root / "cupy"),
            "PYTHONDONTWRITEBYTECODE": "1",
            "GENERATIVEQC_LIBRARY": config["library"],
            # Only the frozen production Python is on the production search path.
            "PYTHONPATH": os.pathsep.join([*config["python_paths"], str(ROOT)]),
        }
    )
    return environment


def new_cache(root: Path) -> None:
    root.mkdir(parents=True, exist_ok=False)
    for name in ("home", "xdg-cache", "xdg-config", "tmp", "work", "cuda-driver"):
        (root / name).mkdir()
    # An empty explicit config prevents default local/remote config discovery.
    (root / "sccache.toml").write_text("", encoding="utf-8")
    (root / "ccache.conf").write_text("remote_storage =\n", encoding="utf-8")


def mark_primed(root: Path, endpoint_sha256: str) -> None:
    """Bind the last completed endpoint to exact reusable cache bytes."""
    contents = [row for row in census(root) if row["path"] != "priming.json"]
    save(
        root / "priming.json",
        {"endpoint_sha256": endpoint_sha256, "contents_sha256": identity(contents)},
    )


def check_primed(root: Path) -> dict:
    """Reject altered/partial caches instead of silently re-priming them."""
    marker = root / "priming.json"
    if not marker.is_file():
        raise ValueError("missing primed cache receipt")
    record = json.loads(marker.read_text(encoding="utf-8"))
    contents = [row for row in census(root) if row["path"] != "priming.json"]
    if record.get("contents_sha256") != identity(contents):
        raise ValueError("primed cache contents changed outside a completed endpoint")
    return record


def schedule(cases: tuple[str, ...] = CASES, repeats: int = 3) -> list[dict]:
    """Fixed balanced order; priming calls are never counted as samples."""
    if not cases or len(set(cases)) != len(cases) or any(c not in CASES for c in cases):
        raise ValueError("unknown or duplicate case")
    if repeats < 1:
        raise ValueError("repeats must be positive")
    rows = [
        {"case": case, "state": "prime", "replicate": None, "cache": f"{case}/primed"}
        for case in cases
    ]
    for repeat in range(repeats):
        for index, case in enumerate(cases):
            states = (
                ("empty", "primed")
                if (repeat + index) % 2 == 0
                else ("primed", "empty")
            )
            for state in states:
                rows.append(
                    {
                        "case": case,
                        "state": state,
                        "replicate": repeat,
                        "cache": f"{case}/empty-{repeat}"
                        if state == "empty"
                        else f"{case}/primed",
                    }
                )
    return rows


def validate_config(config: dict) -> None:
    """A separately audited SDK receipt must bind every production input byte."""
    if config.get("schema") != SCHEMA or config.get("policy") != "standard":
        raise ValueError("requires a Standard frozen experiment config")
    revision = config.get("production_source_revision", "")
    if len(revision) != 40 or any(c not in "0123456789abcdef" for c in revision):
        raise ValueError("missing production source revision")
    if config.get("install_kind") not in (
        "installed-sdk",
        "installed-wheel",
        "build-tree",
    ):
        raise ValueError("missing installation provenance")
    if not config.get("sdk_files") or not config.get("sdk_provenance"):
        raise ValueError("requires full Python/native/AOT SDK manifest and provenance")
    files = config["sdk_files"]
    if len({item["path"] for item in files}) != len(files):
        raise ValueError("duplicate SDK file")
    for item in files:
        path = Path(item["path"])
        if (
            not path.is_absolute()
            or ".env" in path.parts
            or path.name.startswith(".env.")
        ):
            raise ValueError("unsafe SDK inventory path")
        if path.stat().st_size != item["bytes"] or digest(path) != item["sha256"]:
            raise ValueError(f"SDK byte mismatch: {path.name}")
        if "mode" in item and f"{path.stat().st_mode & 0o777:04o}" != item["mode"]:
            raise ValueError(f"SDK mode mismatch: {path.name}")
    for item in config.get("input_files", []):
        path = Path(item["path"])
        if ".env" in path.parts or path.name.startswith(".env."):
            raise ValueError("unsafe scientific input path")
        if path.stat().st_size != item["bytes"] or digest(path) != item["sha256"]:
            raise ValueError(f"scientific input byte mismatch: {path.name}")
    if str(Path(config["library"]).resolve()) not in {
        str(Path(i["path"]).resolve()) for i in files
    }:
        raise ValueError("loaded library absent from SDK inventory")
    environment = config["scientific_environment"]
    manifest = Path(environment["manifest"]["path"])
    if digest(manifest) != environment["manifest"]["sha256"]:
        raise ValueError("scientific environment manifest differs")
    if (
        digest(Path(sys.executable).resolve())
        != environment["python_executable_sha256"]
    ):
        raise ValueError("interpreter differs from frozen scientific environment")


def verify_imported_modules(config: dict) -> list[dict]:
    """Reject production Python imports outside the frozen SDK inventory."""
    inventory = {item["path"]: item["sha256"] for item in config["sdk_files"]}
    imported = []
    for name, module in tuple(sys.modules.items()):
        if name.split(".")[0] not in ("generativeqc", "generativeqc_compiler"):
            continue
        location = getattr(module, "__file__", None)
        if location is None:
            continue
        path = Path(location).resolve()
        if inventory.get(str(path)) != digest(path):
            raise ValueError(f"production Python module outside frozen SDK: {name}")
        imported.append(
            {"module": name, "path": str(path), "sha256": inventory[str(path)]}
        )
    return imported


def water_gate(record: dict, config: dict) -> dict:
    """Replay the original per-sample, grid-stability and CPU/GPU quarter gates."""
    from benchmarks.readme_defaults_qualification import (
        STABILITY_FRACTION,
        _comparison,
        _geometry,
        _oracle,
        _reference_contract,
    )

    endpoint = record["endpoint"]
    method, atoms = endpoint["method"], endpoint["atoms"]
    indexed = {}
    for oracle in config["water_references"]:
        _oracle(oracle)
        key = (oracle["method"], oracle["atoms"], oracle["backend"])
        if key in indexed:
            raise ValueError("duplicate water reference")
        indexed[key] = oracle
    wanted = [(method, atoms, "cuda"), (method, 3, "cpu"), (method, 3, "cuda")]
    if any(
        key not in indexed or indexed[key]["status"] != "complete" for key in wanted
    ):
        return {"status": "INCOMPLETE", "reason": "missing independent reference"}
    target, cpu, gpu = (indexed[key] for key in wanted)
    scientific = {
        "method": method,
        "orbital_basis": "def2-SVP",
        "basis_representation": "spherical",
        "charge": 0,
        "spin": 0,
    }
    measurement = {
        **scientific,
        "atoms": atoms,
        "geometry_bohr": endpoint["geometry_bohr"],
    }
    if _geometry(measurement) != _geometry(target) or _geometry(cpu) != _geometry(gpu):
        raise ValueError("water reference geometry differs")
    for oracle in (target, cpu, gpu):
        if config["reference_reuse"]["entries"].get(
            identity(oracle)
        ) != _reference_contract(scientific, oracle):
            raise ValueError("missing audited water reference reuse contract")
        if record["environment"]["packages"] != oracle["environment"]["packages"]:
            raise ValueError("reference package versions differ")
    stability = [
        _comparison(*oracle["samples"], oracle["atoms"], fraction=STABILITY_FRACTION)
        for oracle in (target, gpu)
    ]
    spots = [
        sample
        for sample in gpu["samples"]
        if sample["level"] == cpu["samples"][0]["level"]
        and (sample.get("nonlocal_grid") or {}).get("level")
        == (cpu["samples"][0].get("nonlocal_grid") or {}).get("level")
    ]
    if len(spots) != 1:
        raise ValueError("CPU/GPU spot quadratures differ")
    spot = _comparison(cpu["samples"][0], spots[0], 3, fraction=STABILITY_FRACTION)
    accuracy = _comparison(endpoint["sample"], target["samples"][-1], atoms)
    return {
        "status": "PASS"
        if all(c["passed"] for c in [*stability, spot, accuracy])
        else "FAIL",
        "reference_sha256": [identity(r) for r in (target, cpu, gpu)],
        "stability": stability,
        "spot": spot,
        "accuracy": accuracy,
    }


def cbs_direction(reference: dict) -> dict:
    """Audit normalized seed identity; all projections still use receipt bytes.

    With n nonnegative squared terms, gamma_n = n*u/(1-n*u) bounds
    summation roundoff. Two norm implementations can each incur that error;
    the square roots halve it. Four additional unit roundoffs cover square,
    root and division rounding. This is a component-relative identity bound,
    never an allowance on actual coordinates, hashes or force observables.
    """
    import numpy as np

    fields = {
        "schema",
        "engine",
        "protocol",
        "source_revision",
        "driver_sha256",
        "status",
        "stage",
        "requested_samples",
        "samples",
        "slurm_job_id",
        "environment",
        "controls",
        "direction",
        "derivatives",
        "energies",
    }
    if set(reference) != fields:
        raise ValueError("unknown or missing independent CBS reference fields")
    shape = (len(reference["protocol"]["atoms_bohr"]), 3)
    expected = np.random.default_rng(709).normal(size=shape)
    expected /= np.linalg.norm(expected)
    actual = np.asarray(reference["direction"])
    unit_roundoff = np.finfo(float).eps / 2
    gamma = expected.size * unit_roundoff / (1 - expected.size * unit_roundoff)
    relative_bound = gamma + 4 * unit_roundoff
    if (
        actual.shape != shape
        or not np.isfinite(actual).all()
        or np.any(np.signbit(actual) != np.signbit(expected))
        or np.any(np.abs(actual - expected) > relative_bound * np.abs(expected))
        or abs(np.linalg.norm(actual) - 1) > relative_bound
    ):
        raise ValueError("independent CBS direction differs from normalized seed 709")
    return {
        "source_of_actual_direction": "unchanged independent receipt",
        "normalization_relative_roundoff_bound": float(relative_bound),
        "maximum_absolute_regeneration_difference": float(
            np.max(np.abs(actual - expected))
        ),
    }


def cbs_gate(record: dict, config: dict) -> dict:
    """Retain independent constituent/composite energy and directional scope."""
    from tools.render_readme_cbs import qualification

    sample = record["endpoint"]["sample"]
    reference = config["cbs_reference"]
    if sample.get("convergence", {}).get("budget") is not None:
        raise ValueError("Standard qualification cannot adopt a Fast budget")
    direction = cbs_direction(reference)
    if identity(reference) != config["cbs_reference_sha256"]:
        raise ValueError("independent CBS reference receipt identity differs")
    result = qualification(sample, record["endpoint"]["protocol"], reference)
    return {
        "status": "PASS"
        if result["execution_gate"] and result["accuracy_gate"]
        else "FAIL",
        "scope": "constituent/composite energies and two directional steps; not full-vector qualification",
        "reference_sha256": identity(reference),
        "direction_identity": direction,
        "checks": result,
    }


def recorded_path(value: str) -> PurePosixPath:
    """Interpret Linux measurement paths without consulting the replay host."""
    path = PurePosixPath(value)
    if not path.is_absolute() or str(path) != value or ".." in path.parts:
        raise ValueError("noncanonical recorded POSIX path")
    return path


def worker_config(config: dict) -> dict:
    """Keep independent oracle numbers in the offline controller only."""
    excluded = {
        "water_references",
        "reference_reuse",
        "cbs_reference",
        "cbs_reference_sha256",
    }
    return {key: value for key, value in config.items() if key not in excluded}


def cache_gate(folder: Path, row: dict, record: dict) -> dict:
    """Reconcile initial emptiness/priming and actual cache-backed subprocesses."""
    request_path = folder / "request.json"
    request = json.loads(request_path.read_text(encoding="utf-8"))
    if digest(request_path) != record["request_sha256"]:
        raise ValueError("endpoint request identity differs")
    before_path = folder / "cache-before.json"
    if digest(before_path) != request["cache_before_sha256"]:
        raise ValueError("initial cache census identity differs")
    before = json.loads(before_path.read_text(encoding="utf-8"))
    if row["state"] in ("empty", "prime"):
        if {item["path"] for item in before} != {"sccache.toml", "ccache.conf"}:
            raise ValueError("empty cache contains prior payloads")
    else:
        contents = [item for item in before if item["path"] != "priming.json"]
        if request["priming_receipt"]["contents_sha256"] != identity(contents):
            raise ValueError("primed cache differs from completed previous endpoint")
    processes = [
        event
        for event in record["observation"]["events"]
        if event["kind"] == "compiler-process" and event["phase"] != "probe"
    ]
    unavailable = []
    owned = recorded_path(request["cache_root"])
    if owned != recorded_path(request["output_root"]) / "caches" / row["cache"]:
        raise ValueError("cache root differs from original campaign")
    for event in record["observation"]["events"]:
        if (
            event["kind"] == "artifact"
            and "path" in event
            and not recorded_path(event["path"]).is_relative_to(owned)
        ):
            raise ValueError("runtime artifact escaped private cache root")
        if event["kind"] == "packaged-aot" and "path" in event:
            installed = {item["path"]: item for item in request["config"]["sdk_files"]}
            item = installed.get(str(recorded_path(event["path"])))
            metadata = event["artifact_metadata"]
            if (
                not item
                or item["sha256"] != metadata["binary_sha256"]
                or item["bytes"] != metadata["binary_bytes"]
            ):
                raise ValueError("observed packaged AOT differs from frozen SDK")
    for event in processes:
        if Path(event["command"][0]).name not in ("sccache", "ccache"):
            raise ValueError(
                "observed compiler/link process bypassed verified cache launcher"
            )
        if set(event["cache_paths"]) != {
            "SCCACHE_DIR",
            "SCCACHE_CONF",
            "SCCACHE_CACHED_CONF",
            "CCACHE_DIR",
            "CCACHE_CONFIGPATH",
        }:
            raise ValueError("incomplete compiler cache path observation")
        for path in event["cache_paths"].values():
            if path is None or not recorded_path(path).is_relative_to(owned):
                raise ValueError("compiler cache configuration escaped private root")
        for phase in ("before", "after"):
            stats = event.get("stats_" + phase)
            available = bool(
                stats and stats.get("returncode") == 0 and stats.get("stdout")
            )
            if available and Path(event["command"][0]).name == "sccache":
                try:
                    parsed = json.loads(stats["stdout"])
                    counters = parsed["stats"]
                    available = (
                        type(counters["compile_requests"]) is int
                        and counters["compile_requests"] >= 0
                        and isinstance(counters["cache_hits"]["counts"], dict)
                        and isinstance(counters["cache_misses"]["counts"], dict)
                        and parsed["cache_location"].startswith("Local disk:")
                        and str(owned) in parsed["cache_location"]
                    )
                except (ValueError, KeyError, TypeError, AttributeError):
                    available = False
            if not available:
                unavailable.append({"command": event["command"], "phase": phase})
    return {
        "status": "INCOMPLETE" if unavailable else "PASS",
        "scoped_compiler_processes": len(processes),
        "missing_stats": unavailable,
        "native_nvrtc_calls": "UNAVAILABLE",
    }


def qualify_campaign(folder: Path) -> dict:
    """Recompute from every retained process, including priming and failures."""
    campaign_path = folder / "campaign.json"
    campaign = json.loads(campaign_path.read_text(encoding="utf-8"))
    config = campaign["config"]
    if campaign["cohort"] not in ("pilot", "formal"):
        raise ValueError("unknown measurement cohort")
    expected = (
        schedule(("water-12-wb97m-v",), 1)
        if campaign["cohort"] == "pilot"
        else schedule()
    )
    if campaign["schedule"] != expected:
        raise ValueError("cache-state population differs from requested matrix")
    checks = []
    devices = set()
    pids = set()
    prior_cache_endpoint = {}
    for index, row in enumerate(campaign["results"]):
        if (
            index >= len(expected)
            or {key: row[key] for key in expected[index]} != expected[index]
        ):
            raise ValueError("measurement rows differ from scheduled order/population")
        path = folder / row["folder"] / "endpoint.json"
        if (
            not path.exists()
            or digest(path) != row["endpoint_sha256"]
            or not row["complete"]
        ):
            checks.append({"folder": row["folder"], "status": "INCOMPLETE"})
            continue
        record = json.loads(path.read_text(encoding="utf-8"))
        request = json.loads((path.parent / "request.json").read_text(encoding="utf-8"))
        process = json.loads((path.parent / "process.json").read_text(encoding="utf-8"))
        if (
            process != row["process"]
            or process["returncode"] != 0
            or process["timed_out"]
            or process["pid"] != record["pid"]
        ):
            raise ValueError("process exit/identity differs from completed endpoint")
        if any(
            request[key] != row[key] for key in ("case", "state", "replicate", "cache")
        ):
            raise ValueError("request differs from scheduled measurement")
        if any(record[key] != request[key] for key in ("case", "state", "replicate")):
            raise ValueError("endpoint differs from requested measurement")
        if (
            identity(request["config"]) != identity(worker_config(config))
            or request["output_root"] != campaign["output_root"]
        ):
            raise ValueError("measurement configuration differs from frozen campaign")
        if row["state"] == "primed" and request["priming_receipt"][
            "endpoint_sha256"
        ] != prior_cache_endpoint.get(row["cache"]):
            raise ValueError(
                "primed cache is not bound to its preceding completed endpoint"
            )
        if row["state"] in ("prime", "primed"):
            prior_cache_endpoint[row["cache"]] = digest(path)
        if record["pid"] in pids:
            raise ValueError("reused endpoint process identity")
        pids.add(record["pid"])
        devices.add(record["physical_gpu_uuid"])
        if record["production_source_revision"] != config["production_source_revision"]:
            raise ValueError("mixed production source revisions")
        if record["driver_sha256"] != campaign["driver_sha256"]:
            raise ValueError("mixed measurement driver revisions")
        build = record["endpoint"]["native_build"]
        if (
            build["source_identity"] != config["native_source_identity"]
            or build["library_sha256"] != config["library_sha256"]
            or build["library_path"] != config["library"]
        ):
            raise ValueError("mixed native SDK identities")
        numerical = (
            cbs_gate(record, config)
            if row["case"] == "cbs-c2"
            else water_gate(record, config)
        )
        cache = cache_gate(folder / row["folder"], row, record)
        gate = {
            "status": "PASS"
            if numerical["status"] == cache["status"] == "PASS"
            else "FAIL"
            if "FAIL" in (numerical["status"], cache["status"])
            else "INCOMPLETE",
            "numerical": numerical,
            "cache": cache,
        }
        checks.append(
            {
                "folder": row["folder"],
                "case": row["case"],
                "state": row["state"],
                **gate,
            }
        )
    complete = (
        len(checks) == len(campaign["schedule"]) and campaign["status"] == "MEASURED"
    )
    passed = (
        complete and len(devices) == 1 and all(c["status"] == "PASS" for c in checks)
    )
    return {
        "schema": SCHEMA,
        "status": "PASS" if passed else "INCOMPLETE" if not complete else "FAIL",
        "cohort": campaign["cohort"],
        "config_sha256": campaign["config_sha256"],
        "campaign_sha256": digest(campaign_path),
        "physical_gpu_uuid": next(iter(devices)) if len(devices) == 1 else None,
        "checks": checks,
    }


class Observer:
    """Observe original adapters, semantic reuse, subprocesses and scoped stats.

    Stats calls use the exact live private server environment, before it is
    reaped by the unmodified runtime. Their wall overhead remains in endpoint
    timings and is reported separately, never subtracted or double-added.
    """

    def __init__(self) -> None:
        from generativeqc_compiler.common import (
            compiler_cache,
            compiler_process,
            native_runtime,
            semantic_source_cache,
        )

        self.originals = {
            compiler_cache.run_cached_compiler: "cached-driver",
            compiler_process.run_compiler: "compiler-process",
            semantic_source_cache.cached_sources: "semantic-source",
            native_runtime.compile_cuda_object: "artifact",
            native_runtime.link_cuda_objects: "artifact",
            native_runtime.compile_runtime: "artifact",
            native_runtime.compile_runtime_bundle: "artifact",
        }
        # Importing scientific plan modules just for observation could warm
        # first-use work. Only hook AOT owners already loaded by the application.
        stationary = sys.modules.get("generativeqc_compiler.method.stationary_cuda")
        self.aot_hooked = stationary is not None
        if stationary is not None:
            self.originals[stationary.load_stationary_aot_artifact] = "packaged-aot"
        self.wrappers: dict[Any, Any] = {}
        self.events: list[dict] = []
        self.stats_wall = 0.0
        self.origin = time.perf_counter()

    def stats(self, command: list[str], environment: dict | None) -> dict | None:
        if (
            not command
            or Path(command[0]).name not in ("sccache", "ccache")
            or any(x.startswith("--") for x in command[1:2])
        ):
            return None
        selected = Path(command[0]).name
        started = time.perf_counter()
        try:
            reply = subprocess.run(
                [
                    command[0],
                    "--show-stats",
                    *(["--stats-format=json"] if selected == "sccache" else []),
                ],
                env=environment,
                capture_output=True,
                text=True,
                timeout=10,
                check=False,
            )
            return {
                "returncode": reply.returncode,
                "stdout": reply.stdout,
                "stderr": reply.stderr,
            }
        except (OSError, subprocess.TimeoutExpired) as error:
            return {"status": "UNAVAILABLE", "error_type": type(error).__name__}
        finally:
            self.stats_wall += time.perf_counter() - started

    def wrap(self, original: Any, kind: str) -> Any:
        @functools.wraps(original)
        def observe(*args: Any, **kwargs: Any) -> Any:
            row = {"kind": kind, "start": time.perf_counter() - self.origin}
            if kind not in ("semantic-source", "artifact", "packaged-aot"):
                command = [str(x) for x in args[0]]
                row.update(command=command, label=kwargs.get("label"))
                row["phase"] = (
                    "probe"
                    if any(
                        x in command
                        for x in (
                            "--version",
                            "--dist-status",
                            "-dumpmachine",
                            "-M",
                            "-MM",
                            "-E",
                        )
                    )
                    else "compile"
                    if "-c" in command
                    else "compile-and-link"
                    if any(x.endswith((".cu", ".cpp", ".cc", ".c")) for x in command)
                    else "link"
                )
                if kind == "compiler-process":
                    env = kwargs.get("environment")
                    row["stats_before"] = self.stats(command, env)
                    # This whitelist contains only task-owned paths, no env dump.
                    row["cache_paths"] = {
                        name: (env or os.environ).get(name)
                        for name in (
                            "SCCACHE_DIR",
                            "SCCACHE_CONF",
                            "SCCACHE_CACHED_CONF",
                            "CCACHE_DIR",
                            "CCACHE_CONFIGPATH",
                        )
                    }
            try:
                result = original(*args, **kwargs)
                if kind == "semantic-source":
                    row["metadata"] = result[1]
                elif kind in ("artifact", "packaged-aot"):
                    row["path"] = str(
                        result.path if hasattr(result, "path") else result.library
                    )
                    # This metadata is historical cache provenance. Its
                    # compile_seconds value is never today's compile wall.
                    row["artifact_metadata"] = result.metadata
                else:
                    row["result"] = dataclasses.asdict(result)
                return result
            except BaseException as error:
                row["error_type"] = type(error).__name__
                raise
            finally:
                row["stop"] = time.perf_counter() - self.origin
                if kind == "compiler-process":
                    row["stats_after"] = self.stats(
                        row["command"], kwargs.get("environment")
                    )
                self.events.append(row)

        return observe

    def __enter__(self) -> Any:
        # Replace aliases too, including those imported before observation.
        # Later imports receive the same forwarding wrappers. This avoids a
        # global profiler charging every Python code-generation call.
        self.wrappers = {
            original: self.wrap(original, kind)
            for original, kind in self.originals.items()
        }
        self.replace_aliases(self.wrappers)
        return self

    def __exit__(self, *unused: object) -> None:
        self.replace_aliases(
            {wrapper: original for original, wrapper in self.wrappers.items()}
        )

    @staticmethod
    def replace_aliases(mapping: dict) -> None:
        for module in tuple(sys.modules.values()):
            if module is None:
                continue
            for name, value in tuple(vars(module).items()):
                for original, replacement in mapping.items():
                    if value is original:
                        setattr(module, name, replacement)

    def receipt(self) -> dict:
        drivers = [e for e in self.events if e["kind"] == "cached-driver"]
        probes = [
            e
            for e in self.events
            if e["kind"] == "compiler-process" and e["phase"] == "probe"
        ]
        return {
            "coverage": "Python cached-driver/compiler-process/semantic-source/native-runtime artifact function aliases",
            "native_nvrtc_calls": "UNAVAILABLE",
            "native_embedded_aot_selections": "UNAVAILABLE",
            "observed_python_aot_loads": sum(
                e["kind"] == "packaged-aot" for e in self.events
            )
            if self.aot_hooked
            else "UNAVAILABLE",
            "internal_compiler_passes": "UNAVAILABLE",
            "cached_driver_invocations": len(drivers),
            "cached_driver_wall_seconds": sum(e["stop"] - e["start"] for e in drivers),
            "identity_dependency_and_cache_status_probes": len(probes),
            "probe_wall_seconds": sum(e["stop"] - e["start"] for e in probes),
            "stats_observer_wall_seconds": self.stats_wall,
            "events": self.events,
        }


def water_endpoint(case: str) -> dict:
    import cupy as cp

    from benchmarks.readme_defaults import (
        _finite_endpoint,
        native_calculator,
        native_settings,
    )
    from benchmarks.readme_hf_scaling import scaling_cases
    from benchmarks.readme_method_endpoints import binary_identity

    _, size, method = case.split("-", 2)
    atoms = scaling_cases()[f"water-{size}"].atoms
    cp.cuda.runtime.deviceSynchronize()
    started = time.perf_counter()
    calculator = native_calculator(method, "standard", "direct", Path("."))
    owner = calculator.prepare_batch([atoms], warm_start=True)
    try:
        item = owner.execute(strict=True, properties=("energy", "forces")).items[0]
        cp.cuda.runtime.deviceSynchronize()
        seconds = time.perf_counter() - started
        if not item.converged or item.executed_backend != "cuda":
            raise RuntimeError("incomplete CUDA E+F endpoint")
        if item.warm_start_used:
            raise RuntimeError(
                "fresh-process endpoint unexpectedly reused a warm density"
            )
        sample = dataclasses.asdict(item)
        sample.update(
            energy=float(item.energy),
            forces=_finite_endpoint(
                float(item.energy), item.forces, len(atoms), seconds
            ),
            seconds=seconds,
            settings=native_settings(calculator, item),
        )
        sample["ks_diagnostic"] = (
            item.ks_diagnostic.to_payload() if item.ks_diagnostic else None
        )
        return {
            "geometry_bohr": atoms,
            "method": method,
            "atoms": len(atoms),
            "sample": sample,
            "native_build": binary_identity(calculator),
        }
    finally:
        owner.close()


def cbs_endpoint(config: dict) -> dict:
    from generativeqc.basis import load_basis

    from benchmarks.readme_cbs import native_call

    frozen = config["cbs_inputs"]
    orbital = tuple(load_basis(p) for p in frozen["orbital_paths"])
    auxiliary = tuple(load_basis(p) for p in frozen["auxiliary_paths"])
    sample = native_call(
        frozen["protocol"]["atoms_bohr"], orbital, auxiliary, 32 << 30, "df", "standard"
    )
    return {"protocol": frozen["protocol"], "sample": sample}


def worker(request: Path) -> None:
    payload = json.loads(request.read_text(encoding="utf-8"))
    config = payload["config"]
    record = {
        "schema": SCHEMA,
        "request_sha256": digest(request),
        "status": "RUNNING",
        "pid": os.getpid(),
        "case": payload["case"],
        "state": payload["state"],
        "replicate": payload["replicate"],
        "production_source_revision": config["production_source_revision"],
        "driver_sha256": digest(Path(__file__)),
    }
    output = request.with_name("endpoint.json")
    save(output, record)
    observer = None
    try:
        started = time.perf_counter()
        validate_config(config)
        record["provenance_verification_wall_seconds"] = time.perf_counter() - started
        from generativeqc_compiler.common.compiler_cache import resolve_compiler_cache

        launcher = resolve_compiler_cache()
        record["launcher"] = {
            "path": str(launcher.path),
            "name": launcher.name,
            "version": launcher.version,
        }
        import cupy as cp

        from benchmarks._support import cuda_accelerator_metadata, environment_metadata

        record["accelerator"] = cuda_accelerator_metadata(cp)
        if record["accelerator"].get("nvidia_smi") is not None:
            record["accelerator"]["nvidia_smi"]["sampling_point"] = "before endpoint"
        record["environment"] = environment_metadata(
            distributions={
                "numpy": ("numpy",),
                "pyscf": ("pyscf",),
                "gpu4pyscf": ("gpu4pyscf-cuda12x", "gpu4pyscf"),
                "cupy": ("cupy-cuda12x",),
            }
        )
        bus = cp.cuda.Device().pci_bus_id
        record["physical_gpu_uuid"] = subprocess.check_output(
            ["nvidia-smi", "--id=" + bus, "--query-gpu=uuid", "--format=csv,noheader"],
            text=True,
            timeout=15,
        ).strip()
        if not record["physical_gpu_uuid"].startswith("GPU-"):
            raise ValueError("physical GPU identity unavailable")
        other_pids = subprocess.check_output(
            [
                "nvidia-smi",
                "--id=" + bus,
                "--query-compute-apps=pid",
                "--format=csv,noheader",
            ],
            text=True,
            timeout=15,
        )
        rows = [value.strip() for value in other_pids.splitlines() if value.strip()]
        if any(not value.isdigit() for value in rows):
            raise RuntimeError("GPU process inventory UNAVAILABLE")
        record["gpu_process_ids_before"] = list(map(int, rows))
        if any(pid != os.getpid() for pid in record["gpu_process_ids_before"]):
            raise RuntimeError(
                "assigned GPU has another compute process before endpoint"
            )
        started = time.perf_counter()
        observer = Observer()
        record["observer_setup_wall_seconds"] = time.perf_counter() - started
        with observer:
            record["endpoint"] = (
                cbs_endpoint(config)
                if payload["case"] == "cbs-c2"
                else water_endpoint(payload["case"])
            )
        if payload["case"] == "cbs-c2":
            from generativeqc import _native

            from benchmarks.readme_method_endpoints import binary_identity

            record["endpoint"]["native_build"] = binary_identity(
                SimpleNamespace(_library=_native.load_library(device="cuda"))
            )
        build = record["endpoint"]["native_build"]
        if (
            build["source_identity"] != config["native_source_identity"]
            or build["library_sha256"] != config["library_sha256"]
            or Path(build["library_path"]).resolve()
            != Path(config["library"]).resolve()
        ):
            raise ValueError("loaded native library differs from frozen SDK")
        record["imported_production_modules"] = verify_imported_modules(config)
        record["accelerator_after"] = cuda_accelerator_metadata(cp)
        if record["accelerator_after"].get("nvidia_smi") is not None:
            record["accelerator_after"]["nvidia_smi"]["sampling_point"] = (
                "after endpoint"
            )
        record["status"] = "COMPLETE"
    except BaseException as error:
        record.update(status="FAIL", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        if observer is not None:
            record["observation"] = observer.receipt()
        save(output, record)


def launch(
    request: Path, environment: dict[str, str], cwd: Path, timeout: float
) -> dict:
    """Retain timeout/failure logs, killing only this new process session."""
    output = request.parent
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "worker",
        "--request",
        str(request),
    ]
    with (
        (output / "stdout.log").open("xb") as stdout,
        (output / "stderr.log").open("xb") as stderr,
    ):
        started = time.perf_counter()
        process = subprocess.Popen(
            command,
            cwd=cwd,
            env=environment,
            stdout=stdout,
            stderr=stderr,
            start_new_session=True,
        )
        timed_out = False
        try:
            process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            timed_out = True
            if os.name == "posix":
                os.killpg(process.pid, signal.SIGKILL)
            else:
                process.kill()
            process.wait()
        seconds = time.perf_counter() - started
    result = {
        "command": command,
        "pid": process.pid,
        "returncode": process.returncode,
        "timed_out": timed_out,
        "launch_to_exit_wall_seconds": seconds,
        "timer_boundary": "before Popen through wait completion; includes startup/imports/observation/cleanup/export",
    }
    save(output / "process.json", result)
    return result


def run(
    config_path: Path,
    output: Path,
    *,
    pilot: bool,
    timeout: float,
    pilot_receipt: Path | None = None,
) -> None:
    if not math.isfinite(timeout) or timeout <= 0:
        raise ValueError("requires finite positive endpoint timeout")
    if not pilot and pilot_receipt is None:
        raise ValueError(
            "formal measurement requires a qualified independent pilot campaign"
        )
    config = json.loads(config_path.read_text(encoding="utf-8"))
    validate_config(config)
    expected_gpu = None
    if not pilot:
        if pilot_receipt is None:
            raise ValueError(
                "formal measurement requires a qualified independent pilot campaign"
            )
        gate = qualify_campaign(pilot_receipt)
        pilot_campaign = json.loads(
            (pilot_receipt / "campaign.json").read_text(encoding="utf-8")
        )
        if (
            gate["status"] != "PASS"
            or gate["cohort"] != "pilot"
            or gate["config_sha256"] != digest(config_path)
            or pilot_campaign["driver_sha256"] != digest(Path(__file__))
        ):
            raise ValueError(
                "pilot lacks matching SDK/config or independent accuracy PASS"
            )
        expected_gpu = gate["physical_gpu_uuid"]
    job = os.environ.get("SLURM_JOB_ID")
    if not job or not os.environ.get("CUDA_VISIBLE_DEVICES"):
        raise ValueError(
            "requires a real Slurm GPU allocation with assigned device visibility"
        )
    allocation = subprocess.check_output(
        ["/usr/bin/scontrol", "show", "job", job], text=True, timeout=15
    )
    if (
        f"JobId={job} " not in allocation
        or "JobState=RUNNING" not in allocation
        or "gres/gpu" not in allocation
    ):
        raise ValueError("Slurm does not confirm a running GPU allocation")
    rows = schedule(("water-12-wb97m-v",), 1) if pilot else schedule()
    output = raw_output_path(output)
    output.mkdir(parents=True, exist_ok=False)
    (output / "allocation.txt").write_text(allocation, encoding="utf-8")
    manifest = {
        "schema": SCHEMA,
        "status": "RUNNING",
        "cohort": "pilot" if pilot else "formal",
        "config": config,
        "config_sha256": digest(config_path),
        "driver_sha256": digest(Path(__file__)),
        "schedule": rows,
        "results": [],
        "accuracy": "INCOMPLETE",
        "slurm_job_id": job,
        "expected_gpu_uuid": expected_gpu,
        "output_root": str(output),
    }
    save(output / "campaign.json", manifest)
    try:
        for index, row in enumerate(rows):
            cache = output / "caches" / row["cache"]
            if row["state"] != "primed":
                new_cache(cache)
            priming = check_primed(cache) if row["state"] == "primed" else None
            folder = output / f"process-{index:02d}"
            folder.mkdir()
            request = folder / "request.json"
            before = census(cache)
            save(folder / "cache-before.json", before)
            save(
                request,
                {
                    **row,
                    "config": worker_config(config),
                    "cache_before_sha256": digest(folder / "cache-before.json"),
                    "priming_receipt": priming,
                    "output_root": str(output),
                    "cache_root": str(cache.resolve()),
                },
            )
            process = launch(
                request,
                cache_environment(cache, config, os.environ),
                cache / "work",
                timeout,
            )
            after = census(cache)
            save(folder / "cache-after.json", after)
            endpoint_path = folder / "endpoint.json"
            endpoint = (
                json.loads(endpoint_path.read_text(encoding="utf-8"))
                if endpoint_path.exists()
                else {}
            )
            complete = (
                process["returncode"] == 0 and endpoint.get("status") == "COMPLETE"
            )
            if complete:
                expected_gpu = expected_gpu or endpoint["physical_gpu_uuid"]
                if endpoint["physical_gpu_uuid"] != expected_gpu:
                    complete = False
            manifest["results"].append(
                {
                    **row,
                    "folder": folder.name,
                    "complete": complete,
                    "process": process,
                    "endpoint_sha256": digest(endpoint_path)
                    if endpoint_path.exists()
                    else None,
                }
            )
            save(output / "campaign.json", manifest)
            if not complete:
                manifest["status"] = "FAIL"
                save(output / "campaign.json", manifest)
                raise RuntimeError(f"endpoint failed; retained process {index}")
            if row["state"] in ("prime", "primed"):
                mark_primed(cache, digest(endpoint_path))
    except BaseException as error:
        manifest.update(status="FAIL", error=f"{type(error).__name__}: {error}")
        save(output / "campaign.json", manifest)
        raise
    manifest["status"] = "MEASURED"
    save(output / "campaign.json", manifest)
    qualification = qualify_campaign(output)
    save(output / "qualification.json", qualification)
    if qualification["status"] != "PASS":
        raise RuntimeError(
            "independent qualification did not PASS; raw measurements retained"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="action", required=True)
    child = commands.add_parser("worker")
    child.add_argument("--request", type=Path, required=True)
    parent = commands.add_parser("run")
    parent.add_argument("--config", type=Path, required=True)
    parent.add_argument("--output", type=raw_output_path, required=True)
    parent.add_argument("--pilot", action="store_true")
    parent.add_argument("--pilot-receipt", type=Path)
    parent.add_argument("--timeout", type=float, default=7200)
    replay = commands.add_parser("qualify")
    replay.add_argument("--campaign", type=Path, required=True)
    replay.add_argument("--output", type=raw_output_path, required=True)
    args = parser.parse_args()
    if args.action == "worker":
        worker(args.request.resolve())
    elif args.action == "qualify":
        output = raw_output_path(args.output.resolve())
        if output.exists():
            raise FileExistsError(output)
        result = qualify_campaign(args.campaign.resolve())
        save(output, result)
        print(json.dumps({"status": result["status"], "cohort": result["cohort"]}))
        if result["status"] != "PASS":
            raise SystemExit(1)
    else:
        run(
            args.config.resolve(),
            args.output.resolve(),
            pilot=args.pilot,
            timeout=args.timeout,
            pilot_receipt=args.pilot_receipt.resolve() if args.pilot_receipt else None,
        )


if __name__ == "__main__":
    main()
