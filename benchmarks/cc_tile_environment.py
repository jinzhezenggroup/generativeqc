"""Explicit semantic and observation environment contracts for CC evidence.

Only named keys are read. Historical receipts missing this contract remain
unbound; an absent workspace value is never inferred from an omitted field.
"""

from __future__ import annotations

import hashlib
import json
import os
import threading
import time
from copy import deepcopy
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import subprocess
    from collections.abc import Mapping

PROFILE_SCHEMA = "generativeqc.cc-profile-environment.v1"
EXECUTION_SCHEMA = "generativeqc.cc-execution-environment.v1"
CHILD_SCHEMA = "generativeqc.cc-child-environment.v1"
OBSERVATION_SCHEMA = "generativeqc.cc-observation-policy.v1"
SEMANTIC_KEYS = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "CUDA_VISIBLE_DEVICES",
    "CUBLAS_WORKSPACE_CONFIG",
    "CUBLASLT_WORKSPACE_SIZE",
    "CUDA_MODULE_LOADING",
    "CUDA_LAUNCH_BLOCKING",
    "CUDA_DEVICE_MAX_CONNECTIONS",
    "CUDA_DEVICE_MAX_COPY_CONNECTIONS",
    "CUDA_CACHE_DISABLE",
    "CUDA_CACHE_MAXSIZE",
    "CUDA_FORCE_PTX_JIT",
    "CUDA_DISABLE_PTX_JIT",
    "NVIDIA_TF32_OVERRIDE",
    "GENERATIVEQC_GRAPH_EIGENSOLVER_OVERRIDE",
    "GENERATIVEQC_DIRECT_HF_SEPARATE_JK",
    "GENERATIVEQC_RHF_RESIDENT_VALUES",
)
OBSERVATION_KEYS = (
    "PATH",
    "LD_LIBRARY_PATH",
    "LANG",
    "LC_ALL",
    "LC_CTYPE",
    "SYSTEMROOT",
    "WINDIR",
    "TEMP",
    "TMP",
    "PYTHONCOERCECLOCALE",
    "PYTHONUTF8",
    "LD_PRELOAD",
    "CUDA_INJECTION64_PATH",
    "NVTX_INJECTION64_PATH",
    "CUBLAS_LOGINFO_DBG",
    "CUBLAS_LOGDEST_DBG",
    "CUSOLVER_LOG_LEVEL",
    "CUSOLVER_LOG_MASK",
    "CUSOLVER_LOG_FILE",
    "GQC_GUARD_TRACE",
)
RUN_PATH_KEYS = {"CUBLAS_LOGDEST_DBG", "CUSOLVER_LOG_FILE", "GQC_GUARD_TRACE"}
INJECTION_KEYS = {"LD_PRELOAD", "CUDA_INJECTION64_PATH", "NVTX_INJECTION64_PATH"}


def canonical_hash(value: Any) -> str:
    """Hash the complete retained object with one deterministic JSON encoding."""
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _snapshot(environment: Mapping[str, str], keys: tuple[str, ...]) -> dict:
    return {
        key: {"present": key in environment, "value": environment.get(key)}
        for key in keys
    }


def _validate(values: Any, keys: tuple[str, ...]) -> None:
    if not isinstance(values, dict) or set(values) != set(keys):
        raise ValueError("complete environment presence/value fields required")
    for item in values.values():
        if (
            not isinstance(item, dict)
            or set(item) != {"present", "value"}
            or type(item["present"]) is not bool
        ):
            raise ValueError("invalid environment presence record")
        if item["present"]:
            if (
                not isinstance(item["value"], str)
                or "\0" in item["value"]
                or len(item["value"]) > 32768
            ):
                raise ValueError("invalid present environment value")
        elif item["value"] is not None:
            raise ValueError("absent environment value must be null")


def profile_environment(environment: Mapping[str, str]) -> dict:
    """Select semantic controls without recording observation/injection controls."""
    return {
        "schema": PROFILE_SCHEMA,
        "variables": _snapshot(environment, SEMANTIC_KEYS),
    }


def validate_profile(profile: Any, expected_hash: str) -> None:
    """Reject unknown/malformed presence contracts even when their hashes agree."""
    if (
        not isinstance(profile, dict)
        or set(profile) != {"schema", "variables"}
        or profile["schema"] != PROFILE_SCHEMA
    ):
        raise ValueError(
            "bound environment profile required; historical state is UNKNOWN"
        )
    _validate(profile["variables"], SEMANTIC_KEYS)
    if canonical_hash(profile) != expected_hash:
        raise ValueError("environment profile hash mismatch")


def execution_environment(environment: Mapping[str, str]) -> dict:
    """Retain all explicitly transmitted semantic and observation values."""
    return {
        "schema": EXECUTION_SCHEMA,
        "profile": profile_environment(environment),
        "observation": _snapshot(environment, OBSERVATION_KEYS),
    }


def observation_policy(clean: dict, diagnostic: dict) -> dict:
    """Freeze two explicit observation roles without collapsing their differences."""
    return {
        "schema": OBSERVATION_SCHEMA,
        "clean": deepcopy(clean),
        "diagnostic": deepcopy(diagnostic),
    }


def validate_observation_policy(policy: Any, expected_hash: str) -> None:
    """Require declared complete roles and forbid injection into clean samples."""
    if (
        not isinstance(policy, dict)
        or set(policy) != {"schema", "clean", "diagnostic"}
        or policy["schema"] != OBSERVATION_SCHEMA
    ):
        raise ValueError("explicit clean/diagnostic observation policy required")
    for role in ("clean", "diagnostic"):
        _validate(policy[role], OBSERVATION_KEYS)
        for key, item in policy[role].items():
            if (
                item["present"]
                and "$RUN_DIR" in item["value"]
                and (
                    key not in RUN_PATH_KEYS
                    or not item["value"].startswith("$RUN_DIR/")
                    or ".." in item["value"].split("/")
                )
            ):
                raise ValueError("invalid run-specific observation path template")
    if any(
        policy["clean"][key]["present"] and policy["clean"][key]["value"]
        for key in INJECTION_KEYS
    ):
        raise ValueError("clean samples cannot preload/inject an observer")
    if canonical_hash(policy) != expected_hash:
        raise ValueError("observation policy hash mismatch")


def validate_observation(
    execution: dict,
    policy: dict,
    policy_hash: str,
    diagnostic: bool,
    run_directory: str | None,
) -> None:
    """Match actual controls to a prior declaration, resolving only owned log paths."""
    validate_observation_policy(policy, policy_hash)
    if type(diagnostic) is not bool:
        raise ValueError("explicit observation role required")
    expected = deepcopy(policy["diagnostic" if diagnostic else "clean"])
    for item in expected.values():
        if item["present"] and "$RUN_DIR" in item["value"]:
            if not isinstance(run_directory, str) or not run_directory:
                raise ValueError("original run directory required for observation path")
            item["value"] = item["value"].replace("$RUN_DIR", run_directory)
    if execution["observation"] != expected:
        raise ValueError("actual observation controls differ from declared recipe")


def validate_execution(environment: Any, profile_hash: str) -> dict[str, str]:
    """Validate both domains and reconstruct exactly the allowed child mapping."""
    if (
        not isinstance(environment, dict)
        or set(environment) != {"schema", "profile", "observation"}
        or environment["schema"] != EXECUTION_SCHEMA
    ):
        raise ValueError("bound execution environment required")
    validate_profile(environment["profile"], profile_hash)
    _validate(environment["observation"], OBSERVATION_KEYS)
    return {
        key: item["value"]
        for group in (environment["profile"]["variables"], environment["observation"])
        for key, item in group.items()
        if item["present"]
    }


def atomic_json(path: Path, value: dict) -> None:
    """Replace one checkpoint without truncating its prior durable version."""
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def validate_child(child: Any, plan: dict, pid: int) -> None:
    """Recompute the child's retained inheritance proof independently of flags."""
    argv = plan.get("argv")
    if (
        not isinstance(argv, list)
        or not argv
        or any(not isinstance(value, str) for value in argv)
    ):
        raise ValueError("nonempty native argv binding required")
    if (
        type(pid) is not int
        or pid <= 0
        or not isinstance(child, dict)
        or child.get("schema") != CHILD_SCHEMA
        or child.get("status") != "PREPARED_FOR_EXEC"
        or type(child.get("pid")) is not int
        or child["pid"] <= 0
        or child["pid"] != pid
        or child.get("argv") != plan["argv"]
        or child.get("environment") != plan["environment"]
        or child.get("environment_sha256") != canonical_hash(plan["environment"])
    ):
        raise ValueError("actual child environment/argv/PID binding mismatch")
    validate_execution(child["environment"], plan["profile_sha256"])


def observe_native(pid: int, executable: Path, libraries: dict[str, dict]) -> dict:
    """Bounded Linux startup attestation; never read another process's environment."""
    deadline = time.monotonic() + 10
    expected = {str(Path(item["path"]).resolve()) for item in libraries.values()}
    while time.monotonic() < deadline:
        try:
            actual_executable = Path(os.readlink(f"/proc/{pid}/exe")).resolve()
            mapped = {}
            for line in Path(f"/proc/{pid}/maps").read_text().splitlines():
                fields = line.split(maxsplit=5)
                if len(fields) == 6 and fields[5].startswith("/"):
                    mapped[fields[5]] = (fields[3], int(fields[4]))
            paths = set(mapped)
            if actual_executable == executable.resolve() and expected <= paths:
                actual = {}
                for name, item in libraries.items():
                    path = Path(item["path"]).resolve()
                    if file_identity(path) != item["file_identity"]:
                        raise ValueError("mapped library changed during execution")
                    device, inode = mapped[str(path)]
                    major, minor = (int(value, 16) for value in device.split(":"))
                    if inode != item["file_identity"]["st_ino"] or (major, minor) != (
                        os.major(item["file_identity"]["st_dev"]),
                        os.minor(item["file_identity"]["st_dev"]),
                    ):
                        raise ValueError(
                            "mapped library inode/device differs from prehashed file"
                        )
                    actual[name] = dict(item, path=str(path))
                return {"executable": str(actual_executable), "libraries": actual}
        except (FileNotFoundError, ProcessLookupError):
            break
        time.sleep(0.01)
    raise ValueError("actual native executable/library bindings unavailable")


def file_identity(path: Path) -> dict[str, int]:
    """Bind a prehashed file without reading its contents during execution."""
    stat = path.stat()
    return {
        key: getattr(stat, key)
        for key in ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns")
    }


class NativeSupervisor:
    """Wait independently of parent attestation with an absolute child deadline."""

    def __init__(self, process: subprocess.Popen, deadline: float) -> None:
        self.process = process
        self.deadline = deadline
        self.finished = threading.Event()
        self.returncode: int | None = None
        self.failure: BaseException | None = None
        self.cleanup_errors: list[tuple[str, BaseException]] = []
        self.cleanup_attempted = False
        self.ended_ns = 0
        self.thread = threading.Thread(target=self._wait, daemon=True)

    def start(self) -> None:
        """Start the sole timed wait before any parent-side attestation."""
        self.thread.start()

    def _wait(self) -> None:
        try:
            self.returncode = self.process.wait(
                timeout=max(0, self.deadline - time.monotonic())
            )
        except BaseException as error:  # noqa: BLE001 -- preserve failure across the thread boundary
            self.failure = error
            self.cleanup_attempted = True
            for stage, action in (
                ("child kill", self.process.kill),
                ("child reap", lambda: self.process.wait(timeout=10)),
            ):
                try:
                    action()
                except BaseException as cleanup_error:  # noqa: BLE001 -- cleanup cannot mask source failure
                    self.cleanup_errors.append((stage, cleanup_error))
        finally:
            self.ended_ns = time.perf_counter_ns()
            self.finished.set()

    def join(self, timeout: float) -> bool:
        """Return whether the independent completion observation is terminal."""
        if self.thread.ident is not None:
            self.thread.join(timeout)
        return self.finished.is_set()


def validate_libraries(libraries: Any) -> None:
    """Reject malformed prehashed library identities without touching remote paths."""
    if not isinstance(libraries, dict) or not libraries:
        raise ValueError("nonempty library binding required")
    keys = {"st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns"}
    for name, item in libraries.items():
        if (
            not isinstance(name, str)
            or not name
            or not isinstance(item, dict)
            or set(item) != {"path", "sha256", "file_identity"}
        ):
            raise ValueError("malformed library binding")
        if (
            not isinstance(item["path"], str)
            or not item["path"]
            or not isinstance(item["sha256"], str)
            or len(item["sha256"]) != 64
            or any(c not in "0123456789abcdef" for c in item["sha256"])
        ):
            raise ValueError("invalid library path/hash")
        stat = item["file_identity"]
        if (
            not isinstance(stat, dict)
            or set(stat) != keys
            or any(type(v) is not int or v < 0 for v in stat.values())
            or stat["st_ino"] == 0
            or stat["st_size"] == 0
        ):
            raise ValueError("invalid library file identity")
