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
EXECUTION_SCHEMA = "generativeqc.cc-execution-environment.v2"
CHILD_SCHEMA = "generativeqc.cc-child-environment.v2"
OBSERVATION_SCHEMA = "generativeqc.cc-observation-policy.v2"
SESSION_SCHEMA = "generativeqc.cc-nsys-session.v1"
SESSION_KEY = "NSYS_PROFILING_SESSION_ID"
SESSION_PLACEHOLDER = "$NSYS_GENERATED_SESSION"
# Local admission bound, not a vendor-specified session identifier format.
SESSION_LIMIT = 1024
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
    SESSION_KEY,
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


def observation_policy(
    clean: dict, diagnostic: dict, *, nsys_profiler: dict | None = None
) -> dict:
    """Freeze two explicit observation roles without collapsing their differences."""
    if diagnostic.get(SESSION_KEY) != {"present": False, "value": None}:
        raise ValueError("declare the session source before launch, not a literal ID")
    result = {
        "schema": OBSERVATION_SCHEMA,
        "clean": deepcopy(clean),
        "diagnostic": deepcopy(diagnostic),
        "session_control": {
            "key": SESSION_KEY,
            "source": "nsight-systems-parent",
            "enabled": nsys_profiler is not None,
            "profiler": deepcopy(nsys_profiler),
        },
    }
    if nsys_profiler is not None:
        result["diagnostic"][SESSION_KEY] = {
            "present": True,
            "value": SESSION_PLACEHOLDER,
        }
    return result


def _session_value(value: Any) -> None:
    if (
        not isinstance(value, str)
        or not value
        or len(value) > SESSION_LIMIT
        or "\0" in value
        or SESSION_PLACEHOLDER in value
    ):
        raise ValueError("bounded nonempty opaque profiler session required")


def require_unprofiled_launch(environment: Mapping[str, str]) -> None:
    """Reject a pre-existing session before an owned outer profiler spawn."""
    if SESSION_KEY in environment:
        raise ValueError("outer launch cannot inherit a manually preset session")


def _process_identity(value: Any) -> None:
    if (
        not isinstance(value, dict)
        or set(value) != {"pid", "starttime"}
        or type(value["pid"]) is not int
        or value["pid"] <= 0
        or not isinstance(value["starttime"], str)
        or not value["starttime"].isascii()
        or not value["starttime"].isdigit()
        or len(value["starttime"]) > 32
    ):
        raise ValueError("PID and Linux process starttime binding required")


def process_identity(pid: int) -> dict:
    """Bind a live Linux process without reading its environment."""
    stat = Path(f"/proc/{pid}/stat").read_text()
    result = {"pid": pid, "starttime": stat[stat.rindex(")") + 2 :].split()[19]}
    _process_identity(result)
    return result


def _profiler_recipe(value: Any) -> None:
    if (
        not isinstance(value, dict)
        or set(value) != {"path", "sha256", "version", "argv", "session_name"}
        or not isinstance(value["path"], str)
        or not value["path"].startswith("/")
        or "\0" in value["path"]
        or not isinstance(value["sha256"], str)
        or len(value["sha256"]) != 64
        or any(c not in "0123456789abcdef" for c in value["sha256"])
        or not isinstance(value["version"], str)
        or not value["version"]
        or len(value["version"]) > 512
        or "\0" in value["version"]
        or not isinstance(value["session_name"], str)
        or not value["session_name"]
        or len(value["session_name"]) > 64
        or any(c.isspace() or c == "\0" for c in value["session_name"])
        or not isinstance(value["argv"], list)
        or not 2 <= len(value["argv"]) <= 256
        or any(
            not isinstance(arg, str) or not arg or "\0" in arg or len(arg) > 32768
            for arg in value["argv"]
        )
        or value["argv"][:2] != [value["path"], "profile"]
        or "--session-new=" + value["session_name"] not in value["argv"]
    ):
        raise ValueError("frozen Nsight executable/version/argv/session name required")


def validate_session_binding(
    binding: Any, binding_hash: Any, policy: dict, execution: dict, parent: Any
) -> None:
    """Check the retained launch chain; this alone cannot prove a real capture.

    The owned outer controller must independently qualify the actual tool and
    retain its raw outputs. The user session name and opaque injected ID are
    separate identities; a session listing does not establish their mapping.
    """
    if (
        not isinstance(binding, dict)
        or set(binding)
        != {
            "schema",
            "policy_sha256",
            "profiler",
            "outer_process",
            "outer_session",
            "profiler_process",
            "parent",
            "session_query",
            "id_mapping",
        }
        or binding["schema"] != SESSION_SCHEMA
        or binding["policy_sha256"] != canonical_hash(policy)
        or canonical_hash(binding) != binding_hash
        or binding["profiler"] != policy["session_control"]["profiler"]
        or binding["outer_session"] != {"present": False, "value": None}
        or binding["id_mapping"] != "UNAVAILABLE"
    ):
        raise ValueError("bound prelaunch policy and outer profiler evidence required")
    _process_identity(binding["outer_process"])
    _process_identity(parent)
    actual_parent = binding["parent"]
    if (
        not isinstance(actual_parent, dict)
        or set(actual_parent) != {"process", "session_id"}
        or actual_parent["process"] != parent
        or actual_parent["session_id"] != execution["observation"][SESSION_KEY]["value"]
    ):
        raise ValueError("actual parent profiler session binding mismatch")
    _session_value(actual_parent["session_id"])
    profiler = binding["profiler_process"]
    recipe = binding["profiler"]
    if (
        not isinstance(profiler, dict)
        or set(profiler) != {"process", "executable", "sha256"}
        or profiler["executable"] != recipe["path"]
        or profiler["sha256"] != recipe["sha256"]
    ):
        raise ValueError("actual profiler executable binding required")
    _process_identity(profiler["process"])
    query = binding["session_query"]
    if (
        not isinstance(query, dict)
        or set(query) != {"argv", "exit_code", "stdout", "stdout_sha256"}
        or query["argv"] != [recipe["path"], "sessions", "list", "--show-header=false"]
        or type(query["exit_code"]) is not int
        or query["exit_code"] != 0
        or not isinstance(query["stdout"], str)
        or len(query["stdout"]) > 65536
        or not any(
            recipe["session_name"] in line.split()
            for line in query["stdout"].splitlines()
        )
        or hashlib.sha256(query["stdout"].encode()).hexdigest()
        != query["stdout_sha256"]
    ):
        raise ValueError("successful retained owned-session query required")


def validate_observation_policy(policy: Any, expected_hash: str) -> None:
    """Require declared complete roles and forbid injection into clean samples."""
    if (
        not isinstance(policy, dict)
        or set(policy) != {"schema", "clean", "diagnostic", "session_control"}
        or policy["schema"] != OBSERVATION_SCHEMA
    ):
        raise ValueError("explicit clean/diagnostic observation policy required")
    control = policy["session_control"]
    if (
        not isinstance(control, dict)
        or set(control) != {"key", "source", "enabled", "profiler"}
        or control["key"] != SESSION_KEY
        or control["source"] != "nsight-systems-parent"
        or type(control["enabled"]) is not bool
    ):
        raise ValueError("explicit named profiler session source required")
    if control["enabled"]:
        _profiler_recipe(control["profiler"])
    elif control["profiler"] is not None:
        raise ValueError("disabled profiler session cannot declare a tool")
    for role in ("clean", "diagnostic"):
        _validate(policy[role], OBSERVATION_KEYS)
        for key, item in policy[role].items():
            if (
                item["present"]
                and SESSION_PLACEHOLDER in item["value"]
                and (
                    role != "diagnostic"
                    or key != SESSION_KEY
                    or not control["enabled"]
                    or item["value"] != SESSION_PLACEHOLDER
                )
            ):
                raise ValueError("only the declared diagnostic session may be dynamic")
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
    absent = {"present": False, "value": None}
    if policy["clean"][SESSION_KEY] != absent:
        raise ValueError("clean samples require an absent profiler session")
    expected_session = (
        {"present": True, "value": SESSION_PLACEHOLDER}
        if control["enabled"]
        else absent
    )
    if policy["diagnostic"][SESSION_KEY] != expected_session:
        raise ValueError("diagnostic session must match its prelaunch source rule")
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
    session_binding: Any = None,
    session_binding_hash: Any = None,
    parent_process: Any = None,
) -> None:
    """Match actual controls to a prior declaration, resolving only owned log paths."""
    validate_observation_policy(policy, policy_hash)
    if type(diagnostic) is not bool:
        raise ValueError("explicit observation role required")
    expected = deepcopy(policy["diagnostic" if diagnostic else "clean"])
    if diagnostic and policy["session_control"]["enabled"]:
        validate_session_binding(
            session_binding, session_binding_hash, policy, execution, parent_process
        )
        expected[SESSION_KEY] = deepcopy(execution["observation"][SESSION_KEY])
        if expected[SESSION_KEY]["present"] is not True:
            raise ValueError("diagnostic profiler session must be present")
    elif session_binding is not None or session_binding_hash is not None:
        raise ValueError("unexpected profiler session evidence")
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
    if environment["observation"][SESSION_KEY]["present"]:
        _session_value(environment["observation"][SESSION_KEY]["value"])
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


def validate_child(
    child: Any, plan: dict, pid: int, native_process: Any = None
) -> None:
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
    if child["environment"]["observation"][SESSION_KEY]["present"]:
        _process_identity(native_process)
        if native_process["pid"] != pid or child.get("process") != native_process:
            raise ValueError("profiled child PID/starttime binding mismatch")


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
