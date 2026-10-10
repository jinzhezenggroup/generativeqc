"""Explicitly enabled CI diagnostics; never changes collection or test outcomes."""

from __future__ import annotations

import ctypes
import importlib.metadata
import json
import os
import resource
import sys
import threading
import time
from pathlib import Path
from typing import TYPE_CHECKING, TypedDict

if TYPE_CHECKING:
    from collections.abc import Generator

import pytest

HISTORY_LIMIT = 4 * 1024 * 1024
MODULE_LIMIT = 128
MODULE_PREFIXES = (
    "libgenerativeqc",
    "libnp_helper",
    "libopenblas",
    "libscipy_openblas",
    "libblas",
    "liblapack",
    "libgomp",
    "libomp",
    "libcint",
    "libcvhf",
    "libxc",
    "libtorch",
    "libc.so",
    "libstdc++",
    "libgcc_s",
    "libpython",
)
_STATE: pytest.StashKey[State] = pytest.StashKey()


class ModuleRecord(TypedDict):
    start: int
    end: int
    file_offset: int
    name: str


class Module(ctypes.Structure):
    _fields_ = [
        ("start", ctypes.c_size_t),
        ("end", ctypes.c_size_t),
        ("file_offset", ctypes.c_size_t),
        ("name", ctypes.c_char * 96),
    ]


def allowed_modules() -> list[ModuleRecord]:
    """Read mapping metadata only, outside the native signal handler."""
    result: list[ModuleRecord] = []
    with Path("/proc/self/maps").open() as stream:
        for line in stream:
            fields = line.split(maxsplit=5)
            if len(fields) != 6 or "x" not in fields[1]:
                continue
            name = Path(fields[5].strip()).name
            if not name.startswith(MODULE_PREFIXES) or ".so" not in name:
                continue
            start, end = (int(value, 16) for value in fields[0].split("-"))
            result.append(
                {
                    "start": start,
                    "end": end,
                    "file_offset": int(fields[2], 16),
                    "name": name[:95],
                }
            )
            if len(result) == MODULE_LIMIT:
                break
    return result


class State:
    def __init__(self, directory: Path, library: Path) -> None:
        self.active = True
        self.refreshing = False
        self.lock = threading.RLock()
        self.last_modules = None
        self.worker = os.environ.get("PYTEST_XDIST_WORKER", "main")
        self.prefix = directory / f"worker-{self.worker}-{os.getpid()}"
        self.history = None
        self.native_fd = None
        self.old_core_limit = None
        self.installed = False
        self.size = 0
        try:
            directory.mkdir(parents=True, exist_ok=True)
            self.history = self.prefix.with_suffix(".jsonl").open("w", encoding="utf-8")
            self.native_fd = os.open(
                self.prefix.with_suffix(".native.log"),
                os.O_WRONLY | os.O_CREAT | os.O_TRUNC,
                0o600,
            )
            self.old_core_limit = resource.getrlimit(resource.RLIMIT_CORE)
            resource.setrlimit(resource.RLIMIT_CORE, (0, self.old_core_limit[1]))
            # These tiny publication functions must retain the GIL, unlike
            # scientific CDLL calls. Published tables remain immutable on faults.
            self.lib = ctypes.PyDLL(str(library))
            self.lib.diagnostic_set_modules.argtypes = [
                ctypes.POINTER(Module),
                ctypes.c_uint,
            ]
            self.lib.diagnostic_set_modules.restype = ctypes.c_int
            self.lib.diagnostic_install.argtypes = [ctypes.c_int]
            self.lib.diagnostic_install.restype = ctypes.c_int
            self.lib.diagnostic_restore.argtypes = []
            self.lib.diagnostic_restore.restype = None
            self.refresh_modules()
            if self.lib.diagnostic_install(self.native_fd):
                raise RuntimeError("Could not install native fault diagnostics")
            self.installed = True
            # GEMM/VHF symbols resolve after PySCF's private libraries load in the test.
            # This passive event replaces no PySCF function or ctypes ABI.
            sys.addaudithook(self.audit)
            packages = {}
            for name in (
                "numpy",
                "scipy",
                "pyscf",
                "torch",
                "pytest",
                "pytest-xdist",
                "pytest-cov",
            ):
                try:
                    packages[name] = importlib.metadata.version(name)
                except importlib.metadata.PackageNotFoundError:
                    pass
                except Exception as error:  # noqa: BLE001 - diagnostics cannot fail scientific work
                    self.write(
                        "diagnostic_error",
                        operation="package_version",
                        error=type(error).__name__,
                    )
            self.write("session", python=sys.version.split()[0], packages=packages)
        except BaseException:
            self.close()
            raise

    def write(self, event: str, **payload: object) -> None:
        with self.lock:
            if not self.active or self.size >= HISTORY_LIMIT:
                return
            try:
                if isinstance(payload.get("nodeid"), str):
                    payload["nodeid"] = str(payload["nodeid"])[:2048]
                record = {
                    "event": event,
                    "worker": self.worker,
                    "pid": os.getpid(),
                    "monotonic_ns": time.monotonic_ns(),
                    **payload,
                }
                text = json.dumps(record) + "\n"
                encoded_size = len(text.encode("utf-8"))
                if self.size + encoded_size >= HISTORY_LIMIT - 128:
                    text = (
                        json.dumps({"event": "history_limit", "bytes": self.size})
                        + "\n"
                    )
                    self.size = HISTORY_LIMIT
                else:
                    self.size += encoded_size
                self.history.write(text)
                self.history.flush()
            except Exception:  # noqa: BLE001 - diagnostics cannot fail scientific work
                # Diagnostic formatting/storage must never replace a test outcome.
                self.size = HISTORY_LIMIT

    def refresh_modules(self) -> None:
        try:
            with self.lock:
                if not self.active:
                    return
                modules = allowed_modules()
                if modules == self.last_modules:
                    return
                native = (Module * len(modules))(
                    *(
                        Module(
                            m["start"],
                            m["end"],
                            m["file_offset"],
                            m["name"].encode("ascii", "replace"),
                        )
                        for m in modules
                    )
                )
                result = self.lib.diagnostic_set_modules(native, len(modules))
                if result == -2:
                    self.write("module_snapshot_limit")
                    return
                if result:
                    raise RuntimeError("Could not publish native module metadata")
                self.last_modules = modules
                temporary = self.prefix.with_suffix(".modules.tmp")
                public_modules = [
                    {
                        "name": module["name"],
                        "file_offset": module["file_offset"],
                        "size": module["end"] - module["start"],
                    }
                    for module in modules
                ]
                temporary.write_text(json.dumps(public_modules), encoding="utf-8")
                temporary.replace(self.prefix.with_suffix(".modules.json"))
        except Exception as error:  # noqa: BLE001 - diagnostics cannot fail scientific work
            self.write(
                "diagnostic_error",
                operation="module_snapshot",
                error=type(error).__name__,
            )

    def audit(self, event: str, args: tuple[object, ...]) -> None:
        if (
            self.active
            and not self.refreshing
            and event == "ctypes.dlsym"
            and len(args) == 2
            and args[1] in ("NPdgemm", "CVHFnrs4_incore_drv", "CVHFnrs8_incore_drv")
        ):
            self.refreshing = True
            try:
                self.refresh_modules()
                self.write("provider_boundary", symbol=str(args[1]))
            finally:
                self.refreshing = False

    def close(self) -> None:
        with self.lock:
            self.active = False
            self.cleanup_errors = []
            operations = []
            if self.installed:
                operations.append(("handlers", self.lib.diagnostic_restore))
                self.installed = False
            if self.history is not None:
                operations.append(("history", self.history.close))
            if self.native_fd is not None:
                fd = self.native_fd
                operations.append(("native_fd", lambda: os.close(fd)))
                self.native_fd = None
            if self.old_core_limit is not None:
                old_limit = self.old_core_limit
                operations.append(
                    (
                        "core_limit",
                        lambda: resource.setrlimit(resource.RLIMIT_CORE, old_limit),
                    )
                )
                self.old_core_limit = None
            for name, operation in operations:
                try:
                    operation()
                except Exception as error:  # noqa: BLE001 - cleanup must preserve the test outcome
                    self.cleanup_errors.append((name, type(error).__name__))


def pytest_addoption(parser: pytest.Parser) -> None:
    group = parser.getgroup("native diagnostics")
    group.addoption("--native-diagnostics-dir", default=None)
    group.addoption("--native-diagnostics-library", default=None)


@pytest.hookimpl(trylast=True)
def pytest_configure(config: pytest.Config) -> None:
    directory = config.getoption("native_diagnostics_dir")
    library = config.getoption("native_diagnostics_library")
    if directory is None and library is None:
        return
    if not directory or not library:
        raise pytest.UsageError("Native diagnostics needs both directory and library")
    if getattr(config.option, "numprocesses", 0) and not hasattr(config, "workerinput"):
        return
    config.stash[_STATE] = State(Path(directory).resolve(), Path(library).resolve())


@pytest.hookimpl(wrapper=True)
def pytest_runtest_protocol(
    item: pytest.Item, nextitem: pytest.Item | None
) -> Generator[None, object, object]:
    state = item.config.stash.get(_STATE, None)
    if state:
        state.refresh_modules()
        state.write("start", nodeid=item.nodeid)
    return (yield)


@pytest.hookimpl(wrapper=True)
def pytest_runtest_makereport(
    item: pytest.Item, call: pytest.CallInfo[object]
) -> Generator[None, pytest.TestReport, pytest.TestReport]:
    report = yield
    state = item.config.stash.get(_STATE, None)
    if state and (report.when == "call" or report.failed):
        state.write(
            "report",
            nodeid=report.nodeid,
            when=report.when,
            outcome=report.outcome,
            duration=report.duration,
        )
    return report


def pytest_sessionfinish(session: pytest.Session, exitstatus: int) -> None:
    state = session.config.stash.get(_STATE, None)
    if state:
        state.write("finish", exitstatus=int(exitstatus))


@pytest.hookimpl(tryfirst=True)
def pytest_unconfigure(config: pytest.Config) -> None:
    state = config.stash.get(_STATE, None)
    if state:
        state.close()
        del config.stash[_STATE]
