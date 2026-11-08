"""Exercise exact CI tool downloads, bounded recovery and fatal-error behavior."""

from __future__ import annotations

import hashlib
import importlib.util
import io
import json
import re
import shlex
import shutil
import subprocess
import sys
import tarfile
import threading
import time
from contextlib import contextmanager
from dataclasses import replace
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from collections.abc import Iterator
    from types import ModuleType

ROOT = Path(__file__).resolve().parents[2]


def _download_commands() -> list[tuple[str, ...]]:
    workflow = (ROOT / ".github/workflows/ci.yml").read_text()
    logical_lines = re.sub(r"\\\n\s*", " ", workflow).splitlines()
    commands = []
    for logical_line in logical_lines:
        line = logical_line.strip().removeprefix("run: ")
        if (
            line.startswith("curl ")
            and "github.com/ccache/ccache/releases/download/" in line
        ) or re.fullmatch(r"python(?:3)?\s+\S+\.py\s+ccache", line):
            commands.append(tuple(shlex.split(line)))
    assert commands, "no compiler-cache download is covered"
    # Equivalent downloads in separate jobs need one transport execution each.
    return list(dict.fromkeys(commands))


def _helper(
    command: tuple[str, ...], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> ModuleType:
    script = ROOT / command[1]
    spec = importlib.util.spec_from_file_location("ci_download_test_helper", script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, module)
    spec.loader.exec_module(module)
    monkeypatch.setattr(sys, "argv", [str(script), *command[2:]])
    monkeypatch.setattr(module.platform, "system", lambda: "Linux")
    monkeypatch.setattr(module.platform, "machine", lambda: "x86_64")
    runner = tmp_path / "runner"
    runner.mkdir()
    monkeypatch.setenv("RUNNER_TEMP", str(runner))
    monkeypatch.setenv("CCACHE_VERSION", "4.14")
    monkeypatch.setenv("GH_TOKEN", hashlib.sha256(b"test credential").hexdigest())
    return module


@pytest.fixture(params=_download_commands())
def command(
    request: pytest.FixtureRequest,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[str, ...]:
    invocation = request.param
    if invocation[0] == "curl":
        return invocation
    helper = _helper(invocation, tmp_path, monkeypatch)
    observed = []

    class CapturedDownload(Exception):
        pass

    def capture(args: list[str], **_kwargs: object) -> None:
        assert args[0] == "curl", "unexpected process before the download"
        observed.append(tuple(args))
        raise CapturedDownload

    # Execute the workflow's CLI and capture its real process-boundary argv;
    # do not copy a second set of curl flags or inspect helper source spelling.
    with monkeypatch.context() as boundary:
        boundary.setattr(subprocess, "run", capture)
        with pytest.raises(CapturedDownload):
            helper.main()
    return observed[0]


def _helper_commands() -> list[tuple[str, ...]]:
    commands = [item for item in _download_commands() if item[0] != "curl"]
    assert commands, "the verified compiler-cache helper is not covered"
    return commands


@pytest.fixture(params=_helper_commands())
def helper(
    request: pytest.FixtureRequest,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> ModuleType:
    return _helper(request.param, tmp_path, monkeypatch)


@contextmanager
def _server(
    statuses: list[int], *, stall: bool = False
) -> Iterator[tuple[str, list[int]]]:
    observed = []

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            status = statuses[min(len(observed), len(statuses) - 1)]
            observed.append(status)
            self.send_response(status)
            self.send_header("Content-Length", "16")
            self.end_headers()
            if stall:
                time.sleep(0.4)
            try:
                self.wfile.write(b"verified archive")
            except (BrokenPipeError, ConnectionResetError):
                pass

        def log_message(self, *_args: object) -> None:
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}/archive", observed
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def _run(
    command: tuple[str, ...], url: str, tmp_path: Path, *, short: bool = False
) -> tuple[subprocess.CompletedProcess[str], Path]:
    curl = shutil.which("curl")
    if curl is None:
        pytest.skip("requires the curl used by the CI download")
    args = list(command)
    args[0] = curl
    destination = tmp_path / "archive"
    args[args.index("--output") + 1] = str(destination)
    args[-1] = url
    # The production helper accepts HTTPS only. This server is loopback HTTP;
    # change only the test transport, retaining all real retry/failure flags.
    for option in ("--proto", "--proto-redir"):
        if option in args:
            args[args.index(option) + 1] = "=http"
    if short:
        # Scale only wall-time limits for a real stalled transfer. The unscaled
        # workflow limits are checked separately, without a minute-long test.
        for option, value in (
            ("--max-time", "0.1"),
            ("--retry-max-time", "1"),
            ("--retry-delay", "0"),
        ):
            args[args.index(option) + 1] = value
    result = subprocess.run(
        ["bash", "-e", "-c", shlex.join(args) + "\nprintf continued"],
        capture_output=True,
        text=True,
        check=False,
        timeout=15,
    )
    return result, destination


def test_download_has_finite_transport_budget(command: tuple[str, ...]) -> None:
    def value(option: str) -> float:
        return float(command[command.index(option) + 1])

    assert 0 < value("--retry") <= 3
    assert 0 < value("--connect-timeout") <= value("--max-time")
    assert 0 < value("--retry-delay") <= value("--retry-max-time")
    # Curl may start its final transfer just before the retry clock expires.
    assert 0 < value("--retry-max-time") + value("--max-time") <= 120


def test_download_recovers_after_transient_server_errors(
    command: tuple[str, ...], tmp_path: Path
) -> None:
    with _server([500, 503, 200]) as (url, observed):
        result, destination = _run(command, url, tmp_path)
    assert result.returncode == 0, result.stderr
    assert observed[-1] == 200 and len(observed) > 1
    assert destination.read_bytes() == b"verified archive"
    assert result.stdout == "continued"


@pytest.mark.parametrize("status", [403, 404])
def test_download_keeps_permanent_http_errors_fatal(
    command: tuple[str, ...], status: int, tmp_path: Path
) -> None:
    with _server([status]) as (url, observed):
        result, _ = _run(command, url, tmp_path)
    assert result.returncode != 0
    assert observed == [status]
    assert "continued" not in result.stdout


def test_download_keeps_exhausted_transient_errors_fatal(
    command: tuple[str, ...], tmp_path: Path
) -> None:
    with _server([503]) as (url, observed):
        result, _ = _run(command, url, tmp_path)
    assert result.returncode != 0
    assert 1 < len(observed) <= 4
    assert "continued" not in result.stdout


def test_download_bounds_stalled_transfers(
    command: tuple[str, ...], tmp_path: Path
) -> None:
    started = time.monotonic()
    with _server([200], stall=True) as (url, observed):
        result, _ = _run(command, url, tmp_path, short=True)
    assert result.returncode == 28, result.stderr
    assert observed
    assert "continued" not in result.stdout
    assert time.monotonic() - started < 5


def test_helper_pins_the_existing_official_asset(helper: ModuleType) -> None:
    asset = helper.CCACHE
    assert asset.name == "ccache-4.14-linux-x86_64-glibc.tar.xz"
    assert asset.url == (
        "https://github.com/ccache/ccache/releases/download/v4.14/" + asset.name
    )
    assert asset.api == (
        "https://api.github.com/repos/ccache/ccache/releases/assets/526043228"
    )
    assert asset.size == 1_344_580
    assert asset.sha256 == (
        "45a91165db7092e67c6208ada03f54700e684c4cd3735f9031de95669ed9272c"
    )


@pytest.fixture
def installer(
    helper: ModuleType, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> Iterator[SimpleNamespace]:
    binary_contents = b"the extracted ccache fixture"
    archive = io.BytesIO()
    with tarfile.open(fileobj=archive, mode="w:xz") as tar:
        member = tarfile.TarInfo("ccache-4.14-linux-x86_64-glibc/ccache")
        member.size = len(binary_contents)
        member.mode = 0o755
        tar.addfile(member, io.BytesIO(binary_contents))
    payload = archive.getvalue()
    # Synthetic bytes replace only the archive's size/digest. The real CLI,
    # upstream URLs, hashing, extraction and executable identity gates run.
    monkeypatch.setattr(
        helper,
        "CCACHE",
        replace(
            helper.CCACHE, size=len(payload), sha256=hashlib.sha256(payload).hexdigest()
        ),
    )
    cache = tmp_path / "compiler-cache"
    cache.mkdir()
    sentinel = cache / "retained-object"
    sentinel.write_bytes(b"existing compiler work")
    monkeypatch.setenv("CCACHE_DIR", str(cache))
    state = SimpleNamespace(
        helper=helper,
        payload=payload,
        transfers=[],
        urls=[],
        installed=False,
        source_version="ccache version 4.14",
        installed_version="ccache version 4.14",
    )

    def run(args: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        assert "GH_TOKEN" not in kwargs["env"]
        assert helper.os.environ["GH_TOKEN"] not in " ".join(args)
        if args[0] == "curl":
            url = args[-1]
            assert url in (helper.CCACHE.url, helper.CCACHE.api)
            state.urls.append(url)
            assert "--location-trusted" not in args
            for option in ("--proto", "--proto-redir"):
                assert args[args.index(option) + 1] == "=https"
            assert int(args[args.index("--max-filesize") + 1]) == len(payload)
            if url == helper.CCACHE.api:
                assert args[args.index("--header") + 1] == "@-"
                assert "Accept: application/octet-stream\n" in kwargs["input"]
                assert (
                    "Authorization: Bearer " + helper.os.environ["GH_TOKEN"]
                ) in kwargs["input"]
            else:
                assert kwargs["input"] is None
            code, body = state.transfers.pop(0)
            Path(args[args.index("--output") + 1]).write_bytes(body)
            return subprocess.CompletedProcess(args, code, "transport result\n")
        if args[0] == "sudo":
            assert args[-1] == "/usr/local/bin/ccache"
            assert Path(args[-2]).read_bytes() == binary_contents
            state.installed = True
            return subprocess.CompletedProcess(args, 0, "")
        assert args[-1] == "--version"
        if args[0] == "/usr/local/bin/ccache":
            assert state.installed
            version = state.installed_version
        else:
            assert Path(args[0]).read_bytes() == binary_contents
            version = state.source_version
        return subprocess.CompletedProcess(args, 0, version + "\n")

    monkeypatch.setattr(subprocess, "run", run)
    yield state
    assert sentinel.read_bytes() == b"existing compiler work"


def test_helper_installs_verified_primary_download(installer: SimpleNamespace) -> None:
    installer.transfers = [(0, installer.payload)]
    assert installer.helper.main() == 0
    assert installer.urls == [installer.helper.CCACHE.url]
    assert installer.installed


@pytest.mark.parametrize("code", [22, 28])
def test_helper_recovers_same_asset_after_transport_failure(
    installer: SimpleNamespace, code: int
) -> None:
    installer.transfers = [(code, b"partial transfer"), (0, installer.payload)]
    assert installer.helper.main() == 0
    assert installer.urls == [installer.helper.CCACHE.url, installer.helper.CCACHE.api]
    assert installer.installed


def test_helper_keeps_both_failed_transport_legs_fatal(
    installer: SimpleNamespace,
) -> None:
    installer.transfers = [(22, b""), (22, b"")]
    assert installer.helper.main() != 0
    assert installer.urls == [installer.helper.CCACHE.url, installer.helper.CCACHE.api]
    assert not installer.installed


@pytest.mark.parametrize("code", [23, 48, 60, 63])
def test_helper_keeps_nontransport_errors_fatal(
    installer: SimpleNamespace, code: int
) -> None:
    installer.transfers = [(code, b"")]
    assert installer.helper.main() != 0
    assert installer.urls == [installer.helper.CCACHE.url]
    assert not installer.installed


@pytest.mark.parametrize("fallback", [False, True])
@pytest.mark.parametrize("corruption", ["size", "digest"])
def test_helper_rejects_corrupt_bytes_before_installing(
    installer: SimpleNamespace,
    fallback: bool,
    corruption: str,
    capsys: pytest.CaptureFixture[str],
) -> None:
    payload = installer.payload
    damaged = (
        payload[:-1] if corruption == "size" else bytes([payload[0] ^ 1]) + payload[1:]
    )
    installer.transfers = ([(22, b"")] if fallback else []) + [(0, damaged)]
    assert installer.helper.main() != 0
    expected_error = "size mismatch" if corruption == "size" else "SHA-256 mismatch"
    assert expected_error in capsys.readouterr().err
    expected = [installer.helper.CCACHE.url]
    if fallback:
        expected.append(installer.helper.CCACHE.api)
    assert installer.urls == expected
    assert not installer.installed


@pytest.mark.parametrize("installed", [False, True])
def test_helper_rejects_wrong_executable_version(
    installer: SimpleNamespace, installed: bool
) -> None:
    installer.transfers = [(0, installer.payload)]
    if installed:
        installer.installed_version = "ccache version 4.13"
    else:
        installer.source_version = "ccache version 4.13"
    assert installer.helper.main() != 0
    assert installer.installed is installed


_REFUSED_STREAM = (
    "  Caused by: client error (SendRequest)\n"
    "  Caused by: http2 error\n"
    "  Caused by: stream error received: refused stream before processing any application logic\n"
)


def _python_download_error(url: str, cause: str) -> str:
    return (
        "error: Request failed after 3 retries\n"
        f"  Caused by: Failed to download {url}\n"
        f"  Caused by: error sending request for url ({url})\n" + cause
    )


def test_helper_pins_the_existing_python_asset(helper: ModuleType) -> None:
    asset = helper.PYTHON
    assert asset.name == (
        "cpython-3.11.14+20260211-x86_64-unknown-linux-gnu-install_only_stripped.tar.gz"
    )
    assert asset.url == (
        "https://github.com/astral-sh/python-build-standalone/releases/download/20260211/"
        "cpython-3.11.14%2B20260211-x86_64-unknown-linux-gnu-install_only_stripped.tar.gz"
    )
    assert asset.api == (
        "https://api.github.com/repos/astral-sh/python-build-standalone/releases/assets/354545489"
    )
    assert asset.size == 31_215_121
    assert asset.sha256 == (
        "1fb6a89daa750f7eff00eb65d901c3c236713812eb9c8c7e07ecdfcd78203c8e"
    )


@pytest.fixture
def python_installer(
    helper: ModuleType, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> Iterator[SimpleNamespace]:
    monkeypatch.setattr(sys, "argv", [sys.argv[0], "python-venv"])
    work = tmp_path / "checkout"
    work.mkdir()
    monkeypatch.chdir(work)
    cache_files = []
    for variable in ("CCACHE_DIR", "UV_PYTHON_INSTALL_DIR", "UV_CACHE_DIR"):
        cache = tmp_path / variable.lower()
        cache.mkdir()
        sentinel = cache / "retained-entry"
        sentinel.write_bytes(b"existing cached work")
        cache_files.append(sentinel)
        monkeypatch.setenv(variable, str(cache))
    monkeypatch.setenv("UV_PYTHON", "3.11")
    members = {
        "python/bin/python3.11": b"fixture interpreter",
        "python/lib/python3.11/encodings/__init__.py": b"retained stdlib",
        "python/lib/libpython3.11.so.1.0": b"retained shared library",
    }
    archive = io.BytesIO()
    with tarfile.open(fileobj=archive, mode="w:gz") as tar:
        for name, data in members.items():
            member = tarfile.TarInfo(name)
            member.size = len(data)
            tar.addfile(member, io.BytesIO(data))
        link = tarfile.TarInfo("python/bin/python3")
        link.type = tarfile.SYMTYPE
        link.linkname = "python3.11"
        tar.addfile(link)
    payload = archive.getvalue()
    monkeypatch.setattr(
        helper,
        "PYTHON",
        replace(
            helper.PYTHON, size=len(payload), sha256=hashlib.sha256(payload).hexdigest()
        ),
    )
    state = SimpleNamespace(
        helper=helper,
        work=work,
        payload=payload,
        original_code=2,
        original_error=_python_download_error(helper.PYTHON.url, _REFUSED_STREAM),
        api_urls=[],
        api_payload=payload,
        recovered_code=0,
        partial_venv=False,
        wrong_base=False,
    )

    def run(args: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        assert "GH_TOKEN" not in kwargs["env"]
        assert helper.os.environ["GH_TOKEN"] not in " ".join(args)
        if args[0] == "uv":
            requested = args[args.index("--python") + 1]
            if requested == "3.11":
                if state.partial_venv or state.original_code == 0:
                    (work / ".venv").mkdir()
                    (work / ".venv/original-state").write_bytes(b"keep original venv")
                return subprocess.CompletedProcess(
                    args, state.original_code, state.original_error
                )
            assert "--no-python-downloads" in args
            executable = Path(requested)
            assert executable.is_absolute() and executable.is_file()
            assert kwargs["env"]["UV_PYTHON"] == str(executable)
            if state.recovered_code == 0:
                (work / ".venv/bin").mkdir(parents=True)
                (work / ".venv/bin/python").symlink_to(executable)
            return subprocess.CompletedProcess(
                args, state.recovered_code, "uv result\n"
            )
        if args[0] == "curl":
            assert args[-1] == helper.PYTHON.api
            assert "--location-trusted" not in args
            assert args[args.index("--header") + 1] == "@-"
            assert "Accept: application/octet-stream\n" in kwargs["input"]
            assert ("Authorization: Bearer " + helper.os.environ["GH_TOKEN"]) in kwargs[
                "input"
            ]
            for option in ("--proto", "--proto-redir"):
                assert args[args.index(option) + 1] == "=https"
            assert int(args[args.index("--max-filesize") + 1]) == len(payload)
            state.api_urls.append(args[-1])
            Path(args[args.index("--output") + 1]).write_bytes(state.api_payload)
            return subprocess.CompletedProcess(args, 0, "")
        executable = Path(args[0])
        real = executable.resolve(strict=True)
        prefix = real.parents[1]
        for name, data in members.items():
            assert (prefix.parent / name).read_bytes() == data
        identity = {
            "implementation": "cpython",
            "version": [3, 11, 14],
            "base_prefix": str(prefix),
            "base_executable": str(real),
        }
        if state.wrong_base and executable == work / ".venv/bin/python":
            identity["base_prefix"] = str(work / "different-python")
        return subprocess.CompletedProcess(args, 0, json.dumps(identity))

    monkeypatch.setattr(subprocess, "run", run)
    yield state
    assert helper.os.environ["UV_PYTHON"] == "3.11"
    for sentinel in cache_files:
        assert sentinel.read_bytes() == b"existing cached work"


@pytest.mark.parametrize(
    "cause",
    ["  Caused by: HTTP status server error (504 Gateway Timeout)\n", _REFUSED_STREAM],
    ids=["gateway-timeout", "pre-response-http2-refusal"],
)
def test_python_cli_recovers_only_the_same_complete_runtime(
    python_installer: SimpleNamespace, cause: str
) -> None:
    state = python_installer
    state.original_error = _python_download_error(state.helper.PYTHON.url, cause)
    assert state.helper.main() == 0
    assert state.api_urls == [state.helper.PYTHON.api]
    executable = (state.work / ".venv/bin/python").resolve(strict=True)
    prefix = executable.parents[1]
    assert prefix.is_relative_to(Path(state.helper.os.environ["RUNNER_TEMP"]))
    assert (
        prefix / "lib/python3.11/encodings/__init__.py"
    ).read_bytes() == b"retained stdlib"
    assert (
        prefix / "lib/libpython3.11.so.1.0"
    ).read_bytes() == b"retained shared library"
    assert (prefix / "bin/python3").resolve(strict=True) == executable


def test_python_cli_preserves_normal_uv_success(
    python_installer: SimpleNamespace,
) -> None:
    state = python_installer
    state.original_code = 0
    state.original_error = (
        "Using CPython 3.11.14\nCreating virtual environment at: .venv\n"
    )
    assert state.helper.main() == 0
    assert not state.api_urls
    assert (state.work / ".venv/original-state").read_bytes() == b"keep original venv"


@pytest.mark.parametrize(
    ("cause", "code", "url_kind"),
    [
        pytest.param(_REFUSED_STREAM, 2, "different-version", id="different-archive"),
        pytest.param(_REFUSED_STREAM, 2, "suffix", id="url-prefix-is-not-identity"),
        pytest.param(_REFUSED_STREAM, 1, "exact", id="different-exit-code"),
        pytest.param(
            _REFUSED_STREAM.replace("  Caused by: client error (SendRequest)\n", ""),
            2,
            "exact",
            id="missing-send-request",
        ),
        pytest.param(
            _REFUSED_STREAM.replace("  Caused by: http2 error\n", ""),
            2,
            "exact",
            id="missing-http2",
        ),
        pytest.param(
            "  Caused by: client error (SendRequest)\n  Caused by: http2 error\n",
            2,
            "exact",
            id="missing-refused-stream",
        ),
        pytest.param(
            "\n".join(reversed(_REFUSED_STREAM.splitlines())) + "\n",
            2,
            "exact",
            id="wrong-cause-order",
        ),
        pytest.param(
            _REFUSED_STREAM.replace(
                "refused stream before processing any application logic",
                "protocol error",
            ),
            2,
            "exact",
            id="other-http2-error",
        ),
        pytest.param(
            "  Caused by: invalid peer certificate: UnknownIssuer\n",
            2,
            "exact",
            id="certificate",
        ),
        pytest.param(
            "  Caused by: Permission denied (os error 13)\n",
            2,
            "exact",
            id="permission",
        ),
        pytest.param(
            "  Caused by: No space left on device (os error 28)\n",
            2,
            "exact",
            id="local-io",
        ),
        pytest.param("  Caused by: 403 Forbidden\n", 2, "exact", id="permission-http"),
        pytest.param(
            "  Caused by: 503 Service Unavailable\n", 2, "exact", id="other-http"
        ),
        pytest.param(
            "  Caused by: Failed to extract archive\n", 2, "exact", id="extract"
        ),
    ],
)
def test_python_cli_keeps_unqualified_failures_fatal(
    python_installer: SimpleNamespace, cause: str, code: int, url_kind: str
) -> None:
    state = python_installer
    url = state.helper.PYTHON.url
    if url_kind == "different-version":
        url = url.replace("3.11.14", "3.11.13")
    elif url_kind == "suffix":
        url += ".other"
    state.original_code = code
    state.original_error = _python_download_error(url, cause)
    assert state.helper.main() == code
    assert not state.api_urls
    assert not (state.work / ".venv").exists()
    assert not list(Path(state.helper.os.environ["RUNNER_TEMP"]).iterdir())


@pytest.mark.parametrize("corruption", ["size", "digest"])
def test_python_cli_authenticates_recovered_archive(
    python_installer: SimpleNamespace,
    corruption: str,
    capsys: pytest.CaptureFixture[str],
) -> None:
    state = python_installer
    payload = state.payload
    state.api_payload = (
        payload[:-1] if corruption == "size" else bytes([payload[0] ^ 1]) + payload[1:]
    )
    assert state.helper.main() != 0
    expected_error = "size mismatch" if corruption == "size" else "SHA-256 mismatch"
    assert expected_error in capsys.readouterr().err
    assert not (state.work / ".venv").exists()


def test_python_cli_keeps_recovered_venv_errors_fatal(
    python_installer: SimpleNamespace,
) -> None:
    state = python_installer
    state.recovered_code = 2
    assert state.helper.main() != 0
    assert state.api_urls == [state.helper.PYTHON.api]
    assert not (state.work / ".venv").exists()


def test_python_cli_checks_the_recovered_venv_base(
    python_installer: SimpleNamespace, capsys: pytest.CaptureFixture[str]
) -> None:
    state = python_installer
    state.wrong_base = True
    assert state.helper.main() != 0
    assert "Venv does not use the verified Python base" in capsys.readouterr().err
    assert (state.work / ".venv/bin/python").resolve(strict=True).is_file()


def test_python_cli_preserves_a_partial_original_venv(
    python_installer: SimpleNamespace,
) -> None:
    state = python_installer
    state.partial_venv = True
    assert state.helper.main() != 0
    assert not state.api_urls
    assert (state.work / ".venv/original-state").read_bytes() == b"keep original venv"
