"""Exercise official compiler-cache download retry and fatal-error behavior."""

from __future__ import annotations

import re
import shlex
import shutil
import subprocess
import threading
import time
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]


def _download_commands() -> list[tuple[str, ...]]:
    workflow = (ROOT / ".github/workflows/ci.yml").read_text()
    logical_lines = re.sub(r"\\\n\s*", " ", workflow).splitlines()
    commands = [
        tuple(shlex.split(line))
        for line in logical_lines
        if line.lstrip().startswith("curl ")
        and "github.com/ccache/ccache/releases/download/" in line
    ]
    assert commands, "no compiler-cache download is covered"
    # Equivalent downloads in separate jobs need one transport execution each.
    return list(dict.fromkeys(commands))


@contextmanager
def _server(statuses: list[int], *, stall: bool = False):
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


def _run(command: tuple[str, ...], url: str, tmp_path: Path, *, short=False):
    curl = shutil.which("curl")
    if curl is None:
        pytest.skip("requires the curl used by the CI download")
    args = list(command)
    args[0] = curl
    destination = tmp_path / "archive"
    args[args.index("--output") + 1] = str(destination)
    args[-1] = url
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


@pytest.mark.parametrize("command", _download_commands())
def test_download_has_finite_transport_budget(command: tuple[str, ...]) -> None:
    def value(option: str) -> float:
        return float(command[command.index(option) + 1])

    assert 0 < value("--retry") <= 3
    assert 0 < value("--connect-timeout") <= value("--max-time")
    assert 0 < value("--retry-delay") <= value("--retry-max-time")
    # Curl may start its final transfer just before the retry clock expires.
    assert 0 < value("--retry-max-time") + value("--max-time") <= 120


@pytest.mark.parametrize("command", _download_commands())
def test_download_recovers_after_transient_server_errors(
    command: tuple[str, ...], tmp_path: Path
) -> None:
    with _server([500, 503, 200]) as (url, observed):
        result, destination = _run(command, url, tmp_path)
    assert result.returncode == 0, result.stderr
    assert observed[-1] == 200 and len(observed) > 1
    assert destination.read_bytes() == b"verified archive"
    assert result.stdout == "continued"


@pytest.mark.parametrize("command", _download_commands())
@pytest.mark.parametrize("status", [403, 404])
def test_download_keeps_permanent_http_errors_fatal(
    command: tuple[str, ...], status: int, tmp_path: Path
) -> None:
    with _server([status]) as (url, observed):
        result, _ = _run(command, url, tmp_path)
    assert result.returncode != 0
    assert observed == [status]
    assert "continued" not in result.stdout


@pytest.mark.parametrize("command", _download_commands())
def test_download_keeps_exhausted_transient_errors_fatal(
    command: tuple[str, ...], tmp_path: Path
) -> None:
    with _server([503]) as (url, observed):
        result, _ = _run(command, url, tmp_path)
    assert result.returncode != 0
    assert 1 < len(observed) <= 4
    assert "continued" not in result.stdout


@pytest.mark.parametrize("command", _download_commands())
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
