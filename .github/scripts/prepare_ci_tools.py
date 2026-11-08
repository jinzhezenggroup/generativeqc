"""Recover exact CI tool downloads without changing compiler or Python contracts."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
import tarfile
import tempfile
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlsplit


@dataclass(frozen=True)
class Asset:
    """One pinned upstream release asset, shared by both transport paths."""

    name: str
    url: str
    api: str
    size: int
    sha256: str


CCACHE = Asset(
    "ccache-4.14-linux-x86_64-glibc.tar.xz",
    "https://github.com/ccache/ccache/releases/download/v4.14/"
    "ccache-4.14-linux-x86_64-glibc.tar.xz",
    "https://api.github.com/repos/ccache/ccache/releases/assets/526043228",
    1_344_580,
    "45a91165db7092e67c6208ada03f54700e684c4cd3735f9031de95669ed9272c",
)
PYTHON = Asset(
    "cpython-3.11.14+20260211-x86_64-unknown-linux-gnu-install_only_stripped.tar.gz",
    "https://github.com/astral-sh/python-build-standalone/releases/download/20260211/"
    "cpython-3.11.14%2B20260211-x86_64-unknown-linux-gnu-install_only_stripped.tar.gz",
    "https://api.github.com/repos/astral-sh/python-build-standalone/releases/assets/354545489",
    31_215_121,
    "1fb6a89daa750f7eff00eb65d901c3c236713812eb9c8c7e07ecdfcd78203c8e",
)
CURL_OPTIONS = [
    "--fail",
    "--silent",
    "--show-error",
    "--location",
    "--retry",
    "3",
    "--retry-delay",
    "2",
    "--retry-max-time",
    "60",
    "--connect-timeout",
    "15",
    "--max-time",
    "60",
    "--proto",
    "=https",
    "--proto-redir",
    "=https",
]
# DNS/connect/partial/HTTP/timeout/empty/send/receive/HTTP2 transfer errors.
# Local I/O, usage, and certificate-verification failures must not be retried.
TRANSFER_ERRORS = {6, 7, 18, 22, 28, 52, 55, 56, 92}


class PreparationError(RuntimeError):
    """A failed tool prerequisite; never permission to skip a downstream gate."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise PreparationError(message)


def _invoke(
    args: list[str],
    *,
    input_text: str | None = None,
    env: dict[str, str] | None = None,
    timeout: float | None = None,
) -> subprocess.CompletedProcess[str]:
    child_env = dict(os.environ if env is None else env)
    # Only the authenticated API request needs the token, supplied via stdin.
    child_env.pop("GH_TOKEN", None)
    try:
        return subprocess.run(
            args,
            input=input_text,
            env=child_env,
            timeout=timeout,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        # Do not include subprocess arguments, headers, or exception context.
        raise PreparationError(f"Could not complete {Path(args[0]).name}") from None


def _runner_temp() -> Path:
    root = Path(os.environ.get("RUNNER_TEMP", ""))
    _require(
        root.is_absolute() and root.is_dir(),
        "RUNNER_TEMP must be an existing absolute directory",
    )
    return root.resolve()


def _curl(
    asset: Asset, payload: Path, *, api: bool
) -> subprocess.CompletedProcess[str]:
    args = [
        "curl",
        *CURL_OPTIONS,
        "--max-filesize",
        str(asset.size),
        "--output",
        str(payload),
    ]
    headers = None
    if api:
        target = urlsplit(asset.api)
        _require(
            target.scheme == "https" and target.netloc == "api.github.com",
            "Release fallback must use the trusted GitHub API host",
        )
        token = os.environ.get("GH_TOKEN", "")
        _require(
            bool(token) and not any(c in token for c in "\r\n\0"),
            "A valid read-only GH_TOKEN is required",
        )
        # Keep the token out of argv. curl strips Authorization on cross-host
        # redirects; never enable --location-trusted or verbose header logging.
        args += ["--header", "@-"]
        headers = (
            "Accept: application/octet-stream\n"
            f"Authorization: Bearer {token}\n"
            "X-GitHub-Api-Version: 2022-11-28\n"
        )
    args.append(asset.api if api else asset.url)
    return _invoke(args, input_text=headers)


def _download(asset: Asset, directory: Path, *, api_only: bool = False) -> Path:
    payload = directory / asset.name
    if not api_only:
        original = _curl(asset, payload, api=False)
        if original.returncode == 0:
            _verify_asset(asset, payload)
            return payload
        print(original.stdout, end="")
        _require(
            original.returncode in TRANSFER_ERRORS,
            f"Original download failed outside the recoverable transport boundary (curl exit {original.returncode})",
        )
    print(f"::notice::Recovering the same {asset.name} through the official asset API")
    payload.unlink(missing_ok=True)
    fallback = _curl(asset, payload, api=True)
    _require(
        fallback.returncode == 0,
        f"Official asset download failed (curl exit {fallback.returncode})",
    )
    _verify_asset(asset, payload)
    return payload


def _verify_asset(asset: Asset, payload: Path) -> None:
    _require(
        payload.is_file() and payload.stat().st_size == asset.size,
        f"{asset.name}: size mismatch",
    )
    with payload.open("rb") as stream:
        actual = hashlib.file_digest(stream, "sha256").hexdigest()
    _require(actual == asset.sha256, f"{asset.name}: SHA-256 mismatch")
    print(f"Verified {asset.name}: {asset.size} bytes, sha256={actual}")


def _extract(payload: Path, directory: Path) -> None:
    try:
        with tarfile.open(payload) as archive:
            archive.extractall(directory, filter="data")
    except (OSError, tarfile.TarError):
        raise PreparationError(f"Could not extract {payload.name}") from None


def _ccache() -> int:
    _require(
        os.environ.get("CCACHE_VERSION") == "4.14",
        "CCACHE_VERSION must remain exactly 4.14",
    )
    with tempfile.TemporaryDirectory(
        prefix="ci-ccache-", dir=_runner_temp()
    ) as temporary:
        directory = Path(temporary)
        _extract(_download(CCACHE, directory), directory)
        binary = directory / "ccache-4.14-linux-x86_64-glibc/ccache"
        version = _invoke([str(binary), "--version"], timeout=30)
        _require(
            version.returncode == 0
            and version.stdout.splitlines()[:1] == ["ccache version 4.14"],
            "Recovered ccache executable version differs from 4.14",
        )
        installed = _invoke(
            ["sudo", "install", "-m", "0755", str(binary), "/usr/local/bin/ccache"],
            timeout=30,
        )
        _require(installed.returncode == 0, "Could not install verified ccache")
        version = _invoke(["/usr/local/bin/ccache", "--version"], timeout=30)
        _require(
            version.returncode == 0
            and version.stdout.splitlines()[:1] == ["ccache version 4.14"],
            "Installed ccache executable version differs from 4.14",
        )
        print(version.stdout, end="")
    return 0


def _python_info(executable: Path) -> dict[str, object]:
    code = (
        "import json,os,sys; print(json.dumps({"
        "'implementation':sys.implementation.name,"
        "'version':list(sys.version_info[:3]),"
        "'base_prefix':os.path.realpath(sys.base_prefix),"
        "'base_executable':os.path.realpath(sys._base_executable)}))"
    )
    result = _invoke([str(executable), "-I", "-c", code], timeout=30)
    _require(
        result.returncode == 0, "Recovered Python could not report its runtime identity"
    )
    try:
        identity = json.loads(result.stdout)
    except ValueError:
        raise PreparationError(
            "Recovered Python returned an invalid runtime identity"
        ) from None
    _require(isinstance(identity, dict), "Invalid Python runtime identity")
    return identity


def _python_venv() -> int:
    venv = Path.cwd() / ".venv"
    _require(not os.path.lexists(venv), "Refusing to overwrite an existing .venv")
    original = _invoke(["uv", "venv", "--python", "3.11"])
    print(original.stdout, end="")
    if original.returncode == 0:
        return 0
    if not (
        original.returncode == 2
        and PYTHON.url in original.stdout
        and "504 Gateway Timeout" in original.stdout
    ):
        return (
            original.returncode
            if original.returncode > 0
            else 128 - original.returncode
        )
    _require(
        not os.path.lexists(venv),
        "Failed uv download left .venv; refusing to overwrite it",
    )
    # A venv uses this interpreter and its complete stdlib/shared libraries.
    # Keep the entire installation until the runner ends, outside uv's caches.
    directory = Path(tempfile.mkdtemp(prefix="ci-python-3.11.14-", dir=_runner_temp()))
    _extract(_download(PYTHON, directory, api_only=True), directory)
    prefix = (directory / "python").resolve()
    executable = prefix / "bin/python3.11"
    expected = {
        "implementation": "cpython",
        "version": [3, 11, 14],
        "base_prefix": str(prefix),
        "base_executable": str(executable.resolve()),
    }
    _require(
        _python_info(executable) == expected,
        "Recovered Python installation identity differs",
    )
    recovered = _invoke(
        ["uv", "venv", "--python", str(executable), "--no-python-downloads"],
        env={**os.environ, "UV_PYTHON": str(executable)},
    )
    print(recovered.stdout, end="")
    _require(
        recovered.returncode == 0, "uv could not create the venv from verified Python"
    )
    _require(
        _python_info(venv / "bin/python") == expected,
        "Venv does not use the verified Python base",
    )
    print(f"Verified CPython 3.11.14 venv base: {prefix}")
    return 0


def main() -> int:
    """Prepare only the requested prerequisite; downstream gates remain required."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("ccache", "python-venv"))
    args = parser.parse_args()
    try:
        _require(
            platform.system() == "Linux" and platform.machine() == "x86_64",
            "Only Linux x86_64 assets are pinned",
        )
        return _ccache() if args.command == "ccache" else _python_venv()
    except (PreparationError, OSError) as error:
        print(f"::error::{error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
