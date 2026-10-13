"""Exercise host dependency setup without network or system package mutations."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
pytestmark = pytest.mark.skipif(
    os.name != "posix" or shutil.which("bash") is None,
    reason="manylinux preparation requires POSIX shell commands",
)


def _executable(path: Path, body: str) -> None:
    path.write_text(f"#!{sys.executable}\n" + textwrap.dedent(body))
    path.chmod(0o755)


def _prepare(
    tmp_path: Path, *, cuda_repo: bool, package_status: int = 0
) -> subprocess.CompletedProcess[str]:
    project = tmp_path / "project with spaces"
    ci = project / "python/ci"
    ci.mkdir(parents=True)
    for name in ("prepare-wheel-build.sh", "install-ccache.sh"):
        shutil.copy2(ROOT / "python/ci" / name, ci / name)
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    _executable(bin_dir / "ccache", 'print("ccache version 4.14")\n')
    _executable(
        bin_dir / "dnf",
        """
        import fnmatch
        import json
        import os
        import sys
        from pathlib import Path

        args = sys.argv[1:]
        assert "install" in args
        packages = {arg for arg in args[args.index("install") + 1:]
                    if not arg.startswith("-")}
        assert packages == {"openssl-devel", "libarchive-devel"}
        repos = {"baseos", "appstream", "extras", "powertools"}
        if os.environ["TEST_CUDA_REPO"] == "1":
            repos.add("cuda-rhel8-x86_64")
        for index, arg in enumerate(args):
            if arg == "--disablerepo":
                patterns = args[index + 1].split(",")
            elif arg.startswith("--disablerepo="):
                patterns = arg.split("=", 1)[1].split(",")
            else:
                continue
            repos = {repo for repo in repos
                     if not any(fnmatch.fnmatchcase(repo, p) for p in patterns)}
        Path(os.environ["TEST_REPOS"]).write_text(json.dumps(sorted(repos)))
        if "cuda-rhel8-x86_64" in repos:
            sys.exit("NVIDIA metadata unavailable: HTTP 404")
        if not {"baseos", "appstream", "extras", "powertools"} <= repos:
            sys.exit("Required distribution repositories are unavailable")
        sys.exit(int(os.environ["TEST_PACKAGE_STATUS"]))
        """,
    )
    _executable(
        bin_dir / "python",
        """
        import os
        import shutil
        import sys
        from pathlib import Path

        with Path(os.environ["TEST_PYTHON_CALLS"]).open("a") as stream:
            stream.write(repr(sys.argv[1:]) + "\\n")
        if sys.argv[1:3] == ["-m", "venv"]:
            target = Path(sys.argv[3]) / "bin/python"
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(__file__, target)
        """,
    )
    result = subprocess.run(
        ["bash", str(ci / "prepare-wheel-build.sh")],
        cwd=tmp_path,
        env={
            **os.environ,
            "PATH": f"{bin_dir}{os.pathsep}{os.environ['PATH']}",
            "CCACHE_DIR": str(tmp_path / "ccache"),
            "TEST_CUDA_REPO": str(int(cuda_repo)),
            "TEST_PACKAGE_STATUS": str(package_status),
            "TEST_REPOS": str(tmp_path / "repos.json"),
            "TEST_PYTHON_CALLS": str(tmp_path / "python-calls.txt"),
        },
        capture_output=True,
        text=True,
        timeout=20,
        check=False,
    )
    return result


@pytest.mark.parametrize("cuda_repo", [False, True])
def test_wheel_host_packages_do_not_require_cuda_repository(
    tmp_path: Path, cuda_repo: bool
) -> None:
    result = _prepare(tmp_path, cuda_repo=cuda_repo)
    assert result.returncode == 0, result.stdout + result.stderr
    assert set(json.loads((tmp_path / "repos.json").read_text())) == {
        "baseos",
        "appstream",
        "extras",
        "powertools",
    }
    calls = (tmp_path / "python-calls.txt").read_text()
    assert "scipy-openblas32==" in calls
    assert "resolve-openblas-wheel.py" in calls


def test_wheel_required_host_dependency_failure_stops_preparation(
    tmp_path: Path,
) -> None:
    result = _prepare(tmp_path, cuda_repo=True, package_status=42)
    assert result.returncode == 42, result.stdout + result.stderr
    assert not (tmp_path / "python-calls.txt").exists()
