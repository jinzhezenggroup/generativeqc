"""Behavioral controls for native CLI archive and required aggregate admission."""

from __future__ import annotations

import ast
import os
import re
import shutil
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

from tools.package_native_cli import package_native_cli

ROOT = Path(__file__).resolve().parents[2]


def _job(name: str) -> str:
    workflow = (ROOT / ".github/workflows/ci.yml").read_text()
    body = workflow.split(f"\n  {name}:\n", 1)[1]
    return re.split(r"\n  [a-zA-Z0-9_-]+:\n", body, maxsplit=1)[0]


def _evaluate(node: ast.AST) -> str | bool:
    """Interpret string equality and short-circuit operators without executing code."""
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    if (
        isinstance(node, ast.Compare)
        and len(node.ops) == 1
        and isinstance(node.ops[0], ast.Eq)
    ):
        return _evaluate(node.left) == _evaluate(node.comparators[0])
    if isinstance(node, ast.BoolOp) and isinstance(node.op, (ast.And, ast.Or)):
        result = _evaluate(node.values[0])
        for value in node.values[1:]:
            if (isinstance(node.op, ast.And) and result) or (
                isinstance(node.op, ast.Or) and not result
            ):
                result = _evaluate(value)
        return result
    raise AssertionError(f"Unsupported workflow expression: {ast.dump(node)}")


@pytest.mark.parametrize(
    "event", ["schedule", "pull_request", "merge_group", "workflow_dispatch"]
)
@pytest.mark.parametrize("result", ["success", "skipped", "failure", "cancelled"])
def test_aggregate_only_admits_intentional_scheduled_artifact_skip(
    event: str, result: str
) -> None:
    job = _job("pass")
    needed = re.search(r"needs: \[([^\]]+)\]", job)
    assert needed is not None
    assert "native-cli-artifact" in {
        name.strip() for name in needed.group(1).split(",")
    }
    allowed = re.search(r"allowed-skips: \$\{\{ (.*?) \}\}", job)
    assert allowed is not None
    # Evaluate the actual event-dependent input passed to the pinned alls-green
    # action, whose contract accepts skips only for explicitly named jobs.
    expression = (
        allowed.group(1)
        .replace("github.event_name", repr(event))
        .replace("&&", " and ")
        .replace("||", " or ")
    )
    allowed_skips = _evaluate(ast.parse(expression, mode="eval").body)
    assert isinstance(allowed_skips, str)
    assert {name.strip() for name in allowed_skips.split(",") if name.strip()} == (
        {"native-cli-artifact"} if event == "schedule" else set()
    )
    accepted = result == "success" or (
        result == "skipped" and "native-cli-artifact" in allowed_skips.split(",")
    )
    assert accepted == (
        result == "success" or (event == "schedule" and result == "skipped")
    )


@pytest.mark.skipif(sys.platform != "linux", reason="Distribution targets Linux ELF")
@pytest.mark.parametrize("mismatch", [False, True])
def test_archive_verification_uses_installed_bytes(
    tmp_path: Path, mismatch: bool
) -> None:
    prefix = tmp_path / "build/native-cli-prefix"
    (prefix / "bin").mkdir(parents=True)
    installed = prefix / "bin/generativeqc"
    shutil.copyfile("/bin/true", installed)
    installed.chmod(0o755)
    build_cli = tmp_path / "build/cuda-release-cli-sm120/generativeqc"
    build_cli.parent.mkdir()
    # Model the distinct ELF bytes caused by CMake's install-time RUNPATH edit.
    shutil.copyfile("/bin/false", build_cli)
    build_cli.chmod(0o755)
    source = tmp_path / "source"
    (source / "LICENSES").mkdir(parents=True)
    for name in ("LICENSE", "THIRD_PARTY_NOTICES.md"):
        (source / name).write_text("fixture\n")
    package_prefix = prefix
    if mismatch:
        package_prefix = tmp_path / "different-prefix"
        (package_prefix / "bin").mkdir(parents=True)
        shutil.copyfile(build_cli, package_prefix / "bin/generativeqc")
    package_native_cli(
        prefix=package_prefix,
        source_root=source,
        output_dir=tmp_path / "dist",
        version="0.1.0",
        commit="a" * 40,
        release=False,
        source_date_epoch=1,
    )
    step = (
        _job("native-cli-artifact")
        .split("      - name: Verify packaged executable and report byte sizes\n", 1)[1]
        .split("      - name:", 1)[0]
    )
    script = textwrap.dedent(step.split("        run: |\n", 1)[1])
    summary = tmp_path / "summary.md"
    result = subprocess.run(
        ["bash", "-c", script],
        cwd=tmp_path,
        env={
            **os.environ,
            "GITHUB_SHA": "a" * 40,
            "GITHUB_STEP_SUMMARY": str(summary),
            "TMPDIR": str(tmp_path),
        },
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    if mismatch:
        assert result.returncode != 0
        assert not summary.exists()
    else:
        assert result.returncode == 0, result.stdout + result.stderr
        assert str(installed.stat().st_size) in summary.read_text()
