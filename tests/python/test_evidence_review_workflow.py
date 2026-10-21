"""Execute evidence review on real synthetic PR merges and the real policy CLI."""

from __future__ import annotations

import json
import os
import shlex
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize(
    "case",
    [
        "stale-base",
        "small",
        "shrink",
        "over-added",
        "over-modified",
        "aggregate",
        "linear",
        "wrong-head",
        "octopus",
        "missing-base",
        "unrelated-base",
        "rollback",
        "empty-base",
        "empty-head",
    ],
)
def test_evidence_review_uses_verified_test_merge(tmp_path: Path, case: str) -> None:
    bash = shutil.which("bash")
    if bash is None and os.name == "nt":
        candidate = Path("C:/Program Files/Git/bin/bash.exe")
        bash = str(candidate) if candidate.is_file() else None
    if bash is None or shutil.which("git") is None:
        pytest.skip("requires Git and Bash")
    env = {
        **os.environ,
        # Isolated fixture identities; never configure the user's repository.
        "GIT_AUTHOR_NAME": "test",
        "GIT_AUTHOR_EMAIL": "test@example.invalid",
        "GIT_COMMITTER_NAME": "test",
        "GIT_COMMITTER_EMAIL": "test@example.invalid",
        "GIT_CONFIG_NOSYSTEM": "1",
        "GIT_CONFIG_GLOBAL": os.devnull,
        "GIT_NO_LAZY_FETCH": "1",
        "GIT_TERMINAL_PROMPT": "0",
        "PYTHONUTF8": "1",
    }
    env.pop("BASH_ENV", None)

    def git(*args: str) -> str:
        return subprocess.run(
            ["git", "-c", "core.autocrlf=false", *args],
            cwd=tmp_path,
            env=env,
            check=True,
            capture_output=True,
            text=True,
            timeout=20,
        ).stdout.strip()

    policy = json.loads(
        (ROOT / "benchmarks/evidence-policy.json").read_text(encoding="utf-8")
    )
    # Keep the real file/change caps; unrelated production exceptions are not
    # copied to an empty fixture repository. The aggregate-negative is stricter.
    rules = {
        key: policy[key]
        for key in (
            "schema",
            "review_size_bytes",
            "change_review_max_bytes",
            "benchmark_results_max_bytes",
        )
    }
    rules["exceptions"] = {}
    if case == "aggregate":
        rules["benchmark_results_max_bytes"] = 16
    policy_path = tmp_path / "benchmarks/evidence-policy.json"
    evidence = tmp_path / "benchmarks/results/synthetic"
    evidence.mkdir(parents=True)
    policy_path.write_text(json.dumps(rules), encoding="utf-8")
    for index in range(3):
        (evidence / f"old-{index}.md").write_text("baseline\n" * 10, encoding="utf-8")
    git("init", "-b", "target")
    git("add", ".")
    git("commit", "-qm", "synthetic fixture baseline")
    event_base = git("rev-parse", "HEAD")
    git("checkout", "-b", "candidate")
    (tmp_path / "source-change").write_text("candidate\n", encoding="utf-8")
    size = rules["change_review_max_bytes"] // 3 + 64
    if case in {"over-added", "over-modified"}:
        for index in range(3):
            name = f"new-{index}.md" if case == "over-added" else f"old-{index}.md"
            (evidence / name).write_bytes(b"e" * size)
    elif case in {"small", "shrink"}:
        name = "new-small.md" if case == "small" else "old-0.md"
        (evidence / name).write_text("small\n", encoding="utf-8")
    git("add", ".")
    git("commit", "-qm", "synthetic candidate")
    head = git("rev-parse", "HEAD")
    git("checkout", "target")
    for index in range(3):
        (evidence / f"upstream-{index}.md").write_bytes(b"u" * size)
    git("add", ".")
    git("commit", "-qm", "synthetic advanced target evidence")
    tested_base = git("rev-parse", "HEAD")
    if case == "octopus":
        git("checkout", "-b", "third", event_base)
        (tmp_path / "third-change").write_text("third\n", encoding="utf-8")
        git("add", ".")
        git("commit", "-qm", "synthetic third parent")
        git("checkout", "target")
        git("merge", "--no-ff", "candidate", "third", "-m", "synthetic octopus")
    elif case == "linear":
        git("checkout", "candidate")
    else:
        git("merge", "--no-ff", "candidate", "-m", "synthetic PR test merge")

    if case == "missing-base":
        event_base = "0" * 40
    elif case == "unrelated-base":
        event_base = git(
            "commit-tree",
            git("rev-parse", tested_base + "^{tree}"),
            "-m",
            "synthetic unrelated root",
        )
    elif case == "rollback":
        event_base = git(
            "commit-tree",
            git("rev-parse", tested_base + "^{tree}"),
            "-p",
            tested_base,
            "-m",
            "synthetic later target",
        )
    elif case == "empty-base":
        event_base = ""
    env["EVIDENCE_BASE"] = event_base
    env["EVIDENCE_HEAD"] = (
        tested_base if case == "wrong-head" else "" if case == "empty-head" else head
    )

    # This thin entry records whether the step reached evidence reading, then
    # runs the actual unchanged public CLI/policy. It never fabricates success.
    marker = tmp_path / "evidence-cli-entered"
    tools = tmp_path / "tools"
    tools.mkdir()
    (tools / "evidence.py").write_text(
        "import runpy,sys\nfrom pathlib import Path\n"
        f"Path({str(marker)!r}).write_text('entered')\n"
        f"sys.argv=[{str(ROOT / 'tools/evidence.py')!r},"
        f"'--root',{str(tmp_path)!r},*sys.argv[1:]]\n"
        f"runpy.run_path({str(ROOT / 'tools/evidence.py')!r},run_name='__main__')\n",
        encoding="utf-8",
    )
    binaries = tmp_path / "bin"
    binaries.mkdir()
    python = sys.executable.replace("\\", "/")
    shim = binaries / "python3"
    shim.write_bytes(("#!/bin/sh\nexec " + shlex.quote(python) + ' "$@"\n').encode())
    shim.chmod(0o755)
    env["PATH"] = str(binaries) + os.pathsep + env.get("PATH", "")
    result = subprocess.run(
        [bash, (ROOT / "tools/check_evidence_change.sh").as_posix()],
        cwd=tmp_path,
        env=env,
        check=False,
        capture_output=True,
        text=True,
        timeout=45,
    )
    invalid = {
        "linear",
        "wrong-head",
        "octopus",
        "missing-base",
        "unrelated-base",
        "rollback",
        "empty-base",
        "empty-head",
    }
    if case in invalid:
        assert result.returncode != 0
        assert not marker.exists(), (
            "Invalid identity must fail before evidence CLI entry"
        )
    else:
        assert marker.exists(), result.stderr
        if case in {"over-added", "over-modified", "aggregate"}:
            assert result.returncode != 0
            assert "exceeds" in result.stdout or "aggregate budget" in result.stdout
        else:
            assert result.returncode == 0, result.stdout + result.stderr
    if case == "stale-base":
        assert "Evidence change: 0 files, 0 bytes" in result.stdout
        # The former event-base invocation must still reject the unrelated
        # target bytes. This negative proves the repaired context is material.
        wrong = subprocess.run(
            [
                bash,
                "-eu",
                "-c",
                'python3 tools/evidence.py check-change --base "$EVIDENCE_BASE"',
            ],
            cwd=tmp_path,
            env=env,
            check=False,
            capture_output=True,
            text=True,
            timeout=45,
        )
        assert wrong.returncode != 0
        assert "new/modified bytes exceeds" in wrong.stdout
