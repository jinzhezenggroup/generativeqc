"""Exercise publishing guards without creating a tag, Release, or network request."""

from __future__ import annotations

import hashlib
import os
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]


def _release_job() -> str:
    workflow = (ROOT / ".github/workflows/ci.yml").read_text()
    return workflow.split("  native-cli-release:\n", 1)[1].split("\n  pass:", 1)[0]


@pytest.mark.parametrize(
    ("event", "ref", "artifact", "expected"),
    [
        ("push", "refs/tags/v0.1.0", "success", True),
        ("push", "refs/heads/master", "success", False),
        ("workflow_dispatch", "refs/tags/v0.1.0", "success", False),
        ("workflow_dispatch", "refs/heads/master", "success", False),
        ("pull_request", "refs/pull/1/merge", "success", False),
        ("merge_group", "refs/heads/gh-readonly-queue/master/pr-1", "success", False),
        ("schedule", "refs/heads/master", "success", False),
        ("push", "refs/tags/v0.1.0", "failure", False),
    ],
)
def test_release_admission_matrix(
    event: str, ref: str, artifact: str, expected: bool
) -> None:
    expression = _release_job().split("    if: >-\n", 1)[1].split("    runs-on:", 1)[0]
    # Interpret the actual conjunction rather than duplicate its expected shape.
    # Unknown expressions fail so a workflow change cannot silently escape this matrix.
    values = {
        "github.event_name == 'push'": event == "push",
        "startsWith(github.ref, 'refs/tags/v')": ref.startswith("refs/tags/v"),
        "needs.native-cli-artifact.result == 'success'": artifact == "success",
    }
    conditions = [values[atom.strip()] for atom in expression.split("&&")]
    admitted = all(conditions)
    assert admitted is expected


@pytest.mark.parametrize(
    ("current_commit", "exists", "missing", "expected", "published"),
    [
        ("a" * 40, False, False, 0, True),
        ("b" * 40, False, False, 1, False),
        ("a" * 40, False, True, 1, False),
        ("b" * 40, True, False, 0, False),
    ],
)
def test_publisher_rechecks_tag_and_never_overwrites(
    tmp_path: Path,
    current_commit: str,
    exists: bool,
    missing: bool,
    expected: int,
    published: bool,
) -> None:
    job = _release_job()
    step = job.split(
        "      - name: Publish only if this tagged version has no Release\n", 1
    )[1]
    script = "\n".join(
        line[10:] for line in step.split("        run: |\n", 1)[1].splitlines()
    )
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    git = bin_dir / "git"
    git.write_text(
        "#!/bin/sh\n"
        'printf "%s\\n" "$*" >> "$GIT_CALLS"\n'
        'if [ "$1" = fetch ]; then [ "$MISSING_TAG" = 0 ]; exit $?; fi\n'
        'if [ "$1" = rev-parse ]; then\n'
        '  if [ "$2" = "FETCH_HEAD^{commit}" ]; then printf "%s\\n" "$CURRENT_TAG_COMMIT";\n'
        '  else printf "%s\\n" "$GITHUB_SHA"; fi; exit 0\n'
        "fi\nexit 90\n"
    )
    gh = bin_dir / "gh"
    gh.write_text(
        "#!/bin/sh\n"
        'if [ "$1 $2" = "release view" ]; then [ "$RELEASE_EXISTS" = 1 ]; exit $?; fi\n'
        'if [ "$1 $2" = "release create" ]; then printf "%s\\n" "$*" > "$PUBLISH_CALL"; exit 0; fi\n'
        "exit 91\n"
    )
    git.chmod(0o755)
    gh.chmod(0o755)
    dist = tmp_path / "dist"
    dist.mkdir()
    archive = dist / "fixture.tar.gz"
    archive.write_bytes(b"synthetic distribution fixture")
    (dist / "SHA256SUMS").write_text(
        f"{hashlib.sha256(archive.read_bytes()).hexdigest()}  {archive.name}\n"
    )
    publish_call = tmp_path / "publish.txt"
    git_calls = tmp_path / "git.txt"
    env = {
        **os.environ,
        "PATH": f"{bin_dir}:{os.environ['PATH']}",
        "GITHUB_REF": "refs/tags/v0.1.0",
        "GITHUB_REF_NAME": "v0.1.0",
        "GITHUB_REPOSITORY": "example/fixture",
        "GITHUB_SHA": "a" * 40,
        "CURRENT_TAG_COMMIT": current_commit,
        "MISSING_TAG": str(int(missing)),
        "RELEASE_EXISTS": str(int(exists)),
        "PUBLISH_CALL": str(publish_call),
        "GIT_CALLS": str(git_calls),
    }
    result = subprocess.run(
        ["bash", "-c", script],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == expected, result.stdout + result.stderr
    assert publish_call.exists() is published
    if exists:
        assert not git_calls.exists()
    elif not missing:
        assert "fetch --no-tags origin refs/tags/v0.1.0" in git_calls.read_text()
        assert "rev-parse FETCH_HEAD^{commit}" in git_calls.read_text()
        if not published:
            assert "Release tag moved" in result.stdout
