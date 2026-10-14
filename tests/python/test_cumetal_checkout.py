"""Bound CuMetal checkout history without changing source/provenance semantics."""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
WORKFLOW = ROOT / ".github/workflows/cumetal-cuda.yml"


def _checkout_inputs(workflow: str, job: str) -> dict[str, str]:
    section = workflow.split(f"\n  {job}:\n", 1)[1]
    section = re.split(r"\n  [\w-]+:\n", section, maxsplit=1)[0]
    checkout = section.split("uses: actions/checkout@", 1)[1]
    checkout = checkout.split("\n      - ", 1)[0]
    return dict(re.findall(r"^          ([\w-]+): (\S+)", checkout, re.MULTILINE))


def _assert_checkout_contract(workflow: str) -> None:
    inputs = _checkout_inputs(workflow, "cuda-tests")
    assert inputs["fetch-depth"] == "1"
    assert inputs["persist-credentials"] == "false"
    # Default checkout must retain the event SHA and every tracked source file.
    assert not {"ref", "repository", "filter", "sparse-checkout"}.intersection(inputs)
    # The upload-only Codecov job is outside this build-history optimization.
    upload = _checkout_inputs(workflow, "upload-test-results")
    assert upload["fetch-depth"] == "0"
    assert upload["persist-credentials"] == "false"


def test_cumetal_build_checkout_is_shallow_and_event_selected() -> None:
    _assert_checkout_contract(WORKFLOW.read_text())


@pytest.mark.parametrize(
    "replacement",
    (
        "fetch-depth: 0",
        "fetch-depth: 1\n          ref: master",
        "fetch-depth: 1\n          sparse-checkout: src",
        "fetch-depth: 1\n          persist-credentials: true",
    ),
)
def test_cumetal_checkout_rejects_history_source_or_credential_regression(
    replacement: str,
) -> None:
    workflow = (
        WORKFLOW.read_text().replace("          persist-credentials: false", "", 1)
        if "persist-credentials" in replacement
        else WORKFLOW.read_text()
    )
    with pytest.raises(AssertionError):
        _assert_checkout_contract(workflow.replace("fetch-depth: 1", replacement, 1))


@pytest.mark.parametrize(
    ("event_ref", "checkout_ref"),
    (
        ("refs/heads/master", "refs/remotes/origin/master"),
        ("refs/pull/7/merge", "refs/remotes/pull/7/merge"),
        (
            "refs/heads/gh-readonly-queue/master/pr-7",
            "refs/remotes/origin/gh-readonly-queue/master/pr-7",
        ),
    ),
)
def test_shallow_event_tree_keeps_codegen_identity(
    tmp_path: Path, event_ref: str, checkout_ref: str
) -> None:
    """Real Git depth-one fetches retain the merge tree, not either parent tree."""
    git = shutil.which("git")
    if git is None:
        pytest.skip("Git is required for shallow-checkout qualification")
    source = tmp_path / "source"
    source.mkdir()
    env = {
        **os.environ,
        "GIT_AUTHOR_NAME": "Checkout fixture",
        "GIT_AUTHOR_EMAIL": "checkout@example.invalid",
        "GIT_COMMITTER_NAME": "Checkout fixture",
        "GIT_COMMITTER_EMAIL": "checkout@example.invalid",
        "GIT_CONFIG_NOSYSTEM": "1",
        "GIT_CONFIG_GLOBAL": os.devnull,
    }

    def run(root: Path, *args: str) -> str:
        result = subprocess.run(
            args,
            cwd=root,
            env=env,
            check=True,
            capture_output=True,
            text=True,
            timeout=30,
        )
        return result.stdout.strip()

    run(source, git, "init", "--initial-branch=master")
    (source / "kernel.cpp").write_text("base source\n")
    (source / "manifest.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "files": ["kernel.cpp", "base.cpp"],
            }
        )
    )
    (source / "base.cpp").write_text("base-only input\n")
    run(source, git, "add", ".")
    run(source, git, "commit", "-m", "base")
    run(source, git, "checkout", "-b", "feature")
    (source / "kernel.cpp").write_text("feature source\n")
    run(source, git, "commit", "-am", "feature")
    feature = run(source, git, "rev-parse", "HEAD")
    run(source, git, "checkout", "master")
    (source / "base.cpp").write_text("new master input\n")
    run(source, git, "commit", "-am", "master")
    master = run(source, git, "rev-parse", "HEAD")
    run(source, git, "merge", "--no-ff", "feature", "-m", "merge")
    event_sha = run(source, git, "rev-parse", "HEAD")
    run(source, git, "update-ref", event_ref, event_sha)
    # The action's SHA refspec must not follow a ref that moved after the event.
    (source / "kernel.cpp").write_text("later source\n")
    run(source, git, "commit", "-am", "later")
    run(source, git, "update-ref", event_ref, "HEAD")
    run(source, git, "checkout", "--detach", event_sha)
    shallow = tmp_path / "shallow"
    shallow.mkdir()
    run(shallow, git, "init")
    depth = _checkout_inputs(WORKFLOW.read_text(), "cuda-tests")["fetch-depth"]
    run(
        shallow,
        git,
        "fetch",
        "--no-tags",
        f"--depth={depth}",
        source.as_uri(),
        f"+{event_sha}:{checkout_ref}",
    )
    run(shallow, git, "checkout", "--detach", checkout_ref)
    assert run(shallow, git, "rev-parse", "HEAD") == event_sha
    assert run(shallow, git, "rev-parse", "--is-shallow-repository") == "true"
    assert run(shallow, git, "rev-list", "--count", "HEAD") == "1"
    assert run(shallow, git, "status", "--porcelain") == ""
    assert run(shallow, git, "ls-tree", "-r", "HEAD") == run(
        source, git, "ls-tree", "-r", "HEAD"
    )

    def identity(root: Path, name: str) -> bytes:
        output = tmp_path / f"{name}.hpp"
        run(
            root,
            sys.executable,
            str(ROOT / "tools/run_codegen.py"),
            "--source-root",
            str(root),
            "--depfile",
            str(output.with_suffix(".d")),
            "--target",
            str(output),
            str(ROOT / "tools/generate_build_identity.py"),
            "--source-root",
            str(root),
            "--manifest",
            str(root / "manifest.json"),
            "--template",
            str(ROOT / "src/api/build_identity.hpp.in"),
            "--output",
            str(output),
            "--cuda-fast-compile",
            "OFF",
            "--release-build",
            "ON",
        )
        return output.read_bytes()

    merged_identity = identity(shallow, "shallow")
    assert merged_identity == identity(source, "full")
    # Using either parent would keep a plausible source tree but qualify wrong code.
    for name, revision in (("master", master), ("feature", feature)):
        run(source, git, "checkout", "--detach", revision)
        assert identity(source, name) != merged_identity
    (shallow / "kernel.cpp").write_text("changed source\n")
    assert identity(shallow, "changed") != merged_identity
