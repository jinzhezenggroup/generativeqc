"""Exercise PR-only compiler reuse, upload bounds and unchanged required gates."""

from __future__ import annotations

import ast
import hashlib
import importlib.util
import json
import os
import re
import shutil
import subprocess
from pathlib import Path
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location(
    "pr_cache", ROOT / "tools/ci_pr_cli_ccache.py"
)
assert SPEC and SPEC.loader
cache = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(cache)
WORKFLOW = (ROOT / ".github/workflows/ci.yml").read_text()
JOB = WORKFLOW.split("\n  native-cli-artifact:\n", 1)[1].split(
    "\n  native-cli-qualification:\n", 1
)[0]


def step(name: str) -> str:
    return JOB.split(f"      - name: {name}\n", 1)[1].split("\n      - ", 1)[0]


def field(section: str, name: str) -> str:
    match = re.search(rf"^\s*{re.escape(name)}: (.+)$", section, re.MULTILINE)
    assert match
    return match[1]


def expression(value: str, context: dict[str, str]) -> Any:
    value = value.removeprefix("${{").removesuffix("}}").strip()
    value = value.replace("always()", "True")
    for key in sorted(context, key=len, reverse=True):
        value = value.replace(key, repr(context[key]))
    value = value.replace("&&", " and ").replace("||", " or ")

    def evaluate(node: ast.AST) -> Any:
        if isinstance(node, ast.Constant):
            return node.value
        if isinstance(node, ast.Compare) and len(node.ops) == 1:
            equal = evaluate(node.left) == evaluate(node.comparators[0])
            assert isinstance(node.ops[0], (ast.Eq, ast.NotEq))
            return equal if isinstance(node.ops[0], ast.Eq) else not equal
        if isinstance(node, ast.BoolOp):
            result = evaluate(node.values[0])
            for child in node.values[1:]:
                if (isinstance(node.op, ast.And) and result) or (
                    isinstance(node.op, ast.Or) and not result
                ):
                    result = evaluate(child)
            return result
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            assert node.func.id == "format" and not node.keywords
            template, *values = map(evaluate, node.args)
            return template.format(*values)
        raise AssertionError(ast.dump(node))

    return evaluate(ast.parse(value, mode="eval").body)


@pytest.fixture
def environment(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    for key, value in {
        "GITHUB_EVENT_NAME": "pull_request",
        "GITHUB_REF": "refs/pull/2498/merge",
        "PR_NUMBER": "2498",
        "CCACHE_VERSION": "4.14",
        "RUNNER_TEMP": str(tmp_path),
        "GITHUB_OUTPUT": str(tmp_path / "outputs"),
    }.items():
        monkeypatch.setenv(key, value)
    return tmp_path


@pytest.mark.parametrize(
    ("event", "ref", "number"),
    [
        ("push", "refs/heads/master", "2498"),
        ("merge_group", "refs/heads/gh-readonly-queue/master/pr-2498", "2498"),
        ("pull_request_target", "refs/heads/master", "2498"),
        ("workflow_dispatch", "refs/heads/master", "2498"),
        ("pull_request", "refs/pull/2499/merge", "2498"),
        ("pull_request", "refs/pull/2498/head", "2498"),
        ("pull_request", "refs/pull/0/merge", "0"),
        ("pull_request", "refs/pull/../merge", ".."),
        ("pull_request", "refs/pull/2498\n/merge", "2498\n"),
        ("pull_request", "refs/pull//merge", ""),
    ],
)
def test_invalid_event_ref_or_number_cannot_prepare_or_bound(
    environment: Path,
    monkeypatch: pytest.MonkeyPatch,
    event: str,
    ref: str,
    number: str,
) -> None:
    monkeypatch.setenv("GITHUB_EVENT_NAME", event)
    monkeypatch.setenv("GITHUB_REF", ref)
    monkeypatch.setenv("PR_NUMBER", number)
    with pytest.raises(ValueError):
        cache.identity()
    assert not (environment / "outputs").exists()


def test_identity_is_one_key_per_pr_toolchain_and_never_trusted(
    environment: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    first = cache.identity()
    assert first[1] == environment / "cuda-cli-pr-ccache-2498"
    assert first[0].startswith("ccache-pr-cli-delta-")
    for key in ("GITHUB_SHA", "GITHUB_RUN_ID", "GITHUB_RUN_ATTEMPT"):
        monkeypatch.setenv(key, "changed")
        assert cache.identity() == first
    monkeypatch.setenv("CCACHE_VERSION", "4.15")
    assert cache.identity()[0] != first[0]
    monkeypatch.setenv("PR_NUMBER", "2499")
    monkeypatch.setenv("GITHUB_REF", "refs/pull/2499/merge")
    other = cache.identity()
    assert other[0] != first[0] and other[1] != first[1]
    for key in (first[0], other[0]):
        assert not key.startswith(("ccache-linux-cuda-", "ccache-native-cli-"))


def test_workflow_is_optional_exact_pr_only_and_preserves_required_builds() -> None:
    prepare = step("Prepare isolated PR CLI compiler delta")
    restore = step("Restore only this PR CLI compiler delta")
    build = step("Build monolithic C++ CUDA CLI (release or fast PR)")
    bound = step("Bound successful PR CLI compiler misses")
    save = step("Save this PR CLI compiler delta once")
    for optional in (prepare, restore, bound, save):
        assert field(optional, "continue-on-error") == "true"
        assert field(optional, "timeout-minutes") in ("1", "2")
        assert "github.event_name == 'pull_request'" in field(optional, "if")
    assert "restore-keys:" not in restore
    assert field(restore, "key") == field(save, "key")
    assert field(restore, "path") == field(save, "path")
    assert "~/.cache/ccache" not in restore + save
    assert field(build, "if") == "steps.cli_scope.outputs.tier != 'skip'"
    assert "continue-on-error:" not in build
    assert field(build, "CCACHE_NORESHARE") == '"true"'
    assert "CCACHE_SLOPPINESS" not in build
    assert "CCACHE_REMOTE_ONLY" not in build
    assert "CCACHE_RESHARE:" not in build
    assert (
        "CCACHE_REMOTE_STORAGE"
        not in JOB.split(
            "      - name: Build monolithic C++ CUDA CLI (release or fast PR)", 1
        )[1].split("      - name: Bound successful PR CLI compiler misses", 1)[1]
    )
    assert "permissions:" not in prepare + restore + bound + save
    assert "GH_TOKEN" not in prepare + restore + bound + save
    for event in (
        "pull_request",
        "push",
        "merge_group",
        "schedule",
        "workflow_dispatch",
    ):
        for outcome in ("success", "failure", "cancelled", "skipped", ""):
            for key, matched, hit in (
                ("key", "", ""),
                ("key", "key", "true"),
                ("key", "key-extra", "false"),
                ("", "", ""),
            ):
                context = {
                    "github.event_name": event,
                    "steps.pr_cli_ccache.outputs.key": key,
                    "steps.pr_cli_ccache.outputs.path": "/tmp/pr-delta",
                    "steps.pr_cli_ccache_restore.outcome": outcome,
                    "steps.pr_cli_ccache_restore.outputs.cache-hit": hit,
                    "steps.pr_cli_ccache_restore.outputs.cache-matched-key": matched,
                    "steps.native_cli_build.outcome": "success",
                    "steps.pr_cli_ccache_bound.outputs.available": "true",
                }
                active = event == "pull_request" and key and outcome == "success"
                expected = (
                    "file:///tmp/pr-delta" + (" read-only" if hit == "true" else "")
                    if active and (hit == "true" or not matched)
                    else ""
                )
                assert (
                    expression(field(build, "CCACHE_REMOTE_STORAGE"), context)
                    == expected
                )
                assert bool(expression(field(bound, "if"), context)) == bool(
                    active and hit != "true" and not matched
                )
                for built in ("failure", "cancelled", "skipped", ""):
                    context["steps.native_cli_build.outcome"] = built
                    assert not expression(field(bound, "if"), context)
                    assert not expression(field(save, "if"), context)
    for available in ("", "false", "true"):
        assert expression(
            field(save, "if"),
            {
                "github.event_name": "pull_request",
                "steps.native_cli_build.outcome": "success",
                "steps.pr_cli_ccache_bound.outputs.available": available,
            },
        ) == (available == "true")


def test_empty_delta_never_uploads(environment: Path) -> None:
    _, path = cache.identity()
    cache.bound(path)
    path.mkdir()
    cache.bound(path)
    assert not (environment / "outputs").exists()


@pytest.mark.parametrize(
    "kind", ["root-link", "file-link", "dir-link", "hard-link", "fifo"]
)
def test_delta_rejects_redirected_or_special_files(
    environment: Path, kind: str
) -> None:
    _, path = cache.identity()
    outside = environment / "outside"
    outside.mkdir()
    sentinel = outside / "sentinel"
    sentinel.write_text("not PR cache data")
    path.mkdir()
    if kind == "root-link":
        path.rmdir()
        path.symlink_to(outside)
        with pytest.raises(ValueError):
            cache.identity()
    elif kind == "file-link":
        (path / "entry").symlink_to(sentinel)
    elif kind == "dir-link":
        (path / "entry").symlink_to(outside)
    elif kind == "hard-link":
        os.link(sentinel, path / "entry")
    else:
        os.mkfifo(path / "entry")
    with pytest.raises(ValueError):
        cache.bound(path)
    assert sentinel.read_text() == "not PR cache data"
    assert not (environment / "outputs").exists()


@pytest.mark.parametrize("failure", ["exception", "over-budget"])
def test_failed_trim_cannot_publish(
    environment: Path, monkeypatch: pytest.MonkeyPatch, failure: str
) -> None:
    _, path = cache.identity()
    path.mkdir()
    entry = path / "result"
    entry.write_text("fixture")

    def trim(command: list[str], **kwargs: Any) -> None:
        assert command == [
            "ccache",
            "--trim-dir",
            str(path),
            "--trim-max-size",
            "256MiB",
        ]
        assert kwargs == {"check": True, "timeout": 45}
        if failure == "exception":
            raise subprocess.TimeoutExpired(command, 45)
        with entry.open("wb") as stream:
            stream.truncate(cache.MAX_BYTES + 1)

    monkeypatch.setattr(cache.subprocess, "run", trim)
    with pytest.raises((ValueError, subprocess.TimeoutExpired)):
        cache.bound(path)
    assert not (environment / "outputs").exists()


def test_real_delta_records_misses_replays_read_only_and_checks_changed_source(
    environment: Path,
) -> None:
    ccache, compiler = shutil.which("ccache"), shutil.which("g++")
    if not ccache or not compiler:
        pytest.skip("ccache and g++ are required for the real compiler-cache fixture")
    subprocess.run([ccache, "--version"], check=True, capture_output=True, timeout=5)
    _, delta = cache.identity()
    source = environment / "source.cpp"
    source.write_text("int value() { return 7; }\n")
    config = environment / "ccache.conf"
    config.write_text("")
    base = {k: v for k, v in os.environ.items() if not k.startswith("CCACHE_")}
    base.update(
        CCACHE_CONFIGPATH=str(config), CCACHE_MAXSIZE="16MiB", CCACHE_NORESHARE="true"
    )

    def compile_one(local: str, remote: str) -> dict[str, int]:
        env = {
            **base,
            "CCACHE_DIR": str(environment / local),
            "CCACHE_REMOTE_STORAGE": remote,
        }
        subprocess.run(
            [
                ccache,
                compiler,
                "-O2",
                "-c",
                str(source),
                "-o",
                str(environment / "source.o"),
            ],
            env=env,
            check=True,
            capture_output=True,
            timeout=10,
        )
        return json.loads(
            subprocess.check_output(
                [ccache, "--print-stats", "--format=json"], env=env, timeout=5
            )
        )

    assert compile_one("trusted-local", "")["cache_miss"] == 1
    assert compile_one("trusted-local", delta.as_uri())["direct_cache_hit"] == 1
    assert not delta.exists(), "local trusted hits must not inflate the delta"
    source.write_text("int value() { return 9; }\n")
    assert compile_one("trusted-local", delta.as_uri())["cache_miss"] == 2
    cache.bound(delta)
    assert (environment / "outputs").read_text() == "available=true\n"
    assert sum(p.stat().st_size for p in cache.cache_files(delta)) <= cache.MAX_BYTES

    def digest() -> dict[str, str]:
        return {
            str(p.relative_to(delta)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in cache.cache_files(delta)
        }

    saved = digest()
    read_only = delta.as_uri() + " read-only"
    stats = compile_one("next-job", read_only)
    assert stats["remote_storage_hit"] >= 1 and stats["cache_miss"] == 0
    source.write_text("int value() { return 11; }\n")
    assert compile_one("changed-source", read_only)["cache_miss"] == 1
    assert digest() == saved
    # The next same-PR run cannot create new entries under the immutable key.
    assert compile_one("another-job", read_only)["cache_miss"] == 1
    assert digest() == saved


def test_real_trim_enforces_cap_without_touching_local_cache(environment: Path) -> None:
    ccache = shutil.which("ccache")
    if not ccache:
        pytest.skip("ccache is required for the real secondary-storage trim")
    subprocess.run([ccache, "--version"], check=True, capture_output=True, timeout=5)
    _, delta = cache.identity()
    delta.mkdir()
    local = environment / "trusted-local"
    local.mkdir()
    sentinel = local / "preserve"
    sentinel.write_text("trusted cache must remain unchanged")
    old = delta / "old-result"
    with old.open("wb") as stream:
        stream.truncate(cache.MAX_BYTES + 1)
    os.utime(old, (1, 1))
    recent = delta / "new-result"
    recent.write_text("recent compiler entry")
    cache.bound(delta)
    assert (environment / "outputs").read_text() == "available=true\n"
    assert sum(p.stat().st_size for p in cache.cache_files(delta)) <= cache.MAX_BYTES
    assert sentinel.read_text() == "trusted cache must remain unchanged"


# Fast CUDA uses the same strict, single-PR compiler-delta contract as CLI,
# but retains an independent cache namespace and a bounded 512 MiB budget.
FAST_SPEC = importlib.util.spec_from_file_location(
    "pr_fast_cache", ROOT / "tools/ci_pr_fast_ccache.py"
)
assert FAST_SPEC and FAST_SPEC.loader
fast_cache = importlib.util.module_from_spec(FAST_SPEC)
FAST_SPEC.loader.exec_module(fast_cache)


def fast_step(name: str) -> str:
    job = WORKFLOW.split("\n  cuda-compile:\n", 1)[1].split(
        "\n  master-cache-cleanup:\n", 1
    )[0]
    return job.split(f"      - name: {name}\n", 1)[1].split("\n      - ", 1)[0]


@pytest.mark.parametrize(
    ("event", "ref", "number"),
    [
        ("push", "refs/heads/master", "2498"),
        ("merge_group", "refs/heads/gh-readonly-queue/master/pr-2498", "2498"),
        ("pull_request_target", "refs/heads/master", "2498"),
        ("pull_request", "refs/pull/2499/merge", "2498"),
        ("pull_request", "refs/pull/2498/head", "2498"),
        ("pull_request", "refs/pull/0/merge", "0"),
    ],
)
def test_fast_cuda_cache_rejects_untrusted_ref(
    environment: Path,
    monkeypatch: pytest.MonkeyPatch,
    event: str,
    ref: str,
    number: str,
) -> None:
    monkeypatch.setenv("GITHUB_EVENT_NAME", event)
    monkeypatch.setenv("GITHUB_REF", ref)
    monkeypatch.setenv("PR_NUMBER", number)
    with pytest.raises(ValueError):
        fast_cache.identity()
    assert not (environment / "outputs").exists()


def test_fast_cuda_cache_identity_and_bound(
    environment: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    key, path = fast_cache.identity()
    assert "ccache-pr-fast-cuda-" in key
    assert path == environment / "cuda-fast-pr-ccache-2498"
    for value in ("GITHUB_SHA", "GITHUB_RUN_ID", "GITHUB_RUN_ATTEMPT"):
        monkeypatch.setenv(value, "another-run")
        assert fast_cache.identity() == (key, path)
    assert fast_cache.MAX_BYTES == 512 * 1024**2
    fast_cache.bound(path)
    assert not (environment / "outputs").exists()
    path.mkdir()
    (path / "new-object").write_bytes(b"compiled-ccache-output")

    def trim(command: list[str], **kwargs: Any) -> None:
        assert command == [
            "ccache",
            "--trim-dir",
            str(path),
            "--trim-max-size",
            "512MiB",
        ]
        assert kwargs == {"check": True, "timeout": 45}

    monkeypatch.setattr(fast_cache.subprocess, "run", trim)
    fast_cache.bound(path)
    assert (environment / "outputs").read_text() == "available=true\n"
    (path / "unsafe-link").symlink_to(environment / "outputs")
    with pytest.raises(ValueError):
        fast_cache.bound(path)


def test_fast_cuda_compiler_delta_does_not_weaken_required_build() -> None:
    prepare = fast_step("Prepare isolated PR fast CUDA compiler delta")
    restore = fast_step("Restore only this PR fast CUDA compiler delta")
    prerequisite = fast_step("Compile production CUDA prerequisite archive")
    build = fast_step("Compile production CUDA and runtime CTest")
    bound = fast_step("Bound successful PR fast CUDA compiler misses")
    save = fast_step("Save this PR fast CUDA compiler delta once")
    for optional in (prepare, restore, bound, save):
        assert field(optional, "continue-on-error") == "true"
        assert field(optional, "timeout-minutes") in ("1", "2")
        assert "github.event_name == 'pull_request'" in field(optional, "if")
    assert "restore-keys:" not in restore
    assert field(restore, "path") == field(save, "path")
    assert field(restore, "key") == field(save, "key")
    assert "~/.cache/ccache" not in restore + save
    assert "id: cuda_runtime_build" in build
    assert "continue-on-error:" not in prerequisite + build
    for step in (prerequisite, build):
        assert 'CCACHE_NORESHARE: "true"' in step
        remote = field(step, "CCACHE_REMOTE_STORAGE")
        assert "cuda-ccache-handoff" in remote
        assert "steps.pr_fast_ccache_restore.outputs.cache-hit == 'true'" in remote
        assert "read-only" in remote
        assert "CCACHE_REMOTE_ONLY" not in step
        assert "CCACHE_SLOPPINESS" not in step
    assert "steps.cuda_runtime_build.outcome == 'success'" in field(bound, "if")
    assert "steps.cuda_runtime_build.outcome == 'success'" in field(save, "if")
    assert "steps.pr_fast_ccache_restore.outputs.cache-hit != 'true'" in field(
        bound, "if"
    )
    assert "steps.pr_fast_ccache_restore.outputs.cache-matched-key == ''" in field(
        bound, "if"
    )
    assert "ctest --test-dir build --output-on-failure --parallel 2" in WORKFLOW


def test_fast_cuda_remote_and_publication_follow_exact_pr_cache_state() -> None:
    """Execute cache expressions for hits, misses and unavailable accelerators."""
    remote_steps = (
        fast_step("Compile production CUDA prerequisite archive"),
        fast_step("Compile production CUDA and runtime CTest"),
    )
    bound = fast_step("Bound successful PR fast CUDA compiler misses")
    for event in ("pull_request", "merge_group", "pull_request_target", "push"):
        for prepared in ("", "exact-pr-key"):
            for outcome in ("success", "failure", "cancelled", "skipped", ""):
                for hit, matched in (
                    ("true", "exact-pr-key"),
                    ("false", ""),
                    ("", ""),
                    ("false", "other-key"),
                ):
                    context = {
                        "github.event_name": event,
                        "github.ref": "refs/pull/2498/merge",
                        "runner.temp": "/tmp/runner",
                        "steps.pr_fast_ccache.outputs.key": prepared,
                        "steps.pr_fast_ccache.outputs.path": "/tmp/runner/cuda-fast-pr-ccache-2498",
                        "steps.pr_fast_ccache_restore.outcome": outcome,
                        "steps.pr_fast_ccache_restore.outputs.cache-hit": hit,
                        "steps.pr_fast_ccache_restore.outputs.cache-matched-key": matched,
                        "steps.cuda_runtime_build.outcome": "success",
                    }
                    allowed = (
                        event == "pull_request"
                        and bool(prepared)
                        and outcome == "success"
                        and (hit == "true" or matched == "")
                    )
                    expected = (
                        "file:///tmp/runner/cuda-fast-pr-ccache-2498" if allowed else ""
                    )
                    if allowed and hit == "true":
                        expected += " read-only"
                    for step in remote_steps:
                        assert (
                            expression(field(step, "CCACHE_REMOTE_STORAGE"), context)
                            == expected
                        )
                    publish = allowed and hit != "true" and matched == ""
                    assert bool(expression(field(bound, "if"), context)) == publish
                    context["steps.cuda_runtime_build.outcome"] = "failure"
                    assert not expression(field(bound, "if"), context)
