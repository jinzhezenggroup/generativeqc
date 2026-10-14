"""Trusted same-run cache deltas accelerate, but never replace, qualification."""

from __future__ import annotations

import ast
import hashlib
import json
import os
import re
import shutil
import subprocess
import textwrap
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
WORKFLOW = (ROOT / ".github/workflows/ci.yml").read_text()
UPLOAD = "Hand fast CUDA cache misses to the master CLI"
DOWNLOAD = "Restore this run's master fast CUDA cache delta"
BOUND = "Bound the master fast CUDA cache handoff"
SEED = "Seed fast CUDA CI objects in the shared cache"


def _job(workflow: str, name: str) -> str:
    return re.split(
        r"\n  [A-Za-z0-9_-]+:\n",
        workflow.split(f"\n  {name}:\n", 1)[1],
        maxsplit=1,
    )[0]


def _step(job: str, name: str) -> str:
    return job.split(f"      - name: {name}\n", 1)[1].split("\n      - ", 1)[0]


def _evaluate(node: ast.AST) -> str | bool:
    if isinstance(node, ast.Constant) and isinstance(node.value, (str, bool)):
        return node.value
    if isinstance(node, ast.Compare) and len(node.ops) == 1:
        equal = _evaluate(node.left) == _evaluate(node.comparators[0])
        if isinstance(node.ops[0], ast.Eq):
            return equal
        if isinstance(node.ops[0], ast.NotEq):
            return not equal
    if isinstance(node, ast.BoolOp) and isinstance(node.op, (ast.And, ast.Or)):
        result = _evaluate(node.values[0])
        for value in node.values[1:]:
            if (isinstance(node.op, ast.And) and result) or (
                isinstance(node.op, ast.Or) and not result
            ):
                result = _evaluate(value)
        return result
    raise AssertionError(f"Unsupported condition: {ast.dump(node)}")


def _expression(expression: str, context: dict[str, str]) -> str | bool:
    expression = expression.removeprefix("${{").removesuffix("}}").strip()
    for key, value in context.items():
        expression = expression.replace(key, repr(value))
    expression = expression.replace("&&", " and ").replace("||", " or ")
    return _evaluate(ast.parse(expression, mode="eval").body)


def _field(section: str, key: str) -> str:
    match = re.search(rf"^\s*{re.escape(key)}: (.+)$", section, re.MULTILINE)
    assert match, f"Missing {key}"
    return match[1]


def _assert_transport(workflow: str) -> None:
    producer = _job(workflow, "cuda-compile")
    consumer = _job(workflow, "native-cli-artifact")
    compile_step = _step(producer, "Compile production CUDA and runtime CTest")
    upload = _step(producer, UPLOAD)
    bound = _step(producer, BOUND)
    download = _step(consumer, DOWNLOAD)
    seed = _step(consumer, SEED)
    assert 'CCACHE_NORESHARE: "true"' in compile_step
    assert "continue-on-error" not in compile_step
    assert "--target generativeqc_cuda_runtime_tests --parallel 3" in compile_step
    for step, minutes in ((bound, 1), (upload, 2), (download, 2)):
        assert _field(step, "continue-on-error") == "true"
        assert _field(step, "timeout-minutes") == str(minutes)
        assert "always()" not in _field(step, "if")
    assert '--trim-dir "$delta" --trim-max-size 2GiB' in bound
    assert 'delta="$RUNNER_TEMP/cuda-ccache-handoff"' in bound
    assert "retention-days: 1" in upload and "compression-level: 0" in upload
    assert (
        "cuda-ccache-handoff-${{ github.run_id }}-${{ github.run_attempt }}" in upload
    )
    assert "path: ${{ runner.temp }}/cuda-ccache-handoff" in upload
    assert "~/.cache/ccache" not in upload and "overwrite:" not in upload
    assert "needs: cuda-compile" in consumer
    assert "needs.cuda-compile.result == 'success'" in consumer
    assert (
        "fast_cache_artifact_id: ${{ steps.master_fast_ccache_upload.outputs.artifact-id }}"
        in producer
    )
    assert "needs.cuda-compile.outputs.fast_cache_artifact_id" in consumer
    assert "needs.cuda-compile.outputs.fast_cache_run_attempt" in consumer
    assert (
        "fast_cache_run_attempt: ${{ steps.master_fast_ccache_upload.outcome == 'success' && github.run_attempt || '' }}"
        in producer
    )
    # No token/run/repository override: download-artifact defaults to this run.
    fields = re.findall(r"^          ([\w-]+):", download, re.MULTILINE)
    assert fields == ["artifact-ids", "path"]
    assert "artifact-ids: ${{ env.FAST_CUDA_CACHE_ARTIFACT_ID }}" in download
    assert (
        producer.index("Check Fock and proposal preflight without a GPU")
        < (producer.index(BOUND))
        < producer.index(UPLOAD)
    )
    assert (
        consumer.index("Restore shared Linux CUDA ccache")
        < consumer.index(DOWNLOAD)
        < consumer.index(SEED)
        < consumer.index("Build monolithic C++ CUDA CLI (release or fast PR)")
    )
    for forbidden in ("CCACHE_SLOPPINESS", "CCACHE_REMOTE_ONLY", "CCACHE_RESHARE:"):
        assert forbidden not in compile_step + seed

    events = (
        ("push", "refs/heads/master"),
        ("push", "refs/heads/feature"),
        ("push", "refs/tags/v1.0.0"),
        ("pull_request", "refs/heads/master"),
        ("merge_group", "refs/heads/master"),
        ("workflow_dispatch", "refs/heads/master"),
        ("schedule", "refs/heads/master"),
    )
    for event, ref in events:
        trusted = event == "push" and ref == "refs/heads/master"
        context = {
            "github.event_name": event,
            "github.ref": ref,
            "github.run_attempt": "2",
            "steps.prose_scope.outcome": "success",
            "steps.prose_scope.outputs.scope": "full",
            "format('file://{0}/cuda-ccache-handoff', runner.temp)": "file:///tmp/delta",
        }
        assert (
            bool(_expression(_field(compile_step, "CCACHE_REMOTE_STORAGE"), context))
            == trusted
        )
        assert _expression(_field(bound, "if"), context) == trusted
        for available in ("", "true", "false"):
            context["steps.master_fast_ccache.outputs.available"] = available
            assert _expression(_field(upload, "if"), context) == (
                trusted and available == "true"
            )
        for tier in ("skip", "fast", "release"):
            for artifact_id in ("", "12345"):
                for attempt in ("", "1", "2"):
                    context.update(
                        {
                            "steps.cli_scope.outputs.tier": tier,
                            "env.FAST_CUDA_CACHE_ARTIFACT_ID": artifact_id,
                            "env.FAST_CUDA_CACHE_RUN_ATTEMPT": attempt,
                        }
                    )
                    assert _expression(_field(download, "if"), context) == (
                        trusted
                        and tier != "skip"
                        and bool(artifact_id)
                        and attempt == "2"
                    )
    for outcome in ("success", "failure", "cancelled", "skipped", ""):
        remote = _expression(
            _field(seed, "CCACHE_REMOTE_STORAGE"),
            {
                "steps.master_fast_ccache_download.outcome": outcome,
                "format('file://{0}/cuda-ccache-handoff read-only', runner.temp)": "file:///tmp/delta read-only",
            },
        )
        assert remote == ("file:///tmp/delta read-only" if outcome == "success" else "")
    # Transport has no authority over the mandatory build conditions.
    assert "master_fast_ccache_download" not in _field(seed, "if")
    assert "continue-on-error" not in seed
    build = _step(consumer, "Build monolithic C++ CUDA CLI (release or fast PR)")
    assert _field(build, "if") == "steps.cli_scope.outputs.tier != 'skip'"
    assert "continue-on-error" not in build


def test_master_delta_transport_keeps_exact_run_trust_and_required_builds() -> None:
    _assert_transport(WORKFLOW)


@pytest.mark.parametrize(
    ("before", "after"),
    [
        ("github.event_name == 'push' && ", ""),
        ("github.ref == 'refs/heads/master' && ", ""),
        (" && env.FAST_CUDA_CACHE_ARTIFACT_ID != ''", ""),
        (" && env.FAST_CUDA_CACHE_RUN_ATTEMPT == github.run_attempt", ""),
        (" read-only", ""),
        ('CCACHE_NORESHARE: "true"', 'CCACHE_RESHARE: "true"'),
        ("--trim-max-size 2GiB", "--trim-max-size 0"),
        ("artifact-ids: ${{ env.FAST_CUDA_CACHE_ARTIFACT_ID }}", "name: latest-cache"),
    ],
    ids=[
        "event",
        "master-ref",
        "artifact-id",
        "attempt",
        "read-only",
        "no-reshare",
        "size-bound",
        "no-name-lookup",
    ],
)
def test_transport_controls_reject_weakened_variants(before: str, after: str) -> None:
    assert before in WORKFLOW
    with pytest.raises((AssertionError, SyntaxError)):
        _assert_transport(WORKFLOW.replace(before, after))


@pytest.mark.parametrize("exists", [False, True])
def test_missing_or_present_delta_is_bounded_without_touching_local_cache(
    tmp_path: Path, exists: bool
) -> None:
    delta = tmp_path / "cuda-ccache-handoff"
    if exists:
        delta.mkdir()
        (delta / "fixture").write_text("cached compiler result")
    binaries = tmp_path / "bin"
    binaries.mkdir()
    calls = tmp_path / "calls"
    fake = binaries / "ccache"
    fake.write_text('#!/bin/sh\nprintf "%s\\n" "$@" >> "$CALLS"\n')
    fake.chmod(0o755)
    output = tmp_path / "output"
    output.touch()
    script = textwrap.dedent(
        _step(_job(WORKFLOW, "cuda-compile"), BOUND).split("        run: |\n", 1)[1]
    )
    subprocess.run(
        ["bash", "-e", "-o", "pipefail", "-c", script],
        env={
            **os.environ,
            "RUNNER_TEMP": str(tmp_path),
            "GITHUB_OUTPUT": str(output),
            "CALLS": str(calls),
            "PATH": f"{binaries}:{os.environ['PATH']}",
        },
        check=True,
        capture_output=True,
        timeout=5,
    )
    assert output.read_text() == ("available=true\n" if exists else "")
    assert calls.exists() is exists
    if exists:
        assert calls.read_text().splitlines() == [
            "--trim-dir",
            str(delta),
            "--trim-max-size",
            "2GiB",
        ]


def test_real_cache_delta_reuses_only_matching_inputs_and_stays_read_only(
    tmp_path: Path,
) -> None:
    ccache = shutil.which("ccache")
    compiler = shutil.which("g++")
    if not ccache or not compiler:
        pytest.skip("ccache and g++ are required for the file-backend contract")
    subprocess.run([ccache, "--version"], check=True, capture_output=True, timeout=5)
    config = tmp_path / "ccache.conf"
    config.write_text("")
    source = tmp_path / "fixture.cpp"
    header = tmp_path / "value.hpp"
    source.write_text('#include "value.hpp"\nint value(){return VALUE;}\n')
    header.write_text("#define VALUE 7\n")
    delta = tmp_path / "delta"
    baseline = {k: v for k, v in os.environ.items() if not k.startswith("CCACHE_")}
    baseline.update(CCACHE_CONFIGPATH=str(config), CCACHE_MAXSIZE="16MiB")
    wrappers = []
    for index in range(2):
        wrapper = tmp_path / f"compiler-{index}" / "g++"
        wrapper.parent.mkdir()
        wrapper.write_text(
            f'#!/bin/sh\n# compiler identity {"x" * index}\nexec "{compiler}" "$@"\n'
        )
        wrapper.chmod(0o755)
        wrappers.append(wrapper)

    def compile_one(
        cache: str,
        remote: str,
        *,
        opt: str = "-O2",
        provider: str = "0",
        executable: Path = wrappers[0],
    ) -> dict[str, int]:
        env = {
            **baseline,
            "CCACHE_DIR": str(tmp_path / cache),
            "CCACHE_REMOTE_STORAGE": remote,
            "CCACHE_NORESHARE": "true",
        }
        subprocess.run(
            [
                ccache,
                str(executable),
                opt,
                f"-DUNUSED_PROVIDER={provider}",
                "-c",
                str(source),
                "-o",
                str(tmp_path / "fixture.o"),
            ],
            cwd=tmp_path,
            env=env,
            check=True,
            capture_output=True,
            timeout=10,
        )
        result = subprocess.run(
            [ccache, "--print-stats", "--format=json"],
            env=env,
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
        return json.loads(result.stdout)

    # An existing local hit must not copy the restored snapshot into the delta.
    assert compile_one("warm", "")["cache_miss"] == 1
    assert compile_one("warm", delta.as_uri())["direct_cache_hit"] == 1
    assert not list(delta.rglob("*"))
    # A new compilation writes through to a separate file backend.
    assert compile_one("producer", delta.as_uri())["cache_miss"] == 1

    def contents() -> dict[str, tuple[str, int]]:
        return {
            str(p.relative_to(delta)): (
                hashlib.sha256(p.read_bytes()).hexdigest(),
                p.stat().st_mtime_ns,
            )
            for p in delta.rglob("*")
            if p.is_file()
        }

    original = contents()
    assert original
    readonly = delta.as_uri() + " read-only"
    reused = compile_one("consumer", readonly, provider="1")
    assert reused["remote_storage_hit"] >= 1 and reused["cache_miss"] == 0
    assert reused["preprocessed_cache_hit"] == 1
    # A remote hit is promoted locally for later producer builds/publication.
    assert compile_one("consumer", "", provider="1")["direct_cache_hit"] == 1
    assert compile_one("changed-flags", readonly, opt="-O0")["cache_miss"] == 1
    assert (
        compile_one("changed-compiler", readonly, executable=wrappers[1])["cache_miss"]
        == 1
    )
    header.write_text("#define VALUE 9\n")
    assert compile_one("changed-header", readonly)["cache_miss"] == 1
    assert contents() == original
    # Missing transport data is simply a miss; the compiler still produces output.
    missing = (tmp_path / "missing-delta").as_uri() + " read-only"
    assert compile_one("missing", missing)["cache_miss"] == 1
    assert (tmp_path / "fixture.o").is_file()
    # A broken secondary destination cannot turn valid compilation into a gate.
    blocked = tmp_path / "not-a-directory"
    blocked.write_text("fixture")
    assert compile_one("unavailable-producer", blocked.as_uri())["cache_miss"] == 1
