"""Execute optional CPU snapshot provenance, transport and activation boundaries."""

from __future__ import annotations

import ast
import copy
import gzip
import importlib.util
import io
import json
import os
import re
import subprocess
import sys
import tarfile
from pathlib import Path
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "tools/ci_cpu_ccache_snapshot.py"
spec = importlib.util.spec_from_file_location("cpu_snapshot", SCRIPT)
assert spec and spec.loader
snapshot = importlib.util.module_from_spec(spec)
spec.loader.exec_module(snapshot)


def fields(section: str, indent: int) -> dict[str, Any]:
    """Read the scalar/mapping fields exercised here without a YAML dependency."""
    result: dict[str, Any] = {}
    lines = section.splitlines()
    for index, line in enumerate(lines):
        match = re.fullmatch(rf" {{{indent}}}([\w-]+):(?: (.*))?", line)
        if not match:
            continue
        key, value = match.groups()
        if value is None:
            following = []
            for nested in lines[index + 1 :]:
                if nested.strip() and len(nested) - len(nested.lstrip()) <= indent:
                    break
                following.append(nested)
            result[key] = fields("\n".join(following), indent + 2)
        elif value in ("true", "false"):
            result[key] = value == "true"
        elif value.isdigit():
            result[key] = int(value)
        else:
            result[key] = value
    return result


def steps(section: str, indent: int) -> list[dict[str, Any]]:
    entries = re.split(rf"(?m)^ {{{indent}}}- ", section)[1:]
    return [fields(" " * (indent + 2) + entry, indent + 2) for entry in entries]


def load_workflow() -> dict[str, Any]:
    text = (ROOT / ".github/workflows/ci.yml").read_text()
    jobs = {}
    sections = re.split(r"(?m)^  ([\w-]+):\n", text.split("\njobs:\n", 1)[1])
    for name, section in zip(sections[1::2], sections[2::2], strict=True):
        jobs[name] = fields(section, 4)
        jobs[name]["steps"] = steps(section, 6)
    return {"jobs": jobs}


WORKFLOW = load_workflow()
ACTION = {
    "runs": {
        "steps": steps(
            (ROOT / ".github/actions/restore-cpu-ccache/action.yml").read_text(), 4
        )
    }
}
DATA_NAME = "a/b/" + "0123456789abcdef" * 2 + "R"
REPO = "jinzhezenggroup/generativeqc"


@pytest.fixture
def environment(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, str]:
    home, temp = tmp_path / "home", tmp_path / "temp"
    home.mkdir()
    temp.mkdir()
    values = {
        "HOME": str(home),
        "RUNNER_TEMP": str(temp),
        "GITHUB_OUTPUT": str(tmp_path / "output"),
        "GITHUB_REPOSITORY": REPO,
        "GITHUB_RUN_ID": "123",
        "GITHUB_RUN_ATTEMPT": "2",
        "GITHUB_SHA": "a" * 40,
        "GITHUB_EVENT_NAME": "push",
        "GITHUB_REF": "refs/heads/master",
        "ImageOS": "ubuntu24",
        "RUNNER_ARCH": "X64",
        "CCACHE_VERSION": "4.14",
    }
    for key, value in values.items():
        monkeypatch.setenv(key, value)
    return os.environ.copy()


def run_command(command: str, env: dict[str, str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(SCRIPT), command],
        env=env,
        text=True,
        capture_output=True,
        timeout=10,
        check=False,
    )


def outputs(env: dict[str, str]) -> dict[str, str]:
    path = Path(env["GITHUB_OUTPUT"])
    return (
        dict(line.split("=", 1) for line in path.read_text().splitlines())
        if path.exists()
        else {}
    )


def fixture_run() -> dict[str, Any]:
    return {
        "id": 456,
        "run_attempt": 3,
        "head_sha": "b" * 40,
        "path": snapshot.WORKFLOW,
        "event": "push",
        "head_branch": "master",
        "status": "completed",
        "conclusion": "success",
        "repository": {"id": 9, "full_name": REPO},
        "head_repository": {"id": 9, "full_name": REPO},
    }


def fixture_artifact(run: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": 789,
        "name": f"{snapshot.ARTIFACT}-{run['run_attempt']}",
        "expired": False,
        "size_in_bytes": 1000,
        "digest": "sha256:" + "c" * 64,
        "workflow_run": {
            "id": run["id"],
            "head_sha": run["head_sha"],
            "head_branch": "master",
            "repository_id": 9,
            "head_repository_id": 9,
        },
    }


def fake_api(
    tmp_path: Path,
    env: dict[str, str],
    run: dict[str, Any],
    artifact: dict[str, Any],
    *,
    failure: str = "",
) -> None:
    (tmp_path / "runs.json").write_text(json.dumps({"workflow_runs": [run]}))
    (tmp_path / "artifacts.json").write_text(json.dumps({"artifacts": [artifact]}))
    binary = tmp_path / "gh"
    binary.write_text(f"""#!{sys.executable}
import os, pathlib, sys
root = pathlib.Path({str(tmp_path)!r})
with (root / 'calls').open('a') as handle:
    handle.write(' '.join(sys.argv[1:]) + '\\n')
kind = 'artifacts' if '/artifacts' in sys.argv[4] else 'runs'
if kind == {failure!r}:
    sys.exit(1)
print((root / (kind + '.json')).read_text())
""")
    binary.chmod(0o755)
    env["PATH"] = str(tmp_path) + os.pathsep + env["PATH"]


def make_stage(
    env: dict[str, str], monkeypatch: pytest.MonkeyPatch, data: bytes = b"cached object"
) -> Path:
    stage = Path(env["HOME"]) / ".cache" / "gqc-cpu-123-2-fixture"
    (stage / "download").mkdir(parents=True)
    expected = snapshot.identity() | {
        "run_id": "456",
        "run_attempt": "3",
        "source_sha": "b" * 40,
    }
    (stage / "expected.json").write_text(json.dumps(expected))
    env["CPU_CACHE_STAGE"] = str(stage)
    monkeypatch.setenv("CPU_CACHE_STAGE", str(stage))
    replace_archive(stage, [(DATA_NAME, tarfile.REGTYPE, data)], expected)
    return stage


def replace_archive(
    stage: Path,
    members: list[tuple[str, bytes, bytes]],
    expected: dict[str, Any] | None = None,
) -> None:
    archive = stage / "download/cache.tar.gz"
    with tarfile.open(archive, "w:gz", format=tarfile.USTAR_FORMAT) as tar:
        for name, kind, data in members:
            info = tarfile.TarInfo(name)
            info.type, info.size = kind, len(data)
            if kind in (tarfile.SYMTYPE, tarfile.LNKTYPE):
                info.linkname = "/tmp/unowned"
            tar.addfile(info, io.BytesIO(data) if kind == tarfile.REGTYPE else None)
    manifest = (expected or json.loads((stage / "expected.json").read_text())) | {
        "sha256": snapshot.digest(archive),
        "archive_bytes": archive.stat().st_size,
        "files": len(members),
        "decoded_bytes": sum(len(data) for _, _, data in members),
    }
    (stage / "download/manifest.json").write_text(json.dumps(manifest))


def refresh_digest(stage: Path) -> None:
    manifest_path = stage / "download/manifest.json"
    manifest = json.loads(manifest_path.read_text())
    archive = stage / "download/cache.tar.gz"
    manifest.update(
        sha256=snapshot.digest(archive), archive_bytes=archive.stat().st_size
    )
    manifest_path.write_text(json.dumps(manifest))


def test_discovery_executes_scoped_api_and_exact_artifact_id(
    environment: dict[str, str], tmp_path: Path
) -> None:
    run = fixture_run()
    fake_api(tmp_path, environment, run, fixture_artifact(run))
    result = run_command("discover", environment)
    assert result.returncode == 0, result.stderr
    selected = outputs(environment)
    assert selected["artifact_id"] == "789" and selected["run_id"] == "456"
    assert (
        json.loads((Path(selected["stage"]) / "expected.json").read_text())[
            "run_attempt"
        ]
        == "3"
    )
    calls = (tmp_path / "calls").read_text().splitlines()
    assert len(calls) == 2
    assert "/actions/workflows/ci.yml/runs" in calls[0]
    assert all(
        part in calls[0] for part in ("branch=master", "event=push", "status=success")
    )
    assert "/actions/runs/456/artifacts" in calls[1]


@pytest.mark.parametrize(
    ("key", "bad"),
    [
        ("path", ".github/workflows/other.yml"),
        ("event", "pull_request"),
        ("event", "merge_group"),
        ("head_branch", "topic"),
        ("status", "in_progress"),
        ("conclusion", "failure"),
        ("conclusion", None),
        ("head_sha", "bad"),
        ("run_attempt", 0),
        ("id", "../wrong"),
        ("repository", {"id": 9, "full_name": "other/repo"}),
        ("head_repository", {"id": 10, "full_name": REPO}),
    ],
)
def test_untrusted_runs_never_reach_artifact_lookup(
    environment: dict[str, str], tmp_path: Path, key: str, bad: Any
) -> None:
    run = fixture_run()
    artifact = fixture_artifact(run)
    run[key] = bad
    fake_api(tmp_path, environment, run, artifact)
    assert run_command("discover", environment).returncode == 0
    assert not outputs(environment)
    assert len((tmp_path / "calls").read_text().splitlines()) == 1


@pytest.mark.parametrize(
    ("key", "bad"),
    [
        ("name", "other"),
        ("expired", True),
        ("size_in_bytes", snapshot.MAX_ARTIFACT + 1),
        ("size_in_bytes", 0),
        ("digest", ""),
        ("digest", "sha256:bad"),
        ("id", 0),
    ],
)
def test_untrusted_artifacts_never_download(
    environment: dict[str, str], tmp_path: Path, key: str, bad: Any
) -> None:
    run = fixture_run()
    artifact = fixture_artifact(run)
    artifact[key] = bad
    fake_api(tmp_path, environment, run, artifact)
    assert run_command("discover", environment).returncode == 0
    assert not outputs(environment)


@pytest.mark.parametrize(
    "field", ["id", "head_sha", "head_branch", "repository_id", "head_repository_id"]
)
def test_artifact_origin_must_match_selected_run(field: str) -> None:
    run = fixture_run()
    artifact = fixture_artifact(run)
    artifact["workflow_run"][field] = "wrong"
    assert not snapshot.trusted_artifact(artifact, run)


@pytest.mark.parametrize("failure", ["runs", "artifacts"])
def test_api_failure_has_no_retry_or_activation(
    environment: dict[str, str], tmp_path: Path, failure: str
) -> None:
    run = fixture_run()
    fake_api(tmp_path, environment, run, fixture_artifact(run), failure=failure)
    result = run_command("discover", environment)
    assert result.returncode == (1 if failure == "runs" else 0)
    assert not outputs(environment)
    assert len((tmp_path / "calls").read_text().splitlines()) == (
        1 if failure == "runs" else 2
    )


def test_snapshot_roundtrip_preserves_data_but_excludes_config(
    environment: dict[str, str], monkeypatch: pytest.MonkeyPatch
) -> None:
    cache = Path(environment["HOME"]) / ".cache/ccache"
    data = cache / DATA_NAME
    data.parent.mkdir(parents=True)
    data.write_bytes(b"compiled cache data")
    (cache / "ccache.conf").write_text("sloppiness = time_macros\n")
    (data.parent / "stats").write_text("stats")
    assert run_command("package", environment).returncode == 0
    packaged = Path(outputs(environment)["path"])
    stage = make_stage(environment, monkeypatch)
    manifest = json.loads((packaged / "manifest.json").read_text())
    (stage / "expected.json").write_text(
        json.dumps(
            {
                k: manifest[k]
                for k in snapshot.identity()
                | {"run_id": "", "run_attempt": "", "source_sha": ""}
            }
        )
    )
    for path in packaged.iterdir():
        (stage / "download" / path.name).write_bytes(path.read_bytes())
    # The source cache is our fixture, not a production cache. Move it aside and
    # verify successful restore never imports the sloppiness configuration.
    cache.rename(cache.parent / "original")
    assert run_command("verify", environment).returncode == 0
    assert not cache.exists()
    assert run_command("activate", environment).returncode == 0
    assert (cache / DATA_NAME).read_bytes() == b"compiled cache data"
    assert not (cache / "ccache.conf").exists()
    assert run_command("cleanup", environment).returncode == 0
    assert (cache / DATA_NAME).exists()
    assert (cache.parent / "original/ccache.conf").exists()


@pytest.mark.parametrize(
    "field",
    [
        "run_id",
        "run_attempt",
        "source_sha",
        "repository",
        "workflow",
        "event",
        "ref",
        "ccache_version",
        "arch",
        "image_os",
        "format",
    ],
)
def test_manifest_identity_mismatch_stays_inactive(
    environment: dict[str, str], monkeypatch: pytest.MonkeyPatch, field: str
) -> None:
    stage = make_stage(environment, monkeypatch)
    manifest_path = stage / "download/manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest[field] = "wrong"
    manifest_path.write_text(json.dumps(manifest))
    assert run_command("verify", environment).returncode != 0
    assert run_command("activate", environment).returncode != 0
    assert not (Path(environment["HOME"]) / ".cache/ccache").exists()


@pytest.mark.parametrize(
    ("name", "kind"),
    [
        ("../../escape", tarfile.REGTYPE),
        ("/tmp/escape", tarfile.REGTYPE),
        ("a/../ccache.conf", tarfile.REGTYPE),
        ("ccache.conf", tarfile.REGTYPE),
        (DATA_NAME, tarfile.SYMTYPE),
        (DATA_NAME, tarfile.LNKTYPE),
        (DATA_NAME, tarfile.FIFOTYPE),
        (DATA_NAME, tarfile.CHRTYPE),
        (DATA_NAME, tarfile.DIRTYPE),
        (DATA_NAME, tarfile.XHDTYPE),
        (DATA_NAME, tarfile.GNUTYPE_LONGNAME),
    ],
)
def test_unsafe_archive_members_stay_inactive(
    environment: dict[str, str], monkeypatch: pytest.MonkeyPatch, name: str, kind: bytes
) -> None:
    stage = make_stage(environment, monkeypatch)
    replace_archive(stage, [(name, kind, b"" if kind != tarfile.REGTYPE else b"bad")])
    assert run_command("verify", environment).returncode != 0
    assert not (stage / "verified").exists()
    assert run_command("activate", environment).returncode != 0


@pytest.mark.parametrize(
    "damage",
    ["digest", "truncated", "truncated-footer", "oversized", "duplicate", "appended"],
)
def test_damaged_archives_stay_inactive(
    environment: dict[str, str], monkeypatch: pytest.MonkeyPatch, damage: str
) -> None:
    stage = make_stage(environment, monkeypatch)
    archive = stage / "download/cache.tar.gz"
    if damage == "duplicate":
        replace_archive(
            stage,
            [
                (DATA_NAME, tarfile.REGTYPE, b"one"),
                (DATA_NAME, tarfile.REGTYPE, b"two"),
            ],
        )
    elif damage == "oversized":
        monkeypatch.setattr(snapshot, "MAX_ARCHIVE", 1)
    else:
        data = archive.read_bytes()
        if damage == "digest":
            archive.write_bytes(data + b"bad")
        elif damage == "appended":
            archive.write_bytes(data + gzip.compress(b"payload"))
            refresh_digest(stage)
        else:
            archive.write_bytes(
                data[: len(data) // 2] if damage == "truncated" else data[:-4]
            )
            refresh_digest(stage)
    with pytest.raises((ValueError, EOFError, OSError, tarfile.TarError)):
        snapshot.verify()
    assert not (stage / "verified").exists()


def test_decoded_and_file_caps_are_enforced_before_activation(
    environment: dict[str, str], monkeypatch: pytest.MonkeyPatch
) -> None:
    stage = make_stage(environment, monkeypatch, b"x" * 4096)
    monkeypatch.setattr(snapshot, "MAX_DECODED", 100)
    with pytest.raises(ValueError, match="decoded"):
        snapshot.verify()
    assert not (stage / "cache" / DATA_NAME).exists()
    assert not (stage / "verified").exists()


def test_encoded_producer_cap_does_not_modify_source(
    environment: dict[str, str], monkeypatch: pytest.MonkeyPatch
) -> None:
    cache = Path(environment["HOME"]) / ".cache/ccache"
    data = cache / DATA_NAME
    data.parent.mkdir(parents=True)
    data.write_bytes(os.urandom(4096))
    before = data.read_bytes()
    monkeypatch.setattr(snapshot, "MAX_ARCHIVE", 50)
    with pytest.raises(ValueError, match="encoded"):
        snapshot.package()
    assert data.read_bytes() == before
    assert not outputs(environment)


@pytest.mark.parametrize("boundary", ["download", "verify", "verified", "activated"])
def test_interruption_cleanup_never_exposes_partial_or_removes_primary(
    environment: dict[str, str], monkeypatch: pytest.MonkeyPatch, boundary: str
) -> None:
    stage = make_stage(environment, monkeypatch)
    target = Path(environment["HOME"]) / ".cache/ccache"
    if boundary == "verify":
        (stage / "cache").mkdir()
        (stage / "cache/partial").write_text("incomplete")
    if boundary in ("verified", "activated"):
        snapshot.verify()
    if boundary == "activated":
        snapshot.activate()
    assert target.exists() == (boundary == "activated")
    snapshot.cleanup()
    assert target.exists() == (boundary == "activated")
    if target.exists():
        assert (target / DATA_NAME).read_bytes() == b"cached object"


def test_existing_normal_cache_and_unowned_staging_are_untouched(
    environment: dict[str, str], monkeypatch: pytest.MonkeyPatch
) -> None:
    stage = make_stage(environment, monkeypatch)
    snapshot.verify()
    target = Path(environment["HOME"]) / ".cache/ccache"
    target.mkdir()
    (target / "sentinel").write_text("existing")
    assert run_command("discover", environment).returncode == 0
    assert run_command("activate", environment).returncode != 0
    for wrong in [target, stage.with_name("gqc-cpu-123-1-old")]:
        wrong.mkdir(exist_ok=True)
        monkeypatch.setenv("CPU_CACHE_STAGE", str(wrong))
        with pytest.raises(ValueError, match="owned"):
            snapshot.cleanup()
    assert (target / "sentinel").read_text() == "existing"


def condition(expression: str, context: dict[str, str]) -> bool:
    for key, value in sorted(context.items(), key=lambda item: -len(item[0])):
        expression = expression.replace(key, repr(value))
    expression = (
        expression.replace("always()", "True")
        .replace("&&", " and ")
        .replace("||", " or ")
    )

    def evaluate(node: ast.AST) -> Any:
        if isinstance(node, ast.Constant):
            return node.value
        if isinstance(node, ast.Compare) and len(node.ops) == 1:
            equal = evaluate(node.left) == evaluate(node.comparators[0])
            return equal if isinstance(node.ops[0], ast.Eq) else not equal
        if isinstance(node, ast.BoolOp):
            values = [bool(evaluate(value)) for value in node.values]
            return all(values) if isinstance(node.op, ast.And) else any(values)
        raise AssertionError(ast.dump(node))

    return bool(evaluate(ast.parse(expression, mode="eval").body))


def assert_policy(workflow: dict[str, Any], action: dict[str, Any]) -> None:
    for job, primary in [
        ("master-cpu-cache", "master_cpu_ccache"),
        ("cpu", "cpu_ccache"),
        ("python", "python_ccache"),
        ("cpu-benchmark", "benchmark_ccache"),
    ]:
        scope = workflow["jobs"][job]
        assert scope["permissions"] == {"contents": "read", "actions": "read"}
        steps = scope["steps"]
        restore = next(step for step in steps if step.get("id") == primary)
        fallback = steps[steps.index(restore) + 1]
        assert fallback["uses"] == "./.github/actions/restore-cpu-ccache"
        assert (
            fallback["continue-on-error"] is True and fallback["timeout-minutes"] == 2
        )
        assert restore["with"]["path"] == "~/.cache/ccache"
        for outcome in ["success", "failure", "cancelled", "skipped", ""]:
            for key in ["", "exact-key", "prefix-key"]:
                for selector_outcome, test_scope in [
                    ("success", "full"),
                    ("success", "prose"),
                    ("success", ""),
                    ("failure", "prose"),
                    ("", "prose"),
                ]:
                    prose = (
                        job in {"cpu", "python"}
                        and selector_outcome == "success"
                        and test_scope == "prose"
                    )
                    assert condition(
                        fallback["if"],
                        {
                            f"steps.{primary}.outcome": outcome,
                            f"steps.{primary}.outputs.cache-matched-key": key,
                            "steps.prose_scope.outcome": selector_outcome,
                            "steps.prose_scope.outputs.scope": test_scope,
                        },
                    ) == (outcome == "success" and key == "" and not prose)
    steps = action["runs"]["steps"]
    download, verify, activate = steps[1:4]
    assert download["with"]["digest-mismatch"] == "error"
    assert (
        download["with"]["artifact-ids"] == "${{ steps.discover.outputs.artifact_id }}"
    )
    assert download["with"]["run-id"] == "${{ steps.discover.outputs.run_id }}"
    assert download["with"]["repository"] == "${{ github.repository }}"
    assert download["with"]["path"] == "${{ steps.discover.outputs.stage }}/download"
    for outcome in ["success", "failure", "cancelled", "skipped", ""]:
        assert condition(verify["if"], {"steps.download.outcome": outcome}) == (
            outcome == "success"
        )
        assert condition(activate["if"], {"steps.verify.outcome": outcome}) == (
            outcome == "success"
        )
    producer = workflow["jobs"]["master-cpu-cache"]["steps"]
    package = next(step for step in producer if step.get("id") == "cpu_snapshot")
    upload = producer[producer.index(package) + 1]
    assert upload["with"]["retention-days"] == 1
    assert upload["with"]["compression-level"] == 0
    assert upload["timeout-minutes"] == package["timeout-minutes"] == 2
    for event in ["push", "pull_request", "merge_group"]:
        for ref in ["refs/heads/master", "refs/heads/topic"]:
            for cpu in ["success", "failure", "skipped"]:
                for python in ["success", "failure", "skipped"]:
                    assert condition(
                        package["if"],
                        {
                            "github.event_name": event,
                            "github.ref": ref,
                            "steps.cpu_seed.outcome": cpu,
                            "steps.python_seed.outcome": python,
                        },
                    ) == (
                        event == "push"
                        and ref == "refs/heads/master"
                        and cpu == python == "success"
                    )


def test_actual_workflow_and_action_decisions() -> None:
    assert_policy(WORKFLOW, ACTION)


@pytest.mark.parametrize(
    "mutation",
    ["timeout", "primary-hit", "digest", "outcome", "artifact", "permission"],
)
def test_policy_rejects_negative_mutations(mutation: str) -> None:
    workflow, action = copy.deepcopy(WORKFLOW), copy.deepcopy(ACTION)
    steps = workflow["jobs"]["cpu"]["steps"]
    fallback = next(
        step
        for step in steps
        if step.get("uses") == "./.github/actions/restore-cpu-ccache"
    )
    if mutation == "timeout":
        fallback["timeout-minutes"] = 20
    elif mutation == "primary-hit":
        fallback["if"] = fallback["if"].replace("cache-matched-key", "cache-hit")
    elif mutation == "digest":
        action["runs"]["steps"][1]["with"]["digest-mismatch"] = "warn"
    elif mutation == "outcome":
        action["runs"]["steps"][3]["if"] = "steps.verify.conclusion == 'success'"
    elif mutation == "artifact":
        action["runs"]["steps"][1]["with"]["artifact-ids"] = "1"
    else:
        workflow["jobs"]["cpu"]["permissions"]["actions"] = "write"
    with pytest.raises((AssertionError, SyntaxError)):
        assert_policy(workflow, action)


@pytest.mark.parametrize("outcome", ["success", "failure", "cancelled", "skipped"])
def test_action_shells_obey_download_outcome(
    environment: dict[str, str], monkeypatch: pytest.MonkeyPatch, outcome: str
) -> None:
    stage = make_stage(environment, monkeypatch)
    steps = ACTION["runs"]["steps"]
    verify, activate = steps[2:4]
    verified = "skipped"
    if condition(verify["if"], {"steps.download.outcome": outcome}):
        result = subprocess.run(
            ["bash", "-e", "-c", verify["run"]],
            cwd=ROOT,
            env=environment,
            capture_output=True,
            timeout=5,
            check=False,
        )
        verified = "success" if result.returncode == 0 else "failure"
    if condition(activate["if"], {"steps.verify.outcome": verified}):
        result = subprocess.run(
            ["bash", "-e", "-c", activate["run"]],
            cwd=ROOT,
            env=environment,
            capture_output=True,
            timeout=5,
            check=False,
        )
        assert result.returncode == 0
    target = Path(environment["HOME"]) / ".cache/ccache"
    assert target.exists() == (outcome == "success")
    result = subprocess.run(
        ["bash", "-e", "-c", steps[4]["run"]],
        cwd=ROOT,
        env=environment,
        capture_output=True,
        timeout=5,
        check=False,
    )
    assert result.returncode == 0 and not stage.exists()
    assert target.exists() == (outcome == "success")


def test_timeout_after_verified_marker_never_means_success(
    environment: dict[str, str], monkeypatch: pytest.MonkeyPatch
) -> None:
    stage = make_stage(environment, monkeypatch)
    verify, activate = ACTION["runs"]["steps"][2:4]
    # Exercise a real shell/process timeout after verification returned but
    # before the runner would mark the step successful.
    result = subprocess.run(
        [
            "timeout",
            "--kill-after=1s",
            "0.3s",
            "bash",
            "-e",
            "-c",
            verify["run"] + "; sleep 10",
        ],
        cwd=ROOT,
        env=environment,
        capture_output=True,
        timeout=3,
        check=False,
    )
    assert result.returncode == 124 and (stage / "verified").exists()
    assert not condition(activate["if"], {"steps.verify.outcome": "failure"})
    assert not (Path(environment["HOME"]) / ".cache/ccache").exists()
    snapshot.cleanup()


def test_file_count_and_producer_decoded_caps(
    environment: dict[str, str], monkeypatch: pytest.MonkeyPatch
) -> None:
    make_stage(environment, monkeypatch)
    monkeypatch.setattr(snapshot, "MAX_FILES", 0)
    with pytest.raises(ValueError, match="decoded"):
        snapshot.verify()
    cache = Path(environment["HOME"]) / ".cache/ccache"
    data = cache / DATA_NAME
    data.parent.mkdir(parents=True)
    data.write_bytes(b"object")
    with pytest.raises(ValueError, match="decoded"):
        snapshot.package()
    assert data.read_bytes() == b"object"


def test_attempt_scoped_cleanup_rejects_symlink(
    environment: dict[str, str], monkeypatch: pytest.MonkeyPatch
) -> None:
    stage = make_stage(environment, monkeypatch)
    link = stage.with_name("gqc-cpu-123-2-link")
    link.symlink_to(stage, target_is_directory=True)
    monkeypatch.setenv("CPU_CACHE_STAGE", str(link))
    with pytest.raises(ValueError, match="owned"):
        snapshot.cleanup()
    assert (stage / "download/manifest.json").exists()
