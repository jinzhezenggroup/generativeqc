"""Execute optional CLI snapshot provenance, fallback and atomic activation."""

from __future__ import annotations

import ast
import copy
import errno
import importlib.util
import re
import subprocess
from pathlib import Path
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location(
    "cli_snapshot", ROOT / "tools/ci_cli_ccache_snapshot.py"
)
assert SPEC and SPEC.loader
snapshot = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(snapshot)
RID = 1347498218
PREFIX = f"/repos/{snapshot.REPOSITORY}/actions"
RUN = {
    "id": 100,
    "path": snapshot.WORKFLOW,
    "event": "push",
    "head_branch": "master",
    "status": "completed",
    "conclusion": "success",
    "head_sha": "a" * 40,
    "repository": {"id": RID, "full_name": snapshot.REPOSITORY},
    "head_repository": {"id": RID, "full_name": snapshot.REPOSITORY},
}
ARTIFACT = {
    "id": 200,
    "name": snapshot.ARTIFACT,
    "expired": False,
    "size_in_bytes": 4977835037,
    "digest": "sha256:" + "b" * 64,
    "workflow_run": {
        "id": 100,
        "head_sha": "a" * 40,
        "head_branch": "master",
        "repository_id": RID,
        "head_repository_id": RID,
    },
}


@pytest.fixture
def environment(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    (tmp_path / "home").mkdir()
    for key, value in {
        "HOME": str(tmp_path / "home"),
        "RUNNER_TEMP": str(tmp_path),
        "GITHUB_REPOSITORY": snapshot.REPOSITORY,
        "GITHUB_REPOSITORY_ID": str(RID),
        "GITHUB_EVENT_NAME": "merge_group",
        "GITHUB_REF": "refs/heads/gh-readonly-queue/master/pr-1",
        "GITHUB_RUN_ID": "300",
        "GITHUB_RUN_ATTEMPT": "2",
        "GITHUB_OUTPUT": str(tmp_path / "outputs"),
        "CLI_CACHE_STAGE": str(tmp_path / "home/.cache/trusted-cli-ccache-300-2"),
    }.items():
        monkeypatch.setenv(key, value)
    return tmp_path


def mock_api(monkeypatch: pytest.MonkeyPatch, runs: list, artifacts: list) -> list:
    calls = []

    def api(endpoint: str, **fields: str) -> dict:
        calls.append((endpoint, fields))
        return (
            {"workflow_runs": copy.deepcopy(runs)}
            if endpoint.endswith("/runs")
            else {"artifacts": copy.deepcopy(artifacts)}
        )

    monkeypatch.setattr(snapshot, "api", api)
    return calls


@pytest.mark.parametrize(
    ("event", "ref", "repo", "accepted"),
    [
        (
            "merge_group",
            "refs/heads/gh-readonly-queue/master/pr-1",
            snapshot.REPOSITORY,
            True,
        ),
        ("pull_request", "refs/pull/1/merge", snapshot.REPOSITORY, True),
        ("push", "refs/heads/master", snapshot.REPOSITORY, True),
        ("push", "refs/heads/topic", snapshot.REPOSITORY, False),
        ("push", "refs/tags/v1.0.0", snapshot.REPOSITORY, False),
        ("workflow_dispatch", "refs/heads/master", snapshot.REPOSITORY, False),
        ("pull_request_target", "refs/heads/master", snapshot.REPOSITORY, False),
        ("push", "refs/heads/master", "fork/generativeqc", False),
    ],
)
def test_consumer_event_scope(
    environment: Path,
    monkeypatch: pytest.MonkeyPatch,
    event: str,
    ref: str,
    repo: str,
    accepted: bool,
) -> None:
    for key, value in [
        ("GITHUB_EVENT_NAME", event),
        ("GITHUB_REF", ref),
        ("GITHUB_REPOSITORY", repo),
    ]:
        monkeypatch.setenv(key, value)
    calls = mock_api(monkeypatch, [RUN], [ARTIFACT])
    snapshot.discover()
    assert (environment / "outputs").exists() == accepted
    assert bool(calls) == accepted
    if accepted:
        assert "run_id=100\nartifact_id=200\n" in (environment / "outputs").read_text()
        assert calls == [
            (
                f"{PREFIX}/workflows/ci.yml/runs",
                {
                    "branch": "master",
                    "event": "push",
                    "status": "success",
                    "per_page": "20",
                },
            ),
            (
                f"{PREFIX}/runs/100/artifacts",
                {"name": snapshot.ARTIFACT, "per_page": "10"},
            ),
        ]


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("id", "100"),
        ("id", True),
        ("path", ".github/workflows/other.yml"),
        ("path", None),
        ("event", "pull_request"),
        ("event", "merge_group"),
        ("head_branch", "topic"),
        ("status", "in_progress"),
        ("conclusion", "failure"),
        ("conclusion", None),
        ("head_sha", "bad"),
        ("repository", {"id": RID, "full_name": "other/repo"}),
        ("head_repository", {"id": 2, "full_name": snapshot.REPOSITORY}),
        ("head_repository", None),
    ],
)
def test_untrusted_run_never_requests_artifact(
    environment: Path, monkeypatch: pytest.MonkeyPatch, key: str, value: Any
) -> None:
    calls = mock_api(monkeypatch, [{**RUN, key: value}], [ARTIFACT])
    snapshot.discover()
    assert not (environment / "outputs").exists() and len(calls) == 1


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("id", "200"),
        ("id", True),
        ("name", "other"),
        ("expired", True),
        ("expired", None),
        ("size_in_bytes", 0),
        ("size_in_bytes", snapshot.MAX_BYTES + 1),
        ("digest", None),
        ("digest", "sha256:bad"),
        ("workflow_run", {}),
        ("workflow_run", "invalid"),
    ],
)
def test_untrusted_artifact_is_rejected(
    environment: Path, monkeypatch: pytest.MonkeyPatch, key: str, value: Any
) -> None:
    mock_api(monkeypatch, [RUN], [{**ARTIFACT, key: value}])
    snapshot.discover()
    assert not (environment / "outputs").exists()


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("id", 101),
        ("head_sha", "c" * 40),
        ("head_branch", "topic"),
        ("repository_id", 1),
        ("head_repository_id", 2),
    ],
)
def test_artifact_run_mismatch(
    environment: Path, monkeypatch: pytest.MonkeyPatch, key: str, value: Any
) -> None:
    artifact = copy.deepcopy(ARTIFACT)
    artifact["workflow_run"][key] = value
    mock_api(monkeypatch, [RUN], [artifact])
    snapshot.discover()
    assert not (environment / "outputs").exists()


@pytest.mark.parametrize("key", list(RUN))
def test_missing_run_provenance(key: str) -> None:
    assert not snapshot.trusted_run({k: v for k, v in RUN.items() if k != key}, RID)


@pytest.mark.parametrize("key", list(ARTIFACT))
def test_missing_artifact_provenance(key: str) -> None:
    assert not snapshot.trusted_artifact(
        {k: v for k, v in ARTIFACT.items() if k != key}, RUN, RID
    )


def test_discovery_has_bounded_attempts(
    environment: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls = []

    def api(endpoint: str, **fields: str) -> dict:
        calls.append(endpoint)
        if endpoint.endswith("/runs"):
            return {"workflow_runs": [{**RUN, "id": index} for index in range(1, 25)]}
        raise subprocess.TimeoutExpired("gh", 5)

    monkeypatch.setattr(snapshot, "api", api)
    snapshot.discover()
    assert len(calls) == 21 and not (environment / "outputs").exists()


def test_discovery_timeout_has_no_selection(
    environment: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def fail(*args: Any, **kwargs: Any) -> dict:
        raise subprocess.TimeoutExpired("gh", 5)

    monkeypatch.setattr(snapshot, "api", fail)
    monkeypatch.setattr("sys.argv", ["snapshot", "discover"])
    assert snapshot.main() == 1 and not (environment / "outputs").exists()


def test_api_transport_is_read_only_and_bounded(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = []

    def run(args: list, **kwargs: Any) -> subprocess.CompletedProcess:
        calls.append((args, kwargs))
        return subprocess.CompletedProcess(args, 0, '{"workflow_runs": []}')

    monkeypatch.setattr(subprocess, "run", run)
    assert snapshot.api(f"{PREFIX}/workflows/ci.yml/runs", per_page="20") == {
        "workflow_runs": []
    }
    args, options = calls[0]
    assert args[:4] == ["gh", "api", "--method", "GET"]
    assert options["timeout"] == 5 and options["check"] is True


def test_atomic_activation_and_cleanup(environment: Path) -> None:
    stage = snapshot.stage_path()
    (stage / "a").mkdir(parents=True)
    entry = stage / "a/strict-source-resultR"
    entry.write_bytes(b"source-and-compiler-bound object")
    inode = entry.stat().st_ino
    snapshot.activate()
    target = Path.home() / ".cache/ccache/a/strict-source-resultR"
    assert target.read_bytes() == b"source-and-compiler-bound object"
    assert target.stat().st_ino == inode and not stage.exists()
    snapshot.cleanup()
    assert target.exists()


@pytest.mark.parametrize("existing", ["empty", "populated", "symlink"])
def test_existing_destination_is_preserved(
    environment: Path, monkeypatch: pytest.MonkeyPatch, existing: str
) -> None:
    snapshot.stage_path().mkdir(parents=True)
    target = Path.home() / ".cache/ccache"
    target.parent.mkdir(exist_ok=True)
    if existing == "symlink":
        target.symlink_to(environment / "other")
    else:
        target.mkdir()
        if existing == "populated":
            (target / "untouched").write_text("existing cache")
    calls = mock_api(monkeypatch, [RUN], [ARTIFACT])
    with pytest.raises(ValueError):
        snapshot.discover()
    assert not calls
    with pytest.raises(ValueError):
        snapshot.activate()
    assert snapshot.stage_path().is_dir()
    if existing == "populated":
        assert (target / "untouched").read_text() == "existing cache"


def test_cross_device_failure_never_copies_partial_data(
    environment: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    stage = snapshot.stage_path()
    stage.mkdir(parents=True)
    (stage / "object").write_text("complete")

    def fail(*args: Any) -> None:
        raise OSError(errno.EXDEV, "cross-device link")

    monkeypatch.setattr(Path, "rename", fail)
    with pytest.raises(OSError):
        snapshot.activate()
    assert not (Path.home() / ".cache/ccache").exists()
    assert (stage / "object").read_text() == "complete"
    snapshot.cleanup()
    assert not stage.exists()


@pytest.mark.parametrize("command", ["activate", "cleanup"])
def test_foreign_and_symlink_stages_are_rejected(
    environment: Path, monkeypatch: pytest.MonkeyPatch, command: str
) -> None:
    other = environment / "other"
    other.mkdir()
    (other / "keep").write_text("untouched")
    monkeypatch.setenv("CLI_CACHE_STAGE", str(other))
    with pytest.raises(ValueError):
        getattr(snapshot, command)()
    monkeypatch.setenv("CLI_CACHE_STAGE", str(snapshot.stage_path()))
    snapshot.stage_path().parent.mkdir(parents=True, exist_ok=True)
    snapshot.stage_path().symlink_to(other, target_is_directory=True)
    with pytest.raises(ValueError):
        getattr(snapshot, command)()
    assert (other / "keep").read_text() == "untouched"


def step(name: str) -> str:
    workflow = (ROOT / ".github/workflows/ci.yml").read_text()
    job = workflow.split("\n  native-cli-artifact:\n", 1)[1].split(
        "\n  native-cli-qualification:\n", 1
    )[0]
    return job.split(f"      - name: {name}\n", 1)[1].split("\n      - ", 1)[0]


def field(section: str, name: str) -> str:
    match = re.search(rf"^\s*{re.escape(name)}: (.+)$", section, re.MULTILINE)
    assert match
    return match[1]


def condition(expression: str, context: dict[str, Any]) -> bool:
    expression = expression.removeprefix("${{ ").removesuffix(" }}")
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
        if isinstance(node, ast.BoolOp):
            values = [bool(evaluate(value)) for value in node.values]
            return all(values) if isinstance(node.op, ast.And) else any(values)
        if isinstance(node, ast.Compare) and len(node.ops) == 1:
            equal = evaluate(node.left) == evaluate(node.comparators[0])
            return equal if isinstance(node.ops[0], ast.Eq) else not equal
        raise AssertionError(ast.dump(node))

    return bool(evaluate(ast.parse(expression, mode="eval").body))


@pytest.mark.parametrize("tier", ["skip", "fast", "release"])
@pytest.mark.parametrize("primary", ["", "exact-primary", "prefix-fallback"])
@pytest.mark.parametrize("outcome", ["success", "failure", "skipped"])
@pytest.mark.parametrize(
    ("event", "ref", "eligible"),
    [
        ("pull_request", "refs/pull/1/merge", True),
        ("merge_group", "refs/heads/gh-readonly-queue/master/pr-1", True),
        ("push", "refs/heads/master", True),
        ("push", "refs/tags/v1.0.0", False),
        ("workflow_dispatch", "refs/heads/master", False),
    ],
)
def test_workflow_only_recovers_true_misses(
    tier: str, primary: str, outcome: str, event: str, ref: str, eligible: bool
) -> None:
    assert condition(
        field(step("Locate trusted master CLI ccache snapshot"), "if"),
        {
            "steps.cli_scope.outputs.tier": tier,
            "steps.cli_ccache.outputs.cache-matched-key": primary,
            "steps.cli_ccache.outcome": outcome,
            "github.event_name": event,
            "github.ref": ref,
        },
    ) == (tier != "skip" and not primary and outcome == "success" and eligible)


@pytest.mark.parametrize("download", ["success", "failure", "cancelled", "skipped", ""])
def test_failed_or_partial_download_never_activates(
    environment: Path, download: str
) -> None:
    stage = snapshot.stage_path()
    stage.mkdir(parents=True)
    (stage / "object").write_text("downloaded bytes")
    accepted = condition(
        field(step("Activate verified trusted CLI ccache snapshot"), "if"),
        {
            "steps.trusted_cli_ccache_download.outcome": download,
        },
    )
    assert accepted == (download == "success")
    if accepted:
        snapshot.activate()
    else:
        snapshot.cleanup()
    assert (Path.home() / ".cache/ccache/object").exists() == accepted
    assert not stage.exists()


def test_workflow_transport_provenance_bounds_and_failure_isolation() -> None:
    discover = step("Locate trusted master CLI ccache snapshot")
    download = step("Restore trusted master CLI ccache snapshot")
    activate = step("Activate verified trusted CLI ccache snapshot")
    cleanup = step("Discard incomplete trusted CLI cache download")
    for item, budget in [(discover, 1), (download, 2), (activate, 1), (cleanup, 1)]:
        assert field(item, "continue-on-error") == "true"
        assert 0 < int(field(item, "timeout-minutes")) <= budget
    assert "timeout --kill-after=2s 25s" in discover
    assert field(download, "digest-mismatch") == "error"
    assert (
        field(download, "artifact-ids")
        == "${{ steps.trusted_cli_ccache.outputs.artifact_id }}"
    )
    assert field(download, "run-id") == "${{ steps.trusted_cli_ccache.outputs.run_id }}"
    assert field(download, "repository") == "${{ github.repository }}"
    assert field(download, "github-token") == "${{ github.token }}"
    assert (
        field(download, "path")
        == field(activate, "CLI_CACHE_STAGE")
        == field(cleanup, "CLI_CACHE_STAGE")
    )
    assert field(download, "path") != field(
        step("Restore shared Linux CUDA ccache"), "path"
    )
    for outcome in ("failure", "cancelled", "skipped", ""):
        assert not condition(
            field(download, "if"),
            {
                "steps.trusted_cli_ccache.outcome": outcome,
                "steps.trusted_cli_ccache.outputs.artifact_id": "200",
            },
        )
    workflow = (
        (ROOT / ".github/workflows/ci.yml")
        .read_text()
        .split("    steps: &native_cli_package_steps", 1)[1]
    )
    positions = [
        workflow.index(name)
        for name in (
            "Restore shared Linux CUDA ccache",
            "Locate trusted master CLI ccache snapshot",
            "Restore trusted master CLI ccache snapshot",
            "Activate verified trusted CLI ccache snapshot",
            "Discard incomplete trusted CLI cache download",
            "Seed fast CUDA CI objects in the shared cache",
            "Build monolithic C++ CUDA CLI (release or fast PR)",
        )
    ]
    assert positions == sorted(positions)


@pytest.mark.parametrize("primary_hit", ["", "false", "true"])
@pytest.mark.parametrize("publish", ["false", "true"])
def test_master_save_and_publication_stay_independent(
    primary_hit: str, publish: str
) -> None:
    context = {
        "github.event_name": "push",
        "github.ref": "refs/heads/master",
        "steps.cli_ccache.outputs.cache-hit": primary_hit,
        "env.PUBLISH_CUDA_CACHE_SNAPSHOT": publish,
    }
    save = step("Save complete shared CUDA ccache on master")
    publish_step = step("Publish trusted master CUDA ccache snapshot")
    assert condition(field(save, "if"), context) == (primary_hit != "true")
    assert condition(field(publish_step, "if"), context) == (publish == "true")
    assert field(save, "key") == "${{ steps.cli_ccache.outputs.cache-primary-key }}"
    assert field(save, "path") == field(publish_step, "path") == "~/.cache/ccache"
    assert "lookup-only" not in step("Restore shared Linux CUDA ccache")


def test_stage_is_a_sibling_of_canonical_cache_not_runner_temp(
    environment: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("RUNNER_TEMP", "/different-filesystem")
    stage = snapshot.stage_path()
    assert stage.parent == (Path.home() / ".cache").resolve()
    assert stage.name == "trusted-cli-ccache-300-2"
    assert stage != Path.home() / ".cache/ccache"
