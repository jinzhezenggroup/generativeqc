"""Exercise snapshot retention with a fake API, never live artifact deletion."""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
WORKFLOW = (ROOT / ".github/workflows/ci.yml").read_text()
JOB = WORKFLOW.split("\n  master-cache-cleanup:\n", 1)[1].split(
    "\n  cuda-resources:\n", 1
)[0]
# Stop at the end of the job's script, excluding comments for the next job.
SCRIPT = textwrap.dedent(JOB.split("        run: |\n", 1)[1].split("\n  #", 1)[0])
PREFIX = "/repos/owner/repo/actions"
NAME = "cuda-ccache-master-12.9-sm120-v2"
pytestmark = pytest.mark.skipif(
    not all(shutil.which(command) for command in ("bash", "jq")),
    reason="requires the workflow's host shell tools",
)
MOCK_GH = """#!{python}
import json
import os
import sys
args = sys.argv[1:]
assert args[0] == 'api'
method = args[args.index('--method') + 1]
endpoint = next(value for value in args if value.startswith('/repos/'))
with open(os.environ['MOCK_CALLS'], 'a') as stream:
    stream.write(json.dumps([method, endpoint, args]) + '\\n')
responses = json.loads(os.environ['MOCK_RESPONSES'])
if method == 'DELETE':
    assert endpoint.startswith('/repos/owner/repo/actions/artifacts/')
    sys.exit(0)
response = responses.get(endpoint)
if response is None:
    sys.exit(23)
if endpoint.endswith('/artifacts'):
    assert '--paginate' in args and '--slurp' in args
    assert 'name=cuda-ccache-master-12.9-sm120-v2' in args
print(json.dumps(response))
"""


def _artifact(artifact_id: int, run_id: int, **changes: object) -> dict:
    return {
        "id": artifact_id,
        "name": NAME,
        "expired": False,
        "created_at": f"2026-10-06T00:{artifact_id:02}:00Z",
        "workflow_run": {"id": run_id},
        **changes,
    }


def _run_metadata(**changes: object) -> dict:
    return {
        "workflow_id": 77,
        "event": "push",
        "head_branch": "master",
        "status": "completed",
        "conclusion": "success",
        **changes,
    }


def _run(
    tmp_path: Path,
    *,
    pages: list[list[dict]],
    runs: dict[int, dict],
    failure: str | None = None,
) -> tuple[subprocess.CompletedProcess, list[list]]:
    executable = tmp_path / "gh"
    executable.write_text(MOCK_GH.format(python=sys.executable))
    executable.chmod(0o755)
    responses = {
        f"{PREFIX}/workflows/ci.yml": {"id": 77},
        f"{PREFIX}/artifacts": [{"artifacts": entries} for entries in pages],
        **{f"{PREFIX}/runs/{run_id}": run for run_id, run in runs.items()},
    }
    if failure is not None:
        responses.pop(f"{PREFIX}/{failure}")
    calls = tmp_path / "calls"
    result = subprocess.run(
        ["bash", "-e", "-o", "pipefail", "-c", SCRIPT],
        env={
            **os.environ,
            "PATH": str(tmp_path) + os.pathsep + os.environ["PATH"],
            "GITHUB_REPOSITORY": "owner/repo",
            "MOCK_RESPONSES": json.dumps(responses),
            "MOCK_CALLS": str(calls),
        },
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
    )
    return result, [json.loads(line) for line in calls.read_text().splitlines()]


def _deleted(calls: list[list]) -> list[int]:
    return [
        int(endpoint.rsplit("/", 1)[1])
        for method, endpoint, _ in calls
        if method == "DELETE"
    ]


def test_cleanup_only_runs_after_successful_master_compile_and_is_non_gating() -> None:
    assert "needs: cuda-compile" in JOB
    assert (
        "if: github.event_name == 'push' && github.ref == 'refs/heads/master' "
        "&& needs.cuda-compile.result == 'success'" in JOB
    )
    assert "continue-on-error: true" in JOB
    assert "actions: write" in JOB
    assert not re.search(r"uses: actions/checkout", JOB)


@pytest.mark.parametrize(
    "untrusted",
    [
        {"workflow_id": 99},
        {"event": "pull_request"},
        {"event": "workflow_dispatch"},
        {"head_branch": "feature"},
        {"conclusion": "failure"},
        {"conclusion": "cancelled"},
        {"status": "in_progress", "conclusion": None},
        {"status": "in_progress"},
    ],
)
def test_newer_untrusted_snapshots_cannot_evict_trusted_pair(
    tmp_path: Path, untrusted: dict
) -> None:
    result, calls = _run(
        tmp_path,
        pages=[
            [
                _artifact(5, 95),
                _artifact(4, 94),
                _artifact(3, 93),
                _artifact(2, 92),
                _artifact(1, 91),
            ]
        ],
        runs={
            95: _run_metadata(**untrusted),
            94: _run_metadata(**untrusted),
            **{run_id: _run_metadata() for run_id in (91, 92, 93)},
        },
    )
    assert result.returncode == 0, result.stderr
    assert _deleted(calls) == [1]


@pytest.mark.parametrize("count", [0, 1, 2])
def test_preserves_all_when_fewer_than_three_trusted_snapshots_exist(
    tmp_path: Path, count: int
) -> None:
    result, calls = _run(
        tmp_path,
        pages=[[_artifact(index, index) for index in range(1, count + 1)]],
        runs={index: _run_metadata() for index in range(1, count + 1)},
    )
    assert result.returncode == 0, result.stderr
    assert _deleted(calls) == []


def test_paginates_and_sorts_artifacts_by_creation_not_run_identity(
    tmp_path: Path,
) -> None:
    result, calls = _run(
        tmp_path,
        pages=[
            [_artifact(1, 99), _artifact(4, 88)],
            [_artifact(2, 77), _artifact(3, 88)],
        ],
        runs={index: _run_metadata() for index in (77, 88, 99)},
    )
    assert result.returncode == 0, result.stderr
    assert _deleted(calls) == [2, 1]
    # Multiple artifacts from a run require only one provenance lookup.
    assert sum(endpoint == f"{PREFIX}/runs/88" for _, endpoint, _ in calls) == 1
    first_delete = next(
        index for index, call in enumerate(calls) if call[0] == "DELETE"
    )
    assert all(call[0] == "GET" for call in calls[:first_delete])
    assert all(call[0] == "DELETE" for call in calls[first_delete:])


def test_expired_unrelated_and_unattributed_artifacts_do_not_count(
    tmp_path: Path,
) -> None:
    result, calls = _run(
        tmp_path,
        pages=[
            [
                _artifact(7, 97, workflow_run=None),
                _artifact(6, 96, workflow_run={}),
                _artifact(5, 95, name="unrelated"),
                _artifact(4, 94, expired=True),
                _artifact(3, 93),
                _artifact(2, 92),
                _artifact(1, 91),
            ]
        ],
        runs={run_id: _run_metadata() for run_id in (91, 92, 93)},
    )
    assert result.returncode == 0, result.stderr
    assert _deleted(calls) == [1]
    assert {endpoint for _, endpoint, _ in calls if "/runs/" in endpoint} == {
        f"{PREFIX}/runs/{run_id}" for run_id in (91, 92, 93)
    }


@pytest.mark.parametrize("failure", ["workflows/ci.yml", "artifacts", "runs/94"])
def test_discovery_failure_never_deletes_any_artifact(
    tmp_path: Path, failure: str
) -> None:
    result, calls = _run(
        tmp_path,
        pages=[[_artifact(index, 90 + index) for index in range(1, 5)]],
        runs={run_id: _run_metadata() for run_id in range(91, 95)},
        failure=failure,
    )
    assert result.returncode != 0
    assert _deleted(calls) == []
