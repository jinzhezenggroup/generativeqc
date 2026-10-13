"""Fail-open lookup, fail-closed qualification, and bounded transport controls."""

from __future__ import annotations

import copy
import io
import json
import os
import re
import subprocess
import sys
import time
import urllib.error
from pathlib import Path
from typing import Any
from unittest.mock import Mock

import pytest

from tools import ci_required_failure_guard as guard

ROOT = Path(__file__).resolve().parents[2]
IDENTITY = guard.RunIdentity("example/project", 12345, 2, "a" * 40, "CI")
REQUIRED = {
    "Merge queue liveness",
    "cpu (gcc)",
    "cpu (clang)",
    "CUDA 12.9 compile (sm_120)",
    "CUDA 12.9 release resources (sm_120)",
    "CUDA 12.9 direct J/K AOT resources (sm_120)",
    "python (core-a)",
    "python (core-b)",
    "python (core-c)",
    "python (uccsd-forces)",
    "python (runtime-heavy)",
    "python (posthf)",
    "python (compiler-heavy)",
    "python (ecp-forces)",
}


def _payload() -> dict[str, Any]:
    jobs = [
        {
            "name": name,
            "id": index,
            "run_id": IDENTITY.run_id,
            "run_attempt": IDENTITY.attempt,
            "head_sha": IDENTITY.head_sha,
            "run_url": IDENTITY.run_url,
            "workflow_name": "CI",
            "status": "completed",
            "conclusion": "success",
            "completed_at": "2026-10-12T22:57:28Z",
        }
        for index, name in enumerate(sorted(REQUIRED), 1)
    ]
    return {"jobs": jobs, "total_count": len(jobs)}


@pytest.fixture
def env(tmp_path: Path) -> dict[str, str]:
    event = tmp_path / "event.json"
    event.write_text(
        json.dumps(
            {
                "repository": {"full_name": IDENTITY.repository},
                "pull_request": {"head": {"sha": IDENTITY.head_sha}},
                "merge_group": {"head_sha": IDENTITY.head_sha},
            }
        )
    )
    return {
        "GITHUB_EVENT_NAME": "pull_request",
        "GITHUB_EVENT_PATH": str(event),
        "GITHUB_API_URL": guard.API_ORIGIN,
        "GITHUB_REPOSITORY": IDENTITY.repository,
        "GITHUB_RUN_ID": str(IDENTITY.run_id),
        "GITHUB_RUN_ATTEMPT": str(IDENTITY.attempt),
        "GITHUB_SHA": "b" * 40,  # Synthetic checkout identity must not be used.
        "GITHUB_WORKFLOW": "CI",
        "GH_TOKEN": "test-only-read-token",
    }


def _run(
    monkeypatch: pytest.MonkeyPatch, env: dict[str, str], payload: dict[str, Any]
) -> int:
    monkeypatch.setattr(guard, "lookup_jobs", lambda identity, token: payload)
    return guard.main(env)


@pytest.mark.parametrize("event", ["pull_request", "merge_group"])
def test_original_event_source_head_is_used_on_reruns(
    env: dict[str, str], event: str
) -> None:
    env["GITHUB_EVENT_NAME"] = event
    assert guard.event_identity(env) == IDENTITY


@pytest.mark.parametrize("name", sorted(REQUIRED))
@pytest.mark.parametrize(
    "conclusion",
    ["failure", "timed_out", "cancelled", "action_required", "startup_failure"],
)
def test_only_explicit_required_terminal_failure_stops_cli(
    monkeypatch: pytest.MonkeyPatch, env: dict[str, str], name: str, conclusion: str
) -> None:
    payload = _payload()
    next(job for job in payload["jobs"] if job["name"] == name)["conclusion"] = (
        conclusion
    )
    assert _run(monkeypatch, env, payload) == 1


@pytest.mark.parametrize("conclusion", ["success", "neutral", "skipped", "stale"])
def test_nonfailure_terminal_states_preserve_qualification(
    monkeypatch: pytest.MonkeyPatch, env: dict[str, str], conclusion: str
) -> None:
    payload = _payload()
    for job in payload["jobs"]:
        job["conclusion"] = conclusion
    assert _run(monkeypatch, env, payload) == 0


@pytest.mark.parametrize(
    "status", ["queued", "in_progress", "waiting", "pending", "requested"]
)
def test_active_siblings_preserve_parallelism_without_polling(
    monkeypatch: pytest.MonkeyPatch, env: dict[str, str], status: str
) -> None:
    payload = _payload()
    payload["jobs"][0].update(status=status, conclusion=None, completed_at=None)
    fetch = Mock(return_value=payload)
    monkeypatch.setattr(guard, "lookup_jobs", fetch)
    assert guard.main(env) == 0
    fetch.assert_called_once_with(IDENTITY, env["GH_TOKEN"])


@pytest.mark.parametrize(
    "name",
    [
        "CPU performance (CodSpeed)",
        "upload-coverage (cpp-core)",
        "Seed shared Linux CPU ccache (GCC, coverage, Clang, Python)",
        "Pass CI checks",
        "unknown advisory report",
        "Native C++ CUDA CLI package (sm_120)",
        "Qualify native C++ CUDA CLI package (sm_120)",
    ],
)
def test_optional_failures_never_stop_cli(
    monkeypatch: pytest.MonkeyPatch, env: dict[str, str], name: str
) -> None:
    payload = _payload()
    optional = {**payload["jobs"][0], "name": name, "id": 100, "conclusion": "failure"}
    payload["jobs"].append(optional)
    payload["total_count"] += 1
    assert _run(monkeypatch, env, payload) == 0


@pytest.mark.parametrize(
    "field,value",
    [
        ("run_id", 54321),
        ("run_id", True),
        ("run_attempt", 1),
        ("run_attempt", "2"),
        ("head_sha", "b" * 40),
        ("workflow_name", "Other CI"),
        ("run_url", IDENTITY.run_url + "0"),
        ("id", True),
        ("name", ""),
        ("status", "unknown"),
        ("conclusion", "unknown"),
        ("conclusion", None),
        ("completed_at", None),
        ("completed_at", True),
        ("completed_at", "2026-10-12"),
        ("completed_at", "not a date"),
    ],
)
def test_bad_job_after_failure_invalidates_whole_page(
    monkeypatch: pytest.MonkeyPatch, env: dict[str, str], field: str, value: object
) -> None:
    payload = _payload()
    payload["jobs"][0]["conclusion"] = "failure"
    payload["jobs"][-1][field] = value
    assert _run(monkeypatch, env, payload) == 0


@pytest.mark.parametrize("field", list(_payload()["jobs"][0]))
def test_missing_job_fields_fail_open(
    monkeypatch: pytest.MonkeyPatch, env: dict[str, str], field: str
) -> None:
    payload = _payload()
    payload["jobs"][0]["conclusion"] = "failure"
    del payload["jobs"][-1][field]
    assert _run(monkeypatch, env, payload) == 0


@pytest.mark.parametrize(
    "change",
    [
        "missing_jobs",
        "missing_count",
        "wrong_count",
        "bool_count",
        "empty",
        "duplicate",
        "duplicate_id",
        "duplicate_name",
        "not_job",
        "not_jobs",
        "full_page",
        "partial_rerun",
    ],
)
def test_incomplete_or_ambiguous_snapshots_fail_open(
    monkeypatch: pytest.MonkeyPatch, env: dict[str, str], change: str
) -> None:
    payload = _payload()
    payload["jobs"][0]["conclusion"] = "failure"
    if change == "missing_jobs":
        del payload["jobs"]
    elif change == "missing_count":
        del payload["total_count"]
    elif change == "wrong_count":
        payload["total_count"] += 1
    elif change == "bool_count":
        payload["total_count"] = True
    elif change == "empty":
        payload = {"jobs": [], "total_count": 0}
    elif change == "duplicate":
        payload["jobs"].append(copy.deepcopy(payload["jobs"][0]))
        payload["total_count"] += 1
    elif change in {"duplicate_id", "duplicate_name"}:
        field = change.removeprefix("duplicate_")
        payload["jobs"][-1][field] = payload["jobs"][0][field]
    elif change == "not_job":
        payload["jobs"].append(None)
        payload["total_count"] += 1
    elif change == "not_jobs":
        payload["jobs"] = {}
    elif change == "full_page":
        payload["jobs"] *= 10
        payload["jobs"] = payload["jobs"][:100]
        payload["total_count"] = 100
    else:
        payload["jobs"].pop()
        payload["total_count"] -= 1
    assert _run(monkeypatch, env, payload) == 0


@pytest.mark.parametrize(
    "field,value",
    [
        ("GITHUB_API_URL", "https://untrusted.example"),
        ("GITHUB_API_URL", "https://api.github.com@untrusted.example"),
        ("GITHUB_API_URL", "http://api.github.com"),
        ("GITHUB_RUN_ID", "0"),
        ("GITHUB_RUN_ATTEMPT", "0"),
        ("GITHUB_RUN_ID", "../1"),
        ("GITHUB_RUN_ATTEMPT", "unknown"),
        ("GITHUB_REPOSITORY", "another/project"),
        ("GITHUB_WORKFLOW", ""),
    ],
)
def test_invalid_context_never_requests_api(
    monkeypatch: pytest.MonkeyPatch, env: dict[str, str], field: str, value: object
) -> None:
    env[field] = value
    fetch = Mock(side_effect=AssertionError("must not contact API"))
    monkeypatch.setattr(guard, "lookup_jobs", fetch)
    assert guard.main(env) == 0
    fetch.assert_not_called()


@pytest.mark.parametrize(
    "raw",
    [
        b"null",
        b"{",
        b"{}",
        b'{"repository": {}, "repository": {}}',
        b"x" * (guard.MAX_RESPONSE_BYTES + 1),
    ],
    ids=[
        "null-json",
        "truncated-json",
        "missing-fields",
        "duplicate-fields",
        "oversized-event",
    ],
)
def test_malformed_event_does_not_request_api(
    monkeypatch: pytest.MonkeyPatch, env: dict[str, str], raw: bytes
) -> None:
    Path(env["GITHUB_EVENT_PATH"]).write_bytes(raw)
    fetch = Mock()
    monkeypatch.setattr(guard, "lookup_jobs", fetch)
    assert guard.main(env) == 0
    fetch.assert_not_called()


@pytest.mark.parametrize("token", ["", "token with space", "token\r\ninjected"])
def test_invalid_token_is_not_transmitted(
    monkeypatch: pytest.MonkeyPatch, env: dict[str, str], token: str
) -> None:
    env["GH_TOKEN"] = token
    opener = _transport(monkeypatch, Response())
    assert guard.main(env) == 0
    opener.open.assert_not_called()


@pytest.mark.parametrize("failure", [False, True])
def test_guard_process_exit_prevents_new_build_on_required_failure(
    tmp_path: Path, env: dict[str, str], failure: bool
) -> None:
    payload = _payload()
    if failure:
        payload["jobs"][0]["conclusion"] = "failure"
    fixture = tmp_path / "jobs.json"
    fixture.write_text(json.dumps(payload))
    script = tmp_path / "guard_fixture.py"
    script.write_text(
        "import json, os\n"
        "from unittest.mock import patch\n"
        "from pathlib import Path\n"
        "from tools import ci_required_failure_guard as guard\n"
        "payload = json.loads(Path(os.environ['JOBS_FIXTURE']).read_text())\n"
        "with patch.object(guard, 'lookup_jobs', return_value=payload):\n"
        "    raise SystemExit(guard.main())\n"
    )
    marker = tmp_path / "build-started"
    result = subprocess.run(
        [
            "bash",
            "-e",
            "-o",
            "pipefail",
            "-c",
            '"$PYTHON" "$SCRIPT"; touch "$BUILD_MARKER"',
        ],
        env={
            **os.environ,
            **env,
            "PYTHON": sys.executable,
            "SCRIPT": str(script),
            "JOBS_FIXTURE": str(fixture),
            "BUILD_MARKER": str(marker),
            "PYTHONPATH": str(ROOT),
        },
        capture_output=True,
        text=True,
        check=False,
        timeout=5,
    )
    assert result.returncode == (1 if failure else 0)
    assert marker.exists() is not failure
    if failure:
        assert "CLI remains unqualified" in result.stdout


@pytest.mark.parametrize(
    "event", ["push", "schedule", "workflow_dispatch", "pull_request_target", "unknown"]
)
def test_non_pr_events_do_not_even_read_event_or_token(
    monkeypatch: pytest.MonkeyPatch, event: str
) -> None:
    fetch = Mock()
    monkeypatch.setattr(guard, "lookup_jobs", fetch)
    assert guard.main({"GITHUB_EVENT_NAME": event}) == 0
    fetch.assert_not_called()


class Response(io.BytesIO):
    def __init__(
        self,
        raw: bytes | None = None,
        *,
        status: int = 200,
        headers: dict[str, str] | None = None,
        url: str = IDENTITY.jobs_url,
    ) -> None:
        super().__init__(json.dumps(_payload()).encode() if raw is None else raw)
        self.status, self.headers, self.url = status, headers or {}, url

    def geturl(self) -> str:
        return self.url


def _transport(monkeypatch: pytest.MonkeyPatch, response: Response) -> Mock:
    monkeypatch.setattr(guard, "lookup_jobs", guard.fetch_jobs)
    opener = Mock()
    opener.open.return_value = response
    monkeypatch.setattr(guard.urllib.request, "build_opener", lambda *handlers: opener)
    return opener


def test_transport_uses_exact_attempt_one_get_and_bounded_timeout(
    monkeypatch: pytest.MonkeyPatch, env: dict[str, str]
) -> None:
    opener = _transport(monkeypatch, Response())
    assert guard.main(env) == 0
    opener.open.assert_called_once()
    args, kwargs = opener.open.call_args
    assert args[0].get_method() == "GET"
    assert (
        args[0].full_url
        == "https://api.github.com/repos/example/project/actions/runs/12345/attempts/2/jobs?per_page=100"
    )
    assert args[0].get_header("Authorization") == "Bearer test-only-read-token"
    assert 0 < kwargs["timeout"] <= 5


@pytest.mark.parametrize(
    "kwargs",
    [
        {"status": 403},
        {"url": "https://untrusted.example/jobs"},
        {"headers": {"Link": '<https://api.github.com/next>; rel="next"'}},
        {"headers": {"Link": "malformed"}},
        {"raw": b'{"jobs": []'},
        {"raw": b'{"total_count": 0, "total_count": 1, "jobs": []}'},
        {"raw": b"x" * (guard.MAX_RESPONSE_BYTES + 1)},
        {"raw": b"null"},
    ],
    ids=[
        "http-error",
        "unexpected-url",
        "pagination-link",
        "malformed-link",
        "truncated-json",
        "duplicate-fields",
        "oversized-response",
        "null-json",
    ],
)
def test_transport_uncertainty_warns_and_runs_full_qualification(
    monkeypatch: pytest.MonkeyPatch,
    env: dict[str, str],
    capsys: pytest.CaptureFixture[str],
    kwargs: dict[str, Any],
) -> None:
    _transport(monkeypatch, Response(**kwargs))
    assert guard.main(env) == 0
    assert "::warning::" in capsys.readouterr().out


@pytest.mark.parametrize("exception", [TimeoutError, OSError, urllib.error.URLError])
def test_api_errors_do_not_expose_tokens_or_fail_cli(
    monkeypatch: pytest.MonkeyPatch,
    env: dict[str, str],
    capsys: pytest.CaptureFixture[str],
    exception: type[Exception],
) -> None:
    opener = _transport(monkeypatch, Response())
    opener.open.side_effect = exception(env["GH_TOKEN"])
    assert guard.main(env) == 0
    output = capsys.readouterr().out
    assert "::warning::" in output and env["GH_TOKEN"] not in output


@pytest.mark.parametrize(
    "destination", ["https://untrusted.example/jobs", IDENTITY.jobs_url]
)
def test_redirects_cannot_forward_authorization(destination: str) -> None:
    request = guard.urllib.request.Request(
        IDENTITY.jobs_url, headers={"Authorization": "Bearer fixture"}
    )
    with pytest.raises(ValueError, match="redirected"):
        guard._NoRedirect().redirect_request(
            request, None, 302, "Found", {}, destination
        )


@pytest.mark.skipif(os.name != "posix", reason="Linux CI native-block deadline")
def test_deadline_kills_and_reaps_native_blocked_lookup(
    monkeypatch: pytest.MonkeyPatch, env: dict[str, str], tmp_path: Path
) -> None:
    worker = tmp_path / "blocked.py"
    pid_file = tmp_path / "worker.pid"
    worker.write_text(
        "import ctypes, os\nfrom pathlib import Path\n"
        f"Path({str(pid_file)!r}).write_text(str(os.getpid()))\n"
        "ctypes.CDLL(None).sleep(10)\n"
    )
    monkeypatch.setattr(guard, "__file__", str(worker))
    monkeypatch.setattr(guard, "LOOKUP_SECONDS", 0.5)
    sentinel = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(10)"])
    try:
        start = time.monotonic()
        assert guard.main(env) == 0
        assert time.monotonic() - start < 3
        assert pid_file.exists(), "worker must enter the native blocking fixture"
        with pytest.raises(ProcessLookupError):
            os.kill(int(pid_file.read_text()), 0)
        assert sentinel.poll() is None, "unrelated processes must survive the deadline"
    finally:
        sentinel.terminate()
        sentinel.wait(timeout=5)


@pytest.mark.parametrize(
    "output,code",
    [(b"null", 0), (b"malformed", 0), (b"", 2)],
    ids=["null-json", "malformed-json", "worker-error"],
)
def test_worker_error_or_bad_output_runs_qualification(
    monkeypatch: pytest.MonkeyPatch,
    env: dict[str, str],
    tmp_path: Path,
    output: bytes,
    code: int,
) -> None:
    worker = tmp_path / "response.py"
    worker.write_text(
        f"import sys\nsys.stdout.buffer.write({output!r})\nsys.exit({code})\n"
    )
    monkeypatch.setattr(guard, "__file__", str(worker))
    assert guard.main(env) == 0


def test_supervisor_keeps_token_out_of_arguments_and_bounds_total_time(
    monkeypatch: pytest.MonkeyPatch, env: dict[str, str]
) -> None:
    command = Mock(
        return_value=subprocess.CompletedProcess(
            [], 0, json.dumps(_payload()).encode(), b""
        )
    )
    monkeypatch.setattr(guard.subprocess, "run", command)
    assert guard.main(env) == 0
    command.assert_called_once()
    args, kwargs = command.call_args
    assert env["GH_TOKEN"] not in repr(args)
    assert env["GH_TOKEN"].encode() not in kwargs["input"]
    assert kwargs["env"]["GH_TOKEN"] == env["GH_TOKEN"]
    assert 0 < kwargs["timeout"] <= 5
    assert kwargs["check"] is True
    assert kwargs["stderr"] == subprocess.DEVNULL
    assert json.loads(kwargs["input"])["attempt"] == IDENTITY.attempt


def _job(name: str) -> str:
    workflow = (ROOT / ".github/workflows/ci.yml").read_text()
    body = workflow.split(f"\n  {name}:\n", 1)[1]
    return re.split(r"\n  [a-zA-Z0-9_-]+:\n", body, maxsplit=1)[0]


def test_guard_precedes_seed_without_changing_dependencies_or_permissions() -> None:
    job = _job("native-cli-artifact")
    steps = job.split("      - name: ")
    index = next(
        i
        for i, step in enumerate(steps)
        if "run: python3 tools/ci_required_failure_guard.py" in step
    )
    assert steps[index + 1].startswith(
        "Seed fast CUDA CI objects in the shared cache\n"
    )
    assert "continue-on-error" not in steps[index]
    assert "GH_TOKEN: ${{ github.token }}" in steps[index]
    assert "needs: cuda-compile\n" in job
    assert "permissions:\n      contents: read\n      actions: read\n" in job


@pytest.mark.parametrize(
    "event", ["pull_request", "merge_group", "push", "schedule", "workflow_dispatch"]
)
@pytest.mark.parametrize("tier", ["skip", "fast", "release"])
def test_workflow_guard_scope(event: str, tier: str) -> None:
    step = next(
        part
        for part in _job("native-cli-artifact").split("      - name: ")
        if "run: python3 tools/ci_required_failure_guard.py" in part
    )
    condition = re.search(r"^        if: (.+)$", step, re.MULTILINE).group(1)
    condition = condition.replace("github.event_name", '"$EVENT"').replace(
        "steps.cli_scope.outputs.tier", '"$TIER"'
    )
    result = subprocess.run(
        ["bash", "-c", f"[[ {condition} ]]"],
        env={**os.environ, "EVENT": event, "TIER": tier},
        check=False,
        timeout=5,
    )
    assert (result.returncode == 0) == (
        event in {"pull_request", "merge_group"} and tier != "skip"
    )


def test_required_sibling_inventory_matches_aggregate_workflow() -> None:
    aggregate = _job("pass")
    needed = {
        name.strip()
        for name in re.search(r"needs: \[([^\]]+)\]", aggregate).group(1).split(",")
    }
    assert needed == {
        "merge_queue_liveness",
        "cpu",
        "cuda-compile",
        "cuda-resources",
        "cuda-direct-aot",
        "python",
        "native-cli-artifact",
        "native-cli-qualification",
    }
    names = set()
    for name in needed - {"native-cli-artifact", "native-cli-qualification"}:
        body = _job(name)
        if name == "cpu":
            compilers = re.search(r"compiler: \[([^\]]+)\]", body).group(1).split(",")
            names.update(f"cpu ({compiler.strip()})" for compiler in compilers)
        elif name == "python":
            shards = re.search(r"shard: \[([^\]]+)\]", body).group(1).split(",")
            names.update(f"python ({shard.strip()})" for shard in shards)
        else:
            names.add(re.search(r"name: (.+)", body).group(1))
    assert names == REQUIRED == guard.REQUIRED_SIBLINGS
    assert "jobs: ${{ toJSON(needs) }}" in aggregate
    assert "allowed-failures" not in aggregate
    assert (
        "continue-on-error"
        not in _job("native-cli-artifact")[
            : _job("native-cli-artifact").index("    steps:")
        ]
    )


def test_partial_rerun_cannot_turn_unqualified_cli_green(
    monkeypatch: pytest.MonkeyPatch, env: dict[str, str]
) -> None:
    payload = _payload()
    payload["jobs"][0]["conclusion"] = "failure"
    cli_status = "failure" if _run(monkeypatch, env, payload) else "success"
    assert cli_status == "failure"
    # The real aggregate includes CLI and permits no failures. The selected
    # PR/queue branch cannot skip (exercised in test_native_cli_workflow.py).
    assert "native-cli-qualification" in re.search(
        r"needs: \[([^\]]+)\]", _job("pass")
    ).group(1)
    repaired_python_status = "success"
    assert {cli_status, repaired_python_status} != {"success"}
    # If CLI itself reruns, an exact-attempt partial list simply builds normally.
    partial = {"jobs": payload["jobs"][:1], "total_count": 1}
    assert _run(monkeypatch, env, partial) == 0


@pytest.mark.parametrize(
    "selected", ["native-cli-artifact", "native-cli-qualification"]
)
def test_cli_branch_records_do_not_hide_a_required_sibling_failure(
    monkeypatch: pytest.MonkeyPatch, env: dict[str, str], selected: str
) -> None:
    payload = _payload()
    for index, key in enumerate(
        ("native-cli-artifact", "native-cli-qualification"), 100
    ):
        name = re.search(r"^    name: (.+)$", _job(key), re.MULTILINE)[1]
        payload["jobs"].append(
            {
                **payload["jobs"][0],
                "name": name,
                "id": index,
                "status": "in_progress" if key == selected else "completed",
                "conclusion": None if key == selected else "skipped",
                "completed_at": None if key == selected else "2026-10-12T22:57:28Z",
            }
        )
    payload["total_count"] = len(payload["jobs"])
    assert _run(monkeypatch, env, payload) == 0
    payload["jobs"][0]["conclusion"] = "failure"
    assert _run(monkeypatch, env, payload) == 1
