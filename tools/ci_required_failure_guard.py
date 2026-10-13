"""Avoid new CLI compilation only after a verified current-attempt CI failure.

This is a one-shot, read-only optimization for PR/merge-group CI, not a success
shortcut. Uncertain API/event data runs normal qualification; a verified required
failure returns nonzero so the unqualified CLI must itself be rerun later.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import urllib.request
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Mapping
    from typing import NoReturn

API_ORIGIN = "https://api.github.com"
LOOKUP_SECONDS = 5.0
MAX_RESPONSE_BYTES = 2 * 1024 * 1024
REQUIRED_SIBLINGS = frozenset(
    {
        "Merge queue liveness",
        "cpu (gcc)",
        "cpu (clang)",
        "CUDA 12.9 compile (sm_120)",
        "CUDA 12.9 release resources (sm_120)",
        "CUDA 12.9 direct J/K AOT resources (sm_120)",
        *(
            f"python ({shard})"
            for shard in (
                "core-a",
                "core-b",
                "core-c",
                "uccsd-forces",
                "runtime-heavy",
                "posthf",
                "compiler-heavy",
                "ecp-forces",
            )
        ),
    }
)
ACTIVE_STATUSES = frozenset(
    {"queued", "in_progress", "waiting", "pending", "requested"}
)
FAILURES = frozenset(
    {"failure", "timed_out", "cancelled", "action_required", "startup_failure"}
)
CONCLUSIONS = FAILURES | {"success", "neutral", "skipped", "stale"}


@dataclass(frozen=True)
class RunIdentity:
    repository: str
    run_id: int
    attempt: int
    head_sha: str
    workflow: str

    @property
    def run_url(self) -> str:
        return f"{API_ORIGIN}/repos/{self.repository}/actions/runs/{self.run_id}"

    @property
    def jobs_url(self) -> str:
        return f"{self.run_url}/attempts/{self.attempt}/jobs?per_page=100"


def _object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    """Reject ambiguous duplicate JSON keys rather than choosing one identity."""
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON field")
        result[key] = value
    return result


def _positive(value: object) -> bool:
    return type(value) is int and value > 0


def event_identity(env: Mapping[str, str]) -> RunIdentity:
    if env["GITHUB_API_URL"] != API_ORIGIN:
        raise ValueError("unsupported API origin")
    repository = env["GITHUB_REPOSITORY"]
    if not re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", repository):
        raise ValueError("invalid repository")
    with Path(env["GITHUB_EVENT_PATH"]).open("rb") as stream:
        raw = stream.read(MAX_RESPONSE_BYTES + 1)
    if len(raw) > MAX_RESPONSE_BYTES:
        raise ValueError("oversize event")
    event = json.loads(raw, object_pairs_hook=_object)
    if event["repository"]["full_name"] != repository:
        raise ValueError("event repository mismatch")
    # The PR event retains the original source head on reruns. GITHUB_SHA is a
    # synthetic merge SHA; fetching the live PR could instead see a newer head.
    if env["GITHUB_EVENT_NAME"] == "pull_request":
        head = event["pull_request"]["head"]["sha"]
    elif env["GITHUB_EVENT_NAME"] == "merge_group":
        head = event["merge_group"]["head_sha"]
    else:
        raise ValueError("unsupported event")
    if not isinstance(head, str) or not re.fullmatch(r"[0-9a-f]{40}", head):
        raise ValueError("invalid source head")
    run_id, attempt = env["GITHUB_RUN_ID"], env["GITHUB_RUN_ATTEMPT"]
    if not all(re.fullmatch(r"[1-9][0-9]*", value) for value in (run_id, attempt)):
        raise ValueError("invalid run or attempt")
    workflow = env["GITHUB_WORKFLOW"]
    if not workflow:
        raise ValueError("missing workflow identity")
    return RunIdentity(repository, int(run_id), int(attempt), head, workflow)


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(
        self,
        req: urllib.request.Request,
        fp: object,
        code: int,
        msg: str,
        headers: object,
        newurl: str,
    ) -> NoReturn:
        # Never forward the step-scoped token, even to another GitHub URL.
        raise ValueError("redirected jobs response")


def fetch_jobs(identity: RunIdentity, token: str) -> object:
    if not token or any(character.isspace() for character in token):
        raise ValueError("missing or invalid read token")
    request = urllib.request.Request(  # noqa: S310 -- fixed HTTPS API origin
        identity.jobs_url,
        headers={
            "Accept": "application/vnd.github+json",
            "Authorization": f"Bearer {token}",
            "X-GitHub-Api-Version": "2022-11-28",
        },
    )
    with urllib.request.build_opener(_NoRedirect()).open(
        request, timeout=LOOKUP_SECONDS
    ) as response:
        if response.status != 200 or response.geturl() != identity.jobs_url:
            raise ValueError("unexpected jobs response")
        # One page only. Even a surprising/malformed Link header is uncertainty.
        if response.headers.get("Link"):
            raise ValueError("paginated jobs response")
        raw = response.read(MAX_RESPONSE_BYTES + 1)
        if len(raw) > MAX_RESPONSE_BYTES:
            raise ValueError("oversize jobs response")
        return json.loads(raw, object_pairs_hook=_object)


def lookup_jobs(identity: RunIdentity, token: str) -> object:
    # A process deadline also bounds libc DNS resolution, where a Python signal
    # handler/socket timeout alone cannot guarantee interruption. run() kills and
    # reaps this one child on timeout; no shared process or workflow is cancelled.
    result = subprocess.run(
        [sys.executable, "-I", str(Path(__file__).resolve()), "--fetch-jobs"],
        input=json.dumps(asdict(identity)).encode(),
        env={**os.environ, "GH_TOKEN": token},
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        timeout=LOOKUP_SECONDS,
        check=True,
    )
    if len(result.stdout) > MAX_RESPONSE_BYTES:
        raise ValueError("oversize worker response")
    return json.loads(result.stdout, object_pairs_hook=_object)


def _fetch_main() -> int:
    try:
        raw = sys.stdin.buffer.read(MAX_RESPONSE_BYTES + 1)
        if len(raw) > MAX_RESPONSE_BYTES:
            raise ValueError("oversize worker input")
        identity = RunIdentity(**json.loads(raw, object_pairs_hook=_object))
        payload = fetch_jobs(identity, os.environ["GH_TOKEN"])
        encoded = json.dumps(payload).encode()
        if len(encoded) > MAX_RESPONSE_BYTES:
            raise ValueError("oversize worker response")
        sys.stdout.buffer.write(encoded)
        return 0
    except Exception:  # noqa: BLE001 -- parent warns and runs full CI
        # No traceback/HTTP error text: those may include authentication details.
        return 2


def required_failures(payload: object, identity: RunIdentity) -> list[str]:
    if not isinstance(payload, dict):
        raise TypeError("invalid jobs payload")
    jobs, count = payload["jobs"], payload["total_count"]
    if not isinstance(jobs, list) or type(count) is not int or count != len(jobs):
        raise ValueError("incomplete jobs page")
    if not 0 < count < 100:
        raise ValueError("empty or potentially truncated jobs page")
    names, ids, failed = set(), set(), []
    # Validate the entire page before acting on any failure. Optional jobs cannot
    # stop qualification, but malformed/foreign records invalidate the snapshot.
    for job in jobs:
        if not isinstance(job, dict):
            raise TypeError("invalid job")
        name, job_id = job["name"], job["id"]
        if not isinstance(name, str) or not name or name in names:
            raise ValueError("missing or duplicate job name")
        if not _positive(job_id) or job_id in ids:
            raise ValueError("missing or duplicate job identity")
        names.add(name)
        ids.add(job_id)
        if (
            not _positive(job["run_id"])
            or job["run_id"] != identity.run_id
            or not _positive(job["run_attempt"])
            or job["run_attempt"] != identity.attempt
            or job["head_sha"] != identity.head_sha
            or job["run_url"] != identity.run_url
            or job["workflow_name"] != identity.workflow
        ):
            raise ValueError("job source identity mismatch")
        status, conclusion = job["status"], job["conclusion"]
        if status == "completed":
            completed = job["completed_at"]
            if (
                conclusion not in CONCLUSIONS
                or not isinstance(completed, str)
                or datetime.fromisoformat(completed.replace("Z", "+00:00")).tzinfo
                is None
            ):
                raise ValueError("unknown terminal job state")
            if name in REQUIRED_SIBLINGS and conclusion in FAILURES:
                failed.append(name)
        elif (
            status not in ACTIVE_STATUSES
            or conclusion is not None
            or job["completed_at"] is not None
        ):
            raise ValueError("unknown active job state")
    if not REQUIRED_SIBLINGS <= names:
        # Partial reruns may reuse earlier successes not listed in this attempt.
        # Do not query an older attempt or treat absence as failure/success.
        raise ValueError("required siblings absent from this attempt")
    return sorted(failed)


def main(env: Mapping[str, str] | None = None) -> int:
    env = os.environ if env is None else env
    if env.get("GITHUB_EVENT_NAME") not in {"pull_request", "merge_group"}:
        print(
            "Required-failure guard is inactive for this event; run CLI qualification."
        )
        return 0
    try:
        identity = event_identity(env)
        failed = required_failures(lookup_jobs(identity, env["GH_TOKEN"]), identity)
    except Exception:  # noqa: BLE001 -- every uncertainty must run full CI
        # API errors, worker failures and malformed data all run
        # normal qualification. Do not log exception text, headers or tokens.
        print(
            "::warning::Cannot verify current-attempt required jobs; run full CLI qualification."
        )
        return 0
    if failed:
        print(
            "::error::Required CI already failed in this attempt: " + ", ".join(failed)
        )
        print("CLI remains unqualified. Rerun it with the repaired required jobs.")
        return 1
    print("No verified required failure; run full CLI qualification without waiting.")
    return 0


if __name__ == "__main__":
    sys.exit(_fetch_main() if sys.argv[1:] == ["--fetch-jobs"] else main())
