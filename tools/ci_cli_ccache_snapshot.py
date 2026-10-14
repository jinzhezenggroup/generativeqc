"""Discover and atomically activate an optional trusted CUDA CLI compiler cache."""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
from pathlib import Path
from typing import Any

REPOSITORY = "jinzhezenggroup/generativeqc"
WORKFLOW = ".github/workflows/ci.yml"
ARTIFACT = "cuda-ccache-master-12.9-sm120-v3"
# The producer caps ccache at 5G; leave room for ZIP metadata.
MAX_BYTES = 6 * 1024**3


def api(endpoint: str, **fields: str) -> dict[str, Any]:
    command = ["gh", "api", "--method", "GET", endpoint]
    for key, value in fields.items():
        command.extend(["-f", f"{key}={value}"])
    result = subprocess.run(
        command, check=True, capture_output=True, text=True, timeout=5
    )
    payload = json.loads(result.stdout)
    if not isinstance(payload, dict):
        raise TypeError("unexpected API payload")
    return payload


def trusted_run(run: dict[str, Any], repository_id: int) -> bool:
    """Require complete successful-master provenance, not merely a run name."""
    return (
        type(run.get("id")) is int
        and run["id"] > 0
        and run.get("path") == WORKFLOW
        and run.get("event") == "push"
        and run.get("head_branch") == "master"
        and run.get("status") == "completed"
        and run.get("conclusion") == "success"
        and bool(re.fullmatch(r"[0-9a-f]{40}", str(run.get("head_sha", ""))))
        and all(
            isinstance(run.get(key), dict)
            and run[key].get("full_name") == REPOSITORY
            and run[key].get("id") == repository_id
            for key in ("repository", "head_repository")
        )
    )


def trusted_artifact(
    artifact: dict[str, Any], run: dict[str, Any], repository_id: int
) -> bool:
    """Bind the immutable artifact to the selected run and its source commit."""
    source = artifact.get("workflow_run") or {}
    return (
        type(artifact.get("id")) is int
        and artifact["id"] > 0
        and artifact.get("name") == ARTIFACT
        and artifact.get("expired") is False
        and type(artifact.get("size_in_bytes")) is int
        and 0 < artifact["size_in_bytes"] <= MAX_BYTES
        and bool(re.fullmatch(r"sha256:[0-9a-f]{64}", str(artifact.get("digest", ""))))
        and isinstance(source, dict)
        and source.get("id") == run["id"]
        and source.get("head_sha") == run["head_sha"]
        and source.get("head_branch") == "master"
        and source.get("repository_id") == repository_id
        and source.get("head_repository_id") == repository_id
    )


def stage_path() -> Path:
    """Return only this attempt's isolated path outside the compiler cache."""
    # A sibling of the absent canonical cache guarantees the same filesystem,
    # even when runner temporary storage is a separate mount.
    root = Path.home() / ".cache"
    identity = f"{os.environ['GITHUB_RUN_ID']}-{os.environ['GITHUB_RUN_ATTEMPT']}"
    if not root.is_absolute() or not re.fullmatch(r"[1-9][0-9]*-[1-9][0-9]*", identity):
        raise ValueError("invalid snapshot staging identity")
    return root.resolve() / f"trusted-cli-ccache-{identity}"


def discover() -> None:
    """Select at most one snapshot within bounded successful-master history."""
    event = os.environ.get("GITHUB_EVENT_NAME")
    if os.environ.get("GITHUB_REPOSITORY") != REPOSITORY or not (
        event in ("pull_request", "merge_group")
        or (event == "push" and os.environ.get("GITHUB_REF") == "refs/heads/master")
    ):
        return
    target = Path.home() / ".cache" / "ccache"
    if target.exists() or target.is_symlink():
        raise ValueError("compiler-cache destination already exists")
    repository_id = int(os.environ["GITHUB_REPOSITORY_ID"])
    if repository_id <= 0:
        raise ValueError("invalid repository identity")
    prefix = f"/repos/{REPOSITORY}/actions"
    runs = api(
        f"{prefix}/workflows/ci.yml/runs",
        branch="master",
        event="push",
        status="success",
        per_page="20",
    )
    for run in runs.get("workflow_runs", [])[:20]:
        if not trusted_run(run, repository_id):
            continue
        try:
            artifacts = api(
                f"{prefix}/runs/{run['id']}/artifacts", name=ARTIFACT, per_page="10"
            )
        except (OSError, ValueError, subprocess.SubprocessError):
            print("::warning::Unable to inspect trusted CLI snapshot; trying next run")
            continue
        for artifact in artifacts.get("artifacts", [])[:10]:
            if trusted_artifact(artifact, run, repository_id):
                stage = stage_path()
                if stage.exists() or stage.is_symlink():
                    raise ValueError("snapshot stage already exists")
                with open(os.environ["GITHUB_OUTPUT"], "a") as output:
                    output.write(f"run_id={run['id']}\nartifact_id={artifact['id']}\n")
                    output.write(f"stage={stage}\n")
                print(f"Trusted CLI cache: run {run['id']}, source {run['head_sha']}")
                return
    print("::notice::No trusted CLI snapshot; using actions/cache or cold compilation")


def owned_stage() -> Path:
    stage = stage_path()
    if os.environ.get("CLI_CACHE_STAGE") != str(stage) or stage.is_symlink():
        raise ValueError("refusing unexpected snapshot staging path")
    return stage


def activate() -> None:
    """Publish only a completed digest-verified download, without copying bytes."""
    stage = owned_stage()
    target = Path.home() / ".cache" / "ccache"
    if not stage.is_dir() or target.exists() or target.is_symlink():
        raise ValueError(
            "snapshot missing or compiler-cache destination already exists"
        )
    target.parent.mkdir(parents=True, exist_ok=True)
    # os.rename is atomic on the same filesystem and fails on cross-device
    # staging. Never fall back to a copy that could expose partial entries.
    stage.rename(target)


def cleanup() -> None:
    """Discard only our unactivated staging directory, never a selected cache."""
    stage = owned_stage()
    if stage.exists():
        shutil.rmtree(stage)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("discover", "activate", "cleanup"))
    args = parser.parse_args()
    try:
        {"discover": discover, "activate": activate, "cleanup": cleanup}[args.command]()
    except (
        OSError,
        ValueError,
        KeyError,
        TypeError,
        subprocess.SubprocessError,
    ) as error:
        print(f"::warning::Optional CLI snapshot unavailable: {error}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
