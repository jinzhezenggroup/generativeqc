"""Bounded trusted-master CPU cache persistence; never replace primary data/config."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import re
import shutil
import stat
import subprocess
import tarfile
import tempfile
from pathlib import Path
from typing import Any, BinaryIO

ARTIFACT = "cpu-ccache-master-ubuntu24-x86_64-v1"
WORKFLOW = ".github/workflows/ci.yml"
MAX_ARTIFACT = 512 * 1024 * 1024
# Allow space for the manifest and upload-artifact's uncompressed ZIP wrapper.
MAX_ARCHIVE = MAX_ARTIFACT - 1024 * 1024
MAX_DECODED = 2 * 1024 * 1024 * 1024
MAX_FILES = 100_000
SHA = re.compile(r"[0-9a-f]{40}")
# Import cache data only: no config, temp files, locks, stats or logs.
CACHE_FILE = re.compile(r"(?:[0-9a-f]/){1,8}[0-9A-Za-z]{16,128}")


def output(**values: object) -> None:
    with Path(os.environ["GITHUB_OUTPUT"]).open("a") as handle:
        handle.writelines(f"{key}={value}\n" for key, value in values.items())


def identity() -> dict[str, str]:
    return {
        "format": ARTIFACT,
        "repository": os.environ["GITHUB_REPOSITORY"],
        "workflow": WORKFLOW,
        "event": "push",
        "ref": "refs/heads/master",
        "image_os": "ubuntu24",
        "arch": "X64",
        "ccache_version": os.environ["CCACHE_VERSION"],
    }


def check_platform() -> None:
    if (
        os.environ.get("ImageOS") != "ubuntu24"
        or os.environ.get("RUNNER_ARCH") != "X64"
    ):
        raise ValueError("unsupported CPU snapshot runner")


def positive(value: object) -> bool:
    return type(value) is int and value > 0


def trusted_run(run: dict[str, Any], repository: str) -> bool:
    return (
        run.get("path") == WORKFLOW
        and run.get("event") == "push"
        and run.get("head_branch") == "master"
        and run.get("status") == "completed"
        and run.get("conclusion") == "success"
        and run.get("repository", {}).get("full_name") == repository
        and run.get("head_repository", {}).get("full_name") == repository
        and positive(run.get("repository", {}).get("id"))
        and run.get("head_repository", {}).get("id") == run["repository"]["id"]
        and positive(run.get("id"))
        and positive(run.get("run_attempt"))
        and SHA.fullmatch(str(run.get("head_sha", ""))) is not None
    )


def trusted_artifact(artifact: dict[str, Any], run: dict[str, Any]) -> bool:
    origin = artifact.get("workflow_run", {})
    return (
        artifact.get("name") == f"{ARTIFACT}-{run['run_attempt']}"
        and artifact.get("expired") is False
        and positive(artifact.get("id"))
        and positive(artifact.get("size_in_bytes"))
        and artifact["size_in_bytes"] <= MAX_ARTIFACT
        and re.fullmatch(r"sha256:[0-9a-f]{64}", str(artifact.get("digest", "")))
        is not None
        and origin.get("id") == run["id"]
        and origin.get("head_branch") == "master"
        and origin.get("head_sha") == run["head_sha"]
        and origin.get("repository_id") == run["repository"]["id"]
        and origin.get("head_repository_id") == run["repository"]["id"]
    )


def api(endpoint: str, **fields: str) -> dict[str, Any]:
    # No pagination/retry loop; an overall discovery deadline additionally bounds
    # the initial request and at most three candidate artifact requests.
    command = ["gh", "api", "--method", "GET", endpoint]
    for key, value in fields.items():
        command.extend(["-f", f"{key}={value}"])
    result = subprocess.run(command, check=True, capture_output=True, timeout=8)
    if len(result.stdout) > 2 * 1024 * 1024:
        raise ValueError("oversized discovery response")
    return json.loads(result.stdout)


def discover() -> None:
    check_platform()
    target = Path.home() / ".cache" / "ccache"
    if target.exists() or target.is_symlink():
        print("::notice::CPU fallback skipped: the normal cache already exists.")
        return
    repository = os.environ["GITHUB_REPOSITORY"]
    runs = api(
        f"/repos/{repository}/actions/workflows/ci.yml/runs",
        branch="master",
        event="push",
        status="success",
        per_page="20",
    )
    for run in [r for r in runs.get("workflow_runs", []) if trusted_run(r, repository)][
        :3
    ]:
        try:
            artifacts = api(
                f"/repos/{repository}/actions/runs/{run['id']}/artifacts",
                name=f"{ARTIFACT}-{run['run_attempt']}",
                per_page="10",
            )
        except (OSError, subprocess.SubprocessError, ValueError):
            continue
        matches = [
            a for a in artifacts.get("artifacts", []) if trusted_artifact(a, run)
        ]
        if len(matches) != 1:
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        stage = Path(
            tempfile.mkdtemp(
                prefix=f"gqc-cpu-{os.environ['GITHUB_RUN_ID']}-{os.environ['GITHUB_RUN_ATTEMPT']}-",
                dir=target.parent,
            )
        )
        expected = identity() | {
            "run_id": str(run["id"]),
            "run_attempt": str(run["run_attempt"]),
            "source_sha": run["head_sha"],
        }
        (stage / "expected.json").write_text(json.dumps(expected))
        output(stage=stage, artifact_id=matches[0]["id"], run_id=run["id"])
        return
    print(
        "::notice::No bounded successful-master CPU snapshot; using the normal cold path."
    )


def digest(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


class LimitedWriter:
    """Stop while compressing instead of checking after filling disk."""

    def __init__(self, handle: BinaryIO) -> None:
        self.handle = handle
        self.size = 0

    def write(self, data: bytes) -> int:
        self.size += len(data)
        if self.size > MAX_ARCHIVE:
            raise ValueError(
                "CPU snapshot exceeds encoded size cap; cache left unchanged"
            )
        return self.handle.write(data)

    def flush(self) -> None:
        self.handle.flush()


def package() -> None:
    check_platform()
    if (
        os.environ.get("GITHUB_EVENT_NAME") != "push"
        or os.environ.get("GITHUB_REF") != "refs/heads/master"
    ):
        raise ValueError("only master push jobs may package CPU snapshots")
    source = Path.home() / ".cache" / "ccache"
    files: list[tuple[Path, str]] = []
    decoded = 0
    for directory, subdirs, names in os.walk(source, followlinks=False):
        subdirs[:] = [
            name for name in subdirs if not (Path(directory) / name).is_symlink()
        ]
        for name in sorted(names):
            path = Path(directory) / name
            relative = path.relative_to(source).as_posix()
            if not CACHE_FILE.fullmatch(relative):
                continue
            info = path.lstat()
            if not stat.S_ISREG(info.st_mode):
                raise ValueError("nonregular cache data")
            decoded += info.st_size
            files.append((path, relative))
            if decoded > MAX_DECODED or len(files) > MAX_FILES:
                raise ValueError("CPU snapshot exceeds decoded size/file cap")
    if not files:
        raise ValueError("no CPU compiler data to publish")
    stage = Path(
        tempfile.mkdtemp(prefix="gqc-cpu-publish-", dir=os.environ["RUNNER_TEMP"])
    )
    archive = stage / "cache.tar.gz"
    with (
        archive.open("wb") as raw,
        gzip.GzipFile(
            fileobj=LimitedWriter(raw), mode="wb", compresslevel=1, mtime=0
        ) as compressed,
        tarfile.open(fileobj=compressed, mode="w|", format=tarfile.USTAR_FORMAT) as tar,
    ):
        for path, relative in files:
            info = tar.gettarinfo(path, arcname=relative)
            info.mtime = 0
            info.uid = info.gid = 0
            info.uname = info.gname = ""
            with path.open("rb") as data:
                tar.addfile(info, data)
    manifest = identity() | {
        "run_id": os.environ["GITHUB_RUN_ID"],
        "run_attempt": os.environ["GITHUB_RUN_ATTEMPT"],
        "source_sha": os.environ["GITHUB_SHA"],
        "sha256": digest(archive),
        "archive_bytes": archive.stat().st_size,
        "decoded_bytes": decoded,
        "files": len(files),
    }
    (stage / "manifest.json").write_text(json.dumps(manifest))
    output(path=stage)
    print(
        f"CPU snapshot: {archive.stat().st_size} encoded bytes, {decoded} cache bytes, {len(files)} files"
    )


def owned_stage() -> Path:
    stage = Path(os.environ["CPU_CACHE_STAGE"])
    prefix = (
        f"gqc-cpu-{os.environ['GITHUB_RUN_ID']}-{os.environ['GITHUB_RUN_ATTEMPT']}-"
    )
    if (
        stage.parent != Path.home() / ".cache"
        or not stage.name.startswith(prefix)
        or stage.is_symlink()
        or not stage.is_dir()
    ):
        raise ValueError("CPU snapshot staging is not owned by this run/attempt")
    return stage


def verify() -> None:
    stage = owned_stage()
    download = stage / "download"
    manifest_path, archive = download / "manifest.json", download / "cache.tar.gz"
    if (
        manifest_path.is_symlink()
        or archive.is_symlink()
        or manifest_path.stat().st_size > 16 * 1024
        or not 0 < archive.stat().st_size <= MAX_ARCHIVE
    ):
        raise ValueError("invalid CPU snapshot payload size/type")
    manifest = json.loads(manifest_path.read_text())
    expected = json.loads((stage / "expected.json").read_text())
    if any(manifest.get(key) != value for key, value in expected.items()):
        raise ValueError("CPU snapshot provenance/toolchain mismatch")
    if manifest.get("archive_bytes") != archive.stat().st_size or manifest.get(
        "sha256"
    ) != digest(archive):
        raise ValueError("CPU snapshot checksum/size mismatch")
    cache = stage / "cache"
    cache.mkdir()  # Never reuse an interrupted extraction.
    decoded = count = 0
    # Parse only ordinary USTAR headers ourselves: tarfile's generic reader
    # expands PAX/longname metadata before yielding members, bypassing a member
    # size check. No links, extensions or special files belong in this format.
    with gzip.open(archive, "rb") as stream:
        while True:
            header = stream.read(512)
            if len(header) != 512:
                raise ValueError("truncated CPU snapshot header")
            if header == bytes(512):
                # Require the second tar end block, bounded zero padding, gzip
                # checksum/footer and EOF; reject hidden appended data as well.
                tail = stream.read(10241)
                if not 512 <= len(tail) <= 10240 or any(tail):
                    raise ValueError("invalid CPU snapshot footer")
                break
            member = tarfile.TarInfo.frombuf(header, "utf-8", "strict")
            if not member.isfile() or not CACHE_FILE.fullmatch(member.name):
                raise ValueError("unsafe CPU snapshot member")
            decoded += member.size
            count += 1
            if member.size < 0 or decoded > MAX_DECODED or count > MAX_FILES:
                raise ValueError("CPU snapshot exceeds decoded size/file cap")
            destination = cache / member.name
            destination.parent.mkdir(parents=True, exist_ok=True)
            with destination.open("xb") as handle:
                remaining = member.size
                while remaining:
                    chunk = stream.read(min(1024 * 1024, remaining))
                    if not chunk:
                        raise ValueError("truncated CPU snapshot member")
                    handle.write(chunk)
                    remaining -= len(chunk)
            padding = (-member.size) % 512
            if stream.read(padding) != bytes(padding):
                raise ValueError("invalid CPU snapshot padding")
    if (
        count == 0
        or manifest.get("files") != count
        or manifest.get("decoded_bytes") != decoded
    ):
        raise ValueError("CPU snapshot inventory mismatch")
    (stage / "verified").write_text("verified\n")


def activate() -> None:
    stage = owned_stage()
    target, cache = Path.home() / ".cache" / "ccache", stage / "cache"
    if (
        not (stage / "verified").is_file()
        or cache.is_symlink()
        or not cache.is_dir()
        or target.exists()
        or target.is_symlink()
    ):
        raise ValueError("CPU snapshot is incomplete or normal cache already exists")
    # Same-filesystem atomic activation: cancellation cannot expose partial data.
    cache.rename(target)
    print("::notice::Activated verified successful-master CPU compiler snapshot.")


def cleanup() -> None:
    shutil.rmtree(owned_stage())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command", choices=("discover", "package", "verify", "activate", "cleanup")
    )
    args = parser.parse_args()
    globals()[args.command]()


if __name__ == "__main__":
    main()
