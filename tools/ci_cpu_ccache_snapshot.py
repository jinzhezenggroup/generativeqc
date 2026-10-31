"""Bounded trusted-master compiler snapshots; never replace primary data/config."""

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
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, BinaryIO

import tomllib

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

PROFILE = "cpu"
LABEL = "CPU"
STAGE_ENV = "CPU_CACHE_STAGE"
CANONICAL_REPOSITORY = "jinzhezenggroup/generativeqc"
WHEEL_IMAGE = "quay.io/manylinux_cuda/manylinux_2_28_x86_64_cuda12_9:2026.06.30-1"


def select_profile(name: str) -> None:
    """Select a closed CLI profile; CPU remains the unchanged default."""
    global PROFILE, LABEL, STAGE_ENV, ARTIFACT, WORKFLOW
    global MAX_ARTIFACT, MAX_ARCHIVE, MAX_DECODED
    if name != "wheel":
        raise ValueError("unsupported snapshot profile")
    PROFILE, LABEL, STAGE_ENV = "wheel", "wheel", "WHEEL_CACHE_STAGE"
    ARTIFACT = "wheel-ccache-master-manylinux-cuda12.9-multiarch-v2"
    WORKFLOW = ".github/workflows/wheels.yml"
    MAX_ARTIFACT = 2 * 1024 * 1024 * 1024
    MAX_ARCHIVE = MAX_ARTIFACT - 1024 * 1024
    MAX_DECODED = 3 * 1024 * 1024 * 1024


def cache_root() -> Path:
    if PROFILE == "wheel":
        root = Path(os.environ["RUNNER_TEMP"])
        if not root.is_absolute():
            raise ValueError("wheel snapshot requires an absolute runner temp")
        return root / "ccache"
    return Path.home() / ".cache" / "ccache"


def stage_prefix() -> str:
    return f"gqc-{PROFILE}-{os.environ['GITHUB_RUN_ID']}-{os.environ['GITHUB_RUN_ATTEMPT']}-"


def wheel_toolchain() -> dict[str, str]:
    # Bind the supported container, not the unrelated Ubuntu CUDA cache. A
    # changed image requires explicit qualification of this optional profile.
    with (Path(__file__).resolve().parents[1] / "pyproject.toml").open("rb") as handle:
        image = tomllib.load(handle)["tool"]["cibuildwheel"]["manylinux-x86_64-image"]
    if image != WHEEL_IMAGE:
        raise ValueError("unsupported wheel snapshot image")
    return {
        "manylinux_image": image,
        "cuda_version": "12.9",
        "cuda_arch": "80-real;86-real;89-real;90-real;100-real;120",
        "ccache_version": "4.14",
    }


def record_wheel_toolchain() -> None:
    """Record observed container tools outside the cache, without affecting builds."""
    if PROFILE != "wheel":
        raise ValueError("toolchain recording is wheel-only")
    expected = wheel_toolchain()
    version = subprocess.check_output(["ccache", "--version"], text=True, timeout=5)
    cuda = subprocess.check_output(["nvcc", "--version"], text=True, timeout=5)
    if (
        version.splitlines()[0] != "ccache version 4.14"
        or not re.search(r"release 12\.9,", cuda)
        or os.environ.get("GENERATIVEQC_CUDA_ARCHITECTURES") != expected["cuda_arch"]
        or os.environ.get("GENERATIVEQC_ENABLE_CUDA") != "ON"
    ):
        raise ValueError("unsupported observed wheel toolchain")
    attempt = os.environ["GQC_WHEEL_SNAPSHOT_ID"]
    if not re.fullmatch(r"[1-9][0-9]*-[1-9][0-9]*", attempt):
        raise ValueError("invalid wheel snapshot attempt")
    receipt = Path(os.environ["CCACHE_DIR"]).with_name("ccache-wheel-toolchain.json")
    receipt.write_text(json.dumps(expected | {"attempt": attempt}))


def output(**values: object) -> None:
    with Path(os.environ["GITHUB_OUTPUT"]).open("a") as handle:
        handle.writelines(f"{key}={value}\n" for key, value in values.items())


def identity() -> dict[str, str]:
    result = {
        "format": ARTIFACT,
        "repository": os.environ["GITHUB_REPOSITORY"],
        "workflow": WORKFLOW,
        "event": "push",
        "ref": "refs/heads/master",
        "image_os": "ubuntu24",
        "arch": "X64",
        "ccache_version": os.environ["CCACHE_VERSION"] if PROFILE == "cpu" else "4.14",
    }
    if PROFILE == "wheel":
        result.update(wheel_toolchain())
    return result


def check_platform() -> None:
    if (
        os.environ.get("ImageOS") != "ubuntu24"
        or os.environ.get("RUNNER_ARCH") != "X64"
    ):
        raise ValueError(f"unsupported {LABEL} snapshot runner")
    if (
        PROFILE == "wheel"
        and os.environ.get("GITHUB_REPOSITORY") != CANONICAL_REPOSITORY
    ):
        raise ValueError("wheel snapshots require the canonical repository")


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


def find_snapshot() -> tuple[dict[str, Any], dict[str, Any]] | None:
    repository = os.environ["GITHUB_REPOSITORY"]
    runs = api(
        f"/repos/{repository}/actions/workflows/{Path(WORKFLOW).name}/runs",
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
        if len(matches) == 1:
            return run, matches[0]
    return None


def discover() -> None:
    check_platform()
    target = cache_root()
    if target.exists() or target.is_symlink():
        print(f"::notice::{LABEL} fallback skipped: the normal cache already exists.")
        return
    found = find_snapshot()
    if found is not None:
        run, artifact = found
        target.parent.mkdir(parents=True, exist_ok=True)
        stage = Path(tempfile.mkdtemp(prefix=stage_prefix(), dir=target.parent))
        expected = identity() | {
            "run_id": str(run["id"]),
            "run_attempt": str(run["run_attempt"]),
            "source_sha": run["head_sha"],
        }
        (stage / "expected.json").write_text(json.dumps(expected))
        output(stage=stage, artifact_id=artifact["id"], run_id=run["id"])
        return
    print(
        f"::notice::No bounded successful-master {LABEL} snapshot; using the normal cold path."
    )


def refresh() -> None:
    """Apply the existing three-hour/input-aware refresh policy to wheel inputs."""
    check_platform()
    if PROFILE != "wheel":
        raise ValueError("refresh is wheel-only")
    found = find_snapshot()
    publish = True
    if found is not None:
        run, artifact = found
        try:
            created = datetime.fromisoformat(
                artifact["created_at"].replace("Z", "+00:00")
            )
            age = (datetime.now(timezone.utc) - created).total_seconds()
            if 0 <= age < 10800:
                # Include all ordinary wheel key inputs plus container setup and
                # toolchain configuration. Missing old source requests refresh.
                result = subprocess.run(
                    [
                        "git",
                        "diff",
                        "--quiet",
                        "--no-ext-diff",
                        "--no-textconv",
                        run["head_sha"],
                        os.environ["GITHUB_SHA"],
                        "--",
                        ".github",
                        "cmake",
                        "python/ci",
                        "python/generativeqc_compiler",
                        "python/generativeqc",
                        "CMakeLists.txt",
                        "CMakePresets.json",
                        "include",
                        "src",
                        "tools",
                        "manifests",
                        "data",
                        "upstream",
                        "pyproject.toml",
                        "uv.lock",
                    ],
                    timeout=8,
                    check=False,
                    capture_output=True,
                )
                publish = result.returncode != 0
        except (KeyError, TypeError, ValueError, OSError, subprocess.SubprocessError):
            pass
    output(publish=str(publish).lower())


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
                f"{LABEL} snapshot exceeds encoded size cap; cache left unchanged"
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
        raise ValueError(f"only master push jobs may package {LABEL} snapshots")
    source = cache_root()
    if PROFILE == "wheel":
        receipt = source.with_name("ccache-wheel-toolchain.json")
        expected = wheel_toolchain() | {
            "attempt": f"{os.environ['GITHUB_RUN_ID']}-{os.environ['GITHUB_RUN_ATTEMPT']}"
        }
        if (
            receipt.is_symlink()
            or receipt.stat().st_size > 4096
            or json.loads(receipt.read_text()) != expected
        ):
            raise ValueError("wheel snapshot lacks the observed toolchain receipt")
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
                raise ValueError(f"{LABEL} snapshot exceeds decoded size/file cap")
    if not files:
        raise ValueError(f"no {LABEL} compiler data to publish")
    stage = Path(
        tempfile.mkdtemp(
            prefix=f"gqc-{PROFILE}-publish-", dir=os.environ["RUNNER_TEMP"]
        )
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
        f"{LABEL} snapshot: {archive.stat().st_size} encoded bytes, {decoded} cache bytes, {len(files)} files"
    )


def owned_stage() -> Path:
    stage = Path(os.environ[STAGE_ENV])
    prefix = stage_prefix()
    if (
        stage.parent != cache_root().parent
        or not stage.name.startswith(prefix)
        or stage.is_symlink()
        or not stage.is_dir()
    ):
        raise ValueError(f"{LABEL} snapshot staging is not owned by this run/attempt")
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
        raise ValueError(f"invalid {LABEL} snapshot payload size/type")
    manifest = json.loads(manifest_path.read_text())
    expected = json.loads((stage / "expected.json").read_text())
    if any(manifest.get(key) != value for key, value in expected.items()):
        raise ValueError(f"{LABEL} snapshot provenance/toolchain mismatch")
    if manifest.get("archive_bytes") != archive.stat().st_size or manifest.get(
        "sha256"
    ) != digest(archive):
        raise ValueError(f"{LABEL} snapshot checksum/size mismatch")
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
                raise ValueError(f"truncated {LABEL} snapshot header")
            if header == bytes(512):
                # Require the second tar end block, bounded zero padding, gzip
                # checksum/footer and EOF; reject hidden appended data as well.
                tail = stream.read(10241)
                if not 512 <= len(tail) <= 10240 or any(tail):
                    raise ValueError(f"invalid {LABEL} snapshot footer")
                break
            member = tarfile.TarInfo.frombuf(header, "utf-8", "strict")
            if not member.isfile() or not CACHE_FILE.fullmatch(member.name):
                raise ValueError(f"unsafe {LABEL} snapshot member")
            decoded += member.size
            count += 1
            if member.size < 0 or decoded > MAX_DECODED or count > MAX_FILES:
                raise ValueError(f"{LABEL} snapshot exceeds decoded size/file cap")
            destination = cache / member.name
            destination.parent.mkdir(parents=True, exist_ok=True)
            with destination.open("xb") as handle:
                remaining = member.size
                while remaining:
                    chunk = stream.read(min(1024 * 1024, remaining))
                    if not chunk:
                        raise ValueError(f"truncated {LABEL} snapshot member")
                    handle.write(chunk)
                    remaining -= len(chunk)
            padding = (-member.size) % 512
            if stream.read(padding) != bytes(padding):
                raise ValueError(f"invalid {LABEL} snapshot padding")
    if (
        count == 0
        or manifest.get("files") != count
        or manifest.get("decoded_bytes") != decoded
    ):
        raise ValueError(f"{LABEL} snapshot inventory mismatch")
    (stage / "verified").write_text("verified\n")


def activate() -> None:
    stage = owned_stage()
    target, cache = cache_root(), stage / "cache"
    if (
        not (stage / "verified").is_file()
        or cache.is_symlink()
        or not cache.is_dir()
        or target.exists()
        or target.is_symlink()
    ):
        raise ValueError(
            f"{LABEL} snapshot is incomplete or normal cache already exists"
        )
    # Same-filesystem atomic activation: cancellation cannot expose partial data.
    cache.rename(target)
    print(f"::notice::Activated verified successful-master {LABEL} compiler snapshot.")


def cleanup() -> None:
    shutil.rmtree(owned_stage())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command",
        choices=(
            "discover",
            "package",
            "verify",
            "activate",
            "cleanup",
            "refresh",
            "record_wheel_toolchain",
        ),
    )
    parser.add_argument("--profile", choices=("cpu", "wheel"), default="cpu")
    args = parser.parse_args()
    if args.profile != "cpu":
        select_profile(args.profile)
    globals()[args.command]()


if __name__ == "__main__":
    main()
