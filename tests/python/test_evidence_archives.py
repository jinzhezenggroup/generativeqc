"""Archive integrity and historical recovery use small, independent fixtures."""

import json
import subprocess
import typing
from hashlib import sha256
from pathlib import Path
from zipfile import ZipFile

import pytest

from tools import restore_retained_evidence as history
from tools.unpack_evidence import unpack

PAYLOADS = {
    "nested/record.json": b'{"values": [0.12345678901234567, 1e-14]}\r\n',
    "arrays.bin": bytes(range(256)) + b"\x00\xff",
    "failure.log": b"deliberate failure retained\n",
}


def sample_archive(
    directory: Path,
    name: str = "record.json",
    *,
    payloads: dict[str, bytes] | None = None,
    schema: str = "generativeqc.evidence-archive.v1",
) -> dict:
    members = payloads if payloads is not None else {name: b'{"passed": true}\n'}
    with ZipFile(directory / "raw-evidence.zip", "w") as archive:
        for path, data in members.items():
            archive.writestr(path, data)
    manifest = {
        "schema": schema,
        "archive_sha256": sha256(
            (directory / "raw-evidence.zip").read_bytes()
        ).hexdigest(),
        "files": [
            {"path": path, "bytes": len(data), "sha256": sha256(data).hexdigest()}
            for path, data in members.items()
        ],
    }
    (directory / "raw-evidence.manifest.json").write_text(json.dumps(manifest))
    return manifest


@pytest.mark.parametrize("prefix", ["generativeqc", "vibeqc"])
def test_archive_restores_text_binary_and_refuses_overwrite(
    tmp_path: Path, prefix: str
) -> None:
    sample_archive(tmp_path, payloads=PAYLOADS, schema=f"{prefix}.evidence-archive.v1")
    output = tmp_path / "restored"
    assert unpack(tmp_path) == len(PAYLOADS)
    assert not output.exists()
    assert unpack(tmp_path, output) == len(PAYLOADS)
    assert {
        path.relative_to(output).as_posix(): path.read_bytes()
        for path in output.rglob("*")
        if path.is_file()
    } == PAYLOADS
    assert json.loads((output / "nested/record.json").read_bytes())["values"] == [
        0.12345678901234567,
        1e-14,
    ]
    with pytest.raises(FileExistsError):
        unpack(tmp_path, output)
    assert (output / "failure.log").read_bytes() == PAYLOADS["failure.log"]


def git(root: Path, *args: str) -> str:
    return subprocess.check_output(
        [
            "git",
            "-c",
            "core.autocrlf=false",
            "-c",
            "user.name=Evidence test",
            "-c",
            "user.email=evidence@example.invalid",
            *args,
        ],
        cwd=root,
        stderr=subprocess.DEVNULL,
        text=True,
    ).strip()


@pytest.fixture
def historical_archive(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[Path, Path, dict, bytes]:
    root = tmp_path / "repo"
    root.mkdir()
    git(root, "init", "--object-format=sha1")
    directory = root / "benchmarks/results/old"
    directory.mkdir(parents=True)
    sample_archive(directory, payloads=PAYLOADS)
    path = directory / "raw-evidence.zip"
    data = path.read_bytes()
    git(root, "add", ".")
    git(root, "commit", "--no-gpg-sign", "-m", "Synthetic historical archive")
    audit = {
        "schema": "generativeqc.git-snapshot.v1",
        "source_revision": git(root, "rev-parse", "HEAD"),
        "file_count": 1,
        "total_bytes": len(data),
        "files": [
            {
                "path": path.relative_to(root).as_posix(),
                "bytes": len(data),
                "sha256": sha256(data).hexdigest(),
            }
        ],
    }
    manifest = tmp_path / "snapshot.manifest.json"
    manifest.write_text(json.dumps(audit))
    path.unlink()
    monkeypatch.setattr(history, "ROOT", root)
    return root, manifest, audit, data


def test_historical_zip_restores_then_unpacks_with_retained_member_manifest(
    historical_archive: tuple[Path, Path, dict, bytes], tmp_path: Path
) -> None:
    root, manifest, audit, original = historical_archive
    path = audit["files"][0]["path"]
    restored = history.restore(path, tmp_path / "recovered.zip", manifest=manifest)
    assert restored.read_bytes() == original
    assert not (root / path).exists()
    output = tmp_path / "members"
    assert unpack((root / path).parent, output, restored) == len(PAYLOADS)
    for name, data in PAYLOADS.items():
        assert (output / name).read_bytes() == data
    with pytest.raises(FileExistsError):
        history.restore(path, restored, manifest=manifest)


@pytest.mark.parametrize("checkout", ["source-archive", "shallow-clone"])
def test_missing_optional_history_fails_offline_without_partial_output(
    historical_archive: tuple[Path, Path, dict, bytes],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    checkout: str,
) -> None:
    root, manifest, audit, _ = historical_archive
    current = tmp_path / checkout
    if checkout == "shallow-clone":
        # A real depth-one local clone lacks the prior archive commit/blob.
        git(root, "add", "-u")
        git(root, "commit", "--no-gpg-sign", "-m", "Retire historical ZIP")
        git(
            tmp_path,
            "-c",
            "protocol.file.allow=always",
            "clone",
            "--depth=1",
            root.as_uri(),
            str(current),
        )
        assert git(current, "rev-parse", "--is-shallow-repository") == "true"
    else:
        current.mkdir()
    monkeypatch.setattr(history, "ROOT", current)
    original_run = subprocess.run
    calls = []

    def offline_read(command: list[str], **kwargs: typing.Any) -> typing.Any:
        calls.append(command)
        assert command[:3] == ["git", "cat-file", "blob"]
        assert kwargs["env"]["GIT_NO_LAZY_FETCH"] == "1"
        assert kwargs["env"]["GIT_ALLOW_PROTOCOL"] == ""
        assert kwargs["env"]["GIT_TERMINAL_PROMPT"] == "0"
        return original_run(command, **kwargs)

    monkeypatch.setattr(history.subprocess, "run", offline_read)
    output = tmp_path / "missing-history"
    with pytest.raises(ValueError, match="historical object unavailable; fetch"):
        history.restore_snapshot(output, manifest=manifest)
    assert len(calls) == 1
    assert calls[0][-1] == (f"{audit['source_revision']}:{audit['files'][0]['path']}")
    assert not output.exists()


@pytest.mark.parametrize("damage", ["archive", "member", "inventory"])
def test_corruption_fails_before_creating_output(tmp_path: Path, damage: str) -> None:
    manifest = sample_archive(tmp_path)
    if damage == "archive":
        with (tmp_path / "raw-evidence.zip").open("ab") as stream:
            stream.write(b"corrupted")
    elif damage == "member":
        manifest["files"][0]["sha256"] = "0" * 64
    else:
        manifest["files"] = []
    (tmp_path / "raw-evidence.manifest.json").write_text(json.dumps(manifest))
    output = tmp_path / "restored"
    with pytest.raises(ValueError):
        unpack(tmp_path, output)
    assert not output.exists()


@pytest.mark.parametrize("name", ["../escape", "/absolute", "C:/escape", "a\\b"])
def test_unsafe_paths_are_rejected(tmp_path: Path, name: str) -> None:
    sample_archive(tmp_path, name)
    # Windows' ZIP writer can normalize backslashes before our inventory check.
    with pytest.raises(ValueError, match="unsafe archive path|members differ"):
        unpack(tmp_path, tmp_path / "restored")
    assert not (tmp_path / "restored").exists()


def test_migrated_unpack_recipes_name_the_recovered_archive() -> None:
    """Every retained recipe must use the ignored ZIP rather than a retired path."""
    results = Path(__file__).resolve().parents[2] / "benchmarks/results"
    snapshot = json.loads(
        (results / "retention-reports-20261003/snapshot.manifest.json").read_bytes()
    )
    for record in snapshot["files"]:
        if not record["path"].endswith("/raw-evidence.zip"):
            continue
        directory = Path(record["path"]).parent
        readme = results.parent.parent / directory / "README.md"
        commands = readme.read_text().replace("\\\n", " ").splitlines()
        unpack = [
            line
            for line in commands
            if "python -m tools.unpack_evidence " + directory.as_posix() in line
        ]
        assert unpack
        assert all("--archive .artifacts/" in line for line in unpack)
