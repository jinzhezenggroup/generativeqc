"""Automatic compaction respects staged ownership and preserves publication data."""

from __future__ import annotations

import gzip
import hashlib
import json
import subprocess
from typing import TYPE_CHECKING

import pytest
from generativeqc_compiler.common.evidence import new_evidence

from tools import compact_evidence_publications as compact
from tools.generativeqc_validation.publication import validate_publication

if TYPE_CHECKING:
    from pathlib import Path


THRESHOLD = 128 << 10
REVIEW = "benchmarks/legacy-evidence-review.json"
SCHEMA = "generativeqc.benchmark-publication.v1"


def _json(value: object) -> bytes:
    return (json.dumps(value, indent=2, ensure_ascii=False) + "\n").encode()


def _digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _git(root: Path, *arguments: str) -> bytes:
    return subprocess.check_output(["git", *arguments], cwd=root)


def _snapshot(root: Path) -> dict[str, bytes]:
    return {
        path.relative_to(root).as_posix(): path.read_bytes()
        for path in root.rglob("*")
        if path.is_file() and ".git" not in path.relative_to(root).parts
    }


def _index(root: Path) -> bytes:
    # Object IDs, modes and paths prove the hook did not stage its own changes.
    return _git(root, "ls-files", "--stage", "-z")


def _samples(size: int = THRESHOLD + 4096) -> bytes:
    prefix = (
        b'{"values":[-1.2345678901234567,-0.0,1e-300,9007199254740993],'
        b'"failed_cases":[{"passed":false,"error":1e-14}],"padding":"'
    )
    suffix = b'"}\n'
    return prefix + b"x" * (size - len(prefix) - len(suffix)) + suffix


def _bundle(
    root: Path,
    name: str = "campaign",
    *,
    schema: str = SCHEMA,
    sample_size: int = THRESHOLD + 4096,
    large_evidence: bool = False,
    large_non_evidence: bool = False,
) -> Path:
    directory = root / "benchmarks/results" / name
    directory.mkdir(parents=True)
    samples = _samples(sample_size)
    evidence = new_evidence(tier="cpu", subject="storage test", inputs_hash="1" * 64)
    evidence.update(
        revision="a" * 40,
        device={"kind": "cpu"},
        toolchain={"python": "fixture"},
        backend_selected="cpu",
        hashes={key: "b" * 64 for key in ("equation", "ir", "source", "schedule")},
        attachments=[
            {
                "path": "samples.json",
                "kind": "samples",
                "bytes": len(samples),
                "sha256": _digest(samples),
            }
        ],
    )
    if large_evidence:
        evidence["retained_explanation"] = "all original observations retained " * 5000
    members = [
        ("samples.json", "samples", samples),
        ("evidence.json", "evidence", _json(evidence)),
        ("README.md", "reproduction", b"Reproduce this retained storage fixture.\n"),
        ("nested/reproduce.py", "reproduction", b"print('retained reproduction')\n"),
    ]
    if large_non_evidence:
        members.extend(
            [
                ("large-readme.md", "summary", b"retain explanation\n" * 10_000),
                ("large-script.py", "reproduction", b"# retain source\n" * 10_000),
                ("custom.json", "input", _samples()),
                ("summary.json", "summary", _samples()),
            ]
        )
    files = []
    for member_name, role, data in members:
        path = directory / member_name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        files.append(
            {
                "path": member_name,
                "role": role,
                "bytes": len(data),
                "sha256": _digest(data),
            }
        )
    manifest = {
        "schema": schema,
        "source": {"revision": "a" * 40, "dirty": False},
        "reproduction": {"command": ["python", "nested/reproduce.py"]},
        "decision": {
            "status": "inconclusive",
            "scope": "numerical",
            "reason": "Storage-only test fixture; no scientific acceptance claim.",
        },
        "archives": [],
        "files": files,
    }
    manifest_path = directory / "publication.json"
    manifest_path.write_bytes(_json(manifest))
    if schema in {SCHEMA, "vibeqc.benchmark-publication.v1"}:
        validate_publication(manifest, {name: data for name, _role, data in members})
    _review(root)
    return manifest_path


def _review(root: Path) -> None:
    families = {}
    rows = []
    for path in sorted(root.glob("benchmarks/results/**/publication.json")):
        manifest = json.loads(path.read_bytes())
        family = path.relative_to(root / "benchmarks/results").parts[0]
        families[family] = {
            "owner": "storage fixture",
            "reason": "Preserve complete original observations.",
            "review_document": (path.parent / "README.md").relative_to(root).as_posix(),
        }
        for entry in manifest["files"]:
            if entry["bytes"] >= THRESHOLD:
                rows.append(
                    {
                        **entry,
                        "path": (path.parent / entry["path"])
                        .relative_to(root)
                        .as_posix(),
                        "family": family,
                        "classification": "required-compact-accepted-evidence",
                    }
                )
    (root / REVIEW).write_bytes(
        _json(
            {
                "schema": "generativeqc.legacy-large-evidence-review.v1",
                "threshold_bytes": THRESHOLD,
                "families": families,
                "files": rows,
            }
        )
    )


def _stage(root: Path) -> None:
    _git(root, "add", "--all", "--", ".")


def _automatic(root: Path, *paths: Path) -> None:
    compact.main(
        ["--pre-commit", *(path.relative_to(root).as_posix() for path in paths)]
    )


@pytest.fixture
def repository(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    root = tmp_path / "repository"
    root.mkdir()
    _git(root, "init", "--quiet")
    monkeypatch.setattr(compact, "ROOT", root)
    monkeypatch.setattr(compact, "REVIEW", root / REVIEW)
    return root


@pytest.mark.parametrize("schema", [SCHEMA, "vibeqc.benchmark-publication.v1"])
@pytest.mark.parametrize(
    "touched", ["samples.json", "publication.json", "README.md", "nested/reproduce.py"]
)
def test_hook_rewrites_selected_bundle_then_requires_explicit_restage(
    repository: Path, schema: str, touched: str
) -> None:
    manifest_path = _bundle(repository, schema=schema, large_evidence=True)
    directory = manifest_path.parent
    before = _snapshot(repository)
    old_manifest = json.loads(manifest_path.read_bytes())
    old_evidence = json.loads((directory / "evidence.json").read_bytes())
    _stage(repository)
    index = _index(repository)

    with pytest.raises(SystemExit, match="(?i)(re.?stage|stage the|git add)"):
        _automatic(repository, directory / touched)

    assert _index(repository) == index
    assert not (directory / "samples.json").exists()
    assert not (directory / "evidence.json").exists()
    packed_samples = (directory / "samples.json.gz").read_bytes()
    assert (
        gzip.decompress(packed_samples)
        == before[(directory / "samples.json").relative_to(repository).as_posix()]
    )
    evidence = json.loads(
        gzip.decompress((directory / "evidence.json.gz").read_bytes())
    )
    attachment = evidence["attachments"][0]
    assert attachment == {
        "path": "samples.json.gz",
        "kind": "samples",
        "bytes": len(packed_samples),
        "sha256": _digest(packed_samples),
    }
    evidence["attachments"] = old_evidence["attachments"]
    assert evidence == old_evidence
    manifest = json.loads(manifest_path.read_bytes())
    assert {key: value for key, value in manifest.items() if key != "files"} == {
        key: value for key, value in old_manifest.items() if key != "files"
    }
    files = {
        entry["path"]: (directory / entry["path"]).read_bytes()
        for entry in manifest["files"]
    }
    validate_publication(manifest, files)
    assert json.loads((repository / REVIEW).read_bytes())["files"] == []

    compacted = _snapshot(repository)
    with pytest.raises((ValueError, SystemExit), match="(?i)(index|stag|work.?tree)"):
        _automatic(repository, manifest_path)
    assert _snapshot(repository) == compacted
    assert _index(repository) == index

    _stage(repository)
    restaged_index = _index(repository)
    _automatic(repository, manifest_path)
    compact.main(["--check"])
    assert _snapshot(repository) == compacted
    assert _index(repository) == restaged_index


@pytest.mark.parametrize("size", [THRESHOLD - 1, THRESHOLD, THRESHOLD + 1])
def test_hook_uses_real_128_kib_boundary(repository: Path, size: int) -> None:
    manifest = _bundle(repository, sample_size=size)
    _stage(repository)
    before = _snapshot(repository)
    index = _index(repository)
    if size >= THRESHOLD:
        with pytest.raises(SystemExit, match="(?i)(re.?stage|stage the|git add)"):
            _automatic(repository, manifest)
        assert gzip.decompress(
            (manifest.parent / "samples.json.gz").read_bytes()
        ) == _samples(size)
        assert (manifest.parent / "evidence.json").is_file()
    else:
        _automatic(repository, manifest)
        assert _snapshot(repository) == before
    assert _index(repository) == index


def test_hook_does_not_compress_non_evidence_roles(repository: Path) -> None:
    manifest = _bundle(repository, large_non_evidence=True)
    _stage(repository)
    before = _snapshot(repository)
    with pytest.raises(SystemExit, match="(?i)(re.?stage|stage the|git add)"):
        _automatic(repository, manifest)
    for name in ("large-readme.md", "large-script.py", "custom.json", "summary.json"):
        path = manifest.parent / name
        assert path.read_bytes() == before[path.relative_to(repository).as_posix()]
        assert not path.with_name(path.name + ".gz").exists()


@pytest.mark.parametrize(
    "touched",
    [None, "README.md", "tools/reproduce.py", "benchmarks/results/custom/data.json"],
)
def test_unrelated_paths_and_empty_hook_invocations_are_noops(
    repository: Path, touched: str | None
) -> None:
    _bundle(repository)
    paths = []
    if touched is not None:
        path = repository / touched
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(_samples())
        paths.append(path)
    _stage(repository)
    before, index = _snapshot(repository), _index(repository)
    _automatic(repository, *paths)
    assert _snapshot(repository) == before
    assert _index(repository) == index


def test_only_selected_tracked_publication_is_compacted(repository: Path) -> None:
    selected = _bundle(repository, "selected")
    unrelated = _bundle(repository, "unrelated")
    _stage(repository)
    # An untracked transient bundle must not be discovered by filesystem globbing.
    review = (repository / REVIEW).read_bytes()
    untracked = _bundle(repository, "untracked")
    (repository / REVIEW).write_bytes(review)
    before = _snapshot(repository)
    index = _index(repository)
    with pytest.raises(SystemExit, match="(?i)(re.?stage|stage the|git add)"):
        _automatic(repository, selected)
    for manifest in (unrelated, untracked):
        for path in manifest.parent.rglob("*"):
            if path.is_file():
                assert (
                    path.read_bytes() == before[path.relative_to(repository).as_posix()]
                )
    assert _index(repository) == index


def test_touched_untracked_publication_is_not_automatically_adopted(
    repository: Path,
) -> None:
    manifest = _bundle(repository)
    before, index = _snapshot(repository), _index(repository)
    _automatic(repository, manifest)
    assert _snapshot(repository) == before
    assert _index(repository) == index


@pytest.mark.parametrize(
    "name", ["publication.json", "samples.json", "evidence.json", "README.md", REVIEW]
)
def test_partially_staged_bundle_or_review_is_rejected_without_writes(
    repository: Path, name: str
) -> None:
    manifest = _bundle(repository)
    _stage(repository)
    target = repository / REVIEW if name == REVIEW else manifest.parent / name
    target.write_bytes(target.read_bytes() + b"\n")
    before, index = _snapshot(repository), _index(repository)
    with pytest.raises((ValueError, SystemExit), match="(?i)(index|stag|work.?tree)"):
        _automatic(repository, manifest)
    assert _snapshot(repository) == before
    assert _index(repository) == index


def test_untracked_member_in_tracked_manifest_is_rejected(repository: Path) -> None:
    manifest = _bundle(repository)
    _stage(repository)
    _git(
        repository,
        "rm",
        "--cached",
        "--",
        (manifest.parent / "samples.json").relative_to(repository).as_posix(),
    )
    before, index = _snapshot(repository), _index(repository)
    with pytest.raises((ValueError, SystemExit), match="(?i)(index|stag|track)"):
        _automatic(repository, manifest)
    assert _snapshot(repository) == before
    assert _index(repository) == index


def test_existing_untracked_gzip_companion_is_never_overwritten(
    repository: Path,
) -> None:
    manifest = _bundle(repository)
    _stage(repository)
    (manifest.parent / "samples.json.gz").write_bytes(b"an unrelated retained artifact")
    before, index = _snapshot(repository), _index(repository)
    with pytest.raises((FileExistsError, ValueError, SystemExit)):
        _automatic(repository, manifest)
    assert _snapshot(repository) == before
    assert _index(repository) == index


@pytest.mark.parametrize(
    "invalid",
    [
        "schema",
        "revision",
        "dirty",
        "member-hash",
        "member-size",
        "review-hash",
        "review-schema",
    ],
)
def test_invalid_metadata_is_rejected_before_any_mutation(
    repository: Path, invalid: str
) -> None:
    manifest_path = _bundle(repository)
    manifest = json.loads(manifest_path.read_bytes())
    if invalid == "schema":
        manifest["schema"] = "custom.unreviewed-publication.v1"
    elif invalid == "revision":
        manifest["source"]["revision"] = "main"
    elif invalid == "dirty":
        manifest["source"]["dirty"] = "false"
    elif invalid == "member-hash":
        manifest["files"][0]["sha256"] = "0" * 64
    elif invalid == "member-size":
        manifest["files"][0]["bytes"] += 1
    else:
        review_path = repository / REVIEW
        review = json.loads(review_path.read_bytes())
        if invalid == "review-hash":
            review["files"][0]["sha256"] = "0" * 64
        else:
            review["schema"] = "custom.unreviewed-large-evidence.v1"
        review_path.write_bytes(_json(review))
    manifest_path.write_bytes(_json(manifest))
    _stage(repository)
    before, index = _snapshot(repository), _index(repository)
    with pytest.raises((ValueError, SystemExit)):
        _automatic(repository, manifest_path)
    assert _snapshot(repository) == before
    assert _index(repository) == index


@pytest.mark.parametrize("invalid", ["member-hash", "schema", "unstaged", "companion"])
def test_later_invalid_publication_does_not_partially_compact_earlier_one(
    repository: Path, invalid: str
) -> None:
    first = _bundle(repository, "a-valid")
    last = _bundle(repository, "z-invalid")
    manifest = json.loads(last.read_bytes())
    if invalid == "member-hash":
        manifest["files"][0]["sha256"] = "0" * 64
    elif invalid == "schema":
        manifest["schema"] = "custom.unreviewed-publication.v1"
    last.write_bytes(_json(manifest))
    _stage(repository)
    if invalid == "unstaged":
        last.write_bytes(last.read_bytes() + b"\n")
    elif invalid == "companion":
        (last.parent / "samples.json.gz").write_bytes(b"pre-existing output")
    before, index = _snapshot(repository), _index(repository)
    with pytest.raises((ValueError, FileExistsError, SystemExit)):
        _automatic(repository, first, last)
    assert _snapshot(repository) == before
    assert _index(repository) == index


def test_check_remains_read_only_with_tracked_large_publication(
    repository: Path,
) -> None:
    _bundle(repository)
    _stage(repository)
    before, index = _snapshot(repository), _index(repository)
    with pytest.raises(SystemExit, match="not compacted"):
        compact.main(["--check"])
    assert _snapshot(repository) == before
    assert _index(repository) == index


def test_identical_payloads_have_identical_compressed_bytes(repository: Path) -> None:
    first = _bundle(repository, "first")
    second = _bundle(repository, "second")
    _stage(repository)
    with pytest.raises(SystemExit, match="(?i)(re.?stage|stage the|git add)"):
        _automatic(repository, first, second)
    assert (first.parent / "samples.json.gz").read_bytes() == (
        second.parent / "samples.json.gz"
    ).read_bytes()


def _reference_bundle(root: Path, *, packed: bool, reference_kind: str) -> Path:
    """Build a valid later bundle whose members cannot trigger compression."""
    manifest_path = _bundle(root, "z-no-candidates", sample_size=1024)
    directory = manifest_path.parent
    manifest = json.loads(manifest_path.read_bytes())
    evidence = json.loads((directory / "evidence.json").read_bytes())
    if reference_kind == "record_parts":
        child = b'[{"energy":-1.2345678901234567,"passed":false}]\n'
        name = "measurements.json"
        (directory / name).write_bytes(child)
        entry = {"path": name, "bytes": len(child), "sha256": _digest(child)}
        manifest["files"].append({**entry, "role": "samples"})
        evidence["record_parts"] = {"retained_measurements": [entry]}
    if packed:
        replacements = {}
        for entry in manifest["files"]:
            if entry["role"] != "samples":
                continue
            old = entry["path"]
            data = gzip.compress((directory / old).read_bytes(), mtime=0)
            new = old + ".gz"
            (directory / new).write_bytes(data)
            (directory / old).unlink()
            replacements[old] = {
                "path": new,
                "bytes": len(data),
                "sha256": _digest(data),
            }
            entry.update(replacements[old])
        references = [
            *evidence["attachments"],
            *[
                entry
                for entries in evidence.get("record_parts", {}).values()
                for entry in entries
            ],
        ]
        for entry in references:
            entry.update(replacements[entry["path"]])
    evidence_name = "evidence.json.gz" if packed else "evidence.json"
    data = _json(evidence)
    if packed:
        data = gzip.compress(data, mtime=0)
        (directory / "evidence.json").unlink()
    (directory / evidence_name).write_bytes(data)
    entry = next(row for row in manifest["files"] if row["role"] == "evidence")
    entry.update(path=evidence_name, bytes=len(data), sha256=_digest(data))
    manifest_path.write_bytes(_json(manifest))
    validate_publication(
        manifest,
        {
            row["path"]: (directory / row["path"]).read_bytes()
            for row in manifest["files"]
        },
    )
    _review(root)
    return manifest_path


@pytest.mark.parametrize("packed", [False, True], ids=["all-small", "all-gzip"])
@pytest.mark.parametrize("reference_kind", ["attachments", "record_parts"])
@pytest.mark.parametrize(
    "damage",
    [
        "stale-hash",
        "stale-size",
        "missing-target",
        "missing-identity",
        "malformed-container",
        "malformed-binding",
    ],
)
def test_later_bundle_without_candidates_still_requires_valid_reference_graph(
    repository: Path, packed: bool, reference_kind: str, damage: str
) -> None:
    first = _bundle(repository, "a-eligible")
    last = _reference_bundle(repository, packed=packed, reference_kind=reference_kind)
    manifest = json.loads(last.read_bytes())
    entry = next(row for row in manifest["files"] if row["role"] == "evidence")
    evidence_path = last.parent / entry["path"]
    raw = evidence_path.read_bytes()
    evidence = json.loads(gzip.decompress(raw) if packed else raw)
    references = (
        evidence["attachments"]
        if reference_kind == "attachments"
        else evidence["record_parts"]["retained_measurements"]
    )
    if damage == "stale-hash":
        references[0]["sha256"] = "0" * 64
    elif damage == "stale-size":
        references[0]["bytes"] += 1
    elif damage == "missing-target":
        references[0]["path"] = "absent.json.gz" if packed else "absent.json"
    elif damage == "missing-identity":
        del references[0]["sha256"]
    elif damage == "malformed-container":
        evidence[reference_kind] = {} if reference_kind == "attachments" else []
    else:
        references[0] = "not a storage binding"
    data = _json(evidence)
    if packed:
        data = gzip.compress(data, mtime=0)
    evidence_path.write_bytes(data)
    # Authenticate the malformed record itself, so failure must come from the
    # reference graph rather than an unrelated stale publication checksum.
    entry.update(bytes=len(data), sha256=_digest(data))
    last.write_bytes(_json(manifest))
    _stage(repository)
    before, index = _snapshot(repository), _index(repository)
    with pytest.raises(
        (ValueError, TypeError), match="(?i)(reference|metadata|retained|identity)"
    ):
        _automatic(repository, first, last)
    assert _snapshot(repository) == before
    assert _index(repository) == index


@pytest.mark.parametrize("packed", [False, True], ids=["all-small", "all-gzip"])
@pytest.mark.parametrize("reference_kind", ["attachments", "record_parts"])
def test_valid_later_bundle_without_candidates_stays_unchanged(
    repository: Path, packed: bool, reference_kind: str
) -> None:
    first = _bundle(repository, "a-eligible")
    last = _reference_bundle(repository, packed=packed, reference_kind=reference_kind)
    _stage(repository)
    before = _snapshot(last.parent)
    index = _index(repository)
    with pytest.raises(SystemExit, match="(?i)(re.?stage|stage the|git add)"):
        _automatic(repository, first, last)
    assert (first.parent / "samples.json.gz").is_file()
    assert _snapshot(last.parent) == before
    assert _index(repository) == index


def _policy(root: Path, exceptions: dict | None = None) -> Path:
    path = root / "benchmarks/evidence-policy.json"
    path.write_bytes(
        _json(
            {
                "schema": "generativeqc.retention-policy.v1",
                "review_size_bytes": THRESHOLD,
                "exceptions": exceptions or {},
            }
        )
    )
    return path


@pytest.mark.parametrize("mismatch", ["modified", "deleted", "untracked"])
def test_policy_worktree_and_index_must_agree_before_compaction(
    repository: Path, mismatch: str
) -> None:
    manifest = _bundle(repository)
    policy = _policy(repository)
    _stage(repository)
    if mismatch == "modified":
        policy.write_bytes(policy.read_bytes() + b"\n")
    elif mismatch == "deleted":
        policy.unlink()
    else:
        _git(
            repository,
            "rm",
            "--cached",
            "--",
            policy.relative_to(repository).as_posix(),
        )
    before, index = _snapshot(repository), _index(repository)
    with pytest.raises(
        (ValueError, SystemExit), match="(?i)(index|stag|work.?tree|policy)"
    ):
        _automatic(repository, manifest)
    assert _snapshot(repository) == before
    assert _index(repository) == index


@pytest.mark.parametrize("name", ["samples.json", "evidence.json", "publication.json"])
def test_hash_pinned_policy_exception_blocks_changed_file(
    repository: Path, name: str
) -> None:
    manifest = _bundle(repository)
    pinned = manifest.parent / name
    _policy(
        repository,
        {
            pinned.relative_to(repository).as_posix(): {
                "owner": "independent storage consumer",
                "reason": "This exact representation has a separate consumer.",
                "sha256": _digest(pinned.read_bytes()),
            }
        },
    )
    _stage(repository)
    before, index = _snapshot(repository), _index(repository)
    with pytest.raises(ValueError, match="(?i)(pin|exception|migration)"):
        _automatic(repository, manifest)
    assert _snapshot(repository) == before
    assert _index(repository) == index


def test_policy_exception_for_unchanged_member_does_not_block_compaction(
    repository: Path,
) -> None:
    manifest = _bundle(repository)
    readme = manifest.parent / "README.md"
    original = readme.read_bytes()
    policy = _policy(
        repository,
        {
            readme.relative_to(repository).as_posix(): {
                "owner": "independent documentation consumer",
                "reason": "The unchanged reproduction guide is pinned.",
                "sha256": _digest(original),
            }
        },
    )
    policy_bytes = policy.read_bytes()
    _stage(repository)
    index = _index(repository)
    with pytest.raises(SystemExit, match="(?i)(re.?stage|stage the|git add)"):
        _automatic(repository, manifest)
    assert readme.read_bytes() == original
    assert policy.read_bytes() == policy_bytes
    assert _index(repository) == index


def test_exact_generated_orphans_are_reused_without_rewriting(repository: Path) -> None:
    manifest = _bundle(repository, large_evidence=True)
    before = _snapshot(repository)
    _stage(repository)
    index = _index(repository)
    with pytest.raises(SystemExit, match="(?i)(re.?stage|stage the|git add)"):
        _automatic(repository, manifest)
    companions = list(manifest.parent.glob("*.json.gz"))
    assert len(companions) == 2
    retained = {
        path: (path.read_bytes(), path.stat().st_ino, path.stat().st_mtime_ns)
        for path in companions
    }
    # Model pre-commit restoring tracked originals after its outer stash
    # conflicts. Its generated, untracked companions remain in the worktree.
    _git(repository, "checkout-index", "--all", "--force")
    for name, data in before.items():
        assert (repository / name).read_bytes() == data
    with pytest.raises(SystemExit, match="(?i)(re.?stage|stage the|git add)"):
        _automatic(repository, manifest)
    for path, (data, inode, mtime) in retained.items():
        assert path.read_bytes() == data
        assert path.stat().st_ino == inode
        assert path.stat().st_mtime_ns == mtime
    assert _index(repository) == index
    _stage(repository)
    compacted = _snapshot(repository)
    _automatic(repository, manifest)
    assert _snapshot(repository) == compacted


def test_equal_decoded_but_different_gzip_companion_is_preserved(
    repository: Path,
) -> None:
    manifest = _bundle(repository)
    _stage(repository)
    source = manifest.parent / "samples.json"
    companion = manifest.parent / "samples.json.gz"
    foreign = gzip.compress(source.read_bytes(), compresslevel=1, mtime=123)
    companion.write_bytes(foreign)
    assert gzip.decompress(foreign) == source.read_bytes()
    before, index = _snapshot(repository), _index(repository)
    with pytest.raises(FileExistsError, match="(?i)(differ|companion|preserve)"):
        _automatic(repository, manifest)
    assert _snapshot(repository) == before
    assert _index(repository) == index


def test_measured_evidence_revision_must_match_publication_source(
    repository: Path,
) -> None:
    manifest_path = _bundle(repository)
    manifest = json.loads(manifest_path.read_bytes())
    evidence_path = manifest_path.parent / "evidence.json"
    evidence = json.loads(evidence_path.read_bytes())
    evidence["revision"] = "c" * 40
    data = _json(evidence)
    evidence_path.write_bytes(data)
    entry = next(row for row in manifest["files"] if row["role"] == "evidence")
    entry.update(bytes=len(data), sha256=_digest(data))
    manifest_path.write_bytes(_json(manifest))
    _stage(repository)
    before, index = _snapshot(repository), _index(repository)
    with pytest.raises(ValueError, match="(?i)(source|revision|measured)"):
        _automatic(repository, manifest_path)
    assert _snapshot(repository) == before
    assert _index(repository) == index


@pytest.mark.parametrize("packed", [False, True], ids=["plain", "gzip"])
def test_historical_reproduction_envelope_keeps_exact_original_storage_links(
    repository: Path, packed: bool
) -> None:
    manifest_path = _bundle(repository)
    directory = manifest_path.parent
    original_samples = (directory / "samples.json").read_bytes()
    historical = json.loads((directory / "evidence.json").read_bytes())
    historical["revision"] = "c" * 40
    historical["attachments"] = [
        {"path": "retired-samples.json", "bytes": 512, "sha256": "0" * 64}
    ]
    historical["record_parts"] = {
        "historical_failures": [
            {"path": "retired-parts.json", "bytes": 17, "sha256": "0" * 64}
        ]
    }
    historical["original_run_notes"] = (
        "retain the historical envelope unchanged " * 4000
    )
    raw = _json(historical)
    assert len(raw) >= THRESHOLD
    name = "historical-run.json.gz" if packed else "historical-run.json"
    if packed:
        raw = gzip.compress(raw, mtime=0)
    retained = directory / name
    retained.write_bytes(raw)
    manifest = json.loads(manifest_path.read_bytes())
    manifest["files"].append(
        {
            "path": name,
            "role": "reproduction",
            "bytes": len(raw),
            "sha256": _digest(raw),
        }
    )
    manifest_path.write_bytes(_json(manifest))
    _review(repository)
    _stage(repository)
    index = _index(repository)
    with pytest.raises(SystemExit, match="(?i)(re.?stage|stage the|git add)"):
        _automatic(repository, manifest_path)
    assert (
        gzip.decompress((directory / "samples.json.gz").read_bytes())
        == original_samples
    )
    assert retained.read_bytes() == raw
    if not packed:
        assert not (directory / (name + ".gz")).exists()
    updated = json.loads(manifest_path.read_bytes())
    assert (
        next(row for row in updated["files"] if row["path"] == name)
        == manifest["files"][-1]
    )
    validate_publication(
        updated,
        {
            row["path"]: (directory / row["path"]).read_bytes()
            for row in updated["files"]
        },
    )
    assert _index(repository) == index
    _stage(repository)
    before = _snapshot(repository)
    _automatic(repository, manifest_path)
    assert _snapshot(repository) == before


def _use_sharded_review(root: Path) -> None:
    """Convert the small publication fixture without changing evidence bytes."""
    previous = json.loads((root / REVIEW).read_bytes())
    directory = root / "benchmarks/evidence-reviews"
    directory.mkdir(parents=True)
    for family, metadata in previous["families"].items():
        shard = {
            "schema": "generativeqc.legacy-large-evidence-family.v1",
            "family": family,
            **metadata,
            "files": [
                entry for entry in previous["files"] if entry["family"] == family
            ],
        }
        (directory / f"{family}.json").write_bytes(_json(shard))
    (root / REVIEW).write_bytes(
        _json(
            {
                "schema": "generativeqc.legacy-large-evidence-index.v2",
                "directory": "benchmarks/evidence-reviews",
                "threshold_bytes": previous["threshold_bytes"],
                "scope": "fixture",
            }
        )
    )


def test_sharded_compaction_rewrites_only_the_owning_review(
    repository: Path,
) -> None:
    first = _bundle(repository, name="first")
    _bundle(repository, name="second")
    _use_sharded_review(repository)
    _stage(repository)
    index_before = (repository / REVIEW).read_bytes()
    sibling = repository / "benchmarks/evidence-reviews/second.json"
    sibling_before = sibling.read_bytes()
    owner = repository / "benchmarks/evidence-reviews/first.json"
    original_owner = owner.read_bytes()
    staged = _index(repository)

    with pytest.raises(SystemExit, match="(?i)(re.?stage|stage the|git add)"):
        _automatic(repository, first)

    assert (repository / REVIEW).read_bytes() == index_before
    assert sibling.read_bytes() == sibling_before
    assert owner.read_bytes() != original_owner
    assert _index(repository) == staged
    _stage(repository)
    saved = _snapshot(repository)
    _automatic(repository, first)
    assert _snapshot(repository) == saved


def test_sharded_compaction_refuses_stale_owner_receipt(
    repository: Path,
) -> None:
    first = _bundle(repository)
    _use_sharded_review(repository)
    owner = repository / "benchmarks/evidence-reviews/campaign.json"
    record = json.loads(owner.read_bytes())
    record["files"][0]["sha256"] = "0" * 64
    owner.write_bytes(_json(record))
    _stage(repository)
    before, staged = _snapshot(repository), _index(repository)
    with pytest.raises(ValueError, match="stale legacy review identity"):
        _automatic(repository, first)
    assert _snapshot(repository) == before
    assert _index(repository) == staged
