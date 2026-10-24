"""Deterministically gzip bulky JSON members of selected evidence publications."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
import lzma
import os
import re
import shutil
import subprocess
import sys
import tempfile
import zlib
from dataclasses import dataclass
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.generativeqc_validation.retention import MAX_TRACKED_BYTES, safe_relative
from tools.generativeqc_validation.storage_json import (
    decode_storage_json,
    replace_metadata,
)

ROOT = Path(__file__).resolve().parents[1]
REVIEW = ROOT / "benchmarks/legacy-evidence-review.json"
THRESHOLD = 128 << 10
ROLES = {"evidence", "samples"}
SCHEMAS = {"generativeqc.benchmark-publication.v1", "vibeqc.benchmark-publication.v1"}
ALL_ROLES = ROLES | {"summary", "input", "reproduction", "source-patch"}
MAX_STORED_BYTES = 16 << 20
MAX_EXPANDED_BYTES = 32 << 20
MAX_TOTAL_STORED = 64 << 20
MAX_TOTAL_EXPANDED = 256 << 20
MAX_MEMBERS = 4096
MAX_BUNDLE_MEMBERS = 512


@dataclass
class Budget:
    """Finite automatic-hook input bounds shared across the selected batch."""

    stored: int = 0
    expanded: int = 0
    members: int = 0

    def add(self, stored: int, expanded: int) -> None:
        self.stored += stored
        self.expanded += expanded
        self.members += 1
        if (
            self.stored > MAX_TOTAL_STORED
            or self.expanded > MAX_TOTAL_EXPANDED
            or self.members > MAX_MEMBERS
        ):
            raise ValueError("publication batch exceeds compaction resource limits")


@dataclass
class Plan:
    """Verified original bytes and proposed storage-only replacements."""

    manifest_path: Path
    manifest: dict
    originals: dict[str, bytes]
    changes: list[tuple[str, str, bytes]]


def publication_paths() -> tuple[str, ...]:
    """Discover only tracked publications, never transient run directories."""
    output = subprocess.check_output(
        ["git", "ls-files", "-z", "--", "benchmarks/results/**/publication.json"],
        cwd=ROOT,
    )
    return tuple(sorted(path.decode() for path in output.split(b"\0") if path))


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def json_bytes(value: object) -> bytes:
    return (
        json.dumps(value, indent=2, allow_nan=False, ensure_ascii=False).encode()
        + b"\n"
    )


def compressed(data: bytes) -> bytes:
    output = io.BytesIO()
    with gzip.GzipFile(
        fileobj=output, mode="wb", filename="", compresslevel=9, mtime=0
    ) as stream:
        stream.write(data)
    packed = output.getvalue()
    if gzip.decompress(packed) != data:
        raise ValueError("compressed payload verification failed")
    return packed


def _commit_changes(
    originals: dict[str, bytes],
    changes: list[tuple[str, str, bytes]],
    *,
    directory: Path,
) -> None:
    """Atomically replace each file and roll back any reported batch failure.

    All originals remain in a recovery directory until every replacement is
    verified. A hard process/host crash can leave that directory; never guess at
    recovery or silently delete it on rerun. This is exception safety, not a
    claim that a filesystem supports an atomic multi-file rename.
    """
    for name, raw in originals.items():
        if _read(ROOT / name) != raw:
            raise ValueError(f"file changed during compaction: {name}")
    stage = Path(tempfile.mkdtemp(prefix=".evidence-stage-", dir=directory))
    backups: dict[str, Path] = {}
    created: list[Path] = []
    replaced: list[str] = []
    retired: list[str] = []
    try:
        for index, (name, raw) in enumerate(originals.items()):
            backup = stage / f"old-{index}"
            backup.write_bytes(raw)
            if backup.read_bytes() != raw:
                raise OSError("incomplete original backup")
            backups[name] = backup
        for index, (_old, _new, data) in enumerate(changes):
            output = stage / f"new-{index}"
            output.write_bytes(data)
            if output.read_bytes() != data:
                raise OSError("incomplete staged output")
        (stage / "recovery.json").write_bytes(
            json_bytes(
                {
                    "originals": {
                        name: backup.name for name, backup in backups.items()
                    },
                    "changes": [
                        {"old": old, "new": new, "sha256": digest(data)}
                        for old, new, data in changes
                    ],
                }
            )
        )
        # Recheck after staging. No input has been changed yet.
        for name, raw in originals.items():
            if _read(ROOT / name) != raw:
                raise ValueError(f"file changed during compaction: {name}")
        for old, new, data in changes:
            if old != new:
                target = ROOT / new
                try:
                    stream = target.open("xb")
                except FileExistsError:
                    _companion(target, data)
                else:
                    created.append(target)
                    with stream:
                        if stream.write(data) != len(data):
                            raise OSError("incomplete compressed output")
                if _read(target) != data:
                    raise OSError("compressed output verification failed")
        # Rebind all metadata only after every new companion is verified.
        for index, (old, new, data) in enumerate(changes):
            if old == new:
                replaced.append(old)
                (stage / f"new-{index}").replace(ROOT / new)
                if _read(ROOT / new) != data:
                    raise OSError("metadata replacement verification failed")
        for old, new, _data in changes:
            if old != new:
                if _read(ROOT / old) != originals[old]:
                    raise ValueError(f"original changed before retirement: {old}")
                retired.append(old)
                (ROOT / old).unlink()
    except BaseException as failure:
        rollback_errors = []
        for name in reversed([*replaced, *retired]):
            try:
                original = ROOT / name
                if not original.is_file() or _read(original) != originals[name]:
                    backups[name].replace(original)
            except (OSError, ValueError) as error:
                rollback_errors.append(error)
        # Do not discard companions if any metadata rollback failed.
        if not rollback_errors:
            for target in reversed(created):
                try:
                    target.unlink(missing_ok=True)
                except OSError as error:
                    rollback_errors.append(error)
        if rollback_errors:
            raise RuntimeError(
                f"Compaction rollback incomplete; retained recovery files: {stage}"
            ) from failure
        shutil.rmtree(stage)
        raise
    else:
        shutil.rmtree(stage)


def _plain(data: bytes, name: str) -> bytes:
    if len(data) > MAX_STORED_BYTES:
        raise ValueError("publication member exceeds stored-byte limit")
    if name.endswith(".gz"):
        decoder = zlib.decompressobj(wbits=31)
        plain = decoder.decompress(data, MAX_EXPANDED_BYTES + 1)
    elif name.endswith(".xz"):
        decoder = lzma.LZMADecompressor(format=lzma.FORMAT_XZ, memlimit=96 << 20)
        plain = decoder.decompress(data, max_length=MAX_EXPANDED_BYTES + 1)
    else:
        plain = data
        decoder = None
    if len(plain) > MAX_EXPANDED_BYTES:
        raise ValueError("publication member exceeds expanded-byte limit")
    if decoder is not None and (not decoder.eof or decoder.unused_data):
        raise ValueError("truncated or trailing compressed publication stream")
    return plain


def _read(path: Path) -> bytes:
    _regular(path, ROOT)
    with path.open("rb") as stream:
        raw = stream.read(MAX_STORED_BYTES + 1)
    if len(raw) > MAX_STORED_BYTES:
        raise ValueError(f"publication input exceeds stored-byte limit: {path}")
    return raw


def _companion(path: Path, expected: bytes) -> None:
    """Reuse only an exact deterministic output; never overwrite a collision."""
    _regular(path, ROOT)
    if _read(path) != expected:
        raise FileExistsError(
            f"compressed companion differs: {path}; preserve and inspect both files. "
            "If pre-commit restored unstaged edits, stage the complete bundle first; "
            "only an exact verified companion can be reused automatically."
        )


def _regular(path: Path, directory: Path) -> None:
    if not path.absolute().is_relative_to(directory.absolute()):
        raise ValueError("publication path escapes its directory")
    if any(part.is_symlink() for part in (path, *path.parents) if part != ROOT.parent):
        raise ValueError("publication path must not contain symlinks")
    if not path.is_file():
        raise ValueError(f"publication path is not a regular file: {path}")


def plan_publication(
    relative: str, *, strict: bool = False, budget: Budget | None = None
) -> Plan:
    budget = Budget() if budget is None else budget
    manifest_path = ROOT / safe_relative(relative)
    if not manifest_path.resolve().is_relative_to(ROOT.resolve()):
        raise ValueError("publication manifest path escapes the checkout")
    directory = manifest_path.parent
    _regular(manifest_path, ROOT)
    if list(directory.glob(".evidence-stage-*")):
        raise ValueError("incomplete compaction recovery directory requires inspection")
    original_manifest = _read(manifest_path)
    budget.add(len(original_manifest), len(original_manifest))
    manifest = decode_storage_json(original_manifest)
    if not isinstance(manifest, dict) or manifest.get("schema") not in SCHEMAS:
        raise ValueError("unsupported publication manifest schema")
    entries = manifest.get("files")
    if (
        not isinstance(entries, list)
        or not entries
        or len(entries) > MAX_BUNDLE_MEMBERS
    ):
        raise ValueError("publication files must be a nonempty list")
    # Storage identities authenticate bytes, not paths. Validate every member
    # before reads or writes so compaction cannot move/delete another bundle.
    names = set()
    for entry in entries:
        if not isinstance(entry, dict) or entry.get("role") not in ALL_ROLES:
            raise ValueError("unsupported publication member role")
        if (
            type(entry.get("bytes")) is not int
            or not 0 <= entry["bytes"] <= MAX_STORED_BYTES
            or not re.fullmatch(r"[0-9a-f]{64}", str(entry.get("sha256", "")))
        ):
            raise ValueError("invalid publication member identity")
        name = safe_relative(entry["path"])
        if name == "publication.json":
            raise ValueError("publication cannot contain its manifest")
        logical = name.removesuffix(".gz").removesuffix(".xz")
        if any(old.removesuffix(".gz").removesuffix(".xz") == logical for old in names):
            raise ValueError("duplicate or mixed publication member representations")
        if name in names:
            raise ValueError("duplicate publication member path")
        names.add(name)
        if not (directory / name).resolve().is_relative_to(directory.resolve()):
            raise ValueError("publication member path escapes its directory")

    # Authenticate every original member before updating any storage identity.
    # Keep these exact bytes so compaction does not re-read unchecked content.
    contents = {}
    for entry in entries:
        path = directory / entry["path"]
        _regular(path, directory)
        raw = _read(path)
        if len(raw) != entry["bytes"] or digest(raw) != entry["sha256"]:
            raise ValueError(f"publication identity mismatch: {path}")
        if str(entry["path"]).endswith((".json", ".json.gz", ".json.xz")):
            plain = _plain(raw, entry["path"])
            budget.add(len(raw), len(plain))
            decode_storage_json(plain)
        else:
            budget.add(len(raw), len(raw))
        contents[entry["path"]] = raw

    if strict and "source" in manifest:
        source = manifest["source"]
        evidence_names = [
            entry["path"] for entry in entries if entry["role"] == "evidence"
        ]
        if (
            not isinstance(source, dict)
            or type(source.get("dirty")) is not bool
            or not re.fullmatch(
                r"[0-9a-f]{40}|[0-9a-f]{64}", str(source.get("revision", ""))
            )
            or len(evidence_names) != 1
        ):
            raise ValueError(
                "automatic compaction requires exact measured source identity and one evidence record"
            )
        evidence = decode_storage_json(
            _plain(contents[evidence_names[0]], evidence_names[0])
        )
        if (
            not isinstance(evidence, dict)
            or evidence.get("revision") != source["revision"]
        ):
            raise ValueError(
                "publication source differs from measured evidence identity"
            )

    # Resolve all JSON reference dependencies before committing. Sample records
    # can themselves contain record_parts; updating only the evidence envelope
    # would leave them pointing at removed plain companions. Resolve leaves
    # first so every parent binds the final stored child bytes.
    decoded = {}
    roles = {entry["path"]: entry["role"] for entry in entries}
    for name, raw in contents.items():
        if roles[name] in ROLES and name.endswith((".json", ".json.gz", ".json.xz")):
            value = decode_storage_json(_plain(raw, name))
            if isinstance(value, dict):
                # Retain only storage bindings and top-level field presence,
                # not every parsed scientific array from the whole bundle.
                decoded[name] = {
                    key: item if key in {"attachments", "record_parts"} else None
                    for key, item in value.items()
                }
    rewritten: dict[str, tuple[str, bytes]] = {}
    visiting: set[str] = set()

    def rewrite(old: str) -> tuple[str, bytes]:
        if old in rewritten:
            return rewritten[old]
        if old in visiting:
            raise ValueError("cyclic publication storage references")
        visiting.add(old)
        raw = contents[old]
        value = decoded.get(old)
        changed = False
        metadata = {}
        if value is not None:
            attachments = value.get("attachments", [])
            parts = value.get("record_parts", {})
            if not isinstance(attachments, list) or not isinstance(parts, dict):
                raise ValueError("invalid publication storage metadata")
            if any(
                not isinstance(part, list) or not part or field in value
                for field, part in parts.items()
            ):
                raise ValueError("record parts overwrite a field or contain no files")
            references = list(attachments) + [
                entry for part in parts.values() for entry in part
            ]
            if any(not isinstance(entry, dict) for entry in references):
                raise ValueError("invalid publication storage reference")
            parent = Path(old).parent
            replacements = {}
            for binding in references:
                reference = safe_relative(binding.get("path"))
                child = (parent / reference).as_posix()
                if child not in contents:
                    raise ValueError(f"publication reference is not retained: {child}")
                original = contents[child]
                if binding.get("sha256") != digest(original) or (
                    "bytes" in binding
                    and (
                        type(binding["bytes"]) is not int
                        or binding["bytes"] != len(original)
                    )
                ):
                    raise ValueError(
                        f"publication reference identity mismatch: {child}"
                    )
                new_child, data = rewrite(child)
                if new_child != child or data != contents[child]:
                    replacements[reference] = (
                        Path(new_child).relative_to(parent).as_posix(),
                        data,
                    )
            # Patch only the storage leaves; never reserialize scientific numbers,
            # string escapes, negative zero or high precision reference values.
            for index, binding in enumerate(attachments):
                if binding["path"] in replacements:
                    new_path, child_data = replacements[binding["path"]]
                    prefix = ("attachments", index)
                    metadata[prefix + ("path",)] = new_path
                    metadata[prefix + ("sha256",)] = digest(child_data)
                    if "bytes" in binding:
                        metadata[prefix + ("bytes",)] = len(child_data)
            for field, bindings in parts.items():
                seen = set()
                for index, binding in enumerate(bindings):
                    if binding["path"] in seen or "bytes" not in binding:
                        raise ValueError("duplicate or incomplete record part")
                    seen.add(binding["path"])
                    if not isinstance(
                        decode_storage_json(
                            _plain(
                                contents[(parent / binding["path"]).as_posix()],
                                binding["path"],
                            )
                        ),
                        list,
                    ):
                        raise TypeError("record part must contain a list")
                    if binding["path"] in replacements:
                        new_path, child_data = replacements[binding["path"]]
                        prefix = ("record_parts", field, index)
                        metadata[prefix + ("path",)] = new_path
                        metadata[prefix + ("sha256",)] = digest(child_data)
                        metadata[prefix + ("bytes",)] = len(child_data)
            changed = bool(metadata)
        data = replace_metadata(_plain(raw, old), metadata) if changed else raw
        pack_plain = (
            old.endswith(".json") and roles[old] in ROLES and len(data) >= THRESHOLD
        )
        new = old + ".gz" if pack_plain else old
        if pack_plain or (changed and old.endswith(".json.gz")):
            data = compressed(data)
        elif changed and old.endswith(".json.xz"):
            data = lzma.compress(data, format=lzma.FORMAT_XZ, check=lzma.CHECK_CRC64)
        rewritten[old] = (new, data)
        visiting.remove(old)
        return new, data

    changes: list[tuple[str, str, bytes]] = []
    for entry in entries:
        old = entry["path"]
        new, data = rewrite(old)
        if new != old or data != contents[old]:
            if len(data) > MAX_TRACKED_BYTES:
                raise ValueError(
                    "compacted member still exceeds the hard 1 MiB limit; "
                    "use a separately reviewed partitioned layout"
                )
            changes.append(
                (
                    (directory / old).relative_to(ROOT).as_posix(),
                    (directory / new).relative_to(ROOT).as_posix(),
                    data,
                )
            )
            entry.update(path=new, bytes=len(data), sha256=digest(data))

    # A pre-existing companion is not ours to overwrite, even in check mode.
    for old, new, _data in changes:
        target = ROOT / new
        if old != new and (target.exists() or target.is_symlink()):
            _companion(target, _data)
    originals = {
        (directory / name).relative_to(ROOT).as_posix(): raw
        for name, raw in contents.items()
    }
    originals[relative] = original_manifest
    return Plan(manifest_path, manifest, originals, changes)


def compact_publication(
    relative: str, *, check: bool = False
) -> list[tuple[str, str, bytes]]:
    plan = plan_publication(relative, strict=not check)
    if not check and plan.changes:
        _commit_changes(
            plan.originals,
            [*plan.changes, (relative, relative, _manifest_bytes(plan))],
            directory=plan.manifest_path.parent,
        )
    return plan.changes


def _review_bytes(
    changes: list[tuple[str, str, bytes]], originals: dict[str, bytes]
) -> bytes:
    _regular(REVIEW, ROOT)
    review = decode_storage_json(_read(REVIEW))
    if (
        not isinstance(review, dict)
        or review.get("schema") != "generativeqc.legacy-large-evidence-review.v1"
        or type(review.get("threshold_bytes")) is not int
        or review["threshold_bytes"] <= 0
        or not isinstance(review.get("files"), list)
    ):
        raise ValueError("unsupported legacy review metadata")
    rows = {}
    for row in review["files"]:
        if not isinstance(row, dict):
            raise TypeError("invalid legacy review row")
        name = safe_relative(row.get("path"))
        if name in rows:
            raise ValueError("duplicate legacy review row")
        rows[name] = row
    for old, new, data in changes:
        row = rows.pop(old, None)
        if row is not None and (
            row.get("sha256") != digest(originals[old])
            or row.get("bytes") != len(originals[old])
        ):
            raise ValueError(f"stale legacy review identity: {old}")
        if len(data) >= review["threshold_bytes"]:
            if row is None:
                raise ValueError(
                    f"missing legacy review row for compressed evidence: {old}"
                )
            row.update(path=new, bytes=len(data), sha256=digest(data))
            rows[new] = row
    review["files"] = sorted(rows.values(), key=lambda row: row["path"])
    return json_bytes(review)


def _manifest_bytes(plan: Plan) -> bytes:
    name = plan.manifest_path.relative_to(ROOT).as_posix()
    updates = {
        ("files", index, field): entry[field]
        for index, entry in enumerate(plan.manifest["files"])
        for field in ("path", "bytes", "sha256")
    }
    return replace_metadata(plan.originals[name], updates)


def _auto_targets(filenames: list[str]) -> tuple[str, ...]:
    names = {safe_relative(name) for name in filenames}
    targets = tuple(
        path
        for path in publication_paths()
        if any(Path(name).is_relative_to(Path(path).parent) for name in names)
    )
    unsupported = [
        name
        for name in sorted(names)
        if name.startswith("benchmarks/results/")
        and name.endswith(".json")
        and not any(Path(name).is_relative_to(Path(path).parent) for path in targets)
    ]
    for name in unsupported:
        print(f"not auto-compacted (no shared publication contract): {name}")
    return targets


def _index_bytes(originals: dict[str, bytes]) -> dict[str, bytes]:
    """Read only this batch's stage-zero ordinary blobs, under byte limits."""
    env = {
        **os.environ,
        "GIT_NO_LAZY_FETCH": "1",
        "GIT_ALLOW_PROTOCOL": "",
        "GIT_TERMINAL_PROMPT": "0",
    }
    rows = subprocess.check_output(
        ["git", "--literal-pathspecs", "ls-files", "--stage", "-z", "--", *originals],
        cwd=ROOT,
        env=env,
    )
    entries = []
    for row in filter(None, rows.split(b"\0")):
        metadata, path = row.split(b"\t", 1)
        mode, oid, stage = metadata.split()
        if stage != b"0" or mode not in {b"100644", b"100755"}:
            raise ValueError("publication requires ordinary files and a resolved index")
        entries.append((path.decode(), oid))
    if not entries:
        return {}
    query = b"\n".join(oid for _name, oid in entries) + b"\n"
    sizes = subprocess.check_output(
        ["git", "cat-file", "--batch-check"], input=query, cwd=ROOT, env=env
    ).splitlines()
    total = 0
    for row in sizes:
        _oid, kind, length = row.split()
        if kind != b"blob" or int(length) > MAX_STORED_BYTES:
            raise ValueError("staged publication input exceeds stored-byte limit")
        total += int(length)
    if total > MAX_TOTAL_STORED:
        raise ValueError("staged publication batch exceeds stored-byte limit")
    batch = subprocess.check_output(
        ["git", "cat-file", "--batch"], input=query, cwd=ROOT, env=env
    )
    offset = 0
    result = {}
    for name, oid in entries:
        end = batch.index(b"\n", offset)
        actual, kind, length = batch[offset:end].split()
        if actual != oid or kind != b"blob":
            raise ValueError("unexpected staged publication object")
        offset = end + 1
        result[name] = batch[offset : offset + int(length)]
        offset += int(length) + 1
    return result


def _check_index(originals: dict[str, bytes], index: dict[str, bytes]) -> None:
    for name, raw in originals.items():
        if index.get(name) != raw:
            raise ValueError(
                f"partial staging or untracked publication input: {name}; "
                "review and stage the complete bundle before compaction"
            )


def _check_publications(targets: tuple[str, ...]) -> None:
    """Check every publication without retaining the repository's raw payloads.

    Each plan keeps the existing member and batch limits. Only path identities
    survive between plans so overlapping inventories still fail closed. This
    read-only path does not change the atomic mutation/pre-commit batch policy.
    """
    seen: set[str] = set()
    for target in targets:
        plan = plan_publication(target, strict=False, budget=Budget())
        if seen & plan.originals.keys():
            raise ValueError("overlapping publication inventories")
        seen.update(plan.originals)
        if plan.changes:
            raise SystemExit("retained evidence JSON is not compacted")
        # Release original bytes before constructing the next bounded plan.
        del plan
    print(
        f"compressed evidence publications: {len(targets)} eligible manifests, up to date"
    )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--check",
        action="store_true",
        help="read-only CI check: fail if compaction would change files",
    )
    mode.add_argument(
        "--pre-commit",
        action="store_true",
        help="compact touched staged publications; never stage changes",
    )
    parser.add_argument(
        "--publication",
        action="append",
        help="explicit reviewed publication path (repeatable)",
    )
    parser.add_argument("filenames", nargs="*", help="pre-commit's touched paths")
    args = parser.parse_args(argv)
    if (args.pre_commit and args.publication) or (
        args.filenames and not args.pre_commit
    ):
        parser.error("filenames require --pre-commit; --publication is manual only")
    if list(ROOT.glob(".evidence-stage-*")):
        raise ValueError("incomplete compaction recovery directory requires inspection")
    targets = (
        _auto_targets(args.filenames)
        if args.pre_commit
        else tuple(args.publication or publication_paths())
    )
    if args.check:
        _check_publications(targets)
        return
    # Plan every selected publication before any mutation, including a later
    # invalid bundle, stale review row or unknown cross-publication binding.
    budget = Budget()
    plans = [
        plan_publication(target, strict=not args.check, budget=budget)
        for target in targets
    ]
    originals = {}
    changes = []
    for plan in plans:
        if args.pre_commit:
            source = plan.manifest.get("source", {})
            if (
                not isinstance(source, dict)
                or type(source.get("dirty")) is not bool
                or not re.fullmatch(
                    r"[0-9a-f]{40}|[0-9a-f]{64}", str(source.get("revision", ""))
                )
            ):
                raise ValueError(
                    "automatic compaction requires exact measured source identity"
                )
            if (
                not plan.manifest_path.relative_to(ROOT)
                .as_posix()
                .startswith("benchmarks/results/")
            ):
                raise ValueError(
                    "automatic compaction is limited to retained publications"
                )
        if originals.keys() & plan.originals.keys():
            raise ValueError("overlapping publication inventories")
        originals.update(plan.originals)
        changes.extend(plan.changes)
    index = _index_bytes(originals) if args.pre_commit and plans else {}
    if args.pre_commit:
        _check_index(originals, index)
    if args.check and changes:
        raise SystemExit("retained evidence JSON is not compacted")
    if not changes:
        print(
            f"compressed evidence publications: {len(plans)} eligible manifests, up to date"
        )
        return
    review_name = REVIEW.relative_to(ROOT).as_posix()
    originals[review_name] = _read(REVIEW)
    budget.add(len(originals[review_name]), len(originals[review_name]))
    next_review = _review_bytes(changes, originals)
    # Hash-pinned exceptional/custom consumers require their separate storage
    # adapter. Do not silently change that policy or claim they were migrated.
    policy_path = ROOT / "benchmarks/evidence-policy.json"
    policy_name = policy_path.relative_to(ROOT).as_posix()
    policy = {}
    if (
        args.pre_commit
        and not policy_path.exists()
        and _index_bytes({policy_name: b""})
    ):
        raise ValueError(
            "partial staging: tracked retention policy is missing from worktree"
        )
    if policy_path.exists():
        policy_raw = _read(policy_path)
        budget.add(len(policy_raw), len(policy_raw))
        originals[policy_name] = policy_raw
        policy = decode_storage_json(policy_raw)
        if (
            not isinstance(policy, dict)
            or policy.get("schema") != "generativeqc.retention-policy.v1"
            or not isinstance(policy.get("exceptions"), dict)
        ):
            raise ValueError("unsupported retention policy metadata")
    all_changes = list(changes)
    for plan in plans:
        if plan.changes:
            name = plan.manifest_path.relative_to(ROOT).as_posix()
            all_changes.append((name, name, _manifest_bytes(plan)))
    if next_review != originals[review_name]:
        all_changes.append((review_name, review_name, next_review))
    if any(old in policy.get("exceptions", {}) for old, _new, _raw in all_changes):
        raise ValueError(
            "hash-pinned retention exception needs explicit storage migration"
        )
    if args.pre_commit:
        _check_index(originals, _index_bytes(originals))
    _commit_changes(originals, all_changes, directory=ROOT)
    for old, new, _raw in changes:
        print(f"compacted: {old} -> {new}")
    if args.pre_commit:
        raise SystemExit(
            "Evidence compacted. Review the visible changes, stage the whole "
            "publication and review metadata, then rerun the commit. Nothing was staged."
        )
    print(f"compressed evidence publications: {len(changes)} changed files")


if __name__ == "__main__":
    main()
