"""Deterministically gzip bulky JSON members of selected evidence publications."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.generativeqc_validation.retention import safe_relative

ROOT = Path(__file__).resolve().parents[1]
REVIEW = ROOT / "benchmarks/legacy-evidence-review.json"
THRESHOLD = 128 << 10
ROLES = {"evidence", "samples"}


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
    return gzip.compress(data, compresslevel=9, mtime=0)


def _update_storage_references(
    evidence: dict, replacements: dict[str, tuple[str, bytes]]
) -> bool:
    changed = False
    for attachment in evidence.get("attachments", []):
        old = attachment.get("path")
        if old in replacements:
            new, data = replacements[old]
            attachment["path"] = new
            attachment["sha256"] = digest(data)
            if "bytes" in attachment:
                attachment["bytes"] = len(data)
            changed = True
    for entries in evidence.get("record_parts", {}).values():
        for entry in entries:
            old = entry.get("path")
            if old in replacements:
                new, data = replacements[old]
                entry.update(path=new, bytes=len(data), sha256=digest(data))
                changed = True
    return changed


def _commit_publication(
    manifest_path: Path, manifest: dict, changes: list[tuple[str, str, bytes]]
) -> None:
    """Stage one publication and roll back reported write/rename failures.

    This is exception safety, not a crash-consistent multi-publication journal.
    """
    stage = Path(tempfile.mkdtemp(prefix=".evidence-stage-", dir=manifest_path.parent))
    backups: list[tuple[Path, Path]] = []
    created: list[Path] = []
    try:
        for index, (_old, _new, data) in enumerate(changes):
            (stage / f"new-{index}").write_bytes(data)
        next_manifest = stage / "next-manifest"
        next_manifest.write_bytes(json_bytes(manifest))

        # Exclusive creation closes the gap after the earlier companion check.
        for old, new, data in changes:
            if old != new:
                target = ROOT / new
                with target.open("xb") as stream:
                    created.append(target)
                    stream.write(data)
        # Retain exact original files until the new manifest is committed.
        for index, (old, _new, _data) in enumerate(changes):
            original, backup = ROOT / old, stage / f"old-{index}"
            original.replace(backup)
            backups.append((backup, original))
        for index, (old, new, _data) in enumerate(changes):
            if old == new:
                target = ROOT / new
                (stage / f"new-{index}").replace(target)
                created.append(target)
        next_manifest.replace(manifest_path)
    except BaseException as failure:
        rollback_errors = []
        for target in reversed(created):
            try:
                target.unlink(missing_ok=True)
            except OSError as error:
                rollback_errors.append(error)
        for backup, original in reversed(backups):
            try:
                backup.replace(original)
            except OSError as error:
                rollback_errors.append(error)
        if rollback_errors:
            # Never remove the only remaining original if the filesystem also
            # refuses rollback. Leave the recovery directory in the exception.
            raise RuntimeError(
                f"Compaction rollback incomplete; retained recovery files: {stage}"
            ) from failure
        else:
            shutil.rmtree(stage)
        raise
    else:
        shutil.rmtree(stage)


def compact_publication(
    relative: str, *, check: bool = False
) -> list[tuple[str, str, bytes]]:
    manifest_path = ROOT / safe_relative(relative)
    if not manifest_path.resolve().is_relative_to(ROOT.resolve()):
        raise ValueError("publication manifest path escapes the checkout")
    directory = manifest_path.parent
    manifest = json.loads(manifest_path.read_text())
    entries = manifest["files"]
    # Storage identities authenticate bytes, not paths. Validate every member
    # before reads or writes so compaction cannot move/delete another bundle.
    names = set()
    for entry in entries:
        name = safe_relative(entry["path"])
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
        raw = path.read_bytes()
        if len(raw) != entry["bytes"] or digest(raw) != entry["sha256"]:
            raise ValueError(f"publication identity mismatch: {path}")
        if str(entry["path"]).endswith(".json.gz"):
            json.loads(gzip.decompress(raw))
        contents[entry["path"]] = raw

    # Select remaining plain members individually. A publication may already
    # contain packed members without having compacted every eligible JSON file.
    candidates = {
        entry["path"]: entry
        for entry in entries
        if entry["role"] in ROLES
        and entry["path"].endswith(".json")
        and entry["bytes"] >= THRESHOLD
    }
    if not candidates:
        return []

    # Resolve all JSON reference dependencies before committing. Sample records
    # can themselves contain record_parts; updating only the evidence envelope
    # would leave them pointing at removed plain companions. Resolve leaves
    # first so every parent binds the final stored child bytes.
    decoded = {}
    for name, raw in contents.items():
        if name.endswith((".json", ".json.gz")):
            value = json.loads(gzip.decompress(raw) if name.endswith(".gz") else raw)
            if isinstance(value, dict):
                decoded[name] = value
    roles = {entry["path"]: entry["role"] for entry in entries}
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
        if value is not None:
            references = [entry["path"] for entry in value.get("attachments", [])]
            references.extend(
                entry["path"]
                for part in value.get("record_parts", {}).values()
                for entry in part
            )
            parent = Path(old).parent
            replacements = {}
            for raw_reference in references:
                reference = safe_relative(raw_reference)
                child = (parent / reference).as_posix()
                if child not in contents:
                    continue
                new_child, data = rewrite(child)
                if new_child != child or data != contents[child]:
                    replacements[reference] = (
                        Path(new_child).relative_to(parent).as_posix(),
                        data,
                    )
            changed = _update_storage_references(value, replacements)
        data = json_bytes(value) if changed else raw
        pack_plain = (
            old.endswith(".json") and roles[old] in ROLES and len(data) >= THRESHOLD
        )
        new = old + ".gz" if pack_plain else old
        if pack_plain or (changed and old.endswith(".json.gz")):
            data = compressed(data)
        rewritten[old] = (new, data)
        visiting.remove(old)
        return new, data

    changes: list[tuple[str, str, bytes]] = []
    for entry in entries:
        old = entry["path"]
        new, data = rewrite(old)
        if new != old or data != contents[old]:
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
            raise FileExistsError(target)
    if check:
        return changes

    _commit_publication(manifest_path, manifest, changes)
    return changes


def update_legacy_review(changes: list[tuple[str, str, bytes]]) -> None:
    review = json.loads(REVIEW.read_text())
    rows = {row["path"]: row for row in review["files"]}
    for old, new, data in changes:
        row = rows.pop(old, None)
        if len(data) >= review["threshold_bytes"]:
            if row is None:
                raise ValueError(
                    f"missing legacy review row for compressed evidence: {old}"
                )
            row.update(path=new, bytes=len(data), sha256=digest(data))
            rows[new] = row
    review["files"] = sorted(rows.values(), key=lambda row: row["path"])
    REVIEW.write_bytes(json_bytes(review))


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check",
        action="store_true",
        help="fail if the compaction would modify the checkout",
    )
    parser.add_argument(
        "--publication",
        action="append",
        help="explicit reviewed publication path (repeatable; default: tracked publications)",
    )
    args = parser.parse_args(argv)
    changes: list[tuple[str, str, bytes]] = []
    for target in args.publication or publication_paths():
        changes.extend(compact_publication(target, check=args.check))
    if args.check and changes:
        raise SystemExit("retained evidence JSON is not compacted")
    if changes:
        update_legacy_review(changes)
    print(
        f"compressed evidence publications: {len(changes)} changed files"
        if changes
        else "compressed evidence publications: up to date"
    )


if __name__ == "__main__":
    main()
