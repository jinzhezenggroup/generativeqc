"""Fail-open persistent cache for deterministic generated-source build outputs."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any

_CACHE_ENV = "GENERATIVEQC_CODEGEN_CACHE"
_CACHE_SCHEMA = "generativeqc.codegen-source-cache/v1"
_CACHE_KEEP_PER_INVOCATION = 4


def _file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_hash(payload: Any) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _cache_root() -> Path | None:
    value = os.environ.get(_CACHE_ENV, "").strip()
    if not value:
        return None
    return Path(value).expanduser().resolve()


def _path_identity(path: Path, source_root: Path) -> str:
    resolved = path.resolve()
    root = source_root.resolve()
    try:
        return f"source:{resolved.relative_to(root).as_posix()}"
    except ValueError:
        return f"absolute:{resolved.as_posix()}"


def _identity_path(identity: str, source_root: Path) -> Path:
    if identity.startswith("source:"):
        return source_root.resolve() / identity.removeprefix("source:")
    if identity.startswith("absolute:"):
        return Path(identity.removeprefix("absolute:"))
    raise ValueError("unknown cached dependency identity")


def _lock_dependencies(source_root: Path) -> list[Path]:
    dependencies: list[Path] = []
    for name in ("pyproject.toml", "uv.lock"):
        candidate = source_root / name
        if candidate.is_file():
            dependencies.append(candidate.resolve())
    return dependencies


def _invocation_payload(
    *,
    generator: Path,
    arguments: list[str],
    targets: list[Path],
    byproducts: list[Path],
    declared_dependencies: list[Path],
    source_root: Path,
) -> dict[str, Any]:
    dependencies = [*declared_dependencies, *_lock_dependencies(source_root)]
    return {
        "schema": _CACHE_SCHEMA,
        "generator": _path_identity(generator, source_root),
        # Keep exact paths in the first rollout. This avoids reusing source that
        # might embed a build/output path while still covering clean CI builds,
        # whose checkout and build paths stay stable across runs.
        "arguments": arguments,
        "targets": [str(path.resolve()) for path in targets],
        "byproducts": [str(path.resolve()) for path in byproducts],
        "declared_dependencies": [
            _path_identity(path, source_root) for path in dependencies
        ],
        "python": [
            sys.implementation.name,
            sys.version_info.major,
            sys.version_info.minor,
        ],
        "environment": sorted(
            (name, value)
            for name, value in os.environ.items()
            if name.startswith("GENERATIVEQC_") and name != _CACHE_ENV
        ),
    }


def _dependency_records(
    dependencies: list[Path], source_root: Path
) -> list[dict[str, str]]:
    records: list[dict[str, str]] = []
    seen: set[str] = set()
    for dependency in sorted({item.resolve() for item in dependencies}):
        if not dependency.is_file():
            raise FileNotFoundError(
                f"codegen cache dependency is missing: {dependency}"
            )
        identity = _path_identity(dependency, source_root)
        if identity in seen:
            continue
        seen.add(identity)
        records.append({"identity": identity, "sha256": _file_hash(dependency)})
    return records


def _validate_dependency_records(records: Any, source_root: Path) -> list[Path] | None:
    if not isinstance(records, list):
        return None
    resolved: list[Path] = []
    seen: set[str] = set()
    for record in records:
        if not isinstance(record, dict):
            return None
        identity = record.get("identity")
        expected = record.get("sha256")
        if not isinstance(identity, str) or not isinstance(expected, str):
            return None
        if identity in seen:
            return None
        seen.add(identity)
        try:
            path = _identity_path(identity, source_root)
        except ValueError:
            return None
        if not path.is_file():
            return None
        try:
            if _file_hash(path) != expected:
                return None
        except OSError:
            return None
        resolved.append(path.resolve())
    return resolved


def _artifact_filename(kind: str, index: int, path: Path) -> str:
    suffix = "".join(path.suffixes[-2:]) if path.suffixes else ""
    return f"{kind}-{index:03d}{suffix}"


def _entry_artifacts(
    entry: Path,
    manifest: dict[str, Any],
    targets: list[Path],
    byproducts: list[Path],
) -> list[tuple[Path, Path]] | None:
    records = manifest.get("artifacts")
    expected_paths = [("target", index, path) for index, path in enumerate(targets)]
    expected_paths.extend(
        ("byproduct", index, path) for index, path in enumerate(byproducts)
    )
    if not isinstance(records, list) or len(records) != len(expected_paths):
        return None
    copies: list[tuple[Path, Path]] = []
    for record, (kind, index, destination) in zip(records, expected_paths, strict=True):
        if not isinstance(record, dict):
            return None
        if record.get("kind") != kind or record.get("index") != index:
            return None
        filename = record.get("filename")
        expected_digest = record.get("sha256")
        if not isinstance(filename, str) or not isinstance(expected_digest, str):
            return None
        source = entry / filename
        if not source.is_file():
            return None
        try:
            if _file_hash(source) != expected_digest:
                return None
        except OSError:
            return None
        copies.append((source, destination))
    return copies


def _invocation_directory(cache_root: Path, invocation: dict[str, Any]) -> Path:
    invocation_key = _canonical_hash(invocation)
    return (
        cache_root
        / _CACHE_SCHEMA.rsplit("/", 1)[-1]
        / invocation_key[:2]
        / invocation_key
    )


def _restore(
    *,
    cache_root: Path,
    invocation: dict[str, Any],
    source_root: Path,
    targets: list[Path],
    byproducts: list[Path],
) -> tuple[bool, list[Path]]:
    directory = _invocation_directory(cache_root, invocation)
    if not directory.is_dir():
        return False, []
    try:
        candidates = sorted(
            (path for path in directory.iterdir() if path.is_dir()),
            key=lambda path: path.stat().st_mtime_ns,
            reverse=True,
        )
    except OSError:
        return False, []
    for entry in candidates:
        try:
            manifest = json.loads((entry / "manifest.json").read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError):
            continue
        if not isinstance(manifest, dict) or manifest.get("invocation") != invocation:
            continue
        dependencies = _validate_dependency_records(
            manifest.get("dependencies"), source_root
        )
        if dependencies is None:
            continue
        depfile_identities = manifest.get("depfile_dependencies")
        if not isinstance(depfile_identities, list) or any(
            not isinstance(identity, str) for identity in depfile_identities
        ):
            continue
        dependency_map = {
            record["identity"]: path
            for record, path in zip(manifest["dependencies"], dependencies, strict=True)
        }
        try:
            depfile_dependencies = [
                dependency_map[identity] for identity in depfile_identities
            ]
        except KeyError:
            continue
        artifacts = _entry_artifacts(entry, manifest, targets, byproducts)
        if artifacts is None:
            continue
        temporaries: list[tuple[Path, Path]] = []
        try:
            for source, destination in artifacts:
                destination.parent.mkdir(parents=True, exist_ok=True)
                descriptor, temporary_name = tempfile.mkstemp(
                    prefix=f".{destination.name}.codegen-cache-",
                    dir=destination.parent,
                )
                os.close(descriptor)
                temporary = Path(temporary_name)
                shutil.copyfile(source, temporary)
                temporaries.append((temporary, destination))
            for temporary, destination in temporaries:
                temporary.replace(destination)
        finally:
            for temporary, _ in temporaries:
                temporary.unlink(missing_ok=True)
        return True, depfile_dependencies
    return False, []


def restore_codegen_outputs(
    *,
    generator: Path,
    arguments: list[str],
    targets: list[Path],
    byproducts: list[Path],
    declared_dependencies: list[Path],
    source_root: Path,
) -> tuple[bool, list[Path]]:
    """Restore verified outputs and return their recorded depfile dependencies."""

    cache_root = _cache_root()
    if cache_root is None:
        return False, []
    invocation = _invocation_payload(
        generator=generator,
        arguments=arguments,
        targets=targets,
        byproducts=byproducts,
        declared_dependencies=declared_dependencies,
        source_root=source_root,
    )
    try:
        return _restore(
            cache_root=cache_root,
            invocation=invocation,
            source_root=source_root,
            targets=targets,
            byproducts=byproducts,
        )
    except (OSError, ValueError):
        return False, []


def _store(
    *,
    cache_root: Path,
    invocation: dict[str, Any],
    source_root: Path,
    targets: list[Path],
    byproducts: list[Path],
    dependencies: list[Path],
    depfile_dependencies: list[Path],
) -> None:
    all_artifacts = [("target", index, path) for index, path in enumerate(targets)]
    all_artifacts.extend(
        ("byproduct", index, path) for index, path in enumerate(byproducts)
    )
    if any(not path.is_file() for _, _, path in all_artifacts):
        return
    records = _dependency_records(dependencies, source_root)
    dependency_key = _canonical_hash(records)
    directory = _invocation_directory(cache_root, invocation)
    entry = directory / dependency_key
    directory.mkdir(parents=True, exist_ok=True)
    if entry.is_dir():
        # Lookup already rejected this exact dependency identity, so replace the
        # corrupt/incomplete entry after successful regeneration.
        shutil.rmtree(entry, ignore_errors=True)
    with tempfile.TemporaryDirectory(
        prefix=".pending-", dir=directory
    ) as temporary_name:
        staged = Path(temporary_name)
        artifact_records: list[dict[str, Any]] = []
        for kind, index, source in all_artifacts:
            filename = _artifact_filename(kind, index, source)
            destination = staged / filename
            shutil.copyfile(source, destination)
            artifact_records.append(
                {
                    "kind": kind,
                    "index": index,
                    "filename": filename,
                    "sha256": _file_hash(destination),
                }
            )
        manifest = {
            "schema": _CACHE_SCHEMA,
            "invocation": invocation,
            "dependencies": records,
            "depfile_dependencies": [
                _path_identity(path, source_root) for path in depfile_dependencies
            ],
            "artifacts": artifact_records,
        }
        (staged / "manifest.json").write_text(
            json.dumps(manifest, sort_keys=True, separators=(",", ":")) + "\n",
            encoding="utf-8",
        )
        try:
            os.rename(staged, entry)
        except OSError:
            if not entry.is_dir():
                raise
    try:
        entries = sorted(
            (path for path in directory.iterdir() if path.is_dir()),
            key=lambda path: path.stat().st_mtime_ns,
            reverse=True,
        )
    except OSError:
        return
    for stale in entries[_CACHE_KEEP_PER_INVOCATION:]:
        shutil.rmtree(stale, ignore_errors=True)


def store_codegen_outputs(
    *,
    generator: Path,
    arguments: list[str],
    targets: list[Path],
    byproducts: list[Path],
    declared_dependencies: list[Path],
    dynamic_dependencies: list[Path],
    runner: Path,
    source_root: Path,
) -> None:
    """Persist one successful deterministic generation without gating the build."""

    cache_root = _cache_root()
    if cache_root is None:
        return
    lock_dependencies = _lock_dependencies(source_root)
    invocation = _invocation_payload(
        generator=generator,
        arguments=arguments,
        targets=targets,
        byproducts=byproducts,
        declared_dependencies=declared_dependencies,
        source_root=source_root,
    )
    dependencies = [
        *dynamic_dependencies,
        *declared_dependencies,
        *lock_dependencies,
        generator,
        runner.resolve(),
        Path(__file__).resolve(),
    ]
    try:
        _store(
            cache_root=cache_root,
            invocation=invocation,
            source_root=source_root,
            targets=targets,
            byproducts=byproducts,
            dependencies=dependencies,
            depfile_dependencies=dynamic_dependencies,
        )
    except (OSError, ValueError):
        # Cache failures are performance misses, never build failures.
        return
