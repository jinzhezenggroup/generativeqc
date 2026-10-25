"""Read-only file/mode/link admission against the original environment metadata."""

from __future__ import annotations

import hashlib
import json
import os
import stat
from pathlib import Path, PurePosixPath

from transport import checked


def safe_relative(value: str) -> PurePosixPath:
    path = PurePosixPath(value)
    if path.is_absolute() or ".." in path.parts or not path.parts or "\\" in value:
        raise ValueError("invalid relative input identity path")
    if path.name in (
        ".env",
        "auth.json",
        "token",
        "token.json",
    ) or path.name.startswith(".env."):
        raise ValueError("credential files are not read by the identity gate")
    return path


def qualified_addition(
    root: Path,
    rows: list,
    additions: dict,
    proof: dict,
    base_sha: str | None,
    proof_sha: str | None,
) -> list:
    """Admit one exact, independently inspected cache derivative, never a pattern."""
    if (
        additions.get("schema") != "generativeqc.environment-derived-additions.v1"
        or not base_sha
        or additions.get("base_manifest_sha256") != base_sha
        or not proof_sha
        or additions.get("derivation_proof_sha256") != proof_sha
    ):
        raise ValueError("derived addition base/proof identity differs")
    extra = additions.get("added_files", [])
    if not isinstance(extra, list) or len(extra) != 1:
        raise ValueError("this cohort qualifies exactly one added cache file")
    added = extra[0]
    original = {row["path"]: row for row in rows}
    source_relative = safe_relative(proof["source_path"])
    cache_relative = safe_relative(added["path"])
    expected_cache = (
        source_relative.parent
        / "__pycache__"
        / (source_relative.stem + ".cpython-311.pyc")
    )
    source_row = original.get(source_relative.as_posix())
    if (
        source_row is None
        or added["path"] in original
        or cache_relative != expected_cache
        or proof.get("path") != added["path"]
        or proof.get("bytes") != added["bytes"]
        or proof.get("sha256") != added["sha256"]
        or proof.get("source_bytes") != source_row["bytes"]
        or proof.get("source_sha256") != source_row["sha256"]
    ):
        raise ValueError(
            "derived cache is not bound to an unchanged original source row"
        )
    required_true = (
        "source_matches_original_manifest",
        "code_objects_equal",
        "marshal_payload_exact",
        "cache_bytes_unchanged_after",
        "source_bytes_unchanged_after",
    )
    if (
        any(proof.get(key) is not True for key in required_true)
        or type(proof.get("header_flags")) is not int
        or proof["header_flags"] != 0
        or type(proof.get("no_site")) is not int
        or proof["no_site"] != 1
        or type(proof.get("isolated")) is not int
        or proof["isolated"] != 1
        or proof.get("distutils_hack_imported") is not False
        or proof.get("header_source_mtime") != proof.get("actual_source_mtime")
        or proof.get("header_source_size") != source_row["bytes"]
        or proof.get("actual_source_size") != source_row["bytes"]
        or proof.get("co_filename") != str(root / str(source_relative))
        or proof.get("expected_filename") != str(root / str(source_relative))
    ):
        raise ValueError(
            "derived cache inspection did not establish exact source/code equivalence"
        )
    actual_mtime = int((root / str(source_relative)).stat().st_mtime) & 0xFFFFFFFF
    if (
        type(proof.get("header_source_mtime")) is not int
        or actual_mtime != proof["header_source_mtime"]
    ):
        raise ValueError(
            "current source timestamp differs from qualified PEP552 pairing"
        )
    return extra


def verify(
    root: Path,
    manifest: dict,
    external: dict,
    *,
    additions: dict | None = None,
    base_manifest_sha256: str | None = None,
    derivation_proof: dict | None = None,
    derivation_proof_sha256: str | None = None,
) -> dict:
    root = root.resolve(strict=True)
    rows = manifest["verified_file_rows"]
    if (
        manifest["status"] != "PASS"
        or manifest["mismatches"]
        or len(rows) != manifest["regular_files"]
        or sum(r["bytes"] for r in rows) != manifest["regular_bytes"]
    ):
        raise ValueError("original readback metadata does not close")
    extra = []
    if additions is not None:
        if derivation_proof is None:
            raise ValueError("derived cache has no independently hash-bound proof")
        extra = qualified_addition(
            root,
            rows,
            additions,
            derivation_proof,
            base_manifest_sha256,
            derivation_proof_sha256,
        )
    all_rows = [*rows, *extra]
    expected = set()
    for row in all_rows:
        relative = safe_relative(row["path"]).as_posix()
        if relative in expected:
            raise ValueError("duplicate file identity")
        expected.add(relative)
        path = root / relative
        if path.is_symlink() or not path.resolve(strict=True).is_relative_to(root):
            raise ValueError("regular input escaped through a symlink")
        metadata = path.lstat()
        if (
            not stat.S_ISREG(metadata.st_mode)
            or metadata.st_size != row["bytes"]
            or oct(stat.S_IMODE(metadata.st_mode)) != row["mode"]
        ):
            raise ValueError("original input file size/mode changed: " + relative)
        checked(path, row["sha256"])
    links = manifest["symlinks"]
    if len(links) != manifest["symlink_count"] or len(
        {r["path"] for r in links}
    ) != len(links):
        raise ValueError("original symlink inventory does not close")
    for row in links:
        path = root / str(safe_relative(row["path"]))
        if (
            not path.is_symlink()
            or os.readlink(path) != row["target"]
            or str(path.resolve(strict=True)) != row["resolved"]
            or oct(stat.S_IMODE(path.lstat().st_mode)) != row["mode"]
        ):
            raise ValueError("original input symlink identity changed")
    for scope in ("source/cbs-qualified", "source/hf-dft-cbd4", "env"):
        wanted = {p for p in expected if p.startswith(scope + "/")}
        if not wanted:
            continue
        actual = {
            p.relative_to(root).as_posix()
            for p in (root / scope).rglob("*")
            if p.is_file() and not p.is_symlink()
        }
        if wanted != actual:
            raise ValueError(
                "frozen input scope has missing or additional regular files: " + scope
            )
        wanted_links = {r["path"] for r in links if r["path"].startswith(scope + "/")}
        actual_links = {
            p.relative_to(root).as_posix()
            for p in (root / scope).rglob("*")
            if p.is_symlink()
        }
        if wanted_links != actual_links:
            raise ValueError(
                "frozen input scope has missing or additional links: " + scope
            )
    for name, row in external["files"].items():
        path = Path(name)
        if path.name in (".env", "auth.json", "token", "token.json"):
            raise ValueError("credential files are not read by external identity gate")
        if (
            str(path.resolve(strict=True)) != row["resolved"]
            or path.stat().st_size != row["bytes"]
            or oct(stat.S_IMODE(path.stat().st_mode)) != row["mode"]
            or (os.readlink(path) if path.is_symlink() else None)
            != row["symlink_target"]
        ):
            raise ValueError("frozen external dependency changed")
        checked(path, row["sha256"])
    result = {
        "status": "PASS",
        "regular_files": len(all_rows),
        "regular_bytes": manifest["regular_bytes"] + sum(row["bytes"] for row in extra),
        "base_regular_files": len(rows),
        "qualified_added_files": len(extra),
        "symlinks": len(links),
        "external_files": len(external["files"]),
        "source_closure_sha256": manifest.get("source_closure_sha256"),
        "source_mode_witness_sha256": manifest.get("source_mode_witness_sha256"),
    }
    if additions is not None:
        result["base_manifest_sha256"] = base_manifest_sha256
        result["derived_proof_sha256"] = derivation_proof_sha256
        result["qualified_input_identity_sha256"] = hashlib.sha256(
            json.dumps(
                {
                    "base_manifest_sha256": base_manifest_sha256,
                    "added_files": extra,
                    "proof_sha256": derivation_proof_sha256,
                },
                sort_keys=True,
                separators=(",", ":"),
            ).encode()
        ).hexdigest()
        result["base_source_closure_sha256"] = result.pop("source_closure_sha256")
        result["base_source_mode_witness_sha256"] = result.pop(
            "source_mode_witness_sha256"
        )
    return result
