"""Source-bound review gate for native high-rank output materialization sites.

Reuses the complexity classifier and shared strict source/provenance boundary.
A passing review gate is not a correctness, required-storage, or speedup proof.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import re
import subprocess
import tarfile
import tempfile
from collections import Counter
from dataclasses import asdict
from pathlib import Path
from types import FunctionType
from typing import Any

try:
    from tools.audit_native_complexity import _mask_comments_and_literals, audit_text
    from tools.audit_native_work import _functions
    from tools.audit_structured_materialization import audit_tree as structured_audit
except ModuleNotFoundError:
    from audit_native_complexity import _mask_comments_and_literals, audit_text
    from audit_native_work import _functions
    from audit_structured_materialization import audit_tree as structured_audit

_SOURCE_SHA256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()

CLASSIFICATION = "high-rank-output-materialization"
INVENTORY_SCHEMA = "generativeqc.materialization-candidates.v1"
MANIFEST_SCHEMA = "generativeqc.materialization-dispositions.v1"
MANIFEST_PATH = "manifests/native_materialization_dispositions.json"
DISPOSITIONS = {
    "retained-pending-evidence",
    "producer-fusable",
    "consumer-fusable",
    "view/index-only",
    "required-materialization",
}


class ReviewError(ValueError):
    """Missing or malformed evidence cannot earn a passing comparison."""


def _digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def source_identity(text: str) -> str:
    """Normalize only ordinary lexical source; unsupported forms keep exact bytes.

    Preprocessing has line/comment/header semantics beyond the existing lexer.
    Even directive spellings inside comments/literals conservatively select exact
    source, avoiding another parser or a claimed preprocessing equivalence.
    """
    if any(marker in text for marker in ("#", "%:", "??=")):
        return _digest(["preprocessed-source", text])
    if "\0" in text or re.search(r'R"|\\\r?\n|\?\?/', text):
        return _digest(["exact-source", text])
    masked = _mask_comments_and_literals(text, literal_mask="\0")
    if re.search(
        r"\b(?:__LINE__|__builtin_LINE|__builtin_COLUMN|source_location)\b", masked
    ):
        return _digest(["line-sensitive-source", text])
    pieces = []
    for match in re.finditer(r"\0+|[^\0]+", masked):
        segment = match.group()
        if segment.startswith("\0"):
            pieces.append(["literal", text[match.start() : match.end()]])
        elif compact := " ".join(segment.split()):
            pieces.append(["code", compact])
    return _digest(pieces)


def _site_id(row: dict[str, Any]) -> str:
    return _digest(
        [row[key] for key in ("path", "function", "lhs", "variables", "occurrence")]
    )


def _bind_analyzers() -> Any:
    # Freeze callable objects and import-time identities, not mutable module
    # globals. A reload mutates a retained function's __globals__ in place.
    bindings = tuple(
        (
            name,
            function,
            function.__globals__,
            Path(function.__globals__["__file__"]),
            function.__globals__["_SOURCE_SHA256"],
        )
        for name, function in (
            ("_mask_comments_and_literals", _mask_comments_and_literals),
            ("audit_text", audit_text),
            ("_functions", _functions),
            ("structured_audit", structured_audit),
        )
    )
    # Also freeze imported helper links inside those analyzers. A dependency
    # may have been reloaded before this gate imports an older dependent module.
    namespaces = {id(binding[2]): binding[2] for binding in bindings}
    paths = {binding[3] for binding in bindings}
    helpers = tuple(
        (namespace, name, function, function.__globals__)
        for namespace in namespaces.values()
        for name, function in namespace.items()
        if isinstance(function, FunctionType)
        and Path(function.__globals__.get("__file__", "")) in paths
    )
    own_path, own_digest = Path(__file__), _SOURCE_SHA256

    def verify() -> dict[str, str]:
        hashes = {own_path.name: own_digest}
        if hashlib.sha256(own_path.read_bytes()).hexdigest() != own_digest:
            raise ReviewError("review analyzer changed after import; restart the audit")
        for name, function, namespace, path, imported in bindings:
            if (
                globals().get(name) is not function
                or namespace.get(function.__name__) is not function
                or namespace.get("_SOURCE_SHA256") != imported
                or hashlib.sha256(path.read_bytes()).hexdigest() != imported
            ):
                raise ReviewError(
                    f"analyzer changed after import: {path.name}; restart the audit"
                )
            hashes[path.name] = imported
        for namespace, name, function, origin in helpers:
            if (
                namespace.get(name) is not function
                or origin.get(function.__name__) is not function
            ):
                raise ReviewError(
                    f"analyzer helper changed after import: {name}; restart the audit"
                )
        return hashes

    return verify


_analyzer_hashes = _bind_analyzers()


def inventory(root: Path, *, _verify: Any = _analyzer_hashes) -> dict[str, Any]:
    analyzers = _verify()
    findings = []
    identities: dict[str, str] = {}
    counts: Counter[str] = Counter()
    occurrences: Counter[str] = Counter()
    scanned = 0

    def visit(path: str, data: bytes) -> None:
        nonlocal scanned
        text = data.decode("utf-8", errors="strict")
        identities[path] = source_identity(text)
        # Match the established complexity-audit production scope exactly.
        if not path.startswith("src/"):
            return
        scanned += 1
        clean = _mask_comments_and_literals(text)
        functions = _functions(clean)
        for finding in audit_text(text, path=path):
            counts[finding.classification] += 1
            if finding.classification != CLASSIFICATION:
                continue
            line_start = sum(
                len(line) for line in text.splitlines(keepends=True)[: finding.line - 1]
            )
            owners = [f for f in functions if f.body <= line_start < f.end]
            if len(owners) != 1:
                raise ReviewError(f"ambiguous function owner: {path}:{finding.line}")
            owner = owners[0]
            row = asdict(finding)
            row["variables"] = list(finding.variables)
            row["function"] = (
                "::".join((*owner.namespace, owner.name))
                + "("
                + " ".join(owner.parameters.split())
                + ")"
            )
            row["lhs"] = " ".join((finding.lhs or "").split())
            row["occurrence"] = 0
            anchor = _site_id(row)
            occurrences[anchor] += 1
            row["occurrence"] = occurrences[anchor]
            row["candidate_id"] = _site_id(row)
            row["source_identity"] = identities[path]
            findings.append(row)

    shared = structured_audit(root, ("src", "include"), _source_visitor=visit)
    provenance = shared["provenance"]
    if _verify() != analyzers:
        raise ReviewError("analyzers changed during inventory")
    provenance["review_scanner_sha256"] = _SOURCE_SHA256
    provenance["consumed_analyzer_hashes"] = analyzers
    return {
        "schema": INVENTORY_SCHEMA,
        "classification": CLASSIFICATION,
        "provenance": provenance,
        "source_identities": identities,
        "scanned_files": scanned,
        "classification_counts": dict(sorted(counts.items())),
        "candidate_count": len(findings),
        "candidates": findings,
    }


def _hash(value: Any) -> bool:
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def _validate_inventory(report: dict[str, Any]) -> dict[str, dict[str, Any]]:
    try:
        if (
            report["schema"] != INVENTORY_SCHEMA
            or report["classification"] != CLASSIFICATION
        ):
            raise ReviewError("unsupported candidate inventory schema or scope")
        sources = report["source_identities"]
        provenance = report["provenance"]
        if type(report["scanned_files"]) is not int or report["scanned_files"] <= 0:
            raise ReviewError("missing native source census")
        if provenance["alias_topology_unverified"]:
            raise ReviewError("source alias topology is unverified")
        for key in ("scanner_digest", "scanned_source_digest"):
            if not _hash(provenance[key]):
                raise ReviewError("missing source/scanner provenance")
        for mapping, digest in (
            ("source_hashes", "scanned_source_digest"),
            ("scanner_hashes", "scanner_digest"),
        ):
            hashes = provenance[mapping]
            if (
                not hashes
                or not all(_hash(value) for value in hashes.values())
                or _digest(hashes) != provenance[digest]
            ):
                raise ReviewError("malformed source/scanner digest")
        consumed = provenance["consumed_analyzer_hashes"]
        required = {
            "ratchet_native_materialization.py",
            "audit_native_complexity.py",
            "audit_native_work.py",
            "audit_structured_materialization.py",
        }
        if set(consumed) != required or not all(
            _hash(value) for value in consumed.values()
        ):
            raise ReviewError("missing consumed analyzer identities")
        if (
            consumed["ratchet_native_materialization.py"]
            != provenance["review_scanner_sha256"]
        ):
            raise ReviewError("review analyzer identity mismatch")
        if any(
            provenance["scanner_hashes"][name] != digest
            for name, digest in consumed.items()
            if name != "ratchet_native_materialization.py"
        ):
            raise ReviewError("consumed analyzer identity mismatch")
        if not sources or not all(_hash(value) for value in sources.values()):
            raise ReviewError("missing or malformed source identities")
        if set(sources) != set(provenance["source_hashes"]):
            raise ReviewError("source provenance and normalized identities disagree")
        if provenance["source_roots"] != ["src", "include"]:
            raise ReviewError("candidate inventory source scope changed")
        if not _hash(provenance["review_scanner_sha256"]):
            raise ReviewError("missing review scanner identity")
        if report["scanned_files"] != sum(path.startswith("src/") for path in sources):
            raise ReviewError("source file census disagrees with inventory")
        rows = report["candidates"]
        if not isinstance(rows, list):
            raise ReviewError("malformed candidate list")
        if type(report["candidate_count"]) is not int or report[
            "candidate_count"
        ] != len(rows):
            raise ReviewError("candidate count disagrees with inventory")
        if report["classification_counts"].get(CLASSIFICATION, 0) != len(rows):
            raise ReviewError("classification count disagrees with inventory")
        indexed = {}
        for row in rows:
            if any(
                not isinstance(row[key], str) or not row[key].strip()
                for key in ("path", "function", "lhs")
            ) or not row["path"].startswith("src/"):
                raise ReviewError("malformed candidate source anchor")
            if any(
                type(row[key]) is not int or row[key] <= 0
                for key in ("line", "occurrence")
            ):
                raise ReviewError("malformed candidate source location")
            identity = row["candidate_id"]
            if identity != _site_id(row) or identity in indexed:
                raise ReviewError("invalid or duplicate candidate identity")
            if (
                row["classification"] != CLASSIFICATION
                or row["source_identity"] != sources[row["path"]]
            ):
                raise ReviewError("candidate is not bound to its source")
            indexed[identity] = row
        return indexed
    except (KeyError, TypeError, AttributeError) as error:
        raise ReviewError("malformed candidate inventory") from error


def _manifest_rows(manifest: dict[str, Any]) -> list[dict[str, Any]]:
    if not isinstance(manifest, dict) or manifest.get("schema") != MANIFEST_SCHEMA:
        raise ReviewError("missing or unsupported disposition manifest")
    rows = manifest.get("candidates")
    if not isinstance(rows, list) or any(not isinstance(row, dict) for row in rows):
        raise ReviewError("malformed disposition manifest candidates")
    return rows


def compare(
    baseline: dict[str, Any],
    candidate: dict[str, Any],
    manifest: dict[str, Any],
    *,
    _verify: Any = _analyzer_hashes,
) -> dict[str, Any]:
    _verify()
    before = _validate_inventory(baseline)
    after = _validate_inventory(candidate)
    for key in ("scanner_digest", "review_scanner_sha256", "consumed_analyzer_hashes"):
        if baseline["provenance"][key] != candidate["provenance"][key]:
            raise ReviewError("baseline and candidate must use the same analyzer")
    errors = []
    reviewed = {}
    for record in _manifest_rows(manifest):
        identity = record.get("candidate_id")
        if not isinstance(identity, str) or identity not in after:
            errors.append(
                {
                    "candidate_id": identity,
                    "reason": "unmatched/stale disposition identity",
                }
            )
            continue
        if identity in reviewed:
            errors.append(
                {"candidate_id": identity, "reason": "duplicate disposition identity"}
            )
            continue
        reviewed[identity] = record
        for field in ("owner", "reason", "residency", "lifetime", "resource_owner"):
            if not isinstance(record.get(field), str) or not record[field].strip():
                errors.append(
                    {"candidate_id": identity, "reason": f"missing explicit {field}"}
                )
        if (
            not isinstance(record.get("disposition"), str)
            or record["disposition"] not in DISPOSITIONS
        ):
            errors.append({"candidate_id": identity, "reason": "unknown disposition"})
        for field in ("evidence_gaps",):
            value = record.get(field)
            if (
                not isinstance(value, list)
                or not value
                or not all(isinstance(item, str) and item.strip() for item in value)
            ):
                errors.append(
                    {"candidate_id": identity, "reason": f"missing explicit {field}"}
                )
        bindings = record.get("source_bindings")
        if not isinstance(bindings, dict) or after[identity]["path"] not in bindings:
            errors.append(
                {"candidate_id": identity, "reason": "missing candidate source binding"}
            )
            continue
        producer = record.get("producer")
        consumers = record.get("consumers")
        references = [producer, *(consumers if isinstance(consumers, list) else [])]
        if not isinstance(consumers, list) or not consumers:
            errors.append(
                {"candidate_id": identity, "reason": "missing explicit consumers"}
            )
        for reference in references:
            if (
                not isinstance(reference, dict)
                or not isinstance(reference.get("symbol"), str)
                or not reference["symbol"].strip()
                or not isinstance(reference.get("path"), str)
                or reference["path"] not in bindings
            ):
                errors.append(
                    {
                        "candidate_id": identity,
                        "reason": "producer/consumer must name a symbol and bound source path",
                    }
                )
        for path, digest in bindings.items():
            if not _hash(digest) or candidate["source_identities"].get(path) != digest:
                errors.append(
                    {
                        "candidate_id": identity,
                        "reason": f"stale/unmatched producer-consumer source binding: {path}",
                    }
                )
    for identity in after.keys() - reviewed.keys():
        errors.append(
            {
                "candidate_id": identity,
                "reason": "candidate needs explicit owner and disposition",
            }
        )
    common = before.keys() & after.keys()
    changed = []
    unchanged = []
    for identity in sorted(common):
        bindings = reviewed.get(identity, {}).get("source_bindings", {})
        paths = {
            after[identity]["path"],
            *(bindings if isinstance(bindings, dict) else {}),
        }
        target = (
            changed
            if any(
                baseline["source_identities"].get(path)
                != candidate["source_identities"].get(path)
                for path in paths
            )
            else unchanged
        )
        target.append(identity)
    result = {
        "schema": "generativeqc.materialization-review-ci.v1",
        "status": "NEEDS_REVIEW" if errors else "PASS",
        "scope": "Source-bound candidate review only; no numerical, required-storage, retirement or profitability proof.",
        "baseline": baseline,
        "candidate": candidate,
        "dispositions": manifest,
        "counts": {
            "baseline": len(before),
            "candidate": len(after),
            "delta": len(after) - len(before),
        },
        "added": [after[key] for key in sorted(after.keys() - before.keys())],
        "removed": [before[key] for key in sorted(before.keys() - after.keys())],
        "changed": [{"before": before[key], "after": after[key]} for key in changed],
        "unchanged": unchanged,
        "review_errors": errors,
        "limitations": [
            "The complexity classifier is lexical; absence of a finding is not proof a pass was retired.",
            "Producer/consumer symbols are reviewer-declared links; their existence and call binding are not verified.",
            "Bindings cover declared source files, not whole-program dispatch, headers or transitive callers unless listed.",
            "Whole-file context changes conservatively require renewed disposition bindings.",
            "Human review records are versioned claims, not evidence that fusion is unsafe or slower.",
        ],
    }

    _verify()
    return result


def _git(root: Path, *arguments: str) -> bytes:
    result = subprocess.run(
        ["git", "-C", str(root), *arguments], capture_output=True, check=False
    )
    if result.returncode:
        raise ReviewError(
            f"git {arguments[0]} failed: {result.stderr.decode(errors='replace').strip()}"
        )
    return result.stdout


def _baseline_tree(root: Path, sha: str, target: Path) -> None:
    # Never execute baseline code; both inventories use this process's analyzers.
    expected = {}
    for entry in _git(root, "ls-tree", "-rz", sha, "--", "src", "include").split(b"\0"):
        if not entry:
            continue
        metadata, name = entry.split(b"\t", 1)
        mode, kind, identity = metadata.split()
        if kind != b"blob" or mode not in {b"100644", b"100755"}:
            raise ReviewError("unsupported baseline source entry")
        expected[name.decode("utf-8")] = identity.decode("ascii")
    seen = set()
    data = _git(root, "archive", "--format=tar", sha, "src", "include")
    with tarfile.open(fileobj=io.BytesIO(data)) as archive:
        for member in archive:
            relative = Path(member.name)
            if (
                relative.is_absolute()
                or ".." in relative.parts
                or relative.parts[0] not in {"src", "include"}
            ):
                raise ReviewError("unsafe baseline archive member")
            if member.isdir():
                continue
            if not member.isfile():
                raise ReviewError(f"unsupported baseline source entry: {member.name}")
            stream = archive.extractfile(member)
            if stream is None:
                raise ReviewError(f"missing baseline source bytes: {member.name}")
            content = stream.read()
            # Archive export-ignore/export-subst attributes must not alter the
            # source census or bytes behind an immutable-baseline claim.
            blob = b"blob " + str(len(content)).encode() + b"\0" + content
            identity = hashlib.sha1(blob, usedforsecurity=False).hexdigest()
            if expected.get(member.name) != identity:
                raise ReviewError(
                    f"baseline archive differs from Git blob: {member.name}"
                )
            seen.add(member.name)
            path = target / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(content)
    if seen != expected.keys():
        raise ReviewError("baseline archive omits source files from the Git tree")


def audit(
    root: Path, base_sha: str, manifest_path: Path, *, _verify: Any = _analyzer_hashes
) -> dict[str, Any]:
    _verify()
    if not re.fullmatch(r"[0-9a-fA-F]{40}", base_sha):
        raise ReviewError("base SHA must be a full 40-character commit digest")
    actual = _git(root, "rev-parse", f"{base_sha}^{{commit}}").decode().strip()
    if actual.lower() != base_sha.lower():
        raise ReviewError("baseline does not identify the requested commit")
    manifest_bytes = manifest_path.read_bytes()
    manifest = json.loads(manifest_bytes)
    with tempfile.TemporaryDirectory(prefix="gqc-materialization-base-") as temporary:
        baseline_root = Path(temporary)
        _baseline_tree(root, actual, baseline_root)
        baseline = inventory(baseline_root)
        baseline["provenance"]["commit"] = actual
        baseline["provenance"]["tree"] = (
            _git(root, "rev-parse", f"{actual}^{{tree}}").decode().strip()
        )
        baseline["provenance"]["source_content_basis"] = "git-archive-at-base-commit"
        candidate = inventory(root)
    report = compare(baseline, candidate, manifest)
    report["base_commit"] = actual
    report["manifest_sha256"] = hashlib.sha256(manifest_bytes).hexdigest()
    _verify()
    return report


def main(argv: list[str] | None = None, *, _verify: Any = _analyzer_hashes) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--base-sha")
    parser.add_argument("--manifest", type=Path, default=Path(MANIFEST_PATH))
    parser.add_argument("--inventory-only", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--fail-on-unreviewed", action="store_true")
    args = parser.parse_args(argv)
    try:
        _verify()
        if args.inventory_only:
            report = inventory(args.root)
        elif args.base_sha is None:
            raise ReviewError("baseline is required; use --base-sha for a comparison")
        else:
            manifest_path = (
                args.manifest
                if args.manifest.is_absolute()
                else args.root / args.manifest
            )
            report = audit(args.root, args.base_sha, manifest_path)
        _verify()
    except (
        ReviewError,
        OSError,
        ValueError,
        TypeError,
        KeyError,
        tarfile.TarError,
    ) as error:
        report = {
            "schema": "generativeqc.materialization-review-ci.v1",
            "status": "INCOMPLETE",
            "reason": str(error),
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    status = report.get("status", "INVENTORY_ONLY")
    print(f"Native materialization review: {status}")
    if "counts" in report:
        print(json.dumps(report["counts"], sort_keys=True))
        for kind in ("added", "removed", "changed"):
            print(f"{kind}: {len(report[kind])}")
        for error in report["review_errors"]:
            print(f"  {error['candidate_id']}: {error['reason']}")
    if "reason" in report:
        print(report["reason"])
    if args.fail_on_unreviewed and (args.inventory_only or status != "PASS"):
        return 2 if status in {"INCOMPLETE", "INVENTORY_ONLY"} else 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
