"""Run NEW API measurements from a checksum-bound, reviewed source snapshot.

Historical measurements and their frozen source/library receipts never change.
A later reviewed source is a new experiment, not an equivalent historical run.
Without --run, this verifies source/byte identities and prints the proposed call.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import runpy
import sys
from pathlib import Path, PurePosixPath

HERE = Path(__file__).resolve().parent


def verify_publication(bundle: Path) -> dict:
    """Check all bound bytes before interpreting mappings or executing the harness."""
    manifest_bytes = (bundle / "publication.json").read_bytes()
    manifest = json.loads(manifest_bytes)
    entries = manifest["files"]
    seen = set()
    for entry in entries:
        name = entry["path"]
        relative = PurePosixPath(name)
        if relative.is_absolute() or ".." in relative.parts or name in seen:
            raise ValueError("unsafe or duplicate publication member")
        seen.add(name)
        path = (bundle / relative).resolve(strict=True)
        if not path.is_relative_to(bundle.resolve()):
            raise ValueError("publication member escapes the evidence directory")
        data = path.read_bytes()
        if (
            len(data) != entry["bytes"]
            or hashlib.sha256(data).hexdigest() != entry["sha256"]
        ):
            raise ValueError(f"publication checksum/size mismatch: {name}")
    required = {
        "run.py",
        "inputs.json",
        "provenance.json",
        "reproduction-sources.json",
        "reproduce.py",
        "new_campaign.py",
    }
    if not required <= seen:
        raise ValueError("publication is missing a required reproduction member")
    historical = json.loads((bundle / "provenance.json").read_text())
    if (
        hashlib.sha256((bundle / "run.py").read_bytes()).hexdigest()
        != historical["harness_sha256"]
    ):
        raise ValueError("frozen historical harness changed")
    if (
        hashlib.sha256((bundle / "inputs.json").read_bytes()).hexdigest()
        != historical["fixture_sha256"]
    ):
        raise ValueError("frozen historical input fixture changed")
    return {
        "publication_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
        "mapping_sha256": hashlib.sha256(
            (bundle / "reproduction-sources.json").read_bytes()
        ).hexdigest(),
    }


def select_mapping(
    snapshot: dict, mappings: dict, historical: dict, requested: str | None = None
) -> dict:
    """Admit only a complete exact source hash, never a prefix or broad exception."""
    entries = mappings["sources"]
    ids = [entry["id"] for entry in entries]
    hashes = [entry["production_source_sha256"] for entry in entries]
    if len(set(ids)) != len(ids) or len(set(hashes)) != len(hashes):
        raise ValueError("ambiguous reproduction source mappings")
    if not all(
        isinstance(value, str)
        and len(value) == 64
        and set(value) <= set("0123456789abcdef")
        for value in hashes
    ):
        raise ValueError("source mappings require complete SHA-256 hashes")
    matches = [
        entry
        for entry in entries
        if entry["production_source_sha256"] == snapshot["source_hash"]
        and (requested is None or entry["id"] == requested)
    ]
    if len(matches) != 1:
        raise ValueError("production source has no exact reviewed reproduction mapping")
    mapping = matches[0]
    if mapping["kind"] not in {"frozen-measured-source", "reviewed-follow-up-source"}:
        raise ValueError("unsupported reproduction source mapping kind")
    if mapping["kind"] == "reviewed-follow-up-source" and not mapping.get(
        "reviewed_production_changes"
    ):
        raise ValueError(
            "follow-up mapping must identify its reviewed production changes"
        )
    if not mapping.get("review_scope"):
        raise ValueError("reproduction mapping needs an explicit review scope")
    if (
        mapping["kind"] == "frozen-measured-source"
        and mapping["production_source_sha256"] != historical["source"]["source_hash"]
    ):
        raise ValueError(
            "frozen-source mapping must match the historical source receipt"
        )
    return mapping


def new_reproduction_plan(
    *,
    source: Path,
    library: Path,
    output: Path,
    revision: str | None,
    actual: dict,
    mapping: dict,
    historical: dict,
    library_hash: str,
    verified: dict,
    timeout: float,
    bundle: Path,
) -> dict:
    """Build a plan whose identities always describe the new source and library."""
    command = [
        sys.executable,
        str(bundle / "reproduce.py"),
        "--run",
        "--source-root",
        str(source),
        "--library",
        str(library),
        "--output",
        str(output),
        "--source-map",
        mapping["id"],
        "--timeout",
        str(timeout),
    ]
    if revision is not None:
        command.extend(["--build-source-revision", revision])
    return {
        "scope": "NEW reproduction; no historical timing or numerical qualification is reused",
        "historical_measurements_reused": False,
        "source_mapping_id": mapping["id"],
        "source_mapping_kind": mapping["kind"],
        "source_mapping_review_scope": mapping["review_scope"],
        "verified_production_source_hash": actual["source_hash"],
        "reproduction_source_revision": actual["revision"],
        "reproduction_source_tree": actual["tree"],
        "reproduction_checkout_dirty": actual["dirty"],
        "caller_declared_build_source": None
        if revision is None
        else {"revision": revision},
        "reproduction_library_build_revision": None,
        "reproduction_library_sha256": library_hash,
        "library_provenance_limit": "Caller-supplied binary bytes are identified; a source-compatibility check is not a compiler/build attestation",
        "historical_source_sha256": historical["source"]["source_hash"],
        "historical_library_sha256": historical["library_sha256"],
        "historical_revision_required_as_git_object": False,
        **verified,
        "source_root": str(source),
        "library_path": str(library),
        "output": str(output),
        "timeout_seconds": timeout,
        "endpoint_boundary": historical["endpoint_boundary"],
        "checkpoint_boundary": historical["checkpoint_boundary"],
        "command": command,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--library", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--source-map", help="Optionally require one named reviewed source mapping"
    )
    parser.add_argument(
        "--build-source-revision",
        help="Optional caller-declared native build source. Omit to retain a known byte-bound library receipt or explicitly unknown rebuild provenance; never inferred from current HEAD",
    )
    parser.add_argument(
        "--run",
        action="store_true",
        help="Otherwise verify provenance and print the command without loading the library or running SCF",
    )
    parser.add_argument("--timeout", type=float, default=300.0)
    args = parser.parse_args()
    if not math.isfinite(args.timeout) or args.timeout <= 0:
        parser.error("--timeout must be finite and positive")
    verified = verify_publication(HERE)
    historical = json.loads((HERE / "provenance.json").read_text())
    mappings = json.loads((HERE / "reproduction-sources.json").read_text())
    # This remains the checksum-verified exact historical driver. Its actual
    # source/library snapshots and output guards apply to every NEW experiment.
    original = runpy.run_path(str(HERE / "run.py"), run_name="frozen_api_runner")
    source = args.source_root.resolve(strict=True)
    library = args.library.resolve(strict=True)
    output = args.output.resolve()
    if output.exists() or output.is_relative_to(HERE.parent):
        raise ValueError("use a new scratch output outside the published evidence tree")
    actual = original["source_snapshot"](source)
    mapping = select_mapping(actual, mappings, historical, args.source_map)
    revision = None
    if args.build_source_revision:
        revision = (
            original["git"](source, "rev-parse", args.build_source_revision)
            .decode()
            .strip()
        )
    plan = new_reproduction_plan(
        source=source,
        library=library,
        output=output,
        revision=revision,
        actual=actual,
        mapping=mapping,
        historical=historical,
        library_hash=original["sha"](library),
        verified=verified,
        timeout=args.timeout,
        bundle=HERE,
    )
    receipts = mappings.get("native_library_receipts", [])
    known = [
        receipt
        for receipt in receipts
        if receipt["library_sha256"] == plan["reproduction_library_sha256"]
    ]
    if len(known) > 1:
        raise ValueError("ambiguous native library receipts")
    receipt = known[0] if known else None
    if receipt is not None and receipt["library_sha256"] not in mapping.get(
        "reviewed_compatible_library_sha256", []
    ):
        raise ValueError(
            "this known native library has no reviewed compatibility with the selected source"
        )
    plan["native_library_build_receipt"] = receipt
    plan["library_build_receipt_status"] = (
        "known-byte-bound"
        if receipt is not None
        else ("caller-declared-only" if revision is not None else "unknown-rebuild")
    )
    if receipt is not None:
        plan["reproduction_library_build_revision"] = receipt["build_source_revision"]
    print(json.dumps(plan, indent=2, sort_keys=True), flush=True)
    if not args.run:
        return 0
    # The original worker/acceptance functions stay byte-exact. The new outer
    # orchestration keeps current Python/header identity separate from the
    # supplied binary's actual older, caller-declared, or unknown build receipt.
    driver = runpy.run_path(
        str(HERE / "new_campaign.py"), run_name="new_reproduction_driver"
    )
    return driver["run_campaign"](plan, original, HERE)


if __name__ == "__main__":
    raise SystemExit(main())
