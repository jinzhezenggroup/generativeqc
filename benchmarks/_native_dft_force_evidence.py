"""Join independently authenticated exact-FP64 bundles for the R1 campaign."""

from __future__ import annotations

import hashlib
import json
from typing import TYPE_CHECKING

from benchmarks._exact_fp64_evidence import load as load_exact
from benchmarks._exact_fp64_evidence import require

if TYPE_CHECKING:
    from pathlib import Path


def require_local_stream(path: Path) -> None:
    """Keep collection indirection inside regular, policy-sized local files."""
    require(not path.is_symlink() and path.is_file(), "R1 stream must be local")
    require(path.stat().st_size <= 1 << 20, "R1 compressed stream size")


def load(folder: Path) -> tuple[dict, dict[str, str]]:
    """Keep transport files bounded without changing the numerical codec.

    Every member is an ordinary exact-FP64 v3 bundle. The collection authenticates
    each member's stream manifest, rejects duplicate logical receipts and never
    searches history or another directory for missing scientific observations.
    """
    index_path = folder / "bundles.json"
    require_local_stream(index_path)
    index = json.loads(index_path.read_text())
    require(
        index.get("schema") == "generativeqc.native-dft-force-evidence-set.v1"
        and isinstance(index.get("bundles"), dict)
        and 0 < len(index["bundles"]) <= 64,
        "R1 evidence collection schema",
    )
    documents: dict[str, str] = {}
    members = {}
    for name, digest in index["bundles"].items():
        require(
            isinstance(name, str)
            and bool(name)
            and all(character.isalnum() or character == "-" for character in name),
            "R1 evidence member name",
        )
        member = folder / name
        require(not member.is_symlink(), "R1 evidence member must be local")
        for filename in ("sha256.txt", "receipts.json.xz", "fp64.xz"):
            require_local_stream(member / filename)
        require(
            hashlib.sha256((member / "sha256.txt").read_bytes()).hexdigest() == digest,
            "R1 evidence member manifest identity",
        )
        metadata, restored = load_exact(member)
        require(not documents.keys() & restored.keys(), "duplicate R1 receipt")
        members[name] = metadata
        documents.update(restored)
    return {"index": index, "members": members}, documents
