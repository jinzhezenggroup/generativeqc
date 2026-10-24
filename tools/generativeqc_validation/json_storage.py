"""Measure JSON representation costs without changing scientific number tokens.

This is a storage audit, not an instruction to recompress every historical file.
In particular, minification changes byte identities even when values are intact.
"""

from __future__ import annotations

import gzip
import lzma
import re
from collections import Counter

from tools.generativeqc_validation.storage_json import decode_storage_json

# Match whole strings before whitespace so escaped quotes and embedded spaces
# remain byte-identical. Validation separately rejects invalid JSON and duplicates.
_LEXEMES = re.compile(rb'"(?:[^"\\]|\\.)*"|[ \t\r\n]+', re.DOTALL)


def compact_json(data: bytes) -> bytes:
    """Remove only JSON whitespace; retain exact strings and number spellings."""
    decode_storage_json(data)
    return _LEXEMES.sub(lambda m: m[0] if m[0].startswith(b'"') else b"", data)


def audit_json(blobs: dict[str, bytes]) -> dict:
    """Compare bounded, indexed JSON records using identical compression levels.

    Totals cover successfully decoded result JSON only. Errors are explicit so
    malformed legacy files cannot silently inflate a claimed saving. Estimates
    exclude any new manifest overhead and are not approved migrations.
    """
    from tools.compact_evidence_publications import _plain

    totals: Counter = Counter()
    formats: dict[str, Counter] = {}
    errors = []
    for path, stored in sorted(blobs.items()):
        suffix = next(
            (s for s in (".json.gz", ".json.xz", ".json") if path.endswith(s)),
            None,
        )
        if not path.startswith("benchmarks/results/") or suffix is None:
            continue
        try:
            plain = _plain(stored, path)
            compact = compact_json(plain)
            row = Counter(
                files=1,
                stored_bytes=len(stored),
                expanded_bytes=len(plain),
                compact_bytes=len(compact),
                gzip_original_bytes=len(gzip.compress(plain, compresslevel=9, mtime=0)),
                gzip_compact_bytes=len(
                    gzip.compress(compact, compresslevel=9, mtime=0)
                ),
                xz_original_bytes=len(lzma.compress(plain, preset=6)),
                xz_compact_bytes=len(lzma.compress(compact, preset=6)),
            )
        except (ValueError, OSError, EOFError, lzma.LZMAError) as error:
            errors.append({"path": path, "error": str(error)})
            continue
        totals.update(row)
        formats.setdefault(suffix, Counter()).update(row)
    return {
        "schema": "generativeqc.json-storage-audit.v1",
        "scope": "tracked result JSON; read-only estimates, not migration approval",
        "totals": dict(totals),
        "formats": {key: dict(value) for key, value in sorted(formats.items())},
        "errors": errors,
    }
