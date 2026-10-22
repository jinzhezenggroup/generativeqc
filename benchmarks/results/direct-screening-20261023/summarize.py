"""Recheck accepted scalar comparisons and summarize the screening campaign.

Full force arrays stay in local raw artifacts. Per-sample errors and cross-arm
maxima were computed from those arrays before compaction; this reader verifies
the retained scalar gates, archive integrity, work counts and timing population.
"""

from __future__ import annotations

import collections
import hashlib
import json
import lzma
import statistics
from pathlib import Path


def summarize() -> dict:
    """Keep clean endpoints, diagnostic profiles and synthetic trials separate."""
    root = Path(__file__).resolve().parent
    payload = (root / "receipts.json.xz").read_bytes()
    manifest = json.loads((root / "manifest.json").read_text())
    entry = manifest["files"]["receipts.json.xz"]
    if len(payload) != entry["bytes"] or hashlib.sha256(payload).hexdigest() != entry["sha256"]:
        raise ValueError("accepted comparison integrity mismatch")
    data = json.loads(lzma.decompress(payload))
    output = {"endpoints": {}, "synthetic": {}, "census": {}}
    populations = collections.defaultdict(list)
    identities = set()
    for record in data["endpoints"].values():
        if record["status"] != "complete":
            raise ValueError("incomplete accepted endpoint")
        identities.add((record["native_build"]["source_identity"], record["native_build"]["library_sha256"]))
        for row in record["samples"]:
            if any(not 0 <= row["errors"][key] <= gate for key, gate in record["gates"].items()):
                raise ValueError("independent scalar acceptance gate failed")
            populations[(record["case"], int(record["fp32_block_screening"]), row["phase"])].append(row)
    if len(identities) != 1 or len(data["endpoints"]) != 18 or sum(map(len, populations.values())) != 78:
        raise ValueError("unexpected source or sample population")
    for (case, arm, phase), rows in populations.items():
        output["endpoints"].setdefault(case, {}).setdefault(str(arm), {})[phase] = {
            "samples": len(rows),
            "seconds_median": statistics.median(r["seconds"] for r in rows),
            "seconds_min": min(r["seconds"] for r in rows),
            "seconds_max": max(r["seconds"] for r in rows),
            "iterations": sorted({r["iterations"] for r in rows}),
            "fock_builds": sorted({r["fock_builds"] for r in rows}, key=str),
        }
    for case, result in output["endpoints"].items():
        rows = [r for (c, _, _), values in populations.items() if c == case for r in values]
        result["all_sample_oracle_max_error"] = {k: max(r["errors"][k] for r in rows) for k in rows[0]["errors"]}
        result["all_sample_baseline_candidate_max"] = data["baseline_candidate_maxima"][case]
    synthetic = collections.defaultdict(list)
    for row in data["synthetic"]:
        if "mode" not in row:
            continue
        if row["mismatches"]:
            raise ValueError("synthetic task mismatch")
        key = f"{row['bounds']}/{'force' if row['force'] else 'fock'}/{row['mode']}/tau={row['tolerance']:g}"
        synthetic[key].append(row)
    for key, rows in synthetic.items():
        output["synthetic"][key] = {
            "total_ms_median": statistics.median(r["total_ms"] for r in rows),
            "packing_ms_median": statistics.median(r["packing_ms"] for r in rows),
            "survivors": sorted({r["survivors"] for r in rows}),
            "coarse_rejections": sorted({r["coarse_rejections"] for r in rows}),
        }
    for case, arms in data["census"].items():
        left, right = arms["0"], arms["1"]
        fields = lambda rows: [{"traversals": r["traversals"], "classes": r["classes"]} for r in rows]
        if not all(r["valid"] for r in left + right) or fields(left) != fields(right):
            raise ValueError("invalid or mismatched census")
        output["census"][case] = {
            "all_work_fields_equal": True,
            "admitted_quartets": [sum(c["admitted"] for c in r["classes"]) for r in left],
            "block_rejected_pages": [sum(t["block_rejected_pages"] for t in r["traversals"]) for r in left],
        }
    output["native_identity"] = next(iter(identities))
    return output


if __name__ == "__main__":
    print(json.dumps(summarize(), indent=2))
