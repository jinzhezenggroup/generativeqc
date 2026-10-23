"""Recompute accepted numerical comparisons and summarize the screening campaign.

Accepted force/reference arrays are retained. This reader recomputes every
independent error and cross-arm maximum, checks their recorded summaries, and
verifies archive integrity, work counts and timing populations without a GPU.
"""

from __future__ import annotations

import collections
import hashlib
import json
import lzma
import math
import statistics
from pathlib import Path


def force_delta(first: list, second: list) -> list[float]:
    """Reject incomplete/nonfinite Cartesian arrays before comparing errors."""
    delta = []
    for left, right in zip(first, second, strict=True):
        if len(left) != 3 or len(right) != 3:
            raise ValueError("force row is not Cartesian")
        delta.extend(a - b for a, b in zip(left, right, strict=True))
    if not delta or not all(math.isfinite(value) for value in delta):
        raise ValueError("invalid force comparison")
    return delta


def summarize() -> dict:
    """Keep clean endpoints, diagnostic profiles and synthetic trials separate."""
    root = Path(__file__).resolve().parent
    payload = (root / "receipts.json.xz").read_bytes()
    manifest = json.loads((root / "manifest.json").read_text())
    entry = manifest["files"]["receipts.json.xz"]
    if (
        len(payload) != entry["bytes"]
        or hashlib.sha256(payload).hexdigest() != entry["sha256"]
    ):
        raise ValueError("accepted comparison integrity mismatch")
    data = json.loads(lzma.decompress(payload))
    output = {"endpoints": {}, "synthetic": {}, "census": {}}
    populations = collections.defaultdict(list)
    identities = set()
    for record in data["endpoints"].values():
        if record["status"] != "complete":
            raise ValueError("incomplete accepted endpoint")
        identities.add(
            (
                record["native_build"]["source_identity"],
                record["native_build"]["library_sha256"],
            )
        )
        for row in record["samples"]:
            reference = data["references"][record["case"]][row["geometry"]]
            geometry = data["geometries_bohr"][record["case"]][row["geometry"]]
            if len(row["forces"]) != len(geometry):
                raise ValueError("force array does not match the atom count")
            delta = force_delta(row["forces"], reference["forces"])
            errors = {
                "energy_per_atom": abs(row["energy"] - reference["energy"])
                / len(row["forces"]),
                "force_max": max(map(abs, delta)),
                "force_rms": math.sqrt(sum(v * v for v in delta) / len(delta)),
            }
            if any(
                not 0 <= errors[key] <= gate for key, gate in record["gates"].items()
            ):
                raise ValueError("independent numerical acceptance gate failed")
            if any(
                not math.isclose(
                    value, row["errors"][key], rel_tol=1e-14, abs_tol=1e-18
                )
                for key, value in errors.items()
            ):
                raise ValueError("recorded error differs from retained arrays")
            row["errors"] = errors
            populations[
                (record["case"], int(record["fp32_block_screening"]), row["phase"])
            ].append(row)
    if (
        len(identities) != 1
        or len(data["endpoints"]) != 18
        or sum(map(len, populations.values())) != 78
    ):
        raise ValueError("unexpected source or sample population")
    if (
        next(iter(identities))[0] != manifest["source_identity"]
        or data["source_identity"] != manifest["source_identity"]
    ):
        raise ValueError("source identity differs from manifest")
    for (case, arm, phase), rows in populations.items():
        if len(rows) != {"cold": 3, "warm": 9, "moved": 1}[phase]:
            raise ValueError("unexpected phase population")
        output["endpoints"].setdefault(case, {}).setdefault(str(arm), {})[phase] = {
            "samples": len(rows),
            "seconds_median": statistics.median(r["seconds"] for r in rows),
            "seconds_min": min(r["seconds"] for r in rows),
            "seconds_max": max(r["seconds"] for r in rows),
            "iterations": sorted({r["iterations"] for r in rows}),
            "fock_builds": sorted({r["fock_builds"] for r in rows}, key=str),
        }
    for case, result in output["endpoints"].items():
        rows = [
            r for (c, _, _), values in populations.items() if c == case for r in values
        ]
        result["all_sample_oracle_max_error"] = {
            k: max(r["errors"][k] for r in rows) for k in rows[0]["errors"]
        }
        maxima = {"energy_abs": 0.0, "force_max": 0.0}
        for geometry in ("original", "moved"):
            arms = [
                [
                    r
                    for (c, a, _), values in populations.items()
                    if c == case and a == arm
                    for r in values
                    if r["geometry"] == geometry
                ]
                for arm in (0, 1)
            ]
            for left in arms[0]:
                for right in arms[1]:
                    maxima["energy_abs"] = max(
                        maxima["energy_abs"], abs(left["energy"] - right["energy"])
                    )
                    difference = max(
                        map(abs, force_delta(left["forces"], right["forces"]))
                    )
                    maxima["force_max"] = max(maxima["force_max"], difference)
        if maxima != data["baseline_candidate_maxima"][case]:
            raise ValueError("recorded cross-arm maximum differs from retained arrays")
        result["all_sample_baseline_candidate_max"] = maxima
        if any(
            not 0 <= value <= data["cross_arm_gates"][key]
            for key, value in result["all_sample_baseline_candidate_max"].items()
        ):
            raise ValueError("cross-arm numerical gate failed")
        for phase in ("cold", "warm", "moved"):
            if any(
                result["0"][phase][key] != result["1"][phase][key]
                for key in ("iterations", "fock_builds")
            ):
                raise ValueError("cross-arm SCF work changed")
    synthetic = collections.defaultdict(list)
    for row in data["synthetic"]:
        if "mode" not in row:
            continue
        if row["mismatches"]:
            raise ValueError("synthetic task mismatch")
        key = f"{row['bounds']}/{'force' if row['force'] else 'fock'}/{row['mode']}/tau={row['tolerance']:g}"
        synthetic[key].append(row)
    for key, rows in synthetic.items():
        if len(rows) != 3 or any(
            r["coarse_rejections"] + r["coarse_acceptances"] + r["refinements"]
            != r["candidates"]
            for r in rows
        ):
            raise ValueError("invalid synthetic population or decision counts")
        output["synthetic"][key] = {
            "total_ms_median": statistics.median(r["total_ms"] for r in rows),
            "packing_ms_median": statistics.median(r["packing_ms"] for r in rows),
            "survivors": sorted({r["survivors"] for r in rows}),
            "coarse_rejections": sorted({r["coarse_rejections"] for r in rows}),
            "coarse_acceptances": sorted({r["coarse_acceptances"] for r in rows}),
            "refinements": sorted({r["refinements"] for r in rows}),
        }
    if len(synthetic) != 7 * 2 * 3 * 2:
        raise ValueError("missing synthetic mode, purpose, size or threshold")
    for case, arms in data["census"].items():
        left, right = arms["0"], arms["1"]
        fields = lambda rows: [
            {"traversals": r["traversals"], "classes": r["classes"]} for r in rows
        ]
        if not all(r["valid"] for r in left + right) or fields(left) != fields(right):
            raise ValueError("invalid or mismatched census")
        output["census"][case] = {
            "all_work_fields_equal": True,
            "admitted_quartets": [
                sum(c["admitted"] for c in r["classes"]) for r in left
            ],
            "block_rejected_pages": [
                sum(t["block_rejected_pages"] for t in r["traversals"]) for r in left
            ],
        }
    output["native_identity"] = next(iter(identities))
    return output


if __name__ == "__main__":
    print(json.dumps(summarize(), indent=2))
