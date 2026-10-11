"""Recheck the retained #1598 negative-closeout gates without CUDA.

Binds every retained file to the publication manifest (path, bytes, sha256),
validates library identity per campaign, recomputes the paired numerical gates
for every retained comparable round and the failure dispositions from the
retained outcomes, and reports which points are not retained so the compact
publication cannot be misread as a complete matrix.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parent

EXPECTED_LIBRARIES = {
    "old": "2998014a18547b813149521c51193cd8913602b5d7e4551e326a4d5c91053a77",
    "new": "30734a0fc741da60389fb817b7429acd4a5164c840728308822fe215eae90a63",
}
MAX_ENERGY_DIFFERENCE = 1.0e-8
MAX_FORCE_DIFFERENCE = 1.0e-7
SIZES = {"old": (12, 24, 48, 96), "new": (24, 48, 96)}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def load_gz(path: Path) -> dict:
    return json.loads(gzip.decompress(path.read_bytes()))


def validate_publication() -> dict[str, tuple[int, str]]:
    """Every member must exist with the exact bytes and digest recorded."""

    manifest_path = ROOT / "publication.json"
    require(manifest_path.exists(), "publication.json is missing")
    manifest = json.loads(manifest_path.read_text())
    require(
        manifest.get("schema") == "generativeqc.benchmark-publication.v1",
        "unsupported publication schema",
    )
    members: dict[str, tuple[int, str]] = {}
    for entry in manifest["files"]:
        path = ROOT / entry["path"]
        require(path.is_file(), f"publication member missing: {entry['path']}")
        data = path.read_bytes().replace(b"\r\n", b"\n")
        require(len(data) == entry["bytes"], f"member size changed: {entry['path']}")
        digest = hashlib.sha256(data).hexdigest()
        require(digest == entry["sha256"], f"member digest changed: {entry['path']}")
        members[entry["path"]] = (entry["bytes"], digest)
    return members


def maximum_abs_difference(first: object, second: object) -> float:
    if isinstance(first, list) or isinstance(second, list):
        require(
            isinstance(first, list) and isinstance(second, list),
            "result shapes differ",
        )
        require(len(first) == len(second) and first, "result shapes differ")
        return max(
            maximum_abs_difference(a, b) for a, b in zip(first, second, strict=True)
        )
    require(
        type(first) in (int, float) and type(second) in (int, float),
        "benchmark results require numeric values",
    )
    a, b = float(first), float(second)
    require(math.isfinite(a) and math.isfinite(b), "non-finite benchmark result")
    return abs(a - b)


def check_campaign(
    name: str, members: dict[str, tuple[int, str]]
) -> dict[str, object]:
    retained: list[str] = []
    failures: list[str] = []
    not_retained: list[str] = []
    for atoms in SIZES[name]:
        for mode in ("baseline", "incremental"):
            for round in (1, 2):
                stem = f"direct-{atoms}-{mode}-{round}"
                record_rel = f"{name}/rhf/{stem}.json.gz"
                outcome_rel = f"{name}/rhf/{stem}.outcome.gz"
                has_record = record_rel in members
                has_outcome = outcome_rel in members
                if not has_record and not has_outcome:
                    not_retained.append(f"{name}/{stem}")
                    continue
                require(has_outcome, f"retained record lacks outcome: {stem}")
                outcome = load_gz(ROOT / outcome_rel)
                if outcome.get("exit_code") != 0:
                    failures.append(f"{name}/{stem}")
                    continue
                require(has_record, f"successful point lacks record: {stem}")
                record = load_gz(ROOT / record_rel)
                require(
                    record.get("benchmark") == "compare_gpu4pyscf_batch",
                    f"{stem} is not a comparator record",
                )
                require(
                    record["native_build"]["library_sha256"]
                    == EXPECTED_LIBRARIES[name],
                    f"{stem} library identity mismatch",
                )
                require(record["gate"]["passed"], f"{stem} failed its gates")
                require(
                    math.isfinite(
                        float(record["generativeqc"]["cold_seconds"])
                    )
                    and float(record["generativeqc"]["cold_seconds"]) > 0,
                    f"{stem} non-finite cold timing",
                )
                retained.append(stem)
        # Paired gates for every retained comparable round pair.
        for round in (1, 2):
            base_rel = f"{name}/rhf/direct-{atoms}-baseline-{round}.json.gz"
            inc_rel = f"{name}/rhf/direct-{atoms}-incremental-{round}.json.gz"
            if base_rel in members and inc_rel in members:
                base = load_gz(ROOT / base_rel)
                inc = load_gz(ROOT / inc_rel)
                require(
                    maximum_abs_difference(
                        base["generativeqc"]["energies_hartree"],
                        inc["generativeqc"]["energies_hartree"],
                    )
                    <= MAX_ENERGY_DIFFERENCE,
                    f"{name}/{atoms}-atom round {round} energy gate failed",
                )
                require(
                    maximum_abs_difference(
                        base["generativeqc"]["forces_hartree_per_bohr"],
                        inc["generativeqc"]["forces_hartree_per_bohr"],
                    )
                    <= MAX_FORCE_DIFFERENCE,
                    f"{name}/{atoms}-atom round {round} force gate failed",
                )
    return {
        "retained": retained,
        "failures": failures,
        "not_retained": not_retained,
    }


def main() -> None:
    members = validate_publication()
    old = check_campaign("old", members)
    new = check_campaign("new", members)
    # The negative-closeout statements must match the retained dispositions:
    # the 192-AO (24-atom) incremental points fail in both campaigns.
    require(
        any("direct-24-incremental" in f for f in old["failures"]),
        "old 192-AO incremental failures missing",
    )
    require(
        any("direct-24-incremental" in f for f in new["failures"]),
        "new 192-AO incremental failures missing",
    )
    print(
        "verified: "
        f"old={len(old['retained'])} retained records, {len(old['failures'])} failed "
        f"points, {len(old['not_retained'])} not retained; "
        f"new={len(new['retained'])} retained records, {len(new['failures'])} failed "
        f"points, {len(new['not_retained'])} not retained; "
        "publication manifest and all retained paired numerical gates pass"
    )


if __name__ == "__main__":
    main()
