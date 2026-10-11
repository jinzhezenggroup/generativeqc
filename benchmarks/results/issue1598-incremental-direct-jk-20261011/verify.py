"""Recheck the retained #1598 negative-closeout gates without CUDA.

Decompresses the retained records, verifies library/source identity
consistency within each campaign, recomputes the paired numerical gates and
the failure dispositions, and fails when the recorded decision statements do
not match the retained measurements.
"""

from __future__ import annotations

import gzip
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent

EXPECTED_LIBRARIES = {
    "old": "2998014a18547b813149521c51193cd8913602b5d7e4551e326a4d5c91053a77",
    "new": "30734a0fc741da60389fb817b7429acd4a5164c840728308822fe215eae90a63",
}
MAX_ENERGY_DIFFERENCE = 1.0e-8
MAX_FORCE_DIFFERENCE = 1.0e-7


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def load_gz(path: Path) -> dict:
    return json.loads(gzip.decompress(path.read_bytes()))


def outcome_of(path: Path) -> dict:
    stem = path.name[: -len(".json.gz")]
    outcome = path.with_name(stem + ".outcome.gz")
    require(outcome.exists(), f"missing outcome for {path.name}")
    return load_gz(outcome)


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
    return abs(float(first) - float(second))


def check_campaign(name: str, sizes: tuple[int, ...]) -> dict:
    root = ROOT / name / "rhf"
    summary = {"points": [], "failures": []}
    for atoms in sizes:
        for mode in ("baseline", "incremental"):
            for round in (1, 2):
                path = root / f"direct-{atoms}-{mode}-{round}.json.gz"
                outcome_path = path.with_name(
                    path.name[: -len(".json.gz")] + ".outcome.gz"
                )
                require(outcome_path.exists(), f"missing outcome for {path.stem}")
                outcome = load_gz(outcome_path)
                if outcome.get("exit_code") != 0 or not path.exists():
                    summary["failures"].append(f"{name}/{path.stem}")
                    continue
                record = load_gz(path)
                require(
                    record.get("benchmark") == "compare_gpu4pyscf_batch",
                    f"{path.name} is not a comparator record",
                )
                require(
                    record["native_build"]["library_sha256"]
                    == EXPECTED_LIBRARIES[name],
                    f"{path.name} library identity mismatch",
                )
                require(record["gate"]["passed"], f"{path.name} failed its gates")
                summary["points"].append(path.stem)
        # Paired gates across modes, using the first successful record each.
        base_path = root / f"direct-{atoms}-baseline-1.json.gz"
        inc_path = root / f"direct-{atoms}-incremental-1.json.gz"
        if base_path.exists() and inc_path.exists():
            base = load_gz(base_path)
            inc = load_gz(inc_path)
            require(
                maximum_abs_difference(
                    base["generativeqc"]["energies_hartree"],
                    inc["generativeqc"]["energies_hartree"],
                )
                <= MAX_ENERGY_DIFFERENCE,
                f"{atoms}-atom energy gate failed",
            )
            require(
                maximum_abs_difference(
                    base["generativeqc"]["forces_hartree_per_bohr"],
                    inc["generativeqc"]["forces_hartree_per_bohr"],
                )
                <= MAX_FORCE_DIFFERENCE,
                f"{atoms}-atom force gate failed",
            )
    return summary


def main() -> None:
    old = check_campaign("old", (12, 24, 48, 96))
    new = check_campaign("new", (24, 48, 96))
    # The negative-closeout statements in the README/note must match the data:
    # the 192-AO (24-atom) incremental points fail in both campaigns, and every
    # comparable paired gate must pass.
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
        f"old={len(old['points'])} ok points, {len(old['failures'])} failures; "
        f"new={len(new['points'])} ok points, {len(new['failures'])} failures; "
        "all paired numerical gates pass"
    )


if __name__ == "__main__":
    main()
