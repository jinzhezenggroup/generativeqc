"""Recheck the retained #1598 negative-closeout gates without CUDA.

Decompresses the retained records, verifies library identity per campaign,
recomputes the paired numerical gates for the retained round-1 pairs and the
failure dispositions from the retained outcomes. Compact publication retains
round-1 records plus every outcome; removed round-2 records are cited from the
README/note tables and are not re-derivable here.
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
    summary = {"records": [], "failures": []}
    for atoms in sizes:
        for mode in ("baseline", "incremental"):
            for round in (1, 2):
                stem = f"direct-{atoms}-{mode}-{round}"
                record_path = root / f"{stem}.json.gz"
                outcome_path = root / f"{stem}.outcome.gz"
                if not outcome_path.exists() and not record_path.exists():
                    # Compact publication omits non-decisive points entirely;
                    # their dispositions are cited from the README/note tables.
                    continue
                require(outcome_path.exists(), f"missing outcome for {name}/{stem}")
                outcome = load_gz(outcome_path)
                if outcome.get("exit_code") != 0:
                    summary["failures"].append(f"{name}/{stem}")
                    continue
                if record_path.exists():
                    record = load_gz(record_path)
                    require(
                        record.get("benchmark") == "compare_gpu4pyscf_batch",
                        f"{name}/{stem} is not a comparator record",
                    )
                    require(
                        record["native_build"]["library_sha256"]
                        == EXPECTED_LIBRARIES[name],
                        f"{name}/{stem} library identity mismatch",
                    )
                    require(record["gate"]["passed"], f"{name}/{stem} failed gates")
                    summary["records"].append(stem)
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
                f"{name}/{atoms}-atom energy gate failed",
            )
            require(
                maximum_abs_difference(
                    base["generativeqc"]["forces_hartree_per_bohr"],
                    inc["generativeqc"]["forces_hartree_per_bohr"],
                )
                <= MAX_FORCE_DIFFERENCE,
                f"{name}/{atoms}-atom force gate failed",
            )
    return summary


def main() -> None:
    old = check_campaign("old", (12, 24, 48, 96))
    new = check_campaign("new", (24, 48, 96))
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
        f"old={len(old['records'])} retained records, {len(old['failures'])} failed "
        f"points; new={len(new['records'])} retained records, "
        f"{len(new['failures'])} failed points; all retained paired numerical "
        "gates pass"
    )


if __name__ == "__main__":
    main()
