"""Reanalyse retained Standard/Fast endpoints without running a calculation.

These are descriptive comparisons of historical receipts, not measurements of
the current checkout or significance tests. Preserve cold and frozen-density
warm populations separately, and distinguish independent error from the paired
Standard/Fast difference. No reference data enter a production calculator.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import lzma
from collections import Counter
from pathlib import Path
from statistics import median

import numpy as np

from benchmarks._retention import raw_output_path
from benchmarks.readme_defaults_qualification import qualify

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "benchmarks/results"


def read(path: Path) -> dict | list:
    """Read JSON or compressed JSON; leave the retained input untouched."""
    data = path.read_bytes()
    return json.loads(lzma.decompress(data) if path.suffix == ".xz" else data)


def differences(standard: dict, fast: dict) -> dict:
    """Measure total-energy and full-vector differences, without an error claim."""
    left, right = (np.asarray(s["forces"], dtype=float) for s in (standard, fast))
    if left.shape != right.shape or left.ndim != 2 or left.shape[1] != 3:
        raise ValueError("incompatible force vectors")
    delta = right - left
    result = {
        "energy_abs_Eh": abs(fast["energy"] - standard["energy"]),
        "force_max_abs_Eh_per_bohr": float(np.max(np.abs(delta))),
        "force_rms_Eh_per_bohr": float(np.sqrt(np.mean(delta**2))),
    }
    if not all(np.isfinite(v) for v in result.values()):
        raise ValueError("nonfinite paired observable")
    return result


def require_equal(left: dict, right: dict, fields: tuple[str, ...]) -> None:
    """Refuse to attribute model, input or binary differences to a policy."""
    for field in fields:
        if left[field] != right[field]:
            raise ValueError(f"unmatched Standard/Fast {field}")


def hfdft_audit(root: Path) -> dict:
    """Retain every population, including fallback routes and accuracy failures."""
    records = {arm: read(root / f"{arm}.json.xz") for arm in ("standard", "fast")}
    indexed = {}
    for arm, rows in records.items():
        if any(r["arm"] != arm for r in rows):
            raise ValueError("receipt is in the wrong policy series")
        indexed[arm] = {(r["mode"], r["method"], r["atoms"]): r for r in rows}
        if len(indexed[arm]) != len(rows):
            raise ValueError("duplicate population")
    if indexed["standard"].keys() != indexed["fast"].keys():
        raise ValueError("unpaired populations")
    # Recompute the independent gates and reference-grid checks, rather than
    # trusting a saved PASS or treating Standard as an independent oracle.
    qualification = qualify(
        records["standard"] + records["fast"],
        read(root / "oracles.json.xz"),
        reference_reuse=read(root / "reference-reuse.json"),
    )
    pairs = []
    for key in sorted(indexed["standard"]):
        standard, fast = (indexed[arm][key] for arm in ("standard", "fast"))
        require_equal(
            standard,
            fast,
            (
                "geometry_bohr",
                "orbital_basis",
                "basis_representation",
                "charge",
                "spin",
                "source_revision",
                "runner_sha256",
                "native_build",
                "protocol",
                "thread_controls",
            ),
        )
        if standard["status"] != "complete" or fast["status"] != "complete":
            raise ValueError("incomplete endpoint")
        for phase, count in (("cold", 1), ("warm", 3)):
            samples = [
                sorted(
                    (s for s in r["samples"] if s["phase"] == phase),
                    key=lambda s: s["repeat"],
                )
                for r in (standard, fast)
            ]
            if any(len(s) != count for s in samples):
                raise ValueError("expected one cold and three warm calls")
            comparisons = []
            for a, b in zip(*samples, strict=True):
                require_equal(a, b, ("phase", "repeat"))
                if not a["converged"] or not b["converged"]:
                    raise ValueError("unconverged endpoint")
                comparisons.append(differences(a, b))
            times = [[s["seconds"] for s in values] for values in samples]
            if not all(np.isfinite(t) and t > 0 for values in times for t in values):
                raise ValueError("invalid complete endpoint timing")
            pairs.append(
                {
                    "mode": key[0],
                    "method": key[1],
                    "atoms": key[2],
                    "phase": phase,
                    "standard_seconds": times[0],
                    "fast_seconds": times[1],
                    "standard_over_fast": median(times[0]) / median(times[1]),
                    "differences": comparisons,
                    "work": {
                        arm: [
                            {
                                "fock_builds": s["fock_builds"],
                                "iterations": s["iterations"],
                                "mixed_stage_fock_builds": (
                                    s.get("precision") or {}
                                ).get("mixed_stage_fock_builds"),
                                "gradient_mode": (
                                    (s.get("precision") or {}).get("gradient") or {}
                                ).get("returned_mode"),
                                "route": (s.get("precision_validation") or {}).get(
                                    "status"
                                ),
                                "controls": {
                                    k: s["settings"][k]
                                    for k in (
                                        "energy_tolerance",
                                        "density_tolerance",
                                        "screening_tolerance",
                                    )
                                },
                            }
                            for s in values
                        ]
                        for arm, values in zip(
                            ("standard", "fast"), samples, strict=True
                        )
                    },
                }
            )
    summary = []
    for mode in sorted({p["mode"] for p in pairs}):
        for phase in ("cold", "warm"):
            selected = [p for p in pairs if (p["mode"], p["phase"]) == (mode, phase)]
            ratios = [p["standard_over_fast"] for p in selected]
            work = [w for p in selected for w in p["work"]["fast"]]
            summary.append(
                {
                    "mode": mode,
                    "phase": phase,
                    "populations": len(selected),
                    "median_standard_over_fast": median(ratios),
                    "ratio_within_5_percent_of_one": sum(
                        0.95 <= r <= 1.05 for r in ratios
                    ),
                    "ratio_above_1_05": sum(r > 1.05 for r in ratios),
                    "ratio_below_0_95": sum(r < 0.95 for r in ratios),
                    "equal_fock_populations": sum(
                        [w["fock_builds"] for w in p["work"]["standard"]]
                        == [w["fock_builds"] for w in p["work"]["fast"]]
                        for p in selected
                    ),
                    "fast_routes": dict(Counter(w["route"] for w in work)),
                    "fast_calls_with_zero_mixed_scf": sum(
                        w["mixed_stage_fock_builds"] == 0 for w in work
                    ),
                }
            )
    return {
        "summary": summary,
        "pairs": pairs,
        "independent_qualification": qualification,
    }


def cbs_audit(path: Path) -> list[dict]:
    """Compare complete two-group calls and expose the coupled stopping/warmup policy."""
    data = read(path)
    pairs = []
    for left, right in zip(data["points"], data["fast"]["points"], strict=True):
        require_equal(left, right, ("carbon",))
        standard, fast = left["native"], right["native"]
        require_equal(
            standard, fast, ("protocol", "source_revision", "library_sha256", "gpu")
        )
        if len(standard["samples"]) != 1 or len(fast["samples"]) != 1:
            raise ValueError("expected a single cold CBS observation per arm")
        a, b = standard["samples"][0], fast["samples"][0]
        work = {}
        for arm, sample in (("standard", a), ("fast", b)):
            record = standard if arm == "standard" else fast
            if (
                record["policy"] != arm
                or record["status"] != "completed"
                or not sample["converged"]
                or not np.isfinite(sample["complete_seconds"])
                or sample["complete_seconds"] <= 0
            ):
                raise ValueError("invalid completed CBS endpoint")
            cc = next(
                c["result"] for c in sample["components"] if c["method"] == "df-ccsd(t)"
            )
            work[arm] = {
                "convergence": sample["convergence"],
                "ccsd_iterations": cc["correlation"]["ccsd_iterations"],
                "ccsd_energy_change": cc["correlation"]["ccsd_energy_change"],
                "ccsd_replay_singles_residual_max": cc["correlation"][
                    "ccsd_replay_singles_residual_max"
                ],
                "cc_performance": cc["cc_performance"],
                "group_seconds": [
                    g["elapsed_seconds"] for g in sample["execution_groups"]
                ],
            }
        pairs.append(
            {
                "carbon": left["carbon"],
                "standard_seconds": a["complete_seconds"],
                "fast_seconds": b["complete_seconds"],
                "time_reduction": 1 - b["complete_seconds"] / a["complete_seconds"],
                "differences": differences(a, b),
                "work": work,
                "independent_reference_available": left["reference"] is not None,
            }
        )
    return pairs


def main() -> None:
    """Write a detached analysis with the original receipt file identities."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--hfdft", type=Path, default=RESULTS / "df-dft48-work-20261025"
    )
    parser.add_argument(
        "--cbs", type=Path, default=RESULTS / "cbs-fast-memory-20261021/summary.json.xz"
    )
    parser.add_argument("--output", type=raw_output_path, required=True)
    args = parser.parse_args()
    inputs = [
        args.hfdft / name
        for name in (
            "standard.json.xz",
            "fast.json.xz",
            "oracles.json.xz",
            "reference-reuse.json",
        )
    ]
    inputs.append(args.cbs)
    result = {
        "scope": "historical receipt reanalysis; no new GPU measurements or default changes",
        "inputs": {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs},
        "hfdft": hfdft_audit(args.hfdft),
        "cbs": cbs_audit(args.cbs),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps(result["hfdft"]["summary"], indent=2))


if __name__ == "__main__":
    main()
