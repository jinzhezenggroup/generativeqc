#!/usr/bin/env python3
"""Join a GenerativeQC arm with the GPU4PySCF arm case by case, on SCC.

This does no chemistry and no timing of its own. It reads the
``generativeqc.readme-defaults.v2`` JSON that ``tools/scc/run.sh`` writes for
each arm and reports, per ``method:mode``:

  * whether both arms ran at all, and if not, the recorded failure;
  * cold and warm wall time for each, and the ratio;
  * SCF iteration counts for each;
  * the energy difference against the common observable gates.

Keeping it read-only is deliberate: the numbers that matter are the ones the
two arms already produced on the same geometry with the same frozen input
tables, and re-deriving anything here would let this script become a second,
unaudited source of truth.

Usage:
    compare_scc.py <standard-dir> <gpu4pyscf-dir> [--json OUT.json]

Both directories are the ``OUT`` paths used for ``GQC_ARM=standard`` and
``GQC_ARM=gpu4pyscf`` with the same ``GQC_GROUP``.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

SCHEMA = "generativeqc.readme-defaults.v2"
# Mirrors COMMON_GATES in benchmarks/readme_defaults.py.
COMMON_GATES = {
    "energy_abs_per_atom_hartree": 1e-5,
    "force_max_abs_hartree_per_bohr": 1e-4,
    "force_rms_hartree_per_bohr": 3e-5,
}
OK_STATUS = "complete"


def is_ok(record: dict) -> bool:
    return record.get("status") == OK_STATUS


def load(directory: Path) -> dict[tuple[str, str], dict]:
    """Index every case JSON in a results directory by (method, mode)."""
    if not directory.is_dir():
        raise SystemExit(f"error: {directory} is not a directory")
    cases: dict[tuple[str, str], dict] = {}
    for path in sorted(directory.glob("*.json")):
        try:
            record = json.loads(path.read_text())
        except json.JSONDecodeError as exc:
            print(f"warning: {path.name}: unreadable ({exc})", file=sys.stderr)
            continue
        if record.get("schema") != SCHEMA:
            continue
        key = (record.get("method"), record.get("mode"))
        if key not in cases:
            record["_path"] = path.name
            cases[key] = record
    return cases


def phases(record: dict) -> dict[str, dict]:
    """First sample of each phase; run.sh never repeats a phase."""
    out: dict[str, dict] = {}
    for sample in record.get("samples") or ():
        out.setdefault(sample.get("phase"), sample)
    return out


def warm_median(record: dict) -> float | None:
    times = [
        float(s["seconds"])
        for s in record.get("samples") or ()
        if s.get("phase") == "warm" and s.get("seconds") is not None
    ]
    return statistics.median(times) if times else None


def describe(record: dict | None) -> str:
    if record is None:
        return "not run"
    if not is_ok(record):
        return f"FAILED: {record.get('error', 'unknown')}"
    return "ok"


def fmt(value: float | None, spec: str = ".3f") -> str:
    return "-" if value is None else format(value, spec)


def energy_agreement(ours: dict, theirs: dict, atoms: int) -> tuple[float, bool] | None:
    """Compare endpoint energies; return (abs per atom, within gate)."""
    a = phases(ours).get("cold", {}).get("energy")
    b = phases(theirs).get("cold", {}).get("energy")
    if a is None or b is None:
        return None
    delta = abs(float(a) - float(b))
    per_atom = delta / max(atoms, 1)
    return delta, per_atom <= COMMON_GATES["energy_abs_per_atom_hartree"]


def iterations(record: dict | None) -> str:
    if record is None or not is_ok(record):
        return "-"
    sample = phases(record).get("cold", {})
    cold = sample.get("iterations")
    warm = phases(record).get("warm", {}).get("iterations")
    if cold is None:
        return "-"
    return f"{cold}" if warm is None else f"{cold}/{warm}"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("standard", type=Path, help="results dir for GQC_ARM=standard")
    parser.add_argument("gpu4pyscf", type=Path, help="results dir for GQC_ARM=gpu4pyscf")
    parser.add_argument("--json", type=Path, help="also write a machine-readable summary")
    args = parser.parse_args(argv)

    ours = load(args.standard)
    theirs = load(args.gpu4pyscf)
    keys = sorted(set(ours) | set(theirs), key=lambda k: (k[0] or "", k[1] or ""))
    if not keys:
        print(f"error: no {SCHEMA} records in either directory", file=sys.stderr)
        return 1

    rows = []
    for key in keys:
        method, mode = key
        a = ours.get(key)
        b = theirs.get(key)
        atoms = (a or b or {}).get("atoms") or 96

        row = {
            "method": method,
            "mode": mode,
            "atoms": atoms,
            "generativeqc": describe(a),
            "gpu4pyscf": describe(b),
            "generativeqc_cold_s": None,
            "gpu4pyscf_cold_s": None,
            "generativeqc_warm_s": None,
            "gpu4pyscf_warm_s": None,
            "cold_ratio": None,
            "warm_ratio": None,
            "iterations": {
                "generativeqc": iterations(a),
                "gpu4pyscf": iterations(b),
            },
            "energy_abs_hartree": None,
            "energy_within_gate": None,
        }
        if a is not None and is_ok(a):
            cold = phases(a).get("cold", {}).get("seconds")
            row["generativeqc_cold_s"] = float(cold) if cold is not None else None
            row["generativeqc_warm_s"] = warm_median(a)
        if b is not None and is_ok(b):
            cold = phases(b).get("cold", {}).get("seconds")
            row["gpu4pyscf_cold_s"] = float(cold) if cold is not None else None
            row["gpu4pyscf_warm_s"] = warm_median(b)
        for phase in ("cold", "warm"):
            x = row[f"generativeqc_{phase}_s"]
            y = row[f"gpu4pyscf_{phase}_s"]
            if x is not None and y:
                row[f"{phase}_ratio"] = x / y
        if a is not None and b is not None and is_ok(a) and is_ok(b):
            agreement = energy_agreement(a, b, atoms)
            if agreement:
                row["energy_abs_hartree"] = agreement[0]
                row["energy_within_gate"] = agreement[1]
        rows.append(row)

    header = (
        f"{'case':<16}{'gqc':<9}{'gpu4pyscf':<9}"
        f"{'gqc cold':>10}{'py cold':>10}{'ratio':>8}"
        f"{'gqc warm':>10}{'py warm':>10}{'ratio':>8}"
        f"{'iters g/p':>22}{'dE/atom':>11}"
    )
    print(header)
    print("-" * len(header))
    for row in rows:
        case = f"{row['method']}:{row['mode']}"
        # A failure is the interesting result, so surface the reason inline.
        gqc_state = "ok" if row["generativeqc"] == "ok" else "FAIL"
        py_state = "ok" if row["gpu4pyscf"] == "ok" else "FAIL"
        print(
            f"{case:<16}{gqc_state:<9}{py_state:<9}"
            f"{fmt(row['generativeqc_cold_s']):>10}{fmt(row['gpu4pyscf_cold_s']):>10}"
            f"{fmt(row['cold_ratio'], '.2f'):>8}"
            f"{fmt(row['generativeqc_warm_s']):>10}{fmt(row['gpu4pyscf_warm_s']):>10}"
            f"{fmt(row['warm_ratio'], '.2f'):>8}"
            f"{row['iterations']['generativeqc'] + '/' + row['iterations']['gpu4pyscf']:>22}"
            f"{fmt(row['energy_abs_hartree'], '.2e'):>11}"
        )
    print()
    print("ratio = GenerativeQC / GPU4PySCF; >1 means GPU4PySCF is faster.")
    print(f"dE/atom gate = {COMMON_GATES['energy_abs_per_atom_hartree']:g} hartree "
          "(cold endpoint energies).")

    problems = [r for r in rows if "FAIL" in (describe(ours.get((r['method'], r['mode']))),
                                              describe(theirs.get((r['method'], r['mode']))))]
    for row in problems:
        for arm in ("generativeqc", "gpu4pyscf"):
            if row[arm] != "ok":
                print(f"\n{row['method']}:{row['mode']} [{arm}] {row[arm]}")
    off_gate = [r for r in rows if r["energy_within_gate"] is False]
    if off_gate:
        print("\nenergy outside the common gate:")
        for row in off_gate:
            print(f"  {row['method']}:{row['mode']}  dE={row['energy_abs_hartree']:.3e}")

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(
            json.dumps(
                {
                    "standard_dir": str(args.standard),
                    "gpu4pyscf_dir": str(args.gpu4pyscf),
                    "common_gates": COMMON_GATES,
                    "rows": rows,
                },
                indent=2,
            )
            + "\n"
        )
        print(f"\nwrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
