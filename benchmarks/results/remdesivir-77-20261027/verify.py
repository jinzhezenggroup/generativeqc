"""Recompute the remdesivir-77 tables from the retained ``.xz`` record bundles.

The bundles are the authoritative artifacts; this script never needs a GPU, the
native library or the benchmark drivers. With ``--check`` it recomputes
``timings.md``, ``timings.csv`` and ``comparisons.csv`` in memory and fails when
they disagree with the committed files; with ``--write`` it regenerates them.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import lzma
import statistics
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
METHOD_ORDER = (
    "hf",
    "pbe",
    "pbe0",
    "b3lyp",
    "r2scan",
    "r2scan-3c",
    "m06-2x",
    "wb97m-v",
)
MODE_ORDER = ("direct", "df")
CAMPAIGN_BUNDLES = (
    ("def2-svp", "standard", "standard.json.xz"),
    ("def2-svp", "gpu4pyscf", "gpu4pyscf.json.xz"),
    ("def2-tzvp", "standard", "tzvp-standard.json.xz"),
    ("def2-tzvp", "gpu4pyscf", "tzvp-gpu4pyscf.json.xz"),
)
ENGINE = {"standard": "GenerativeQC Standard", "gpu4pyscf": "GPU4PySCF 1.8.1"}
CSV_FIELDS = (
    "basis",
    "method",
    "mode",
    "engine",
    "status",
    "cold_seconds",
    "warm_median_seconds",
    "warm_min_seconds",
    "warm_max_seconds",
    "warm_n",
    "note",
)
COMPARISON_FIELDS = (
    "basis",
    "method",
    "mode",
    "native_cold_seconds",
    "reference_cold_seconds",
    "cold_speedup",
    "native_warm_seconds",
    "reference_warm_seconds",
    "warm_speedup",
    "energy_abs_per_atom_hartree",
    "force_max_abs_hartree_per_bohr",
    "force_rms_hartree_per_bohr",
)


def load_bundle(name: str) -> dict[str, Any]:
    with lzma.open(HERE / name, "rt", encoding="utf-8") as stream:
        return json.load(stream)


def reason(record: dict[str, Any]) -> str:
    """Return a one-line reason for a record without a retained endpoint."""
    error = (record.get("error") or "").strip()
    if error:
        return error.splitlines()[0]
    status = record.get("status") or "unknown"
    if status == "running":
        return "interrupted: no endpoint retained (point timeout or job time limit)"
    return f"no retained endpoint (status={status})"


def measured(record: dict[str, Any]) -> dict[str, Any] | None:
    """Reduce one record to its measured endpoint, if it completed."""
    if record.get("status") != "complete":
        return None
    samples = record.get("samples") or []
    if not samples:
        return None
    warm = [sample["seconds"] for sample in samples[1:]]
    return {
        "cold": float(samples[0]["seconds"]),
        "warm": statistics.median(warm) if warm else float("nan"),
        "warm_min": min(warm) if warm else float("nan"),
        "warm_max": max(warm) if warm else float("nan"),
        "warm_n": len(warm),
        "energy": float(samples[0]["energy"]),
        "forces": samples[0]["forces"],
        "atoms": int(record["atoms"]),
    }


def collect() -> dict[tuple[str, str, str, str], dict[str, Any]]:
    rows: dict[tuple[str, str, str, str], dict[str, Any]] = {}
    for basis, arm, name in CAMPAIGN_BUNDLES:
        bundle = load_bundle(name)
        for entry in bundle["records"]:
            record = entry["record"]
            key = (basis, record["method"], record["mode"], arm)
            value = measured(record)
            rows[key] = {
                "record": record,
                "measured": value,
                "error": record.get("error"),
                "status": record.get("status"),
            }
    return rows


def sort_key(key: tuple[str, str, str, str]) -> tuple[int, int, int, int]:
    basis, method, mode, arm = key
    return (
        ("def2-svp", "def2-tzvp").index(basis),
        METHOD_ORDER.index(method) if method in METHOD_ORDER else 99,
        MODE_ORDER.index(mode),
        ("standard", "gpu4pyscf").index(arm),
    )


def render_timings(rows: dict[tuple[str, str, str, str], dict[str, Any]]) -> str:
    lines = [
        "# Remdesivir-77 endpoint timings",
        "",
        "Absolute latency of one cold and three frozen-density warm complete",
        "energy-plus-force calls per process. Failure rows retain the reason that",
        "prevented a measurement; they are not imputed or hidden.",
        "",
        "| Basis | Mode | Method | Engine | Cold (s) | Warm median [range] (s) | Status |",
        "| --- | --- | --- | --- | ---: | --- | --- |",
    ]
    for key in sorted(rows, key=sort_key):
        basis, method, mode, arm = key
        row = rows[key]
        value = row["measured"]
        if value is None:
            detail = reason(row["record"])
            lines.append(
                f"| {basis} | {mode} | {method} | {ENGINE[arm]} | - | - | "
                f"failed: {detail[:150]} |"
            )
            continue
        lines.append(
            "| %s | %s | %s | %s | %.4f | %.4f [%.4f, %.4f] | complete |"
            % (
                basis,
                mode,
                method,
                ENGINE[arm],
                value["cold"],
                value["warm"],
                value["warm_min"],
                value["warm_max"],
            )
        )
    lines.append("")
    return "\n".join(lines)


def render_timings_csv(rows: dict[tuple[str, str, str, str], dict[str, Any]]) -> str:
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=CSV_FIELDS, lineterminator="\n")
    writer.writeheader()
    for key in sorted(rows, key=sort_key):
        basis, method, mode, arm = key
        row = rows[key]
        value = row["measured"]
        if value is None:
            writer.writerow(
                {
                    "basis": basis,
                    "method": method,
                    "mode": mode,
                    "engine": ENGINE[arm],
                    "status": "failed",
                    "note": reason(row["record"])[:160],
                }
            )
            continue
        writer.writerow(
            {
                "basis": basis,
                "method": method,
                "mode": mode,
                "engine": ENGINE[arm],
                "status": "complete",
                "cold_seconds": "%.6f" % value["cold"],
                "warm_median_seconds": "%.6f" % value["warm"],
                "warm_min_seconds": "%.6f" % value["warm_min"],
                "warm_max_seconds": "%.6f" % value["warm_max"],
                "warm_n": value["warm_n"],
            }
        )
    return buffer.getvalue()


def render_comparisons(
    rows: dict[tuple[str, str, str, str], dict[str, Any]]
) -> tuple[str, int]:
    buffer = io.StringIO()
    writer = csv.DictWriter(
        buffer, fieldnames=COMPARISON_FIELDS, lineterminator="\n"
    )
    writer.writeheader()
    pairs = 0
    for key in sorted(rows, key=sort_key):
        basis, method, mode, arm = key
        if arm != "standard":
            continue
        native = rows[key]["measured"]
        reference = rows.get((basis, method, mode, "gpu4pyscf"), {}).get("measured")
        if native is None or reference is None:
            continue
        atoms = native["atoms"]
        deltas = [
            abs(a - b)
            for native_row, reference_row in zip(native["forces"], reference["forces"])
            for a, b in zip(native_row, reference_row)
        ]
        writer.writerow(
            {
                "basis": basis,
                "method": method,
                "mode": mode,
                "native_cold_seconds": "%.6f" % native["cold"],
                "reference_cold_seconds": "%.6f" % reference["cold"],
                "cold_speedup": "%.4f" % (reference["cold"] / native["cold"]),
                "native_warm_seconds": "%.6f" % native["warm"],
                "reference_warm_seconds": "%.6f" % reference["warm"],
                "warm_speedup": "%.4f" % (reference["warm"] / native["warm"]),
                "energy_abs_per_atom_hartree": "%.3e"
                % (abs(native["energy"] - reference["energy"]) / atoms),
                "force_max_abs_hartree_per_bohr": "%.3e" % max(deltas),
                "force_rms_hartree_per_bohr": "%.3e"
                % (sum(delta * delta for delta in deltas) / len(deltas)) ** 0.5,
            }
        )
        pairs += 1
    return buffer.getvalue(), pairs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()
    rows = collect()
    timings = render_timings(rows)
    timing_csv = render_timings_csv(rows)
    comparison_csv, pairs = render_comparisons(rows)
    if args.write:
        (HERE / "timings.md").write_text(timings, encoding="utf-8")
        (HERE / "timings.csv").write_text(timing_csv, encoding="utf-8")
        (HERE / "comparisons.csv").write_text(comparison_csv, encoding="utf-8")
    failures = 0
    for name, rendered in (
        ("timings.md", timings),
        ("timings.csv", timing_csv),
        ("comparisons.csv", comparison_csv),
    ):
        committed = (HERE / name).read_text(encoding="utf-8")
        if committed != rendered:
            print(f"MISMATCH {name}")
            failures += 1
    if failures:
        raise SystemExit(f"{failures} retained file(s) disagree with the bundles")
    print(f"ok: {len(rows)} rows, {pairs} complete engine pairs")


if __name__ == "__main__":
    main()
