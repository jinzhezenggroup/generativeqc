"""Reduce complete default endpoints and redraw README latency figures offline.

No speedup or equal-accuracy claim is inferred from these measurements. Partial
warm populations and unsuccessful calls remain in the report, never as medians.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import lzma
import math
import statistics
from pathlib import Path

from benchmarks._retention import raw_output_path
from benchmarks.readme_defaults import ARMS, MEASUREMENT_ARMS, METHODS, SCHEMA, SIZES

LABELS = {
    "hf": "HF",
    "pbe": "PBE",
    "pbe0": "PBE0",
    "b3lyp": "B3LYP",
    "r2scan": "r²SCAN",
    "r2scan-3c": "r²SCAN-3c",
    "m06-2x": "M06-2X",
    "wb97m-v": "ωB97M-V",
}
ARM_LABELS = {
    "standard": "GenerativeQC Standard",
    "fast": "GenerativeQC Fast",
    "gpu4pyscf": "GPU4PySCF Default",
    "gpu4pyscf-fast": "GPU4PySCF Fast",
}
COLORS = dict(zip(MEASUREMENT_ARMS, ("#1768ac", "#00a087", "#d95f02"), strict=True))


def checked_sample(sample: dict, record: dict) -> None:
    """Reject failed/nonfinite endpoints and missing executed Fast evidence."""
    if sample.get("converged") is not True:
        raise ValueError("unconverged endpoint")
    for key in ("seconds", "energy"):
        value = sample.get(key)
        if type(value) not in (int, float) or not math.isfinite(value):
            raise ValueError("nonfinite endpoint")
    if sample["seconds"] <= 0:
        raise ValueError("nonpositive latency")
    forces = sample.get("forces")
    if (
        not isinstance(forces, list)
        or len(forces) != record["atoms"]
        or any(
            not isinstance(row, list)
            or len(row) != 3
            or any(type(x) not in (float, int) or not math.isfinite(x) for x in row)
            for row in forces
        )
    ):
        raise ValueError("incomplete analytic force export")
    if type(sample.get("iterations")) is not int or sample["iterations"] < 1:
        raise ValueError("missing SCF work")
    if record["arm"] == "fast":
        report = sample.get("precision_validation")
        if (
            not isinstance(report, dict)
            or report.get("profile") != "fast"
            or not report.get("solves")
        ):
            raise ValueError("missing Fast solve provenance")
        if report.get("solve_count") != len(report["solves"]):
            raise ValueError("incomplete Fast solve census")


def reduce_record(record: dict) -> dict:
    """Keep valid completed phases from a failed or externally timed-out job."""
    if record.get("schema") != SCHEMA or record.get("arm") not in ARMS:
        raise ValueError("wrong benchmark schema or arm")
    if record.get("method") not in METHODS or record.get("mode") not in (
        "direct",
        "df",
    ):
        raise ValueError("unknown scientific workload")
    if (
        record.get("atoms") not in SIZES
        or type(record.get("warm_repeats")) is not int
        or record["warm_repeats"] < 1
    ):
        raise ValueError("invalid population dimensions")
    samples = record.get("samples", [])
    for index, sample in enumerate(samples):
        expected = ("cold", 0) if index == 0 else ("warm", index - 1)
        if (sample.get("phase"), sample.get("repeat")) != expected:
            raise ValueError("duplicate, missing, or reordered endpoint")
        checked_sample(sample, record)
    if len(samples) > record["warm_repeats"] + 1:
        raise ValueError("too many repeats")
    result = {key: record[key] for key in ("arm", "method", "mode", "atoms", "status")}
    result.update(
        error=record.get("error"),
        slurm_job_id=record.get("slurm_job_id"),
        source_revision=record.get("source_revision"),
        cold=None,
        warm=None,
    )
    if str(record.get("error", "")).startswith("NotImplementedError:"):
        result["status"] = "unsupported"
    for phase in ("cold", "warm"):
        rows = [row for row in samples if row["phase"] == phase]
        required = 1 if phase == "cold" else record["warm_repeats"]
        if len(rows) != required:
            continue
        times = [row["seconds"] for row in rows]
        result[phase] = {
            "seconds": times,
            "median": statistics.median(times),
            "minimum": min(times),
            "maximum": max(times),
            "iterations": [row["iterations"] for row in rows],
            "fock_builds": [
                row.get("fock_builds", row.get("effective_potential_builds"))
                for row in rows
            ],
            "fast_status": [
                (row.get("precision_validation") or {}).get("status") for row in rows
            ],
            "fast_solve_count": [
                (row.get("precision_validation") or {}).get("solve_count")
                for row in rows
            ],
        }
    if record["status"] == "complete" and (
        result["cold"] is None or result["warm"] is None
    ):
        raise ValueError("completed job has an incomplete population")
    return result


def render(
    records: list[dict], destination: Path, *, qualifications: list[dict] | None = None
) -> list[dict]:
    """Render absolute latencies; leave missing points as visible curve gaps."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    rows = [reduce_record(record) for record in records]
    indexed = {(r["mode"], r["method"], r["arm"], r["atoms"]): r for r in rows}
    if len(indexed) != len(rows):
        raise ValueError("duplicate measurement population")
    if qualifications is not None:
        from benchmarks.readme_defaults_qualification import identity

        gates = {q["receipt_sha256"]: q for q in qualifications}
        if len(gates) != len(qualifications) or set(gates) != {
            identity(r) for r in records
        }:
            raise ValueError("qualification does not bind exactly these receipts")
        for record, row in zip(records, rows, strict=True):
            row["qualification"] = gates[identity(record)]
    revisions = {row["source_revision"] for row in rows}
    if len(revisions) != 1 or None in revisions:
        raise ValueError("mixed or missing measured revisions")
    repeat_counts = {record["warm_repeats"] for record in records}
    if len(repeat_counts) != 1:
        raise ValueError("mixed warm repeat counts")
    warm_repeats = next(iter(repeat_counts))
    destination.mkdir(parents=True, exist_ok=True)
    with plt.rc_context(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "svg.fonttype": "none",
            "svg.hashsalt": "readme-defaults-v1",
        }
    ):
        for mode in ("direct", "df"):
            fig, axes = plt.subplots(2, 4, figsize=(15, 6.8))
            for ax, method in zip(axes.flat, METHODS, strict=True):
                for arm in MEASUREMENT_ARMS:
                    for phase, style, marker in (
                        ("cold", "--", "s"),
                        ("warm", "-", "o"),
                    ):
                        values, lower, upper = [], [], []
                        for size in SIZES:
                            row = indexed.get((mode, method, arm, size), {})
                            population = row.get(phase)
                            if (
                                qualifications is not None
                                and row.get("qualification", {})
                                .get(phase, {})
                                .get("status")
                                != "qualified"
                            ):
                                population = None
                            value = population["median"] if population else math.nan
                            values.append(value)
                            lower.append(
                                value - population["minimum"] if population else 0
                            )
                            upper.append(
                                population["maximum"] - value if population else 0
                            )
                        ax.errorbar(
                            SIZES,
                            values,
                            yerr=[lower, upper],
                            color=COLORS[arm],
                            ls=style,
                            marker=marker,
                            ms=4,
                            lw=1.4,
                            capsize=2,
                        )
                basis = (
                    "def2-mTZVPP · D4 + gCP" if method == "r2scan-3c" else "def2-SVP"
                )
                ax.set_title(
                    f"{LABELS[method]}\n{basis}", fontsize=11, loc="left", pad=10
                )
                ax.set(yscale="log", xscale="log", xticks=SIZES)
                ax.set_xlabel("Atoms in water cluster")
                ax.set_ylabel("Energy + forces (s)")
                ax.set_xticklabels([str(size) for size in SIZES])
                ax.minorticks_off()
                ax.grid(axis="y", alpha=0.18)
                ax.spines[["top", "right"]].set_visible(False)
            handles = [
                Line2D([], [], color=COLORS[arm], lw=2, label=ARM_LABELS[arm])
                for arm in MEASUREMENT_ARMS
                if any(
                    row["mode"] == mode and row["arm"] == arm and row["cold"]
                    for row in rows
                )
            ]
            handles.extend(
                [
                    Line2D(
                        [],
                        [],
                        color="#555555",
                        ls=style,
                        marker=marker,
                        ms=4,
                        label=label,
                    )
                    for style, marker, label in (
                        ("--", "s", "Cold"),
                        ("-", "o", f"Warm ({warm_repeats} calls)"),
                    )
                ]
            )
            fig.legend(
                handles=handles,
                loc="upper center",
                bbox_to_anchor=(0.5, 1.0),
                ncol=len(handles),
                frameon=False,
            )
            fig.text(
                0.5,
                0.018,
                f"{'Direct' if mode == 'direct' else 'Density fitting'} · RTX 5090 · cold includes setup · warm: frozen cold density · bars: min–max"
                + (" · gaps: unqualified" if qualifications is not None else ""),
                ha="center",
                fontsize=9,
                color="#555555",
            )
            fig.tight_layout(rect=(0, 0.06, 1, 0.93), h_pad=2, w_pad=2)
            svg = destination / f"{mode}.svg"
            fig.savefig(svg, metadata={"Date": None})
            # Matplotlib emits trailing blanks inside multiline SVG paths.
            # Newlines already delimit path tokens; stripping blanks preserves
            # the drawing while keeping generated assets clean in Git diffs.
            svg.write_text(
                "\n".join(line.rstrip() for line in svg.read_text().splitlines()) + "\n"
            )
            plt.close(fig)
    return rows


def load_records(paths: list[Path], *, schema: str = SCHEMA) -> list[dict]:
    """Accept raw receipts or retained compressed bundles for exact offline replay."""
    records = []
    for path in paths:
        data = (
            lzma.decompress(path.read_bytes())
            if path.suffix == ".xz"
            else path.read_bytes()
        )
        value = json.loads(data)
        if isinstance(value, list):
            records.extend(value)
        elif value.get("schema") == schema:
            records.append(value)
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--receipts", type=Path, nargs="+", required=True)
    parser.add_argument("--oracles", type=Path, nargs="+")
    parser.add_argument("--output", type=raw_output_path, required=True)
    args = parser.parse_args()
    records = load_records(args.receipts)
    qualifications = None
    if args.oracles:
        from benchmarks.readme_defaults_qualification import ORACLE_SCHEMA, qualify

        qualifications = qualify(
            records, load_records(args.oracles, schema=ORACLE_SCHEMA)
        )
    rows = render(records, args.output, qualifications=qualifications)
    if qualifications is not None:
        (args.output / "qualification.json").write_text(
            json.dumps(qualifications, indent=2, allow_nan=False) + "\n"
        )
    (args.output / "summary.json").write_text(
        json.dumps(rows, indent=2, allow_nan=False) + "\n"
    )
    lines = [
        "# Default endpoint timings",
        "",
        "Absolute latency; qualification is limited to the sampled static geometries."
        if qualifications is not None
        else "Absolute latency; accuracy unverified.",
        "",
        "| Mode | Method | Atoms | Engine | Cold (s) | Warm median [range] (s) | Status |",
        "|---|---|---:|---|---:|---|---|",
    ]
    for row in sorted(
        [row for row in rows if row["arm"] in MEASUREMENT_ARMS],
        key=lambda r: (
            r["mode"],
            list(METHODS).index(r["method"]),
            r["atoms"],
            r["arm"],
        ),
    ):
        cold = f"{row['cold']['median']:.4f}" if row["cold"] else "—"
        warm = row["warm"]
        warm_text = (
            f"{warm['median']:.4f} [{warm['minimum']:.4f}, {warm['maximum']:.4f}]"
            if warm
            else "—"
        )
        status = row["status"]
        if qualifications is not None:
            status += "; accuracy " + row["qualification"]["status"]
        lines.append(
            f"| {row['mode']} | {LABELS[row['method']]} | {row['atoms']} | {ARM_LABELS[row['arm']]} | {cold} | {warm_text} | {status} |"
        )
    (args.output / "timings.md").write_text("\n".join(lines) + "\n")
    identities = {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in [*args.receipts, *(args.oracles or [])]
    }
    (args.output / "receipt-identities.json").write_text(
        json.dumps(identities, indent=2) + "\n"
    )


if __name__ == "__main__":
    main()
