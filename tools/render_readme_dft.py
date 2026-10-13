"""Publish independently gated cold/warm DFT endpoints from retained evidence.

Only complete, accurate runs become plotted timings. Raw arrays remain in the
ignored run directory; the compact archive keeps work counts, residuals,
independent cold/moved oracles, errors and hashes of every original record.
"""

from __future__ import annotations

import argparse
import gzip
import json
import lzma
import math
from pathlib import Path
from statistics import median
from typing import Any

from benchmarks.readme_dft import METHODS, SCHEMA
from tools.render_omol25_benchmarks import collect

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
COLORS = {"native": "#087e8b", "reference": "#cf5c36"}


def phase_rows(entry: dict[str, Any], phase: str) -> list[dict[str, Any]]:
    """Failed or partial endpoints never supply timings to the figure or table."""
    if entry["status"] != "measured":
        return []
    return [row for row in entry["records"] if row["phase"] == phase]


def stable_work(rows: list[dict[str, Any]]) -> bool:
    """Identify differing SCF work without discarding complete endpoint samples."""
    return (
        len(
            {
                (row["iterations"], row.get("fock_builds", row.get("scf_jk_builds")))
                for row in rows
            }
        )
        == 1
    )


def timing_cell(entry: dict[str, Any], phase: str) -> str:
    """Report an endpoint median, marking the full range when SCF work varies."""
    rows = phase_rows(entry, phase)
    if not rows:
        return entry["status"].replace("_", " ")
    values = [row["complete_seconds"] for row in rows]
    if not stable_work(rows):
        return f"{median(values):.3f} [{min(values):.3f}–{max(values):.3f}]†"
    return f"{median(values):.3f}"


def work_cell(entry: dict[str, Any], phase: str) -> str:
    """Keep each observed iteration/build pair visible beside endpoint timings."""
    pairs = {
        (row["iterations"], row.get("fock_builds", row.get("scf_jk_builds")))
        for row in phase_rows(entry, phase)
    }
    return (
        ", ".join(
            f"{iterations}/{builds if builds is not None else '—'}"
            for iterations, builds in sorted(pairs)
        )
        or "—"
    )


def collect_point(directory: Path, size: int) -> dict[str, Any]:
    """Retain complete endpoint medians and correct the provenance sampling label."""
    point = collect(directory, size, schema=SCHEMA)
    for entry in point["engines"].values():
        # Slurm can terminate the whole runner before its per-point timeout
        # writes an outcome. Preserve the observed signal exit, not a fake 124.
        if (entry.get("outcome") or {}).get("reason") == "slurm_time_limit":
            entry["status"] = "timeout"
        elif (entry.get("outcome") or {}).get("reason") in (
            "operator_cancelled",
            "stopped_after_12_atom_energy_gate_failure",
        ):
            # An explicit diagnostic stop is distinct from exhausting a time
            # limit. Preserve partial records, but never plot their timings.
            entry["status"] = "interrupted"
        state = entry.get("environment", {}).get("accelerator", {}).get("nvidia_smi")
        if state:
            # This driver samples provenance before setup/measurement. The
            # helper's legacy label describes the HF caller's sampling order.
            state["raw_sampling_point_label"] = state["sampling_point"]
            state["sampling_point"] = "before endpoint measurements"
    return point


def phase_statistics(entry: dict[str, Any], phase: str) -> tuple[float, float, float]:
    """Only missing/failed endpoints break a line; SCF variability stays measured.

    These are wall-time statistics of the requested endpoint, not estimates for
    fixed SCF work. Keep every accepted repeat and display its full range.
    """
    rows = phase_rows(entry, phase)
    if not rows:
        return float("nan"), 0.0, 0.0
    times = [row["complete_seconds"] for row in rows]
    center = median(times)
    return center, center - min(times), max(times) - center


def render(summary: dict[str, Any], destination: Path) -> None:
    """Draw all measured sizes, including cold and fixed-density warm execution."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    matplotlib.rcParams.update(
        {
            "svg.fonttype": "none",
            "svg.hashsalt": "readme-dft-cold-warm",
            "font.family": "DejaVu Sans",
        }
    )
    methods = list(summary["methods"])
    composite_note = (
        "r²SCAN-3c: native CPU gCP; reference CPU D4 + gCP. "
        if "r2scan-3c" in methods
        else ""
    )
    ncols = 4 if len(methods) >= 7 or len(methods) == 4 else min(3, len(methods))
    nrows = math.ceil(len(methods) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(15, 3.4 * nrows), squeeze=False)
    for ax, method in zip(axes.flat, methods, strict=False):
        points = summary["methods"][method]
        for engine, color in COLORS.items():
            for phase, style in (("cold", "--"), ("warm", "-")):
                xs, ys, lows, highs = [], [], [], []
                for point in points:
                    rows = phase_rows(point["engines"][engine], phase)
                    x = point["atoms"]
                    center, low, high = phase_statistics(
                        point["engines"][engine], phase
                    )
                    if rows and not stable_work(rows):
                        ax.scatter(
                            [x] * len(rows),
                            [row["complete_seconds"] for row in rows],
                            color=color,
                            marker="x",
                            s=24,
                        )
                    xs.append(x)
                    ys.append(center)
                    lows.append(low)
                    highs.append(high)
                ax.errorbar(
                    xs,
                    ys,
                    yerr=[lows, highs],
                    color=color,
                    linestyle=style,
                    marker="o" if phase == "warm" else "s",
                    markersize=4,
                    linewidth=1.7,
                    capsize=3,
                )
        basis = (
            "def2-mTZVPP · includes D4 + gCP" if method == "r2scan-3c" else "def2-SVP"
        )
        ax.set_title(f"{LABELS[method]}\n{basis}", fontsize=11, loc="left", pad=10)
        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        ax.set_xticks(summary["sizes"], [str(size) for size in summary["sizes"]])
        ax.set_xlabel("Atoms in water cluster")
        ax.set_ylabel("Energy + forces (s)")
        ax.grid(axis="y", alpha=0.18)
        ax.spines[["top", "right"]].set_visible(False)
        missing = []
        for engine, label in (
            ("native", "Native not qualified"),
            ("reference", "Reference incomplete"),
        ):
            sizes = [
                str(point["atoms"])
                for point in points
                if point["engines"][engine]["status"] != "measured"
            ]
            if sizes:
                missing.append(f"{label}: {', '.join(sizes)} atoms")
        if missing:
            ax.text(
                0.98,
                0.03,
                "\n".join(missing),
                transform=ax.transAxes,
                va="bottom",
                ha="right",
                fontsize=7.5,
                bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.85},
            )
    for ax in list(axes.flat)[len(methods) :]:
        ax.axis("off")
        ax.text(0.03, 0.80, "Lower is faster", fontsize=15, weight="bold")
        ax.text(
            0.03,
            0.68,
            "Cold: preparation + first calculation\nWarm: fixed post-cold density\nError bars: minimum–maximum\nCrosses: differing SCF work\n\n"
            + composite_note,
            va="top",
            fontsize=10,
            linespacing=1.7,
        )
    legend = [
        Line2D(
            [],
            [],
            color=COLORS[engine],
            linestyle=style,
            marker=marker,
            label=f"{label} · {phase}",
        )
        for engine, label in (("native", "GenerativeQC"), ("reference", "GPU4PySCF"))
        for phase, style, marker in (("cold", "--", "s"), ("warm", "-", "o"))
    ]
    fig.legend(
        handles=legend,
        loc="upper center",
        ncol=4,
        frameon=False,
        bbox_to_anchor=(0.5, 1.0),
    )
    route = "DF · cc-pVDZ-JKFIT" if summary.get("density_fitting") else "Direct"
    fig.text(
        0.5,
        0.018,
        f"{route} · lower is faster · warm: median of 5 fixed-density replays · bars: min–max · crosses: varying SCF work\n"
        f"Cold includes preparation. {composite_note}Failed/incomplete points remain gaps.",
        ha="center",
        fontsize=9,
    )
    fig.tight_layout(rect=(0, 0.08, 1, 0.93), h_pad=2, w_pad=2)
    prefix = "df" if summary.get("density_fitting") else "dft"
    figure = destination / f"{prefix}-cold-warm.svg"
    fig.savefig(figure, metadata={"Date": None}, facecolor="white")
    plt.close(fig)
    figure.write_text(
        "\n".join(line.rstrip() for line in figure.read_text().splitlines()) + "\n"
    )

    lines = [
        f"# Complete {route} endpoint timings",
        "",
        "| Functional | Atoms (AOs) | GenerativeQC cold (s) | GPU4PySCF cold (s) | GenerativeQC warm (s) | GPU4PySCF warm (s) |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for method, points in summary["methods"].items():
        for point in points:
            cells = [
                timing_cell(point["engines"][engine], phase)
                for phase in ("cold", "warm")
                for engine in ("native", "reference")
            ]
            lines.append(
                f"| {LABELS[method]} | {point['atoms']} ({point.get('protocol', {}).get('aos', '—')}) | "
                + " | ".join(cells)
                + " |"
            )
    lines += [
        "",
        "Warm cells show the median of all five complete endpoints. † marks differing",
        "SCF work and includes the full range; timings are never normalized by iterations.",
        *([composite_note.strip()] if composite_note else []),
        "",
        "## Observed SCF work",
        "",
        "Each cell lists iterations/builds. Builds are native Fock builds or reference",
        "SCF J/K calls, including its initial potential. Multiple pairs retain all",
        "observed warm branches. Counts do not imply equivalent work or stopping rules",
        "between engines; complete endpoint time includes analytic forces as well.",
        "",
        "| Functional | Atoms | GenerativeQC cold | GPU4PySCF cold | GenerativeQC warm | GPU4PySCF warm |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for method, points in summary["methods"].items():
        for point in points:
            cells = [
                work_cell(point["engines"][engine], phase)
                for phase in ("cold", "warm")
                for engine in ("native", "reference")
            ]
            lines.append(
                f"| {LABELS[method]} | {point['atoms']} | " + " | ".join(cells) + " |"
            )
    lines += [
        "",
        "The [measurement report](README.md) describes the protocol and unsuccessful",
        "points; "
        + (
            "[df-summary.json.xz](df-summary.json.xz)"
            if summary.get("density_fitting")
            else "[summary.json.xz](summary.json.xz)"
        )
        + " retains all scalar observations,",
        "residuals, errors and independent cold/moved reference forces.",
        "",
    ]
    (
        destination
        / ("df-timings.md" if summary.get("density_fitting") else "timings.md")
    ).write_text("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--raw", type=Path)
    inputs.add_argument("--summary", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sizes", nargs="+", type=int, default=[3, 12, 48, 96])
    parser.add_argument("--methods", nargs="+", choices=METHODS, default=list(METHODS))
    parser.add_argument(
        "--density-fitting",
        action="store_true",
        help="Label a separately gated DF campaign",
    )
    parser.add_argument(
        "--source-commit",
        help="Exact native source revision, recorded alongside binary and consumer hashes",
    )
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    if args.raw:
        summary = {
            "schema": "generativeqc.readme-dft-summary.v1",
            "source_commit": args.source_commit,
            "sizes": args.sizes,
            "density_fitting": args.density_fitting,
            "methods": {
                method: [collect_point(args.raw / method, size) for size in args.sizes]
                for method in args.methods
            },
        }
        payload = (json.dumps(summary, indent=2, allow_nan=False) + "\n").encode()
        name = "df-summary.json.xz" if args.density_fitting else "summary.json.xz"
        (args.output / name).write_bytes(lzma.compress(payload))
    else:
        # Older committed campaigns use gzip; xz retains the same JSON exactly
        # while compressing repeated force/provenance diagnostics more compactly.
        decompress = (
            lzma.decompress if args.summary.suffix == ".xz" else gzip.decompress
        )
        summary = json.loads(decompress(args.summary.read_bytes()))
    render(summary, args.output)


if __name__ == "__main__":
    main()
