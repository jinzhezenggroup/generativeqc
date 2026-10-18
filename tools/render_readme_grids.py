"""Reduce matched WB97M-V grid sweeps without pooling quadrature populations."""

from __future__ import annotations

import argparse
import json
import lzma
from dataclasses import asdict
from pathlib import Path
from typing import Any

from generativeqc import GridSpec

from benchmarks._readme_grid import grid_directory, parse_grid
from benchmarks.readme_omol25 import SIZES
from tools.render_readme_dft import (
    collect_point,
    phase_statistics,
    phase_unchecked,
    timing_cell,
    work_cell,
)


def collect_grid(
    root: Path, grid: tuple[int, int, int], sizes: list[int]
) -> dict[str, Any]:
    """Reject mislabeled data rather than plotting it under a directory's name."""
    points = [
        collect_point(root / grid_directory(grid) / "wb97m-v", atoms) for atoms in sizes
    ]
    expected = json.dumps(asdict(GridSpec(*grid)), sort_keys=True)
    for point in points:
        scientific = point.get("protocol")
        if scientific is not None and (
            scientific["atoms"] != point["atoms"]
            or scientific["method"] != "WB97M-V/RKS"
            or json.dumps(scientific["grid"], sort_keys=True) != expected
        ):
            raise ValueError(
                "grid timing population is filed under the wrong grid or case"
            )
    return {"dimensions": list(grid), "points": points}


def collect_sweep(
    root: Path,
    dimensions: list[tuple[int, int, int]],
    sizes: list[int],
    modes: list[str],
) -> dict[str, Any]:
    """Retain every endpoint's independent gates, work, outcomes and raw hashes."""
    return {
        "schema": "generativeqc.readme-grid-timings.v1",
        "sizes": sizes,
        "modes": {
            mode: [collect_grid(root / mode, grid, sizes) for grid in dimensions]
            for mode in modes
        },
    }


def render(summary: dict[str, Any], destination: Path) -> None:
    """Plot complete cold/warm phases with separate grid and engine identities."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    matplotlib.rcParams.update(
        {"svg.fonttype": "none", "svg.hashsalt": "readme-vv10-grids"}
    )
    modes = summary["modes"]
    figure, axes = plt.subplots(
        2, len(modes), figsize=(6 * len(modes), 7), squeeze=False
    )
    colors = {"native": "#087e8b", "reference": "#cf5c36"}
    styles = ("-", "--", "-.", ":")
    lines = [
        "# Matched-grid complete energy + analytic forces",
        "",
        "Warm values retain all requested fixed-density replays. Columns show median seconds and observed SCF iterations/builds.",
        "Same-grid engine parity and cross-grid convergence are separate gates; see grid-convergence.md.",
        "",
        "| Mode | Grid | Atoms | Engine | Cold (s) | Warm (s) | Cold work | Warm work |",
        "| --- | --- | ---: | --- | ---: | ---: | --- | --- |",
    ]
    grid_handles = []
    for column, (mode, grids) in enumerate(modes.items()):
        for index, population in enumerate(grids):
            dimensions = "×".join(str(value) for value in population["dimensions"])
            style = styles[index % len(styles)]
            if column == 0:
                grid_handles.append(
                    Line2D([], [], color="#555555", linestyle=style, label=dimensions)
                )
            for engine, color in colors.items():
                for row, phase in enumerate(("cold", "warm")):
                    values, lows, highs = [], [], []
                    for point in population["points"]:
                        entry = point["engines"][engine]
                        center, low, high = phase_statistics(entry, phase)
                        if phase_unchecked(entry, phase):
                            center, low, high = float("nan"), 0.0, 0.0
                        values.append(center)
                        lows.append(low)
                        highs.append(high)
                    axes[row, column].errorbar(
                        summary["sizes"],
                        values,
                        yerr=[lows, highs],
                        color=color,
                        linestyle=style,
                        marker="o",
                        capsize=3,
                    )
                for point in population["points"]:
                    entry = point["engines"][engine]
                    lines.append(
                        f"| {mode} | {dimensions} | {point['atoms']} | {engine} | {timing_cell(entry, 'cold')} | {timing_cell(entry, 'warm')} | {work_cell(entry, 'cold')} | {work_cell(entry, 'warm')} |"
                    )
        for row, phase in enumerate(("cold", "warm")):
            axis = axes[row, column]
            axis.set_title(
                f"{'Direct J/K' if mode == 'none' else 'Density fitting'} · {phase}"
            )
            axis.set_xticks(summary["sizes"])
            axis.set_xlabel("Atoms in water cluster")
            axis.set_ylabel("Complete E+F (s)")
            axis.set_yscale("log")
            axis.grid(axis="y", alpha=0.2)
            axis.spines[["top", "right"]].set_visible(False)
    handles = [
        Line2D([], [], color=color, label=label)
        for (engine, color), label in zip(
            colors.items(), ("GenerativeQC", "GPU4PySCF"), strict=True
        )
    ] + grid_handles
    figure.suptitle("ωB97M-V / def2-SVP · FP64 · matched moving XC/VV10 grids")
    figure.legend(
        handles=handles,
        loc="lower center",
        ncol=min(5, len(handles)),
        frameon=False,
        bbox_to_anchor=(0.5, 0.035),
    )
    figure.text(
        0.5,
        0.012,
        "Cold: fresh process with existing compile caches, preparation + first E+F. Warm: median, bars: min–max. Unchecked phases omitted.",
        ha="center",
        fontsize=9,
    )
    figure.tight_layout(rect=(0, 0.09, 1, 0.95))
    graphic = destination / "wb97mv-grids.svg"
    figure.savefig(graphic, metadata={"Date": None}, facecolor="white")
    plt.close(figure)
    graphic.write_text(
        "\n".join(line.rstrip() for line in graphic.read_text().splitlines()) + "\n"
    )
    (destination / "grid-timings.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--raw", type=Path)
    inputs.add_argument("--summary", type=Path)
    parser.add_argument(
        "--grids", type=parse_grid, nargs="+", default=[(48, 12, 24), (48, 16, 32)]
    )
    parser.add_argument("--sizes", type=int, choices=SIZES, nargs="+", default=[3, 12])
    parser.add_argument(
        "--modes", choices=("none", "cuda"), nargs="+", default=["none", "cuda"]
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    summary = (
        collect_sweep(args.raw, args.grids, args.sizes, args.modes)
        if args.raw
        else json.loads(lzma.decompress(args.summary.read_bytes()))
    )
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "grid-timings.json.xz").write_bytes(
        lzma.compress((json.dumps(summary, indent=2, allow_nan=False) + "\n").encode())
    )
    render(summary, args.output)


if __name__ == "__main__":
    main()
