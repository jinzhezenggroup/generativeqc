"""Draw the README's cold/warm WB97M-V comparison from retained endpoint reports.

The figure describes the measured opt-in integration, not the master default.
Reuse the evidence verifier before plotting so incomplete or inaccurate pairs
cannot silently become benchmark points. No GPU or native library is needed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import lzma
import math
import statistics
import subprocess
import sys
from pathlib import Path

from tools.render_readme_benchmarks import COLORS, plt, save_svg, style_axes


def checked_reports(directory: Path) -> tuple[dict, list[dict]]:
    """Check all retained controls and bind plotted reports to their manifest."""
    manifest = json.loads((directory / "manifest.json").read_text())
    reports = []
    for atoms in sorted(map(int, manifest["comparison_modes"])):
        subprocess.run(
            [sys.executable, str(directory / "verify.py"), str(atoms)], check=True
        )
        path = directory / f"matched{atoms}-sparse.json.xz"
        stored = path.read_bytes()
        payload = lzma.decompress(stored)
        identity = manifest["files"][path.name]
        if (
            hashlib.sha256(stored).hexdigest() != identity["sha256"]
            or hashlib.sha256(payload).hexdigest() != identity["decoded_sha256"]
        ):
            raise ValueError(f"report changed after verification: {path}")
        report = json.loads(payload)
        for engine in ("native", "reference"):
            values = [row["seconds"] for row in report[f"{engine}_samples"]]
            values.append(report[f"{engine}_complete_cold"]["seconds"])
            if not all(math.isfinite(value) and value > 0 for value in values):
                raise ValueError(f"invalid endpoint time: {path}, {engine}")
        reports.append(report)
    return manifest, sorted(reports, key=lambda row: row["aos"])


def draw(directory: Path) -> None:
    """Keep one-shot cold totals separate from warm medians and observed ranges."""
    manifest, reports = checked_reports(directory)
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "svg.fonttype": "none",
            "svg.hashsalt": "generativeqc-wb97mv-active-ao",
        }
    )
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.8))
    aos = [row["aos"] for row in reports]
    for key, engine, label, style in (
        ("native", "GenerativeQC", "GenerativeQC (opt-in candidate)", "o-"),
        ("reference", "GPU4PySCF", "GPU4PySCF 1.8.1", "s--"),
    ):
        axes[0].plot(
            aos,
            [row[f"{key}_complete_cold"]["seconds"] for row in reports],
            style,
            color=COLORS[engine],
            label=label,
            linewidth=1.8,
            markersize=4,
        )
        values = [[s["seconds"] for s in row[f"{key}_samples"]] for row in reports]
        medians = [statistics.median(samples) for samples in values]
        axes[1].errorbar(
            aos,
            medians,
            yerr=[
                [m - min(s) for m, s in zip(medians, values, strict=True)],
                [max(s) - m for m, s in zip(medians, values, strict=True)],
            ],
            fmt=style,
            color=COLORS[engine],
            label=label,
            capsize=3,
            linewidth=1.8,
            markersize=4,
        )
    for ax, title in zip(
        axes,
        ("Cold · prepare + first energy / forces", "Warm · energy + analytic forces"),
        strict=True,
    ):
        style_axes(ax, title, aos)
        ax.set_ylabel("Complete endpoint / s")
        ax.margins(x=0.10, y=0.15)
    fig.suptitle("ωB97M-V · spherical def2-SVP · RTX 5090", weight="bold", y=0.98)
    fig.text(
        0.5,
        0.91,
        f"Candidate {manifest['source']['source_commit'][:9]} · matched unpruned grids · lower is better",
        ha="center",
        fontsize=9,
    )
    fig.legend(
        *axes[0].get_legend_handles_labels(),
        loc="upper center",
        bbox_to_anchor=(0.5, 0.89),
        ncol=2,
        frameon=False,
        fontsize=9,
    )
    fig.text(
        0.5,
        0.065,
        "Cold: one run / size, including engine construction and preparation",
        ha="center",
        fontsize=9,
    )
    fig.text(
        0.5,
        0.025,
        "Warm: median and min–max of 3 fixed-density runs / engine · one SCF iteration each",
        ha="center",
        fontsize=9,
    )
    fig.tight_layout(rect=(0, 0.11, 1, 0.82))
    save_svg(fig, directory / "wb97mv.svg")
    plt.close(fig)


def main() -> None:
    """Regenerate the tracked SVG from the hash-bound benchmark evidence."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--directory",
        type=Path,
        default=Path("benchmarks/results/wb97mv-active-ao-20261003"),
    )
    draw(parser.parse_args().directory)


if __name__ == "__main__":
    main()
