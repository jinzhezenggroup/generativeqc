"""Render only the measured Standard complete endpoints from retained records."""

from __future__ import annotations

import argparse
import json
import lzma
from pathlib import Path
from statistics import median


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--directory", type=Path, default=Path(__file__).resolve().parent
    )
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    payload = json.loads(
        lzma.decompress((args.directory / "records.json.xz").read_bytes())
    )["qm7"]
    if payload["status"] != "completed" or any(
        r["policy"] != "standard" for r in payload["samples"]
    ):
        raise ValueError("only complete Standard endpoints can be plotted")
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    matplotlib.rcParams.update(
        {
            "svg.fonttype": "none",
            "svg.hashsalt": "cbs-routine-standard-20261025",
            "font.family": "DejaVu Sans",
        }
    )
    figure, axis = plt.subplots(figsize=(10, 4.8))
    carbons = (1, 2, 4, 6)
    populations = [
        [r["complete_seconds"] for r in payload["samples"] if r["carbon"] == c]
        for c in carbons
    ]
    if any(len(p) != 2 for p in populations):
        raise ValueError("expected two fresh calls per population")
    values = [median(p) for p in populations]
    axis.plot(
        carbons, values, "-o", color="#087e8b", linewidth=1.7, label="GenerativeQC"
    )
    for carbon, seconds, samples in zip(carbons, values, populations, strict=True):
        axis.vlines(carbon, min(samples), max(samples), color="#087e8b")
        axis.annotate(
            f"{seconds:.1f} s",
            (carbon, seconds),
            xytext=(0, 10),
            textcoords="offset points",
            ha="center",
            fontsize=10,
        )
    axis.set(
        yscale="log",
        xlim=(0.6, 6.4),
        xticks=carbons,
        xlabel="QM7 alkane representative (carbon count)",
        ylabel="Complete E+F wall time / s",
        title="DF-CCSD(T)/CBS · cc-pV[D,T]Z + DZ correction · RTX 5090",
    )
    axis.margins(y=0.3)
    axis.grid(axis="y", alpha=0.2)
    axis.legend(loc="upper left")
    figure.text(
        0.5,
        0.025,
        "Default settings · Full DF / RI-JK · matching JKFIT · all electrons\nMedian of two fresh owners; process/CUDA startup retained separately.",
        ha="center",
        fontsize=9,
    )
    figure.tight_layout(rect=(0, 0.09, 1, 1))
    figure.savefig(args.output, metadata={"Date": None})
    plt.close(figure)
    args.output.write_text(
        "\n".join(line.rstrip() for line in args.output.read_text().splitlines()) + "\n"
    )


if __name__ == "__main__":
    main()
