"""Replot exact endpoint samples and report their recorded independent errors.

Each engine's per-call energy/force errors against the common dense reference
are reported as values (``--error-table``). The historical acceptance-gate
counts remain in the replayed summary for archive integrity only; README
publications compare timings and report both engines' errors, not gate counts.

The compact publication retains every timing and per-call oracle error, not full
force arrays or an independent replay of the original vector subtraction. Those
source-bound original receipts remain in the disclosed ignored acquisition tree.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import lzma
import math
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from benchmarks._retention import raw_output_path
from benchmarks.readme_defaults import COMMON_GATES, METHODS, SIZES
from benchmarks.readme_defaults_report import COLORS, LABELS

SAMPLE_COLUMNS = (
    "seconds",
    "scf_iterations",
    "fock_or_effective_potential_builds",
    "preliminary_guess_focks",
    *COMMON_GATES,
)
SCHEMA = "generativeqc.readme-defaults-summary.v1"


def audit(data: dict) -> dict:
    """Reject partial populations and recompute gates from unrounded error scalars."""
    if data["schema"] != SCHEMA:
        raise ValueError("unknown compact summary schema")
    if data["common_observable_gates"] != COMMON_GATES:
        raise ValueError("scientific gates changed")
    if data["sample_columns"] != list(SAMPLE_COLUMNS):
        raise ValueError("unknown timing/error sample layout")
    expected = {
        (arm, method, mode, atoms)
        for arm in ("standard", "gpu4pyscf")
        for method in METHODS
        for mode in ("direct", "df")
        for atoms in SIZES
    }
    rows = data["populations"]
    if len(rows) != 128 or {tuple(row["key"]) for row in rows} != expected:
        raise ValueError("incomplete or duplicate population matrix")
    indexed = {tuple(row["key"]): row for row in rows}
    qualified = {}
    superseded = data.get("superseded_populations", [])
    if not isinstance(superseded, list):
        raise TypeError("invalid superseded acquisition records")
    for row, complete in [(row, True) for row in rows] + [
        (row, False) for row in superseded
    ]:
        samples = row["samples"]
        valid_count = len(samples) == 4 if complete else 1 <= len(samples) <= 4
        if (
            not valid_count
            or tuple(row["key"]) not in expected
            or any(len(sample) != 7 for sample in samples)
        ):
            raise ValueError("incomplete cold/three-warm population")
        if any(
            not all(
                type(value) in (int, float) and math.isfinite(value) and value >= 0
                for value in sample
            )
            or sample[0] <= 0
            or sample[1] < 1
            for sample in samples
        ):
            raise ValueError("invalid timing, work or oracle error")
        gates = [
            all(
                error <= gate
                for error, gate in zip(sample[4:], COMMON_GATES.values(), strict=True)
            )
            for sample in samples
        ]
        if complete:
            qualified[tuple(row["key"])] = gates
    counts = []
    for mode in ("direct", "df"):
        for phase, selection in (("cold", slice(0, 1)), ("warm", slice(1, 4))):
            wins = admitted = admitted_wins = 0
            for method in METHODS:
                for atoms in SIZES:
                    keys = [
                        (arm, method, mode, atoms) for arm in ("standard", "gpu4pyscf")
                    ]
                    native, reference = (indexed[key] for key in keys)
                    if (
                        not all(native["allocation"])
                        or native["allocation"] != reference["allocation"]
                    ):
                        raise ValueError("paired engines used different allocated GPUs")
                    medians = [
                        statistics.median(
                            sample[0] for sample in indexed[key]["samples"][selection]
                        )
                        for key in keys
                    ]
                    passed = all(all(qualified[key][selection]) for key in keys)
                    wins += medians[0] < medians[1]
                    admitted += passed
                    admitted_wins += passed and medians[0] < medians[1]
            counts.append(
                {
                    "mode": mode,
                    "phase": phase,
                    "pairs": 32,
                    "wins": wins,
                    "qualified_pairs": admitted,
                    "qualified_wins": admitted_wins,
                }
            )
    return {
        "populations": len(rows),
        "complete_endpoints": sum(len(row["samples"]) for row in rows),
        "superseded_complete_endpoints": sum(len(row["samples"]) for row in superseded),
        "wins": counts,
        "qualification": {
            arm: {
                "qualified": sum(
                    key[0] == arm and all(value) for key, value in qualified.items()
                ),
                "failed": sum(
                    key[0] == arm and not all(value) for key, value in qualified.items()
                ),
            }
            for arm in ("standard", "gpu4pyscf")
        },
    }


ERROR_LABELS = ("E/atom (Eh)", "Fmax (Eh/Bohr)", "Frms (Eh/Bohr)")


def error_table(data: dict) -> str:
    """Tabulate each engine's largest per-call error over all cold and warm calls."""
    audit(data)
    indexed = {tuple(row["key"]): row for row in data["populations"]}

    def largest(key: tuple) -> list[float]:
        samples = indexed[key]["samples"]
        return [max(sample[column] for sample in samples) for column in (4, 5, 6)]

    engines = (("standard", "GenerativeQC"), ("gpu4pyscf", "GPU4PySCF"))
    header = ["Mode", "Method", "Atoms"] + [
        f"{name} {label}" for _, name in engines for label in ERROR_LABELS
    ]
    lines = [
        "# Energy and force errors of both engines",
        "",
        "Each value is the largest absolute error over the cold call and three warm",
        "calls, measured against the same untimed dense Direct reference (tighter",
        "SCF, denser XC grids, no density fitting, full grid-response forces).",
        "DF rows therefore include density-fitting error. Generated by",
        "`benchmarks/readme_defaults_summary.py --error-table`.",
        "",
        "| " + " | ".join(header) + " |",
        "| " + " | ".join(["---"] * 3 + ["---:"] * 6) + " |",
    ]
    for mode in ("direct", "df"):
        for method in METHODS:
            for atoms in SIZES:
                values = [
                    f"{value:.1e}"
                    for arm, _ in engines
                    for value in largest((arm, method, mode, atoms))
                ]
                cells = ["Direct" if mode == "direct" else "DF", LABELS[method]]
                lines.append("| " + " | ".join([*cells, str(atoms), *values]) + " |")
    for mode in ("direct", "df"):
        maxima = [
            max(
                largest((arm, method, mode, atoms))[column]
                for method in METHODS
                for atoms in SIZES
            )
            for arm, _ in engines
            for column in range(3)
        ]
        cells = ["Direct" if mode == "direct" else "DF", "**largest**", "all"]
        lines.append(
            "| " + " | ".join([*cells, *(f"{value:.1e}" for value in maxima)]) + " |"
        )
    return "\n".join(lines) + "\n"


def render(data: dict, destination: Path) -> None:
    """Keep all completed points, linear axes, and the homepage's cold/warm encoding."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    indexed = {tuple(row["key"]): row for row in data["populations"]}
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
            figure, axes = plt.subplots(2, 4, figsize=(15, 6.8))
            for axis, method in zip(axes.flat, METHODS, strict=True):
                for arm in ("standard", "gpu4pyscf"):
                    for selection, style, marker in (
                        (slice(0, 1), "--", "s"),
                        (slice(1, 4), "-", "o"),
                    ):
                        values = [
                            [
                                sample[0]
                                for sample in indexed[(arm, method, mode, atoms)][
                                    "samples"
                                ][selection]
                            ]
                            for atoms in SIZES
                        ]
                        medians = [statistics.median(samples) for samples in values]
                        axis.errorbar(
                            SIZES,
                            medians,
                            yerr=[
                                [
                                    value - min(samples)
                                    for value, samples in zip(
                                        medians, values, strict=True
                                    )
                                ],
                                [
                                    max(samples) - value
                                    for value, samples in zip(
                                        medians, values, strict=True
                                    )
                                ],
                            ],
                            color=COLORS[arm],
                            ls=style,
                            marker=marker,
                            ms=4,
                            lw=1.7,
                            capsize=2,
                        )
                basis = (
                    "def2-mTZVPP · D4 + gCP" if method == "r2scan-3c" else "def2-SVP"
                )
                axis.set_title(
                    f"{LABELS[method]}\n{basis}", fontsize=11, loc="left", pad=10
                )
                axis.set(
                    xlim=(0, SIZES[-1] * 1.06),
                    ylim=(0, None),
                    xticks=SIZES,
                    xlabel="Atoms in water cluster",
                    ylabel="Energy + forces (s)",
                )
                axis.minorticks_off()
                axis.grid(axis="y", alpha=0.18)
                axis.spines[["top", "right"]].set_visible(False)
            handles = [
                Line2D([], [], color=COLORS[arm], lw=2, label=label)
                for arm, label in (
                    ("standard", "GenerativeQC"),
                    ("gpu4pyscf", "GPU4PySCF"),
                )
            ]
            handles.extend(
                Line2D(
                    [], [], color="#555555", ls=style, marker=marker, ms=4, label=label
                )
                for style, marker, label in (
                    ("--", "s", "Cold"),
                    ("-", "o", "Warm (3 calls)"),
                )
            )
            figure.legend(
                handles=handles,
                loc="upper center",
                bbox_to_anchor=(0.5, 1.0),
                ncol=4,
                frameon=False,
            )
            figure.text(
                0.5,
                0.018,
                f"{'Direct' if mode == 'direct' else 'Density fitting'} · {data.get('device_label', 'RTX 5090')} · cold includes setup · warm: frozen cold density · bars: min–max · all complete points",
                ha="center",
                fontsize=9,
                color="#555555",
            )
            figure.tight_layout(rect=(0, 0.06, 1, 0.93), h_pad=2, w_pad=2)
            path = destination / (mode + ".svg")
            figure.savefig(path, metadata={"Date": None})
            path.write_text(
                "\n".join(line.rstrip() for line in path.read_text().splitlines())
                + "\n"
            )
            figure.savefig(destination / (mode + ".png"), dpi=140)
            plt.close(figure)


def main() -> None:
    """Verify compact evidence before optionally regenerating the homepage figures."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", type=Path, required=True)
    parser.add_argument("--output", type=raw_output_path)
    parser.add_argument(
        "--error-table",
        type=Path,
        help="write both engines' largest per-pair errors as Markdown",
    )
    args = parser.parse_args()
    manifest = json.loads((args.campaign / "manifest.json").read_text())
    payload = (args.campaign / "samples.json.xz").read_bytes()
    if hashlib.sha256(payload).hexdigest() != manifest["samples_sha256"]:
        raise ValueError("compact samples changed")
    data = json.loads(lzma.decompress(payload))
    for name, checksum in data["references"].items():
        if hashlib.sha256((ROOT / name).read_bytes()).hexdigest() != checksum:
            raise ValueError("reviewed independent reference changed: " + name)
    summary = audit(data)
    if summary != manifest["summary"]:
        raise ValueError("recomputed timing or accuracy summary differs")
    if args.output is not None:
        render(data, args.output)
    if args.error_table is not None:
        args.error_table.write_text(error_table(data))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
