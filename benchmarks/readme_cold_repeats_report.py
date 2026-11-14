"""Replay all independent cold repeats, full-vector errors and paper-size figures.

No timing or accuracy filtering is performed. Missing or failed processes stop
publication. Accuracy is reported separately, including historical gate failures
from approximation/grid errors, without promoting them to scientific acceptance.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import lzma
from itertools import pairwise
from pathlib import Path

from benchmarks._retention import raw_output_path
from benchmarks.readme_cold_repeats import SCHEMA, aggregate, population_order
from benchmarks.readme_defaults import COMMON_GATES, METHODS, SIZES, save
from benchmarks.readme_defaults_qualification import qualify

COLORS = {"standard": "#087e8b", "gpu4pyscf": "#cf5c36"}
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


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(campaign: Path) -> tuple[list[dict], list[dict]]:
    """Authenticate every original receipt before deriving a complete summary."""
    status = json.loads((campaign / "campaign.json").read_text())
    if status["schema"] != SCHEMA or status["status"] != "complete":
        raise ValueError("publication requires a complete campaign")
    records = json.loads((campaign / "records.json").read_text())["records"]
    expected = {
        f"{method}-{mode}-{atoms}-{arm}-r{repeat}": repeat
        for repeat in range(3)
        for method, mode, atoms, arm in population_order(repeat)
    }
    commands = {}
    for command in status.get("commands", []):
        if command.get("stage") != "formal":
            continue
        name = command.get("name")
        if (
            name not in expected
            or name in commands
            or type(command.get("process_repeat")) is not int
            or command["process_repeat"] != expected[name]
            or command.get("status") != "complete"
            or type(command.get("returncode")) is not int
            or command["returncode"] != 0
        ):
            raise ValueError("invalid formal process command census")
        commands[name] = command
    if set(commands) != set(expected):
        raise ValueError("incomplete formal process command census")
    seen = set()
    for record in records:
        name = "{method}-{mode}-{atoms}-{arm}-r{process_repeat}".format(**record)
        filename = Path(record["raw_file"]).name
        if name not in commands or name in seen or filename != name + ".json":
            raise ValueError("raw receipt does not own a distinct formal process")
        seen.add(name)
        command = commands[name]
        argv = command.get("argv", [])
        controls = {
            "--method": record["method"],
            "--mode": record["mode"],
            "--atoms": str(record["atoms"]),
            "--warm-repeats": "3",
            "--source-revision": record["source_revision"],
            "--output": record["raw_file"],
        }
        if (
            not isinstance(argv, list)
            or argv[1:4] != ["-m", "benchmarks.readme_defaults", record["arm"]]
            or any(
                argv.count(flag) != 1
                or argv.index(flag) + 1 >= len(argv)
                or argv[argv.index(flag) + 1] != value
                for flag, value in controls.items()
            )
            or command.get("raw_sha256") != record["raw_sha256"]
        ):
            raise ValueError("raw receipt differs from its formal process command")
        path = campaign / "formal" / filename
        if sha(path) != record["raw_sha256"]:
            raise ValueError("raw endpoint differs: " + path.name)
        raw = json.loads(path.read_text())
        if raw != {
            k: v
            for k, v in record.items()
            if k not in ("process_repeat", "raw_file", "raw_sha256")
        }:
            raise ValueError("collected endpoint differs from original")
    if seen != set(commands):
        raise ValueError("unconsumed formal process command")
    return records, aggregate(records)


def verify_build(
    records: list[dict], build: dict, execution: dict | None = None
) -> None:
    """Bind measured binaries and recorder bytes to the completed build receipt."""
    if build.get("status") != "BUILD_PASS_SCIENCE_NOT_RUN":
        raise ValueError("native build did not complete")
    for stage in ("configure", "build", "aot-audit"):
        matches = [row for row in build["stages"] if row["name"] == stage]
        if (
            len(matches) != 1
            or matches[0].get("status") != "PASS"
            or matches[0].get("returncode") != 0
        ):
            raise ValueError("missing successful native build/AOT stage")
    source = build["source"]
    runner = source["overlay_sha256"]["benchmarks/readme_defaults.py"]
    device_owner = build
    if execution is not None:
        if (
            execution.get("schema") != "generativeqc.paper-cold3-execution.v1"
            or execution.get("status")
            != "READY_SOURCE_MATCHED_AUTHENTICATED_BUILD_REUSE"
            or execution.get("source_revision") != source["revision"]
            or execution.get("native_source_identity")
            != build["native_source_identity"]
            or execution.get("library_sha256") != build["library_sha256"]
            or not execution.get("scheduler", {}).get("SLURM_JOB_ID")
        ):
            raise ValueError(
                "execution allocation does not bind the authenticated build"
            )
        device_owner = execution
    expected_uuid = next(csv.reader([device_owner["device"]]))[1].strip()
    packages_by_arm = {}
    for record in records:
        if (
            record["source_revision"] != source["revision"]
            or record["runner_sha256"] != runner
        ):
            raise ValueError("measured source/recorder differs from frozen build")
        accelerator = record["accelerator"]
        if (
            accelerator["nvidia_smi"]["uuid"] != expected_uuid
            or accelerator["pci_bus_id"].lower() != device_owner["pci_bus_id"].lower()
        ):
            raise ValueError("measured physical GPU differs from allocated build")
        if (
            execution is not None
            and record.get("slurm_job_id") != execution["scheduler"]["SLURM_JOB_ID"]
        ):
            raise ValueError("measured Slurm allocation differs from execution binding")
        packages = record["environment"]["packages"]
        if (
            not packages
            or packages_by_arm.setdefault(record["arm"], packages) != packages
        ):
            raise ValueError("measurement software environment changed")
        if record["arm"] == "standard":
            native = record["native_build"]
            if (
                native["library_sha256"] != build["library_sha256"]
                or native["source_identity"] != build["native_source_identity"]
            ):
                raise ValueError("measured native image differs from completed build")


def accuracy(records: list[dict], oracles: list[dict], reuse: dict) -> list[dict]:
    """Use reviewed scientific-input bindings; list actual measurement environments."""
    # The caller supplies an explicit reuse contract. Never synthesize permission
    # to substitute a different oracle, geometry, package set or source identity.
    return [
        {**result, "process_repeat": repeat}
        for repeat in range(3)
        for result in qualify(
            [r for r in records if r["process_repeat"] == repeat],
            oracles,
            reference_reuse=reuse,
        )
    ]


def diagnostics(rows: list[dict]) -> dict:
    """Report nonmonotonic medians without repairing or excluding any point."""
    index = {tuple(row["key"]): row for row in rows}
    decreases, comparisons = [], []
    for mode in ("direct", "df"):
        for method in METHODS:
            for arm in COLORS:
                for left, right in pairwise(SIZES):
                    a, b = [
                        index[(arm, method, mode, n)]["cold"]["median"]
                        for n in (left, right)
                    ]
                    if b < a:
                        decreases.append(
                            {
                                "arm": arm,
                                "method": method,
                                "mode": mode,
                                "atoms": [left, right],
                                "seconds": [a, b],
                                "relative_drop": (a - b) / a,
                            }
                        )
            for atoms in SIZES:
                a, b = [index[(arm, method, mode, atoms)] for arm in COLORS]
                comparisons.append(
                    {
                        "method": method,
                        "mode": mode,
                        "atoms": atoms,
                        **{
                            phase + "_reference_over_native": b[phase]["median"]
                            / a[phase]["median"]
                            for phase in ("cold", "warm")
                        },
                    }
                )
    by_process = {str(repeat): [] for repeat in range(3)}
    for repeat in range(3):
        for mode in ("direct", "df"):
            for method in METHODS:
                for arm in COLORS:
                    for left, right in pairwise(SIZES):
                        a, b = [
                            index[(arm, method, mode, n)]["cold"]["seconds"][repeat]
                            for n in (left, right)
                        ]
                        if b < a:
                            by_process[str(repeat)].append(
                                {
                                    "arm": arm,
                                    "method": method,
                                    "mode": mode,
                                    "atoms": [left, right],
                                    "seconds": [a, b],
                                    "relative_drop": (a - b) / a,
                                }
                            )
    return {
        "cold_decreases": decreases,
        "cold_decreases_by_process": by_process,
        "comparisons": comparisons,
        "wins": {
            phase: sum(row[phase + "_reference_over_native"] > 1 for row in comparisons)
            for phase in ("cold", "warm")
        },
        "interpretation": "Descriptive n=3 process ranges; no statistical/hardware-only speedup or causal noise attribution.",
    }


def render(rows: list[dict], label: str, output: Path) -> None:
    """Reuse the paper's palette/layout with explicit independent-process bars."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.ticker import MaxNLocator

    indexed = {tuple(row["key"]): row for row in rows}
    style = {
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
        "font.size": 8,
        "axes.titlesize": 8,
        "axes.labelsize": 8,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "legend.fontsize": 7,
        "pdf.fonttype": 42,
        "svg.fonttype": "none",
    }
    with plt.rc_context(style):
        fig, axes = plt.subplots(4, 4, figsize=(7.2, 7.6))
        for number, axis in enumerate(axes.flat):
            mode = ("direct", "df")[number // 8]
            method = list(METHODS)[number % 8]
            highest = 0.0
            for arm, color in COLORS.items():
                for phase, line, marker in (("cold", "--", "s"), ("warm", "-", "o")):
                    points = [
                        indexed[(arm, method, mode, atoms)][phase] for atoms in SIZES
                    ]
                    values = [p["median"] for p in points]
                    axis.errorbar(
                        SIZES,
                        values,
                        yerr=[
                            [p["median"] - p["min"] for p in points],
                            [p["max"] - p["median"] for p in points],
                        ],
                        color=color,
                        linestyle=line,
                        marker=marker,
                        markersize=2.6,
                        linewidth=1,
                        capsize=1.5,
                        elinewidth=0.7,
                    )
                    highest = max(highest, *(p["max"] for p in points))
            locator = MaxNLocator(nbins=5, steps=[1, 2, 5, 10])
            axis.set_ylim(
                0, min(t for t in locator.tick_values(0, highest) if t >= highest)
            )
            axis.yaxis.set_major_locator(locator)
            axis.set_xlim(0, 100)
            axis.set_xticks(SIZES)
            axis.set_title(
                f"{mode.upper() if mode == 'df' else 'Direct'} {LABELS[method]}",
                loc="left",
                pad=3,
            )
            axis.grid(axis="y", alpha=0.18)
            axis.spines[["top", "right"]].set_visible(False)
            if number >= 12:
                axis.set_xlabel("Atoms")
            if number % 4 == 0:
                axis.set_ylabel("Time (s)")
        handles = [
            Line2D([], [], color=COLORS[a], label=name)
            for a, name in (("standard", "GenerativeQC"), ("gpu4pyscf", "GPU4PySCF"))
        ]
        handles += [
            Line2D(
                [],
                [],
                color="#555555",
                linestyle="--",
                marker="s",
                markersize=3,
                label="Cold (3 processes)",
            ),
            Line2D(
                [],
                [],
                color="#555555",
                linestyle="-",
                marker="o",
                markersize=3,
                label="Warm (3 process medians)",
            ),
        ]
        fig.legend(
            handles=handles,
            loc="upper center",
            bbox_to_anchor=(0.5, 0.987),
            ncol=4,
            frameon=False,
        )
        fig.suptitle(label, x=0.5, y=1.005, fontsize=9)
        fig.text(
            0.5,
            0.005,
            "Lines: medians; bars: min–max. Each process: 1 cold + 3 warm energy-and-force calls.",
            ha="center",
            fontsize=7,
        )
        fig.tight_layout(rect=(0, 0.02, 1, 0.955), h_pad=0.9, w_pad=0.6)
        for suffix in (".pdf", ".svg", ".png"):
            fig.savefig(output.with_suffix(suffix), dpi=300, bbox_inches="tight")
        plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", type=Path, required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--oracles", type=Path, required=True)
    parser.add_argument("--reference-reuse", type=Path, required=True)
    parser.add_argument("--build-receipt", type=Path, required=True)
    parser.add_argument(
        "--execution-binding",
        type=Path,
        help="separate actual allocation for an authenticated reused build",
    )
    parser.add_argument("--output", type=raw_output_path, required=True)
    args = parser.parse_args()
    records, rows = load(args.campaign)
    execution = (
        json.loads(args.execution_binding.read_text())
        if args.execution_binding
        else None
    )
    if execution is not None and execution.get("build_receipt_sha256") != sha(
        args.build_receipt
    ):
        raise ValueError("execution binding names a different original build receipt")
    verify_build(records, json.loads(args.build_receipt.read_text()), execution)
    oracles = json.loads(lzma.decompress(args.oracles.read_bytes()))
    reuse = json.loads(args.reference_reuse.read_text())
    checks = accuracy(records, oracles, reuse)
    args.output.mkdir(parents=True, exist_ok=False)
    save(
        args.output / "summary.json",
        {
            "schema": SCHEMA,
            "label": args.label,
            "populations": rows,
            "diagnostics": diagnostics(rows),
            "oracles_sha256": sha(args.oracles),
            "reference_reuse_sha256": sha(args.reference_reuse),
            "build_receipt_sha256": sha(args.build_receipt),
            "execution_binding_sha256": sha(args.execution_binding)
            if args.execution_binding
            else None,
            "accuracy": checks,
            "common_observable_gates": COMMON_GATES,
        },
    )
    with (args.output / "source-data.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            [
                "engine",
                "method",
                "mode",
                "atoms",
                "process_repeat",
                "phase",
                "within_process_repeat",
                "seconds",
                "scf_iterations",
                "fock_or_veff_builds",
            ]
        )
        for record in records:
            for sample in record["samples"]:
                writer.writerow(
                    [
                        *(
                            record[k]
                            for k in (
                                "arm",
                                "method",
                                "mode",
                                "atoms",
                                "process_repeat",
                            )
                        ),
                        sample["phase"],
                        sample["repeat"],
                        sample["seconds"],
                        sample["iterations"],
                        sample.get(
                            "fock_builds", sample.get("effective_potential_builds")
                        ),
                    ]
                )
    render(rows, args.label, args.output / "results-hfdft")


if __name__ == "__main__":
    main()
