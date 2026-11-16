"""Authenticate original processes and reproduce the two PR3093 figure tables."""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import io
import json
import lzma
import math
import os
import re
import statistics
import subprocess
import sys
import tarfile
from collections import Counter
from pathlib import Path, PurePosixPath
from typing import NoReturn

from defusedxml import ElementTree as ET

from benchmarks._retention import raw_output_path
from benchmarks.readme_defaults import METHODS, SIZES

ROOT = Path(__file__).resolve().parents[1]
DEFAULT = ROOT / "benchmarks/results/paper-hfdft-cold3-20261115"
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


def digest(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _object(pairs: list[tuple[str, object]]) -> dict:
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON key")
        result[key] = value
    return result


def _invalid_constant(value: str) -> NoReturn:
    raise ValueError("non-finite JSON constant: " + value)


def read_json(raw: bytes) -> dict:
    return json.loads(raw, object_pairs_hook=_object, parse_constant=_invalid_constant)


def read_xz(raw: bytes) -> dict:
    decoder = lzma.LZMADecompressor(format=lzma.FORMAT_XZ, memlimit=96 << 20)
    decoded = decoder.decompress(raw, max_length=(32 << 20) + 1)
    if len(decoded) > 32 << 20 or not decoder.eof or decoder.unused_data:
        raise ValueError("truncated, concatenated or oversized XZ record")
    return read_json(decoded)


def raw_receipt(record: dict) -> bytes:
    """Restore the acquisition serializer, not a substitute scientific projection."""
    raw = {
        k: v
        for k, v in record.items()
        if k not in ("raw_file", "raw_sha256", "process_repeat")
    }
    return json.dumps(raw, indent=2, allow_nan=False).encode() + b"\n"


def publication_files(folder: Path, manifest: dict) -> dict[str, bytes]:
    """Validate paths before opening any declared publication member."""
    from tools.generativeqc_validation.retention import safe_relative

    files = {}
    for entry in manifest["files"]:
        name = safe_relative(entry["path"])
        path = folder / name
        if (
            name in files
            or any(p.startswith(".env") for p in PurePosixPath(name).parts)
            or not path.resolve().is_relative_to(folder.resolve())
        ):
            raise ValueError("unsafe or duplicate publication member")
        files[name] = path.read_bytes()
    return files


def authenticate(
    records: list[dict], commands: list[dict], expected: set[tuple]
) -> None:
    """Bind each repeat to its distinct successful command and original raw bytes."""
    census = {}
    outputs = set()
    for command in commands:
        if (
            command["stage"] != "formal"
            or command["status"] != "complete"
            or command["returncode"] != 0
        ):
            raise ValueError("unsuccessful formal process")
        argv = command["argv"]
        output = argv[argv.index("--output") + 1]
        if command["name"] in census or output in outputs:
            raise ValueError("duplicate command/output ownership")
        census[command["name"]] = command
        outputs.add(output)
    seen = set()
    owned = set()
    for record in records:
        key = tuple(
            record[k] for k in ("arm", "method", "mode", "atoms", "process_repeat")
        )
        if key not in expected or key in seen:
            raise ValueError("duplicate, missing or unexpected process population")
        seen.add(key)
        arm, method, mode, atoms, repeat = key
        name = f"{method}-{mode}-{atoms}-{arm}-r{repeat}"
        command = census.get(name)
        if command is None or command["process_repeat"] != repeat:
            raise ValueError("repeat has no matching successful command")
        argv = command["argv"]
        path = record["raw_file"]
        if (
            path in owned
            or PurePosixPath(path).name != name + ".json"
            or argv[argv.index("--output") + 1] != path
        ):
            raise ValueError("raw path/repeat ownership mismatch")
        owned.add(path)
        for flag, value in (
            ("--method", method),
            ("--mode", mode),
            ("--atoms", str(atoms)),
            ("--warm-repeats", "3"),
            ("--source-revision", record["source_revision"]),
        ):
            if argv[argv.index(flag) + 1] != value:
                raise ValueError("command/scientific population mismatch")
        if (
            digest(raw_receipt(record)) != record["raw_sha256"]
            or record["raw_sha256"] != command["raw_sha256"]
        ):
            raise ValueError("original raw receipt hash mismatch")
        samples = record["samples"]
        if record["status"] != "complete" or [
            (s["phase"], s["repeat"]) for s in samples
        ] != [("cold", 0), ("warm", 0), ("warm", 1), ("warm", 2)]:
            raise ValueError("incomplete cold/warm calls")
        for sample in samples:
            if (
                sample["converged"] is not True
                or not math.isfinite(sample["seconds"])
                or sample["seconds"] <= 0
                or not math.isfinite(sample["energy"])
            ):
                raise ValueError("invalid scientific endpoint")
            force = sample["forces"]
            if len(force) != atoms or any(
                len(row) != 3 or not all(math.isfinite(v) for v in row) for row in force
            ):
                raise ValueError("missing or invalid full force vector")
    if seen != expected or len(census) != len(expected):
        raise ValueError("incomplete or extra process census")


def populations(records: list[dict]) -> list[dict]:
    result = []
    for arm in COLORS:
        for method in METHODS:
            for mode in ("direct", "df"):
                for atoms in SIZES:
                    selected = sorted(
                        (
                            r
                            for r in records
                            if (r["arm"], r["method"], r["mode"], r["atoms"])
                            == (arm, method, mode, atoms)
                        ),
                        key=lambda r: r["process_repeat"],
                    )
                    cold = [r["samples"][0]["seconds"] for r in selected]
                    warm = [
                        statistics.median(s["seconds"] for s in r["samples"][1:])
                        for r in selected
                    ]
                    result.append(
                        {
                            "key": [arm, method, mode, atoms],
                            "cold": {
                                "seconds": cold,
                                "median": statistics.median(cold),
                                "min": min(cold),
                                "max": max(cold),
                            },
                            "warm": {
                                "process_medians": warm,
                                "median": statistics.median(warm),
                                "min": min(warm),
                                "max": max(warm),
                                "all_seconds": [
                                    s["seconds"]
                                    for r in selected
                                    for s in r["samples"][1:]
                                ],
                            },
                            "cold_iterations": [
                                r["samples"][0]["iterations"] for r in selected
                            ],
                            "cold_focks": [
                                r["samples"][0].get(
                                    "fock_builds",
                                    r["samples"][0].get("effective_potential_builds"),
                                )
                                for r in selected
                            ],
                        }
                    )
    return result


def timing_csv(records: list[dict]) -> bytes:
    stream = io.StringIO(newline="")
    writer = csv.writer(stream, lineterminator="\n")
    writer.writerow(
        [
            "arm",
            "method",
            "mode",
            "atoms",
            "process_repeat",
            "cold_seconds",
            "warm0_seconds",
            "warm1_seconds",
            "warm2_seconds",
            "cold_scf_iterations",
            "cold_fock_or_effective_potential_builds",
        ]
    )
    for record in records:
        samples = record["samples"]
        writer.writerow(
            [
                *(
                    record[k]
                    for k in ("arm", "method", "mode", "atoms", "process_repeat")
                ),
                *(s["seconds"] for s in samples),
                samples[0]["iterations"],
                samples[0].get(
                    "fock_builds", samples[0].get("effective_potential_builds")
                ),
            ]
        )
    return stream.getvalue().encode()


def summary_csv(rows: list[dict]) -> bytes:
    stream = io.StringIO(newline="")
    writer = csv.writer(stream, lineterminator="\n")
    writer.writerow(
        [
            "arm",
            "method",
            "mode",
            "atoms",
            "phase",
            "median_seconds",
            "min_seconds",
            "max_seconds",
        ]
    )
    for row in rows:
        for phase in ("cold", "warm"):
            writer.writerow(
                [*row["key"], phase, *(row[phase][k] for k in ("median", "min", "max"))]
            )
    return stream.getvalue().encode()


def bound_figure_bytes(raw: bytes, binding: dict) -> bytes:
    """Accept only the published LF asset or its explicitly retained CRLF view."""
    if binding.get("schema") != "generativeqc.published-svg-binding.v1":
        raise ValueError("unsupported published figure binding")
    if digest(raw) == binding["sha256"] and b"\r" not in raw:
        return raw
    historical = binding["recorded_publication_checkout"]
    if digest(raw) == historical["sha256"] and historical["line_endings"] == "CRLF":
        published = raw.replace(b"\r\n", b"\n")
        if (
            b"\r" not in published
            and digest(published) == binding["sha256"]
            and published.replace(b"\n", b"\r\n") == raw
        ):
            return published
    raise ValueError("associated figure bytes changed")


def compare_svg(rows: list[dict], raw: bytes) -> dict:
    """Read plotted coordinates back from SVG, independently of rendering code."""
    namespace = "{http://www.w3.org/2000/svg}"
    tree = ET.fromstring(raw)
    indexed = {tuple(row["key"]): row for row in rows}
    axes = [
        e
        for e in tree.iter(namespace + "g")
        if re.fullmatch(r"axes_\d+", e.get("id", ""))
    ]
    if len(axes) != 16:
        raise ValueError("figure panel census mismatch")
    maximum = 0.0

    def points(path: str) -> list[float]:
        if re.sub(r"[ML\d.eE+\-\s]", "", path):
            return []
        return [
            float(v)
            for v in re.findall(r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?", path)
        ]

    for number, axis in enumerate(axes):
        ticks = []
        for group in axis.iter(namespace + "g"):
            if not group.get("id", "").startswith("ytick_"):
                continue
            label = "".join(group.find(".//" + namespace + "text").itertext())
            use = group.find(".//" + namespace + "use")
            if use is not None:
                ticks.append((float(use.get("y")), float(label)))
        if len(ticks) < 2:
            raise ValueError("figure lacks readable y calibration")
        (y0, v0), (y1, v1) = ticks[:2]

        def value(
            y: float, y0: float = y0, v0: float = v0, y1: float = y1, v1: float = v1
        ) -> float:
            return v0 + (y - y0) * (v1 - v0) / (y1 - y0)

        lines, ranges = [], []
        for child in axis:
            ident = child.get("id", "")
            if ident.startswith("line2d_"):
                for path in child.findall(namespace + "path"):
                    coords = points(path.get("d", ""))
                    style = path.get("style", "")
                    if len(coords) == 8 and any(
                        color in style for color in COLORS.values()
                    ):
                        lines.append((style, [value(y) for y in coords[1::2]]))
            elif ident.startswith("LineCollection_"):
                paths = child.findall(namespace + "path")
                if len(paths) == 4:
                    coords = [points(p.get("d", "")) for p in paths]
                    if all(len(p) == 4 for p in coords):
                        ranges.append(
                            (
                                paths[0].get("style", ""),
                                [sorted((value(p[1]), value(p[3]))) for p in coords],
                            )
                        )
        if len(lines) != 4 or len(ranges) != 4:
            raise ValueError("figure series/range census mismatch")
        mode = ("direct", "df")[number // 8]
        method = list(METHODS)[number % 8]
        for series, (arm, phase) in enumerate(
            (a, p) for a in COLORS for p in ("cold", "warm")
        ):
            if (
                COLORS[arm] not in lines[series][0]
                or COLORS[arm] not in ranges[series][0]
                or ("stroke-dasharray" in lines[series][0]) != (phase == "cold")
            ):
                raise ValueError("figure color/phase binding mismatch")
            for size, observed, bounds in zip(
                SIZES, lines[series][1], ranges[series][1], strict=True
            ):
                expected = indexed[(arm, method, mode, size)][phase]
                error = max(
                    abs(observed - expected["median"]),
                    abs(bounds[0] - expected["min"]),
                    abs(bounds[1] - expected["max"]),
                )
                maximum = max(maximum, error)
                if error > 2e-5:
                    raise ValueError("SVG values differ from raw data")
    return {
        "medians_checked": 256,
        "range_endpoints_checked": 512,
        "maximum_svg_rounding_error_seconds": maximum,
    }


def verify(directory: Path = DEFAULT) -> dict:
    sys.path.insert(0, str(ROOT / "python"))
    from benchmarks.readme_defaults_qualification import qualify
    from tools.generativeqc_validation.publication import validate_publication

    oracle = ROOT / "benchmarks/results/readme-refresh-20261026/oracles.json.xz"
    if (
        digest(oracle.read_bytes())
        != "d61a3e8ba0a8f8a8c08ee4c4b30c8e3e397e67863b8d75e7ba974e90cf0f85a4"
    ):
        raise ValueError("independent reference changed")
    references = read_xz(oracle.read_bytes())
    result = {}
    for device in ("h100", "rtx5090"):
        folder = directory / device
        manifest = read_json((folder / "publication.json").read_bytes())
        files = publication_files(folder, manifest)
        validate_publication(manifest, files)
        payload = read_xz(files["samples.json.xz"])
        provenance = read_xz(files["provenance.json.xz"])
        source = read_xz(files["measured-source.json.xz"])
        inputs = read_xz(files["inputs.json.xz"])
        for entry in inputs["files"].values():
            raw = json.dumps(entry["value"], indent=2).encode() + b"\n"
            if digest(raw) != entry["sha256"]:
                raise ValueError("basis/input bytes differ")
        if (
            payload["schema"] != "generativeqc.paper-hfdft-raw.v1"
            or payload["device"] != device
            or payload["prewarm_processes"] != 128
        ):
            raise ValueError("wrong cohort/protocol")
        records = payload["records"]
        expected = {
            (a, m, d, n, r)
            for a in COLORS
            for m in METHODS
            for d in ("direct", "df")
            for n in SIZES
            for r in range(3)
        }
        authenticate(records, payload["commands"], expected)
        if (
            provenance["source_revision"] != manifest["source"]["revision"]
            or source["native_source_identity"] != provenance["native_source_identity"]
        ):
            raise ValueError("measured source binding differs")
        for record in records:
            if (
                record["source_revision"] != provenance["source_revision"]
                or record["runner_sha256"] != provenance["runner_sha256"]
            ):
                raise ValueError("recorder/source identity differs")
            if record["arm"] == "standard" and (
                record["native_build"]["source_identity"]
                != provenance["native_source_identity"]
                or record["native_build"]["library_sha256"]
                != provenance["native_library_sha256"]
            ):
                raise ValueError("loaded native source/image differs")
            if (
                record["mode"] == "df"
                and inputs["files"]["inputs/" + record["method"] + "-auxiliary.json"][
                    "sha256"
                ]
                != record["auxiliary_input_sha256"]
            ):
                raise ValueError("measured auxiliary basis differs")
        rows = populations(records)
        if (
            timing_csv(records) != files["timings.csv"]
            or summary_csv(rows) != files["summary.csv"]
        ):
            raise ValueError("published CSV differs from original samples")
        checks = []
        for repeat in range(3):
            checks.extend(
                {"process_repeat": repeat, **q}
                for q in qualify(
                    [r for r in records if r["process_repeat"] == repeat],
                    references,
                    reference_reuse=provenance["reference_reuse"],
                )
            )
        for check in checks:
            legacy = (check["method"] == "hf" and check["mode"] == "df") or (
                check["method"] == "m06-2x" and check["atoms"] == 3
            )
            if check["status"] != ("failed" if legacy else "qualified"):
                raise ValueError("new/unverified numerical outcome")
            if check["method"] == "hf" and check["mode"] == "df":
                from benchmarks.readme_defaults import COMMON_GATES

                if any(
                    error[key] > limit
                    for phase in ("cold", "warm")
                    for error in check[phase]["errors"]
                    for key, limit in COMMON_GATES.items()
                    if key.startswith("force_")
                ):
                    raise ValueError(
                        "DF-HF force regression beyond retained energy-only failure"
                    )
        counts = Counter(c["status"] for c in checks)
        if counts != Counter(qualified=348, failed=36):
            raise ValueError("numerical outcome census mismatch")
        indexed = {tuple(r["key"]): r for r in rows}
        wins = {
            phase: sum(
                r[phase]["median"]
                < indexed[("gpu4pyscf", *r["key"][1:])][phase]["median"]
                for r in rows
                if r["key"][0] == "standard"
            )
            for phase in ("cold", "warm")
        }
        if wins != {p: provenance["expected"][p + "_wins"] for p in ("cold", "warm")}:
            raise ValueError("timing summary differs")
        figure = ROOT / provenance["figure"]["path"]
        published_figure = bound_figure_bytes(figure.read_bytes(), provenance["figure"])
        plotted = compare_svg(rows, published_figure)
        result[device] = {
            "qualification": checks,
            "inputs": inputs,
            "records": records,
            "populations": rows,
            "wins": wins,
            "numeric_counts": dict(counts),
            "source": source,
            "plot_replay": plotted,
        }
    return result


def reconstruct(source: dict, destination: Path) -> None:
    """Recover measured code from a durable Git base plus exact retained bytes."""
    from tools.generativeqc_validation.retention import safe_relative

    if destination.exists():
        raise FileExistsError(destination)
    base = source["base_revision"]
    if not re.fullmatch(r"[0-9a-f]{40}", base):
        raise ValueError("invalid public source base")
    for name in [
        *source["keep_git_bytes"],
        *source["patch_paths"],
        *source["crlf_extra_paths"],
    ]:
        safe_relative(name)
    raw = subprocess.check_output(
        ["git", "-c", "core.autocrlf=false", "archive", "--format=tar", base],
        cwd=ROOT,
        env={
            **os.environ,
            "GIT_NO_LAZY_FETCH": "1",
            "GIT_ALLOW_PROTOCOL": "",
            "GIT_TERMINAL_PROMPT": "0",
        },
        timeout=180,
    )
    destination.mkdir(parents=True)
    with tarfile.open(fileobj=io.BytesIO(raw)) as archive:
        archive.extractall(destination, filter="data")
    spec = importlib.util.spec_from_file_location(
        "measured_identity", destination / "tools/generate_build_identity.py"
    )
    identity = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(identity)
    inputs = identity._inventory(
        destination, destination / "cmake/GenerativeQCSourceIdentity.json"
    )
    for relative in inputs:
        name = relative.as_posix()
        path = destination / relative
        if name not in source["keep_git_bytes"]:
            path.write_bytes(
                path.read_bytes().replace(b"\r\n", b"\n").replace(b"\n", b"\r\n")
            )
    for name in source["crlf_extra_paths"]:
        path = destination / name
        path.write_bytes(
            path.read_bytes().replace(b"\r\n", b"\n").replace(b"\n", b"\r\n")
        )
    subprocess.run(["git", "init", "--quiet", str(destination)], check=True)
    subprocess.run(
        ["git", "-C", str(destination), "config", "core.autocrlf", "false"], check=True
    )
    patch = source["patch_utf8"].encode()
    subprocess.run(
        ["git", "apply", "--check", "--whitespace=nowarn", "-"],
        cwd=destination,
        input=patch,
        check=True,
    )
    subprocess.run(
        ["git", "apply", "--whitespace=nowarn", "-"],
        cwd=destination,
        input=patch,
        check=True,
    )
    inputs = identity._inventory(
        destination, destination / "cmake/GenerativeQCSourceIdentity.json"
    )
    if (
        len(inputs) != source["native_inputs"]
        or identity._source_identity(destination, inputs)
        != source["native_source_identity"]
    ):
        raise ValueError("reconstructed native input bytes differ")
    for name, expected in source["producer_sha256"].items():
        if digest((destination / name).read_bytes()) != expected:
            raise ValueError("reconstructed measured recorder differs")


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
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="authenticate and replay without writing outputs (the default)",
    )
    parser.add_argument("--directory", type=Path, default=DEFAULT)
    parser.add_argument(
        "--output",
        type=raw_output_path,
        help="write derived tables and PNG/PDF/SVG outside retained results",
    )
    parser.add_argument(
        "--reconstruct-source",
        type=raw_output_path,
        help="recover measured native and recorder source bytes; does not compile or run a GPU",
    )
    parser.add_argument(
        "--restore-raw",
        type=raw_output_path,
        help="restore exact original formal JSON receipts outside retained results",
    )
    args = parser.parse_args()
    if args.verify_only and (
        args.output or args.reconstruct_source or args.restore_raw
    ):
        parser.error("--verify-only cannot write outputs")
    result = verify(args.directory)
    if args.output:
        args.output.mkdir(parents=True, exist_ok=False)
        for device, data in result.items():
            folder = args.output / device
            folder.mkdir()
            (folder / "timings.csv").write_bytes(timing_csv(data["records"]))
            (folder / "summary.csv").write_bytes(summary_csv(data["populations"]))
            (folder / "qualification.json").write_text(
                json.dumps(data["qualification"], indent=2, allow_nan=False) + "\n",
                encoding="utf-8",
            )
            render(
                data["populations"],
                "H100" if device == "h100" else "RTX 5090",
                folder / "results-hfdft",
            )
    if args.reconstruct_source:
        for device, data in result.items():
            target = args.reconstruct_source / device
            reconstruct(data["source"], target)
            for name, entry in data["inputs"]["files"].items():
                from tools.generativeqc_validation.retention import safe_relative

                path = target / "measurement-inputs" / safe_relative(name)
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(json.dumps(entry["value"], indent=2).encode() + b"\n")
    if args.restore_raw:
        args.restore_raw.mkdir(parents=True, exist_ok=False)
        for device, data in result.items():
            folder = args.restore_raw / device
            folder.mkdir()
            for record in data["records"]:
                (folder / PurePosixPath(record["raw_file"]).name).write_bytes(
                    raw_receipt(record)
                )
    print(
        json.dumps(
            {
                "status": "RAW_BYTES_TABLES_FULL_VECTOR_GATES_AND_PLOTTED_VALUES_REPLAYED",
                "devices": {
                    d: {
                        "formal_processes": len(v["records"]),
                        "full_EF_calls": 4 * len(v["records"]),
                        "wins": v["wins"],
                        "numeric_counts": v["numeric_counts"],
                        "plot_replay": v["plot_replay"],
                    }
                    for d, v in result.items()
                },
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
