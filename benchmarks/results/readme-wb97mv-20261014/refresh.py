"""Replay the README panels without duplicating unchanged historical evidence.

Only the WB97M-V population is replaced. Other functionals keep their original
source/build identities in the linked baseline campaign, not the new revision.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import lzma
import math
from collections import Counter
from pathlib import Path
from typing import Any
from xml.etree import ElementTree

from tools.render_readme_dft import phase_rows, render

ROOT = Path(__file__).resolve().parent
BASELINE = ROOT.parent / "readme-dft-20261013"
SOURCE_COMMIT = "810f081ef293256ebe20beb7298f9d21996fec81"
PHASE_COUNTS = {"cold": 1, "warm": 5, "moved": 1, "moved-warm": 5}


def read_summary(path: Path) -> dict[str, Any]:
    """Read the losslessly compressed scientific reduction, not raw run debris."""
    return json.loads(lzma.decompress(path.read_bytes()))


def combined_summary(name: str) -> dict[str, Any]:
    """Join fresh native points with the unchanged, explicitly historical panels."""
    summary = copy.deepcopy(read_summary(BASELINE / name))
    update = read_summary(ROOT / name)
    if update["source_commit"] != SOURCE_COMMIT:
        raise ValueError("WB97M-V source identity differs from this publication")
    if update["sizes"] != summary["sizes"]:
        raise ValueError("WB97M-V sizes differ from the README protocol")
    if set(update["methods"]) != {"wb97m-v"}:
        raise ValueError("refresh may replace only the WB97M-V population")
    if update["density_fitting"] != summary["density_fitting"]:
        raise ValueError("direct and density-fitted campaigns cannot be combined")
    summary["methods"]["wb97m-v"] = update["methods"]["wb97m-v"]
    summary["panel_annotations"] = update.get("panel_annotations", {})
    return summary


def compact_svg_presentation(svg: str) -> str:
    """Share repeated presentation attributes without changing plot geometry.

    This operates only on renderer-owned SVGs without existing CSS classes.
    Stroke cap/join attributes stay untouched because Matplotlib's universal
    rule already controls their cascade. Text whitespace remains significant.
    """
    from defusedxml.ElementTree import fromstring

    namespace = "http://www.w3.org/2000/svg"
    root = fromstring(svg)
    if any("class" in element.attrib for element in root.iter()):
        return svg
    properties = {
        "fill",
        "stroke",
        "stroke-width",
        "stroke-dasharray",
        "stroke-dashoffset",
        "font-family",
        "font-size",
        "font-style",
        "font-weight",
        "text-anchor",
        "opacity",
        "fill-opacity",
        "stroke-opacity",
    }
    combinations = [
        tuple(
            sorted(
                (key, value)
                for key, value in element.attrib.items()
                if key in properties
            )
        )
        for element in root.iter()
    ]
    classes = {}
    rules = []
    for attributes, count in Counter(combinations).items():
        declarations = ";".join(
            f"{key}:{value}px" if key == "stroke-width" else f"{key}:{value}"
            for key, value in attributes
        )
        original_bytes = sum(len(key) + len(value) + 4 for key, value in attributes)
        if count < 2 or count * original_bytes <= count * 18 + len(declarations) + 18:
            continue
        name = f"gqc{len(classes)}"
        classes[attributes] = name
        rules.append(f".{name}{{{declarations}}}")
    for element, attributes in zip(root.iter(), combinations, strict=True):
        if attributes in classes:
            for key, _value in attributes:
                del element.attrib[key]
            element.set("class", classes[attributes])
        if element.tag not in {f"{{{namespace}}}text", f"{{{namespace}}}tspan"}:
            if element.text is not None and not element.text.strip():
                element.text = None
            for child in element:
                if child.tail is not None and not child.tail.strip():
                    child.tail = None
    definitions = root.find(f"{{{namespace}}}defs")
    if definitions is None:
        definitions = ElementTree.SubElement(root, f"{{{namespace}}}defs")
    ElementTree.SubElement(definitions, f"{{{namespace}}}style").text = "".join(rules)
    ElementTree.register_namespace("", namespace)
    ElementTree.register_namespace("xlink", "http://www.w3.org/1999/xlink")
    return ElementTree.tostring(root, encoding="unicode") + "\n"


def render_publication(summary: dict[str, Any], destination: Path) -> None:
    """Compact generated SVG markup without changing the scientific reduction.

    Scour 0.38.2 removes unused IDs and redundant markup; eight significant
    digits retain sub-pixel positioning. Historical SVGs
    are archived losslessly instead, preserving their exact original bytes.
    """
    from scour.scour import parse_args, scourString

    render(summary, destination)
    options = parse_args(
        [
            "--enable-id-stripping",
            "--shorten-ids",
            "--strip-xml-prolog",
            "--remove-metadata",
            "--indent=none",
            "--set-precision=8",
            "--set-c-precision=8",
        ]
    )
    name = "df-cold-warm.svg" if summary["density_fitting"] else "dft-cold-warm.svg"
    path = destination / name
    path.write_text(
        compact_svg_presentation(scourString(path.read_text(), options=options))
    )


def verify() -> None:
    """Audit stored identities, full phase populations and honest accuracy labels.

    This scalar/integrity audit cannot recompute omitted native force vectors or
    qualify a different binary. Missing independent geometry oracles remain null.
    Checked raw rows may encode execution in their combined scientific gate;
    unchecked rows must retain a separate successful execution gate.
    """
    manifest = json.loads((ROOT / "manifest.json").read_text())
    if manifest["native_source_commit"] != SOURCE_COMMIT:
        raise ValueError("manifest source identity differs from this publication")
    for directory, key in ((ROOT, "files"), (BASELINE, "baseline_files")):
        for name, receipt in manifest[key].items():
            payload = (directory / name).read_bytes()
            if (
                len(payload) != receipt["bytes"]
                or hashlib.sha256(payload).hexdigest() != receipt["sha256"]
            ):
                raise ValueError(f"publication checksum mismatch: {name}")
            if "uncompressed" in receipt:
                original = lzma.decompress(payload)
                identity = receipt["uncompressed"]
                if (
                    len(original) != identity["bytes"]
                    or hashlib.sha256(original).hexdigest() != identity["sha256"]
                ):
                    raise ValueError(f"historical figure identity mismatch: {name}")
    for name in ("summary.json.xz", "df-summary.json.xz"):
        summary = read_summary(ROOT / name)
        points = summary["methods"]["wb97m-v"]
        expected_sizes = [3, 12, 48] if summary["density_fitting"] else [3, 12, 48, 96]
        if [point["atoms"] for point in points] != expected_sizes:
            raise ValueError("publication must retain every completed native size")
        for point in points:
            native = point["engines"]["native"]
            if (
                native["native_build"]["library_sha256"]
                != manifest["native_library_sha256"]
            ):
                raise ValueError("native point uses a different binary")
            if native["source_file_sha256"] != manifest["native_source_file_sha256"]:
                raise ValueError(
                    "native point uses a different source/consumer inventory"
                )
            if native["status"] not in ("measured", "completed_unchecked"):
                raise ValueError(
                    f"incomplete native WB97M-V population: {point['atoms']}"
                )
            if point["atoms"] in (3, 12):
                priming = point.get("cache_priming", {})
                if not (
                    priming.get("exit_code") == 0
                    and priming.get("phase_counts") == PHASE_COUNTS
                    and priming.get("independently_checked_calls") == 12
                    and priming.get("fresh_measurement_process") is True
                    and priming.get("scheduler") == native["scheduler"]
                    and priming.get("reproducer_sha256")
                    == hashlib.sha256((ROOT / "reproduce.sh").read_bytes()).hexdigest()
                    and native["native_experiment"]["public_initialization_policy"]
                    is True
                    and native["native_experiment"]["preliminary_provider"] == "none"
                ):
                    raise ValueError(
                        "small native point lacks a successful separate cache primer"
                    )
            if len(native["records"]) != sum(PHASE_COUNTS.values()):
                raise ValueError("native population must contain all twelve endpoints")
            for phase, expected in PHASE_COUNTS.items():
                rows = phase_rows(native, phase)
                if len(rows) != expected:
                    raise ValueError(f"incomplete phase: {phase}")
                for row in rows:
                    if not (
                        row.get("execution_gate", row["gate"]) is True
                        and row["converged"] is True
                        and row["status"] == 0
                        and math.isfinite(row["energy"])
                    ):
                        raise ValueError("native E+F execution did not complete")
                    if (
                        not math.isfinite(row["complete_seconds"])
                        or row["complete_seconds"] <= 0
                    ):
                        raise ValueError("invalid complete endpoint timing")
                    if row["gate"] is None:
                        if native["phase_status"][phase] != "completed_unchecked":
                            raise ValueError(
                                "missing accuracy cannot have a measured label"
                            )
                        if (
                            row["energy_error"] is not None
                            or row["force_error"] is not None
                        ):
                            raise ValueError(
                                "unchecked accuracy must retain null errors"
                            )
                    elif not (
                        row["gate"] is True
                        and 0 <= row["energy_error"] <= 1e-8
                        and 0 <= row["force_error"] <= 1e-7
                    ):
                        raise ValueError("independent energy/force acceptance failed")
        combined_summary(name)
    archive = read_summary(ROOT / "unprimed-small-points.json.xz")
    for mode, points in archive["populations"].items():
        current = read_summary(
            ROOT / ("df-summary.json.xz" if mode == "cuda" else "summary.json.xz")
        )
        for point in points:
            retained = next(
                candidate
                for candidate in current["methods"]["wb97m-v"]
                if candidate["atoms"] == point["atoms"]
            )
            for key, value in (
                ("reference_engine_sha256", retained["engines"]["reference"]),
                ("independent_references_sha256", retained["independent_references"]),
            ):
                digest = hashlib.sha256(
                    json.dumps(value, sort_keys=True, allow_nan=False).encode()
                ).hexdigest()
                if digest != point["reference_bindings"][key]:
                    raise ValueError(
                        "unprimed observations lost their original reference binding"
                    )
            original = point["engines"]["native"]
            if (
                original["source_file_sha256"] != manifest["native_source_file_sha256"]
                or original["native_build"]["library_sha256"]
                != manifest["native_library_sha256"]
            ):
                raise ValueError(
                    "unprimed observations have a different source or binary"
                )
            if Counter(
                row["phase"] for row in original["records"]
            ) != PHASE_COUNTS or not all(
                row["gate"] is True for row in original["records"]
            ):
                raise ValueError(
                    "unprimed original observations were not preserved completely"
                )


def main() -> None:
    """Verify the retained campaign and optionally replay its tables and SVGs."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    verify()
    if args.output:
        args.output.mkdir(parents=True, exist_ok=True)
        for name in ("summary.json.xz", "df-summary.json.xz"):
            render_publication(combined_summary(name), args.output)
    print("WB97M-V identities, complete phases and scalar gates verified.")


if __name__ == "__main__":
    main()
